from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Sequence

from .context import select_history_messages
from .contracts import ModelGateway
from .telemetry import model_usage_stage, timed_stage


logger = logging.getLogger(__name__)
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": ["single", "rewrite", "decompose"]},
        "standalone_query": {"type": "string"},
        "subqueries": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["strategy", "standalone_query", "subqueries"],
}
_HISTORY_TOKENS = 1_200
_MAX_QUERY_CHARS = 500
_MAX_SUBQUERIES = 4
_REFERENCE_PATTERN = re.compile(r"(?:它|这个|那个|这些|那些|上述|上面|前面|刚才|前者|后者|其中|该方案|该文件|该井|该层|this|that|it|they|above)", re.IGNORECASE)
_SHORT_FOLLOW_UP_PATTERN = re.compile(r"(?:呢|怎么样|如何|多少|是什么|怎么办|有何影响|有什么区别)[？?]?$", re.IGNORECASE)
_STRONG_COMPLEX_PATTERN = re.compile(r"(?:综合|分别|对比|比较|异同|各自|逐项|多个方面|哪些方面)", re.IGNORECASE)
_WEAK_COMPLEX_PATTERN = re.compile(r"(?:以及|并且|同时|还要|还需|并说明|并分析|并给出|及其|又有哪些)", re.IGNORECASE)

QUERY_PLANNING_PROMPT = """你负责生成知识库检索查询，不负责回答问题。
策略：single 表示问题独立且只有一个目标；rewrite 表示依赖历史指代；decompose 表示有多个取证目标，需要生成 2 到 4 个独立子问题。
必须保留井号、层位、年份、标准号、数值和专有名词。只能改写或拆分已有目标，不得补充答案、结论或新事实。
standalone_query 是消除指代后的完整原问题。只输出符合 schema 的 JSON。"""


@dataclass(frozen=True)
class QueryPlan:
    strategy: str
    standalone_query: str
    subqueries: tuple[str, ...] = ()
    trigger: str | None = None
    fallback: bool = False

    @classmethod
    def single(cls, question: str, *, trigger: str | None = None, fallback: bool = False) -> "QueryPlan":
        return cls("single", _clean_query(question), trigger=trigger, fallback=fallback)

    @property
    def model_invoked(self) -> bool:
        return self.trigger is not None

    def retrieval_queries(self, original_question: str) -> list[str]:
        queries = [_clean_query(original_question)]
        if self.strategy == "rewrite":
            queries.append(self.standalone_query)
        elif self.strategy == "decompose":
            queries.extend(self.subqueries)
        return _unique_queries(queries)

    def rerank_query(self, original_question: str) -> str:
        return self.standalone_query or _clean_query(original_question)

    def as_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "standalone_query": self.standalone_query, "subqueries": list(self.subqueries), "trigger": self.trigger, "model_invoked": self.model_invoked, "fallback": self.fallback}


class QueryPlanningAgent:
    """Conditionally rewrites follow-ups and decomposes multi-evidence queries."""

    name = "query_planning"

    def __init__(self, models: ModelGateway):
        self.models = models

    def run(self, question: str, history: Sequence[dict[str, Any]]) -> QueryPlan:
        trigger = query_planning_trigger(question, history)
        if trigger is None:
            return QueryPlan.single(question)
        try:
            with timed_stage("planning.generation"), model_usage_stage("query_planning"):
                output = self.models.complete(
                    query_planning_messages(question, history, trigger),
                    temperature=0, max_tokens=384, reasoning=False, response_schema=_PLAN_SCHEMA,
                )
            return parse_query_plan(output, question, trigger)
        except Exception:
            logger.warning("Query planning failed; falling back to the original question", exc_info=True)
            return QueryPlan.single(question, trigger=trigger, fallback=True)


def query_planning_trigger(question: str, history: Sequence[dict[str, Any]]) -> str | None:
    normalized = _clean_query(question)
    compact_length = len(re.sub(r"\s+", "", normalized))
    has_history = any(item.get("role") in {"user", "assistant"} and str(item.get("content") or "").strip() for item in history)
    if has_history and (_REFERENCE_PATTERN.search(normalized) or compact_length <= 18 and _SHORT_FOLLOW_UP_PATTERN.search(normalized)):
        return "context_reference"
    if compact_length >= 18 and _STRONG_COMPLEX_PATTERN.search(normalized):
        return "complex_query"
    if compact_length >= 36 and len(_WEAK_COMPLEX_PATTERN.findall(normalized)) >= 2:
        return "complex_query"
    return None


def query_planning_messages(question: str, history: Sequence[dict[str, Any]], trigger: str) -> list[dict[str, str]]:
    history_view = select_history_messages(history, _HISTORY_TOKENS)
    transcript = "\n".join(f"{item['role']}: {item['content']}" for item in history_view.messages)
    if history_view.omission_notice:
        transcript = f"{history_view.omission_notice}\n{transcript}"
    return [
        {"role": "system", "content": QUERY_PLANNING_PROMPT},
        {"role": "user", "content": f"触发原因：{trigger}\n对话历史：\n{transcript or '（无）'}\n\n当前问题：\n{question}"},
    ]


def parse_query_plan(output: str, question: str, trigger: str) -> QueryPlan:
    normalized = output.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*|\s*```$", "", normalized, flags=re.IGNORECASE)
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError("Query planner output must be a JSON object")
    strategy = str(payload.get("strategy") or "").strip().lower()
    if strategy not in {"single", "rewrite", "decompose"}:
        raise ValueError("Query planner returned an unsupported strategy")
    original = _clean_query(question)
    standalone_query = _clean_query(str(payload.get("standalone_query") or original))
    raw_subqueries = payload.get("subqueries") or []
    if not isinstance(raw_subqueries, list):
        raise ValueError("Query planner subqueries must be a list")
    subqueries = tuple(_unique_queries([_clean_query(str(item)) for item in raw_subqueries if str(item).strip()]))[:_MAX_SUBQUERIES]
    if strategy == "rewrite" and standalone_query == original:
        strategy = "single"
    if strategy == "decompose" and len(subqueries) < 2:
        strategy = "rewrite" if standalone_query != original else "single"
        subqueries = ()
    if strategy != "decompose":
        subqueries = ()
    return QueryPlan(strategy, standalone_query, subqueries, trigger=trigger)


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()[:_MAX_QUERY_CHARS]


def _unique_queries(queries: Sequence[str]) -> list[str]:
    result = []
    seen = set()
    for query in queries:
        cleaned = _clean_query(query)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
