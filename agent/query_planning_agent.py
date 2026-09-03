from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Sequence
import unicodedata

from .context import select_history_messages
from .contracts import ModelGateway
from .telemetry import model_usage_stage, timed_stage


logger = logging.getLogger(__name__)
_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["single", "rewrite", "decompose"],
            "description": "single=一个取证目标；rewrite=消除历史指代；decompose=多个可独立检索的取证目标",
        },
        "standalone_query": {
            "type": "string",
            "maxLength": 240,
            "description": "仅 rewrite 使用；必须是完整、自包含的问题",
        },
        "subqueries": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "maxItems": 4,
            "description": "仅 decompose 使用；每项对应一个独立且自包含的取证目标",
        },
    },
    "required": ["strategy", "standalone_query", "subqueries"],
}
_HISTORY_TOKENS = 1_200
_MAX_HISTORY_MESSAGES = 4
_MAX_HISTORY_CHARS = 800
_MAX_QUERY_CHARS = 500
_MAX_SUBQUERIES = 4
_QUERY_CONTAMINATION_MARKERS = (
    "trigger reason",
    "current question:",
    "standalone query:",
    "rewrite:",
    "触发原因：",
    "对话历史：",
    "当前问题：",
    "独立问题：",
    "改写：",
)
_FILE_SOURCE_PATTERN = re.compile(
    r"[\w\u4e00-\u9fff][\w\u4e00-\u9fff._()\uFF08\uFF09-]{1,100}"
    r"\.(?:docx?|pdf|xlsx?|pptx?|txt|csv|gdb|att|md)",
    re.IGNORECASE,
)
_SOURCE_CAPTURE_PATTERN = re.compile(r"\u300a([^\u300b]{2,160})\u300b")
_IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z\u4e00-\u9fff]{1,8}\s*\d+(?:\s*[-\u2010-\u2015./]\s*\d+)+"
)
_YEAR_PATTERN = re.compile(r"(?<!\d)(?:(?:19|20)\d{2}|\d{2})\s*(?:年|年度)")
_ANAPHORIC_SUBQUERY_PATTERN = re.compile(
    r"^(?:其(?:中)?|该(?:项|对象|文件|记录)?|上述|前者|后者|另一个(?:文件|对象)?)"
)

QUERY_PLANNING_PROMPT = """你负责生成知识库检索计划，不负责回答问题。
single 表示一个检索视角足够；rewrite 仅用于消除对话历史中的指代；decompose 表示需要多个独立检索视角。
如果用户明确写出多个来源，只有在它们对应不同取证目标时才拆分，并由你直接输出来源绑定的子查询；不要根据文件名推断领域或主题。程序不会在你输出之后自动补充来源子查询。
多个指标如果共享主体、来源和时间上下文，可以保持 single。只有每个子问题都能独立检索、且检索约束不同，才使用 decompose。
必须保留井号、层位、年份、标准号、数值、专有名词和用户明确写出的来源。只能改写或拆分已有目标，不得补充答案、结论或新事实。
single 时 standalone_query 输出空字符串；rewrite 时只输出一句不超过 120 字的消除指代后的问题；decompose 时 standalone_query 输出空字符串，subqueries 每项只写一个不超过 120 字的检索问题。
每个 subquery 必须是自包含的完整检索问题，重复必要的主体、来源、年份和指标；禁止只写“其……”“该文件……”“前者……”“另一个……”等依赖上下文的片段。
用户问题下方的“机械提取硬约束”只用于保留原文中的来源、编号、年份和数值，不代表必须拆分，也不能据此推断主题或答案。
例如：用户问“根据《合同甲》和《验收记录乙》，比较付款节点与到账情况”，可拆为“《合同甲》中的付款节点是什么？”和“《验收记录乙》中的到账情况是什么？”，不能写成“其付款节点”和“另一个文件的到账情况”。
严禁重复句子、回答问题、计算数值、生成 SQL 或添加原问题中没有的事实。只输出符合 schema 的 JSON。"""


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
        if self.strategy == "rewrite":
            return self.standalone_query or _clean_query(original_question)
        return _clean_query(original_question)

    def as_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "standalone_query": self.standalone_query, "subqueries": list(self.subqueries), "trigger": self.trigger, "model_invoked": self.model_invoked, "fallback": self.fallback}


class QueryPlanningAgent:
    """Conditionally rewrites follow-ups and decomposes multi-evidence queries."""

    name = "query_planning"

    def __init__(self, models: ModelGateway):
        self.models = models

    def run(
        self,
        question: str,
        history: Sequence[dict[str, Any]],
        trigger: str = "planner",
    ) -> QueryPlan:
        try:
            with timed_stage("planning.generation"), model_usage_stage("query_planning"):
                output = self.models.complete(
                    query_planning_messages(question, history, trigger),
                    temperature=0, max_tokens=384, reasoning=False, response_schema=_PLAN_SCHEMA,
                )
            return parse_query_plan(output, question, trigger)
        except Exception as exc:
            logger.warning(
                "Query planning output invalid; using the original question (%s: %s)",
                type(exc).__name__,
                exc,
            )
            logger.debug("Query planning failure details", exc_info=True)
            return QueryPlan.single(question, trigger=trigger, fallback=True)


def query_planning_messages(question: str, history: Sequence[dict[str, Any]], trigger: str) -> list[dict[str, str]]:
    history_view = select_history_messages(
        history,
        _HISTORY_TOKENS if trigger == "context_reference" else 0,
    )
    selected_history = history_view.messages[-_MAX_HISTORY_MESSAGES:]
    transcript = "\n".join(
        f"{item['role']}: {_clip_history_text(item['content'])}"
        for item in selected_history
    )
    if history_view.omission_notice:
        transcript = f"{history_view.omission_notice}\n{transcript}"
    constraints = format_query_constraints(question)
    return [
        {"role": "system", "content": QUERY_PLANNING_PROMPT},
        {
            "role": "user",
            "content": (
                f"触发原因：{trigger}\n"
                f"对话历史：\n{transcript or '（无）'}\n\n"
                f"机械提取硬约束：\n{constraints}\n\n"
                f"当前问题：\n{question}"
            ),
        },
    ]


def format_query_constraints(question: str) -> str:
    """Expose literal anchors without inferring a domain or decomposition."""
    normalized = unicodedata.normalize("NFKC", _clean_query(question))
    sources = _unique_queries([
        *_SOURCE_CAPTURE_PATTERN.findall(normalized),
        *_FILE_SOURCE_PATTERN.findall(normalized),
    ])
    identifiers = _unique_queries(
        match.group(0) for match in _IDENTIFIER_PATTERN.finditer(normalized)
    )
    years = _unique_queries(match.group(0) for match in _YEAR_PATTERN.finditer(normalized))
    fields = []
    if sources:
        fields.append(f"来源={'、'.join(sources[:_MAX_SUBQUERIES])}")
    if identifiers:
        fields.append(f"编号={'、'.join(identifiers[:_MAX_SUBQUERIES])}")
    if years:
        fields.append(f"时间={'、'.join(years[:_MAX_SUBQUERIES])}")
    return "；".join(fields) or "无"


def parse_query_plan(output: str, question: str, trigger: str) -> QueryPlan:
    payload = _load_json_object(output)
    if not isinstance(payload, dict):
        raise ValueError("Query planner output must be a JSON object")
    partially_recovered = bool(payload.pop("_partial", False))
    strategy = str(payload.get("strategy") or "").strip().lower()
    if strategy not in {"single", "rewrite", "decompose"}:
        raise ValueError("Query planner returned an unsupported strategy")
    original = _clean_query(question)
    try:
        standalone_query = _validated_generated_query(
            payload.get("standalone_query") or original,
            fallback=original,
        )
    except ValueError:
        standalone_query = original
        partially_recovered = True
    raw_subqueries = payload.get("subqueries") or []
    if not isinstance(raw_subqueries, list):
        raise ValueError("Query planner subqueries must be a list")
    valid_subqueries: list[str] = []
    for item in raw_subqueries:
        if not str(item).strip():
            continue
        try:
            valid_subqueries.append(
                _validated_generated_query(item, require_self_contained=True)
            )
        except ValueError:
            partially_recovered = True
    subqueries = tuple(_unique_queries(valid_subqueries))[:_MAX_SUBQUERIES]
    if strategy == "single":
        subqueries = ()
    elif strategy == "rewrite":
        subqueries = ()
        if standalone_query == original:
            return QueryPlan.single(
                original,
                trigger=trigger,
                fallback=True,
            )
    elif len(subqueries) < 2:
        return QueryPlan.single(
            original,
            trigger=trigger,
            fallback=True,
        )
    if strategy == "decompose":
        # The standalone field is not used for decomposed retrieval. Keep the
        # original question as the rerank anchor regardless of model text.
        standalone_query = original
    if strategy != "decompose":
        subqueries = ()
    return QueryPlan(
        strategy,
        standalone_query,
        subqueries,
        trigger=trigger,
        fallback=partially_recovered,
    )


def fallback_query_plan(question: str, trigger: str | None) -> QueryPlan:
    """Return a retrieval-safe plan when structured model output is unusable."""
    return QueryPlan.single(question, trigger=trigger, fallback=True)


def _load_json_object(output: str) -> dict[str, Any]:
    """Parse strict JSON while tolerating a model's prose or code-fence wrapper."""
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Query planner returned empty output")
    normalized = output.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[^\r\n]*\r?\n?", "", normalized, count=1)
        normalized = re.sub(r"\r?\n?```\s*$", "", normalized).strip()

    candidates = [normalized]
    extracted = _extract_json_object(normalized)
    if extracted and extracted != normalized:
        candidates.append(extracted)
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise ValueError("Query planner output must be a JSON object")
        return payload
    partial = _load_partial_json_object(normalized)
    if partial is not None:
        return partial
    if last_error is not None:
        raise last_error
    raise ValueError("Query planner output must be a JSON object")


def _load_partial_json_object(value: str) -> dict[str, Any] | None:
    """Recover completed planner fields from a truncated JSON response."""
    strategy, strategy_complete = _extract_partial_json_string(value, "strategy")
    if not strategy_complete or strategy.strip().lower() not in {"single", "rewrite", "decompose"}:
        if re.search(r'"decision"\s*:\s*"(?:RETRIEVE|SKIP)"', value, re.IGNORECASE):
            return {"strategy": "single", "subqueries": [], "_partial": True}
        return None
    strategy = strategy.strip().lower()

    payload: dict[str, Any] = {"strategy": strategy}
    standalone, standalone_complete = _extract_partial_json_string(value, "standalone_query")
    if standalone_complete and standalone:
        payload["standalone_query"] = standalone
    payload["subqueries"] = _extract_partial_json_array(value, "subqueries")
    payload["_partial"] = True
    return payload


def _extract_partial_json_string(value: str, field: str) -> tuple[str, bool]:
    marker = re.search(rf'"{re.escape(field)}"\s*:\s*"', value)
    if not marker:
        return "", False
    raw: list[str] = []
    escaped = False
    for character in value[marker.end():]:
        if escaped:
            raw.append(character)
            escaped = False
            continue
        if character == chr(92):
            raw.append(character)
            escaped = True
            continue
        if character == '"':
            return _decode_partial_json_string("".join(raw)), True
        raw.append(" " if character in "\r\n" else character)
    return _decode_partial_json_string("".join(raw)), False


def _extract_partial_json_array(value: str, field: str) -> list[str]:
    marker = re.search(rf'"{re.escape(field)}"\s*:\s*\[', value)
    if not marker:
        return []
    body = value[marker.end():]
    result: list[str] = []
    index = 0
    while index < len(body):
        while index < len(body) and body[index] in "\r\n\t ,":
            index += 1
        if index >= len(body) or body[index] == "]":
            break
        if body[index] != '"':
            index += 1
            continue
        index += 1
        raw: list[str] = []
        escaped = False
        complete = False
        while index < len(body):
            character = body[index]
            index += 1
            if escaped:
                raw.append(character)
                escaped = False
                continue
            if character == chr(92):
                raw.append(character)
                escaped = True
                continue
            if character == '"':
                complete = True
                break
            raw.append(" " if character in "\r\n" else character)
        if not complete:
            break
        decoded = _decode_partial_json_string("".join(raw))
        if decoded:
            result.append(decoded)
    return result[:_MAX_SUBQUERIES]


def _decode_partial_json_string(value: str) -> str:
    try:
        decoded = json.loads(f'"{value}"')
    except json.JSONDecodeError:
        decoded = value
    return _clean_query(str(decoded))


def _extract_json_object(value: str) -> str | None:
    """Return the first balanced JSON object, ignoring braces inside strings."""
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == chr(92):
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return value[start:index + 1]
    return None


def _validated_generated_query(
    value: object,
    fallback: str = "",
    require_self_contained: bool = False,
) -> str:
    query = _clean_query(str(value or ""))
    lowered = query.casefold()
    if any(marker in lowered or marker in query for marker in _QUERY_CONTAMINATION_MARKERS):
        raise ValueError("Query planner copied prompt text into a query")
    if _is_repetitive_query(query):
        raise ValueError("Query planner repeated the same text")
    if re.search(
        r"(?:答案|结论|结果)\s*(?:是|为|：|:)|\b(?:select|from|where)\b",
        query,
        re.IGNORECASE,
    ):
        raise ValueError("Query planner generated an answer instead of a query")
    if require_self_contained and _ANAPHORIC_SUBQUERY_PATTERN.match(query):
        raise ValueError("Query planner generated a context-dependent subquery")
    if not query:
        if fallback:
            return fallback
        raise ValueError("Query planner returned an empty query")
    return query


def _is_repetitive_query(query: str) -> bool:
    compact = re.sub(r"\s+", "", query)
    if len(compact) < 32:
        return False
    max_unit_length = min(80, len(compact) // 2)
    for unit_length in range(12, max_unit_length + 1):
        if compact[:unit_length] == compact[unit_length:unit_length * 2]:
            return True
    return False


def _clip_history_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= _MAX_HISTORY_CHARS:
        return text
    half = (_MAX_HISTORY_CHARS - 18) // 2
    return f"{text[:half]} ...[省略]... {text[-half:]}"


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
