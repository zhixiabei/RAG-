from dataclasses import dataclass
import json
import re
from typing import Any

from .context import select_history_messages
from .contracts import ModelGateway
from .query_intent import QueryIntent, analyze_query_intent
from .query_planning_agent import QueryPlan, parse_query_plan, query_planning_trigger
from .telemetry import model_usage_stage, timed_stage


RETRIEVAL_DECISION_PROMPT = """判断当前消息是否需要检索新的知识库文档。
需要知识库事实、原文、出处或校验信息时输出 RETRIEVE；问候、致谢、改写已有回答、助手身份问题或完全可以依据对话历史回答时输出 SKIP。
只输出 JSON，不要解释：{\"decision\":\"RETRIEVE\"} 或 {\"decision\":\"SKIP\"}。不确定时输出 RETRIEVE。"""

RETRIEVAL_DECISION_PROMPT = """你负责一次性完成检索判断和查询规划，不负责回答问题。
需要知识库事实、原文、出处或校验信息时 decision=RETRIEVE；问候、致谢、改写已有回答、助手身份问题或完全可以依据对话历史回答时 decision=SKIP。不确定时输出 RETRIEVE。
strategy=single 表示问题独立且只有一个目标；strategy=rewrite 表示依赖历史指代；strategy=decompose 表示有多个取证目标，需要生成 2 到 4 个独立子问题。
必须保留井号、层位、年份、标准号、数值和专有名词。只能改写或拆分已有目标，不得补充答案、结论或新事实。
没有历史指代或明显多目标时使用 single，standalone_query 原样保留当前问题，subqueries 输出空数组。
只输出符合 schema 的 JSON，不要解释。"""

RETRIEVAL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["RETRIEVE", "SKIP"]},
        "strategy": {"type": "string", "enum": ["single", "rewrite", "decompose"]},
        "standalone_query": {"type": "string"},
        "subqueries": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": ["decision", "strategy", "standalone_query", "subqueries"],
}
RETRIEVAL_DECISION_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["RETRIEVE", "SKIP"]},
    },
    "required": ["decision"],
}
RETRIEVAL_DECISION_HISTORY_TOKENS = 512


def retrieval_decision_messages(question: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
    history_view = select_history_messages(history, RETRIEVAL_DECISION_HISTORY_TOKENS)
    transcript = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in history_view.messages
    )
    if history_view.omission_notice:
        transcript = f"{history_view.omission_notice}\n{transcript}"
    return [
        {"role": "system", "content": RETRIEVAL_DECISION_PROMPT},
        {"role": "user", "content": f"对话历史：\n{transcript or '（无）'}\n\n当前消息：\n{question}"},
    ]


def should_retrieve(decision: str) -> bool:
    normalized = decision.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*|\s*```$", "", normalized, flags=re.IGNORECASE)
    try:
        value = json.loads(normalized).get("decision", "")
    except (AttributeError, json.JSONDecodeError):
        incomplete_json = re.fullmatch(
            r'\s*\{\s*"decision"\s*:\s*"(RETRIEVE|SKIP)"\s*\}?\s*',
            normalized,
            flags=re.IGNORECASE,
        )
        decision_field = re.search(
            r'"decision"\s*:\s*"(RETRIEVE|SKIP)"',
            normalized,
            flags=re.IGNORECASE,
        )
        value = (
            incomplete_json.group(1)
            if incomplete_json
            else decision_field.group(1)
            if decision_field
            else normalized
        )
    # Unexpected output must not silently bypass relevant knowledge.
    return str(value).strip().upper() != "SKIP"


@dataclass(frozen=True)
class RetrievalDecision:
    should_retrieve: bool
    query_plan: QueryPlan | None = None

    @property
    def outcome(self) -> str:
        return "retrieve" if self.should_retrieve else "skip"


class RetrievalDecisionAgent:
    """Decides whether the current turn needs fresh knowledge-base context."""

    name = "retrieval_decision"

    def __init__(self, models: ModelGateway, query_planning_enabled: bool = False):
        self.models = models
        self.query_planning_enabled = query_planning_enabled

    def run(
        self,
        question: str,
        history: list[dict[str, Any]],
        intent: QueryIntent | None = None,
    ) -> RetrievalDecision:
        intent = intent or analyze_query_intent(question)
        if intent.skips_retrieval:
            return RetrievalDecision(False)
        trigger = (
            query_planning_trigger(question, history)
            if self.query_planning_enabled
            else None
        )
        with timed_stage("decision.generation"), model_usage_stage("retrieval_decision"):
            output = self.models.complete(
                retrieval_decision_messages(question, history),
                temperature=0,
                max_tokens=384 if self.query_planning_enabled else 16,
                reasoning=False,
                response_schema=(
                    RETRIEVAL_DECISION_SCHEMA
                    if self.query_planning_enabled
                    else RETRIEVAL_DECISION_ONLY_SCHEMA
                ),
            )
        should_retrieve_now = should_retrieve(output)
        if not should_retrieve_now or trigger is None:
            return RetrievalDecision(should_retrieve_now, QueryPlan.single(question))
        try:
            query_plan = parse_query_plan(output, question, trigger)
        except Exception:
            query_plan = QueryPlan.single(question, trigger=trigger, fallback=True)
        return RetrievalDecision(should_retrieve_now, query_plan)
