from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from .context import select_history_messages
from .contracts import ModelGateway
from .query_intent import QueryIntent, analyze_query_intent
from .query_planning_agent import (
    QueryPlan,
    fallback_query_plan,
    parse_query_plan,
    query_planning_trigger,
)
from .telemetry import model_usage_stage, timed_stage


logger = logging.getLogger(__name__)


RETRIEVAL_DECISION_PROMPT = """你负责一次性完成检索判断和查询规划，不负责回答问题。
需要知识库事实、原文、出处或校验信息时 decision=RETRIEVE；问候、致谢、改写已有回答、助手身份问题或完全可以依据对话历史回答时 decision=SKIP。不确定时输出 RETRIEVE。
strategy=single 表示一个检索视角足够；strategy=rewrite 仅用于消除历史指代；strategy=decompose 表示有多个独立取证目标，需要生成 2 到 4 个子问题。
显式写出多个来源时，只有它们对应不同取证目标才拆分，并由你直接输出来源绑定的子查询；不要根据文件名或领域词推断主题。程序不会在你输出之后自动补充子查询。
多个指标如果共享主体、来源和时间上下文，可以使用 single。只有每个子问题都能独立检索且检索约束不同，才使用 decompose。
必须保留井号、层位、年份、标准号、数值、专有名词和用户明确写出的来源。只能改写或拆分已有目标，不得补充答案、结论或新事实。
single 时 standalone_query 输出空字符串；rewrite 时只输出一句不超过 120 字的消除指代后的问题；decompose 时 standalone_query 输出空字符串，subqueries 每项只写一个不超过 120 字的检索问题。
严禁重复句子、回答问题、计算数值、生成 SQL 或添加原问题中没有的事实。
只输出符合 schema 的 JSON，不要解释。"""
RETRIEVAL_DECISION_ONLY_PROMPT = """你只负责判断当前消息是否需要检索知识库，不负责回答问题。
需要知识库事实、原文、出处或校验信息时 decision=RETRIEVE；问候、致谢、改写已有回答、助手身份问题或完全可以依据对话历史回答时 decision=SKIP。
不确定时输出 RETRIEVE。只输出 JSON：{"decision":"RETRIEVE"} 或 {"decision":"SKIP"}，不要输出其他字段或解释。"""

RETRIEVAL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["RETRIEVE", "SKIP"]},
        "strategy": {"type": "string", "enum": ["single", "rewrite", "decompose"]},
        "standalone_query": {"type": "string", "maxLength": 240},
        "subqueries": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "maxItems": 4,
        },
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
RETRIEVAL_DECISION_PLAN_MAX_TOKENS = 384


def retrieval_decision_messages(
    question: str,
    history: list[dict[str, Any]],
    include_plan: bool = True,
) -> list[dict[str, str]]:
    history_view = select_history_messages(history, RETRIEVAL_DECISION_HISTORY_TOKENS)
    transcript = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in history_view.messages
    )
    if history_view.omission_notice:
        transcript = f"{history_view.omission_notice}\n{transcript}"
    return [
        {
            "role": "system",
            "content": RETRIEVAL_DECISION_PROMPT if include_plan else RETRIEVAL_DECISION_ONLY_PROMPT,
        },
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
        planning_requested = self.query_planning_enabled and trigger is not None
        try:
            with timed_stage("decision.generation"), model_usage_stage("retrieval_decision"):
                output = self.models.complete(
                    retrieval_decision_messages(
                        question,
                        history,
                        include_plan=planning_requested,
                    ),
                    temperature=0,
                    max_tokens=(
                        RETRIEVAL_DECISION_PLAN_MAX_TOKENS
                        if planning_requested
                        else 16
                    ),
                    reasoning=False,
                    response_schema=(
                        RETRIEVAL_DECISION_SCHEMA
                        if planning_requested
                        else RETRIEVAL_DECISION_ONLY_SCHEMA
                    ),
                )
        except Exception as exc:
            logger.warning(
                "Retrieval decision model failed; defaulting to retrieval (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return RetrievalDecision(True, fallback_query_plan(question, trigger))
        should_retrieve_now = should_retrieve(output)
        planning_trigger = trigger or "planner"
        if not should_retrieve_now:
            # Preserve a valid combined plan for callers that may force
            # retrieval later; do not let structural keywords override the
            # model's retrieval decision.
            if planning_requested:
                try:
                    return RetrievalDecision(
                        False,
                        parse_query_plan(output, question, planning_trigger),
                    )
                except Exception:
                    pass
            return RetrievalDecision(False)
        if not planning_requested:
            return RetrievalDecision(
                True,
                QueryPlan.single(question, trigger=planning_trigger),
            )
        try:
            query_plan = parse_query_plan(output, question, planning_trigger)
        except Exception as exc:
            logger.warning(
                "Combined retrieval decision/query plan was invalid; using deterministic fallback (%s: %s)",
                type(exc).__name__,
                exc,
            )
            query_plan = fallback_query_plan(question, planning_trigger)
        return RetrievalDecision(True, query_plan)
