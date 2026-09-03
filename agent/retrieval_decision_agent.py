from dataclasses import dataclass
import json
import logging
import re
from typing import Any

from .context import select_history_messages
from .contracts import ModelGateway
from .query_intent import QueryIntent
from .query_planning_agent import (
    QueryPlan,
    QueryPlanningAgent,
)
from .telemetry import model_usage_stage, timed_stage


logger = logging.getLogger(__name__)


RETRIEVAL_DECISION_PROMPT = """你只负责判断当前消息是否需要知识库检索，以及它是否包含多个独立取证目标；不负责回答问题，也不生成子查询。
需要知识库事实、原文、出处或校验信息时 decision=RETRIEVE；问候、致谢、助手身份问题、纯改写请求或完全可以依据对话历史回答时 decision=SKIP。不确定时输出 RETRIEVE。
complexity=complex 仅表示存在两个或以上可以分别检索、且检索约束不同的取证目标；共享同一主体、来源和时间上下文的多个指标仍然是 simple。
如果当前问题依赖对话历史中的“它、这个、上述、前者”等指代，needs_rewrite=true；否则为 false。complexity 只判断问题结构，不要因为出现某个领域词、文件名或连接词就机械判定 complex。
只输出 decision、complexity 和 needs_rewrite 三个字段，不要输出 strategy、subqueries、答案、结论、计算结果或解释。"""
RETRIEVAL_DECISION_ONLY_PROMPT = """你只负责判断当前消息是否需要检索知识库，不负责回答问题。
需要知识库事实、原文、出处或校验信息时 decision=RETRIEVE；问候、致谢、改写已有回答、助手身份问题或完全可以依据对话历史回答时 decision=SKIP。
不确定时输出 RETRIEVE。只输出 JSON：{"decision":"RETRIEVE"} 或 {"decision":"SKIP"}，不要输出其他字段或解释。"""

RETRIEVAL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["RETRIEVE", "SKIP"]},
        "complexity": {"type": "string", "enum": ["simple", "complex"]},
        "needs_rewrite": {"type": "boolean"},
    },
    "required": ["decision", "complexity", "needs_rewrite"],
}
RETRIEVAL_DECISION_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["RETRIEVE", "SKIP"]},
    },
    "required": ["decision"],
}
RETRIEVAL_DECISION_HISTORY_TOKENS = 512
RETRIEVAL_DECISION_MAX_TOKENS = 96


def retrieval_decision_messages(
    question: str,
    history: list[dict[str, Any]],
    include_complexity: bool = True,
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
            "content": RETRIEVAL_DECISION_PROMPT if include_complexity else RETRIEVAL_DECISION_ONLY_PROMPT,
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
class RetrievalAssessment:
    should_retrieve: bool
    complexity: str = "simple"
    needs_rewrite: bool = False
    valid: bool = True


def parse_retrieval_assessment(output: str) -> RetrievalAssessment:
    """Parse the first-stage decision without treating structural regex as intent."""
    normalized = str(output or "").strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*|\s*```$", "", normalized, flags=re.IGNORECASE)
    payload: dict[str, Any] | None = None
    try:
        decoded = json.loads(normalized)
        if isinstance(decoded, dict):
            payload = decoded
    except json.JSONDecodeError:
        payload = None

    decision_match = re.search(r'"decision"\s*:\s*"(RETRIEVE|SKIP)"', normalized, re.IGNORECASE)
    complexity_match = re.search(r'"complexity"\s*:\s*"(simple|complex)"', normalized, re.IGNORECASE)
    rewrite_match = re.search(r'"needs_rewrite"\s*:\s*(true|false)', normalized, re.IGNORECASE)

    raw_decision = payload.get("decision") if payload else None
    raw_complexity = payload.get("complexity") if payload else None
    raw_rewrite = payload.get("needs_rewrite") if payload else None
    decision = str(raw_decision or (decision_match.group(1) if decision_match else "RETRIEVE")).strip().upper()
    complexity = str(raw_complexity or (complexity_match.group(1) if complexity_match else "simple")).strip().lower()
    needs_rewrite = (
        raw_rewrite
        if isinstance(raw_rewrite, bool)
        else rewrite_match.group(1).lower() == "true"
        if rewrite_match
        else False
    )
    valid = (
        decision in {"RETRIEVE", "SKIP"}
        and complexity in {"simple", "complex"}
        and (isinstance(raw_rewrite, bool) or rewrite_match is not None)
    )
    return RetrievalAssessment(
        should_retrieve=decision != "SKIP",
        complexity=complexity if complexity in {"simple", "complex"} else "simple",
        needs_rewrite=bool(needs_rewrite),
        valid=valid,
    )


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

    def __init__(self, models: ModelGateway, query_planning_enabled: bool = True):
        self.models = models
        self.query_planning_enabled = query_planning_enabled

    def run(
        self,
        question: str,
        history: list[dict[str, Any]],
        intent: QueryIntent | None = None,
        force_retrieval: bool = False,
    ) -> RetrievalDecision:
        # QueryIntent is metadata-only; it must never bypass the model decision.
        _ = intent
        try:
            with timed_stage("decision.generation"), model_usage_stage("retrieval_decision"):
                output = self.models.complete(
                    retrieval_decision_messages(
                        question,
                        history,
                        include_complexity=self.query_planning_enabled,
                    ),
                    temperature=0,
                    max_tokens=RETRIEVAL_DECISION_MAX_TOKENS if self.query_planning_enabled else 16,
                    reasoning=False,
                    response_schema=(
                        RETRIEVAL_DECISION_SCHEMA
                        if self.query_planning_enabled
                        else RETRIEVAL_DECISION_ONLY_SCHEMA
                    ),
                )
        except Exception as exc:
            logger.warning(
                "Retrieval decision model failed; defaulting to retrieval (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return RetrievalDecision(
                True,
                QueryPlan.single(question, trigger="decision", fallback=True),
            )

        assessment = (
            parse_retrieval_assessment(output)
            if self.query_planning_enabled
            else RetrievalAssessment(should_retrieve(output))
        )
        if not assessment.should_retrieve and not force_retrieval:
            return RetrievalDecision(False)
        if not self.query_planning_enabled or not assessment.valid:
            return RetrievalDecision(
                True,
                QueryPlan.single(
                    question,
                    trigger="decision",
                    fallback=not assessment.valid,
                ),
            )
        if assessment.complexity == "complex" or assessment.needs_rewrite:
            planning_trigger = (
                "complex_query"
                if assessment.complexity == "complex"
                else "context_reference"
            )
            query_plan = QueryPlanningAgent(self.models).run(
                question,
                history,
                trigger=planning_trigger,
            )
            return RetrievalDecision(True, query_plan)
        return RetrievalDecision(
            True,
            QueryPlan.single(question, trigger="decision"),
        )
