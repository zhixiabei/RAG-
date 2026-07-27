from dataclasses import dataclass
import json
import re
from typing import Any

from .contracts import ModelGateway
from .query_intent import is_assistant_identity_question, is_knowledge_catalog_inventory_question


RETRIEVAL_DECISION_PROMPT = """判断回答当前消息是否需要检索新的知识库文档。
需要检索时 decision 为 RETRIEVE，不需要时 decision 为 SKIP。

以下情况输出 RETRIEVE：
- 用户提出新的事实性问题、需要查找知识库内容或核验信息；
- 用户要求引用、出处、原文或更具体的知识库细节；
- 仅凭对话历史无法可靠回答。

只有以下情况输出 SKIP：
- 问候、致谢、告别等日常交流；
- 询问当前助手身份、当前使用的模型或助手自身能力；
- 对已有回答做改写、翻译、总结、格式调整或简单澄清，不需要新事实；
- 问题可以完全依据对话历史回答。

同时生成 search_query：
- decision 为 RETRIEVE 时，search_query 必须是可脱离对话独立理解的知识库检索词；
- 结合对话历史补全省略的信息和指代，但不要把此前助手的回答当作新的事实来源；
- 保留用户真正的信息需求、主题、对象和限定条件，去掉不影响检索的口语套话；
- 当用户提到文件名或扩展名时，必须原样保留完整文件名及后缀，不得删除或改写后缀；
- 不得添加用户没有表达的领域、结论或限制；
- decision 为 SKIP 时，search_query 为空字符串。

只输出 JSON，例如 {"decision":"RETRIEVE","search_query":"可独立理解的检索问题"}
或 {"decision":"SKIP","search_query":""}，不要解释。遇到不确定的情况输出 RETRIEVE。"""

RETRIEVAL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["RETRIEVE", "SKIP"]},
        "search_query": {"type": "string"},
    },
    "required": ["decision", "search_query"],
}


def retrieval_decision_messages(question: str, history: list[dict[str, Any]]) -> list[dict[str, str]]:
    transcript = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
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
        value = incomplete_json.group(1) if incomplete_json else normalized
    # Unexpected output must not silently bypass relevant knowledge.
    return str(value).strip().upper() != "SKIP"


def retrieval_search_query(output: str, question: str) -> str:
    normalized = output.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*|\s*```$", "", normalized, flags=re.IGNORECASE)
    try:
        payload = json.loads(normalized)
    except (TypeError, json.JSONDecodeError):
        return question
    if not isinstance(payload, dict) or str(payload.get("decision", "")).strip().upper() == "SKIP":
        return ""
    query = payload.get("search_query")
    resolved = query.strip() if isinstance(query, str) and query.strip() else question
    suffixes = dict.fromkeys(
        re.findall(r"\.[A-Za-z][A-Za-z0-9]{0,9}(?=[^A-Za-z0-9]|$)", question)
    )
    missing_suffixes = [suffix for suffix in suffixes if suffix.casefold() not in resolved.casefold()]
    return " ".join([resolved, *missing_suffixes])


@dataclass(frozen=True)
class RetrievalDecision:
    should_retrieve: bool
    search_query: str = ""

    @property
    def outcome(self) -> str:
        return "retrieve" if self.should_retrieve else "skip"


class RetrievalDecisionAgent:
    """Decides whether the current turn needs fresh knowledge-base context."""

    name = "retrieval_decision"

    def __init__(self, models: ModelGateway):
        self.models = models

    def run(self, question: str, history: list[dict[str, Any]]) -> RetrievalDecision:
        if is_assistant_identity_question(question) or is_knowledge_catalog_inventory_question(question):
            return RetrievalDecision(False)
        output = self.models.complete(
            retrieval_decision_messages(question, history),
            temperature=0,
            reasoning=False,
            response_schema=RETRIEVAL_DECISION_SCHEMA,
        )
        retrieve = should_retrieve(output)
        return RetrievalDecision(retrieve, retrieval_search_query(output, question) if retrieve else "")
