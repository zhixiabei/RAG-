from dataclasses import dataclass
from typing import Any

from .contracts import ModelGateway


RETRIEVAL_DECISION_PROMPT = """判断回答当前消息是否需要检索新的知识库文档。
需要检索时输出 RETRIEVE，不需要时输出 SKIP。

以下情况输出 RETRIEVE：
- 用户提出新的事实性问题、需要查找知识库内容或核验信息；
- 用户要求引用、出处、原文或更具体的知识库细节；
- 仅凭对话历史无法可靠回答。

只有以下情况输出 SKIP：
- 问候、致谢、告别等日常交流；
- 对已有回答做改写、翻译、总结、格式调整或简单澄清，不需要新事实；
- 问题可以完全依据对话历史回答。

只输出 RETRIEVE 或 SKIP，不要解释。遇到不确定的情况输出 RETRIEVE。"""


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
    # Unexpected output must not silently bypass relevant knowledge.
    return decision.strip().upper() != "SKIP"


@dataclass(frozen=True)
class RetrievalDecision:
    should_retrieve: bool

    @property
    def outcome(self) -> str:
        return "retrieve" if self.should_retrieve else "skip"


class RetrievalDecisionAgent:
    """Decides whether the current turn needs fresh knowledge-base context."""

    name = "retrieval_decision"

    def __init__(self, models: ModelGateway):
        self.models = models

    def run(self, question: str, history: list[dict[str, Any]]) -> RetrievalDecision:
        output = self.models.complete(
            retrieval_decision_messages(question, history),
            temperature=0,
            max_tokens=8,
            reasoning=False,
        )
        return RetrievalDecision(should_retrieve(output))
