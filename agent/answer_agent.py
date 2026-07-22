from typing import Any, Sequence

from .contracts import ModelGateway, SearchHit


ANSWER_SYSTEM_PROMPT = (
    "你是知识库问答助手，只能依据检索上下文和此前对话中已有的知识库信息回答；"
    "信息不足时明确说明，不得编造。"
)


def answer_messages(
    question: str,
    context: str,
    history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
    messages.append({"role": "user", "content": f"当前问题：{question}\n\n本轮检索上下文：\n{context}"})
    return messages


class AnswerAgent:
    """Produces the only user-facing answer from retrieved context and history."""

    name = "answer"

    def __init__(self, models: ModelGateway):
        self.models = models

    def run(
        self,
        question: str,
        history: list[dict[str, Any]],
        hits: Sequence[SearchHit],
        retrieval_used: bool,
        model: str | None = None,
    ) -> str:
        if hits:
            context = "\n\n".join(
                f"[文档] {hit.title}\n[页码] {hit.page_number or '未知'}\n[内容] {hit.text}"
                for hit in hits
            )
        elif not retrieval_used:
            context = (
                "本轮问题不需要检索新的知识库内容。请仅依据此前对话回答；"
                "若是问候、致谢等日常交流，可直接简洁回应。"
            )
        elif history:
            context = "本轮未检索到新的相关文档片段，请仅依据历史对话中已有的信息回答。"
        else:
            return "知识库中没有足够信息回答这个问题。"
        return self.models.complete(answer_messages(question, context, history), model=model, temperature=0.1)
