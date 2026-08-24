from pydantic import BaseModel, Field, field_validator


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=200)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    include_retrieved_content: bool = False
    force_retrieval: bool = False


class ParsedChatAttachment(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    context: str = Field(min_length=1, max_length=12000)
    citations: list[dict] = Field(default_factory=list, max_length=200)


class ParsedAttachmentChatRequest(ChatRequest):
    attachments: list[ParsedChatAttachment] = Field(min_length=1, max_length=10)


class EvaluationRequest(BaseModel):
    question_ids: list[str] = Field(default_factory=list, max_length=500)
    dataset_source: str = Field(default="workshop", pattern="^(workshop|local)$")
    dataset_id: str | None = Field(default=None, max_length=255)

    @field_validator("question_ids")
    @classmethod
    def normalize_question_ids(cls, values: list[str]) -> list[str]:
        normalized = []
        seen = set()
        for value in values:
            question_id = value.strip()
            if question_id and question_id not in seen:
                normalized.append(question_id)
                seen.add(question_id)
        return normalized


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("对话名称不能为空")
        return title


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("对话名称不能为空")
        return title
