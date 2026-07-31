from pydantic import BaseModel, Field, field_validator


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=200)
    model: str | None = Field(default=None, min_length=1, max_length=200)


class ParsedChatAttachment(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    context: str = Field(min_length=1, max_length=12000)
    citations: list[dict] = Field(default_factory=list, max_length=200)


class ParsedAttachmentChatRequest(ChatRequest):
    attachments: list[ParsedChatAttachment] = Field(min_length=1, max_length=10)


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
