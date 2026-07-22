from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=200)
    model: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=200)
