from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

