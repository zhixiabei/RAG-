from typing import Any, Protocol, Sequence


class SearchHit(Protocol):
    chunk_id: str
    document_id: str
    knowledge_base_id: str
    title: str
    text: str
    score: float
    folder_path: str
    page_number: int | None


class ModelGateway(Protocol):
    chat_model: str
    embedding_model: str

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        reasoning: bool | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def search(
        self,
        knowledge_base_id: str,
        vector: list[float],
        limit: int,
    ) -> Sequence[SearchHit]: ...
