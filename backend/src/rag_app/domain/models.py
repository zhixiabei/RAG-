from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedChunk:
    index: int
    text: str
    page_number: int | None = None
    section_path: str | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    knowledge_base_id: str
    title: str
    text: str
    score: float
    page_number: int | None = None


@dataclass(frozen=True)
class Citation:
    document_id: str
    chunk_id: str
    title: str
    page_number: int | None
    score: float
    relevance_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "page_number": self.page_number,
            "score": self.score,
            "relevance_score": self.relevance_score,
        }
