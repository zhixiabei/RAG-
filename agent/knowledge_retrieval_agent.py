import re
from typing import Any
import unicodedata

from .contracts import ModelGateway, SearchHit, VectorStore


_IDENTIFIER_PATTERNS = (
    re.compile(r"[A-Za-z\u4e00-\u9fff]{1,8}\s*\d+(?:\s*[-\u2010-\u2015./]\s*\d+)+"),
    re.compile(r"[A-Za-z]{1,12}\s*\d+(?:\.\d+)+", re.IGNORECASE),
)


def _normalize_keyword(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"\s*([-./])\s*", r"\1", normalized)
    return re.sub(r"\s+", "", normalized)


def extract_keyword_terms(question: str, limit: int = 32) -> list[str]:
    """Extract exact identifiers plus short Chinese phrases for lexical recall."""
    terms = []
    seen = set()

    def add(value: str) -> None:
        term = _normalize_keyword(value)
        if len(term) < 2 or term in seen:
            return
        seen.add(term)
        terms.append(term)

    normalized_question = unicodedata.normalize("NFKC", question)
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(normalized_question):
            add(match.group(0))

    for run in re.findall(r"[\u4e00-\u9fff]{4,}", normalized_question):
        for size in (6, 5, 4):
            if len(run) < size:
                continue
            for start in range(len(run) - size + 1):
                add(run[start:start + size])
                if len(terms) >= limit:
                    return terms
    return terms[:limit]


class KnowledgeRetrievalAgent:
    """Embeds a question and returns its nearest knowledge-base chunks."""

    name = "knowledge_retrieval"

    def __init__(self, vectors: VectorStore, models: ModelGateway, top_k: int):
        if top_k <= 0:
            raise ValueError("Top-K 必须大于 0")
        self.vectors = vectors
        self.models = models
        self.top_k = top_k

    def run(self, knowledge_base: dict[str, Any], question: str) -> list[SearchHit]:
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 "
                f"{self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        query_vector = self.models.embed([question])[0]
        vector_hits = list(
            self.vectors.search(knowledge_base["id"], query_vector, self.top_k)
        )
        keywords = extract_keyword_terms(question)
        keyword_limit = max(1, self.top_k // 2)
        keyword_hits = list(
            self.vectors.search_keywords(
                knowledge_base["id"],
                keywords,
                keyword_limit,
            )
        ) if keywords else []

        combined = []
        seen_chunk_ids = set()
        for hit in [*keyword_hits, *vector_hits]:
            if hit.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(hit.chunk_id)
            combined.append(hit)
            if len(combined) >= self.top_k:
                break
        return combined
