import logging
import re
from typing import Any
import unicodedata

from .contracts import ModelGateway, Reranker, SearchHit, VectorStore
from .telemetry import model_usage_stage


_IDENTIFIER_PATTERNS = (
    re.compile(r"[A-Za-z\u4e00-\u9fff]{1,8}\s*\d+(?:\s*[-\u2010-\u2015./]\s*\d+)+"),
    re.compile(r"[A-Za-z]{1,12}\s*\d+(?:\.\d+)+", re.IGNORECASE),
)
_RRF_RANK_CONSTANT = 60
_KEYWORD_RRF_WEIGHT = 1.25
logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        vectors: VectorStore,
        models: ModelGateway,
        top_k: int,
        candidate_k: int | None = None,
        reranker: Reranker | None = None,
    ):
        if top_k <= 0:
            raise ValueError("Top-K 必须大于 0")
        candidate_k = top_k if candidate_k is None else candidate_k
        if candidate_k < top_k:
            raise ValueError("Candidate-K must be greater than or equal to Top-K")
        self.vectors = vectors
        self.models = models
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.reranker = reranker

    def run(self, knowledge_base: dict[str, Any], question: str) -> list[SearchHit]:
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 "
                f"{self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        with model_usage_stage("query_embedding"):
            query_vector = self.models.embed([question])[0]
        vector_hits = list(
            self.vectors.search(knowledge_base["id"], query_vector, self.candidate_k)
        )
        keywords = extract_keyword_terms(question)
        keyword_limit = max(1, self.candidate_k // 2)
        keyword_hits = list(
            self.vectors.search_keywords(
                knowledge_base["id"],
                keywords,
                keyword_limit,
            )
        ) if keywords else []
        candidates = _reciprocal_rank_fusion(
            vector_hits,
            keyword_hits,
            len(vector_hits) + len(keyword_hits),
        )
        if self.reranker is None or not candidates:
            return candidates[:self.top_k]
        try:
            with model_usage_stage("reranking"):
                reranked = list(
                    self.reranker.rerank(question, candidates, self.top_k)
                )
            if not reranked:
                raise RuntimeError("Reranker returned no results")
            seen = {hit.chunk_id for hit in reranked}
            reranked.extend(
                hit
                for hit in candidates
                if hit.chunk_id not in seen
            )
            return reranked[:self.top_k]
        except Exception:
            logger.warning(
                "Reranking failed; falling back to fused retrieval order",
                exc_info=True,
            )
            return candidates[:self.top_k]


def _reciprocal_rank_fusion(
    vector_hits: list[SearchHit],
    keyword_hits: list[SearchHit],
    limit: int,
) -> list[SearchHit]:
    """Fuse lexical and vector rankings without comparing incompatible raw scores."""
    hits_by_id: dict[str, SearchHit] = {}
    fused_scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for weight, hits in ((1.0, vector_hits), (_KEYWORD_RRF_WEIGHT, keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            hits_by_id.setdefault(hit.chunk_id, hit)
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + weight / (
                _RRF_RANK_CONSTANT + rank
            )
            best_rank[hit.chunk_id] = min(best_rank.get(hit.chunk_id, rank), rank)
    ranked_ids = sorted(
        hits_by_id,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            best_rank[chunk_id],
            -float(hits_by_id[chunk_id].score),
            chunk_id,
        ),
    )
    return [hits_by_id[chunk_id] for chunk_id in ranked_ids[:limit]]
