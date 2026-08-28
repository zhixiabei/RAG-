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
    """Routes queries through documents before retrieving their nearest chunks."""

    name = "knowledge_retrieval"

    def __init__(
        self,
        vectors: VectorStore,
        models: ModelGateway,
        top_k: int,
        candidate_k: int | None = None,
        reranker: Reranker | None = None,
        document_candidate_k: int | None = None,
        document_score_threshold: float = 0.45,
    ):
        if top_k <= 0:
            raise ValueError("Top-K 必须大于 0")
        candidate_k = top_k if candidate_k is None else candidate_k
        document_candidate_k = (
            candidate_k if document_candidate_k is None else document_candidate_k
        )
        if candidate_k < top_k:
            raise ValueError("Candidate-K must be greater than or equal to Top-K")
        if document_candidate_k <= 0:
            raise ValueError("Document-Candidate-K must be greater than 0")
        if not 0 <= document_score_threshold <= 1:
            raise ValueError("Document score threshold must be between 0 and 1")
        self.vectors = vectors
        self.models = models
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.document_candidate_k = document_candidate_k
        self.document_score_threshold = document_score_threshold
        self.reranker = reranker

    def run(self, knowledge_base: dict[str, Any], question: str) -> list[SearchHit]:
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 "
                f"{self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        with model_usage_stage("query_embedding"):
            query_vector = self.models.embed([question])[0]
        knowledge_base_id = knowledge_base["id"]
        keywords = extract_keyword_terms(question)

        document_vector_hits: list[SearchHit] = []
        document_ids: list[str] = []
        search_documents = getattr(self.vectors, "search_documents", None)
        if callable(search_documents):
            document_vector_hits = _best_document_hits(list(
                search_documents(
                    knowledge_base_id,
                    query_vector,
                    self.document_candidate_k,
                )
            ))
            document_ids = [
                hit.document_id
                for hit in document_vector_hits
                if float(hit.score) >= self.document_score_threshold
            ]

        if document_ids:
            candidates = self._retrieve_chunks(
                knowledge_base_id,
                query_vector,
                keywords,
                document_ids,
            )
        elif not document_vector_hits:
            candidates = self._retrieve_chunks(
                knowledge_base_id,
                query_vector,
                keywords,
                None,
            )
        else:
            candidates = []

        # A stale or incomplete document index must not turn routing into a hard failure.
        if document_ids and not candidates:
            candidates = self._retrieve_chunks(
                knowledge_base_id,
                query_vector,
                keywords,
                None,
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

    def _retrieve_chunks(
        self,
        knowledge_base_id: str,
        query_vector: list[float],
        keywords: list[str],
        document_ids: list[str] | None,
    ) -> list[SearchHit]:
        if document_ids is None:
            vector_hits = list(
                self.vectors.search(
                    knowledge_base_id,
                    query_vector,
                    self.candidate_k,
                )
            )
        else:
            vector_hits = list(
                self.vectors.search(
                    knowledge_base_id,
                    query_vector,
                    self.candidate_k,
                    document_ids,
                )
            )

        keyword_limit = max(1, self.candidate_k // 2)
        if not keywords:
            keyword_hits = []
        elif document_ids is None:
            keyword_hits = list(
                self.vectors.search_keywords(
                    knowledge_base_id,
                    keywords,
                    keyword_limit,
                )
            )
        else:
            keyword_hits = list(
                self.vectors.search_keywords(
                    knowledge_base_id,
                    keywords,
                    keyword_limit,
                    document_ids,
                )
            )
        return _reciprocal_rank_fusion(
            vector_hits,
            keyword_hits,
            len(vector_hits) + len(keyword_hits),
        )


def _reciprocal_rank_fusion(
    vector_hits: list[SearchHit],
    keyword_hits: list[SearchHit],
    limit: int,
    key_attribute: str = "chunk_id",
    second_weight: float = _KEYWORD_RRF_WEIGHT,
) -> list[SearchHit]:
    """Fuse lexical and vector rankings without comparing incompatible raw scores."""
    hits_by_id: dict[str, SearchHit] = {}
    fused_scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for weight, hits in ((1.0, vector_hits), (second_weight, keyword_hits)):
        for rank, hit in enumerate(hits, start=1):
            hit_id = str(getattr(hit, key_attribute))
            hits_by_id.setdefault(hit_id, hit)
            fused_scores[hit_id] = fused_scores.get(hit_id, 0.0) + weight / (
                _RRF_RANK_CONSTANT + rank
            )
            best_rank[hit_id] = min(best_rank.get(hit_id, rank), rank)
    ranked_ids = sorted(
        hits_by_id,
        key=lambda hit_id: (
            -fused_scores[hit_id],
            best_rank[hit_id],
            -float(hits_by_id[hit_id].score),
            hit_id,
        ),
    )
    return [hits_by_id[hit_id] for hit_id in ranked_ids[:limit]]


def _best_document_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Deduplicate route nodes while keeping the retriever's first-seen order."""
    best_by_document: dict[str, SearchHit] = {}
    document_order: list[str] = []
    for hit in hits:
        document_id = str(hit.document_id)
        current = best_by_document.get(document_id)
        if current is None:
            document_order.append(document_id)
        if current is None or float(hit.score) > float(current.score):
            best_by_document[document_id] = hit
    return [best_by_document[document_id] for document_id in document_order]
