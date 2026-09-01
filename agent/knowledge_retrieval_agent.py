import logging
import re
from typing import Any, Sequence
import unicodedata

from .contracts import ModelGateway, Reranker, SearchHit, VectorStore
from .query_planning_agent import QueryPlan
from .telemetry import model_usage_stage, timed_stage


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

    def run(
        self,
        knowledge_base: dict[str, Any],
        question: str,
        query_plan: QueryPlan | None = None,
    ) -> list[SearchHit]:
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 "
                f"{self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        plan = query_plan or QueryPlan.single(question)
        queries = plan.retrieval_queries(question)
        with timed_stage("retrieval.query_embedding"), model_usage_stage("query_embedding"):
            query_vectors = self.models.embed(queries)
        if len(query_vectors) != len(queries) or any(not vector for vector in query_vectors):
            raise RuntimeError("embedding service returned incomplete query vectors")
        knowledge_base_id = knowledge_base["id"]
        query_candidates = [
            self._retrieve_for_query(knowledge_base_id, query_vector, query)
            for query, query_vector in zip(queries, query_vectors)
        ]
        candidates = _multi_query_fusion(
            query_candidates,
            limit=max(self.candidate_k, self.top_k * 6),
        )
        if self.reranker is None or not candidates:
            return candidates[:self.top_k]

        # A decomposed question has multiple independent evidence targets. A
        # single rerank over the fused pool lets one target crowd out another,
        # so rank each subquery against its own candidates before interleaving
        # the results for the answer model.
        if plan.strategy == "decompose":
            return self._rerank_decomposed_queries(
                queries,
                query_candidates,
                plan.subqueries,
            )
        try:
            with timed_stage("retrieval.rerank"), model_usage_stage("reranking"):
                reranked = list(
                    self.reranker.rerank(plan.rerank_query(question), candidates, self.top_k)
                )
            if not reranked:
                raise RuntimeError("Reranker returned no results")
            seen = {hit.chunk_id for hit in reranked}
            reranked.extend(hit for hit in candidates if hit.chunk_id not in seen)
            return reranked[:self.top_k]
        except Exception:
            logger.warning(
                "Reranking failed; falling back to fused retrieval order",
                exc_info=True,
            )
            return candidates[:self.top_k]


    def _rerank_decomposed_queries(
        self,
        queries: list[str],
        query_candidates: list[list[SearchHit]],
        subqueries: Sequence[str],
    ) -> list[SearchHit]:
        candidates_by_query = {
            query.casefold(): candidates
            for query, candidates in zip(queries, query_candidates)
        }
        ranked_groups: list[list[SearchHit]] = []
        reranked_query_keys: set[str] = set()
        for query in subqueries:
            query_key = query.casefold()
            if query_key in reranked_query_keys:
                continue
            reranked_query_keys.add(query_key)
            candidates = candidates_by_query.get(query_key, [])
            if not candidates:
                continue
            ranked_groups.append(self._rerank_one_query(query, candidates))

        # Once at least one subquery has usable evidence, keep the generation
        # context focused on those independent targets. The original query is
        # still retrieved above and is used as the all-groups-empty fallback.
        if not ranked_groups:
            return _multi_query_fusion(
                query_candidates,
                limit=max(self.candidate_k, self.top_k * 6),
            )[:self.top_k]
        return _interleave_ranked_groups(ranked_groups)

    def _rerank_one_query(
        self,
        query: str,
        candidates: list[SearchHit],
    ) -> list[SearchHit]:
        try:
            with timed_stage("retrieval.rerank"), model_usage_stage("reranking"):
                reranked = list(self.reranker.rerank(query, candidates, self.top_k))
            if not reranked:
                raise RuntimeError("Reranker returned no results")
            candidate_ids = {hit.chunk_id for hit in candidates}
            seen: set[str] = set()
            valid_reranked = []
            for hit in reranked:
                hit_id = str(hit.chunk_id)
                if hit_id in candidate_ids and hit_id not in seen:
                    seen.add(hit_id)
                    valid_reranked.append(hit)
            if not valid_reranked:
                raise RuntimeError("Reranker returned no candidate results")
            valid_reranked.extend(hit for hit in candidates if hit.chunk_id not in seen)
            return valid_reranked[:self.top_k]
        except Exception:
            logger.warning(
                "Reranking failed for decomposed query; falling back to its retrieval order query=%r",
                query,
                exc_info=True,
            )
            return candidates[:self.top_k]
    def _retrieve_for_query(
        self,
        knowledge_base_id: str,
        query_vector: list[float],
        question: str,
    ) -> list[SearchHit]:
        keywords = extract_keyword_terms(question)
        document_vector_hits: list[SearchHit] = []
        document_keyword_hits: list[SearchHit] = []
        document_ids: list[str] = []
        search_documents = getattr(self.vectors, "search_documents", None)
        if callable(search_documents):
            with timed_stage("retrieval.document_vector_search"):
                document_vector_hits = _best_document_hits(list(
                    search_documents(knowledge_base_id, query_vector, self.document_candidate_k)
                ))
        search_keyword_documents = getattr(self.vectors, "search_keyword_documents", None)
        if keywords and callable(search_keyword_documents):
            with timed_stage("retrieval.document_keyword_search"):
                document_keyword_hits = _best_document_hits(list(
                    search_keyword_documents(knowledge_base_id, keywords, self.document_candidate_k)
                ))

        document_ids = [
            hit.document_id
            for hit in document_vector_hits
            if float(hit.score) >= self.document_score_threshold
        ]
        document_ids.extend(
            hit.document_id
            for hit in document_keyword_hits
            if hit.document_id not in document_ids
        )
        if document_ids:
            candidates = self._retrieve_chunks(
                knowledge_base_id, query_vector, keywords, document_ids
            )
        elif not document_vector_hits:
            candidates = self._retrieve_chunks(
                knowledge_base_id, query_vector, keywords, None
            )
        else:
            candidates = []
        if document_ids and not candidates:
            candidates = self._retrieve_chunks(
                knowledge_base_id, query_vector, keywords, None
            )
        return candidates

    def _retrieve_chunks(
        self,
        knowledge_base_id: str,
        query_vector: list[float],
        keywords: list[str],
        document_ids: list[str] | None,
    ) -> list[SearchHit]:
        with timed_stage("retrieval.chunk_vector_search"):
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
        with timed_stage("retrieval.chunk_keyword_search"):
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
        with timed_stage("retrieval.rrf_fusion"):
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


def _multi_query_fusion(
    query_results: list[list[SearchHit]],
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse per-query rankings while giving the original query a small priority."""
    if not query_results:
        return []
    if len(query_results) == 1:
        return query_results[0]
    hits_by_id: dict[str, SearchHit] = {}
    fused_scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for query_index, hits in enumerate(query_results):
        weight = 1.0 if query_index == 0 else 0.85
        for rank, hit in enumerate(hits, start=1):
            hit_id = str(hit.chunk_id)
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
    if limit is not None:
        ranked_ids = ranked_ids[:max(0, limit)]
    return [hits_by_id[hit_id] for hit_id in ranked_ids]



def _interleave_ranked_groups(groups: Sequence[Sequence[SearchHit]]) -> list[SearchHit]:
    """Interleave per-query rankings so each evidence target gets fair context space."""
    result: list[SearchHit] = []
    seen: set[str] = set()
    max_group_size = max((len(group) for group in groups), default=0)
    for rank in range(max_group_size):
        for group in groups:
            if rank >= len(group):
                continue
            hit = group[rank]
            hit_id = str(hit.chunk_id)
            if hit_id in seen:
                continue
            seen.add(hit_id)
            result.append(hit)
    return result
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
