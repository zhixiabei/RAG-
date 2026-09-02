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
_QUOTED_TITLE_PATTERN = re.compile(r"\u300a([^\u300b]{2,160})\u300b")
_FILE_NAME_PATTERN = re.compile(
    r"(?<![\w\u4e00-\u9fff])"
    r"[\w\u4e00-\u9fff][\w\u4e00-\u9fff./\\ ()\uFF08\uFF09_-]{1,159}"
    r"\.(?:docx?|pdf|xlsx?|pptx?|txt|csv|gdb|att|md)",
    re.IGNORECASE,
)
_NUMERIC_TERM_PATTERN = re.compile(
    r"(?<![\w\u4e00-\u9fff])"
    r"(?:(?:19\d{2}|20\d{2})\u5e74|\d+(?:\.\d+)?(?:%|\uFF05|\u4e07\u5143|\u5428|\u53e3|\u4e95|\u7c73|m|\u5929|\u4e2a\u6708))"
)
_SOURCE_CAPTURE_PATTERN = re.compile(r"\u300a([^\u300b]{2,160})\u300b")
_RRF_RANK_CONSTANT = 60
_KEYWORD_RRF_WEIGHT = 1.25
logger = logging.getLogger(__name__)


def _normalize_keyword(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"\s*([-./])\s*", r"\1", normalized)
    return re.sub(r"\s+", "", normalized)


def extract_keyword_terms(question: str, limit: int = 32) -> list[str]:
    """Extract source names, exact identifiers, and short phrases for lexical recall."""
    terms = []
    seen = set()

    def add(value: str) -> None:
        term = _normalize_keyword(value)
        if len(term) < 2 or term in seen:
            return
        seen.add(term)
        terms.append(term)

    normalized_question = unicodedata.normalize("NFKC", question)
    # Source names are high-value lexical anchors in cross-file questions. Add
    # them before generic Chinese sliding windows so they cannot be crowded out.
    for pattern in (_QUOTED_TITLE_PATTERN, _FILE_NAME_PATTERN):
        for match in pattern.finditer(normalized_question):
            add(match.group(1) if pattern is _QUOTED_TITLE_PATTERN else match.group(0))
            if len(terms) >= limit:
                return terms

    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(normalized_question):
            add(match.group(0))
            # The broad pattern can absorb a preceding Chinese verb such as
            # “比较化309-5”. Also add the compact identifier suffix so lexical
            # search can match the exact spreadsheet row.
            compact = _normalize_keyword(match.group(0))
            suffix_matches = list(re.finditer(
                r"(?=([A-Za-z\u4e00-\u9fff]{1,4}\d+(?:[-\u2010-\u2015./]\d+)+)$)",
                compact,
            ))
            suffix = suffix_matches[-1].group(1) if suffix_matches else None
            if suffix:
                add(suffix)
            if len(terms) >= limit:
                return terms

    for match in _NUMERIC_TERM_PATTERN.finditer(normalized_question):
        add(match.group(0))
        if len(terms) >= limit:
            return terms

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
        result_limit = self._result_limit(plan)
        # A decomposed question has multiple independent evidence targets. A
        # single rerank over the fused pool lets one target crowd out another,
        # so rank each subquery against its own candidates before interleaving
        # the results for the answer model. Do this before constructing any
        # cross-query fusion pool so the reranker never sees mixed intents.
        if plan.strategy == "decompose" and self.reranker is not None:
            return self._rerank_decomposed_queries(
                queries,
                query_candidates,
                plan.subqueries,
                result_limit,
            )

        candidates = _multi_query_fusion(
            query_candidates,
            limit=max(self.candidate_k, self.top_k * 6),
        )
        if self.reranker is None or not candidates:
            return candidates[:result_limit]
        return self._rerank_one_query(plan.rerank_query(question), candidates)


    def _rerank_decomposed_queries(
        self,
        queries: list[str],
        query_candidates: list[list[SearchHit]],
        subqueries: Sequence[str],
        result_limit: int,
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
            )[:result_limit]
        return _interleave_ranked_groups(ranked_groups, result_limit)

    def _result_limit(self, plan: QueryPlan) -> int:
        if plan.strategy != "decompose":
            return self.top_k
        # Keep a minimum evidence budget for every independent target. Cap at
        # three top-k blocks so four-way questions do not flood the answer
        # context with low-ranked duplicates.
        subquery_count = max(2, len(plan.subqueries))
        return self.top_k * min(3, subquery_count)

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
            candidate_by_id = {str(hit.chunk_id): hit for hit in candidates}
            source_anchors = []
            query_terms = extract_keyword_terms(query, limit=64)
            ranked_by_id = {str(hit.chunk_id): hit for hit in valid_reranked}
            source_names = _SOURCE_CAPTURE_PATTERN.findall(query)
            anchor_limit = max(
                1,
                min(3, self.top_k // max(1, len(source_names))),
            )
            for source_name in source_names:
                matching_hits = [
                    candidate_by_id[str(hit.chunk_id)]
                    for hit in candidates
                    if _hit_matches_source_name(hit, source_name)
                ]
                matching_hits.sort(
                    key=lambda hit: _source_anchor_score(hit, query_terms),
                    reverse=True,
                )
                for anchor in matching_hits[:anchor_limit]:
                    source_anchors.append(
                        ranked_by_id.get(str(anchor.chunk_id), anchor)
                    )
            ordered = []
            ordered_ids = set()
            # Explicit source names are hard evidence about which document the
            # user meant. Keep one matching chunk at the front even if a noisy
            # cross-document reranker scores it below its top-N cutoff.
            for hit in source_anchors:
                hit_id = str(hit.chunk_id)
                if hit_id not in ordered_ids:
                    ordered.append(hit)
                    ordered_ids.add(hit_id)
            for hit in valid_reranked:
                hit_id = str(hit.chunk_id)
                if hit_id not in ordered_ids:
                    ordered.append(hit)
                    ordered_ids.add(hit_id)
            for hit in candidates:
                hit_id = str(hit.chunk_id)
                if hit_id not in ordered_ids:
                    ordered.append(hit)
                    ordered_ids.add(hit_id)
            return ordered[:self.top_k]
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

        # Document routing is a recall boost, not a gate. A profile can miss a
        # row-level identifier (especially in large spreadsheets), while the
        # chunk index can still contain the exact answer. Always keep the
        # global search and fuse routed candidates into it when available.
        global_candidates = self._retrieve_chunks(
            knowledge_base_id, query_vector, keywords, None
        )
        if document_ids:
            routed_candidates = self._retrieve_chunks(
                knowledge_base_id, query_vector, keywords, document_ids
            )
            return _multi_query_fusion(
                [global_candidates, routed_candidates],
                limit=max(self.candidate_k, self.top_k * 6),
            )
        return global_candidates

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

        # Exact identifiers are often the only reliable signal for row-level
        # facts. Give lexical search a pool as large as dense search so a
        # common phrase in another file cannot evict the exact row.
        keyword_limit = max(1, self.candidate_k)
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



def _interleave_ranked_groups(
    groups: Sequence[Sequence[SearchHit]],
    limit: int | None = None,
) -> list[SearchHit]:
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
            if limit is not None and len(result) >= limit:
                return result
    return result


def _best_document_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Deduplicate route nodes and rank documents by their best route score."""
    best_by_document: dict[str, SearchHit] = {}
    for hit in hits:
        document_id = str(hit.document_id)
        current = best_by_document.get(document_id)
        if current is None or float(hit.score) > float(current.score):
            best_by_document[document_id] = hit
    return sorted(
        best_by_document.values(),
        key=lambda hit: (-float(hit.score), str(hit.document_id)),
    )


def _hit_matches_source_name(hit: SearchHit, source_name: str) -> bool:
    file_name = _normalize_keyword(str(hit.file_name or hit.title or ""))
    return _normalize_keyword(source_name) in file_name


def _source_anchor_score(hit: SearchHit, query_terms: Sequence[str]) -> tuple[int, float, float]:
    text = _normalize_keyword(str(hit.text or ""))
    lexical_score = sum(
        len(term) ** 2
        for term in query_terms
        if len(term) >= 3 and _normalize_keyword(term) in text
    )
    return (
        lexical_score,
        float(hit.relevance_score if hit.relevance_score is not None else -1.0),
        float(hit.score),
    )
