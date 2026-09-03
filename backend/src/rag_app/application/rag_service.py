import logging
from time import perf_counter
from typing import Callable, TypeVar

from agent import (
    AnswerAgent,
    KnowledgeRetrievalAgent,
    RetrievalDecision,
    RetrievalDecisionAgent,
    QueryPlan,
)
from agent.context import format_file_lookup_answer
from agent.telemetry import collect_model_usage, collect_timing, timed_stage
from agent.query_intent import (
    analyze_query_intent,
)

from ..domain.models import Citation
from ..domain.ports import MetadataRepository


logger = logging.getLogger(__name__)
T = TypeVar("T")
# Kept for API compatibility; model-facing history is now token-budgeted.
CONTEXT_HISTORY_MESSAGE_LIMIT = 12


class RagStageError(RuntimeError):
    """Identifies the RAG stage that failed without discarding the root cause."""

    def __init__(self, stage: str, elapsed_seconds: float, cause: Exception):
        self.stage = stage
        self.elapsed_seconds = elapsed_seconds
        self.cause = cause
        super().__init__(
            f"{stage}失败（{elapsed_seconds:.2f} 秒）: {type(cause).__name__}: {cause}"
        )


def _run_stage(stage: str, operation: Callable[[], T]) -> T:
    started_at = perf_counter()
    try:
        with timed_stage(stage):
            result = operation()
    except Exception as exc:
        elapsed = perf_counter() - started_at
        logger.exception("RAG 阶段失败 stage=%s elapsed_seconds=%.3f", stage, elapsed)
        raise RagStageError(stage, elapsed, exc) from exc
    elapsed = perf_counter() - started_at
    logger.info("RAG 阶段完成 stage=%s elapsed_seconds=%.3f", stage, elapsed)
    return result


def _deduplicate_citations(citations: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for citation in citations:
        key = citation.get("chunk_id") or citation.get("document_id") or citation.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(citation)
    return result


class RagService:
    def __init__(
        self,
        repository: MetadataRepository,
        decision_agent: RetrievalDecisionAgent,
        retrieval_agent: KnowledgeRetrievalAgent,
        answer_agent: AnswerAgent,
    ):
        self.repository = repository
        self.decision_agent = decision_agent
        self.retrieval_agent = retrieval_agent
        self.answer_agent = answer_agent

    def answer(
        self,
        knowledge_base_id: str,
        conversation_id: str,
        question: str,
        model: str | None = None,
        attachment_context: str = "",
        attachment_citations: list[dict] | None = None,
        include_retrieved_content: bool = False,
        force_retrieval: bool = False,
    ) -> dict:
        started_at = perf_counter()
        with collect_timing() as timing, collect_model_usage() as collector:
            response = self._build_answer(
                knowledge_base_id,
                conversation_id,
                question,
                model=model,
                attachment_context=attachment_context,
                attachment_citations=attachment_citations,
                include_retrieved_content=include_retrieved_content,
                force_retrieval=force_retrieval,
            )
        response_time_ms = round((perf_counter() - started_at) * 1_000, 2)
        token_usage = collector.summary()
        timing_summary = timing.summary(response_time_ms)
        metrics = {
            "responseTimeMs": response_time_ms,
            "serverResponseTimeMs": response_time_ms,
            "tokenUsage": token_usage,
            "timing": timing_summary,
        }
        _run_stage(
            "保存回答",
            lambda: self.repository.add_message(
                conversation_id,
                knowledge_base_id,
                question,
                response["answer"],
                response["citations"],
                metrics,
            ),
        )
        response["response_time_ms"] = response_time_ms
        response["token_usage"] = token_usage
        response["timing"] = timing_summary
        return response

    def _build_answer(
        self,
        knowledge_base_id: str,
        conversation_id: str,
        question: str,
        model: str | None = None,
        attachment_context: str = "",
        attachment_citations: list[dict] | None = None,
        include_retrieved_content: bool = False,
        force_retrieval: bool = False,
    ) -> dict:
        knowledge_base = _run_stage(
            "读取知识库",
            lambda: self.repository.get_knowledge_base(knowledge_base_id),
        )
        if not knowledge_base:
            raise ValueError("知识库不存在")
        history = _run_stage(
            "读取对话历史",
            lambda: self.repository.list_messages(conversation_id),
        )

        knowledge_catalog = ""
        catalog_answer = ""
        intent = analyze_query_intent(question)
        file_lookup_question = intent.catalog_file_lookup and not attachment_context
        if file_lookup_question:
            documents = _run_stage(
                "metadata.list_documents",
                lambda: self.repository.list_documents(knowledge_base_id),
            )
            catalog_answer = _run_stage(
                "metadata.file_lookup_answer",
                lambda: format_file_lookup_answer(
                    question,
                    documents,
                    history,
                ),
            )

        decision = _run_stage(
            "检索判断",
            lambda: self.decision_agent.run(
                question,
                history,
                intent=intent,
                force_retrieval=force_retrieval,
            ),
        )
        if force_retrieval and not decision.should_retrieve:
            # Evaluation and explicit callers may require retrieval while
            # still needing the planner's decompose/rewrite query plan.
            decision = RetrievalDecision(
                True,
                decision.query_plan or QueryPlan.single(question),
            )
        retrieval_used = decision.should_retrieve
        hits = []
        query_plan = decision.query_plan or QueryPlan.single(question)
        if retrieval_used:
            hits = _run_stage(
                "问题向量化并执行相似度检索",
                lambda: self.retrieval_agent.run(
                    knowledge_base,
                    question,
                    query_plan=query_plan,
                ),
            )

        answer_result = _run_stage(
            "生成最终回答",
            lambda: self.answer_agent.run_with_context(
                question,
                history,
                hits,
                retrieval_used,
                model,
                knowledge_catalog=knowledge_catalog,
                catalog_answer=catalog_answer,
                attachment_context=attachment_context,
                intent=intent,
            ),
        )
        answer = answer_result.answer
        context_hits = answer_result.selected_hits
        citations = _deduplicate_citations([
            *[
                Citation(
                    document_id=hit.document_id,
                    chunk_id=hit.chunk_id,
                    title=hit.file_name or hit.title,
                    page_number=hit.page_number,
                    score=hit.score,
                    relevance_score=hit.relevance_score,
                    excerpt=hit.text[:500],
                    section_path=hit.section_path,
                    chunk_index=hit.chunk_index,
                ).as_dict()
                for hit in context_hits
            ],
            *list(attachment_citations or []),
        ])
        retrieval_queries = query_plan.retrieval_queries(question) if retrieval_used else []
        reranker_instance = self.retrieval_agent.reranker
        reranker_name = getattr(reranker_instance, "name", None)
        reranker_provider = getattr(reranker_instance, "provider_name", None)
        reranker_enabled = self.retrieval_agent.reranker is not None
        decompose_rerank = retrieval_used and reranker_enabled and query_plan.strategy == "decompose"
        response = {
            "conversation_id": conversation_id,
            "model": model or self.answer_agent.models.chat_model,
            "answer": answer,
            "citations": citations,
            "retrieval_used": retrieval_used,
            "retrieved_count": len(hits),
            "retrieval_k": self.retrieval_agent.top_k,
            "retrieval_k_per_query": self.retrieval_agent.top_k if query_plan.strategy == "decompose" else None,
            "retrieval_total_k": len(hits),
            "retrieval_candidate_k": self.retrieval_agent.candidate_k,
            "reranker": reranker_name,
            "reranker_provider": reranker_provider,
            "query_plan": query_plan.as_dict(),
            "retrieval_trace": {
                "strategy": query_plan.strategy,
                "queries": retrieval_queries,
                "query_count": len(retrieval_queries),
                "rerank_mode": (
                    "per_query" if decompose_rerank
                    else "fused" if retrieval_used and reranker_enabled
                    else "disabled"
                ),
                "rerank_query_count": (
                    len(query_plan.subqueries) if decompose_rerank
                    else 1 if retrieval_used and reranker_enabled
                    else 0
                ),
                "rerank_top_k": self.retrieval_agent.top_k if reranker_enabled else None,
                "reranker_provider": reranker_provider,
                "reranker_model": reranker_name,
            },
            "retrieved_document_ids": list(dict.fromkeys(
                hit.document_id for hit in hits if hit.document_id
            )),
            "retrieved_chunk_ids": list(dict.fromkeys(
                hit.chunk_id for hit in hits if hit.chunk_id
            )),
            "context_chunk_ids": [hit.chunk_id for hit in context_hits],
            "context_document_count": len({hit.document_id for hit in context_hits}),
            "context_trace": answer_result.context_trace,
            "catalog_used": bool(catalog_answer),
            "attachments_used": bool(attachment_context),
            "agent_trace": [
                {
                    "agent": self.decision_agent.name,
                    "status": "forced" if force_retrieval else "completed",
                    "outcome": decision.outcome,
                },
                {
                    "agent": self.retrieval_agent.name,
                    "status": "completed" if retrieval_used else "skipped",
                    "retrieved_count": len(hits),
                    "top_k": self.retrieval_agent.top_k,
                    "candidate_k": self.retrieval_agent.candidate_k,
                    "reranker": getattr(self.retrieval_agent.reranker, "name", None),
                    "reranker_provider": getattr(
                        self.retrieval_agent.reranker, "provider_name", None
                    ),
                },
                {
                    "agent": self.answer_agent.name,
                    "status": "completed",
                },
            ],
        }
        if include_retrieved_content:
            response["retrieved_chunks"] = [
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "title": hit.file_name or hit.title,
                    "page_number": hit.page_number,
                    "section_path": hit.section_path,
                    "chunk_index": hit.chunk_index,
                    "text": hit.text,
                }
                for hit in hits
            ]
        return response
