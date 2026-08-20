import logging
from time import perf_counter
from typing import Callable, TypeVar

from agent import AnswerAgent, KnowledgeRetrievalAgent, RetrievalDecisionAgent
from agent.context import format_knowledge_catalog, format_knowledge_catalog_answer
from agent.telemetry import capture_request_metrics
from agent.query_intent import (
    is_knowledge_catalog_file_lookup_question,
    is_knowledge_catalog_inventory_question,
    needs_knowledge_catalog,
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

    @capture_request_metrics
    def answer(
        self,
        knowledge_base_id: str,
        conversation_id: str,
        question: str,
        model: str | None = None,
        attachment_context: str = "",
        attachment_citations: list[dict] | None = None,
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
        file_lookup_question = is_knowledge_catalog_file_lookup_question(question) and not attachment_context
        inventory_question = is_knowledge_catalog_inventory_question(question) and not attachment_context
        if needs_knowledge_catalog(question) or inventory_question:
            documents = self.repository.list_documents(knowledge_base_id)
            knowledge_catalog = format_knowledge_catalog(documents)
            if inventory_question:
                catalog_answer = format_knowledge_catalog_answer(
                    question,
                    documents,
                    history,
                    file_lookup=file_lookup_question,
                )

        decision = _run_stage(
            "检索判断",
            lambda: self.decision_agent.run(question, history),
        )
        retrieval_used = decision.should_retrieve
        hits = []
        if retrieval_used:
            hits = _run_stage(
                "问题向量化并执行相似度检索",
                lambda: self.retrieval_agent.run(knowledge_base, question),
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
            ),
        )
        answer = answer_result.answer
        context_hits = answer_result.selected_hits
        citations = _deduplicate_citations([
            *[
                Citation(
                    hit.document_id,
                    hit.chunk_id,
                    hit.title,
                    hit.page_number,
                    hit.score,
                    hit.relevance_score,
                    hit.text[:500],
                ).as_dict()
                for hit in context_hits
            ],
            *list(attachment_citations or []),
        ])
        _run_stage(
            "保存回答",
            lambda: self.repository.add_message(
                conversation_id,
                knowledge_base_id,
                question,
                answer,
                citations,
            ),
        )
        return {
            "conversation_id": conversation_id,
            "model": model or self.answer_agent.models.chat_model,
            "answer": answer,
            "citations": citations,
            "retrieval_used": retrieval_used,
            "retrieved_count": len(hits),
            "retrieval_k": self.retrieval_agent.top_k,
            "retrieval_candidate_k": self.retrieval_agent.candidate_k,
            "reranker": getattr(self.retrieval_agent.reranker, "name", None),
            "retrieved_document_ids": list(dict.fromkeys(
                hit.document_id for hit in hits if hit.document_id
            )),
            "retrieved_chunk_ids": list(dict.fromkeys(
                hit.chunk_id for hit in hits if hit.chunk_id
            )),
            "context_chunk_ids": [hit.chunk_id for hit in context_hits],
            "context_document_count": len({hit.document_id for hit in context_hits}),
            "context_trace": answer_result.context_trace,
            "catalog_used": bool(knowledge_catalog),
            "attachments_used": bool(attachment_context),
            "agent_trace": [
                {
                    "agent": self.decision_agent.name,
                    "status": "completed",
                    "outcome": decision.outcome,
                },
                {
                    "agent": self.retrieval_agent.name,
                    "status": "completed" if retrieval_used else "skipped",
                    "retrieved_count": len(hits),
                    "top_k": self.retrieval_agent.top_k,
                    "candidate_k": self.retrieval_agent.candidate_k,
                    "reranker": getattr(self.retrieval_agent.reranker, "name", None),
                },
                {
                    "agent": self.answer_agent.name,
                    "status": "completed",
                },
            ],
        }
