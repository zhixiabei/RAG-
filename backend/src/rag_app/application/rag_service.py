import re

from ..domain.models import Citation, SearchHit
from ..domain.ports import MetadataRepository
from agent import (
    AnswerAgent,
    ContextCompressionAgent,
    KnowledgeRetrievalAgent,
    RelevanceGradingAgent,
    RetrievalDecisionAgent,
)
from agent.context import format_knowledge_catalog, format_knowledge_catalog_answer
from agent.query_intent import (
    is_knowledge_catalog_file_lookup_question,
    is_knowledge_catalog_inventory_question,
    needs_knowledge_catalog,
)


def _lexical_search_terms(query: str) -> list[str]:
    terms = []
    terms.extend(match.group(0) for match in re.finditer(r"[^\s，。？！；：]+\.[A-Za-z0-9]{1,10}", query))
    for token in re.findall(r"[A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query):
        if token in {"什么", "怎么", "哪些", "是否", "这个", "文件", "文档", "资料", "内容", "知识库", "我问的是"}:
            continue
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            terms.extend(token[index:index + 2] for index in range(len(token) - 1))
    return list(dict.fromkeys(terms))[:16]


def _select_diverse_hits(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    """Prefer one chunk per document before filling remaining context slots."""
    if limit <= 0:
        return []
    selected = []
    deferred = []
    seen_documents = set()
    for hit in hits:
        if hit.document_id in seen_documents:
            deferred.append(hit)
            continue
        seen_documents.add(hit.document_id)
        selected.append(hit)
        if len(selected) >= limit:
            return selected
    selected.extend(deferred[: limit - len(selected)])
    return selected


def _deduplicate_citations(citations: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for citation in citations:
        key = citation.get("document_id") or citation.get("title") or citation.get("chunk_id")
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
        relevance_agent: RelevanceGradingAgent,
        compression_agent: ContextCompressionAgent,
        answer_agent: AnswerAgent,
    ):
        self.repository = repository
        self.decision_agent = decision_agent
        self.retrieval_agent = retrieval_agent
        self.relevance_agent = relevance_agent
        self.compression_agent = compression_agent
        self.answer_agent = answer_agent

    def answer(
        self,
        knowledge_base_id: str,
        conversation_id: str,
        question: str,
        model: str | None = None,
        attachment_context: str = "",
        attachment_citations: list[dict] | None = None,
    ) -> dict:
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if not knowledge_base:
            raise ValueError("知识库不存在")
        history = self.repository.list_messages(conversation_id)[-12:]
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
        decision = self.decision_agent.run(question, history)
        retrieval_used = decision.should_retrieve
        retrieved_hits = []
        relevant_hits = []
        hits = []
        relevance_result = None
        compression_result = None
        all_relevant_hits = []
        retrieval_fallback = None
        relevance_fallback = False
        lexical_chunk_ids: set[str] = set()
        search_query = question
        if retrieval_used:
            search_query = decision.search_query or question
            retrieved_hits = self.retrieval_agent.run(knowledge_base, search_query)
            lexical_hits = self.repository.search_document_chunks(
                knowledge_base_id,
                _lexical_search_terms(search_query),
                self.retrieval_agent.retrieval_top_k,
            )
            if lexical_hits:
                vector_chunk_ids = {hit.chunk_id for hit in retrieved_hits}
                lexical_chunk_ids = {hit.chunk_id for hit in lexical_hits}
                retrieved_hits = [
                    *lexical_hits,
                    *(hit for hit in retrieved_hits if hit.chunk_id not in lexical_chunk_ids),
                ][: self.retrieval_agent.retrieval_top_k]
                retrieval_fallback = "postgres_lexical" if not vector_chunk_ids else "postgres_lexical_augmented"
            relevance_result = self.relevance_agent.run(question, retrieved_hits, search_query)
            all_relevant_hits = list(relevance_result.relevant_hits)
            relevant_hits = _select_diverse_hits(all_relevant_hits, self.retrieval_agent.context_top_k)
            hits = relevant_hits
            if not hits and lexical_chunk_ids:
                hits = [
                    hit
                    for hit in retrieved_hits
                    if hit.chunk_id in lexical_chunk_ids and hit.score >= self.relevance_agent.threshold
                ]
                hits = _select_diverse_hits(hits, self.retrieval_agent.context_top_k)
                relevance_fallback = bool(hits)
            if hits:
                compression_result = self.compression_agent.run(question, hits, search_query)
                hits = [hit for hit in hits if hit.chunk_id in compression_result.kept_chunk_ids]
        citations = _deduplicate_citations([
            Citation(
                hit.document_id,
                hit.chunk_id,
                hit.title,
                hit.page_number,
                hit.score,
                relevance_result.score_for(hit.chunk_id) if relevance_result else 0.0,
            ).as_dict()
            for hit in hits
        ] + list(attachment_citations or []))
        answer = self.answer_agent.run(
            question,
            history,
            hits,
            retrieval_used,
            model,
            context_texts=compression_result.text_by_chunk_id if compression_result else None,
            knowledge_catalog=knowledge_catalog,
            catalog_answer=catalog_answer,
            attachment_context=attachment_context,
        )
        self.repository.add_message(conversation_id, knowledge_base_id, question, answer, citations)
        return {
            "conversation_id": conversation_id,
            "model": model or self.answer_agent.models.chat_model,
            "answer": answer,
            "citations": citations,
            "retrieval_used": retrieval_used,
            "retrieved_count": len(retrieved_hits),
            "relevant_count": len(all_relevant_hits),
            "relevant_document_count": len({hit.document_id for hit in all_relevant_hits}),
            "context_document_count": len({hit.document_id for hit in hits}),
            "retrieval_fallback": retrieval_fallback,
            "relevance_fallback": relevance_fallback,
            "catalog_used": bool(knowledge_catalog),
            "attachments_used": bool(attachment_context),
            "agent_trace": [
                {
                    "agent": self.decision_agent.name,
                    "status": "completed",
                    "outcome": decision.outcome,
                    "search_query": decision.search_query if retrieval_used else None,
                },
                {
                    "agent": self.retrieval_agent.name,
                    "status": "completed" if retrieval_used else "skipped",
                    "retrieved_count": len(retrieved_hits),
                    "fallback": retrieval_fallback,
                },
                {
                    "agent": self.relevance_agent.name,
                    "status": "completed" if retrieval_used else "skipped",
                    "candidate_count": len(retrieved_hits),
                    "relevant_count": len(all_relevant_hits),
                    "relevant_document_count": len({hit.document_id for hit in all_relevant_hits}),
                    "threshold": self.relevance_agent.threshold,
                    "grading_complete": relevance_result.grading_complete if relevance_result else True,
                    "fallback_used": relevance_fallback,
                },
                {
                    "agent": self.compression_agent.name,
                    "status": "completed" if compression_result else "skipped",
                    "triggered": compression_result.triggered if compression_result else False,
                    "original_chars": compression_result.original_chars if compression_result else 0,
                    "compressed_chars": compression_result.compressed_chars if compression_result else 0,
                    "max_chars": self.compression_agent.max_chars,
                    "kept_count": len(hits),
                },
                {
                    "agent": self.answer_agent.name,
                    "status": "completed",
                },
            ],
        }
