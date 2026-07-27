from ..domain.models import Citation
from ..domain.ports import MetadataRepository
from agent import (
    AnswerAgent,
    ContextCompressionAgent,
    KnowledgeRetrievalAgent,
    RelevanceGradingAgent,
    RetrievalDecisionAgent,
)
from agent.context import format_knowledge_catalog, format_knowledge_catalog_answer
from agent.query_intent import is_knowledge_catalog_inventory_question, needs_knowledge_catalog


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

    def answer(self, knowledge_base_id: str, conversation_id: str, question: str, model: str | None = None) -> dict:
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if not knowledge_base:
            raise ValueError("知识库不存在")
        history = self.repository.list_messages(conversation_id)[-12:]
        knowledge_catalog = ""
        catalog_answer = ""
        inventory_question = is_knowledge_catalog_inventory_question(question)
        if needs_knowledge_catalog(question) or inventory_question:
            documents = self.repository.list_documents(knowledge_base_id)
            knowledge_catalog = format_knowledge_catalog(documents)
            if inventory_question:
                catalog_answer = format_knowledge_catalog_answer(question, documents, history)
        decision = self.decision_agent.run(question, history)
        retrieval_used = decision.should_retrieve
        retrieved_hits = []
        relevant_hits = []
        hits = []
        relevance_result = None
        compression_result = None
        search_query = question
        if retrieval_used:
            search_query = decision.search_query or question
            retrieved_hits = self.retrieval_agent.run(knowledge_base, search_query)
            relevance_result = self.relevance_agent.run(question, retrieved_hits, search_query)
            relevant_hits = list(relevance_result.relevant_hits[: self.retrieval_agent.context_top_k])
            hits = relevant_hits
            if hits:
                compression_result = self.compression_agent.run(question, hits, search_query)
                hits = [hit for hit in hits if hit.chunk_id in compression_result.kept_chunk_ids]
        citations = [
            Citation(
                hit.document_id,
                hit.chunk_id,
                hit.title,
                hit.page_number,
                hit.score,
                relevance_result.score_for(hit.chunk_id) if relevance_result else 0.0,
            ).as_dict()
            for hit in hits
        ]
        answer = self.answer_agent.run(
            question,
            history,
            hits,
            retrieval_used,
            model,
            context_texts=compression_result.text_by_chunk_id if compression_result else None,
            knowledge_catalog=knowledge_catalog,
            catalog_answer=catalog_answer,
        )
        self.repository.add_message(conversation_id, knowledge_base_id, question, answer, citations)
        return {
            "conversation_id": conversation_id,
            "model": model or self.answer_agent.models.chat_model,
            "answer": answer,
            "citations": citations,
            "retrieval_used": retrieval_used,
            "retrieved_count": len(retrieved_hits),
            "relevant_count": len(relevant_hits),
            "catalog_used": bool(knowledge_catalog),
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
                },
                {
                    "agent": self.relevance_agent.name,
                    "status": "completed" if retrieval_used else "skipped",
                    "candidate_count": len(retrieved_hits),
                    "relevant_count": len(relevant_hits),
                    "threshold": self.relevance_agent.threshold,
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
