from ..domain.models import Citation
from ..domain.ports import MetadataRepository
from agent import AnswerAgent, KnowledgeRetrievalAgent, RelevanceGradingAgent, RetrievalDecisionAgent


class RagService:
    def __init__(
        self,
        repository: MetadataRepository,
        decision_agent: RetrievalDecisionAgent,
        retrieval_agent: KnowledgeRetrievalAgent,
        relevance_agent: RelevanceGradingAgent,
        answer_agent: AnswerAgent,
    ):
        self.repository = repository
        self.decision_agent = decision_agent
        self.retrieval_agent = retrieval_agent
        self.relevance_agent = relevance_agent
        self.answer_agent = answer_agent

    def answer(self, knowledge_base_id: str, conversation_id: str, question: str, model: str | None = None) -> dict:
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if not knowledge_base:
            raise ValueError("知识库不存在")
        history = self.repository.list_messages(conversation_id)[-12:]
        decision = self.decision_agent.run(question, history)
        retrieval_used = decision.should_retrieve
        retrieved_hits = []
        hits = []
        relevance_result = None
        if retrieval_used:
            retrieved_hits = self.retrieval_agent.run(knowledge_base, question)
            relevance_result = self.relevance_agent.run(question, retrieved_hits)
            hits = list(relevance_result.relevant_hits)
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
        answer = self.answer_agent.run(question, history, hits, retrieval_used, model)
        self.repository.add_message(conversation_id, knowledge_base_id, question, answer, citations)
        return {
            "conversation_id": conversation_id,
            "model": model or self.answer_agent.models.chat_model,
            "answer": answer,
            "citations": citations,
            "retrieval_used": retrieval_used,
            "retrieved_count": len(retrieved_hits),
            "relevant_count": len(hits),
            "agent_trace": [
                {
                    "agent": self.decision_agent.name,
                    "status": "completed",
                    "outcome": decision.outcome,
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
                    "relevant_count": len(hits),
                    "threshold": self.relevance_agent.threshold,
                },
                {
                    "agent": self.answer_agent.name,
                    "status": "completed",
                },
            ],
        }
