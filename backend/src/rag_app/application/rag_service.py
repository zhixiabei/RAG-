from ..domain.models import Citation
from ..domain.ports import MetadataRepository, ModelGateway, VectorStore


class RagService:
    def __init__(self, repository: MetadataRepository, vectors: VectorStore, models: ModelGateway, retrieval_top_k: int, context_top_k: int):
        self.repository = repository
        self.vectors = vectors
        self.models = models
        self.retrieval_top_k = retrieval_top_k
        self.context_top_k = context_top_k

    def answer(self, knowledge_base_id: str, question: str) -> dict:
        query_vector = self.models.embed([question])[0]
        hits = self.vectors.search(knowledge_base_id, query_vector, self.retrieval_top_k)[: self.context_top_k]
        citations = [Citation(hit.document_id, hit.chunk_id, hit.title, hit.page_number, hit.score).as_dict() for hit in hits]
        if hits:
            context = "\n\n".join(f"[文档] {hit.title}\n[页码] {hit.page_number or '未知'}\n[内容] {hit.text}" for hit in hits)
            answer = self.models.answer(question, context)
        else:
            answer = "知识库中没有足够信息回答这个问题。"
        self.repository.add_message(knowledge_base_id, question, answer, citations)
        return {"answer": answer, "citations": citations, "retrieved_count": len(hits)}

