from ..domain.models import Citation
from ..domain.ports import MetadataRepository, ModelGateway, VectorStore


class RagService:
    def __init__(self, repository: MetadataRepository, vectors: VectorStore, models: ModelGateway, retrieval_top_k: int, context_top_k: int):
        self.repository = repository
        self.vectors = vectors
        self.models = models
        self.retrieval_top_k = retrieval_top_k
        self.context_top_k = context_top_k

    def answer(self, knowledge_base_id: str, conversation_id: str, question: str, model: str | None = None) -> dict:
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if not knowledge_base:
            raise ValueError("知识库不存在")
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 {self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        history = self.repository.list_messages(conversation_id)[-12:]
        query_vector = self.models.embed([question])[0]
        hits = self.vectors.search(knowledge_base_id, query_vector, self.retrieval_top_k)[: self.context_top_k]
        citations = [Citation(hit.document_id, hit.chunk_id, hit.title, hit.page_number, hit.score).as_dict() for hit in hits]
        if hits:
            context = "\n\n".join(f"[文档] {hit.title}\n[页码] {hit.page_number or '未知'}\n[内容] {hit.text}" for hit in hits)
            answer = self.models.answer(question, context, history, model)
        elif history:
            answer = self.models.answer(question, "本轮未检索到新的相关文档片段，请仅依据历史对话中已有的信息回答。", history, model)
        else:
            answer = "知识库中没有足够信息回答这个问题。"
        self.repository.add_message(conversation_id, knowledge_base_id, question, answer, citations)
        return {
            "conversation_id": conversation_id,
            "model": model or self.models.chat_model,
            "answer": answer,
            "citations": citations,
            "retrieved_count": len(hits),
        }
