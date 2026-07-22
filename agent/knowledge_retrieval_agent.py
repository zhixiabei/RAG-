from typing import Any

from .contracts import ModelGateway, SearchHit, VectorStore


class KnowledgeRetrievalAgent:
    """Embeds a question and retrieves the bounded context used for answering."""

    name = "knowledge_retrieval"

    def __init__(self, vectors: VectorStore, models: ModelGateway, retrieval_top_k: int, context_top_k: int):
        self.vectors = vectors
        self.models = models
        self.retrieval_top_k = retrieval_top_k
        self.context_top_k = context_top_k

    def run(self, knowledge_base: dict[str, Any], question: str) -> list[SearchHit]:
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 "
                f"{self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        query_vector = self.models.embed([question])[0]
        hits = self.vectors.search(knowledge_base["id"], query_vector, self.retrieval_top_k)
        return list(hits[: self.context_top_k])
