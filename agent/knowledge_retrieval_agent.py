from typing import Any

from .contracts import ModelGateway, SearchHit, VectorStore


class KnowledgeRetrievalAgent:
    """Embeds a question and returns its nearest knowledge-base chunks."""

    name = "knowledge_retrieval"

    def __init__(self, vectors: VectorStore, models: ModelGateway, top_k: int):
        if top_k <= 0:
            raise ValueError("Top-K 必须大于 0")
        self.vectors = vectors
        self.models = models
        self.top_k = top_k

    def run(self, knowledge_base: dict[str, Any], question: str) -> list[SearchHit]:
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 "
                f"{self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        query_vector = self.models.embed([question])[0]
        return list(self.vectors.search(knowledge_base["id"], query_vector, self.top_k))[: self.top_k]
