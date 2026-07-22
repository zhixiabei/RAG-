from typing import Any

from qdrant_client import QdrantClient, models

from ...domain.ids import vector_point_id
from ...domain.models import SearchHit


class QdrantVectorStore:
    def __init__(self, url: str, collection: str):
        self.collection = collection
        self.client = QdrantClient(url=url, timeout=3)

    def check_connection(self) -> None:
        self.client.get_collections()

    def ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            return
        collection = self.client.get_collection(self.collection)
        if collection.config.params.vectors.size != vector_size:
            raise RuntimeError("Qdrant collection 向量维度与 embedding 模型不一致")

    def upsert(self, points: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        batch_size = 256
        for start in range(0, len(points), batch_size):
            point_batch = points[start:start + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=models.Batch(
                    ids=[vector_point_id(point["chunk_id"]) for point in point_batch],
                    vectors=vectors[start:start + batch_size],
                    payloads=point_batch,
                ),
            )

    def search(self, knowledge_base_id: str, vector: list[float], limit: int) -> list[SearchHit]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=models.Filter(must=[models.FieldCondition(key="knowledge_base_id", match=models.MatchValue(value=knowledge_base_id))]),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                chunk_id=str(point.payload["chunk_id"]),
                document_id=str(point.payload["document_id"]),
                knowledge_base_id=str(point.payload["knowledge_base_id"]),
                title=str(point.payload["title"]),
                text=str(point.payload["text"]),
                page_number=point.payload.get("page_number"),
                score=float(point.score),
            )
            for point in result.points
        ]
