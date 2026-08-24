import time
from typing import Any
import unicodedata

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from ...domain.ids import vector_point_id
from ...domain.models import SearchHit


class QdrantVectorStore:
    payload_index_fields = ("knowledge_base_id", "document_id", "node_id")
    text_index_field = "text"

    def __init__(
        self,
        url: str,
        collection: str,
        timeout_seconds: float = 30.0,
        upsert_batch_size: int = 32,
        upsert_max_retries: int = 2,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
        hnsw_full_scan_threshold: int = 10_000,
        search_hnsw_ef: int = 64,
    ):
        self.collection = collection
        self.client = QdrantClient(url=url, timeout=timeout_seconds)
        self.upsert_batch_size = max(1, upsert_batch_size)
        self.upsert_max_retries = max(0, upsert_max_retries)
        self.hnsw_m = max(4, hnsw_m)
        self.hnsw_ef_construct = max(self.hnsw_m, hnsw_ef_construct)
        self.hnsw_full_scan_threshold = max(0, hnsw_full_scan_threshold)
        self.search_hnsw_ef = max(1, search_hnsw_ef)
        self._collection_configured = False

    def check_connection(self) -> None:
        self.client.get_collections()
        if self.client.collection_exists(self.collection):
            self._configure_existing_collection(self.client.get_collection(self.collection))

    def ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                hnsw_config=self._hnsw_config(),
            )
            payload_schema = {}
        else:
            collection = self.client.get_collection(self.collection)
            if collection.config.params.vectors.size != vector_size:
                raise RuntimeError("Qdrant collection 向量维度与 embedding 模型不一致")
            if not self._collection_configured:
                self._configure_existing_collection(collection)
            return
        self._ensure_payload_indexes(payload_schema)
        self._collection_configured = True

    def _configure_existing_collection(self, collection: Any) -> None:
        if not self._uses_configured_hnsw(collection):
            self.client.update_collection(
                collection_name=self.collection,
                hnsw_config=self._hnsw_config(),
            )
        self._ensure_payload_indexes(collection.payload_schema or {})
        self._collection_configured = True

    def _hnsw_config(self) -> models.HnswConfigDiff:
        return models.HnswConfigDiff(
            m=self.hnsw_m,
            ef_construct=self.hnsw_ef_construct,
            full_scan_threshold=self.hnsw_full_scan_threshold,
        )

    def _uses_configured_hnsw(self, collection: Any) -> bool:
        current = getattr(getattr(collection, "config", None), "hnsw_config", None)
        return current is not None and all(
            getattr(current, field, None) == expected
            for field, expected in (
                ("m", self.hnsw_m),
                ("ef_construct", self.hnsw_ef_construct),
                ("full_scan_threshold", self.hnsw_full_scan_threshold),
            )
        )

    def _ensure_payload_indexes(self, payload_schema: Any) -> None:
        indexed_fields = set(payload_schema) if isinstance(payload_schema, dict) else set()
        for field in self.payload_index_fields:
            if field in indexed_fields:
                continue
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                # Index construction may take time for an existing large collection.
                wait=False,
            )
        if self.text_index_field not in indexed_fields:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=self.text_index_field,
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=1,
                lowercase=True,
                phrase_matching=True,
            ),
                wait=True,
            )

    def upsert(self, points: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        for start in range(0, len(points), self.upsert_batch_size):
            point_batch = points[start:start + self.upsert_batch_size]
            vector_batch = vectors[start:start + self.upsert_batch_size]
            for attempt in range(self.upsert_max_retries + 1):
                try:
                    self.client.upsert(
                        collection_name=self.collection,
                        points=models.Batch(
                            ids=[vector_point_id(point["chunk_id"]) for point in point_batch],
                            vectors=vector_batch,
                            payloads=point_batch,
                        ),
                        # Stable point IDs make retries idempotent. Acknowledgement is enough
                        # here; waiting for indexing can exceed the HTTP timeout under load.
                        wait=False,
                    )
                    break
                except ResponseHandlingException:
                    if attempt >= self.upsert_max_retries:
                        raise
                    time.sleep(0.5 * (2 ** attempt))

    def search(self, knowledge_base_id: str, vector: list[float], limit: int) -> list[SearchHit]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=models.Filter(must=[models.FieldCondition(key="knowledge_base_id", match=models.MatchValue(value=knowledge_base_id))]),
            search_params=models.SearchParams(hnsw_ef=self.search_hnsw_ef, exact=False),
            limit=limit,
            with_payload=True,
        )
        return [self._to_search_hit(point) for point in result.points]

    def search_keywords(
        self,
        knowledge_base_id: str,
        keywords: list[str],
        limit: int,
    ) -> list[SearchHit]:
        if not keywords or limit <= 0:
            return []
        conditions = [
            models.FieldCondition(
                key=self.text_index_field,
                match=models.MatchText(text=keyword),
            )
            for keyword in keywords
        ]
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="knowledge_base_id",
                    match=models.MatchValue(value=knowledge_base_id),
                )],
                min_should=models.MinShould(conditions=conditions, min_count=1),
            ),
            limit=min(256, max(64, limit * 16)),
            with_payload=True,
            with_vectors=False,
        )
        normalized_keywords = [self._normalize_lexical(keyword) for keyword in keywords]

        def keyword_score(point) -> int:
            text = self._normalize_lexical(str(point.payload.get("text") or ""))
            return sum(
                len(keyword) ** 2 + (100 if any(char.isdigit() for char in keyword) else 0)
                for keyword in normalized_keywords
                if keyword and keyword in text
            )

        ranked = sorted(
            points,
            key=lambda point: (
                keyword_score(point),
                str(point.payload.get("chunk_id") or ""),
            ),
            reverse=True,
        )[:limit]
        max_score = max((keyword_score(point) for point in ranked), default=1)
        return [
            self._to_search_hit(point, keyword_score(point) / max_score)
            for point in ranked
        ]

    @staticmethod
    def _normalize_lexical(value: str) -> str:
        return "".join(unicodedata.normalize("NFKC", value).lower().split())

    @staticmethod
    def _to_search_hit(point, score: float | None = None) -> SearchHit:
        payload = point.payload
        return SearchHit(
            chunk_id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            knowledge_base_id=str(payload["knowledge_base_id"]),
            title=str(payload["title"]),
            text=str(payload["text"]),
            folder_path=str(payload.get("folder_path", "")),
            page_number=payload.get("page_number"),
            score=float(point.score if score is None else score),
            file_name=str(payload.get("file_name") or payload["title"]),
            relative_path=str(
                payload.get("relative_path")
                or "/".join(
                    part.strip("/\\")
                    for part in (
                        str(payload.get("folder_path", "")),
                        str(payload.get("file_name") or payload["title"]),
                    )
                    if part.strip("/\\")
                )
            ),
            section_path=payload.get("section_path"),
            chunk_index=payload.get("chunk_index"),
        )

    def delete_document(self, document_id: str) -> None:
        self._delete_by_payload("document_id", document_id)

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        self._delete_by_payload("knowledge_base_id", knowledge_base_id)

    def _delete_by_payload(self, field: str, value: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))]
                )
            ),
            # Qdrant has accepted the durable delete operation at this point. Waiting for
            # every matching point to be physically removed makes large-document deletes
            # block the API until the HTTP client times out.
            wait=False,
        )
