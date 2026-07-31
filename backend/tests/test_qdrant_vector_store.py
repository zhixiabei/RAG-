import unittest
from unittest.mock import patch

from qdrant_client.http.exceptions import ResponseHandlingException

from rag_app.infrastructure.qdrant.vector_store import QdrantVectorStore


class QdrantVectorStoreTest(unittest.TestCase):
    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_new_collection_uses_explicit_hnsw_and_payload_indexes(self, client_class):
        client = client_class.return_value
        client.collection_exists.return_value = False
        store = QdrantVectorStore(
            "http://qdrant:6333",
            "chunks",
            hnsw_m=24,
            hnsw_ef_construct=160,
            hnsw_full_scan_threshold=5000,
        )

        store.ensure_collection(1024)

        create_call = client.create_collection.call_args
        self.assertEqual(create_call.kwargs["vectors_config"].size, 1024)
        hnsw = create_call.kwargs["hnsw_config"]
        self.assertEqual(hnsw.m, 24)
        self.assertEqual(hnsw.ef_construct, 160)
        self.assertEqual(hnsw.full_scan_threshold, 5000)
        self.assertEqual(
            {call.kwargs["field_name"] for call in client.create_payload_index.call_args_list},
            {"knowledge_base_id", "document_id", "node_id"},
        )
        self.assertTrue(all(call.kwargs["wait"] is False for call in client.create_payload_index.call_args_list))

    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_search_uses_approximate_hnsw_with_configured_ef(self, client_class):
        client = client_class.return_value
        client.query_points.return_value.points = []
        store = QdrantVectorStore("http://qdrant:6333", "chunks", search_hnsw_ef=96)

        hits = store.search("kb-1", [0.1, 0.2], 12)

        self.assertEqual(hits, [])
        params = client.query_points.call_args.kwargs["search_params"]
        self.assertEqual(params.hnsw_ef, 96)
        self.assertFalse(params.exact)

    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_existing_collection_updates_changed_hnsw_and_only_missing_payload_indexes(self, client_class):
        client = client_class.return_value
        client.collection_exists.return_value = True
        collection = client.get_collection.return_value
        collection.config.params.vectors.size = 1024
        collection.config.hnsw_config.m = 16
        collection.config.hnsw_config.ef_construct = 100
        collection.config.hnsw_config.full_scan_threshold = 10_000
        collection.payload_schema = {"knowledge_base_id": object()}
        store = QdrantVectorStore(
            "http://qdrant:6333",
            "chunks",
            hnsw_m=16,
            hnsw_ef_construct=128,
            hnsw_full_scan_threshold=10_000,
        )

        store.ensure_collection(1024)

        update = client.update_collection.call_args.kwargs["hnsw_config"]
        self.assertEqual(update.m, 16)
        self.assertEqual(update.ef_construct, 128)
        self.assertEqual(update.full_scan_threshold, 10_000)
        self.assertEqual(
            {call.kwargs["field_name"] for call in client.create_payload_index.call_args_list},
            {"document_id", "node_id"},
        )

    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_existing_collection_keeps_matching_hnsw_config(self, client_class):
        client = client_class.return_value
        client.collection_exists.return_value = True
        collection = client.get_collection.return_value
        collection.config.params.vectors.size = 1024
        collection.config.hnsw_config.m = 16
        collection.config.hnsw_config.ef_construct = 128
        collection.config.hnsw_config.full_scan_threshold = 10_000
        collection.payload_schema = {
            "knowledge_base_id": object(),
            "document_id": object(),
            "node_id": object(),
        }
        store = QdrantVectorStore("http://qdrant:6333", "chunks")

        store.ensure_collection(1024)

        client.update_collection.assert_not_called()
        client.create_payload_index.assert_not_called()

    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_connection_check_configures_an_existing_collection(self, client_class):
        client = client_class.return_value
        client.collection_exists.return_value = True
        collection = client.get_collection.return_value
        collection.config.hnsw_config.m = 16
        collection.config.hnsw_config.ef_construct = 100
        collection.config.hnsw_config.full_scan_threshold = 10_000
        collection.payload_schema = {}
        store = QdrantVectorStore("http://qdrant:6333", "chunks")

        store.check_connection()

        client.get_collections.assert_called_once_with()
        client.update_collection.assert_called_once()
        self.assertEqual(client.create_payload_index.call_count, 3)

    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_document_delete_returns_after_qdrant_accepts_operation(self, client_class):
        client = client_class.return_value
        client.collection_exists.return_value = True
        store = QdrantVectorStore("http://qdrant:6333", "chunks", timeout_seconds=30)

        store.delete_document("doc-1")

        client_class.assert_called_once_with(url="http://qdrant:6333", timeout=30)
        client.delete.assert_called_once()
        call = client.delete.call_args
        self.assertEqual(call.kwargs["collection_name"], "chunks")
        self.assertFalse(call.kwargs["wait"])
        condition = call.kwargs["points_selector"].filter.must[0]
        self.assertEqual(condition.key, "document_id")
        self.assertEqual(condition.match.value, "doc-1")

    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_delete_is_a_noop_when_collection_does_not_exist(self, client_class):
        client = client_class.return_value
        client.collection_exists.return_value = False
        store = QdrantVectorStore("http://qdrant:6333", "chunks")

        store.delete_document("doc-1")

        client.delete.assert_not_called()

    @patch("rag_app.infrastructure.qdrant.vector_store.time.sleep")
    @patch("rag_app.infrastructure.qdrant.vector_store.QdrantClient")
    def test_upsert_retries_transport_timeout_with_stable_ids(self, client_class, sleep):
        client = client_class.return_value
        client.upsert.side_effect = [
            ResponseHandlingException(TimeoutError("timed out")),
            None,
            None,
        ]
        store = QdrantVectorStore(
            "http://qdrant:6333",
            "chunks",
            upsert_batch_size=2,
            upsert_max_retries=2,
        )
        points = [
            {"chunk_id": f"doc-1:{index}", "text": f"content {index}"}
            for index in range(3)
        ]
        vectors = [[0.1, 0.2] for _ in points]

        store.upsert(points, vectors)

        self.assertEqual(client.upsert.call_count, 3)
        self.assertTrue(all(call.kwargs["wait"] is False for call in client.upsert.call_args_list))
        first_ids = client.upsert.call_args_list[0].kwargs["points"].ids
        retry_ids = client.upsert.call_args_list[1].kwargs["points"].ids
        self.assertEqual(first_ids, retry_ids)
        self.assertEqual(len(client.upsert.call_args_list[2].kwargs["points"].ids), 1)
        sleep.assert_called_once_with(0.5)


if __name__ == "__main__":
    unittest.main()
