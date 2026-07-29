import unittest
from unittest.mock import patch

from qdrant_client.http.exceptions import ResponseHandlingException

from rag_app.infrastructure.qdrant.vector_store import QdrantVectorStore


class QdrantVectorStoreTest(unittest.TestCase):
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
