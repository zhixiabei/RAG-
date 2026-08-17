import json
import unittest

import httpx

from agent.telemetry import collect_model_usage, model_usage_stage
from rag_app.domain.models import SearchHit
from rag_app.infrastructure.rerank import HttpReranker


class HttpRerankerTest(unittest.TestCase):
    def test_posts_candidates_and_returns_relevance_order(self):
        captured = {}

        def handler(request):
            captured["request"] = request
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.42},
                        {"index": 1, "relevance_score": 0.91},
                    ]
                },
            )

        gateway = HttpReranker(
            "SiliconFlow",
            "https://api.example/v1",
            "secret",
            "BAAI/bge-reranker-v2-m3",
            max_document_chars=200,
            transport=httpx.MockTransport(handler),
        )
        hits = [
            SearchHit(
                "chunk-1",
                "doc-1",
                "kb-1",
                "one.txt",
                "first candidate",
                0.9,
                page_number=2,
                relative_path="docs/one.txt",
            ),
            SearchHit(
                "chunk-2",
                "doc-2",
                "kb-1",
                "two.txt",
                "second candidate",
                0.8,
            ),
        ]

        try:
            with collect_model_usage() as collector, model_usage_stage("reranking"):
                result = gateway.rerank("policy question", hits, limit=2)
        finally:
            gateway.close()

        request = captured["request"]
        payload = captured["payload"]
        self.assertEqual(str(request.url), "https://api.example/v1/rerank")
        self.assertEqual(request.headers["authorization"], "Bearer secret")
        self.assertEqual(payload["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(payload["query"], "policy question")
        self.assertEqual(payload["top_n"], 2)
        self.assertFalse(payload["return_documents"])
        self.assertIn("Title: one.txt", payload["documents"][0])
        self.assertIn("Path: docs/one.txt", payload["documents"][0])
        self.assertIn("Page: 2", payload["documents"][0])
        self.assertIn("first candidate", payload["documents"][0])
        self.assertEqual([hit.chunk_id for hit in result], ["chunk-2", "chunk-1"])
        self.assertEqual([hit.relevance_score for hit in result], [0.91, 0.42])
        summary = collector.summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["unreported_calls"], 1)
        self.assertEqual(summary["by_stage"]["reranking"]["calls"], 1)

    def test_rejects_response_without_valid_scores(self):
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"results": [{"index": 99, "relevance_score": 1}]}
            )
        )
        gateway = HttpReranker(
            "Provider", "https://api.example/v1", "", "reranker", transport=transport
        )
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "one.txt", "text", 0.9)

        try:
            with self.assertRaisesRegex(RuntimeError, "no valid scores"):
                gateway.rerank("question", [hit], limit=1)
        finally:
            gateway.close()


if __name__ == "__main__":
    unittest.main()
