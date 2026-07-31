import unittest

from agent.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from rag_app.domain.models import SearchHit


class FakeModels:
    embedding_model = "test-embedding"

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeVectors:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, knowledge_base_id, vector, limit):
        self.calls.append((knowledge_base_id, vector, limit))
        return self.hits


class KnowledgeRetrievalAgentTest(unittest.TestCase):
    def test_returns_similarity_ordered_top_k_candidates(self):
        hits = [
            SearchHit(f"chunk-{index}", "doc-1", "kb-1", "制度.pdf", f"内容 {index}", 0.9)
            for index in range(4)
        ]
        vectors = FakeVectors(hits)
        agent = KnowledgeRetrievalAgent(vectors, FakeModels(), top_k=2)

        result = agent.run({"id": "kb-1", "embedding_model": "test-embedding"}, "报销制度")

        self.assertEqual(len(result), 2)
        self.assertEqual(vectors.calls, [("kb-1", [0.1, 0.2], 2)])

    def test_rejects_incompatible_embedding_model(self):
        agent = KnowledgeRetrievalAgent(FakeVectors([]), FakeModels(), 2)

        with self.assertRaisesRegex(RuntimeError, "embedding 模型"):
            agent.run({"id": "kb-1", "embedding_model": "old-embedding"}, "问题")


if __name__ == "__main__":
    unittest.main()
