import unittest

from agent.knowledge_retrieval_agent import KnowledgeRetrievalAgent, extract_keyword_terms
from rag_app.domain.models import SearchHit


class FakeModels:
    embedding_model = "test-embedding"

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeVectors:
    def __init__(self, hits, keyword_hits=None):
        self.hits = hits
        self.keyword_hits = keyword_hits or []
        self.calls = []
        self.keyword_calls = []

    def search(self, knowledge_base_id, vector, limit):
        self.calls.append((knowledge_base_id, vector, limit))
        return self.hits

    def search_keywords(self, knowledge_base_id, keywords, limit):
        self.keyword_calls.append((knowledge_base_id, keywords, limit))
        return self.keyword_hits


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
        self.assertEqual(vectors.keyword_calls[0][0], "kb-1")

    def test_prioritizes_normalized_keyword_hits_before_vector_hits(self):
        keyword_hit = SearchHit(
            "source:2", "source", "kb-1", "实例.docx", "化348-4 年累计增油 66.36", 1.0
        )
        vector_hit = SearchHit(
            "other:1", "other", "kb-1", "方案.doc", "其他油井产量", 0.9
        )
        vectors = FakeVectors([vector_hit], [keyword_hit])
        agent = KnowledgeRetrievalAgent(vectors, FakeModels(), top_k=2)

        result = agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "化 348-4 井的年累计增油是多少？",
        )

        self.assertEqual([hit.chunk_id for hit in result], ["source:2", "other:1"])
        self.assertIn("化348-4", vectors.keyword_calls[0][1])

    def test_extracts_identifier_and_chinese_phrase_terms(self):
        terms = extract_keyword_terms("化 348-4 井的年累计增油是多少？")

        self.assertIn("化348-4", terms)
        self.assertIn("年累计增油", terms)

    def test_rejects_incompatible_embedding_model(self):
        agent = KnowledgeRetrievalAgent(FakeVectors([]), FakeModels(), 2)

        with self.assertRaisesRegex(RuntimeError, "embedding 模型"):
            agent.run({"id": "kb-1", "embedding_model": "old-embedding"}, "问题")


if __name__ == "__main__":
    unittest.main()
