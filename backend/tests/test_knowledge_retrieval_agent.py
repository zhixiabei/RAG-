import unittest

from agent.knowledge_retrieval_agent import KnowledgeRetrievalAgent, extract_keyword_terms
from agent.query_planning_agent import QueryPlan
from rag_app.domain.models import SearchHit


class FakeModels:
    embedding_model = "test-embedding"

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeVectors:
    def __init__(
        self,
        hits,
        keyword_hits=None,
        document_hits=None,
        document_keyword_hits=None,
    ):
        self.hits = hits
        self.keyword_hits = keyword_hits or []
        self.document_hits = document_hits or []
        self.document_keyword_hits = document_keyword_hits or []
        self.calls = []
        self.keyword_calls = []
        self.document_calls = []
        self.document_keyword_calls = []

    def search(self, knowledge_base_id, vector, limit, document_ids=None):
        call = (knowledge_base_id, vector, limit)
        if document_ids is not None:
            call = (*call, tuple(document_ids))
        self.calls.append(call)
        if document_ids is None:
            return self.hits
        allowed = set(document_ids)
        return [hit for hit in self.hits if hit.document_id in allowed]

    def search_keywords(self, knowledge_base_id, keywords, limit, document_ids=None):
        call = (knowledge_base_id, keywords, limit)
        if document_ids is not None:
            call = (*call, tuple(document_ids))
        self.keyword_calls.append(call)
        if document_ids is None:
            return self.keyword_hits
        allowed = set(document_ids)
        return [hit for hit in self.keyword_hits if hit.document_id in allowed]

    def search_documents(self, knowledge_base_id, vector, limit):
        self.document_calls.append((knowledge_base_id, vector, limit))
        return self.document_hits

    def search_keyword_documents(self, knowledge_base_id, keywords, limit):
        self.document_keyword_calls.append((knowledge_base_id, keywords, limit))
        return self.document_keyword_hits


class FakeReranker:
    name = "test-reranker"

    def __init__(self, result=None, error=None):
        self.result = result or []
        self.error = error
        self.calls = []

    def rerank(self, query, hits, limit):
        self.calls.append((query, list(hits), limit))
        if self.error:
            raise self.error
        return self.result


class KnowledgeRetrievalAgentTest(unittest.TestCase):
    def test_batches_planned_queries_and_reranks_once(self):
        shared = SearchHit("shared", "doc-1", "kb-1", "hse.txt", "共同要求", 0.8)
        emergency = SearchHit("emergency", "doc-1", "kb-1", "hse.txt", "井喷处置", 0.9)
        environment = SearchHit("environment", "doc-2", "kb-1", "env.txt", "污染物管理", 0.9)

        class PlannedModels:
            embedding_model = "test-embedding"

            def __init__(self):
                self.calls = []

            def embed(self, texts):
                self.calls.append(texts)
                return [[float(index), 0.2] for index, _text in enumerate(texts)]

        class PlannedVectors:
            def __init__(self):
                self.calls = []

            def search(self, knowledge_base_id, vector, limit, document_ids=None):
                index = int(vector[0])
                self.calls.append((knowledge_base_id, index, limit))
                return {
                    0: [shared],
                    1: [shared, emergency],
                    2: [environment],
                }[index]

            def search_keywords(self, knowledge_base_id, keywords, limit, document_ids=None):
                return []

        models = PlannedModels()
        vectors = PlannedVectors()
        reranker = FakeReranker([emergency, environment])
        agent = KnowledgeRetrievalAgent(
            vectors, models, top_k=2, candidate_k=3, reranker=reranker
        )
        question = "综合说明井喷处置和污染物管理要求"
        plan = QueryPlan(
            "decompose", question, ("井喷处置要求", "污染物管理要求"),
            trigger="complex_query",
        )

        result = agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            question,
            query_plan=plan,
        )

        self.assertEqual(models.calls, [[question, "井喷处置要求", "污染物管理要求"]])
        self.assertEqual(len(vectors.calls), 3)
        self.assertEqual(len(reranker.calls), 1)
        self.assertEqual([hit.chunk_id for hit in reranker.calls[0][1]], [
            "shared", "environment", "emergency",
        ])
        self.assertEqual([hit.chunk_id for hit in result], ["emergency", "environment"])
        self.assertEqual(len(plan.retrieval_queries(question)), 3)

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

    def test_fusion_promotes_a_chunk_recalled_by_both_search_paths(self):
        vector_only = SearchHit(
            "vector:1", "doc-1", "kb-1", "方案.doc", "向量命中", 0.95
        )
        shared = SearchHit(
            "shared:1", "doc-2", "kb-1", "实例.docx", "化348-4 年累计增油", 0.80
        )
        vectors = FakeVectors([vector_only, shared], [shared])
        agent = KnowledgeRetrievalAgent(vectors, FakeModels(), top_k=2)

        result = agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "化348-4 年累计增油",
        )

        self.assertEqual([hit.chunk_id for hit in result], ["shared:1", "vector:1"])

    def test_widens_candidates_and_uses_reranked_order(self):
        vector_hits = [
            SearchHit("vector:1", "doc-1", "kb-1", "one.txt", "first", 0.9),
            SearchHit("vector:2", "doc-2", "kb-1", "two.txt", "second", 0.8),
            SearchHit("vector:3", "doc-3", "kb-1", "three.txt", "third", 0.7),
        ]
        keyword_hit = SearchHit(
            "keyword:1", "doc-4", "kb-1", "keyword.txt", "Policy 12-3", 1.0
        )
        reranker = FakeReranker([vector_hits[2]])
        vectors = FakeVectors(vector_hits, [keyword_hit])
        agent = KnowledgeRetrievalAgent(
            vectors,
            FakeModels(),
            top_k=2,
            candidate_k=4,
            reranker=reranker,
        )

        result = agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "Policy 12-3 details",
        )

        self.assertEqual(vectors.calls, [("kb-1", [0.1, 0.2], 4)])
        self.assertEqual(vectors.keyword_calls[0][2], 2)
        query, candidates, limit = reranker.calls[0]
        self.assertEqual(query, "Policy 12-3 details")
        self.assertEqual(limit, 2)
        self.assertEqual(
            [hit.chunk_id for hit in candidates],
            ["keyword:1", "vector:1", "vector:2", "vector:3"],
        )
        self.assertEqual(
            [hit.chunk_id for hit in result],
            ["vector:3", "keyword:1"],
        )

    def test_falls_back_to_fused_order_when_reranker_fails(self):
        vector_hits = [
            SearchHit("vector:1", "doc-1", "kb-1", "one.txt", "first", 0.9),
            SearchHit("vector:2", "doc-2", "kb-1", "two.txt", "second", 0.8),
        ]
        keyword_hit = SearchHit(
            "keyword:1", "doc-3", "kb-1", "keyword.txt", "Policy 12-3", 1.0
        )
        vectors = FakeVectors(vector_hits, [keyword_hit])
        reranker = FakeReranker(error=TimeoutError("timed out"))
        agent = KnowledgeRetrievalAgent(
            vectors, FakeModels(), top_k=2, candidate_k=3, reranker=reranker
        )

        with self.assertLogs("agent.knowledge_retrieval_agent", level="WARNING"):
            result = agent.run(
                {"id": "kb-1", "embedding_model": "test-embedding"},
                "Policy 12-3 details",
            )

        self.assertEqual(
            [hit.chunk_id for hit in result],
            ["keyword:1", "vector:1"],
        )

    def test_rejects_candidate_k_smaller_than_top_k(self):
        with self.assertRaisesRegex(ValueError, "Candidate-K"):
            KnowledgeRetrievalAgent(
                FakeVectors([]), FakeModels(), top_k=3, candidate_k=2
            )

    def test_extracts_identifier_and_chinese_phrase_terms(self):
        terms = extract_keyword_terms("化 348-4 井的年累计增油是多少？")

        self.assertIn("化348-4", terms)
        self.assertIn("年累计增油", terms)

    def test_rejects_incompatible_embedding_model(self):
        agent = KnowledgeRetrievalAgent(FakeVectors([]), FakeModels(), 2)

        with self.assertRaisesRegex(RuntimeError, "embedding 模型"):
            agent.run({"id": "kb-1", "embedding_model": "old-embedding"}, "问题")

    def test_document_threshold_filters_without_reranking(self):
        chunks = [
            SearchHit("doc-1:1", "doc-1", "kb-1", "one.txt", "目标内容", 0.9),
            SearchHit("doc-2:1", "doc-2", "kb-1", "two.txt", "其他内容", 0.8),
        ]
        document_hits = [
            SearchHit("document:doc-1", "doc-1", "kb-1", "one.txt", "", 0.92),
            SearchHit("document:doc-2", "doc-2", "kb-1", "two.txt", "", 0.40),
        ]
        vectors = FakeVectors(chunks, document_hits=document_hits)
        agent = KnowledgeRetrievalAgent(
            vectors,
            FakeModels(),
            top_k=2,
            candidate_k=4,
            document_candidate_k=20,
            document_score_threshold=0.45,
        )

        result = agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "目标内容是什么？",
        )

        self.assertEqual([hit.chunk_id for hit in result], ["doc-1:1"])
        self.assertEqual(vectors.document_calls, [("kb-1", [0.1, 0.2], 20)])
        self.assertEqual(vectors.calls, [("kb-1", [0.1, 0.2], 4, ("doc-1",))])

    def test_all_documents_above_threshold_enter_one_chunk_search(self):
        chunks = [
            SearchHit(f"doc-{index}:1", f"doc-{index}", "kb-1", f"{index}.txt", "内容", 0.9)
            for index in range(1, 4)
        ]
        document_hits = [
            SearchHit(
                f"document:doc-{index}",
                f"doc-{index}",
                "kb-1",
                f"{index}.txt",
                "",
                score,
            )
            for index, score in ((1, 0.90), (2, 0.88), (3, 0.86))
        ]
        vectors = FakeVectors(chunks, document_hits=document_hits)
        agent = KnowledgeRetrievalAgent(
            vectors,
            FakeModels(),
            top_k=3,
            candidate_k=4,
            document_candidate_k=50,
            document_score_threshold=0.45,
        )

        agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "比较这些内容",
        )

        self.assertEqual(
            vectors.calls,
            [("kb-1", [0.1, 0.2], 4, ("doc-1", "doc-2", "doc-3"))],
        )
        self.assertEqual(vectors.document_calls, [("kb-1", [0.1, 0.2], 50)])

    def test_document_threshold_keeps_retriever_order(self):
        chunks = [
            SearchHit("doc-2:1", "doc-2", "kb-1", "two.txt", "second", 0.8),
            SearchHit("doc-1:1", "doc-1", "kb-1", "one.txt", "first", 0.9),
        ]
        document_hits = [
            SearchHit("document:doc-2", "doc-2", "kb-1", "two.txt", "", 0.8),
            SearchHit("document:doc-1", "doc-1", "kb-1", "one.txt", "", 0.9),
        ]
        vectors = FakeVectors(chunks, document_hits=document_hits)
        agent = KnowledgeRetrievalAgent(
            vectors,
            FakeModels(),
            top_k=2,
            candidate_k=4,
            document_candidate_k=50,
            document_score_threshold=0.45,
        )

        agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "compare documents",
        )

        self.assertEqual(
            vectors.calls,
            [("kb-1", [0.1, 0.2], 4, ("doc-2", "doc-1"))],
        )

    def test_documents_below_threshold_do_not_bypass_filter(self):
        global_hit = SearchHit(
            "doc-low:1", "doc-low", "kb-1", "low.txt", "content", 0.9
        )
        document_hit = SearchHit(
            "document:doc-low", "doc-low", "kb-1", "low.txt", "", 0.44
        )
        vectors = FakeVectors([global_hit], document_hits=[document_hit])
        agent = KnowledgeRetrievalAgent(
            vectors,
            FakeModels(),
            top_k=1,
            document_candidate_k=50,
            document_score_threshold=0.45,
        )

        result = agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "unrelated question",
        )

        self.assertEqual(result, [])
        self.assertEqual(vectors.calls, [])
        self.assertEqual(vectors.document_keyword_calls, [])

    def test_empty_routed_search_falls_back_to_global_chunks(self):
        global_hit = SearchHit(
            "doc-source:1", "doc-source", "kb-1", "source.txt", "目标内容", 0.9
        )
        routed_document = SearchHit(
            "document:doc-stale", "doc-stale", "kb-1", "stale.txt", "", 0.95
        )
        vectors = FakeVectors([global_hit], document_hits=[routed_document])
        agent = KnowledgeRetrievalAgent(
            vectors,
            FakeModels(),
            top_k=1,
            document_candidate_k=50,
            document_score_threshold=0.45,
        )

        result = agent.run(
            {"id": "kb-1", "embedding_model": "test-embedding"},
            "目标内容",
        )

        self.assertEqual([hit.chunk_id for hit in result], ["doc-source:1"])
        self.assertEqual(
            vectors.calls,
            [
                ("kb-1", [0.1, 0.2], 1, ("doc-stale",)),
                ("kb-1", [0.1, 0.2], 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
