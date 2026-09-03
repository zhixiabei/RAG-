from dataclasses import replace
import json
import unittest

from agent import (
    AnswerAgent,
    KnowledgeRetrievalAgent,
    RetrievalDecisionAgent,
)
from rag_app.application.rag_service import RagService, RagStageError
from rag_app.domain.models import SearchHit


class FakeRepository:
    def __init__(self, history=None):
        self.history = history or []
        self.saved = []
        self.documents = []

    def get_knowledge_base(self, knowledge_base_id):
        return {"id": knowledge_base_id, "embedding_model": "test-embedding"}

    def list_messages(self, conversation_id):
        return self.history

    def list_documents(self, knowledge_base_id):
        return self.documents

    def add_message(self, conversation_id, knowledge_base_id, question, answer, citations, metrics=None):
        self.saved.append((conversation_id, knowledge_base_id, question, answer, citations, metrics))


class FakeVectorStore:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.search_calls = []

    def search(self, knowledge_base_id, vector, limit):
        self.search_calls.append((knowledge_base_id, vector, limit))
        return self.hits[:limit]

    def search_keywords(self, knowledge_base_id, keywords, limit):
        return []


class FakeModelGateway:
    chat_model = "test-chat"
    embedding_model = "test-embedding"

    def __init__(self, retrieval_needed, planner_output=None, complexity="simple", needs_rewrite=False):
        self.retrieval_needed = retrieval_needed
        self.planner_output = planner_output
        self.complexity = complexity
        self.needs_rewrite = needs_rewrite
        self.embed_calls = []
        self.completion_calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.completion_calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        required = response_schema.get("required", []) if response_schema else []
        if required == ["decision"]:
            decision = "RETRIEVE" if self.retrieval_needed else "SKIP"
            return json.dumps({"decision": decision})
        if required == ["decision", "complexity", "needs_rewrite"]:
            return json.dumps({
                "decision": "RETRIEVE" if self.retrieval_needed else "SKIP",
                "complexity": self.complexity,
                "needs_rewrite": self.needs_rewrite,
            })
        if required == ["strategy", "standalone_query", "subqueries"]:
            return self.planner_output or json.dumps({
                "strategy": "single",
                "standalone_query": "",
                "subqueries": [],
            })
        return "测试回答"

    def embed(self, texts):
        self.embed_calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class RagServiceTest(unittest.TestCase):
    @staticmethod
    def build_service(
        repository, vectors, models, top_k=3, query_planning=False, reranker=None
    ):
        return RagService(
            repository,
            RetrievalDecisionAgent(models, query_planning_enabled=query_planning),
            KnowledgeRetrievalAgent(vectors, models, top_k=top_k, reranker=reranker),
            AnswerAgent(models),
        )

    def test_simple_retrieval_uses_only_the_decision_model(self):
        repository = FakeRepository()
        vectors = FakeVectorStore([
            SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "测试证据", 0.9, 1),
        ])
        models = FakeModelGateway(retrieval_needed=True)

        result = self.build_service(
            repository, vectors, models, query_planning=True
        ).answer("kb-1", "conversation-1", "报销制度是什么？")

        self.assertEqual(len(models.completion_calls), 2)
        self.assertEqual(models.embed_calls, [["报销制度是什么？"]])
        self.assertEqual(result["query_plan"]["strategy"], "single")
        self.assertTrue(result["query_plan"]["model_invoked"])
        self.assertEqual(result["retrieval_trace"]["query_count"], 1)
        self.assertNotIn("planning.generation", result["timing"]["by_stage"])

    def test_rewrite_plans_after_decision_and_batches_original_with_standalone_query(self):
        repository = FakeRepository([
            {"role": "user", "content": "井控方案有哪些审批要求？"},
            {"role": "assistant", "content": "已有审批要求回答"},
        ])
        vectors = FakeVectorStore([
            SearchHit("chunk-1", "doc-1", "kb-1", "方案.pdf", "审批要求", 0.9, 1),
        ])
        models = FakeModelGateway(
            retrieval_needed=True,
            needs_rewrite=True,
            planner_output=json.dumps({
                "strategy": "rewrite",
                "standalone_query": "井控方案的审批要求有哪些？",
                "subqueries": [],
            }),
        )

        result = self.build_service(
            repository, vectors, models, query_planning=True
        ).answer("kb-1", "conversation-1", "这个方案的审批要求呢？")

        self.assertEqual(len(models.completion_calls), 3)
        self.assertEqual(
            models.embed_calls,
            [["这个方案的审批要求呢？", "井控方案的审批要求有哪些？"]],
        )
        self.assertEqual(result["query_plan"]["strategy"], "rewrite")
        self.assertTrue(result["query_plan"]["model_invoked"])
        self.assertEqual(
            result["retrieval_trace"]["queries"],
            ["这个方案的审批要求呢？", "井控方案的审批要求有哪些？"],
        )
        self.assertEqual(result["retrieval_trace"]["query_count"], 2)
        self.assertIn("decision.generation", result["timing"]["by_stage"])
        self.assertIn("planning.generation", result["timing"]["by_stage"])

    def test_reports_per_query_rerank_trace_for_decomposed_plan(self):
        repository = FakeRepository()
        vectors = FakeVectorStore([
            SearchHit("chunk-1", "doc-1", "kb-1", "资料.pdf", "证据一", 0.9),
            SearchHit("chunk-2", "doc-2", "kb-1", "资料.pdf", "证据二", 0.8),
        ])
        models = FakeModelGateway(
            retrieval_needed=True,
            complexity="complex",
            planner_output=json.dumps({
                "strategy": "decompose",
                "standalone_query": "综合说明井喷处置和污染物管理要求分别有哪些？",
                "subqueries": ["目标一", "目标二"],
            }),
        )

        class TraceReranker:
            name = "test-reranker"

            def __init__(self):
                self.calls = []

            def rerank(self, query, hits, limit):
                self.calls.append((query, list(hits), limit))
                return list(hits)[:limit]

        reranker = TraceReranker()
        result = self.build_service(
            repository,
            vectors,
            models,
            top_k=2,
            query_planning=True,
            reranker=reranker,
        ).answer("kb-1", "conversation-1", "综合说明井喷处置和污染物管理要求分别有哪些？")

        self.assertEqual([call[0] for call in reranker.calls], ["目标一", "目标二"])
        self.assertEqual(result["retrieval_trace"]["rerank_mode"], "per_query")
        self.assertEqual(result["retrieval_trace"]["rerank_query_count"], 2)
        self.assertEqual(result["retrieval_trace"]["rerank_top_k"], 2)
        self.assertEqual(result["retrieval_k_per_query"], 2)
        self.assertEqual(result["retrieval_total_k"], result["retrieved_count"])
    def test_skips_embedding_and_vector_search_when_retrieval_is_not_needed(self):
        repository = FakeRepository([{"role": "assistant", "content": "已有回答"}])
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=False)

        result = self.build_service(repository, vectors, models).answer(
            "kb-1", "conversation-1", "把上面的回答总结一下"
        )

        self.assertFalse(result["retrieval_used"])
        self.assertEqual(result["retrieved_count"], 0)
        self.assertEqual(result["citations"], [])
        self.assertGreaterEqual(result["response_time_ms"], 0)
        self.assertFalse(result["token_usage"]["available"])
        self.assertEqual(result["token_usage"]["calls"], 0)
        self.assertEqual(models.embed_calls, [])
        self.assertEqual(vectors.search_calls, [])
        self.assertEqual(len(models.completion_calls), 2)
        saved_metrics = repository.saved[0][5]
        self.assertEqual(saved_metrics["responseTimeMs"], result["response_time_ms"])
        self.assertEqual(saved_metrics["serverResponseTimeMs"], result["response_time_ms"])
        self.assertEqual(saved_metrics["tokenUsage"], result["token_usage"])


    def test_short_history_is_token_budgeted_instead_of_fixed_to_twelve_messages(self):
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"<message-{index:02d}>",
            }
            for index in range(14)
        ]
        repository = FakeRepository(history)
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=False)

        self.build_service(repository, vectors, models).answer(
            "kb-1", "conversation-1", "current question"
        )

        self.assertEqual(len(models.completion_calls), 2)
        decision_prompt = models.completion_calls[0][0][1]["content"]
        self.assertIn("<message-00>", decision_prompt)
        self.assertIn("<message-13>", decision_prompt)

        answer_messages = models.completion_calls[1][0]
        answer_history = answer_messages[1:1 + len(history)]
        self.assertEqual(answer_history, history)
        self.assertEqual(repository.saved[0][2], "current question")

    def test_force_retrieval_preserves_query_planning(self):
        repository = FakeRepository()
        vectors = FakeVectorStore([
            SearchHit("chunk-1", "doc-1", "kb-1", "test.pdf", "test evidence", 0.9, 1),
        ])
        question = "\u7efc\u5408\u8bf4\u660e\u6d4b\u8bd5\u95ee\u9898\u4ee5\u53ca\u5176\u4ed6\u76ee\u6807\u7684\u8981\u6c42\u9879"
        target_one = "\u6d4b\u8bd5\u76ee\u6807\u4e00"
        target_two = "\u6d4b\u8bd5\u76ee\u6807\u4e8c"
        models = FakeModelGateway(
            retrieval_needed=False,
            complexity="complex",
            planner_output=json.dumps({
                "strategy": "decompose",
                "standalone_query": question,
                "subqueries": [target_one, target_two],
            }),
        )

        result = self.build_service(
            repository, vectors, models, query_planning=True
        ).answer(
            "kb-1",
            "conversation-1",
            question,
            force_retrieval=True,
        )

        self.assertTrue(result["retrieval_used"])
        self.assertEqual(models.embed_calls, [[question, target_one, target_two]])
        self.assertEqual(len(models.completion_calls), 3)
        self.assertEqual(result["query_plan"]["strategy"], "decompose")
        self.assertEqual(result["retrieval_trace"]["query_count"], 3)
        self.assertEqual(result["agent_trace"][0], {
            "agent": "retrieval_decision",
            "status": "forced",
            "outcome": "retrieve",
        })
    def test_vectorizes_the_question_and_returns_qdrant_ordered_top_k_chunks(self):
        hits = [
            SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "第一段", 0.92, 3),
            SearchHit("chunk-2", "doc-1", "kb-1", "制度.pdf", "第二段", 0.87, 4),
            SearchHit("chunk-3", "doc-2", "kb-1", "流程.pdf", "第三段", 0.81, 1),
            SearchHit("chunk-4", "doc-3", "kb-1", "其他.pdf", "第四段", 0.70, 1),
        ]
        hits[0] = replace(
            hits[0],
            file_name="制度原文.pdf",
            relevance_score=0.97,
            section_path="第一章/报销标准",
            chunk_index=0,
        )
        repository = FakeRepository()
        vectors = FakeVectorStore(hits)
        models = FakeModelGateway(retrieval_needed=True)

        result = self.build_service(repository, vectors, models, top_k=3).answer(
            "kb-1", "conversation-1", "报销制度是什么？", include_retrieved_content=True
        )

        self.assertEqual(models.embed_calls, [["报销制度是什么？"]])
        self.assertEqual(vectors.search_calls, [("kb-1", [0.1, 0.2], 3)])
        self.assertEqual(result["retrieved_count"], 3)
        self.assertEqual(result["retrieved_document_ids"], ["doc-1", "doc-2"])
        self.assertEqual(result["retrieval_k"], 3)
        self.assertEqual(result["retrieval_candidate_k"], 3)
        self.assertIsNone(result["reranker"])
        self.assertEqual(result["retrieved_chunk_ids"], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(result["context_chunk_ids"], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual([item["chunk_id"] for item in result["citations"]], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(result["citations"][0]["score"], 0.92)
        self.assertEqual(result["citations"][0]["relevance_score"], 0.97)
        self.assertEqual(result["citations"][0]["excerpt"], "第一段")
        self.assertEqual(result["citations"][0]["title"], "制度原文.pdf")
        self.assertEqual(result["citations"][0]["section_path"], "第一章/报销标准")
        self.assertEqual(result["citations"][0]["chunk_index"], 0)
        self.assertEqual(result["retrieved_chunks"][0]["section_path"], "第一章/报销标准")
        self.assertEqual(result["retrieved_chunks"][0]["chunk_index"], 0)
        self.assertEqual(result["retrieved_chunks"][0]["title"], "制度原文.pdf")
        self.assertEqual(result["agent_trace"], [
            {"agent": "retrieval_decision", "status": "completed", "outcome": "retrieve"},
            {"agent": "knowledge_retrieval", "status": "completed", "retrieved_count": 3,
             "top_k": 3, "candidate_k": 3, "reranker": None,
             "reranker_provider": None},
            {"agent": "answer", "status": "completed"},
        ])

        answer_payload = json.loads(models.completion_calls[-1][0][-2]["content"].split("\n", 1)[1])
        self.assertIn("第一段", answer_payload["retrieved_context"])
        self.assertIn("第二段", answer_payload["retrieved_context"])
        self.assertIn("第三段", answer_payload["retrieved_context"])
        self.assertNotIn("第四段", answer_payload["retrieved_context"])
        self.assertEqual(result["context_trace"]["evidence"]["selected"], 3)

    def test_file_listing_request_uses_normal_retrieval_flow(self):
        repository = FakeRepository()
        repository.documents = [
            {"status": "ready", "folder_path": "资料", "file_name": "制度.pdf"},
        ]
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=True)

        result = self.build_service(repository, vectors, models).answer(
            "kb-1", "conversation-1", "请列出知识库中的全部文件"
        )

        self.assertTrue(result["retrieval_used"])
        self.assertEqual(models.embed_calls, [["请列出知识库中的全部文件"]])
        self.assertEqual(len(models.completion_calls), 1)
        self.assertFalse(result["catalog_used"])

    def test_explicit_file_lookup_uses_metadata_after_decision_model(self):
        repository = FakeRepository()
        repository.documents = [
            {"status": "ready", "folder_path": "资料", "file_name": "制度.pdf"},
        ]
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=False)
        service = self.build_service(repository, vectors, models)

        result = service.answer(
            "kb-1",
            "conversation-1",
            "知识库里是否存在制度.pdf文件？",
        )

        self.assertFalse(result["retrieval_used"])
        self.assertEqual(vectors.search_calls, [])
        self.assertEqual(len(models.completion_calls), 1)
        self.assertTrue(result["catalog_used"])
        self.assertIn("制度.pdf", result["answer"])
        self.assertEqual(result["agent_trace"][0]["status"], "completed")

    def test_reports_vector_stage_failures(self):
        repository = FakeRepository()
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=True)
        models.embed = lambda _texts: (_ for _ in ()).throw(TimeoutError("timed out"))

        with self.assertLogs("rag_app.application.rag_service", level="ERROR"):
            with self.assertRaisesRegex(RagStageError, "问题向量化并执行相似度检索失败.*TimeoutError: timed out"):
                self.build_service(repository, vectors, models).answer(
                    "kb-1", "conversation-1", "报销制度是什么？"
                )


if __name__ == "__main__":
    unittest.main()
