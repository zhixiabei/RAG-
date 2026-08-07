import json
import unittest

from agent import AnswerAgent, KnowledgeRetrievalAgent, RetrievalDecisionAgent
from rag_app.application.rag_service import CONTEXT_HISTORY_MESSAGE_LIMIT, RagService, RagStageError
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

    def add_message(self, conversation_id, knowledge_base_id, question, answer, citations):
        self.saved.append((conversation_id, knowledge_base_id, question, answer, citations))


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

    def __init__(self, retrieval_needed):
        self.retrieval_needed = retrieval_needed
        self.embed_calls = []
        self.completion_calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.completion_calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        if response_schema and response_schema.get("required") == ["decision"]:
            decision = "RETRIEVE" if self.retrieval_needed else "SKIP"
            return json.dumps({"decision": decision})
        return "测试回答"

    def embed(self, texts):
        self.embed_calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class RagServiceTest(unittest.TestCase):
    @staticmethod
    def build_service(repository, vectors, models, top_k=3):
        return RagService(
            repository,
            RetrievalDecisionAgent(models),
            KnowledgeRetrievalAgent(vectors, models, top_k=top_k),
            AnswerAgent(models),
        )

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
        self.assertEqual(models.embed_calls, [])
        self.assertEqual(vectors.search_calls, [])
        self.assertEqual(len(models.completion_calls), 2)

    def test_only_latest_twelve_messages_are_added_to_model_context(self):
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"<message-{index:02d}>",
            }
            for index in range(CONTEXT_HISTORY_MESSAGE_LIMIT + 2)
        ]
        repository = FakeRepository(history)
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=False)

        self.build_service(repository, vectors, models).answer(
            "kb-1", "conversation-1", "current question"
        )

        self.assertEqual(len(models.completion_calls), 2)
        decision_prompt = models.completion_calls[0][0][1]["content"]
        self.assertNotIn("<message-00>", decision_prompt)
        self.assertNotIn("<message-01>", decision_prompt)

        answer_messages = models.completion_calls[1][0]
        answer_history = answer_messages[1:1 + CONTEXT_HISTORY_MESSAGE_LIMIT]
        self.assertEqual(answer_history, history[-CONTEXT_HISTORY_MESSAGE_LIMIT:])
        self.assertNotIn("<message-00>", str(answer_messages))
        self.assertNotIn("<message-01>", str(answer_messages))

    def test_vectorizes_the_question_and_returns_qdrant_ordered_top_k_chunks(self):
        hits = [
            SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "第一段", 0.92, 3),
            SearchHit("chunk-2", "doc-1", "kb-1", "制度.pdf", "第二段", 0.87, 4),
            SearchHit("chunk-3", "doc-2", "kb-1", "流程.pdf", "第三段", 0.81, 1),
            SearchHit("chunk-4", "doc-3", "kb-1", "其他.pdf", "第四段", 0.70, 1),
        ]
        repository = FakeRepository()
        vectors = FakeVectorStore(hits)
        models = FakeModelGateway(retrieval_needed=True)

        result = self.build_service(repository, vectors, models, top_k=3).answer(
            "kb-1", "conversation-1", "报销制度是什么？"
        )

        self.assertEqual(models.embed_calls, [["报销制度是什么？"]])
        self.assertEqual(vectors.search_calls, [("kb-1", [0.1, 0.2], 3)])
        self.assertEqual(result["retrieved_count"], 3)
        self.assertEqual(result["retrieved_document_ids"], ["doc-1", "doc-2"])
        self.assertEqual(result["retrieved_chunk_ids"], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual([item["chunk_id"] for item in result["citations"]], ["chunk-1", "chunk-3"])
        self.assertEqual(result["citations"][0]["score"], 0.92)
        self.assertIsNone(result["citations"][0]["relevance_score"])
        self.assertEqual(result["agent_trace"], [
            {"agent": "retrieval_decision", "status": "completed", "outcome": "retrieve"},
            {"agent": "knowledge_retrieval", "status": "completed", "retrieved_count": 3, "top_k": 3},
            {"agent": "answer", "status": "completed"},
        ])

        answer_payload = json.loads(models.completion_calls[-1][0][-2]["content"].split("\n", 1)[1])
        self.assertIn("第一段", answer_payload["retrieved_context"])
        self.assertIn("第二段", answer_payload["retrieved_context"])
        self.assertIn("第三段", answer_payload["retrieved_context"])
        self.assertNotIn("第四段", answer_payload["retrieved_context"])

    def test_catalog_inventory_uses_metadata_without_vector_search(self):
        repository = FakeRepository()
        repository.documents = [{"status": "ready", "folder_path": "资料", "file_name": "制度.pdf"}]
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=True)

        result = self.build_service(repository, vectors, models).answer(
            "kb-1", "conversation-1", "知识库里有哪些文件？"
        )

        self.assertFalse(result["retrieval_used"])
        self.assertEqual(vectors.search_calls, [])
        self.assertEqual(models.completion_calls, [])
        self.assertTrue(result["catalog_used"])

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
