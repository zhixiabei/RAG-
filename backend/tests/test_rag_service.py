import unittest

from agent import AnswerAgent, KnowledgeRetrievalAgent, RelevanceGradingAgent, RetrievalDecisionAgent
from rag_app.application.rag_service import RagService
from rag_app.domain.models import SearchHit


class FakeRepository:
    def __init__(self, history=None):
        self.history = history or []
        self.saved = []

    def get_knowledge_base(self, knowledge_base_id):
        return {"id": knowledge_base_id, "embedding_model": "test-embedding"}

    def list_messages(self, conversation_id):
        return self.history

    def add_message(self, conversation_id, knowledge_base_id, question, answer, citations):
        self.saved.append((conversation_id, knowledge_base_id, question, answer, citations))


class FakeVectorStore:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.search_calls = []

    def search(self, knowledge_base_id, vector, limit):
        self.search_calls.append((knowledge_base_id, vector, limit))
        return self.hits


class FakeModelGateway:
    chat_model = "test-chat"
    embedding_model = "test-embedding"

    def __init__(self, retrieval_needed):
        self.retrieval_needed = retrieval_needed
        self.relevance_scores = {}
        self.embed_calls = []
        self.completion_calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.completion_calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        if response_schema and response_schema.get("required") == ["decision", "search_query"]:
            decision = "RETRIEVE" if self.retrieval_needed else "SKIP"
            query = "时变电磁场 核心概念 基本规律" if self.retrieval_needed else ""
            return f'{{"decision":"{decision}","search_query":"{query}"}}'
        if response_schema and response_schema.get("required") == ["items"]:
            import json

            candidates = json.loads(messages[-1]["content"])["candidates"]
            items = [
                {
                    "chunk_id": item["chunk_id"],
                    "score": self.relevance_scores.get(f"chunk-{index}", 0.9),
                }
                for index, item in enumerate(candidates, 1)
            ]
            return json.dumps({"items": items})
        return "测试回答"

    def embed(self, texts):
        self.embed_calls.append(texts)
        return [[0.1, 0.2]]

class RagServiceTest(unittest.TestCase):
    @staticmethod
    def build_service(repository, vectors, models):
        return RagService(
            repository,
            RetrievalDecisionAgent(models),
            KnowledgeRetrievalAgent(vectors, models, retrieval_top_k=20, context_top_k=8),
            RelevanceGradingAgent(models, threshold=0.65),
            AnswerAgent(models),
        )

    def test_skips_embedding_and_vector_search_when_retrieval_is_not_needed(self):
        repository = FakeRepository([{"role": "assistant", "content": "已有回答"}])
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=False)
        service = self.build_service(repository, vectors, models)

        result = service.answer("kb-1", "conversation-1", "把上面的回答总结一下")

        self.assertFalse(result["retrieval_used"])
        self.assertEqual(result["retrieved_count"], 0)
        self.assertEqual(result["citations"], [])
        self.assertEqual(models.embed_calls, [])
        self.assertEqual(vectors.search_calls, [])
        self.assertEqual(len(models.completion_calls), 2)
        self.assertEqual(result["agent_trace"][0]["outcome"], "skip")
        self.assertEqual(result["agent_trace"][1]["status"], "skipped")

    def test_searches_and_returns_citations_when_retrieval_is_needed(self):
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "知识库内容", 0.92, 3)
        repository = FakeRepository()
        vectors = FakeVectorStore([hit])
        models = FakeModelGateway(retrieval_needed=True)
        service = self.build_service(repository, vectors, models)

        result = service.answer("kb-1", "conversation-1", "报销制度是什么？")

        self.assertTrue(result["retrieval_used"])
        self.assertEqual(result["retrieved_count"], 1)
        self.assertEqual(result["citations"][0]["chunk_id"], "chunk-1")
        self.assertEqual(models.embed_calls, [["时变电磁场 核心概念 基本规律"]])
        self.assertEqual(len(vectors.search_calls), 1)
        self.assertEqual(result["agent_trace"][0]["outcome"], "retrieve")
        self.assertEqual(result["agent_trace"][0]["search_query"], "时变电磁场 核心概念 基本规律")
        self.assertEqual(result["agent_trace"][1]["retrieved_count"], 1)
        self.assertEqual(result["agent_trace"][2]["relevant_count"], 1)
        self.assertEqual(result["citations"][0]["relevance_score"], 0.9)
        self.assertEqual(len(models.completion_calls), 3)

    def test_returns_no_related_content_when_all_candidates_score_below_threshold(self):
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "无关内容", 0.92, 3)
        repository = FakeRepository([{"role": "assistant", "content": "旧回答"}])
        vectors = FakeVectorStore([hit])
        models = FakeModelGateway(retrieval_needed=True)
        models.relevance_scores = {"chunk-1": 0.2}
        service = self.build_service(repository, vectors, models)

        result = service.answer("kb-1", "conversation-1", "完全不同的问题")

        self.assertEqual(result["answer"], "知识库中无相关内容。")
        self.assertEqual(result["retrieved_count"], 1)
        self.assertEqual(result["relevant_count"], 0)
        self.assertEqual(result["citations"], [])
        self.assertEqual(len(models.completion_calls), 2)

    def test_model_identity_question_does_not_retrieve_and_reports_selected_model(self):
        repository = FakeRepository([{"role": "assistant", "content": "旧的文档回答"}])
        vectors = FakeVectorStore()
        models = FakeModelGateway(retrieval_needed=True)
        service = self.build_service(repository, vectors, models)

        result = service.answer("kb-1", "conversation-1", "你是什么大模型啊", "qwen3:4b")

        self.assertEqual(result["answer"], "我是知识库助手，当前回答使用的模型是 qwen3:4b。")
        self.assertFalse(result["retrieval_used"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(models.completion_calls, [])
        self.assertEqual(vectors.search_calls, [])


if __name__ == "__main__":
    unittest.main()
