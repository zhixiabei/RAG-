import unittest

from agent.answer_agent import AnswerAgent
from rag_app.domain.models import SearchHit


class FakeModels:
    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning))
        return "最终回答"


class AnswerAgentTest(unittest.TestCase):
    def test_returns_safe_fallback_without_hits_or_history(self):
        models = FakeModels()

        answer = AnswerAgent(models).run("未知问题", [], [], retrieval_used=True)

        self.assertEqual(answer, "知识库中没有足够信息回答这个问题。")
        self.assertEqual(models.calls, [])

    def test_builds_context_from_hits(self):
        models = FakeModels()
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.9, 3)

        answer = AnswerAgent(models).run("上限是多少？", [], [hit], retrieval_used=True)

        self.assertEqual(answer, "最终回答")
        user_message = models.calls[0][0][-1]["content"]
        self.assertIn("制度.pdf", user_message)
        self.assertIn("报销上限为 500 元", user_message)


if __name__ == "__main__":
    unittest.main()
