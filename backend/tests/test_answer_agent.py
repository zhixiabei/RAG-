import unittest

from agent.answer_agent import AnswerAgent
from rag_app.domain.models import SearchHit


class FakeModels:
    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        return "最终回答"


class AnswerAgentTest(unittest.TestCase):
    def test_returns_no_relevant_content_after_unsuccessful_retrieval(self):
        models = FakeModels()

        answer = AnswerAgent(models).run("未知问题", [], [], retrieval_used=True)

        self.assertEqual(answer, "知识库中无相关内容。")
        self.assertEqual(models.calls, [])

    def test_unsuccessful_retrieval_does_not_fall_back_to_unrelated_history(self):
        models = FakeModels()

        answer = AnswerAgent(models).run(
            "新的问题",
            [{"role": "assistant", "content": "旧问题的回答"}],
            [],
            retrieval_used=True,
        )

        self.assertEqual(answer, "知识库中无相关内容。")
        self.assertEqual(models.calls, [])

    def test_builds_context_from_hits(self):
        models = FakeModels()
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.9, 3)

        answer = AnswerAgent(models).run("上限是多少？", [], [hit], retrieval_used=True)

        self.assertEqual(answer, "最终回答")
        messages = models.calls[0][0]
        self.assertEqual(messages[-1], {"role": "user", "content": "上限是多少？"})
        self.assertIn("行内数学公式必须用 $...$", messages[0]["content"])
        self.assertIn("制度.pdf", messages[-2]["content"])
        self.assertIn("报销上限为 500 元", messages[-2]["content"])

    def test_returns_selected_model_for_model_identity_question_without_completion(self):
        models = FakeModels()

        answer = AnswerAgent(models).run("你是什么大模型啊", [], [], retrieval_used=False, model="qwen3:4b")

        self.assertEqual(answer, "我是知识库助手，当前回答使用的模型是 qwen3:4b。")
        self.assertEqual(models.calls, [])


if __name__ == "__main__":
    unittest.main()
