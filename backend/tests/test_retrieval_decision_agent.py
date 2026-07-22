import unittest

from agent.retrieval_decision_agent import (
    RetrievalDecisionAgent,
    retrieval_decision_messages,
    should_retrieve,
)


class FakeModels:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        return self.output


class RetrievalDecisionAgentTest(unittest.TestCase):
    def test_only_exact_skip_bypasses_retrieval(self):
        self.assertFalse(should_retrieve(" SKIP \n"))
        self.assertFalse(should_retrieve('{"decision":"SKIP"}'))
        self.assertFalse(should_retrieve('```json\n{"decision":"SKIP"}\n```'))
        self.assertFalse(should_retrieve('{"decision":"SKIP"'))
        self.assertTrue(should_retrieve('{"decision":"RETRIEVE"}'))
        self.assertTrue(should_retrieve('先分析，最终应该 SKIP'))
        self.assertTrue(should_retrieve(""))
        self.assertTrue(should_retrieve("SKIP because history is enough"))
        self.assertTrue(should_retrieve("RETRIEVE"))

    def test_prompt_contains_history_and_current_question(self):
        messages = retrieval_decision_messages(
            "总结一下",
            [{"role": "assistant", "content": "这是已有回答"}, {"role": "tool", "content": "忽略"}],
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("assistant: 这是已有回答", messages[1]["content"])
        self.assertNotIn("tool: 忽略", messages[1]["content"])
        self.assertIn("总结一下", messages[1]["content"])

    def test_agent_uses_bounded_deterministic_completion(self):
        models = FakeModels('{"decision":"SKIP"}')

        decision = RetrievalDecisionAgent(models).run("谢谢", [])

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(decision.outcome, "skip")
        self.assertEqual(models.calls[0][2:5], (0, 32, False))
        self.assertEqual(models.calls[0][5]["required"], ["decision"])

    def test_model_identity_question_skips_retrieval_without_completion(self):
        models = FakeModels('{"decision":"RETRIEVE"}')

        decision = RetrievalDecisionAgent(models).run("你是什么大模型啊", [])

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(models.calls, [])


if __name__ == "__main__":
    unittest.main()
