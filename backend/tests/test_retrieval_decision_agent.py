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

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning))
        return self.output


class RetrievalDecisionAgentTest(unittest.TestCase):
    def test_only_exact_skip_bypasses_retrieval(self):
        self.assertFalse(should_retrieve(" SKIP \n"))
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
        models = FakeModels("SKIP")

        decision = RetrievalDecisionAgent(models).run("谢谢", [])

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(decision.outcome, "skip")
        self.assertEqual(models.calls[0][2:], (0, 8, False))


if __name__ == "__main__":
    unittest.main()
