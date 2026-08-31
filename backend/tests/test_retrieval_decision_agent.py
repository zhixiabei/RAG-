import unittest

from agent.retrieval_decision_agent import (
    RetrievalDecisionAgent,
    retrieval_decision_messages,
    should_retrieve,
)
from agent.query_intent import analyze_query_intent


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
        self.assertFalse(should_retrieve("```json\n{\"decision\":\"SKIP\"}\n```"))
        self.assertFalse(should_retrieve('{"decision":"SKIP"'))
        self.assertTrue(should_retrieve('{"decision":"RETRIEVE"}'))
        self.assertTrue(should_retrieve("先分析，最终应该 SKIP"))
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

    def test_conversation_keyword_skips_without_completion(self):
        models = FakeModels('{"decision":"RETRIEVE"}')

        decision = RetrievalDecisionAgent(models).run("谢谢", [])

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(decision.outcome, "skip")
        self.assertEqual(models.calls, [])

    def test_agent_requests_only_retrieval_decision(self):
        models = FakeModels('{"decision":"SKIP"}')

        decision = RetrievalDecisionAgent(models).run(
            "Can you rewrite the previous answer?",
            [],
        )

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(decision.outcome, "skip")
        self.assertEqual(models.calls[0][2:5], (0, 16, False))
        self.assertEqual(models.calls[0][5]["required"], ["decision"])

    def test_model_identity_question_skips_retrieval_without_completion(self):
        models = FakeModels('{"decision":"RETRIEVE"}')

        decision = RetrievalDecisionAgent(models).run("你是什么大模型啊", [])

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(models.calls, [])

    def test_requested_materials_checklist_retrieves_document_content(self):
        question = "黑山梁化学驱方案所需资料清单帮我列出来"
        models = FakeModels('{"decision":"RETRIEVE"}')

        decision = RetrievalDecisionAgent(models).run(question, [])

        self.assertFalse(analyze_query_intent(question).skips_retrieval)
        self.assertTrue(decision.should_retrieve)
        self.assertEqual(len(models.calls), 1)

    def test_full_file_listing_no_longer_bypasses_decision_model(self):
        question = "请列出知识库中的全部文件"
        models = FakeModels('{"decision":"RETRIEVE"}')

        decision = RetrievalDecisionAgent(models).run(question, [])

        self.assertFalse(analyze_query_intent(question).skips_retrieval)
        self.assertTrue(decision.should_retrieve)
        self.assertEqual(len(models.calls), 1)

    def test_file_content_listing_is_not_misclassified_as_catalog_inventory(self):
        question = (
            "化167-3井的基线测井文件与同位素测井文件的测量深度范围是否一致？"
            "请分别列出两个文件的起始深度和结束深度，并说明同位素线文件与它们的深度范围有何不同。"
        )
        models = FakeModels('{"decision":"RETRIEVE"}')

        decision = RetrievalDecisionAgent(models).run(question, [])

        self.assertFalse(analyze_query_intent(question).skips_retrieval)
        self.assertTrue(decision.should_retrieve)
        self.assertEqual(len(models.calls), 1)


if __name__ == "__main__":
    unittest.main()
