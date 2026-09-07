import unittest

from agent.retrieval_decision_agent import (
    RetrievalDecisionAgent,
    retrieval_decision_messages,
    should_retrieve,
)
from agent.query_intent import analyze_query_intent


class FakeModels:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        return self.outputs.pop(0) if self.outputs else '{"decision":"RETRIEVE","complexity":"simple","needs_rewrite":false}'


class RetrievalDecisionAgentTest(unittest.TestCase):
    def test_conversation_only_turns_skip_models_with_or_without_precomputed_intent(self):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "知识库中无相关内容。"},
        ]
        for question in ("你好", " 您好！ ", "你好呀", "Hi!", "谢谢", "收到", "再见", "?", "？？"):
            for planning_enabled in (False, True):
                for intent in (None, analyze_query_intent(question)):
                    with self.subTest(question=question, planning=planning_enabled, intent=intent):
                        models = FakeModels('{"decision":"RETRIEVE","complexity":"complex","needs_rewrite":true}')

                        decision = RetrievalDecisionAgent(models, planning_enabled).run(
                            question, history, intent=intent
                        )

                        self.assertFalse(decision.should_retrieve)
                        self.assertIsNone(decision.query_plan)
                        self.assertEqual(models.calls, [])

    def test_greeting_prefix_and_short_factual_questions_still_use_decision_model(self):
        for question in ("你好，报销标准是什么？", "谢谢，继续分析报告", "你好文档的内容是什么？", "？报销上限呢", "井深？", "《你好》"):
            with self.subTest(question=question):
                models = FakeModels('{"decision":"RETRIEVE","complexity":"simple","needs_rewrite":false}')

                decision = RetrievalDecisionAgent(models).run(question, [])

                self.assertFalse(analyze_query_intent(question).conversation_only)
                self.assertTrue(decision.should_retrieve)
                self.assertEqual(len(models.calls), 1)

    def test_forced_retrieval_overrides_conversation_shortcut(self):
        models = FakeModels('{"decision":"SKIP","complexity":"simple","needs_rewrite":false}')

        decision = RetrievalDecisionAgent(models).run("你好", [], force_retrieval=True)

        self.assertTrue(decision.should_retrieve)
        self.assertEqual(decision.query_plan.retrieval_queries("你好"), ["你好"])
        self.assertEqual(len(models.calls), 1)

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

        self.assertFalse(should_retrieve('{"decision":"SKIP","strategy":"rewrite"'))
        self.assertTrue(should_retrieve('{"decision":"RETRIEVE","strategy":"rewrite"'))

    def test_prompt_contains_history_and_current_question(self):
        messages = retrieval_decision_messages(
            "总结一下",
            [{"role": "assistant", "content": "这是已有回答"}, {"role": "tool", "content": "忽略"}],
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("assistant: 这是已有回答", messages[1]["content"])
        self.assertNotIn("tool: 忽略", messages[1]["content"])
        self.assertIn("总结一下", messages[1]["content"])

    def test_conversation_keyword_skips_the_decision_model(self):
        models = FakeModels('{"decision":"SKIP","complexity":"simple","needs_rewrite":false}')

        decision = RetrievalDecisionAgent(models).run("谢谢", [])

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(decision.outcome, "skip")
        self.assertEqual(models.calls, [])

    def test_agent_requests_only_retrieval_decision(self):
        models = FakeModels('{"decision":"SKIP"}')

        decision = RetrievalDecisionAgent(models, query_planning_enabled=False).run(
            "Can you rewrite the previous answer?",
            [],
        )

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(decision.outcome, "skip")
        self.assertEqual(models.calls[0][2:5], (0, 16, False))
        self.assertEqual(models.calls[0][5]["required"], ["decision"])

    def test_complexity_decision_and_rewrite_are_two_separate_model_calls(self):
        models = FakeModels(
            '{"decision":"RETRIEVE","complexity":"simple","needs_rewrite":true}',
            '{"strategy":"rewrite","standalone_query":"化348-4井的含油率是多少？","subqueries":[]}',
        )

        result = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run(
            "那它的含油率呢？",
            [{"role": "user", "content": "介绍化348-4井"}],
        )

        self.assertTrue(result.should_retrieve)
        self.assertEqual(result.query_plan.strategy, "rewrite")
        self.assertEqual(result.query_plan.standalone_query, "化348-4井的含油率是多少？")
        self.assertEqual(models.calls[0][3], 96)
        self.assertEqual(
            models.calls[0][5]["required"],
            ["decision", "complexity", "needs_rewrite"],
        )
        self.assertEqual(models.calls[1][5]["required"], ["strategy", "standalone_query", "subqueries"])

    def test_simple_question_uses_assessment_schema_without_planner_call(self):
        models = FakeModels('{"decision":"RETRIEVE","complexity":"simple","needs_rewrite":false}')

        result = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run("报销制度是什么？", [])

        self.assertTrue(result.should_retrieve)
        self.assertEqual(result.query_plan.strategy, "single")
        self.assertEqual(models.calls[0][3], 96)
        self.assertEqual(models.calls[0][5]["required"], ["decision", "complexity", "needs_rewrite"])
        self.assertEqual(len(models.calls), 1)

    def test_domain_word_does_not_force_complexity(self):
        question = "根据26年度预算方案，分析研发费用变化及预算增幅。"
        models = FakeModels(
            '{"decision":"RETRIEVE","complexity":"simple","needs_rewrite":false}'
        )

        result = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run(question, [])

        self.assertTrue(result.should_retrieve)
        self.assertEqual(result.query_plan.strategy, "single")
        self.assertEqual(len(models.calls), 1)

    def test_truncated_combined_plan_recovers_decision_and_original_query(self):
        models = FakeModels(
            '{\n'
            '  "decision": "RETRIEVE",\n'
            '  "complexity": "complex",\n'
            '  "needs_rewrite": false\n'
            '}',
            '{\n'
            '  "strategy": "decompose",\n'
            '  "standalone_query": "'
        )

        result = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run(
            "请分别说明合同付款节点和验收到账情况。",
            [],
        )

        self.assertTrue(result.should_retrieve)
        self.assertEqual(result.query_plan.strategy, "single")
        self.assertEqual(
            result.query_plan.retrieval_queries("请分别说明合同付款节点和验收到账情况。"),
            ["请分别说明合同付款节点和验收到账情况。"],
        )

    def test_truncated_combined_plan_keeps_completed_subqueries(self):
        models = FakeModels(
            '{"decision":"RETRIEVE","complexity":"complex","needs_rewrite":false}',
            '{"strategy":"decompose",'
            '"standalone_query":"原问题",'
            '"subqueries":["合同付款节点","验收到账情况"'
        )

        result = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run("请比较合同和验收记录中的付款节点与到账情况。", [])

        self.assertTrue(result.should_retrieve)
        self.assertEqual(result.query_plan.strategy, "decompose")
        self.assertEqual(result.query_plan.subqueries, ("合同付款节点", "验收到账情况"))

    def test_model_identity_question_is_judged_by_the_decision_model(self):
        models = FakeModels('{"decision":"SKIP","complexity":"simple","needs_rewrite":false}')

        decision = RetrievalDecisionAgent(models).run("你是什么大模型啊", [])

        self.assertFalse(decision.should_retrieve)
        self.assertEqual(len(models.calls), 1)

    def test_requested_materials_checklist_retrieves_document_content(self):
        question = "黑山梁化学驱方案所需资料清单帮我列出来"
        models = FakeModels('{"decision":"RETRIEVE","complexity":"simple","needs_rewrite":false}')

        decision = RetrievalDecisionAgent(models, query_planning_enabled=True).run(question, [])

        self.assertFalse(analyze_query_intent(question).skips_retrieval)
        self.assertTrue(decision.should_retrieve)
        self.assertEqual(len(models.calls), 1)

    def test_full_file_listing_no_longer_bypasses_decision_model(self):
        question = "请列出知识库中的全部文件"
        models = FakeModels('{"decision":"RETRIEVE","complexity":"simple","needs_rewrite":false}')

        decision = RetrievalDecisionAgent(models).run(question, [])

        self.assertFalse(analyze_query_intent(question).skips_retrieval)
        self.assertTrue(decision.should_retrieve)
        self.assertEqual(len(models.calls), 1)

    def test_file_content_listing_is_not_misclassified_as_catalog_inventory(self):
        question = (
            "化167-3井的基线测井文件与同位素测井文件的测量深度范围是否一致？"
            "请分别列出两个文件的起始深度和结束深度，并说明同位素线文件与它们的深度范围有何不同。"
        )
        models = FakeModels('{"decision":"RETRIEVE","complexity":"complex","needs_rewrite":false}', '{"strategy":"decompose","standalone_query":"","subqueries":["合同付款节点","验收到账情况"]}')

        decision = RetrievalDecisionAgent(models, query_planning_enabled=True).run(question, [])

        self.assertFalse(analyze_query_intent(question).skips_retrieval)
        self.assertTrue(decision.should_retrieve)
        self.assertEqual(len(models.calls), 2)
    def test_malformed_complex_plan_falls_back_to_original_question(self):
        question = (
            "根据《设计报告》和《实际报告》，分别统计设计数量和实际数量。"
        )
        models = FakeModels(
            '{"decision":"RETRIEVE","complexity":"complex","needs_rewrite":false}',
            '{"strategy":"decompose","standalone_query":"'
        )

        decision = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run(question, [])

        self.assertTrue(decision.should_retrieve)
        self.assertTrue(decision.query_plan.fallback)
        self.assertEqual(decision.query_plan.strategy, "single")
        self.assertEqual(decision.query_plan.subqueries, ())
        self.assertEqual(decision.query_plan.retrieval_queries(question), [question])

    def test_model_skip_is_respected_even_for_structurally_complex_question(self):
        question = "请分别说明井喷处置和污染物管理有哪些要求？"
        models = FakeModels('{"decision":"SKIP","complexity":"complex","needs_rewrite":false}')

        decision = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run(question, [])

        self.assertFalse(decision.should_retrieve)
        self.assertIsNone(decision.query_plan)

    def test_complexity_is_model_owned_even_without_structural_keywords(self):
        question = "根据两份年度材料，核对研发支出并给出差异。"
        models = FakeModels(
            '{"decision":"RETRIEVE","complexity":"complex","needs_rewrite":false}',
            '{"strategy":"decompose","standalone_query":"","subqueries":["第一份年度材料中的研发支出是多少？","第二份年度材料中的研发支出是多少？"]}',
        )

        decision = RetrievalDecisionAgent(
            models,
            query_planning_enabled=True,
        ).run(question, [])

        self.assertTrue(decision.should_retrieve)
        self.assertEqual(decision.query_plan.strategy, "decompose")
        self.assertEqual(len(models.calls), 2)

    def test_decision_model_failure_defaults_to_retrieval(self):
        class FailingModels:
            def complete(self, *args, **kwargs):
                raise TimeoutError("timed out")

        models = FailingModels()

        decision = RetrievalDecisionAgent(models).run("报销制度是什么？", [])

        self.assertTrue(decision.should_retrieve)
        self.assertTrue(decision.query_plan.fallback)


if __name__ == "__main__":
    unittest.main()
