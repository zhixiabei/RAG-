import json
import unittest

from agent.query_planning_agent import (
    QueryPlanningAgent,
    query_planning_messages,
    query_planning_trigger,
)


class FakeModels:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return self.output


class QueryPlanningAgentTest(unittest.TestCase):
    def test_simple_lookup_uses_a_single_planning_result(self):
        question = "化348-4井的年累计增油是多少？"
        models = FakeModels(json.dumps({
            "strategy": "single",
            "standalone_query": question,
            "subqueries": [],
        }, ensure_ascii=False))
        plan = QueryPlanningAgent(models).run(question, [])
        self.assertEqual(plan.strategy, "single")
        self.assertTrue(plan.model_invoked)
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(plan.retrieval_queries(question), [question])

    def test_rewrites_a_context_dependent_follow_up(self):
        models = FakeModels(json.dumps({
            "strategy": "rewrite",
            "standalone_query": "化348-4井的含油率是多少？",
            "subqueries": [],
        }, ensure_ascii=False))
        history = [
            {"role": "user", "content": "介绍化348-4井"},
            {"role": "assistant", "content": "这是该井的基本情况。"},
        ]
        plan = QueryPlanningAgent(models).run("那它的含油率呢？", history)
        self.assertEqual(plan.strategy, "rewrite")
        self.assertEqual(plan.retrieval_queries("那它的含油率呢？"), [
            "那它的含油率呢？", "化348-4井的含油率是多少？",
        ])
        self.assertEqual(plan.trigger, "context_reference")
        self.assertEqual(len(models.calls), 1)

    def test_decomposes_a_multi_evidence_question(self):
        models = FakeModels(json.dumps({
            "strategy": "decompose",
            "standalone_query": "综合井喷处置和污染物管理要求，应如何建立HSE控制链？",
            "subqueries": ["井喷失控后的应急处置要求是什么？", "钻井污染物应如何管理？"],
        }, ensure_ascii=False))
        plan = QueryPlanningAgent(models).run(
            "综合井喷处置和污染物管理要求，应如何建立HSE控制链？", []
        )
        self.assertEqual(plan.strategy, "decompose")
        self.assertEqual(len(plan.subqueries), 2)
        self.assertEqual(plan.trigger, "complex_query")

    def test_triggers_planning_for_two_explicit_source_documents(self):
        question = (
            "根据《化166-2井组化学示踪剂监测设计》和《化166-2示踪剂报告》，"
            "统计设计阶段井数与实际见剂井数并判断整体连通性。"
        )

        self.assertEqual(query_planning_trigger(question, []), "complex_query")

    def test_model_single_is_not_overridden_by_explicit_sources(self):
        question = (
            "根据《化166-2井组化学示踪剂监测设计》和《化166-2示踪剂报告》，"
            "统计设计阶段井数与实际见剂井数并判断整体连通性。"
        )
        models = FakeModels(json.dumps({
            "strategy": "single",
            "standalone_query": question,
            "subqueries": [],
        }, ensure_ascii=False))

        plan = QueryPlanningAgent(models).run(question, [])

        self.assertEqual(plan.strategy, "single")
        self.assertEqual(plan.subqueries, ())
        self.assertEqual(plan.retrieval_queries(question), [question])

    def test_single_budget_source_does_not_select_a_domain_topic(self):
        question = "根据《26年度预算方案》，分析研发费用变化及预算增幅"
        models = FakeModels(json.dumps({
            "strategy": "single",
            "standalone_query": question,
            "subqueries": [],
        }, ensure_ascii=False))

        plan = QueryPlanningAgent(models).run(question, [])

        self.assertEqual(plan.strategy, "single")
        self.assertEqual(plan.subqueries, ())
        self.assertEqual(plan.retrieval_queries(question), [question])

    def test_model_owned_source_decomposition_preserves_model_queries(self):
        question = "根据《合同甲》和《验收记录乙》，比较付款节点与到账情况"
        models = FakeModels(json.dumps({
            "strategy": "decompose",
            "standalone_query": question,
            "subqueries": [
                "《合同甲》中的付款节点",
                "《验收记录乙》中的到账情况",
            ],
        }, ensure_ascii=False))

        plan = QueryPlanningAgent(models).run(question, [])

        self.assertEqual(plan.strategy, "decompose")
        self.assertEqual(plan.subqueries, (
            "《合同甲》中的付款节点",
            "《验收记录乙》中的到账情况",
        ))

    def test_ignores_hallucinated_standalone_query_for_explicit_source_comparison(self):
        question = (
            "根据《2024年化子坪测试解释结果汇总》和《2024年杏子川采油厂油水井测试解释成果汇总表》，"
            "比较化309-5井与杏6329-1井的绝对吸水量：哪口井更高，高出多少方/天，"
            "约为另一口井的多少倍？结果保留两位小数。"
        )
        models = FakeModels(json.dumps({
            "strategy": "decompose",
            "standalone_query": "化309-5井为10000方/天，杏6329-1井为5000方/天，结果为2.00倍。",
            "subqueries": [
                "检索化309-5井绝对吸水量",
                "检索杏6329-1井绝对吸水量",
            ],
        }, ensure_ascii=False))

        plan = QueryPlanningAgent(models).run(question, [])

        self.assertEqual(plan.strategy, "decompose")
        self.assertEqual(plan.standalone_query, question)
        self.assertEqual(len(plan.subqueries), 2)
        self.assertEqual(plan.subqueries, (
            "检索化309-5井绝对吸水量",
            "检索杏6329-1井绝对吸水量",
        ))

    def test_planner_failure_falls_back_to_original_question(self):
        models = FakeModels(error=TimeoutError("timed out"))
        question = "分别说明井喷处置和污染物管理有哪些要求？"
        with self.assertLogs("agent.query_planning_agent", level="WARNING") as logs:
            plan = QueryPlanningAgent(models).run(question, [])
        self.assertEqual(plan.strategy, "single")
        self.assertTrue(plan.fallback)
        self.assertEqual(plan.retrieval_queries(question), [question])
        self.assertNotIn("Traceback", logs.output[0])

    def test_truncated_json_falls_back_without_using_partial_query(self):
        question = "请分别说明井喷处置和污染物管理要求？"
        models = FakeModels(
            '{"strategy":"rewrite","standalone_query":"井喷处置和污染物管理要求'
        )

        plan = QueryPlanningAgent(models).run(question, [])

        self.assertEqual(plan.strategy, "single")
        self.assertTrue(plan.fallback)
        self.assertEqual(plan.standalone_query, question)

    def test_rejects_prompt_text_copied_into_generated_query(self):
        question = "请对比两个井的测试设计并分别说明验收要求？"
        models = FakeModels(
            '{"strategy":"rewrite","standalone_query":"trigger reason: complex_query; current question: x","subqueries":[]}'
        )

        plan = QueryPlanningAgent(models).run(question, [])

        self.assertEqual(plan.strategy, "single")
        self.assertTrue(plan.fallback)

    def test_accepts_json_code_fence_and_leading_prose(self):
        models = FakeModels(
            '结果如下：\n```json\n{"strategy":"rewrite","standalone_query":"化348-4井的含油率是多少？","subqueries":[]}\n```'
        )

        plan = QueryPlanningAgent(models).run("那它的含油率呢？", [
            {"role": "user", "content": "介绍化348-4井"},
        ])

        self.assertEqual(plan.strategy, "rewrite")
        self.assertEqual(plan.standalone_query, "化348-4井的含油率是多少？")

    def test_planner_history_is_bounded_to_recent_short_messages(self):
        history = [
            {"role": "user", "content": "旧问题 " * 500},
            {"role": "assistant", "content": "旧回答 " * 500},
            {"role": "user", "content": "最近问题 " * 500},
            {"role": "assistant", "content": "最近回答 " * 500},
        ]

        messages = query_planning_messages("这个问题呢？", history, "context_reference")
        prompt = messages[1]["content"]

        self.assertIn("最近问题", prompt)
        self.assertIn("最近回答", prompt)
        self.assertLessEqual(len(prompt), 2_000)

    def test_short_follow_up_requires_history(self):
        self.assertIsNone(query_planning_trigger("含油率呢？", []))
        self.assertEqual(query_planning_trigger(
            "含油率呢？", [{"role": "assistant", "content": "上一轮内容"}]
        ), "context_reference")


    def test_subqueries_override_an_incorrect_strategy_without_posthoc_rewriting(self):
        question = (
            "根据《设计报告》和《实际报告》，分别统计设计数量和实际数量。"
        )
        models = FakeModels(json.dumps({
            "strategy": "rewrite",
            "standalone_query": "分别统计设计数量和实际数量",
            "subqueries": ["统计设计数量", "统计实际数量"],
        }, ensure_ascii=False))

        plan = QueryPlanningAgent(models).run(question, [])

        self.assertEqual(plan.strategy, "decompose")
        self.assertEqual(plan.standalone_query, question)
        self.assertEqual(plan.subqueries, ("统计设计数量", "统计实际数量"))

if __name__ == "__main__":
    unittest.main()
