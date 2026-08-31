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
    def test_simple_lookup_skips_planning_model(self):
        models = FakeModels()
        plan = QueryPlanningAgent(models).run("化348-4井的年累计增油是多少？", [])
        self.assertEqual(plan.strategy, "single")
        self.assertFalse(plan.model_invoked)
        self.assertEqual(models.calls, [])

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


if __name__ == "__main__":
    unittest.main()
