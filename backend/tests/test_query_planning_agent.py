import json
import unittest

from agent.query_planning_agent import QueryPlanningAgent, query_planning_trigger


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
        with self.assertLogs("agent.query_planning_agent", level="WARNING"):
            plan = QueryPlanningAgent(models).run(question, [])
        self.assertEqual(plan.strategy, "single")
        self.assertTrue(plan.fallback)
        self.assertEqual(plan.retrieval_queries(question), [question])

    def test_short_follow_up_requires_history(self):
        self.assertIsNone(query_planning_trigger("含油率呢？", []))
        self.assertEqual(query_planning_trigger(
            "含油率呢？", [{"role": "assistant", "content": "上一轮内容"}]
        ), "context_reference")


if __name__ == "__main__":
    unittest.main()
