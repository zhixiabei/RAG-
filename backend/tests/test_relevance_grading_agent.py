import unittest

from agent.relevance_grading_agent import RelevanceGradingAgent
from rag_app.domain.models import SearchHit


class FakeModels:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        return self.output


class RelevanceGradingAgentTest(unittest.TestCase):
    def test_filters_hits_below_threshold_and_preserves_scores(self):
        hits = [
            SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.91),
            SearchHit("chunk-2", "doc-2", "kb-1", "通讯录.pdf", "张三的电话", 0.88),
        ]
        models = FakeModels('{"items":[{"chunk_id":"chunk-1","score":0.92},{"chunk_id":"chunk-2","score":0.12}]}')

        result = RelevanceGradingAgent(models, threshold=0.65).run("报销上限是多少？", hits)

        self.assertEqual([hit.chunk_id for hit in result.relevant_hits], ["chunk-1"])
        self.assertEqual(result.score_for("chunk-1"), 0.92)
        self.assertEqual(result.score_for("chunk-2"), 0.12)
        self.assertEqual(models.calls[0][2:5], (0, 256, False))
        self.assertEqual(models.calls[0][5]["required"], ["items"])

    def test_invalid_output_fails_closed(self):
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "内容", 0.91)

        result = RelevanceGradingAgent(FakeModels("无法判断")).run("问题", [hit])

        self.assertEqual(result.relevant_hits, ())
        self.assertEqual(result.score_for("chunk-1"), 0.0)

    def test_empty_candidates_do_not_call_model(self):
        models = FakeModels("unused")

        result = RelevanceGradingAgent(models).run("问题", [])

        self.assertEqual(result.relevant_hits, ())
        self.assertEqual(models.calls, [])


if __name__ == "__main__":
    unittest.main()
