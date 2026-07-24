import json
import unittest

from agent.context import format_retrieved_context
from agent.context_compression_agent import ContextCompressionAgent
from rag_app.domain.models import SearchHit


class FakeModels:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        return self.output


class FailingModels(FakeModels):
    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        raise RuntimeError("compression unavailable")


class ContextCompressionAgentTest(unittest.TestCase):
    @staticmethod
    def hit(chunk_id, text):
        return SearchHit(chunk_id, "doc-1", "kb-1", "制度.pdf", text, 0.9, 3)

    def test_short_context_is_unchanged_without_model_call(self):
        models = FakeModels("unused")
        hit = self.hit("chunk-1", "报销上限为 500 元。")

        result = ContextCompressionAgent(models, max_chars=500).run("上限是多少？", [hit])

        self.assertFalse(result.triggered)
        self.assertEqual(result.text_by_chunk_id, {"chunk-1": hit.text})
        self.assertEqual(result.original_chars, result.compressed_chars)
        self.assertEqual(models.calls, [])

    def test_long_context_extracts_verified_evidence_within_budget(self):
        first = self.hit("chunk-1", "背景信息。" * 80 + "报销上限为 500 元。" + "其他信息。" * 80)
        second = self.hit("chunk-2", "住宿费需要主管审批。" * 100)
        output = json.dumps(
            {
                "items": [
                    {"chunk_id": "c1", "excerpts": ["报销上限为 500 元。"]},
                    {"chunk_id": "c2", "excerpts": ["住宿费需要主管审批。"]},
                ]
            },
            ensure_ascii=False,
        )
        models = FakeModels(output)

        result = ContextCompressionAgent(models, max_chars=180).run("报销规则是什么？", [first, second])
        kept_hits = [hit for hit in [first, second] if hit.chunk_id in result.kept_chunk_ids]
        rendered = format_retrieved_context(kept_hits, result.text_by_chunk_id)

        self.assertTrue(result.triggered)
        self.assertLessEqual(len(rendered), 180)
        self.assertIn("报销上限为 500 元。", rendered)
        self.assertIn("住宿费需要主管审批。", rendered)
        self.assertEqual(result.compressed_chars, len(rendered))
        self.assertEqual(models.calls[0][2], 0)
        self.assertEqual(models.calls[0][4], False)
        self.assertEqual(models.calls[0][5]["required"], ["items"])

    def test_unverified_model_excerpt_is_rejected_and_falls_back(self):
        source = "报销流程要求先提交发票，然后由主管审批。" * 40
        hit = self.hit("chunk-1", source)
        models = FakeModels('{"items":[{"chunk_id":"c1","excerpts":["报销无限额"]}]}')

        result = ContextCompressionAgent(models, max_chars=100).run("报销流程", [hit])

        compressed = result.text_by_chunk_id["chunk-1"]
        self.assertNotIn("报销无限额", compressed)
        self.assertIn(compressed, source)
        self.assertLessEqual(result.compressed_chars, 100)

    def test_invalid_model_output_still_enforces_budget(self):
        hits = [
            self.hit("chunk-1", "甲" * 500),
            self.hit("chunk-2", "乙" * 500),
        ]
        models = FakeModels("not-json")

        result = ContextCompressionAgent(models, max_chars=140).run("问题", hits)
        kept_hits = [hit for hit in hits if hit.chunk_id in result.kept_chunk_ids]

        self.assertTrue(result.triggered)
        self.assertLessEqual(len(format_retrieved_context(kept_hits, result.text_by_chunk_id)), 140)

    def test_model_failure_uses_deterministic_fallback(self):
        hit = self.hit("chunk-1", "报销流程要求主管审批。" * 50)
        models = FailingModels("")

        result = ContextCompressionAgent(models, max_chars=100).run("报销流程", [hit])

        self.assertTrue(result.triggered)
        self.assertLessEqual(result.compressed_chars, 100)
        self.assertTrue(result.text_by_chunk_id)


if __name__ == "__main__":
    unittest.main()
