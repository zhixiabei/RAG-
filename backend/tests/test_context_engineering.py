import json
import unittest

from agent.answer_agent import AnswerAgent
from agent.context import (
    ContextPolicy,
    build_answer_context,
    estimate_messages_tokens,
    estimate_text_tokens,
    select_history_messages,
)
from rag_app.domain.models import SearchHit


class OverflowOnceModels:
    chat_model = "qwen3:4b"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, **_kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise RuntimeError("maximum context length exceeded")
        return "重试成功"


class ContextEngineeringTest(unittest.TestCase):
    def test_chinese_token_estimate_does_not_use_ascii_chars_divided_by_four(self):
        text = "中" * 100

        self.assertGreaterEqual(estimate_text_tokens(text), 100)

    def test_history_keeps_complete_recent_rounds(self):
        history = []
        for index in range(4):
            history.extend([
                {"role": "user", "content": f"问题{index}" + "甲" * 100},
                {"role": "assistant", "content": f"回答{index}" + "乙" * 100},
            ])

        view = select_history_messages(history, token_budget=240)

        self.assertEqual([message["role"] for message in view.messages], ["user", "assistant"])
        self.assertIn("问题3", view.messages[0]["content"])
        self.assertIn("回答3", view.messages[1]["content"])
        self.assertEqual(view.omitted_count, 6)
        self.assertIn("持久会话中仍保留原文", view.omission_notice)

    def test_oversized_latest_round_is_clipped_without_dropping_one_side(self):
        history = [
            {"role": "user", "content": "问题" + "甲" * 1_000},
            {"role": "assistant", "content": "回答" + "乙" * 1_000},
        ]

        view = select_history_messages(history, token_budget=120)

        self.assertEqual([message["role"] for message in view.messages], ["user", "assistant"])
        self.assertTrue(view.truncated)
        self.assertLessEqual(view.estimated_tokens, 120)

    def test_context_deduplicates_evidence_and_reports_omissions(self):
        hits = [
            SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "甲" * 1_200, 0.9, 1),
            SearchHit("chunk-2", "doc-1", "kb-1", "制度.pdf", "甲" * 1_200, 0.8, 2),
            SearchHit("chunk-3", "doc-2", "kb-1", "流程.pdf", "乙" * 1_200, 0.7, 1),
        ]

        view = build_answer_context(
            system_prompt="系统约束",
            question="当前问题",
            history=[],
            hits=hits,
            policy=ContextPolicy(max_input_tokens=1_200, output_reserve_tokens=200),
        )

        self.assertEqual(view.trace["evidence"]["received"], 3)
        self.assertEqual(view.trace["evidence"]["deduplicated"], 2)
        self.assertEqual(view.trace["evidence"]["selected"], 1)
        self.assertEqual(view.trace["evidence"]["chunk_ids"], ["chunk-1"])
        self.assertLessEqual(
            view.trace["estimated_input_tokens"],
            view.trace["input_budget_tokens"],
        )
        self.assertEqual([hit.chunk_id for hit in view.selected_hits], ["chunk-1"])
        payload = json.loads(view.messages[-2]["content"].split("\n", 1)[1])
        self.assertIn("证据中段因上下文预算省略", payload["retrieved_context"])
        self.assertNotIn("chunk-3", payload["retrieved_context"])

    def test_answer_agent_rebuilds_a_smaller_view_after_context_overflow(self):
        models = OverflowOnceModels()
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "内容" * 2_000, 0.9, 1)
        agent = AnswerAgent(
            models,
            ContextPolicy(max_input_tokens=4_096, output_reserve_tokens=512),
        )

        result = agent.run_with_context("问题", [], [hit], retrieval_used=True)

        self.assertEqual(result.answer, "重试成功")
        self.assertTrue(result.context_trace["overflow_retry"])
        self.assertEqual(len(models.calls), 2)
        self.assertLess(
            estimate_messages_tokens(models.calls[1]),
            estimate_messages_tokens(models.calls[0]),
        )


if __name__ == "__main__":
    unittest.main()
