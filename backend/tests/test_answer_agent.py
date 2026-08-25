import unittest

from agent.answer_agent import AnswerAgent
from rag_app.domain.models import SearchHit


class FakeModels:
    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None, temperature=0.1, max_tokens=None, reasoning=None, response_schema=None):
        self.calls.append((messages, model, temperature, max_tokens, reasoning, response_schema))
        return "最终回答"


class AnswerAgentTest(unittest.TestCase):
    def test_returns_no_relevant_content_after_unsuccessful_retrieval(self):
        models = FakeModels()

        answer = AnswerAgent(models).run("未知问题", [], [], retrieval_used=True)

        self.assertEqual(answer, "知识库中无相关内容。")
        self.assertEqual(models.calls, [])

    def test_rejects_when_all_reranker_scores_are_below_threshold(self):
        models = FakeModels()
        low_hit = SearchHit(
            "chunk-low", "doc-1", "kb-1", "unrelated.pdf", "unrelated content",
            0.9, relevance_score=0.05,
        )

        answer = AnswerAgent(models).run("out-of-scope question", [], [low_hit], retrieval_used=True)

        self.assertEqual(answer, "知识库中无相关内容。")
        self.assertEqual(models.calls, [])

    def test_only_passes_hits_at_or_above_relevance_threshold(self):
        models = FakeModels()
        low_hit = SearchHit(
            "chunk-low", "doc-1", "kb-1", "low.pdf", "irrelevant evidence",
            0.9, relevance_score=0.05,
        )
        high_hit = SearchHit(
            "chunk-high", "doc-2", "kb-1", "high.pdf", "relevant evidence",
            0.8, relevance_score=0.75,
        )

        result = AnswerAgent(models).run_with_context(
            "supported question", [], [low_hit, high_hit], retrieval_used=True
        )

        self.assertEqual([hit.chunk_id for hit in result.selected_hits], ["chunk-high"])
        context_message = models.calls[0][0][-2]["content"]
        self.assertIn("relevant evidence", context_message)
        self.assertNotIn("irrelevant evidence", context_message)

    def test_keeps_hits_without_reranker_relevance_scores(self):
        models = FakeModels()
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "source.pdf", "source content", 0.01)

        answer = AnswerAgent(models).run("supported question", [], [hit], retrieval_used=True)

        self.assertEqual(answer, "最终回答")
        self.assertEqual(len(models.calls), 1)

    def test_answers_folder_question_from_catalog_without_retrieved_chunks(self):
        models = FakeModels()
        catalog = "[文件夹]\n- 井资料/化163-1井\n[文件]\n- 井资料/化163-1井/施工设计.docx"

        answer = AnswerAgent(models).run(
            "化163-1井文件夹里有什么？",
            [],
            [],
            retrieval_used=True,
            knowledge_catalog=catalog,
        )

        self.assertEqual(answer, "最终回答")
        context_message = models.calls[0][0][-2]["content"]
        self.assertIn("knowledge_base_catalog", context_message)
        self.assertIn("井资料/化163-1井/施工设计.docx", context_message)

    def test_returns_deterministic_catalog_answer_without_model_call(self):
        models = FakeModels()

        answer = AnswerAgent(models).run(
            "化163-1井文件夹里有哪些文件？",
            [],
            [],
            retrieval_used=False,
            catalog_answer="化163-1井中共有 **2** 个已入库文件：\n- `施工设计.docx`\n- `施工总结.pdf`",
        )

        self.assertIn("施工设计.docx", answer)
        self.assertIn("施工总结.pdf", answer)
        self.assertEqual(models.calls, [])

    def test_unsuccessful_retrieval_does_not_fall_back_to_unrelated_history(self):
        models = FakeModels()

        answer = AnswerAgent(models).run(
            "新的问题",
            [{"role": "assistant", "content": "旧问题的回答"}],
            [],
            retrieval_used=True,
        )

        self.assertEqual(answer, "知识库中无相关内容。")
        self.assertEqual(models.calls, [])

    def test_builds_context_from_hits(self):
        models = FakeModels()
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.9, 3)

        answer = AnswerAgent(models).run("上限是多少？", [], [hit], retrieval_used=True)

        self.assertEqual(answer, "最终回答")
        messages = models.calls[0][0]
        self.assertEqual(messages[-1], {"role": "user", "content": "上限是多少？"})
        self.assertIn("行内数学公式必须用 $...$", messages[0]["content"])
        self.assertIn("默认使用 2 至 5 个简短段落", messages[0]["content"])
        self.assertIn("每个要点最多两句", messages[0]["content"])
        self.assertIn("[证据:证据ID]", messages[0]["content"])
        self.assertEqual(models.calls[0][3], 640)
        self.assertFalse(models.calls[0][4])
        self.assertIn("制度.pdf", messages[-2]["content"])
        self.assertIn("报销上限为 500 元", messages[-2]["content"])

    def test_uses_tighter_output_limit_for_brief_questions(self):
        models = FakeModels()
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.9, 3)

        AnswerAgent(models).run("请简要概括上限", [], [hit], retrieval_used=True)

        self.assertEqual(models.calls[0][3], 512)


    def test_drops_incomplete_tail_block_from_brief_answer(self):
        models = FakeModels()
        models.complete = lambda *args, **kwargs: "结论。[证据:chunk-1]\n\n- 未完成 [证据:chunk-"
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.9, 3)

        answer = AnswerAgent(models).run("只概括上限", [], [hit], retrieval_used=True)

        self.assertEqual(answer, "结论。[证据:chunk-1]")

    def test_keeps_complete_brief_answer(self):
        models = FakeModels()
        models.complete = lambda *args, **kwargs: "第一点。[证据:chunk-1]\n\n第二点。"
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.9, 3)

        answer = AnswerAgent(models).run("只概括上限", [], [hit], retrieval_used=True)

        self.assertEqual(answer, "第一点。[证据:chunk-1]\n\n第二点。")
    def test_allows_configured_output_limit_for_explicit_detailed_questions(self):
        models = FakeModels()
        hit = SearchHit("chunk-1", "doc-1", "kb-1", "制度.pdf", "报销上限为 500 元", 0.9, 3)

        AnswerAgent(models, max_output_tokens=900).run(
            "请详细分析并形成完整报告",
            [],
            [hit],
            retrieval_used=True,
        )

    def test_context_uses_original_file_name_with_suffix(self):
        models = FakeModels()
        hit = SearchHit(
            "chunk-1",
            "doc-1",
            "kb-1",
            "长63渗透率",
            "渗透率数据",
            0.9,
            folder_path="井资料/长63/渗透率",
            file_name="长63渗透率.gdb",
        )

        AnswerAgent(models).run("这是什么文件？", [], [hit], retrieval_used=True)

        context_message = models.calls[0][0][-2]["content"]
        self.assertIn("长63渗透率.gdb", context_message)
        self.assertIn("井资料/长63/渗透率/长63渗透率.gdb", context_message)

    def test_returns_selected_model_for_model_identity_question_without_completion(self):
        models = FakeModels()

        answer = AnswerAgent(models).run("你是什么大模型啊", [], [], retrieval_used=False, model="qwen3:4b")

        self.assertEqual(answer, "我是知识库助手，当前回答使用的模型是 qwen3:4b。")
        self.assertEqual(models.calls, [])


if __name__ == "__main__":
    unittest.main()
