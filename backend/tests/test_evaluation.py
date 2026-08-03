import json
import unittest

import httpx

from rag_app.evaluation import (
    EvaluationError,
    character_f1,
    find_samples_outside_knowledge_base,
    is_refusal,
    judge_answer,
    load_dataset_from_testset_tool,
    parse_answer_judgement,
    score_response,
    summarize,
)


class FakeJudgeModels:
    chat_model = "judge-model"

    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.output


class EvaluationTest(unittest.TestCase):
    def test_character_f1_ignores_spacing_case_and_punctuation(self):
        self.assertEqual(character_f1("Ab，中文!", "a b 中文"), 1.0)

    def test_scores_retrieval_answer_and_keywords(self):
        sample = {
            "question_id": "q1",
            "question": "测试问题",
            "expected_answer": "长6储层属于低孔、特低渗储层",
            "source_document_ids": ["doc-1"],
            "source_chunk_ids": ["doc-1:3"],
            "keywords": ["低孔", "特低渗"],
            "should_refuse": False,
        }
        response = {
            "answer": "长6储层总体属于低孔、特低渗储层。",
            "citations": [
                {"document_id": "doc-1", "chunk_id": "doc-1:3"},
            ],
            "retrieval_used": True,
            "retrieved_count": 3,
        }

        result = score_response(
            sample,
            response,
            {
                "correctness": 4,
                "completeness": 4,
                "faithfulness": 4,
                "relevance": 4,
                "reason": "关键事实完整且有证据支持。",
            },
            min_answer_score=3,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["document_hit"])
        self.assertTrue(result["chunk_hit"])
        self.assertEqual(result["answer_score"], 4.0)
        self.assertEqual(result["keyword_recall"], 1.0)

    def test_scores_expected_refusal(self):
        sample = {
            "question_id": "q2",
            "question": "库外问题",
            "expected_answer": "",
            "source_document_ids": [],
            "source_chunk_ids": [],
            "keywords": [],
            "should_refuse": True,
        }

        result = score_response(
            sample,
            {
                "answer": "知识库中无相关内容。",
                "citations": [],
                "retrieval_used": True,
            },
            None,
            min_answer_score=3,
        )

        self.assertTrue(is_refusal(result["answer"]))
        self.assertTrue(result["passed"])
        self.assertIsNone(result["document_hit"])

    def test_detects_explanatory_refusal(self):
        answer = (
            "当前知识库没有任何关于北京市2024年PM2.5月均浓度的数据。"
            "因此，我无法基于这些资料分析季节变化原因。"
        )

        self.assertTrue(is_refusal(answer))

    def test_semantic_judge_replaces_character_f1_as_pass_condition(self):
        sample = {
            "question_id": "q3",
            "question": "说明构造演化阶段",
            "expected_answer": "经历六个构造演化阶段。",
            "source_document_ids": ["doc-1"],
            "source_chunk_ids": ["doc-1:1"],
            "should_refuse": False,
        }
        response = {
            "answer": "详细背景。" * 100 + "该盆地经历六个构造演化阶段。",
            "citations": [{"document_id": "doc-1", "chunk_id": "doc-1:1"}],
        }

        result = score_response(
            sample,
            response,
            {
                "correctness": 4,
                "completeness": 4,
                "faithfulness": 3,
                "relevance": 3,
                "reason": "答案较长，但结论完整且扩展内容有证据支持。",
            },
            min_answer_score=3,
        )

        self.assertLess(result["answer_char_f1"], 0.1)
        self.assertTrue(result["answer_judge_passed"])
        self.assertTrue(result["passed"])

    def test_judge_requires_every_dimension_to_reach_threshold(self):
        result = score_response(
            {
                "question_id": "q4",
                "question": "问题",
                "expected_answer": "答案",
                "should_refuse": False,
            },
            {"answer": "候选答案", "citations": []},
            {
                "correctness": 4,
                "completeness": 2,
                "faithfulness": 4,
                "relevance": 4,
                "reason": "遗漏关键点。",
            },
            min_answer_score=3,
        )

        self.assertEqual(result["answer_score"], 3.5)
        self.assertFalse(result["answer_judge_passed"])
        self.assertFalse(result["passed"])

    def test_judge_uses_reference_evidence_and_structured_output(self):
        models = FakeJudgeModels(json.dumps({
            "correctness": 4,
            "completeness": 3,
            "faithfulness": 4,
            "relevance": 3,
            "reason": "答案覆盖关键点。",
        }, ensure_ascii=False))

        judgement = judge_answer(
            models,
            {
                "question": "测试问题",
                "expected_answer": "参考答案",
                "evidence_texts": ["证据原文"],
            },
            "候选答案",
        )

        self.assertEqual(judgement["completeness"], 3.0)
        messages, kwargs = models.calls[0]
        judge_input = json.loads(messages[1]["content"])
        self.assertEqual(judge_input["reference_answer"], "参考答案")
        self.assertIn("证据原文", judge_input["evidence"])
        self.assertEqual(kwargs["temperature"], 0)
        self.assertIsNotNone(kwargs["response_schema"])

    def test_rejects_invalid_judge_scores(self):
        with self.assertRaisesRegex(EvaluationError, "correctness 超出"):
            parse_answer_judgement(json.dumps({
                "correctness": 5,
                "completeness": 4,
                "faithfulness": 4,
                "relevance": 4,
                "reason": "invalid",
            }))

    def test_summary_excludes_missing_optional_metrics(self):
        summary = summarize([
            {
                "passed": True,
                "document_hit": True,
                "chunk_hit": False,
                "refusal_correct": True,
                "expected_refusal": False,
                "answer_score": 3.5,
                "answer_judge": {
                    "correctness": 4,
                    "completeness": 3,
                    "faithfulness": 4,
                    "relevance": 3,
                },
                "answer_char_f1": 0.8,
                "keyword_recall": None,
            },
            {
                "passed": False,
                "document_hit": None,
                "chunk_hit": None,
                "refusal_correct": False,
                "expected_refusal": True,
                "answer_score": None,
                "answer_judge": None,
                "answer_char_f1": 0.2,
                "keyword_recall": 0.5,
            },
        ])

        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["document_hit_rate"], 1.0)
        self.assertEqual(summary["chunk_hit_rate"], 0.0)
        self.assertEqual(summary["average_answer_score"], 3.5)
        self.assertEqual(summary["average_answer_char_f1"], 0.8)
        self.assertEqual(summary["average_keyword_recall"], 0.5)

    def test_loads_approved_samples_directly_from_testset_tool(self):
        def handle(request):
            self.assertEqual(request.url.path, "/api/datasets/export")
            self.assertEqual(json.loads(request.content)["scope"], "approved")
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "metadata": {"dataset_name": "rag_eval", "sample_count": 1},
                        "samples": [
                            {
                                "question_id": "q1",
                                "question": "question",
                                "status": "approved",
                            }
                        ],
                    },
                },
            )

        samples, metadata = load_dataset_from_testset_tool(
            "http://testset.local/",
            10,
            transport=httpx.MockTransport(handle),
        )

        self.assertEqual(samples[0]["question_id"], "q1")
        self.assertEqual(metadata["sample_count"], 1)

    def test_finds_legacy_source_ids_before_remote_evaluation(self):
        mismatched = find_samples_outside_knowledge_base(
            [
                {"question_id": "old", "source_document_ids": ["doc_hua31_report"]},
                {"question_id": "new", "source_document_ids": ["uuid-document"]},
                {"question_id": "refusal", "source_document_ids": []},
            ],
            {"uuid-document"},
        )

        self.assertEqual([sample["question_id"] for sample in mismatched], ["old"])


if __name__ == "__main__":
    unittest.main()
