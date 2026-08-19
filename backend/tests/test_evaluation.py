import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from rag_app.evaluation import (
    EvaluationError,
    evaluate_sample,
    find_samples_outside_knowledge_base,
    load_dataset_from_testset_tool,
    measure_retrieval_hits,
    run_evaluation,
    select_dataset_samples,
    summarize,
)


class EvaluationTest(unittest.TestCase):
    def test_measures_document_and_chunk_hits(self):
        sample = {
            "question_id": "q1",
            "question": "测试问题",
            "source_document_ids": ["doc-1"],
            "source_chunk_ids": ["doc-1:3", "doc-1:5"],
        }
        response = {
            "answer": "标准答案",
            "citations": [
                {"document_id": "doc-1", "chunk_id": "doc-1:3"},
            ],
            "retrieved_document_ids": ["doc-1"],
            "retrieved_chunk_ids": ["doc-1:4", "doc-1:3"],
            "retrieval_k": 2,
            "retrieval_used": True,
            "retrieved_count": 3,
            "response_time_ms": 125.5,
            "client_response_time_ms": 150.25,
            "token_usage": {
                "available": True,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        }

        result = measure_retrieval_hits(sample, response)

        self.assertTrue(result["document_hit"])
        self.assertTrue(result["chunk_hit"])
        self.assertEqual(result["document_reciprocal_rank"], 1.0)
        self.assertEqual(result["chunk_precision"], 0.5)
        self.assertEqual(result["chunk_recall"], 0.5)
        self.assertEqual(result["chunk_reciprocal_rank"], 0.5)
        self.assertEqual(result["retrieval_k"], 2)
        self.assertEqual(result["retrieval_recall_at_k"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        self.assertEqual(result["retrieved_chunk_ids"], ["doc-1:4", "doc-1:3"])
        self.assertFalse(any(key.endswith("_f1") for key in result))
        self.assertEqual(result["response_time_ms"], 150.25)
        self.assertEqual(result["server_response_time_ms"], 125.5)
        self.assertEqual(result["token_usage"]["total_tokens"], 120)
        self.assertNotIn("answer_score", result)
        self.assertNotIn("answer_char_f1", result)
        self.assertNotIn("keyword_recall", result)
        self.assertNotIn("passed", result)

    def test_missing_source_ids_are_excluded_from_hit_rates(self):
        sample = {
            "question_id": "q2",
            "question": "没有来源的问题",
            "source_document_ids": [],
            "source_chunk_ids": [],
        }

        result = measure_retrieval_hits(sample, {"citations": []})

        self.assertIsNone(result["document_hit"])
        self.assertIsNone(result["chunk_hit"])

    def test_expected_chunk_must_be_in_full_retrieval_results(self):
        result = measure_retrieval_hits(
            {
                "question_id": "q5",
                "question": "问题",
                "source_document_ids": ["doc-1"],
                "source_chunk_ids": ["doc-1:7"],
            },
            {
                "answer": "答案",
                "citations": [{"document_id": "doc-1", "chunk_id": "doc-1:1"}],
                "retrieved_document_ids": ["doc-1"],
                "retrieved_chunk_ids": ["doc-1:1", "doc-1:2"],
            },
        )

        self.assertTrue(result["document_hit"])
        self.assertFalse(result["chunk_hit"])

    def test_summary_excludes_missing_optional_metrics(self):
        summary = summarize([
            {
                "document_hit": True,
                "chunk_hit": False,
                "document_precision": 0.5,
                "document_recall": 1.0,
                "document_reciprocal_rank": 1.0,
                "chunk_precision": 0.25,
                "chunk_recall": 0.5,
                "chunk_reciprocal_rank": 0.5,
                "retrieval_k": 2,
                "retrieval_recall_at_k": 0.5,
                "retrieval_reciprocal_rank": 0.5,
                "response_time_ms": 100,
                "token_usage": {
                    "available": True,
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                },
            },
            {
                "document_hit": None,
                "chunk_hit": None,
            },
        ])

        self.assertEqual(summary["document_hit_rate"], 1.0)
        self.assertEqual(summary["chunk_hit_rate"], 0.0)
        self.assertEqual(summary["document_mrr"], 1.0)
        self.assertEqual(summary["chunk_mrr"], 0.5)
        self.assertEqual(summary["retrieval_k"], 2)
        self.assertEqual(summary["retrieval_recall_at_k"], 0.5)
        self.assertEqual(summary["mrr"], 0.5)
        self.assertFalse(any(key.endswith("_f1") for key in summary))
        self.assertEqual(summary["average_response_time_ms"], 100)
        self.assertEqual(summary["total_tokens"], 100)
        self.assertEqual(summary["average_tokens"], 100)
        self.assertNotIn("pass_rate", summary)
        self.assertIsNone(summary["average_answer_score"])
        self.assertEqual(summary["judge_sample_count"], 0)
        self.assertEqual(summary["judge_error_count"], 0)
        self.assertNotIn("average_keyword_recall", summary)

    def test_evaluate_sample_keeps_rag_result_when_judge_fails(self):
        chat_payloads = []

        def handle(request):
            if request.method == "POST" and request.url.path.endswith("/conversations"):
                return httpx.Response(200, json={"id": "conversation-1"})
            if request.method == "POST" and request.url.path.endswith("/chat"):
                chat_payloads.append(json.loads(request.content))
                return httpx.Response(
                    200,
                    json={
                        "answer": "生成答案",
                        "citations": [],
                        "retrieval_used": True,
                        "retrieved_count": 0,
                        "retrieval_k": 10,
                    },
                )
            if request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(404)

        class FailingJudge:
            model_name = "judge-model"

            def run(self, sample, answer):
                raise RuntimeError("judge unavailable")

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            result = evaluate_sample(
                client,
                "http://backend.local",
                "kb-1",
                {"question_id": "q1", "question": "问题"},
                None,
                True,
                FailingJudge(),
            )

        self.assertEqual(result["answer"], "生成答案")
        self.assertIn("judge unavailable", result["judge_error"])
        self.assertFalse(result["judge_token_usage"]["available"])
        self.assertEqual(
            chat_payloads,
            [{
                "conversation_id": "conversation-1",
                "question": "问题",
                "model": None,
            }],
        )
    def test_summarizes_judge_scores_and_usage(self):
        summary = summarize([
            {
                "document_hit": True,
                "chunk_hit": True,
                "judge": {
                    "score": 0.8,
                    "correctness_score": 0.9,
                    "completeness_score": 0.7,
                    "faithfulness_score": 0.8,
                    "passed": True,
                },
                "judge_token_usage": {
                    "available": True,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            },
            {
                "document_hit": False,
                "chunk_hit": False,
                "judge": {
                    "score": 0.6,
                    "correctness_score": 0.5,
                    "completeness_score": 0.7,
                    "faithfulness_score": 0.6,
                    "passed": False,
                },
                "judge_token_usage": {"available": False},
            },
            {
                "document_hit": True,
                "chunk_hit": True,
                "judge_error": "Judge unavailable",
                "judge_token_usage": {"available": False},
            },
        ])

        self.assertEqual(summary["judge_sample_count"], 2)
        self.assertEqual(summary["judge_error_count"], 1)
        self.assertEqual(summary["answer_pass_rate"], 0.5)
        self.assertEqual(summary["average_answer_score"], 0.7)
        self.assertEqual(summary["average_correctness_score"], 0.7)
        self.assertEqual(summary["average_completeness_score"], 0.7)
        self.assertEqual(summary["average_faithfulness_score"], 0.7)
        self.assertEqual(summary["judge_total_tokens"], 120)
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

    def test_exports_only_selected_question_ids_from_testset_tool(self):
        def handle(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["scope"], "selected")
            self.assertEqual(payload["questionIds"], ["q2", "q1"])
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "metadata": {"dataset_name": "rag_eval", "sample_count": 2},
                        "samples": [
                            {"question_id": "q1", "question": "first", "status": "approved"},
                            {"question_id": "q2", "question": "second", "status": "approved"},
                        ],
                    },
                },
            )

        samples, _ = load_dataset_from_testset_tool(
            "http://testset.local",
            10,
            question_ids=["q2", "q1"],
            transport=httpx.MockTransport(handle),
        )

        self.assertEqual({sample["question_id"] for sample in samples}, {"q1", "q2"})

    def test_rejects_missing_selected_question_id(self):
        with self.assertRaisesRegex(EvaluationError, "不存在"):
            select_dataset_samples(
                [{"question_id": "q1", "question": "first"}],
                ["q1", "missing"],
            )

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

    @patch("rag_app.evaluation.evaluate_sample")
    @patch("rag_app.evaluation.httpx.Client")
    def test_stops_remaining_samples_after_request_timeout(self, client_class, evaluate_sample):
        client = client_class.return_value.__enter__.return_value
        client.get.return_value = httpx.Response(200)
        evaluate_sample.side_effect = [
            {
                "question_id": "q1",
                "question": "first",
                "document_hit": True,
                "chunk_hit": True,
            },
            httpx.ReadTimeout("timed out"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "evaluation.jsonl"
            dataset.write_text(
                "\n".join(
                    json.dumps({
                        "question_id": f"q{index}",
                        "question": f"question {index}",
                        "status": "approved",
                    })
                    for index in range(1, 4)
                ),
                encoding="utf-8",
            )
            report = run_evaluation(
                dataset,
                "kb-1",
                "http://backend.local",
                None,
                180.0,
                True,
                False,
            )

        self.assertEqual(evaluate_sample.call_count, 2)
        self.assertTrue(report["summary"]["stopped_early"])
        self.assertEqual(report["summary"]["requested_count"], 3)
        self.assertEqual(report["summary"]["remaining_count"], 1)
        self.assertIn("q2", report["stop_reason"])
        self.assertIn("180 秒", report["results"][-1]["error"])


if __name__ == "__main__":
    unittest.main()
