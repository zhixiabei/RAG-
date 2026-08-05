import json
import unittest

import httpx

from rag_app.evaluation import (
    EvaluationError,
    character_f1,
    cosine_similarity,
    find_samples_outside_knowledge_base,
    is_refusal,
    load_dataset_from_testset_tool,
    select_dataset_samples,
    score_response,
    semantic_answer_score,
    summarize,
)


class FakeEmbeddingModels:
    embedding_model = "fake-embedding"

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return self.embeddings


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
            "retrieved_document_ids": ["doc-1"],
            "retrieved_chunk_ids": ["doc-1:3", "doc-1:4"],
            "retrieval_used": True,
            "retrieved_count": 3,
        }

        result = score_response(
            sample,
            response,
            96.5,
            min_answer_score=75,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["document_hit"])
        self.assertTrue(result["chunk_hit"])
        self.assertEqual(result["answer_score"], 96.5)
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
            min_answer_score=75,
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

    def test_detects_no_result_refusal_variant(self):
        answer = "没有找到任何关于北京市2024年PM2.5月均浓度的数据或分析内容。"

        self.assertTrue(is_refusal(answer))

    def test_vector_score_replaces_character_f1_as_pass_condition(self):
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
            88.0,
            min_answer_score=75,
        )

        self.assertLess(result["answer_char_f1"], 0.1)
        self.assertTrue(result["answer_score_passed"])
        self.assertTrue(result["passed"])

    def test_vector_score_must_reach_threshold(self):
        result = score_response(
            {
                "question_id": "q4",
                "question": "问题",
                "expected_answer": "答案",
                "should_refuse": False,
            },
            {"answer": "候选答案", "citations": []},
            74.9,
            min_answer_score=75,
        )

        self.assertEqual(result["answer_score"], 74.9)
        self.assertFalse(result["answer_score_passed"])
        self.assertFalse(result["passed"])

    def test_expected_chunk_must_be_in_full_retrieval_results(self):
        result = score_response(
            {
                "question_id": "q5",
                "question": "问题",
                "expected_answer": "答案",
                "source_document_ids": ["doc-1"],
                "source_chunk_ids": ["doc-1:7"],
                "should_refuse": False,
            },
            {
                "answer": "答案",
                "citations": [{"document_id": "doc-1", "chunk_id": "doc-1:1"}],
                "retrieved_document_ids": ["doc-1"],
                "retrieved_chunk_ids": ["doc-1:1", "doc-1:2"],
            },
            100.0,
            min_answer_score=75,
        )

        self.assertTrue(result["document_hit"])
        self.assertFalse(result["chunk_hit"])
        self.assertFalse(result["passed"])

    def test_cosine_similarity_and_semantic_percentage(self):
        self.assertEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

        models = FakeEmbeddingModels([[1.0, 0.0], [0.8, 0.6]])
        score = semantic_answer_score(models, "标准答案", "实际答案")

        self.assertEqual(score, 80.0)
        self.assertEqual(models.calls, [["标准答案", "实际答案"]])

    def test_rejects_zero_norm_answer_vector(self):
        models = FakeEmbeddingModels([[1.0, 0.0], [0.0, 0.0]])

        with self.assertRaisesRegex(EvaluationError, "范数为 0"):
            semantic_answer_score(models, "标准答案", "实际答案")

    def test_summary_excludes_missing_optional_metrics(self):
        summary = summarize([
            {
                "passed": True,
                "document_hit": True,
                "chunk_hit": False,
                "refusal_correct": True,
                "expected_refusal": False,
                "answer_score": 87.5,
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
                "answer_char_f1": 0.2,
                "keyword_recall": 0.5,
            },
        ])

        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["document_hit_rate"], 1.0)
        self.assertEqual(summary["chunk_hit_rate"], 0.0)
        self.assertEqual(summary["average_answer_score"], 87.5)
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


if __name__ == "__main__":
    unittest.main()
