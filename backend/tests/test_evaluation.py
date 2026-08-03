import unittest

from rag_app.evaluation import character_f1, is_refusal, score_response, summarize


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

        result = score_response(sample, response, min_answer_f1=0.35)

        self.assertTrue(result["passed"])
        self.assertTrue(result["document_hit"])
        self.assertTrue(result["chunk_hit"])
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
            min_answer_f1=0.35,
        )

        self.assertTrue(is_refusal(result["answer"]))
        self.assertTrue(result["passed"])
        self.assertIsNone(result["document_hit"])

    def test_summary_excludes_missing_optional_metrics(self):
        summary = summarize([
            {
                "passed": True,
                "document_hit": True,
                "chunk_hit": False,
                "refusal_correct": True,
                "answer_char_f1": 0.8,
                "keyword_recall": None,
            },
            {
                "passed": False,
                "document_hit": None,
                "chunk_hit": None,
                "refusal_correct": False,
                "answer_char_f1": 0.2,
                "keyword_recall": 0.5,
            },
        ])

        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["document_hit_rate"], 1.0)
        self.assertEqual(summary["chunk_hit_rate"], 0.0)
        self.assertEqual(summary["average_answer_char_f1"], 0.5)
        self.assertEqual(summary["average_keyword_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
