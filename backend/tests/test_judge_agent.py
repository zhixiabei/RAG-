import json
import unittest

from agent.judge_agent import AnswerJudgeAgent, JudgeOutputError


class FakeModels:
    chat_model = "judge-model"
    embedding_model = "embed-model"

    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete(
        self,
        messages,
        model=None,
        temperature=0.1,
        max_tokens=None,
        reasoning=None,
        response_schema=None,
    ):
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": reasoning,
            "response_schema": response_schema,
        })
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class AnswerJudgeAgentTest(unittest.TestCase):
    def test_scores_answer_with_weighted_dimensions(self):
        models = FakeModels(json.dumps({
            "correctness_score": 90,
            "completeness_score": 80,
            "faithfulness_score": 70,
            "reason": "事实正确，遗漏一个次要要点。",
        }))
        agent = AnswerJudgeAgent(models, pass_threshold=0.75)

        judgment = agent.run(
            {
                "question": "项目海拔是多少？",
                "expected_answer": "1012 至 1731 米",
                "evidence_texts": ["项目海拔为 1012 至 1731 米。"],
            },
            "项目海拔约为 1012 至 1731 米。",
        )

        self.assertEqual(judgment.score, 0.81)
        self.assertTrue(judgment.passed)
        self.assertEqual(judgment.model, "judge-model")
        call = models.calls[0]
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["max_tokens"], 300)
        self.assertFalse(call["reasoning"])
        self.assertEqual(
            set(call["response_schema"]["required"]),
            {
                "correctness_score",
                "completeness_score",
                "faithfulness_score",
                "reason",
            },
        )

    def test_bounds_evidence_in_model_input(self):
        models = FakeModels(json.dumps({
            "correctness_score": 100,
            "completeness_score": 100,
            "faithfulness_score": 100,
            "reason": "完整。",
        }))
        agent = AnswerJudgeAgent(models, max_evidence_chars=5)

        agent.run(
            {
                "question": "问题",
                "expected_answer": "答案",
                "evidence_texts": ["1234", "5678"],
            },
            "答案",
        )

        payload = json.loads(models.calls[0]["messages"][1]["content"])
        self.assertEqual(payload["evidence_texts"], ["1234", "5"])

    def test_rejects_invalid_score(self):
        models = FakeModels(json.dumps({
            "correctness_score": 101,
            "completeness_score": 80,
            "faithfulness_score": 80,
            "reason": "无效。",
        }))

        with self.assertRaisesRegex(JudgeOutputError, "0 到 100"):
            AnswerJudgeAgent(models).run({"question": "问题"}, "答案")

    def test_accepts_json_code_fence(self):
        fence = chr(96) * 3
        models = FakeModels(
            fence
            + "json\n"
            + json.dumps({
                "correctness_score": 80,
                "completeness_score": 80,
                "faithfulness_score": 80,
                "reason": "通过。",
            })
            + "\n"
            + fence
        )

        judgment = AnswerJudgeAgent(models).run({"question": "问题"}, "答案")

        self.assertEqual(judgment.score, 0.8)


if __name__ == "__main__":
    unittest.main()