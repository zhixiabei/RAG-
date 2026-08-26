import unittest
from types import SimpleNamespace

from rag_app.infrastructure.ollama.gateway import OllamaGateway
from rag_app.infrastructure.openai_compatible.gateway import OpenAICompatibleGateway
from rag_app.model_gateway_factory import build_judge_gateway, build_model_gateway


def settings(**overrides):
    values = {
        "rag_judge_enabled": True,
        "rag_judge_provider_name": "",
        "rag_judge_base_url": "",
        "rag_judge_api_key": "",
        "rag_judge_model": "",
        "rag_judge_timeout_seconds": 60.0,
        "rag_judge_max_retries": 1,
        "rag_judge_retry_base_delay_seconds": 1.0,
        "rag_judge_retry_max_delay_seconds": 5.0,
        "model_mode": "local",
        "ollama_url": "http://ollama",
        "ollama_embedding_model": "embed",
        "remote_llm_provider_name": "Remote",
        "remote_llm_base_url": "https://chat.example/v1",
        "remote_llm_api_key": "chat-key",
        "remote_llm_models": "answer-model",
        "remote_embedding_provider_name": "Embed",
        "remote_embedding_base_url": "https://embed.example/v1",
        "remote_embedding_api_key": "embed-key",
        "remote_embedding_model": "embed-model",
        "remote_llm_timeout_seconds": 60.0,
        "remote_llm_max_retries": 0,
        "remote_embedding_max_retries": 2,
        "remote_llm_retry_base_delay_seconds": 0.5,
        "remote_llm_retry_max_delay_seconds": 5.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class JudgeGatewayFactoryTest(unittest.TestCase):
    def test_builds_remote_answer_gateway_with_bounded_timeout(self):
        result = build_model_gateway(settings(
            model_mode="remote",
            remote_default_chat_model="answer-model",
        ))

        self.assertIsInstance(result, OpenAICompatibleGateway)
        self.assertEqual(result.request_timeout_seconds, 60.0)
        self.assertEqual(result.max_transient_retries, 0)
        self.assertEqual(result.embedding_max_retries, 2)
        self.assertEqual(result.retry_base_delay_seconds, 0.5)
        self.assertEqual(result.retry_max_delay_seconds, 5.0)
        result.close()

    def test_reuses_default_gateway_when_model_is_empty(self):
        default = SimpleNamespace(chat_model="answer-model")

        result = build_judge_gateway(settings(), default)

        self.assertIs(result, default)

    def test_can_disable_judge(self):
        default = SimpleNamespace(chat_model="answer-model")

        result = build_judge_gateway(
            settings(rag_judge_enabled=False),
            default,
        )

        self.assertIsNone(result)

    def test_builds_alternate_ollama_model(self):
        default = SimpleNamespace(chat_model="answer-model")

        result = build_judge_gateway(
            settings(rag_judge_model="judge-model"),
            default,
        )

        self.assertIsInstance(result, OllamaGateway)
        self.assertEqual(result.chat_model, "judge-model")
        self.assertEqual(result.base_url, "http://ollama")

    def test_builds_dedicated_openai_compatible_judge(self):
        default = SimpleNamespace(chat_model="answer-model")

        result = build_judge_gateway(
            settings(
                rag_judge_provider_name="JudgeProvider",
                rag_judge_base_url="https://judge.example/v1/",
                rag_judge_api_key="judge-key",
                rag_judge_model="judge-model",
            ),
            default,
        )

        self.assertIsInstance(result, OpenAICompatibleGateway)
        self.assertEqual(result.provider_name, "JudgeProvider")
        self.assertEqual(result.base_url, "https://judge.example/v1")
        self.assertEqual(result.chat_model, "judge-model")
        self.assertEqual(result.models, ["judge-model"])
        self.assertEqual(result.request_timeout_seconds, 60.0)
        self.assertEqual(result.max_transient_retries, 1)
        self.assertEqual(result.retry_base_delay_seconds, 1.0)
        self.assertEqual(result.retry_max_delay_seconds, 5.0)
        result.check_connection(require_embedding_model=False)
        result.close()

    def test_dedicated_endpoint_requires_model(self):
        default = SimpleNamespace(chat_model="answer-model")

        with self.assertRaisesRegex(ValueError, "RAG_JUDGE_MODEL"):
            build_judge_gateway(
                settings(rag_judge_base_url="https://judge.example/v1"),
                default,
            )


if __name__ == "__main__":
    unittest.main()
