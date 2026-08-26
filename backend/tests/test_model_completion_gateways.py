import unittest
from unittest.mock import Mock, patch

import httpx

from rag_app.infrastructure.ollama.gateway import OllamaGateway
from rag_app.infrastructure.openai_compatible.gateway import OpenAICompatibleGateway
from agent.telemetry import collect_model_usage, model_usage_stage


class ModelCompletionGatewayTest(unittest.TestCase):
    @patch("rag_app.infrastructure.ollama.gateway.httpx.get")
    def test_ollama_can_check_a_chat_only_compression_model(self, get):
        response = Mock()
        response.json.return_value = {"models": [{"name": "qwen3:4b"}]}
        get.return_value = response
        gateway = OllamaGateway("http://ollama", "qwen3:4b", "missing-embed")

        gateway.check_connection(require_embedding_model=False)

        response.raise_for_status.assert_called_once()

    @patch("rag_app.infrastructure.ollama.gateway.httpx.post")
    def test_ollama_maps_completion_options(self, post):
        response = Mock()
        response.json.return_value = {
            "message": {"content": " SKIP "},
            "prompt_eval_count": 40,
            "eval_count": 5,
        }
        post.return_value = response
        gateway = OllamaGateway("http://ollama", "qwen", "embed")
        messages = [{"role": "user", "content": "question"}]

        with collect_model_usage() as collector, model_usage_stage("retrieval_decision"):
            result = gateway.complete(messages, temperature=0, max_tokens=8, reasoning=False)

        self.assertEqual(result, "SKIP")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen")
        self.assertEqual(payload["options"], {"temperature": 0, "num_predict": 8})
        self.assertFalse(payload["think"])
        self.assertEqual(collector.summary()["total_tokens"], 45)
        self.assertEqual(collector.summary()["by_stage"]["retrieval_decision"]["input_tokens"], 40)
        response.raise_for_status.assert_called_once()

    @patch("rag_app.infrastructure.ollama.gateway.httpx.post")
    def test_ollama_maps_response_schema(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": '{"decision":"SKIP"}'}}
        post.return_value = response
        gateway = OllamaGateway("http://ollama", "qwen", "embed")
        schema = {"type": "object", "required": ["decision"]}

        gateway.complete([], response_schema=schema)

        self.assertEqual(post.call_args.kwargs["json"]["format"], schema)

    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_maps_completion_options(self, client_class):
        post = client_class.return_value.post
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": " answer "}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
        }
        post.return_value = response
        gateway = OpenAICompatibleGateway(
            "Provider",
            "https://chat.example/v1",
            "chat-key",
            ["chat-model"],
            "chat-model",
            "EmbeddingProvider",
            "https://embed.example/v1",
            "embed-key",
            "embed-model",
        )
        messages = [{"role": "user", "content": "question"}]

        with collect_model_usage() as collector, model_usage_stage("answer_generation"):
            result = gateway.complete(messages, temperature=0, max_tokens=8, reasoning=False)

        self.assertEqual(result, "answer")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "chat-model")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(collector.summary()["input_tokens"], 80)
        self.assertEqual(collector.summary()["output_tokens"], 20)
        self.assertEqual(collector.summary()["total_tokens"], 100)
        response.raise_for_status.assert_called_once()

    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_qwen_completion_disables_thinking(self, client_class):
        post = client_class.return_value.post
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": "answer"}}],
        }
        post.return_value = response
        gateway = OpenAICompatibleGateway(
            "SiliconFlow", "https://chat.example/v1", "chat-key",
            ["Qwen/Qwen3-8B"], "Qwen/Qwen3-8B",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
        )

        gateway.complete([], reasoning=False)

        self.assertIs(post.call_args.kwargs["json"]["enable_thinking"], False)
    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_reuses_client_across_completions(self, client_class):
        post = client_class.return_value.post
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": "answer"}}],
        }
        post.return_value = response
        gateway = OpenAICompatibleGateway(
            "Provider", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
        )

        gateway.complete([])
        gateway.complete([])
        gateway.close()

        client_class.assert_called_once()
        self.assertEqual(post.call_count, 2)
        client_class.return_value.close.assert_called_once()

    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_requests_json_for_response_schema(self, client_class):
        post = client_class.return_value.post
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": '{"decision":"SKIP"}'}}]}
        post.return_value = response
        gateway = OpenAICompatibleGateway(
            "Provider", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
        )

        gateway.complete([], response_schema={"type": "object"})

        self.assertEqual(
            post.call_args.kwargs["json"]["response_format"],
            {"type": "json_object"},
        )

    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_retries_when_json_mode_is_unsupported(self, client_class):
        post = client_class.return_value.post
        unsupported = Mock(status_code=400)
        success = Mock(status_code=200)
        success.json.return_value = {"choices": [{"message": {"content": '{"decision":"SKIP"}'}}]}
        post.side_effect = [unsupported, success]
        gateway = OpenAICompatibleGateway(
            "Provider", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
        )

        result = gateway.complete([], response_schema={"type": "object"})

        self.assertEqual(result, '{"decision":"SKIP"}')
        self.assertEqual(post.call_count, 2)
        self.assertNotIn("response_format", post.call_args.kwargs["json"])
        success.raise_for_status.assert_called_once()

    @patch("rag_app.infrastructure.openai_compatible.gateway.time.sleep")
    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_retries_transient_http_failures(self, client_class, sleep):
        post = client_class.return_value.post
        unavailable = Mock(status_code=503, headers={})
        success = Mock(status_code=200)
        success.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        post.side_effect = [unavailable, success]
        gateway = OpenAICompatibleGateway(
            "DeepSeek", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
        )

        result = gateway.complete([{"role": "user", "content": "question"}])

        self.assertEqual(result, "answer")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)
        success.raise_for_status.assert_called_once()

    @patch("rag_app.infrastructure.openai_compatible.gateway.time.sleep")
    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_retries_connection_failure(self, client_class, sleep):
        post = client_class.return_value.post
        success = Mock(status_code=200)
        success.json.return_value = {
            "choices": [{"message": {"content": "answer"}}],
        }
        post.side_effect = [httpx.ConnectError("connection reset"), success]
        gateway = OpenAICompatibleGateway(
            "Provider", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
            max_transient_retries=1,
        )

        self.assertEqual(gateway.complete([]), "answer")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)

    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_identifies_chat_timeout(self, client_class):
        post = client_class.return_value.post
        post.side_effect = httpx.ReadTimeout("timed out")
        gateway = OpenAICompatibleGateway(
            "DeepSeek", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
            max_transient_retries=0,
        )

        with self.assertRaisesRegex(RuntimeError, "DeepSeek 聊天接口连续 1 次超时"):
            gateway.complete([{"role": "user", "content": "question"}])

    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_identifies_embedding_timeout(self, client_class):
        post = client_class.return_value.post
        post.side_effect = httpx.ReadTimeout("timed out")
        gateway = OpenAICompatibleGateway(
            "DeepSeek", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "EmbeddingProvider", "https://embed.example/v1", "embed-key", "embed-model",
            max_transient_retries=0,
        )

        with self.assertRaisesRegex(RuntimeError, "EmbeddingProvider 接口连续 1 次超时"):
            gateway.embed(["question"])

    @patch("rag_app.infrastructure.openai_compatible.gateway.time.sleep")
    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_retries_embedding_http_400(self, client_class, sleep):
        post = client_class.return_value.post
        bad_request = Mock(status_code=400, headers={}, text="")
        bad_request.json.return_value = {"error": {"message": "temporary failure"}}
        success = Mock(status_code=200)
        success.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }
        post.side_effect = [bad_request, success]
        gateway = OpenAICompatibleGateway(
            "DeepSeek", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "SiliconFlow", "https://embed.example/v1", "embed-key", "embed-model",
            max_transient_retries=0,
            embedding_max_retries=2,
        )

        result = gateway.embed(["question"])

        self.assertEqual(result, [[0.1, 0.2]])
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)

    @patch("rag_app.infrastructure.openai_compatible.gateway.time.sleep")
    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.Client")
    def test_openai_compatible_reports_embedding_http_error_detail(self, client_class, sleep):
        post = client_class.return_value.post
        bad_request = Mock(status_code=400, headers={}, text="")
        bad_request.json.return_value = {
            "error": {"message": "provider rejected the embedding request"},
        }
        post.return_value = bad_request
        gateway = OpenAICompatibleGateway(
            "DeepSeek", "https://chat.example/v1", "chat-key", ["chat-model"], "chat-model",
            "SiliconFlow", "https://embed.example/v1", "embed-key", "embed-model",
            max_transient_retries=0,
            embedding_max_retries=1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "SiliconFlow 接口连续 2 次返回 HTTP 400: provider rejected the embedding request",
        ):
            gateway.embed(["question"])

        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.5)


if __name__ == "__main__":
    unittest.main()
