import unittest
from unittest.mock import Mock, patch

from rag_app.infrastructure.ollama.gateway import OllamaGateway
from rag_app.infrastructure.openai_compatible.gateway import OpenAICompatibleGateway


class ModelCompletionGatewayTest(unittest.TestCase):
    @patch("rag_app.infrastructure.ollama.gateway.httpx.post")
    def test_ollama_maps_completion_options(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": " SKIP "}}
        post.return_value = response
        gateway = OllamaGateway("http://ollama", "qwen", "embed")
        messages = [{"role": "user", "content": "question"}]

        result = gateway.complete(messages, temperature=0, max_tokens=8, reasoning=False)

        self.assertEqual(result, "SKIP")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen")
        self.assertEqual(payload["options"], {"temperature": 0, "num_predict": 8})
        self.assertFalse(payload["think"])
        response.raise_for_status.assert_called_once()

    @patch("rag_app.infrastructure.openai_compatible.gateway.httpx.post")
    def test_openai_compatible_maps_completion_options(self, post):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": " answer "}}]}
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

        result = gateway.complete(messages, temperature=0, max_tokens=8, reasoning=False)

        self.assertEqual(result, "answer")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "chat-model")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 8)
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
