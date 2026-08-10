import httpx


class OllamaGateway:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def check_connection(
        self,
        require_chat_model: bool = True,
        require_embedding_model: bool = True,
    ) -> None:
        response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        available = {
            item.get("name") or item.get("model")
            for item in response.json().get("models", [])
        }
        required = [self.embedding_model] if require_embedding_model else []
        if require_chat_model:
            required.append(self.chat_model)
        missing = [model for model in required if model not in available]
        if missing:
            raise RuntimeError(f"缺少 Ollama 模型: {', '.join(missing)}")

    def list_chat_models(self) -> list[dict]:
        response = httpx.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = []
        for item in response.json().get("models", []):
            model_id = item.get("name") or item.get("model")
            if not model_id or model_id == self.embedding_model or "embed" in model_id.lower():
                continue
            models.append(
                {
                    "id": model_id,
                    "name": model_id,
                    "provider": "Ollama",
                    "is_default": model_id == self.chat_model,
                    "size": item.get("size"),
                }
            )
        return sorted(models, key=lambda item: (not item["is_default"], item["name"].lower()))

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        reasoning: bool | None = None,
        response_schema: dict | None = None,
    ) -> str:
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": model or self.chat_model,
            "stream": False,
            "options": options,
            "messages": messages,
        }
        if reasoning is not None:
            payload["think"] = reasoning
        if response_schema is not None:
            payload["format"] = response_schema
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        output = response.json().get("message", {}).get("content", "").strip()
        if not output:
            raise RuntimeError("Ollama 未返回文本")
        return output

    def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embedding_model, "input": batch},
                timeout=180,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise RuntimeError("Ollama 未返回完整 embedding")
            result.extend(embeddings)
        return result
