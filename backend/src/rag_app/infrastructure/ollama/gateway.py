import httpx


class OllamaGateway:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def check_connection(self, require_chat_model: bool = True) -> None:
        response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        available = {
            item.get("name") or item.get("model")
            for item in response.json().get("models", [])
        }
        required = [self.embedding_model]
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

    def answer(self, question: str, context: str, history: list[dict], model: str | None = None) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是知识库问答助手，只能依据检索上下文和此前对话中已有的知识库信息回答；信息不足时明确说明，不得编造。",
            }
        ]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )
        messages.append({"role": "user", "content": f"当前问题：{question}\n\n本轮检索上下文：\n{context}"})
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.chat_model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": messages,
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama 未返回答案")
        return answer
