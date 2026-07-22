import httpx


class OllamaGateway:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def check_connection(self) -> None:
        response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        available = {
            item.get("name") or item.get("model")
            for item in response.json().get("models", [])
        }
        missing = [model for model in (self.chat_model, self.embedding_model) if model not in available]
        if missing:
            raise RuntimeError(f"缺少 Ollama 模型: {', '.join(missing)}")

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

    def answer(self, question: str, context: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.chat_model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": "你是知识库问答助手，只能依据上下文回答；上下文不足时明确说明，不得编造。"},
                    {"role": "user", "content": f"问题：{question}\n\n上下文：\n{context}"},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama 未返回答案")
        return answer
