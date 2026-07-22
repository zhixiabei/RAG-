import httpx


class OpenAICompatibleGateway:
    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str,
        models: list[str],
        default_chat_model: str,
        embedding_provider_name: str,
        embedding_base_url: str,
        embedding_api_key: str,
        embedding_model: str,
    ):
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.models = models
        self.chat_model = default_chat_model
        self.embedding_provider_name = embedding_provider_name
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_api_key = embedding_api_key
        self.remote_embedding_model = embedding_model
        self.embedding_model = f"{embedding_provider_name}::{embedding_model}"

    def check_connection(self) -> None:
        missing = []
        if not self.api_key or not self.base_url or not self.models:
            missing.append("远程聊天 API")
        if self.chat_model not in self.models:
            missing.append(f"默认聊天模型 {self.chat_model}")
        if not self.embedding_api_key or not self.embedding_base_url or not self.remote_embedding_model:
            missing.append("远程 embedding API")
        if missing:
            raise RuntimeError(f"远程模型配置不完整: {', '.join(missing)}")

    def list_chat_models(self) -> list[dict]:
        return [
            {
                "id": model,
                "name": model,
                "provider": self.provider_name,
                "is_default": model == self.chat_model,
            }
            for model in self.models
        ]

    def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = httpx.post(
                f"{self.embedding_base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.embedding_api_key}", "Content-Type": "application/json"},
                json={"model": self.remote_embedding_model, "input": batch},
                timeout=180,
            )
            response.raise_for_status()
            data = sorted(response.json().get("data") or [], key=lambda item: item.get("index", 0))
            embeddings = [item.get("embedding") for item in data]
            if len(embeddings) != len(batch) or any(not embedding for embedding in embeddings):
                raise RuntimeError(f"{self.embedding_provider_name} 未返回完整 embedding")
            result.extend(embeddings)
        return result

    def answer(self, question: str, context: str, history: list[dict], model: str | None = None) -> str:
        model = model or self.chat_model
        if model not in self.models:
            raise RuntimeError(f"远程聊天模型未配置: {model}")
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
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "stream": False, "temperature": 0.1},
            timeout=180,
        )
        response.raise_for_status()
        choices = response.json().get("choices") or []
        answer = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        if not answer:
            raise RuntimeError(f"{self.provider_name} 未返回答案")
        return answer
