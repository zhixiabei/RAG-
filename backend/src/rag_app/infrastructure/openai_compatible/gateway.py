import time

import httpx

from agent.telemetry import record_model_usage


_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 3
_RETRY_BASE_DELAY_SECONDS = 0.5
_RETRY_MAX_DELAY_SECONDS = 8.0


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

    def check_connection(
        self,
        require_chat_model: bool = True,
        require_embedding_model: bool = True,
    ) -> None:
        missing = []
        if require_chat_model and (not self.api_key or not self.base_url or not self.models):
            missing.append("远程聊天 API")
        if require_chat_model and self.chat_model not in self.models:
            missing.append(f"默认聊天模型 {self.chat_model}")
        if (
            require_embedding_model
            and (
                not self.embedding_api_key
                or not self.embedding_base_url
                or not self.remote_embedding_model
            )
        ):
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

    @staticmethod
    def _post_json(url: str, api_key: str, payload: dict, operation: str) -> httpx.Response:
        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                response = httpx.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=180,
                )
            except httpx.TimeoutException as exc:
                raise RuntimeError(f"{operation}超时（180 秒）") from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"{operation}连接失败: {exc}") from exc

            if response.status_code not in _TRANSIENT_STATUS_CODES or attempt >= _MAX_TRANSIENT_RETRIES:
                return response

            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = _RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            time.sleep(min(max(delay, 0.0), _RETRY_MAX_DELAY_SECONDS))

        raise AssertionError("unreachable")

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        reasoning: bool | None = None,
        response_schema: dict | None = None,
    ) -> str:
        selected_model = model or self.chat_model
        if selected_model not in self.models:
            raise RuntimeError(f"远程聊天模型未配置: {selected_model}")
        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        response = self._post_json(
            f"{self.base_url}/chat/completions",
            self.api_key,
            payload,
            f"{self.provider_name} 聊天接口",
        )
        if response_schema is not None and response.status_code in {400, 422}:
            # Some OpenAI-compatible providers do not implement response_format.
            payload.pop("response_format", None)
            response = self._post_json(
                f"{self.base_url}/chat/completions",
                self.api_key,
                payload,
                f"{self.provider_name} 聊天接口",
            )
        response.raise_for_status()
        response_payload = response.json()
        choices = response_payload.get("choices") or []
        output = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        if not output:
            raise RuntimeError(f"{self.provider_name} 未返回文本")
        usage = response_payload.get("usage") or {}
        record_model_usage(
            operation="completion",
            provider=self.provider_name,
            model=selected_model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        return output

    def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = self._post_json(
                f"{self.embedding_base_url}/embeddings",
                self.embedding_api_key,
                {"model": self.remote_embedding_model, "input": batch},
                f"{self.embedding_provider_name or '远程 embedding'} 接口",
            )
            response.raise_for_status()
            response_payload = response.json()
            data = sorted(response_payload.get("data") or [], key=lambda item: item.get("index", 0))
            embeddings = [item.get("embedding") for item in data]
            if len(embeddings) != len(batch) or any(not embedding for embedding in embeddings):
                raise RuntimeError(f"{self.embedding_provider_name} 未返回完整 embedding")
            usage = response_payload.get("usage") or {}
            record_model_usage(
                operation="embedding",
                provider=self.embedding_provider_name or "远程 embedding",
                model=self.remote_embedding_model,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=0 if usage else None,
                total_tokens=usage.get("total_tokens"),
            )
            result.extend(embeddings)
        return result
