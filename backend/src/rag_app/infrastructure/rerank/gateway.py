from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import httpx

from agent.telemetry import record_model_usage

from ...domain.models import SearchHit


class HttpReranker:
    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_document_chars: int = 4_000,
        transport: httpx.BaseTransport | None = None,
    ):
        if not base_url.strip():
            raise ValueError("Reranker base URL is required")
        if not model.strip():
            raise ValueError("Reranker model is required")
        self.provider_name = provider_name.strip() or "HTTP reranker"
        self.model = model.strip()
        self.name = self.model
        self.max_document_chars = max(1, max_document_chars)
        endpoint = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.endswith("/rerank") else f"{endpoint}/rerank"
        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        self.client = httpx.Client(
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        candidates = list(hits)
        if not candidates or limit <= 0:
            return []
        response = self.client.post(
            self.endpoint,
            json={
                "model": self.model,
                "query": query,
                "documents": [self._document(hit) for hit in candidates],
                "top_n": min(limit, len(candidates)),
                "return_documents": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results:
            raise RuntimeError("Reranker response is missing results")

        scored: list[tuple[float, int]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item["index"])
                score = float(item.get("relevance_score", item.get("score")))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(candidates):
                scored.append((score, index))
        if not scored:
            raise RuntimeError("Reranker response contains no valid scores")

        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [
            replace(candidates[index], relevance_score=score)
            for score, index in scored[:limit]
        ]
        record_model_usage(
            operation="rerank",
            provider=self.provider_name,
            model=self.model,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
        return ranked

    def close(self) -> None:
        self.client.close()

    def _document(self, hit: SearchHit) -> str:
        metadata = [f"Title: {hit.title}"]
        if hit.relative_path:
            metadata.append(f"Path: {hit.relative_path}")
        if hit.page_number is not None:
            metadata.append(f"Page: {hit.page_number}")
        document = "\n".join([*metadata, "", hit.text])
        return document[:self.max_document_chars]
