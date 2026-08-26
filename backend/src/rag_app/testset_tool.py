from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .domain.models import ParsedChunk


class TestsetToolSyncError(RuntimeError):
    pass


# The workshop rejects requests above either limit. Keep a little headroom for
# JSON encoding and transport headers so large documents can always be split.
MAX_IMPORT_CHUNKS = 100
MAX_IMPORT_PAYLOAD_BYTES = 3_500_000


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value:
        return str(value)
    return datetime.now(timezone.utc).isoformat()


class TestsetToolClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 60.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def save_question(self, question: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post("/api/questions", json=question)
        except httpx.HTTPError as exc:
            raise TestsetToolSyncError(
                f"Cannot reach test-set tool at {self.base_url}: {exc}"
            ) from exc

        try:
            response_payload = response.json()
        except ValueError:
            response_payload = None
        if (
            not response.is_success
            or not isinstance(response_payload, dict)
            or response_payload.get("success") is not True
        ):
            detail = None
            if isinstance(response_payload, dict):
                error = response_payload.get("error")
                if isinstance(error, dict):
                    detail = error.get("message")
            raise TestsetToolSyncError(
                f"Test-set question import failed ({response.status_code}): "
                f"{detail or response.text or 'invalid response'}"
            )
        return response_payload

    def sync_document(self, document: dict[str, Any], chunks: list[ParsedChunk]) -> dict[str, Any]:
        document_id = str(document["document_id"])
        knowledge_base_id = str(document["knowledge_base_id"])
        file_name = str(document.get("file_name") or document.get("title") or document_id)
        relative_path = str(document.get("relative_path") or file_name)
        folder_path = str(document.get("folder_path") or "")
        suffix = Path(file_name).suffix.lower().lstrip(".")
        chunk_records = [
            {
                "id": f"{document_id}:{chunk.index}",
                "documentId": document_id,
                "documentName": file_name,
                "page": chunk.page_number,
                "section": chunk.section_path,
                "chunkIndex": chunk.index,
                "text": chunk.text,
                "metadata": {
                    "source": "rag",
                    "knowledgeBaseId": knowledge_base_id,
                    "folderPath": folder_path,
                    "relativePath": relative_path,
                },
            }
            for chunk in chunks
        ]
        document_record = {
            "id": document_id,
            "name": file_name,
            "format": suffix or "unknown",
            "sourcePath": relative_path,
            "importedAt": _iso_timestamp(document.get("created_at")),
            "chunkCount": len(chunk_records),
            "metadata": {
                "source": "rag",
                "knowledgeBaseId": knowledge_base_id,
                "mimeType": str(document.get("mime_type") or ""),
                "contentHash": str(document.get("content_hash") or ""),
                "folderPath": folder_path,
            },
        }

        for batch in self._chunk_batches(document_record, chunk_records):
            self._import_payload({"documents": [document_record], "chunks": batch})

        return {
            "document_id": document_id,
            "chunk_count": len(chunk_records),
        }

    @staticmethod
    def _chunk_batches(
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
    ):
        if not chunks:
            yield []
            return

        batch: list[dict[str, Any]] = []
        for chunk in chunks:
            candidate = batch + [chunk]
            payload = {"documents": [document], "chunks": candidate}
            payload_size = len(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            if batch and (
                len(candidate) > MAX_IMPORT_CHUNKS
                or payload_size > MAX_IMPORT_PAYLOAD_BYTES
            ):
                yield batch
                batch = [chunk]
                single_size = len(
                    json.dumps(
                        {"documents": [document], "chunks": batch},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                if single_size > MAX_IMPORT_PAYLOAD_BYTES:
                    raise TestsetToolSyncError(
                        f"Chunk {chunk.get('id', 'unknown')} cannot fit within the "
                        f"{MAX_IMPORT_PAYLOAD_BYTES} byte import limit"
                    )
            elif not batch and (
                len(candidate) > MAX_IMPORT_CHUNKS
                or payload_size > MAX_IMPORT_PAYLOAD_BYTES
            ):
                raise TestsetToolSyncError(
                    f"Chunk {chunk.get('id', 'unknown')} cannot fit within the "
                    f"{MAX_IMPORT_PAYLOAD_BYTES} byte import limit"
                )
            else:
                batch = candidate

        if batch:
            yield batch

    def _import_payload(self, payload: dict[str, Any]) -> None:
        try:
            response = self._client.post("/api/documents/import", json=payload)
        except httpx.HTTPError as exc:
            raise TestsetToolSyncError(
                f"Cannot reach test-set tool at {self.base_url}: {exc}"
            ) from exc

        try:
            response_payload = response.json()
        except ValueError:
            response_payload = None
        if not response.is_success or not isinstance(response_payload, dict) or response_payload.get("success") is not True:
            detail = None
            if isinstance(response_payload, dict):
                error = response_payload.get("error")
                if isinstance(error, dict):
                    detail = error.get("message")
            raise TestsetToolSyncError(
                f"Test-set tool import failed ({response.status_code}): {detail or response.text or 'invalid response'}"
            )


class TestsetSyncService:
    def __init__(self, repository, client: TestsetToolClient):
        self.repository = repository
        self.client = client

    def close(self) -> None:
        self.client.close()

    def sync_document(self, document: dict[str, Any], chunks: list[ParsedChunk]) -> dict[str, Any]:
        document_id = str(document["document_id"])
        self.repository.update_document(
            document_id,
            testset_sync_status="syncing",
            testset_sync_error=None,
        )
        try:
            result = self.client.sync_document(document, chunks)
        except Exception as exc:
            self.repository.update_document(
                document_id,
                testset_sync_status="failed",
                testset_sync_error=str(exc),
            )
            raise
        self.repository.update_document(
            document_id,
            testset_sync_status="synced",
            testset_sync_error=None,
            testset_synced_at=datetime.now(timezone.utc),
        )
        return result

    def sync_knowledge_base(self, knowledge_base_id: str) -> dict[str, Any]:
        synced_count = 0
        synced_chunk_count = 0
        failures = []
        skipped_count = 0
        for document in self.repository.list_documents(knowledge_base_id):
            if document.get("status") != "ready":
                skipped_count += 1
                continue
            rows = self.repository.list_document_chunks(document["id"])
            chunks = [
                ParsedChunk(
                    index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    page_number=row.get("page_number"),
                    section_path=row.get("section_path"),
                )
                for row in rows
            ]
            sync_document = {
                **document,
                "document_id": document["id"],
                "relative_path": (
                    f"{document.get('folder_path')}/{document['file_name']}"
                    if document.get("folder_path")
                    else document["file_name"]
                ),
            }
            try:
                result = self.sync_document(sync_document, chunks)
            except Exception as exc:
                failures.append({"document_id": document["id"], "error": str(exc)})
                continue
            synced_count += 1
            synced_chunk_count += int(result["chunk_count"])

        return {
            "knowledge_base_id": knowledge_base_id,
            "synced_document_count": synced_count,
            "synced_chunk_count": synced_chunk_count,
            "failed_document_count": len(failures),
            "skipped_document_count": skipped_count,
            "failures": failures,
        }
