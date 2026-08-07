import json
import unittest

import httpx

from rag_app.domain.models import ParsedChunk
from rag_app.testset_tool import TestsetSyncService, TestsetToolClient, TestsetToolSyncError


class FakeRepository:
    def __init__(self):
        self.documents = [
            {
                "id": "doc-1",
                "knowledge_base_id": "kb-1",
                "file_name": "report.pdf",
                "mime_type": "application/pdf",
                "folder_path": "reports",
                "content_hash": "abc",
                "status": "ready",
            }
        ]
        self.chunks = {
            "doc-1": [
                {
                    "chunk_index": 0,
                    "text": "first chunk",
                    "page_number": 2,
                    "section_path": "summary",
                }
            ]
        }
        self.updates = []

    def update_document(self, document_id, **fields):
        self.updates.append((document_id, fields))

    def list_documents(self, knowledge_base_id):
        return [item for item in self.documents if item["knowledge_base_id"] == knowledge_base_id]

    def list_document_chunks(self, document_id):
        return self.chunks[document_id]


class TestsetToolClientTest(unittest.TestCase):
    def test_sends_the_exact_rag_document_and_chunk_ids(self):
        captured = {}

        def handle(request):
            captured["path"] = request.url.path
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True})

        client = TestsetToolClient(
            "http://testset.local",
            transport=httpx.MockTransport(handle),
        )
        try:
            result = client.sync_document(
                {
                    "document_id": "8f-document",
                    "knowledge_base_id": "kb-1",
                    "file_name": "report.pdf",
                    "mime_type": "application/pdf",
                    "folder_path": "reports",
                    "relative_path": "reports/report.pdf",
                    "content_hash": "abc",
                },
                [ParsedChunk(0, "first chunk", 2, "summary")],
            )
        finally:
            client.close()

        self.assertEqual(captured["path"], "/api/documents/import")
        self.assertEqual(captured["payload"]["documents"][0]["id"], "8f-document")
        self.assertEqual(captured["payload"]["chunks"][0]["id"], "8f-document:0")
        self.assertEqual(captured["payload"]["chunks"][0]["text"], "first chunk")
        self.assertEqual(result["chunk_count"], 1)

    def test_reports_testset_api_errors(self):
        client = TestsetToolClient(
            "http://testset.local",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    400,
                    json={"success": False, "error": {"message": "invalid chunk"}},
                )
            ),
        )
        try:
            with self.assertRaisesRegex(TestsetToolSyncError, "invalid chunk"):
                client.sync_document(
                    {
                        "document_id": "doc-1",
                        "knowledge_base_id": "kb-1",
                        "file_name": "report.pdf",
                    },
                    [ParsedChunk(0, "content")],
                )
        finally:
            client.close()

    def test_saves_a_question_to_the_workshop(self):
        captured = {}

        def handle(request):
            captured["path"] = request.url.path
            captured["payload"] = json.loads(request.content)
            return httpx.Response(201, json={"success": True, "issues": []})

        client = TestsetToolClient(
            "http://testset.local",
            transport=httpx.MockTransport(handle),
        )
        try:
            result = client.save_question({"id": "generated_0001", "question": "问题"})
        finally:
            client.close()

        self.assertEqual(captured["path"], "/api/questions")
        self.assertEqual(captured["payload"]["id"], "generated_0001")
        self.assertTrue(result["success"])


class TestsetSyncServiceTest(unittest.TestCase):
    def test_resyncs_all_ready_documents_from_persisted_chunks(self):
        repository = FakeRepository()
        captured = []

        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True})

        client = TestsetToolClient(
            "http://testset.local",
            transport=httpx.MockTransport(handle),
        )
        service = TestsetSyncService(repository, client)
        try:
            result = service.sync_knowledge_base("kb-1")
        finally:
            service.close()

        self.assertEqual(result["synced_document_count"], 1)
        self.assertEqual(result["synced_chunk_count"], 1)
        self.assertEqual(captured[0]["chunks"][0]["id"], "doc-1:0")
        self.assertEqual(repository.updates[-1][1]["testset_sync_status"], "synced")


if __name__ == "__main__":
    unittest.main()
