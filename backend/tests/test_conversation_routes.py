import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_app.api.routes import router


class FakeRepository:
    def __init__(self):
        self.documents = {
            "document-1": {
                "id": "document-1",
                "knowledge_base_id": "kb-1",
                "status": "processing",
                "progress": 35,
                "stage": "embedding",
            }
        }
        self.conversations = {
            "conversation-1": {
                "id": "conversation-1",
                "knowledge_base_id": "kb-1",
                "title": "原名称",
            }
        }

    def get_conversation(self, conversation_id, owner_id=None):
        conversation = self.conversations.get(conversation_id)
        if conversation and owner_id not in {None, "personal"}:
            return None
        return conversation

    def get_knowledge_base(self, knowledge_base_id, owner_id=None):
        if knowledge_base_id == "kb-1" and owner_id in {None, "personal"}:
            return {"id": knowledge_base_id}
        return None

    def get_document(self, document_id):
        return self.documents.get(document_id)

    def update_conversation_title(self, conversation_id, title):
        self.conversations[conversation_id]["title"] = title
        return self.conversations[conversation_id]

    def delete_conversation(self, conversation_id):
        self.conversations.pop(conversation_id, None)


class ConversationRoutesTest(unittest.TestCase):
    def setUp(self):
        self.repository = FakeRepository()
        app = FastAPI()
        app.state.startup_error = None
        parser = SimpleNamespace(
            supports=lambda file_name: file_name.endswith(".txt"),
            parse=lambda file_name, content: [SimpleNamespace(index=0, text=content.decode(), page_number=None)],
        )
        self.rag_calls = []
        self.ingestion_calls = []

        def answer(*args, **kwargs):
            self.rag_calls.append((args, kwargs))
            return {"answer": "附件回答", "citations": kwargs["attachment_citations"]}

        def enqueue_stream(*args, **kwargs):
            self.ingestion_calls.append((args, kwargs))
            return {"id": "document-1", "status": "processing", "progress": 10, "stage": "parsing"}

        app.state.services = SimpleNamespace(
            repository=self.repository,
            ingestion=SimpleNamespace(
                parser=parser,
                enqueue_stream=enqueue_stream,
                parse_stream=lambda file_name, stream, max_bytes: parser.parse(
                    file_name,
                    stream.read(max_bytes + 1),
                ),
            ),
            rag=SimpleNamespace(answer=answer),
            settings=SimpleNamespace(
                auth_username="admin",
                auth_password="test-password",
                auth_secret="test-secret",
                auth_owner_id="personal",
                auth_session_ttl_seconds=3600,
                auth_cookie_secure=False,
                rag_context_max_chars=12000,
                max_document_bytes=0,
            ),
        )
        app.include_router(router)
        self.client = TestClient(app)
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "test-password"},
        )
        self.assertEqual(response.status_code, 200)

    def test_renames_conversation(self):
        response = self.client.patch(
            "/api/v1/conversations/conversation-1",
            json={"title": "新名称"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "新名称")

    def test_upload_accepts_document_for_background_ingestion(self):
        response = self.client.post(
            "/api/v1/knowledge-bases/kb-1/documents",
            files={"file": ("notes.txt", b"background content", "text/plain")},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "processing")
        self.assertEqual(self.ingestion_calls[0][0][1], "notes.txt")

    def test_gets_background_ingestion_status(self):
        response = self.client.get(
            "/api/v1/knowledge-bases/kb-1/documents/document-1",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["stage"], "embedding")

    def test_does_not_return_document_from_another_knowledge_base(self):
        self.repository.documents["document-other"] = {
            "id": "document-other",
            "knowledge_base_id": "kb-other",
            "status": "ready",
        }

        response = self.client.get(
            "/api/v1/knowledge-bases/kb-1/documents/document-other",
        )

        self.assertEqual(response.status_code, 404)

    def test_deletes_conversation(self):
        response = self.client.delete("/api/v1/conversations/conversation-1")

        self.assertEqual(response.status_code, 204)
        self.assertIsNone(self.repository.get_conversation("conversation-1"))

    def test_returns_not_found_for_missing_conversation(self):
        response = self.client.delete("/api/v1/conversations/missing")

        self.assertEqual(response.status_code, 404)

    def test_rejects_blank_conversation_title(self):
        response = self.client.patch(
            "/api/v1/conversations/conversation-1",
            json={"title": "   "},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.repository.get_conversation("conversation-1")["title"], "原名称")

    def test_answers_with_temporary_attachment_without_ingesting_it(self):
        response = self.client.post(
            "/api/v1/knowledge-bases/kb-1/chat-with-attachments",
            data={"conversation_id": "conversation-1", "question": "总结附件"},
            files=[("files", ("notes.txt", b"temporary evidence", "text/plain"))],
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "附件回答")
        args, kwargs = self.rag_calls[0]
        self.assertIn("附件：notes.txt", args[2])
        self.assertIn("temporary evidence", kwargs["attachment_context"])
        self.assertTrue(kwargs["attachment_citations"][0]["temporary"])

    def test_parses_attachment_before_chat(self):
        response = self.client.post(
            "/api/v1/knowledge-bases/kb-1/chat-attachments/parse",
            files={"file": ("notes.txt", b"temporary evidence", "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "notes.txt")
        self.assertIn("temporary evidence", response.json()["context"])
        self.assertEqual(response.json()["chunk_count"], 1)
        self.assertEqual(self.rag_calls, [])

    def test_answers_with_preparsed_attachment(self):
        response = self.client.post(
            "/api/v1/knowledge-bases/kb-1/chat-with-parsed-attachments",
            json={
                "conversation_id": "conversation-1",
                "question": "总结附件",
                "model": None,
                "attachments": [{
                    "name": "notes.txt",
                    "context": "[临时附件] notes.txt\n[内容] temporary evidence",
                    "citations": [{"chunk_id": "attachment:0:0", "title": "notes.txt"}],
                }],
            },
        )

        self.assertEqual(response.status_code, 200)
        args, kwargs = self.rag_calls[0]
        self.assertIn("附件：notes.txt", args[2])
        self.assertIn("temporary evidence", kwargs["attachment_context"])
        self.assertTrue(kwargs["attachment_citations"][0]["temporary"])


if __name__ == "__main__":
    unittest.main()
