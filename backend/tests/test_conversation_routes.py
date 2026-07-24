import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_app.api.routes import router


class FakeRepository:
    def __init__(self):
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
        app.state.services = SimpleNamespace(
            repository=self.repository,
            settings=SimpleNamespace(
                auth_username="admin",
                auth_password="test-password",
                auth_secret="test-secret",
                auth_owner_id="personal",
                auth_session_ttl_seconds=3600,
                auth_cookie_secure=False,
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


if __name__ == "__main__":
    unittest.main()
