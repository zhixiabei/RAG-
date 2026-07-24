import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_app.api.routes import router


class FakeRepository:
    def __init__(self):
        self.items = [
            {"id": "kb-personal", "owner_id": "personal", "name": "我的知识库"},
            {"id": "kb-other", "owner_id": "other", "name": "其他知识库"},
        ]

    def list_knowledge_bases(self, owner_id=None):
        return [item for item in self.items if owner_id is None or item["owner_id"] == owner_id]

    def get_knowledge_base(self, knowledge_base_id, owner_id=None):
        return next(
            (
                item
                for item in self.items
                if item["id"] == knowledge_base_id and (owner_id is None or item["owner_id"] == owner_id)
            ),
            None,
        )


class AuthRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.state.startup_error = None
        app.state.services = SimpleNamespace(
            repository=FakeRepository(),
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

    def login(self):
        return self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "test-password"},
        )

    def test_requires_login_for_business_api(self):
        response = self.client.get("/api/v1/knowledge-bases")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "请先登录")

    def test_rejects_invalid_credentials(self):
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong"},
        )

        self.assertEqual(response.status_code, 401)

    def test_session_lists_only_owned_knowledge_bases(self):
        self.assertEqual(self.login().status_code, 200)

        response = self.client.get("/api/v1/knowledge-bases")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()], ["kb-personal"])

    def test_other_owner_resource_is_hidden(self):
        self.login()

        response = self.client.get("/api/v1/knowledge-bases/kb-other")

        self.assertEqual(response.status_code, 404)

    def test_logout_invalidates_browser_session(self):
        self.login()
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 200)

        response = self.client.post("/api/v1/auth/logout")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 401)


if __name__ == "__main__":
    unittest.main()
