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


class PublicRoutesTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.state.startup_error = None
        app.state.services = SimpleNamespace(
            repository=FakeRepository(),
            settings=SimpleNamespace(owner_id="personal"),
        )
        app.include_router(router)
        self.client = TestClient(app)

    def test_business_api_is_available_without_login(self):
        response = self.client.get("/api/v1/knowledge-bases")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()], ["kb-personal"])

    def test_other_owner_resource_is_hidden(self):
        response = self.client.get("/api/v1/knowledge-bases/kb-other")

        self.assertEqual(response.status_code, 404)

    def test_auth_endpoints_do_not_exist(self):
        self.assertEqual(self.client.post("/api/v1/auth/login", json={}).status_code, 404)
        self.assertEqual(self.client.post("/api/v1/auth/logout").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 404)


if __name__ == "__main__":
    unittest.main()
