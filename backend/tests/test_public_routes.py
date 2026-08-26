import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
            models=SimpleNamespace(),
            settings=SimpleNamespace(
                owner_id="personal",
                testset_tool_base_url="http://testset.local",
                evaluation_dataset_dir="testsets",
                evaluation_request_timeout_seconds=45.0,
                evaluation_max_concurrency=3,
            ),
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

    @patch("rag_app.api.routes.load_dataset_from_testset_tool")
    def test_lists_approved_evaluation_samples(self, load_samples):
        load_samples.return_value = (
            [{"question_id": "q1", "question": "问题一", "question_type": "numeric"}],
            {},
        )

        response = self.client.get("/api/v1/knowledge-bases/kb-personal/evaluation-samples")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["question_id"], "q1")
        load_samples.assert_called_once()

    @patch("rag_app.api.routes._load_evaluation_samples")
    def test_lists_local_evaluation_samples_when_selected(self, load_samples):
        load_samples.return_value = (
            [{"question_id": "local-1", "question": "local question"}],
            {},
        )

        response = self.client.get(
            "/api/v1/knowledge-bases/kb-personal/evaluation-samples?source=local&dataset=local.jsonl"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["question_id"], "local-1")
        load_samples.assert_called_once()

    @patch("rag_app.api.routes.run_evaluation")
    def test_runs_evaluation_for_selected_questions(self, run_evaluation):
        run_evaluation.return_value = {"summary": {"sample_count": 1}}

        response = self.client.post(
            "/api/v1/knowledge-bases/kb-personal/evaluation",
            json={"question_ids": ["q1"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"]["sample_count"], 1)
        self.assertEqual(run_evaluation.call_args.args[-1], ["q1"])
        self.assertEqual(run_evaluation.call_args.args[4], 45.0)
        self.assertEqual(run_evaluation.call_args.kwargs["max_concurrency"], 3)

    @patch("rag_app.api.routes.load_dataset_from_testset_tool")
    @patch("rag_app.api.routes._load_evaluation_samples")
    @patch("rag_app.api.routes._local_dataset_path")
    @patch("rag_app.api.routes.run_evaluation")
    def test_local_evaluation_does_not_use_testset_tool(
        self,
        run_evaluation,
        local_dataset_path,
        load_samples,
        load_from_testset_tool,
    ):
        dataset_path = Path("testsets/local.jsonl")
        local_dataset_path.return_value = dataset_path
        load_samples.return_value = ([{"question_id": "local-1", "question": "local"}], {})
        run_evaluation.return_value = {"summary": {"sample_count": 1}}

        response = self.client.post(
            "/api/v1/knowledge-bases/kb-personal/evaluation",
            json={
                "question_ids": ["local-1"],
                "dataset_source": "local",
                "dataset_id": "local.jsonl",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(run_evaluation.call_args.args[0], dataset_path)
        self.assertIsNone(run_evaluation.call_args.args[7])
        load_samples.assert_called_once_with(
            self.client.app.state.services.settings,
            "local",
            ["local-1"],
            "local.jsonl",
        )
        load_from_testset_tool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
