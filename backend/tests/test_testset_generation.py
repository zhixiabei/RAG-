import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import httpx

from rag_app.testset_generation import (
    TestsetGenerationError,
    TestsetGeneratorClient,
    build_generator,
    generate_testset,
    parse_generated_question,
    to_workshop_question,
)
from rag_app.evaluation import load_dataset


class FakeRepository:
    def __init__(self):
        self.knowledge_base = {"id": "kb-1"}
        self.documents = [
            {
                "id": "doc-1",
                "knowledge_base_id": "kb-1",
                "file_name": "report.pdf",
                "status": "ready",
            },
            {
                "id": "doc-processing",
                "knowledge_base_id": "kb-1",
                "file_name": "pending.pdf",
                "status": "processing",
            },
        ]
        self.chunks = {
            "doc-1": [
                {
                    "id": "doc-1:0",
                    "chunk_index": 0,
                    "text": "长六储层属于低孔、特低渗储层，主要储集空间为粒间孔。",
                    "page_number": 3,
                    "section_path": "储层特征",
                },
                {
                    "id": "doc-1:1",
                    "chunk_index": 1,
                    "text": "短内容",
                    "page_number": 4,
                    "section_path": "附注",
                },
            ],
            "doc-processing": [],
        }

    def get_knowledge_base(self, knowledge_base_id):
        return self.knowledge_base if knowledge_base_id == "kb-1" else None

    def list_documents(self, knowledge_base_id):
        return [item for item in self.documents if item["knowledge_base_id"] == knowledge_base_id]

    def list_document_chunks(self, document_id):
        return self.chunks[document_id]


class FakeGenerator:
    model = "independent-generator-model"

    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        target_difficulty = json.loads(messages[-1]["content"])["target_difficulty"]
        return json.dumps({
            "question": "长六储层的孔渗特征、主要储集空间及其相互关系是什么？",
            "expected_answer": (
                "长六储层属于低孔、特低渗储层，主要储集空间为粒间孔。"
                "这些特征共同说明储层物性较差，储集能力受到孔隙度和渗透率的双重限制。"
            ),
            "difficulty": target_difficulty,
        })


class TestsetGenerationTest(unittest.TestCase):
    def test_generates_retrieval_samples_with_exact_source_ids(self):
        generator = FakeGenerator()

        samples = generate_testset(
            FakeRepository(),
            generator,
            "kb-1",
            questions_per_document=2,
            min_chunk_chars=20,
            status="draft",
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["source_document_ids"], ["doc-1"])
        self.assertEqual(samples[0]["source_chunk_ids"], ["doc-1:0"])
        self.assertEqual(samples[0]["source_pages"], [3])
        self.assertEqual(samples[0]["status"], "draft")
        self.assertEqual(samples[0]["created_by"], "ai")
        self.assertIn("expected_answer", samples[0])
        self.assertEqual(len(generator.calls), 1)

    def test_reports_progress_before_each_model_request(self):
        events = []

        generate_testset(
            FakeRepository(),
            FakeGenerator(),
            "kb-1",
            questions_per_document=1,
            min_chunk_chars=20,
            on_progress=lambda index, total, document, chunks, difficulty: events.append(
                (index, total, document["id"], chunks[0]["id"], difficulty)
            ),
        )

        self.assertEqual(events, [(1, 1, "doc-1", "doc-1:0", "medium")])

    def test_hard_questions_use_three_chunks_and_workshop_evidence(self):
        repository = FakeRepository()
        repository.chunks["doc-1"] = [
            {
                "id": f"doc-1:{index}",
                "chunk_index": index,
                "text": (
                    f"第 {index + 1} 部分描述储层条件、治理措施及实施限制，"
                    "包含足够长的技术信息用于构造综合分析问题。"
                ),
                "page_number": index + 1,
                "section_path": "综合分析",
            }
            for index in range(3)
        ]

        sample = generate_testset(
            repository,
            FakeGenerator(),
            "kb-1",
            questions_per_document=1,
            min_chunk_chars=20,
            difficulty_profile="hard",
            max_source_chunks=3,
        )[0]
        payload = to_workshop_question(sample)

        self.assertEqual(sample["difficulty"], "hard")
        self.assertEqual(sample["question_type"], "multi_chunk")
        self.assertTrue(sample["requires_multiple_chunks"])
        self.assertEqual(len(sample["source_chunk_ids"]), 3)
        self.assertEqual(len(payload["evidence"]), 3)

    def test_rejects_non_json_generator_output(self):
        with self.assertRaisesRegex(TestsetGenerationError, "合法 JSON"):
            parse_generated_question("not json")

    def test_approved_output_can_be_loaded_by_the_evaluator(self):
        samples = generate_testset(
            FakeRepository(),
            FakeGenerator(),
            "kb-1",
            questions_per_document=1,
            min_chunk_chars=20,
            status="approved",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.jsonl"
            path.write_text(
                "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
                encoding="utf-8",
            )
            loaded = load_dataset(path)

        self.assertEqual(loaded[0]["source_chunk_ids"], ["doc-1:0"])

    def test_maps_a_sample_to_the_workshop_question_contract(self):
        sample = generate_testset(
            FakeRepository(),
            FakeGenerator(),
            "kb-1",
            questions_per_document=1,
            min_chunk_chars=20,
            status="draft",
        )[0]

        payload = to_workshop_question(sample)

        self.assertEqual(payload["id"], sample["question_id"])
        self.assertEqual(payload["expectedAnswer"], sample["expected_answer"])
        self.assertEqual(payload["createdBy"], "ai")
        self.assertEqual(payload["evidence"], [
            {"chunkId": "doc-1:0", "position": 0, "isPrimary": True},
        ])

    def test_dedicated_client_sends_only_its_configured_model(self):
        captured = {}

        def handle(request):
            captured["path"] = request.url.path
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"question":"测试问题是什么？","difficulty":"easy"}'}}
                    ]
                },
            )

        client = TestsetGeneratorClient(
            "Independent",
            "http://generator.local/v1",
            "generator-key",
            "generator-only-model",
            transport=httpx.MockTransport(handle),
        )
        try:
            output = client.complete([{"role": "user", "content": "source"}])
        finally:
            client.close()

        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "generator-only-model")
        self.assertIn("测试问题", output)

    def test_rejects_the_active_rag_model_as_generator(self):
        settings = SimpleNamespace(
            model_mode="remote",
            ollama_chat_model="local-model",
            remote_default_chat_model="rag-model",
            testset_generator_provider_name="Independent",
            testset_generator_base_url="http://generator.local/v1",
            testset_generator_api_key="key",
            testset_generator_model="rag-model",
        )

        with self.assertRaisesRegex(TestsetGenerationError, "不能与当前 RAG 模型相同"):
            build_generator(settings)

    def test_build_generator_uses_cli_timeout_override(self):
        settings = SimpleNamespace(
            model_mode="remote",
            ollama_chat_model="local-model",
            remote_default_chat_model="rag-model",
            testset_generator_provider_name="Independent",
            testset_generator_base_url="http://generator.local/v1",
            testset_generator_api_key="key",
            testset_generator_model="generator-model",
            testset_generator_timeout_seconds=90.0,
        )

        client = build_generator(settings, timeout_seconds=12.0)
        try:
            self.assertEqual(client.timeout_seconds, 12.0)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
