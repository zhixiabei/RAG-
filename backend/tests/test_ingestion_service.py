import unittest

from rag_app.application.ingestion_service import IngestionService
from rag_app.domain.models import ParsedChunk


class FakeRepository:
    def __init__(self):
        self.document = None
        self.updates = []
        self.replaced_chunks = []

    def get_knowledge_base(self, knowledge_base_id):
        return {"id": knowledge_base_id, "embedding_model": "test-embedding"}

    def create_document(self, item):
        self.document = {**item, "progress": 0, "stage": "queued", "chunk_count": 0}

    def update_document(self, document_id, **fields):
        self.updates.append(fields)
        self.document.update(fields)

    def replace_chunks(self, document_id, knowledge_base_id, chunks):
        self.replaced_chunks = chunks

    def get_document(self, document_id):
        return self.document


class FakeObjects:
    def __init__(self):
        self.items = []

    def put_bytes(self, object_key, content, content_type):
        self.items.append((object_key, content, content_type))


class FakeVectors:
    def ensure_collection(self, vector_size):
        self.vector_size = vector_size

    def upsert(self, points, vectors):
        self.points = points
        self.vectors = vectors


class FakeParser:
    def __init__(self, failure=None):
        self.failure = failure

    def supports(self, file_name):
        return True

    def parse(self, file_name, content):
        if self.failure:
            raise self.failure
        return [ParsedChunk(0, "知识内容", 1)]


class FakeModels:
    embedding_model = "test-embedding"

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class IngestionServiceTest(unittest.TestCase):
    def build_service(self, parser):
        self.repository = FakeRepository()
        return IngestionService(
            self.repository,
            FakeObjects(),
            FakeVectors(),
            parser,
            FakeModels(),
        )

    def test_persists_progress_through_each_ingestion_stage(self):
        service = self.build_service(FakeParser())

        result = service.ingest("kb-1", "制度.pdf", "application/pdf", b"content")

        self.assertEqual(
            [(update.get("progress"), update.get("stage")) for update in self.repository.updates],
            [(10, "parsing"), (35, "embedding"), (70, "indexing"), (90, "saving"), (100, "ready")],
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["progress"], 100)
        self.assertEqual(result["chunk_count"], 1)

    def test_marks_document_failed_without_losing_last_progress(self):
        service = self.build_service(FakeParser(RuntimeError("parse failed")))

        with self.assertRaisesRegex(RuntimeError, "parse failed"):
            service.ingest("kb-1", "broken.pdf", "application/pdf", b"content")

        self.assertEqual(self.repository.document["status"], "failed")
        self.assertEqual(self.repository.document["stage"], "failed")
        self.assertEqual(self.repository.document["progress"], 10)


if __name__ == "__main__":
    unittest.main()
