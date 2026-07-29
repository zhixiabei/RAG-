import unittest
from io import BytesIO

from rag_app.application.ingestion_service import DocumentTooLargeError, DuplicateDocumentError, IngestionService
from rag_app.domain.models import ParsedChunk


class FakeRepository:
    def __init__(self):
        self.document = None
        self.updates = []
        self.replaced_chunks = []
        self._existing_files = set()
        self._existing_hashes = set()
        self._missing_hash_documents = []

    def get_knowledge_base(self, knowledge_base_id):
        return {"id": knowledge_base_id, "embedding_model": "test-embedding"}

    def create_document(self, item):
        self.document = {**item, "progress": 0, "stage": "queued", "chunk_count": 0}
        self._existing_files.add((item["knowledge_base_id"], item["file_name"], item.get("folder_path") or ""))
        self._existing_hashes.add((item["knowledge_base_id"], item["content_hash"]))

    def update_document(self, document_id, **fields):
        self.updates.append(fields)
        if self.document and self.document["id"] == document_id:
            self.document.update(fields)
            return
        for document in self._missing_hash_documents:
            if document["id"] == document_id:
                document.update(fields)
                if content_hash := fields.get("content_hash"):
                    self._existing_hashes.add((document["knowledge_base_id"], content_hash))
                return

    def replace_chunks(self, document_id, knowledge_base_id, chunks, folder_path=""):
        self.replaced_chunks = chunks

    def get_document(self, document_id):
        return self.document

    def document_exists_by_file(self, knowledge_base_id, file_name, folder_path):
        return (knowledge_base_id, file_name, folder_path or "") in self._existing_files

    def document_exists_by_content_hash(self, knowledge_base_id, content_hash):
        return (knowledge_base_id, content_hash) in self._existing_hashes

    def list_documents_without_content_hash(self, knowledge_base_id):
        return [
            document
            for document in self._missing_hash_documents
            if document["knowledge_base_id"] == knowledge_base_id and not document.get("content_hash")
        ]


class FakeObjects:
    def __init__(self):
        self.items = []

    def put_bytes(self, object_key, content, content_type):
        self.items.append((object_key, content, content_type))

    def put_stream(self, object_key, stream, length, content_type):
        content = stream.read()
        if len(content) != length:
            raise AssertionError("stream length mismatch")
        self.items.append((object_key, content, content_type))

    def calculate_hash(self, object_key):
        from hashlib import sha256

        content = next(item[1] for item in self.items if item[0] == object_key)
        return sha256(content).hexdigest()


class FakeVectors:
    def __init__(self):
        self.points = []
        self.vectors = []
        self.upsert_batch_sizes = []
        self.deleted_documents = []

    def ensure_collection(self, vector_size):
        self.vector_size = vector_size

    def upsert(self, points, vectors):
        self.points.extend(points)
        self.vectors.extend(vectors)
        self.upsert_batch_sizes.append(len(points))

    def delete_document(self, document_id):
        self.deleted_documents.append(document_id)


class FakeParser:
    def __init__(self, failure=None, chunks=None):
        self.failure = failure
        self.chunks = chunks

    def supports(self, file_name):
        return True

    def parse(self, file_name, content):
        if self.failure:
            raise self.failure
        return self.chunks or [ParsedChunk(0, "知识内容", 1)]

    def parse_stream(self, file_name, stream):
        return self.parse(file_name, stream.read())


class FakeModels:
    embedding_model = "test-embedding"

    def __init__(self):
        self.embedded_texts = []
        self.embed_batch_sizes = []

    def embed(self, texts):
        self.embedded_texts.extend(texts)
        self.embed_batch_sizes.append(len(texts))
        return [[0.1, 0.2] for _ in texts]


class IngestionServiceTest(unittest.TestCase):
    def build_service(self, parser, **kwargs):
        self.repository = FakeRepository()
        return IngestionService(
            self.repository,
            FakeObjects(),
            FakeVectors(),
            parser,
            FakeModels(),
            **kwargs,
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

    def test_preserves_file_suffix_in_metadata_and_embedding_input(self):
        service = self.build_service(FakeParser())

        result = service.ingest(
            "kb-1",
            "长63渗透率.att",
            "application/octet-stream",
            b"content",
            "井资料\\长63/渗透率",
        )

        self.assertEqual(result["title"], "长63渗透率.att")
        self.assertEqual(service.vectors.points[0]["title"], "长63渗透率.att")
        self.assertEqual(service.vectors.points[0]["file_name"], "长63渗透率.att")
        self.assertEqual(service.vectors.points[0]["folder_path"], "井资料/长63/渗透率")
        self.assertEqual(
            service.vectors.points[0]["relative_path"],
            "井资料/长63/渗透率/长63渗透率.att",
        )
        self.assertIn("完整路径: 井资料/长63/渗透率/长63渗透率.att", service.models.embedded_texts[0])
        self.assertIn("长63渗透率.att", service.models.embedded_texts[0])
        self.assertIn("文件后缀: .att", service.models.embedded_texts[0])

    def test_batches_embeddings_and_vector_upserts(self):
        chunks = [ParsedChunk(index, f"知识内容 {index}", index + 1) for index in range(5)]
        service = self.build_service(FakeParser(chunks=chunks), embedding_batch_size=2)

        service.ingest("kb-1", "制度.pdf", "application/pdf", b"content")

        self.assertEqual(service.models.embed_batch_sizes, [2, 2, 1])
        self.assertEqual(service.vectors.upsert_batch_sizes, [2, 2, 1])
        self.assertEqual(len(service.vectors.points), 5)

    def test_skips_same_path_without_storing_or_parsing_again(self):
        service = self.build_service(FakeParser())
        self.repository._existing_files.add(("kb-1", "制度.pdf", "项目/制度"))

        with self.assertRaisesRegex(DuplicateDocumentError, "相同路径") as raised:
            service.ingest("kb-1", "制度.pdf", "application/pdf", b"content", "项目/制度")

        self.assertEqual(raised.exception.kind, "path")
        self.assertEqual(service.objects.items, [])
        self.assertIsNone(self.repository.document)

    def test_continues_ingesting_after_skipping_a_duplicate(self):
        service = self.build_service(FakeParser())
        self.repository._existing_files.add(("kb-1", "重复.pdf", "资料"))

        with self.assertRaises(DuplicateDocumentError):
            service.ingest("kb-1", "重复.pdf", "application/pdf", b"duplicate", "资料")
        result = service.ingest("kb-1", "新文件.pdf", "application/pdf", b"new content", "资料")

        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(service.objects.items), 1)

    def test_skips_same_content_at_a_different_path(self):
        service = self.build_service(FakeParser())

        first = service.ingest("kb-1", "制度.pdf", "application/pdf", b"same content", "旧目录")
        with self.assertRaisesRegex(DuplicateDocumentError, "内容相同") as raised:
            service.ingest("kb-1", "副本.pdf", "application/pdf", b"same content", "新目录")

        self.assertEqual(raised.exception.kind, "content")
        self.assertEqual(len(service.objects.items), 1)
        self.assertEqual(first["content_hash"], self.repository.document["content_hash"])

    def test_backfills_existing_hashes_without_holding_up_ingestion(self):
        service = self.build_service(FakeParser())
        object_key = "kb-1/legacy/source/制度.pdf"
        service.objects.items.append((object_key, b"legacy content", "application/pdf"))
        self.repository._missing_hash_documents.append({
            "id": "legacy",
            "knowledge_base_id": "kb-1",
            "source_object_key": object_key,
        })

        service._backfill_content_hashes("kb-1")

        self.assertIsNotNone(self.repository._missing_hash_documents[0].get("content_hash"))
        self.assertEqual(len(service.objects.items), 1)

    def test_rejects_oversized_stream_before_creating_document(self):
        service = self.build_service(FakeParser())

        with self.assertRaisesRegex(DocumentTooLargeError, "不能超过"):
            service.ingest_stream(
                "kb-1",
                "large.pdf",
                "application/pdf",
                BytesIO(b"12345"),
                max_bytes=4,
            )

        self.assertIsNone(self.repository.document)

    def test_zero_size_limit_accepts_document_stream(self):
        service = self.build_service(FakeParser())

        result = service.ingest_stream(
            "kb-1",
            "large.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            BytesIO(b"12345"),
            max_bytes=0,
        )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(service.objects.items[0][1], b"12345")


if __name__ == "__main__":
    unittest.main()
