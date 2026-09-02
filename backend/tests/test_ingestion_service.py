import unittest
from io import BytesIO
from threading import Event

from rag_app.application.deletion_service import DeletionService
from rag_app.application.ingestion_service import DocumentTooLargeError, DuplicateDocumentError, IngestionService
from rag_app.domain.models import ParsedChunk


class FakeRepository:
    def __init__(self):
        self.document = None
        self.updates = []
        self.replaced_chunks = []
        self._existing_hashes = set()
        self._missing_hash_documents = []

    def get_knowledge_base(self, knowledge_base_id):
        return {"id": knowledge_base_id, "embedding_model": "test-embedding"}

    def create_document(self, item):
        self.document = {**item, "progress": 0, "stage": "queued", "chunk_count": 0}
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

    def delete_document(self, document_id):
        if self.document and self.document["id"] == document_id:
            self.document = None

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

    def open_stream(self, object_key):
        content = next(item[1] for item in self.items if item[0] == object_key)
        return BytesIO(content)

    def delete_object(self, object_key):
        self.items = [item for item in self.items if item[0] != object_key]


class FakeVectors:
    def __init__(self):
        self.points = []
        self.vectors = []
        self.document_points = []
        self.document_vectors = []
        self.upsert_batch_sizes = []
        self.deleted_documents = []

    def ensure_collection(self, vector_size):
        self.vector_size = vector_size

    def upsert(self, points, vectors):
        self.points.extend(points)
        self.vectors.extend(vectors)
        self.upsert_batch_sizes.append(len(points))

    def upsert_documents(self, points, vectors):
        self.document_points.extend(points)
        self.document_vectors.extend(vectors)

    def delete_document(self, document_id):
        self.deleted_documents.append(document_id)
        self.points = [point for point in self.points if point["document_id"] != document_id]
        self.document_points = [
            point for point in self.document_points
            if point["document_id"] != document_id
        ]


class BlockingVectors(FakeVectors):
    def __init__(self):
        super().__init__()
        self.upsert_started = Event()
        self.upsert_release = Event()

    def upsert(self, points, vectors):
        self.upsert_started.set()
        self.upsert_release.wait(timeout=2)
        super().upsert(points, vectors)


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


class BlockingParser(FakeParser):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.release = Event()

    def parse_stream(self, file_name, stream):
        self.started.set()
        self.release.wait(timeout=2)
        return super().parse_stream(file_name, stream)


class FakeModels:
    embedding_model = "test-embedding"

    def __init__(self):
        self.embedded_texts = []
        self.embed_batch_sizes = []

    def embed(self, texts):
        self.embedded_texts.extend(texts)
        self.embed_batch_sizes.append(len(texts))
        return [[0.1, 0.2] for _ in texts]

    def complete(self, messages, **kwargs):
        return '{"summary":"","topics":[]}'


class FakeTestsetSync:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    def sync_document(self, document, chunks):
        self.calls.append((document, chunks))
        if self.failure:
            raise self.failure
        return {"document_id": document["document_id"], "chunk_count": len(chunks)}


class IngestionServiceTest(unittest.TestCase):
    def build_service(self, parser, *, vectors=None, models=None, **kwargs):
        self.repository = FakeRepository()
        return IngestionService(
            self.repository,
            FakeObjects(),
            vectors or FakeVectors(),
            parser,
            models or FakeModels(),
            **kwargs,
        )

    def test_persists_progress_through_each_ingestion_stage(self):
        service = self.build_service(FakeParser())

        result = service.ingest("kb-1", "制度.pdf", "application/pdf", b"content")

        self.assertEqual(
            [(update.get("progress"), update.get("stage")) for update in self.repository.updates],
            [(10, "parsing"), (30, "profiling"), (40, "embedding"), (70, "indexing"), (90, "saving"), (100, "ready")],
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
        chunk_embedding = next(
            text for text in service.models.embedded_texts
            if "文件后缀: .att" in text
        )
        self.assertIn("文件后缀: .att", chunk_embedding)

    def test_embeds_and_indexes_the_chunk_section_path(self):
        chunks = [ParsedChunk(0, "安装步骤", 2, "安装/Windows")]
        service = self.build_service(FakeParser(chunks=chunks))

        service.ingest(
            "kb-1",
            "指南.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            b"content",
        )

        chunk_embedding = next(
            text for text in service.models.embedded_texts
            if "章节路径: 安装/Windows" in text
        )
        self.assertIn("章节路径: 安装/Windows", chunk_embedding)
        self.assertEqual(service.vectors.points[0]["section_path"], "安装/Windows")
        self.assertEqual(service.vectors.points[0]["chunk_index"], 0)

    def test_batches_embeddings_and_vector_upserts(self):
        chunks = [ParsedChunk(index, f"知识内容 {index}", index + 1) for index in range(5)]
        service = self.build_service(FakeParser(chunks=chunks), embedding_batch_size=2)

        service.ingest("kb-1", "制度.pdf", "application/pdf", b"content")

        self.assertEqual(service.models.embed_batch_sizes, [3, 2, 2, 1])
        self.assertEqual(service.vectors.upsert_batch_sizes, [2, 2, 1])
        self.assertEqual(len(service.vectors.points), 5)

    def test_syncs_the_same_parsed_chunks_after_indexing(self):
        sync = FakeTestsetSync()
        chunks = [ParsedChunk(0, "first", 1), ParsedChunk(1, "second", 2)]
        service = self.build_service(FakeParser(chunks=chunks), testset_sync=sync)

        result = service.ingest("kb-1", "report.pdf", "application/pdf", b"content")

        self.assertEqual(result["status"], "ready")
        self.assertEqual(sync.calls[0][0]["document_id"], result["id"])
        self.assertIs(sync.calls[0][1], chunks)
        self.assertIn((95, "syncing_testset"), [
            (update.get("progress"), update.get("stage"))
            for update in self.repository.updates
        ])

    def test_sync_failure_does_not_discard_a_ready_rag_document(self):
        sync = FakeTestsetSync(RuntimeError("test-set tool unavailable"))
        service = self.build_service(FakeParser(), testset_sync=sync)

        with self.assertLogs("rag_app.application.ingestion_service", level="ERROR"):
            result = service.ingest("kb-1", "report.pdf", "application/pdf", b"content")

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(len(service.vectors.points), 1)

    def test_allows_same_name_and_path_when_content_is_different(self):
        service = self.build_service(FakeParser())

        first = service.ingest("kb-1", "report.pdf", "application/pdf", b"old content", "project")
        second = service.ingest("kb-1", "report.pdf", "application/pdf", b"new content", "project")

        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(second["status"], "ready")
        self.assertEqual([item[1] for item in service.objects.items], [b"old content", b"new content"])


    def test_continues_ingesting_after_skipping_same_content(self):
        service = self.build_service(FakeParser())
        service.ingest("kb-1", "original.pdf", "application/pdf", b"duplicate", "documents")

        with self.assertRaises(DuplicateDocumentError):
            service.ingest("kb-1", "renamed.pdf", "application/pdf", b"duplicate", "documents")
        result = service.ingest("kb-1", "new.pdf", "application/pdf", b"new content", "documents")

        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(service.objects.items), 2)


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

    def test_enqueue_returns_before_background_parsing_finishes(self):
        parser = BlockingParser()
        service = self.build_service(parser, max_concurrency=1)

        result = service.enqueue_stream(
            "kb-1",
            "large.pdf",
            "application/pdf",
            BytesIO(b"content"),
        )

        self.assertEqual(result["status"], "processing")
        self.assertEqual(result["progress"], 10)
        self.assertTrue(parser.started.wait(timeout=1))
        self.assertEqual(self.repository.document["status"], "processing")
        self.assertTrue(service.is_pending(result["id"]))

        parser.release.set()
        service.wait_for_pending()
        self.assertEqual(self.repository.document["status"], "ready")
        self.assertFalse(service.is_pending(result["id"]))

    def test_delete_cancels_background_ingestion_and_cleans_external_data(self):
        parser = BlockingParser()
        service = self.build_service(parser, max_concurrency=1)
        deletion = DeletionService(service.repository, service.objects, service.vectors, service.cancel)

        result = service.enqueue_stream(
            "kb-1",
            "cancelled.pdf",
            "application/pdf",
            BytesIO(b"content"),
        )
        self.assertTrue(parser.started.wait(timeout=1))

        self.assertTrue(deletion.delete_document("kb-1", result["id"]))
        self.assertIsNone(self.repository.document)
        parser.release.set()
        service.wait_for_pending()

        self.assertFalse(service.is_pending(result["id"]))
        self.assertIsNone(self.repository.document)
        self.assertEqual(service.vectors.deleted_documents, [result["id"]])
        self.assertEqual(service.objects.items, [])

    def test_delete_during_upsert_removes_vectors_written_after_cancellation(self):
        vectors = BlockingVectors()
        service = self.build_service(FakeParser(), vectors=vectors, max_concurrency=1)
        deletion = DeletionService(service.repository, service.objects, service.vectors, service.cancel)

        result = service.enqueue_stream(
            "kb-1",
            "upserting.pdf",
            "application/pdf",
            BytesIO(b"content"),
        )
        self.assertTrue(vectors.upsert_started.wait(timeout=1))

        self.assertTrue(deletion.delete_document("kb-1", result["id"]))
        vectors.upsert_release.set()
        service.wait_for_pending()

        self.assertIsNone(self.repository.document)
        self.assertEqual(vectors.points, [])
        self.assertEqual(vectors.deleted_documents, [result["id"]])


if __name__ == "__main__":
    unittest.main()
