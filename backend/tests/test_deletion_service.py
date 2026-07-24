import unittest

from rag_app.application.deletion_service import DeletionService


class FakeRepository:
    def __init__(self):
        self.knowledge_bases = {"kb-1": {"id": "kb-1"}}
        self.documents = {
            "doc-1": {
                "id": "doc-1",
                "knowledge_base_id": "kb-1",
                "source_object_key": "kb-1/doc-1/source/test.pdf",
            },
            "doc-2": {
                "id": "doc-2",
                "knowledge_base_id": "kb-1",
                "source_object_key": "kb-1/doc-2/source/other.pdf",
            },
        }
        self.deleted_documents = []
        self.deleted_knowledge_bases = []

    def get_document(self, document_id):
        return self.documents.get(document_id)

    def get_knowledge_base(self, knowledge_base_id):
        return self.knowledge_bases.get(knowledge_base_id)

    def list_documents(self, knowledge_base_id):
        return [item for item in self.documents.values() if item["knowledge_base_id"] == knowledge_base_id]

    def delete_document(self, document_id):
        self.deleted_documents.append(document_id)

    def delete_knowledge_base(self, knowledge_base_id):
        self.deleted_knowledge_bases.append(knowledge_base_id)


class FakeObjectStore:
    def __init__(self):
        self.deleted = []
        self.failure = None

    def delete_object(self, object_key):
        if self.failure:
            raise self.failure
        self.deleted.append(object_key)


class FakeVectorStore:
    def __init__(self):
        self.deleted_documents = []
        self.deleted_knowledge_bases = []

    def delete_document(self, document_id):
        self.deleted_documents.append(document_id)

    def delete_knowledge_base(self, knowledge_base_id):
        self.deleted_knowledge_bases.append(knowledge_base_id)


class DeletionServiceTest(unittest.TestCase):
    def setUp(self):
        self.repository = FakeRepository()
        self.objects = FakeObjectStore()
        self.vectors = FakeVectorStore()
        self.service = DeletionService(self.repository, self.objects, self.vectors)

    def test_deletes_document_from_all_stores(self):
        deleted = self.service.delete_document("kb-1", "doc-1")

        self.assertTrue(deleted)
        self.assertEqual(self.vectors.deleted_documents, ["doc-1"])
        self.assertEqual(self.objects.deleted, ["kb-1/doc-1/source/test.pdf"])
        self.assertEqual(self.repository.deleted_documents, ["doc-1"])

    def test_rejects_document_from_another_knowledge_base(self):
        deleted = self.service.delete_document("kb-other", "doc-1")

        self.assertFalse(deleted)
        self.assertEqual(self.vectors.deleted_documents, [])
        self.assertEqual(self.objects.deleted, [])
        self.assertEqual(self.repository.deleted_documents, [])

    def test_keeps_database_record_when_external_cleanup_fails(self):
        self.objects.failure = RuntimeError("MinIO unavailable")

        with self.assertRaisesRegex(RuntimeError, "MinIO unavailable"):
            self.service.delete_document("kb-1", "doc-1")

        self.assertEqual(self.repository.deleted_documents, [])

    def test_deletes_knowledge_base_and_all_source_objects(self):
        deleted = self.service.delete_knowledge_base("kb-1")

        self.assertTrue(deleted)
        self.assertEqual(self.vectors.deleted_knowledge_bases, ["kb-1"])
        self.assertEqual(
            self.objects.deleted,
            ["kb-1/doc-1/source/test.pdf", "kb-1/doc-2/source/other.pdf"],
        )
        self.assertEqual(self.repository.deleted_knowledge_bases, ["kb-1"])


if __name__ == "__main__":
    unittest.main()
