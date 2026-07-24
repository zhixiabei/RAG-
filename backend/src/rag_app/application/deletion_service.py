from ..domain.ports import MetadataRepository, ObjectStore, VectorStore


class DeletionService:
    def __init__(self, repository: MetadataRepository, objects: ObjectStore, vectors: VectorStore):
        self.repository = repository
        self.objects = objects
        self.vectors = vectors

    def delete_document(self, knowledge_base_id: str, document_id: str) -> bool:
        document = self.repository.get_document(document_id)
        if not document or document["knowledge_base_id"] != knowledge_base_id:
            return False

        self.vectors.delete_document(document_id)
        self.objects.delete_object(document["source_object_key"])
        self.repository.delete_document(document_id)
        return True

    def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        if not self.repository.get_knowledge_base(knowledge_base_id):
            return False

        documents = self.repository.list_documents(knowledge_base_id)
        self.vectors.delete_knowledge_base(knowledge_base_id)
        for document in documents:
            self.objects.delete_object(document["source_object_key"])
        self.repository.delete_knowledge_base(knowledge_base_id)
        return True
