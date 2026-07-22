from pathlib import Path
from uuid import uuid4

from ..domain.ports import DocumentParser, MetadataRepository, ModelGateway, ObjectStore, VectorStore


class IngestionService:
    def __init__(self, repository: MetadataRepository, objects: ObjectStore, vectors: VectorStore, parser: DocumentParser, models: ModelGateway):
        self.repository = repository
        self.objects = objects
        self.vectors = vectors
        self.parser = parser
        self.models = models

    def ingest(self, knowledge_base_id: str, file_name: str, mime_type: str, content: bytes) -> dict:
        document_id = str(uuid4())
        safe_name = Path(file_name).name
        title = Path(safe_name).stem.replace("_", " ").replace("-", " ") or safe_name
        object_key = f"{knowledge_base_id}/{document_id}/source/{safe_name}"
        self.objects.put_bytes(object_key, content, mime_type)
        self.repository.create_document({
            "id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "title": title,
            "file_name": safe_name,
            "mime_type": mime_type,
            "source_object_key": object_key,
            "status": "processing",
        })
        try:
            chunks = self.parser.parse(safe_name, content)
            if not chunks:
                raise ValueError("文档没有可索引的文本")
            embeddings = self.models.embed([chunk.text for chunk in chunks])
            self.vectors.ensure_collection(len(embeddings[0]))
            points = [
                {
                    "chunk_id": f"{document_id}:{chunk.index}",
                    "knowledge_base_id": knowledge_base_id,
                    "document_id": document_id,
                    "title": title,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                }
                for chunk in chunks
            ]
            self.vectors.upsert(points, embeddings)
            self.repository.replace_chunks(document_id, knowledge_base_id, chunks)
            self.repository.update_document(document_id, status="ready", chunk_count=len(chunks), error_message=None)
        except Exception as exc:
            self.repository.update_document(document_id, status="failed", error_message=str(exc))
            raise
        return self.repository.get_document(document_id) or {"id": document_id, "status": "ready"}

