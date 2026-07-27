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

    def ingest(self, knowledge_base_id: str, file_name: str, mime_type: str, content: bytes, folder_path: str = "") -> dict:
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        if not knowledge_base:
            raise ValueError("知识库不存在")
        if knowledge_base["embedding_model"] != self.models.embedding_model:
            raise RuntimeError(
                f"知识库使用 {knowledge_base['embedding_model']} 建立索引，当前 embedding 模型是 {self.models.embedding_model}，请重新建立知识库并导入文档"
            )
        safe_name = Path(file_name).name
        folder_parts = []
        for part in folder_path.replace("\\", "/").split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                raise ValueError("文件夹路径不能包含 ..")
            folder_parts.append(part)
        folder_path = "/".join(folder_parts)
        relative_path = f"{folder_path}/{safe_name}" if folder_path else safe_name
        if not self.parser.supports(safe_name):
            suffix = Path(safe_name).suffix.lower()
            raise ValueError(f"暂不支持的文件类型: {suffix or '无扩展名'}")
        if not content:
            raise ValueError("文件内容为空")

        if self.repository.document_exists_by_file(knowledge_base_id, safe_name, folder_path):
            raise ValueError("文件重复：该知识库中已存在相同路径的同名文件")

        document_id = str(uuid4())
        suffix = Path(safe_name).suffix.lower()
        title = safe_name
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
            "folder_path": folder_path,
        })
        try:
            self.repository.update_document(document_id, progress=10, stage="parsing")
            chunks = self.parser.parse(safe_name, content)
            if not chunks:
                raise ValueError("文档没有可索引的文本内容")
            self.repository.update_document(document_id, progress=35, stage="embedding")
            try:
                embeddings = self.models.embed([
                    f"完整路径: {relative_path}\n文件名: {safe_name}\n文件后缀: {suffix or '无后缀'}\n内容:\n{chunk.text}"
                    for chunk in chunks
                ])
            except Exception as exc:
                raise RuntimeError(f"向量化失败（embedding 模型可能未就绪）: {exc}") from exc
            self.repository.update_document(document_id, progress=70, stage="indexing")
            try:
                self.vectors.ensure_collection(len(embeddings[0]))
                points = [
                    {
                        "chunk_id": f"{document_id}:{chunk.index}",
                        "knowledge_base_id": knowledge_base_id,
                        "document_id": document_id,
                        "title": title,
                        "file_name": safe_name,
                        "folder_path": folder_path,
                        "relative_path": relative_path,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ]
                self.vectors.upsert(points, embeddings)
            except Exception as exc:
                raise RuntimeError(f"向量索引写入失败: {exc}") from exc
            self.repository.update_document(document_id, progress=90, stage="saving")
            self.repository.replace_chunks(document_id, knowledge_base_id, chunks, folder_path)
            self.repository.update_document(
                document_id,
                status="ready",
                progress=100,
                stage="ready",
                chunk_count=len(chunks),
                error_message=None,
            )
        except Exception as exc:
            self.repository.update_document(document_id, status="failed", stage="failed", error_message=str(exc))
            raise
        return self.repository.get_document(document_id) or {"id": document_id, "status": "ready"}
