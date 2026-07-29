from hashlib import sha256
from io import BytesIO, SEEK_END
from pathlib import Path
from threading import BoundedSemaphore, Lock, Thread
from typing import BinaryIO
from uuid import uuid4

from ..domain.ports import DocumentParser, MetadataRepository, ModelGateway, ObjectStore, VectorStore


class DocumentTooLargeError(ValueError):
    pass


class DuplicateDocumentError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class IngestionService:
    def __init__(
        self,
        repository: MetadataRepository,
        objects: ObjectStore,
        vectors: VectorStore,
        parser: DocumentParser,
        models: ModelGateway,
        *,
        max_concurrency: int = 2,
        embedding_max_concurrency: int = 1,
        embedding_batch_size: int = 32,
    ):
        self.repository = repository
        self.objects = objects
        self.vectors = vectors
        self.parser = parser
        self.models = models
        self._ingestion_slots = BoundedSemaphore(max(1, max_concurrency))
        self._embedding_slots = BoundedSemaphore(max(1, embedding_max_concurrency))
        self._deduplication_lock = Lock()
        self._backfill_lock = Lock()
        self._backfill_started: set[str] = set()
        self.embedding_batch_size = max(1, embedding_batch_size)

    def ingest_stream(
        self,
        knowledge_base_id: str,
        file_name: str,
        mime_type: str,
        stream: BinaryIO,
        max_bytes: int | None = None,
        folder_path: str = "",
    ) -> dict:
        with self._ingestion_slots:
            content_length = self._measure_stream(stream)
            if max_bytes and max_bytes > 0 and content_length > max_bytes:
                raise DocumentTooLargeError(f"单个文档不能超过 {max_bytes // (1024 * 1024)} MB")
            return self._ingest_stream(
                knowledge_base_id,
                file_name,
                mime_type,
                stream,
                content_length,
                folder_path,
            )

    def parse_stream(self, file_name: str, stream: BinaryIO, max_bytes: int):
        with self._ingestion_slots:
            content_length = self._measure_stream(stream)
            if content_length > max_bytes:
                raise DocumentTooLargeError(f"单个临时附件不能超过 {max_bytes // (1024 * 1024)} MB")
            return self.parser.parse_stream(file_name, stream)

    def ingest(self, knowledge_base_id: str, file_name: str, mime_type: str, content: bytes, folder_path: str = "") -> dict:
        with self._ingestion_slots:
            return self._ingest_stream(
                knowledge_base_id,
                file_name,
                mime_type,
                BytesIO(content),
                len(content),
                folder_path,
            )

    @staticmethod
    def _measure_stream(stream: BinaryIO) -> int:
        try:
            stream.seek(0, SEEK_END)
            content_length = stream.tell()
            stream.seek(0)
        except (AttributeError, OSError) as exc:
            raise ValueError("上传文件流必须支持定位") from exc
        return content_length

    @staticmethod
    def _content_hash(stream: BinaryIO) -> str:
        digest = sha256()
        try:
            stream.seek(0)
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            stream.seek(0)
        except (AttributeError, OSError) as exc:
            raise ValueError("上传文件流必须支持定位") from exc
        return digest.hexdigest()

    def _backfill_content_hashes(self, knowledge_base_id: str) -> None:
        for document in self.repository.list_documents_without_content_hash(knowledge_base_id):
            try:
                existing_hash = self.objects.calculate_hash(document["source_object_key"])
            except Exception:
                continue
            self.repository.update_document(document["id"], content_hash=existing_hash)

    def _start_content_hash_backfill(self, knowledge_base_id: str) -> None:
        with self._backfill_lock:
            if knowledge_base_id in self._backfill_started:
                return
            self._backfill_started.add(knowledge_base_id)
        Thread(
            target=self._backfill_content_hashes,
            args=(knowledge_base_id,),
            name=f"content-hash-backfill-{knowledge_base_id}",
            daemon=True,
        ).start()

    def _ingest_stream(
        self,
        knowledge_base_id: str,
        file_name: str,
        mime_type: str,
        stream: BinaryIO,
        content_length: int,
        folder_path: str = "",
    ) -> dict:
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
        if content_length <= 0:
            raise ValueError("文件内容为空")

        document_id = str(uuid4())
        suffix = Path(safe_name).suffix.lower()
        title = safe_name
        object_key = f"{knowledge_base_id}/{document_id}/source/{safe_name}"
        with self._deduplication_lock:
            if self.repository.document_exists_by_file(knowledge_base_id, safe_name, folder_path):
                raise DuplicateDocumentError("path", "文件重复：该知识库中已存在相同路径的同名文件")

        content_hash = self._content_hash(stream)
        self._start_content_hash_backfill(knowledge_base_id)
        with self._deduplication_lock:
            if self.repository.document_exists_by_file(knowledge_base_id, safe_name, folder_path):
                raise DuplicateDocumentError("path", "文件重复：该知识库中已存在相同路径的同名文件")
            if self.repository.document_exists_by_content_hash(knowledge_base_id, content_hash):
                raise DuplicateDocumentError("content", "文件重复：该知识库中已存在内容相同的文件")
            self.repository.create_document({
                "id": document_id,
                "knowledge_base_id": knowledge_base_id,
                "title": title,
                "file_name": safe_name,
                "mime_type": mime_type,
                "source_object_key": object_key,
                "status": "processing",
                "folder_path": folder_path,
                "content_hash": content_hash,
            })
        try:
            stream.seek(0)
            self.objects.put_stream(object_key, stream, content_length, mime_type)
            stream.seek(0)
            self.repository.update_document(document_id, progress=10, stage="parsing")
            chunks = self.parser.parse_stream(safe_name, stream)
            if not chunks:
                raise ValueError("文档没有可索引的文本内容")
            self.repository.update_document(document_id, progress=35, stage="embedding")
            with self._embedding_slots:
                for start in range(0, len(chunks), self.embedding_batch_size):
                    chunk_batch = chunks[start:start + self.embedding_batch_size]
                    try:
                        embeddings = self.models.embed([
                            f"完整路径: {relative_path}\n文件名: {safe_name}\n文件后缀: {suffix or '无后缀'}\n内容:\n{chunk.text}"
                            for chunk in chunk_batch
                        ])
                        if not embeddings or len(embeddings) != len(chunk_batch):
                            raise RuntimeError("embedding 服务未返回完整向量")
                    except Exception as exc:
                        raise RuntimeError(f"向量化失败（embedding 模型可能未就绪）: {exc}") from exc
                    try:
                        if start == 0:
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
                            for chunk in chunk_batch
                        ]
                        self.vectors.upsert(points, embeddings)
                    except Exception as exc:
                        raise RuntimeError(f"向量索引写入失败: {exc}") from exc
            self.repository.update_document(document_id, progress=70, stage="indexing")
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
            try:
                self.vectors.delete_document(document_id)
            except Exception:
                pass
            self.repository.update_document(document_id, status="failed", stage="failed", error_message=str(exc))
            raise
        return self.repository.get_document(document_id) or {"id": document_id, "status": "ready"}
