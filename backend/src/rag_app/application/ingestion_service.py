from hashlib import sha256
from io import BytesIO, SEEK_END
import logging
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Lock, Thread
from typing import BinaryIO
from uuid import uuid4

from ..domain.ports import DocumentParser, MetadataRepository, ModelGateway, ObjectStore, VectorStore


logger = logging.getLogger(__name__)


class DocumentTooLargeError(ValueError):
    pass


class DuplicateDocumentError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class IngestionCancelled(RuntimeError):
    pass


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
        testset_sync=None,
    ):
        self.repository = repository
        self.objects = objects
        self.vectors = vectors
        self.parser = parser
        self.models = models
        self.testset_sync = testset_sync
        self._ingestion_slots = BoundedSemaphore(max(1, max_concurrency))
        self._embedding_slots = BoundedSemaphore(max(1, embedding_max_concurrency))
        self._deduplication_lock = Lock()
        self._backfill_lock = Lock()
        self._backfill_started: set[str] = set()
        self._worker_lock = Lock()
        self._pending_lock = Lock()
        self._pending_document_ids: set[str] = set()
        self._cancelled_document_ids: set[str] = set()
        self._worker_count = max(1, max_concurrency)
        self._workers: list[Thread] = []
        self._ingestion_queue: Queue[dict | None] = Queue()
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

    def enqueue_stream(
        self,
        knowledge_base_id: str,
        file_name: str,
        mime_type: str,
        stream: BinaryIO,
        max_bytes: int | None = None,
        folder_path: str = "",
    ) -> dict:
        content_length = self._measure_stream(stream)
        if max_bytes and max_bytes > 0 and content_length > max_bytes:
            raise DocumentTooLargeError(f"单个文档不能超过 {max_bytes // (1024 * 1024)} MB")
        prepared = self._prepare_ingestion(
            knowledge_base_id,
            file_name,
            mime_type,
            stream,
            content_length,
            folder_path,
        )
        document_id = prepared["document_id"]
        with self._pending_lock:
            self._pending_document_ids.add(document_id)
        try:
            self._ensure_workers_started()
            self._ingestion_queue.put(prepared)
        except Exception:
            with self._pending_lock:
                self._pending_document_ids.discard(document_id)
            raise
        return self.repository.get_document(prepared["document_id"]) or {
            "id": prepared["document_id"],
            "status": "processing",
            "progress": 10,
            "stage": "parsing",
        }

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

    def _ensure_workers_started(self) -> None:
        with self._worker_lock:
            if self._workers:
                return
            for index in range(self._worker_count):
                worker = Thread(
                    target=self._run_ingestion_worker,
                    name=f"ingestion-worker-{index + 1}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

    def _run_ingestion_worker(self) -> None:
        while True:
            prepared = self._ingestion_queue.get()
            try:
                if prepared is None:
                    return
                with self._ingestion_slots:
                    try:
                        self._raise_if_cancelled(prepared["document_id"])
                        stream = self.objects.open_stream(prepared["object_key"])
                        try:
                            self._process_ingestion(prepared, stream)
                        finally:
                            stream.close()
                    except IngestionCancelled:
                        self._cleanup_cancelled_ingestion(prepared)
                    except Exception as exc:
                        try:
                            self.repository.update_document(
                                prepared["document_id"],
                                status="failed",
                                stage="failed",
                                error_message=str(exc),
                            )
                        except Exception:
                            pass
            finally:
                if prepared is not None:
                    with self._pending_lock:
                        self._pending_document_ids.discard(prepared["document_id"])
                        self._cancelled_document_ids.discard(prepared["document_id"])
                self._ingestion_queue.task_done()

    def is_pending(self, document_id: str) -> bool:
        with self._pending_lock:
            return document_id in self._pending_document_ids

    def cancel(self, document_id: str) -> bool:
        with self._pending_lock:
            if document_id not in self._pending_document_ids:
                return False
            self._cancelled_document_ids.add(document_id)
            return True

    def _raise_if_cancelled(self, document_id: str) -> None:
        with self._pending_lock:
            cancelled = document_id in self._cancelled_document_ids
        if cancelled:
            raise IngestionCancelled(f"文档导入已取消: {document_id}")

    def _cleanup_cancelled_ingestion(self, prepared: dict) -> None:
        document_id = prepared["document_id"]
        try:
            self.vectors.delete_document(document_id)
        except Exception:
            pass
        try:
            self.objects.delete_object(prepared["object_key"])
        except Exception:
            pass

    def wait_for_pending(self) -> None:
        self._ingestion_queue.join()

    def _ingest_stream(
        self,
        knowledge_base_id: str,
        file_name: str,
        mime_type: str,
        stream: BinaryIO,
        content_length: int,
        folder_path: str = "",
    ) -> dict:
        prepared = self._prepare_ingestion(
            knowledge_base_id,
            file_name,
            mime_type,
            stream,
            content_length,
            folder_path,
        )
        return self._process_ingestion(prepared, stream)

    def _prepare_ingestion(
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
        content_hash = self._content_hash(stream)
        self._start_content_hash_backfill(knowledge_base_id)
        with self._deduplication_lock:
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
                "testset_sync_status": "pending" if self.testset_sync else "disabled",
            })
        try:
            stream.seek(0)
            self.objects.put_stream(object_key, stream, content_length, mime_type)
            stream.seek(0)
            self.repository.update_document(document_id, progress=10, stage="parsing")
        except Exception as exc:
            try:
                self.objects.delete_object(object_key)
            except Exception:
                pass
            self.repository.update_document(document_id, status="failed", stage="failed", error_message=str(exc))
            raise
        return {
            "knowledge_base_id": knowledge_base_id,
            "document_id": document_id,
            "safe_name": safe_name,
            "suffix": suffix,
            "title": title,
            "object_key": object_key,
            "folder_path": folder_path,
            "relative_path": relative_path,
            "mime_type": mime_type,
            "content_hash": content_hash,
        }

    def _process_ingestion(self, prepared: dict, stream: BinaryIO) -> dict:
        knowledge_base_id = prepared["knowledge_base_id"]
        document_id = prepared["document_id"]
        safe_name = prepared["safe_name"]
        suffix = prepared["suffix"]
        title = prepared["title"]
        folder_path = prepared["folder_path"]
        relative_path = prepared["relative_path"]
        try:
            self._raise_if_cancelled(document_id)
            stream.seek(0)
            chunks = self.parser.parse_stream(safe_name, stream)
            self._raise_if_cancelled(document_id)
            if not chunks:
                raise ValueError("文档没有可索引的文本内容")
            self.repository.update_document(document_id, progress=35, stage="embedding")
            with self._embedding_slots:
                for start in range(0, len(chunks), self.embedding_batch_size):
                    chunk_batch = chunks[start:start + self.embedding_batch_size]
                    try:
                        self._raise_if_cancelled(document_id)
                        embeddings = self.models.embed([
                            f"完整路径: {relative_path}\n文件名: {safe_name}\n文件后缀: {suffix or '无后缀'}"
                            f"\n章节路径: {chunk.section_path or '无'}\n内容:\n{chunk.text}"
                            for chunk in chunk_batch
                        ])
                        self._raise_if_cancelled(document_id)
                        if not embeddings or len(embeddings) != len(chunk_batch):
                            raise RuntimeError("embedding 服务未返回完整向量")
                    except IngestionCancelled:
                        raise
                    except Exception as exc:
                        raise RuntimeError(f"向量化失败（embedding 模型可能未就绪）: {exc}") from exc
                    try:
                        self._raise_if_cancelled(document_id)
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
                                "section_path": chunk.section_path,
                                "chunk_index": chunk.index,
                                "text": chunk.text,
                            }
                            for chunk in chunk_batch
                        ]
                        self.vectors.upsert(points, embeddings)
                        self._raise_if_cancelled(document_id)
                    except IngestionCancelled:
                        raise
                    except Exception as exc:
                        raise RuntimeError(f"向量索引写入失败: {exc}") from exc
            self._raise_if_cancelled(document_id)
            self.repository.update_document(document_id, progress=70, stage="indexing")
            self.repository.update_document(document_id, progress=90, stage="saving")
            self._raise_if_cancelled(document_id)
            self.repository.replace_chunks(document_id, knowledge_base_id, chunks, folder_path)
            self._raise_if_cancelled(document_id)
            if self.testset_sync:
                self.repository.update_document(document_id, progress=95, stage="syncing_testset")
                try:
                    self.testset_sync.sync_document(prepared, chunks)
                except Exception:
                    logger.exception("Test-set tool synchronization failed for document %s", document_id)
            self.repository.update_document(
                document_id,
                status="ready",
                progress=100,
                stage="ready",
                chunk_count=len(chunks),
                error_message=None,
            )
            self._raise_if_cancelled(document_id)
        except IngestionCancelled:
            raise
        except Exception as exc:
            try:
                self.vectors.delete_document(document_id)
            except Exception:
                pass
            self.repository.update_document(document_id, status="failed", stage="failed", error_message=str(exc))
            raise
        return self.repository.get_document(document_id) or {"id": document_id, "status": "ready"}
