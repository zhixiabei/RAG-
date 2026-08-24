from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile

from ..application.ingestion_service import DocumentTooLargeError, DuplicateDocumentError
from fastapi.responses import JSONResponse
from uuid import uuid4

from ..config import PROJECT_ROOT
from ..evaluation import (
    EvaluationError,
    load_dataset,
    load_dataset_from_testset_tool,
    run_evaluation,
    select_dataset_samples,
)
from .schemas import (
    ChatRequest,
    ConversationCreate,
    ConversationUpdate,
    EvaluationRequest,
    KnowledgeBaseCreate,
    ParsedAttachmentChatRequest,
)

router = APIRouter()

MAX_CHAT_ATTACHMENTS = 10
MAX_CHAT_ATTACHMENT_BYTES = 30 * 1024 * 1024
MAX_CHAT_ATTACHMENT_CONTEXT_CHARS = 12_000
def _evaluation_request_timeout(settings) -> float:
    return max(1.0, float(getattr(settings, "evaluation_request_timeout_seconds", 90.0)))


def _evaluation_max_concurrency(settings) -> int:
    return max(1, int(getattr(settings, "evaluation_max_concurrency", 2)))


def _local_dataset_dir(settings) -> Path:
    raw_dir = str(getattr(settings, "evaluation_dataset_dir", "testsets") or "testsets").strip()
    path = Path(raw_dir).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _local_dataset_path(settings, dataset_id: str | None) -> Path:
    if not dataset_id:
        raise EvaluationError("dataset_id is required for local evaluation")
    directory = _local_dataset_dir(settings).resolve()
    candidate = (directory / dataset_id).resolve()
    if candidate.stem.casefold().endswith("_chunk_candidates"):
        raise EvaluationError("chunk candidate files are not formal evaluation datasets")
    if candidate.parent != directory or candidate.suffix.casefold() != ".jsonl":
        raise EvaluationError("invalid local dataset_id")
    if not candidate.is_file():
        raise EvaluationError(f"Local evaluation dataset does not exist: {dataset_id}")
    return candidate


def _list_local_datasets(settings) -> list[dict]:
    directory = _local_dataset_dir(settings)
    if not directory.is_dir():
        return []
    datasets = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem.casefold().endswith("_chunk_candidates"):
            continue
        if not path.is_file():
            continue
        try:
            samples = load_dataset(path, allow_empty=True)
            error = None
        except EvaluationError as exc:
            samples = []
            error = str(exc)
        datasets.append({
            "id": path.name,
            "name": path.stem,
            "sample_count": len(samples),
            "error": error,
        })
    return datasets


def _load_evaluation_samples(
    settings,
    source: str,
    question_ids=None,
    dataset_id: str | None = None,
):
    if source == "local":
        path = _local_dataset_path(settings, dataset_id)
        samples = load_dataset(path, bool(question_ids))
        return select_dataset_samples(samples, question_ids), {
            "dataset_source": "local",
            "dataset_path": str(path),
        }
    testset_url = str(getattr(settings, "testset_tool_base_url", "") or "").strip()
    if not testset_url:
        raise EvaluationError("TESTSET_TOOL_BASE_URL is not configured")
    return load_dataset_from_testset_tool(
        testset_url,
        _evaluation_request_timeout(settings),
        question_ids=question_ids,
    )


def services(request: Request):
    startup_error = getattr(request.app.state, "startup_error", None)
    if startup_error:
        raise HTTPException(
            503,
            f"基础设施或模型服务未就绪，请检查 PostgreSQL、MinIO、Qdrant 和模型配置。原因: {startup_error}",
        )
    return request.app.state.services


def owned_knowledge_base(request: Request, knowledge_base_id: str) -> tuple[object, dict]:
    service = services(request)
    item = service.repository.get_knowledge_base(knowledge_base_id, service.settings.owner_id)
    if not item:
        raise HTTPException(404, "知识库不存在")
    return service, item


@router.get("/health")
def health(request: Request):
    service = request.app.state.services
    answer_judge = getattr(service, "answer_judge", None)
    startup_error = getattr(request.app.state, "startup_error", None)
    payload = {
        "ok": startup_error is None,
        "storage": "postgresql+minio+qdrant",
        "model_mode": service.settings.model_mode,
        "embedding_model": service.models.embedding_model,
        "judge_model": answer_judge.model_name if answer_judge else None,
    }
    if startup_error:
        payload["detail"] = f"基础设施初始化失败: {startup_error}"
    return JSONResponse(status_code=503 if startup_error else 200, content=payload)


@router.get("/api/v1/models")
def list_models(request: Request):
    return services(request).models.list_chat_models()


@router.post("/api/v1/knowledge-bases")
def create_knowledge_base(request: Request, payload: KnowledgeBaseCreate):
    service = services(request)
    return service.repository.create_knowledge_base(
        str(uuid4()),
        payload.name,
        payload.description,
        service.models.embedding_model,
        service.settings.owner_id,
    )


@router.get("/api/v1/knowledge-bases")
def list_knowledge_bases(request: Request):
    service = services(request)
    return service.repository.list_knowledge_bases(service.settings.owner_id)


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}")
def get_knowledge_base(request: Request, knowledge_base_id: str):
    _, item = owned_knowledge_base(request, knowledge_base_id)
    return item


@router.delete("/api/v1/knowledge-bases/{knowledge_base_id}", status_code=204)
def delete_knowledge_base(request: Request, knowledge_base_id: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    try:
        deleted = service.deletion.delete_knowledge_base(knowledge_base_id)
    except Exception as exc:
        raise HTTPException(503, f"知识库删除失败: {exc}") from exc
    if not deleted:
        raise HTTPException(404, "知识库不存在")
    return Response(status_code=204)


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/documents")
def list_documents(request: Request, knowledge_base_id: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    return service.repository.list_documents(knowledge_base_id)


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/testset-sync")
def sync_knowledge_base_to_testset(request: Request, knowledge_base_id: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    testset_sync = getattr(service, "testset_sync", None)
    if not testset_sync:
        raise HTTPException(503, "TESTSET_TOOL_BASE_URL is not configured")
    return testset_sync.sync_knowledge_base(knowledge_base_id)


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/evaluation-datasets")
def list_evaluation_datasets(request: Request, knowledge_base_id: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    return {
        "default_source": "workshop",
        "workshop_available": bool(service.settings.testset_tool_base_url.strip()),
        "local_datasets": _list_local_datasets(service.settings),
    }


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/evaluation-samples")
def list_evaluation_samples(
    request: Request,
    knowledge_base_id: str,
    source: str = "workshop",
    dataset: str | None = None,
):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    if source not in {"workshop", "local"}:
        raise HTTPException(422, "source must be workshop or local")
    try:
        samples, _ = _load_evaluation_samples(
            service.settings,
            source,
            dataset_id=dataset,
        )
    except EvaluationError as exc:
        raise HTTPException(503, f"读取测试集失败: {exc}") from exc
    return [
        {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "question_type": sample.get("question_type"),
            "difficulty": sample.get("difficulty"),
        }
        for sample in samples
    ]


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/evaluation")
def evaluate_knowledge_base(
    request: Request,
    knowledge_base_id: str,
    payload: EvaluationRequest,
):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    source = payload.dataset_source
    try:
        local_path = (
            _local_dataset_path(service.settings, payload.dataset_id)
            if source == "local"
            else None
        )
        testset_url = str(getattr(service.settings, "testset_tool_base_url", "") or "").strip()
        if source == "local":
            _load_evaluation_samples(
                service.settings,
                source,
                payload.question_ids,
                payload.dataset_id,
            )
        elif not testset_url:
            raise EvaluationError("TESTSET_TOOL_BASE_URL is not configured")
        return run_evaluation(
            local_path,
            knowledge_base_id,
            str(request.base_url).rstrip("/"),
            None,
            _evaluation_request_timeout(service.settings),
            True,
            False,
            None if source == "local" else testset_url,
            payload.question_ids,
            judge_agent=getattr(service, "answer_judge", None),
            evidence_embedding_models=service.models,
            max_concurrency=_evaluation_max_concurrency(service.settings),
        )
    except EvaluationError as exc:
        raise HTTPException(422, f"评测无法启动: {exc}") from exc
    except Exception as exc:
        raise HTTPException(503, f"评测失败: {exc}") from exc


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
def get_document(request: Request, knowledge_base_id: str, document_id: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    document = service.repository.get_document(document_id)
    if not document or document["knowledge_base_id"] != knowledge_base_id:
        raise HTTPException(404, "文档不存在")
    return document


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/documents", status_code=202)
def upload_document(request: Request, knowledge_base_id: str, file: UploadFile = File(...), folder_path: str = Form("")):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")
    if not service.ingestion.parser.supports(file.filename):
        suffix = Path(file.filename).suffix.lower()
        raise HTTPException(415, f"暂不支持的文件类型: {suffix or '无扩展名'}")
    try:
        return service.ingestion.enqueue_stream(
            knowledge_base_id,
            Path(file.filename).name,
            file.content_type or "application/octet-stream",
            file.file,
            service.settings.max_document_bytes,
            folder_path,
        )
    except DocumentTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except DuplicateDocumentError as exc:
        return JSONResponse({
            "status": "skipped",
            "reason": "duplicate",
            "duplicate_kind": exc.kind,
            "detail": str(exc),
        })
    except ValueError as exc:
        raise HTTPException(500, f"文档入库失败: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"文档入库失败: {exc}") from exc


@router.delete("/api/v1/knowledge-bases/{knowledge_base_id}/documents")
def delete_document_folder(request: Request, knowledge_base_id: str, folder_path: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    try:
        deleted_count = service.deletion.delete_document_folder(knowledge_base_id, folder_path)
    except Exception as exc:
        raise HTTPException(503, f"文件夹删除失败: {exc}") from exc
    if not deleted_count:
        raise HTTPException(404, "文件夹不存在或文件夹中没有文档")
    return {"deleted_count": deleted_count}


@router.delete("/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}", status_code=204)
def delete_document(request: Request, knowledge_base_id: str, document_id: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    try:
        deleted = service.deletion.delete_document(knowledge_base_id, document_id)
    except Exception as exc:
        raise HTTPException(503, f"文档删除失败: {exc}") from exc
    if not deleted:
        raise HTTPException(404, "文档不存在")
    return Response(status_code=204)


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/chat")
def chat(request: Request, knowledge_base_id: str, payload: ChatRequest):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    conversation = service.repository.get_conversation(payload.conversation_id, service.settings.owner_id)
    if not conversation or conversation["knowledge_base_id"] != knowledge_base_id:
        raise HTTPException(404, "对话不存在")
    try:
        return service.rag.answer(
            knowledge_base_id,
            payload.conversation_id,
            payload.question,
            payload.model,
            include_retrieved_content=payload.include_retrieved_content,
            force_retrieval=payload.force_retrieval,
        )
    except Exception as exc:
        raise HTTPException(503, f"问答服务不可用: {exc}") from exc


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/chat-attachments/parse")
def parse_chat_attachment(request: Request, knowledge_base_id: str, file: UploadFile = File(...)):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    file_name = Path(file.filename or "").name
    if not file_name:
        raise HTTPException(400, "附件文件名不能为空")
    if not service.ingestion.parser.supports(file_name):
        suffix = Path(file_name).suffix.lower()
        raise HTTPException(415, f"暂不支持的文件类型: {suffix or '无扩展名'}")
    try:
        chunks = service.ingestion.parse_stream(file_name, file.file, MAX_CHAT_ATTACHMENT_BYTES)
    except DocumentTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"附件解析失败: {exc}") from exc
    if not chunks:
        raise HTTPException(422, f"附件没有可读取的文本内容: {file_name}")

    context_parts: list[str] = []
    citations: list[dict] = []
    remaining_chars = MAX_CHAT_ATTACHMENT_CONTEXT_CHARS
    for chunk in chunks:
        if remaining_chars <= 0:
            break
        chunk_id = f"attachment:0:{chunk.index}"
        header = f"[证据ID] {chunk_id}\n[临时附件] {file_name}\n[页码] {chunk.page_number or '未知'}\n[内容] "
        available = max(0, remaining_chars - len(header) - 2)
        if not available:
            break
        text = chunk.text[:available]
        context_parts.append(f"{header}{text}")
        citations.append({
            "document_id": None,
            "chunk_id": chunk_id,
            "title": file_name,
            "page_number": chunk.page_number,
            "score": 1.0,
            "relevance_score": None,
            "temporary": True,
            "excerpt": text[:500],
        })
        remaining_chars -= len(header) + len(text) + 2
    return {
        "name": file_name,
        "context": "\n\n".join(context_parts),
        "citations": citations,
        "chunk_count": len(chunks),
    }


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/chat-with-parsed-attachments")
def chat_with_parsed_attachments(
    request: Request,
    knowledge_base_id: str,
    payload: ParsedAttachmentChatRequest,
):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    conversation = service.repository.get_conversation(payload.conversation_id, service.settings.owner_id)
    if not conversation or conversation["knowledge_base_id"] != knowledge_base_id:
        raise HTTPException(404, "对话不存在")

    remaining_chars = MAX_CHAT_ATTACHMENT_CONTEXT_CHARS
    context_parts: list[str] = []
    citations: list[dict] = []
    attachment_names: list[str] = []
    for attachment_index, attachment in enumerate(payload.attachments):
        if remaining_chars <= 0:
            break
        context = attachment.context
        attachment_citations = []
        for citation_index, citation in enumerate(attachment.citations):
            normalized = dict(citation)
            original_chunk_id = str(citation.get("chunk_id") or citation_index)
            chunk_id = f"attachment:{attachment_index}:{citation_index}"
            context = context.replace(f"[证据ID] {original_chunk_id}", f"[证据ID] {chunk_id}")
            normalized["chunk_id"] = chunk_id
            normalized["title"] = attachment.name
            normalized["temporary"] = True
            attachment_citations.append(normalized)
        if attachment_citations and "[证据ID]" not in context:
            context = f"[证据ID] {attachment_citations[0]['chunk_id']}\n{context}"
        context = context[:remaining_chars]
        context_parts.append(context)
        remaining_chars -= len(context) + 2
        attachment_names.append(attachment.name)
        citations.extend(attachment_citations)


    question_with_attachments = f"{payload.question}\n\n附件：{'、'.join(attachment_names)}"
    try:
        return service.rag.answer(
            knowledge_base_id,
            payload.conversation_id,
            question_with_attachments,
            payload.model,
            attachment_context="\n\n".join(context_parts),
            attachment_citations=citations,
        )
    except Exception as exc:
        raise HTTPException(503, f"问答服务不可用: {exc}") from exc


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/chat-with-attachments")
def chat_with_attachments(
    request: Request,
    knowledge_base_id: str,
    conversation_id: str = Form(...),
    question: str = Form(...),
    model: str | None = Form(None),
    files: list[UploadFile] = File(...),
):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    conversation = service.repository.get_conversation(conversation_id, service.settings.owner_id)
    if not conversation or conversation["knowledge_base_id"] != knowledge_base_id:
        raise HTTPException(404, "对话不存在")
    try:
        payload = ChatRequest(conversation_id=conversation_id, question=question, model=model or None)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not files or len(files) > MAX_CHAT_ATTACHMENTS:
        raise HTTPException(400, f"每次可上传 1 至 {MAX_CHAT_ATTACHMENTS} 个临时附件")

    context_parts: list[str] = []
    citations: list[dict] = []
    total_bytes = 0
    remaining_chars = MAX_CHAT_ATTACHMENT_CONTEXT_CHARS
    try:
        for file_index, file in enumerate(files):
            file_name = Path(file.filename or "").name
            if not file_name:
                raise HTTPException(400, "附件文件名不能为空")
            if not service.ingestion.parser.supports(file_name):
                suffix = Path(file_name).suffix.lower()
                raise HTTPException(415, f"暂不支持的文件类型: {suffix or '无扩展名'}")
            remaining_bytes = MAX_CHAT_ATTACHMENT_BYTES - total_bytes
            content = file.file.read(remaining_bytes + 1)
            total_bytes += len(content)
            if total_bytes > MAX_CHAT_ATTACHMENT_BYTES:
                raise HTTPException(413, "临时附件总大小不能超过 30 MB")
            chunks = service.ingestion.parser.parse(file_name, content)
            if not chunks:
                raise HTTPException(422, f"附件没有可读取的文本内容: {file_name}")
            for chunk in chunks:
                if remaining_chars <= 0:
                    break
                chunk_id = f"attachment:{file_index}:{chunk.index}"
                header = f"[证据ID] {chunk_id}\n[临时附件] {file_name}\n[页码] {chunk.page_number or '未知'}\n[内容] "
                available = max(0, remaining_chars - len(header) - 2)
                if not available:
                    break
                text = chunk.text[:available]
                context_parts.append(f"{header}{text}")
                citations.append({
                    "document_id": None,
                    "chunk_id": chunk_id,
                    "title": file_name,
                    "page_number": chunk.page_number,
                    "score": 1.0,
                    "relevance_score": None,
                    "temporary": True,
                    "excerpt": text[:500],
                })
                remaining_chars -= len(header) + len(text) + 2
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"附件解析失败: {exc}") from exc

    try:
        attachment_names = "、".join(Path(file.filename or "").name for file in files)
        question_with_attachments = f"{payload.question}\n\n附件：{attachment_names}"
        return service.rag.answer(
            knowledge_base_id,
            payload.conversation_id,
            question_with_attachments,
            payload.model,
            attachment_context="\n\n".join(context_parts),
            attachment_citations=citations,
        )
    except Exception as exc:
        raise HTTPException(503, f"问答服务不可用: {exc}") from exc


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/conversations")
def list_conversations(request: Request, knowledge_base_id: str):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    return service.repository.list_conversations(knowledge_base_id, service.settings.owner_id)


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/conversations")
def create_conversation(request: Request, knowledge_base_id: str, payload: ConversationCreate):
    service, _ = owned_knowledge_base(request, knowledge_base_id)
    return service.repository.create_conversation(str(uuid4()), knowledge_base_id, payload.title.strip())


@router.get("/api/v1/conversations/{conversation_id}/messages")
def list_messages(request: Request, conversation_id: str):
    service = services(request)
    if not service.repository.get_conversation(conversation_id, service.settings.owner_id):
        raise HTTPException(404, "对话不存在")
    return service.repository.list_messages(conversation_id)


@router.patch("/api/v1/conversations/{conversation_id}")
def update_conversation(request: Request, conversation_id: str, payload: ConversationUpdate):
    service = services(request)
    if not service.repository.get_conversation(conversation_id, service.settings.owner_id):
        raise HTTPException(404, "对话不存在")
    return service.repository.update_conversation_title(conversation_id, payload.title)


@router.delete("/api/v1/conversations/{conversation_id}", status_code=204)
def delete_conversation(request: Request, conversation_id: str):
    service = services(request)
    if not service.repository.get_conversation(conversation_id, service.settings.owner_id):
        raise HTTPException(404, "对话不存在")
    service.repository.delete_conversation(conversation_id)
    return Response(status_code=204)
