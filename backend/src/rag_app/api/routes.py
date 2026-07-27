from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from uuid import uuid4

from ..application.auth import SESSION_COOKIE_NAME, create_session, credentials_match, verify_session
from .schemas import ChatRequest, ConversationCreate, ConversationUpdate, KnowledgeBaseCreate, LoginRequest

router = APIRouter()


def services(request: Request):
    startup_error = getattr(request.app.state, "startup_error", None)
    if startup_error:
        raise HTTPException(
            503,
            f"基础设施或模型服务未就绪，请检查 PostgreSQL、MinIO、Qdrant 和模型配置。原因: {startup_error}",
        )
    return request.app.state.services


def current_user(request: Request) -> dict:
    settings = request.app.state.services.settings
    payload = verify_session(request.cookies.get(SESSION_COOKIE_NAME), settings.auth_secret)
    if (
        not payload
        or payload["sub"] != settings.auth_username
        or payload["owner_id"] != settings.auth_owner_id
    ):
        raise HTTPException(401, "请先登录")
    return {"username": payload["sub"], "owner_id": payload["owner_id"]}


def owned_knowledge_base(request: Request, knowledge_base_id: str) -> tuple[object, dict, dict]:
    service = services(request)
    user = current_user(request)
    item = service.repository.get_knowledge_base(knowledge_base_id, user["owner_id"])
    if not item:
        raise HTTPException(404, "知识库不存在")
    return service, user, item


@router.post("/api/v1/auth/login")
def login(request: Request, response: Response, payload: LoginRequest):
    settings = request.app.state.services.settings
    if not credentials_match(payload.username, payload.password, settings.auth_username, settings.auth_password):
        raise HTTPException(401, "用户名或密码错误")
    token = create_session(
        settings.auth_username,
        settings.auth_owner_id,
        settings.auth_secret,
        settings.auth_session_ttl_seconds,
    )
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=settings.auth_session_ttl_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )
    return {"username": settings.auth_username}


@router.post("/api/v1/auth/logout", status_code=204)
def logout():
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


@router.get("/api/v1/auth/me")
def me(request: Request):
    user = current_user(request)
    return {"username": user["username"]}


@router.get("/health")
def health(request: Request):
    service = request.app.state.services
    startup_error = getattr(request.app.state, "startup_error", None)
    payload = {
        "ok": startup_error is None,
        "storage": "postgresql+minio+qdrant",
        "model_mode": service.settings.model_mode,
        "embedding_model": service.models.embedding_model,
    }
    if startup_error:
        payload["detail"] = f"基础设施初始化失败: {startup_error}"
    return JSONResponse(status_code=503 if startup_error else 200, content=payload)


@router.get("/api/v1/models")
def list_models(request: Request):
    current_user(request)
    return services(request).models.list_chat_models()


@router.post("/api/v1/knowledge-bases")
def create_knowledge_base(request: Request, payload: KnowledgeBaseCreate):
    service = services(request)
    user = current_user(request)
    return service.repository.create_knowledge_base(
        str(uuid4()),
        payload.name,
        payload.description,
        service.models.embedding_model,
        user["owner_id"],
    )


@router.get("/api/v1/knowledge-bases")
def list_knowledge_bases(request: Request):
    user = current_user(request)
    return services(request).repository.list_knowledge_bases(user["owner_id"])


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}")
def get_knowledge_base(request: Request, knowledge_base_id: str):
    _, _, item = owned_knowledge_base(request, knowledge_base_id)
    return item


@router.delete("/api/v1/knowledge-bases/{knowledge_base_id}", status_code=204)
def delete_knowledge_base(request: Request, knowledge_base_id: str):
    service, _, _ = owned_knowledge_base(request, knowledge_base_id)
    try:
        deleted = service.deletion.delete_knowledge_base(knowledge_base_id)
    except Exception as exc:
        raise HTTPException(503, f"知识库删除失败: {exc}") from exc
    if not deleted:
        raise HTTPException(404, "知识库不存在")
    return Response(status_code=204)


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/documents")
def list_documents(request: Request, knowledge_base_id: str):
    service, _, _ = owned_knowledge_base(request, knowledge_base_id)
    return service.repository.list_documents(knowledge_base_id)


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/documents")
def upload_document(request: Request, knowledge_base_id: str, file: UploadFile = File(...), folder_path: str = Form("")):
    service, _, _ = owned_knowledge_base(request, knowledge_base_id)
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")
    if not service.ingestion.parser.supports(file.filename):
        suffix = Path(file.filename).suffix.lower()
        raise HTTPException(415, f"暂不支持的文件类型: {suffix or '无扩展名'}")
    try:
        return service.ingestion.ingest(
            knowledge_base_id,
            Path(file.filename).name,
            file.content_type or "application/octet-stream",
            file.file.read(),
            folder_path,
        )
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("文件重复"):
            raise HTTPException(409, msg) from exc
        raise HTTPException(500, f"文档入库失败: {msg}") from exc
    except Exception as exc:
        raise HTTPException(500, f"文档入库失败: {exc}") from exc


@router.delete("/api/v1/knowledge-bases/{knowledge_base_id}/documents")
def delete_document_folder(request: Request, knowledge_base_id: str, folder_path: str):
    service, _, _ = owned_knowledge_base(request, knowledge_base_id)
    try:
        deleted_count = service.deletion.delete_document_folder(knowledge_base_id, folder_path)
    except Exception as exc:
        raise HTTPException(503, f"文件夹删除失败: {exc}") from exc
    if not deleted_count:
        raise HTTPException(404, "文件夹不存在或文件夹中没有文档")
    return {"deleted_count": deleted_count}


@router.delete("/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}", status_code=204)
def delete_document(request: Request, knowledge_base_id: str, document_id: str):
    service, _, _ = owned_knowledge_base(request, knowledge_base_id)
    try:
        deleted = service.deletion.delete_document(knowledge_base_id, document_id)
    except Exception as exc:
        raise HTTPException(503, f"文档删除失败: {exc}") from exc
    if not deleted:
        raise HTTPException(404, "文档不存在")
    return Response(status_code=204)


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/chat")
def chat(request: Request, knowledge_base_id: str, payload: ChatRequest):
    service, user, _ = owned_knowledge_base(request, knowledge_base_id)
    conversation = service.repository.get_conversation(payload.conversation_id, user["owner_id"])
    if not conversation or conversation["knowledge_base_id"] != knowledge_base_id:
        raise HTTPException(404, "对话不存在")
    try:
        return service.rag.answer(knowledge_base_id, payload.conversation_id, payload.question, payload.model)
    except Exception as exc:
        raise HTTPException(503, f"问答服务不可用: {exc}") from exc


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/conversations")
def list_conversations(request: Request, knowledge_base_id: str):
    service, user, _ = owned_knowledge_base(request, knowledge_base_id)
    return service.repository.list_conversations(knowledge_base_id, user["owner_id"])


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/conversations")
def create_conversation(request: Request, knowledge_base_id: str, payload: ConversationCreate):
    service, _, _ = owned_knowledge_base(request, knowledge_base_id)
    return service.repository.create_conversation(str(uuid4()), knowledge_base_id, payload.title.strip())


@router.get("/api/v1/conversations/{conversation_id}/messages")
def list_messages(request: Request, conversation_id: str):
    service = services(request)
    user = current_user(request)
    if not service.repository.get_conversation(conversation_id, user["owner_id"]):
        raise HTTPException(404, "对话不存在")
    return service.repository.list_messages(conversation_id)


@router.patch("/api/v1/conversations/{conversation_id}")
def update_conversation(request: Request, conversation_id: str, payload: ConversationUpdate):
    service = services(request)
    user = current_user(request)
    if not service.repository.get_conversation(conversation_id, user["owner_id"]):
        raise HTTPException(404, "对话不存在")
    return service.repository.update_conversation_title(conversation_id, payload.title)


@router.delete("/api/v1/conversations/{conversation_id}", status_code=204)
def delete_conversation(request: Request, conversation_id: str):
    service = services(request)
    user = current_user(request)
    if not service.repository.get_conversation(conversation_id, user["owner_id"]):
        raise HTTPException(404, "对话不存在")
    service.repository.delete_conversation(conversation_id)
    return Response(status_code=204)
