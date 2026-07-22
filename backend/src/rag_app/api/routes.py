from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from uuid import uuid4

from .schemas import ChatRequest, KnowledgeBaseCreate

router = APIRouter()


def services(request: Request):
    startup_error = getattr(request.app.state, "startup_error", None)
    if startup_error:
        raise HTTPException(
            503,
            f"基础设施未就绪，请先启动 PostgreSQL、MinIO 和 Qdrant。原因: {startup_error}",
        )
    return request.app.state.services


@router.get("/health")
def health(request: Request):
    service = request.app.state.services
    startup_error = getattr(request.app.state, "startup_error", None)
    payload = {
        "ok": startup_error is None,
        "storage": "postgresql+minio+qdrant",
        "ollama": service.settings.ollama_url,
    }
    if startup_error:
        payload["detail"] = f"基础设施初始化失败: {startup_error}"
    return JSONResponse(status_code=503 if startup_error else 200, content=payload)


@router.post("/api/v1/knowledge-bases")
def create_knowledge_base(request: Request, payload: KnowledgeBaseCreate):
    service = services(request)
    return service.repository.create_knowledge_base(str(uuid4()), payload.name, payload.description, service.settings.ollama_embedding_model)


@router.get("/api/v1/knowledge-bases")
def list_knowledge_bases(request: Request):
    return services(request).repository.list_knowledge_bases()


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}")
def get_knowledge_base(request: Request, knowledge_base_id: str):
    item = services(request).repository.get_knowledge_base(knowledge_base_id)
    if not item:
        raise HTTPException(404, "知识库不存在")
    return item


@router.get("/api/v1/knowledge-bases/{knowledge_base_id}/documents")
def list_documents(request: Request, knowledge_base_id: str):
    service = services(request)
    if not service.repository.knowledge_base_exists(knowledge_base_id):
        raise HTTPException(404, "知识库不存在")
    return service.repository.list_documents(knowledge_base_id)


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/documents")
def upload_document(request: Request, knowledge_base_id: str, file: UploadFile = File(...)):
    service = services(request)
    if not service.repository.knowledge_base_exists(knowledge_base_id):
        raise HTTPException(404, "知识库不存在")
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
        )
    except Exception as exc:
        raise HTTPException(500, f"文档入库失败: {exc}") from exc


@router.post("/api/v1/knowledge-bases/{knowledge_base_id}/chat")
def chat(request: Request, knowledge_base_id: str, payload: ChatRequest):
    service = services(request)
    if not service.repository.knowledge_base_exists(knowledge_base_id):
        raise HTTPException(404, "知识库不存在")
    try:
        return service.rag.answer(knowledge_base_id, payload.question)
    except Exception as exc:
        raise HTTPException(503, f"问答服务不可用: {exc}") from exc
