from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    __package__ = "rag_app"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .application.ingestion_service import IngestionService
from .application.rag_service import RagService
from .config import Settings
from .infrastructure.minio.object_store import MinioObjectStore
from .infrastructure.ollama.gateway import OllamaGateway
from .infrastructure.parsing.document_parser import DocumentParser
from .infrastructure.postgres.repository import PostgresRepository
from .infrastructure.qdrant.vector_store import QdrantVectorStore


logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    repository: PostgresRepository
    objects: MinioObjectStore
    vectors: QdrantVectorStore
    models: OllamaGateway
    ingestion: IngestionService
    rag: RagService


def build_services(settings: Settings) -> Services:
    repository = PostgresRepository(settings.database_url)
    objects = MinioObjectStore(
        settings.minio_endpoint,
        settings.minio_access_key,
        settings.minio_secret_key,
        settings.minio_secure,
        settings.minio_bucket,
    )
    vectors = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection)
    models = OllamaGateway(settings.ollama_url, settings.ollama_chat_model, settings.ollama_embedding_model)
    return Services(
        settings=settings,
        repository=repository,
        objects=objects,
        vectors=vectors,
        models=models,
        ingestion=IngestionService(repository, objects, vectors, DocumentParser(), models),
        rag=RagService(repository, vectors, models, settings.rag_retrieval_top_k, settings.rag_context_top_k),
    )


def initialize_services(services: Services) -> None:
    checks = (
        ("PostgreSQL", services.repository.initialize),
        ("MinIO", services.objects.ensure_bucket),
        ("Qdrant", services.vectors.check_connection),
        ("Ollama", services.models.check_connection),
    )
    errors = []
    for name, check in checks:
        try:
            check()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError(" | ".join(errors))


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = build_services(Settings())
    app.state.services = services
    app.state.startup_error = None
    try:
        initialize_services(services)
    except Exception as exc:
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
        logger.error("基础设施初始化失败: %s", app.state.startup_error)
    try:
        yield
    finally:
        services.repository.close()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
