from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __package__ in {None, ""}:
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    __package__ = "rag_app"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent import (
    AnswerAgent,
    ContextPolicy,
    KnowledgeRetrievalAgent,
    RetrievalDecisionAgent,
)
from .api.routes import router
from .application.deletion_service import DeletionService
from .application.ingestion_service import IngestionService
from .application.rag_service import RagService
from .config import Settings
from .infrastructure.minio.object_store import MinioObjectStore
from .infrastructure.parsing.document_parser import DocumentParser
from .infrastructure.postgres.repository import PostgresRepository
from .infrastructure.qdrant.vector_store import QdrantVectorStore
from .model_gateway_factory import ModelGateway, build_model_gateway
from .testset_tool import TestsetSyncService, TestsetToolClient


logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    repository: PostgresRepository
    objects: MinioObjectStore
    vectors: QdrantVectorStore
    models: ModelGateway
    ingestion: IngestionService
    deletion: DeletionService
    rag: RagService
    testset_sync: TestsetSyncService | None


def build_services(settings: Settings) -> Services:
    repository = PostgresRepository(settings.database_url)
    objects = MinioObjectStore(
        settings.minio_endpoint,
        settings.minio_access_key,
        settings.minio_secret_key,
        settings.minio_secure,
        settings.minio_bucket,
    )
    vectors = QdrantVectorStore(
        settings.qdrant_url,
        settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
        upsert_batch_size=settings.qdrant_upsert_batch_size,
        upsert_max_retries=settings.qdrant_upsert_max_retries,
        hnsw_m=settings.qdrant_hnsw_m,
        hnsw_ef_construct=settings.qdrant_hnsw_ef_construct,
        hnsw_full_scan_threshold=settings.qdrant_hnsw_full_scan_threshold,
        search_hnsw_ef=settings.qdrant_search_hnsw_ef,
    )
    models = build_model_gateway(settings)
    decision_agent = RetrievalDecisionAgent(models)
    retrieval_agent = KnowledgeRetrievalAgent(
        vectors,
        models,
        settings.rag_top_k,
    )
    answer_agent = AnswerAgent(
        models,
        ContextPolicy(
            max_input_tokens=settings.rag_context_max_input_tokens or None,
            output_reserve_tokens=settings.rag_context_output_reserve_tokens,
            history_max_tokens=settings.rag_context_history_max_tokens,
            catalog_max_tokens=settings.rag_context_catalog_max_tokens,
            attachment_max_tokens=settings.rag_context_attachment_max_tokens,
        ),
    )
    testset_sync = None
    if settings.testset_tool_base_url.strip():
        testset_sync = TestsetSyncService(
            repository,
            TestsetToolClient(
                settings.testset_tool_base_url,
                settings.testset_tool_sync_timeout_seconds,
            ),
        )
    ingestion = IngestionService(
        repository,
        objects,
        vectors,
        DocumentParser(),
        models,
        max_concurrency=settings.ingestion_max_concurrency,
        embedding_max_concurrency=settings.ingestion_embedding_max_concurrency,
        embedding_batch_size=settings.ingestion_embedding_batch_size,
        testset_sync=testset_sync,
    )
    return Services(
        settings=settings,
        repository=repository,
        objects=objects,
        vectors=vectors,
        models=models,
        ingestion=ingestion,
        deletion=DeletionService(repository, objects, vectors, ingestion.cancel),
        rag=RagService(repository, decision_agent, retrieval_agent, answer_agent),
        testset_sync=testset_sync,
    )


def initialize_services(services: Services) -> None:
    checks = (
        ("PostgreSQL", services.repository.initialize),
        ("MinIO", services.objects.ensure_bucket),
        ("Qdrant", services.vectors.check_connection),
        ("模型服务", services.models.check_connection),
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
        if services.testset_sync:
            services.testset_sync.close()
        services.repository.close()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0", lifespan=lifespan)
    cors_origins = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
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
