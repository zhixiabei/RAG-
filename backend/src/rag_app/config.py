from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = 8080
    database_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool = False
    minio_bucket: str = "rag-documents"
    qdrant_url: str
    qdrant_collection: str
    qdrant_timeout_seconds: float = 30.0
    qdrant_upsert_batch_size: int = 32
    qdrant_upsert_max_retries: int = 2
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 128
    qdrant_hnsw_full_scan_threshold: int = 10_000
    qdrant_search_hnsw_ef: int = 64
    ingestion_max_concurrency: int = 2
    ingestion_embedding_max_concurrency: int = 1
    ingestion_embedding_batch_size: int = 32
    max_document_bytes: int = 0
    model_mode: str = "local"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3:4b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    remote_llm_provider_name: str = "DeepSeek"
    remote_llm_base_url: str = "https://api.deepseek.com"
    remote_llm_api_key: str = ""
    remote_llm_models: str = "deepseek-chat,deepseek-reasoner"
    remote_default_chat_model: str = "deepseek-chat"
    remote_embedding_provider_name: str = ""
    remote_embedding_base_url: str = ""
    remote_embedding_api_key: str = ""
    remote_embedding_model: str = ""
    rag_retrieval_top_k: int = 100
    rag_context_top_k: int = 50
    rag_relevance_threshold: float = 0.65
    rag_context_max_chars: int = 30_000
    auth_username: str = "admin"
    auth_password: str = "admin"
    auth_secret: str = "local-development-secret-change-me"
    auth_owner_id: str = "personal"
    auth_session_ttl_seconds: int = 30 * 24 * 60 * 60
    auth_cookie_secure: bool = False
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
