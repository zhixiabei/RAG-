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
    rag_top_k: int = 10
    rag_context_max_input_tokens: int = 0
    rag_context_output_reserve_tokens: int = 4_096
    rag_context_history_max_tokens: int = 6_000
    rag_context_catalog_max_tokens: int = 3_000
    rag_context_attachment_max_tokens: int = 10_000
    rag_context_compression_enabled: bool = True
    rag_context_compression_model: str = "qwen3:4b"
    rag_context_compression_input_tokens: int = 6_000
    rag_context_compression_output_tokens: int = 1_000
    testset_tool_base_url: str = ""
    testset_tool_sync_timeout_seconds: float = 60.0
    testset_generator_provider_name: str = "Testset Generator"
    testset_generator_base_url: str = ""
    testset_generator_api_key: str = ""
    testset_generator_model: str = ""
    testset_generator_timeout_seconds: float = 90.0
    owner_id: str = "personal"
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
