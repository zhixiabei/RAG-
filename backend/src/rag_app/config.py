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
    rag_retrieval_top_k: int = 20
    rag_context_top_k: int = 8
    rag_relevance_threshold: float = 0.65
