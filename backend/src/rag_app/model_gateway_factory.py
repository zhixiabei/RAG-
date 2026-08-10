from .config import Settings
from .infrastructure.ollama.gateway import OllamaGateway
from .infrastructure.openai_compatible.gateway import OpenAICompatibleGateway


ModelGateway = OllamaGateway | OpenAICompatibleGateway


def build_model_gateway(settings: Settings) -> ModelGateway:
    if settings.model_mode == "local":
        return OllamaGateway(
            settings.ollama_url,
            settings.ollama_chat_model,
            settings.ollama_embedding_model,
        )
    if settings.model_mode == "remote":
        return OpenAICompatibleGateway(
            settings.remote_llm_provider_name,
            settings.remote_llm_base_url,
            settings.remote_llm_api_key,
            [model.strip() for model in settings.remote_llm_models.split(",") if model.strip()],
            settings.remote_default_chat_model,
            settings.remote_embedding_provider_name,
            settings.remote_embedding_base_url,
            settings.remote_embedding_api_key,
            settings.remote_embedding_model,
        )
    raise ValueError("MODEL_MODE 只能是 local 或 remote")


def build_context_compression_gateway(settings: Settings) -> OllamaGateway | None:
    if not settings.rag_context_compression_enabled:
        return None
    return OllamaGateway(
        settings.ollama_url,
        settings.rag_context_compression_model,
        settings.ollama_embedding_model,
    )
