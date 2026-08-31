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
            request_timeout_seconds=settings.remote_llm_timeout_seconds,
            max_transient_retries=settings.remote_llm_max_retries,
            embedding_max_retries=settings.remote_embedding_max_retries,
            retry_base_delay_seconds=settings.remote_llm_retry_base_delay_seconds,
            retry_max_delay_seconds=settings.remote_llm_retry_max_delay_seconds,
        )
    raise ValueError("MODEL_MODE 只能是 local 或 remote")


def build_decision_gateway(
    settings: Settings,
    default_gateway: ModelGateway,
) -> ModelGateway:
    """Build the lightweight local gateway used for retrieval decisions.

    The answer model may be remote, but intent classification must stay on its
    own small Ollama model so it never reuses the main generation model.
    """
    decision_model = settings.rag_decision_model.strip()
    if not decision_model:
        raise ValueError("RAG_DECISION_MODEL 不能为空")
    if decision_model == getattr(default_gateway, "chat_model", None):
        raise ValueError("RAG_DECISION_MODEL 不能使用主聊天模型")
    return OllamaGateway(
        settings.ollama_url,
        decision_model,
        settings.ollama_embedding_model,
        keep_alive=settings.rag_decision_keep_alive,
    )


def build_context_compression_gateway(settings: Settings) -> OllamaGateway | None:
    if not settings.rag_context_compression_enabled:
        return None
    return OllamaGateway(
        settings.ollama_url,
        settings.rag_context_compression_model,
        settings.ollama_embedding_model,
    )


def build_judge_gateway(
    settings: Settings,
    default_gateway: ModelGateway,
) -> ModelGateway | None:
    if not settings.rag_judge_enabled:
        return None

    judge_model = settings.rag_judge_model.strip()
    judge_base_url = settings.rag_judge_base_url.strip()
    if judge_base_url:
        if not judge_model:
            raise ValueError("配置 RAG_JUDGE_BASE_URL 时必须同时配置 RAG_JUDGE_MODEL")
        return OpenAICompatibleGateway(
            settings.rag_judge_provider_name.strip() or "Judge",
            judge_base_url,
            settings.rag_judge_api_key,
            [judge_model],
            judge_model,
            "",
            "",
            "",
            "",
            request_timeout_seconds=settings.rag_judge_timeout_seconds,
            max_transient_retries=settings.rag_judge_max_retries,
            retry_base_delay_seconds=settings.rag_judge_retry_base_delay_seconds,
            retry_max_delay_seconds=settings.rag_judge_retry_max_delay_seconds,
        )

    if not judge_model or judge_model == default_gateway.chat_model:
        return default_gateway

    if settings.model_mode == "local":
        return OllamaGateway(
            settings.ollama_url,
            judge_model,
            settings.ollama_embedding_model,
        )

    models = [
        model.strip()
        for model in settings.remote_llm_models.split(",")
        if model.strip()
    ]
    if judge_model not in models:
        models.append(judge_model)
    return OpenAICompatibleGateway(
        settings.remote_llm_provider_name,
        settings.remote_llm_base_url,
        settings.remote_llm_api_key,
        models,
        judge_model,
        settings.remote_embedding_provider_name,
        settings.remote_embedding_base_url,
        settings.remote_embedding_api_key,
        settings.remote_embedding_model,
        request_timeout_seconds=settings.rag_judge_timeout_seconds,
        max_transient_retries=settings.rag_judge_max_retries,
        retry_base_delay_seconds=settings.rag_judge_retry_base_delay_seconds,
        retry_max_delay_seconds=settings.rag_judge_retry_max_delay_seconds,
    )
