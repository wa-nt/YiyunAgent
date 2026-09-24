from app.config import settings
from app.llm.types import LLMClient


def get_llm() -> LLMClient:
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic import AnthropicClient

        return AnthropicClient(
            api_key=settings.anthropic_api_key, model=settings.anthropic_model
        )
    from app.llm.openai_compat import OpenAICompatClient

    return OpenAICompatClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
    )
