from app.config import settings
from app.llm.types import LLMClient


def get_llm(
    db_path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """按配置建一个 LLM 客户端；db_path 决定这一轮 llm trace 落哪个库。

    默认 None = 落默认库 settings.db_path，HTTP 主流程用这条；只有**用自定义库**
    的调用方（评测 / CLI）才把自己的库传进来，客户端埋点时再透传给 record_llm。

    provider 非空时覆盖全局 settings.llm_provider（per-session 供应商切换）；
    model 非空时覆盖该供应商的全局默认模型（per-session 模型切换）；
    密钥/base_url 仍取该供应商的全局配置——会话级不另存密钥。
    """
    if (provider or settings.llm_provider) == "anthropic":
        from app.llm.anthropic import AnthropicClient

        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=model or settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            db_path=db_path,
        )
    from app.llm.openai_compat import OpenAICompatClient

    return OpenAICompatClient(
        api_key=settings.openai_api_key,
        model=model or settings.openai_model,
        base_url=settings.openai_base_url,
        db_path=db_path,
    )