from openai import AsyncOpenAI

from app.config import settings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """OpenAI 兼容 embeddings 接口（通义/OpenAI 等均可配）。"""
    client = AsyncOpenAI(
        api_key=settings.embed_api_key,
        base_url=settings.embed_base_url,
        timeout=60.0,
        max_retries=1,
    )
    resp = await client.embeddings.create(model=settings.embed_model, input=texts)
    return [item.embedding for item in resp.data]
