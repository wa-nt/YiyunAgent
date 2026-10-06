from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Protocol

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

# 思考强度：可移植的四档枚举，在各家客户端层映射成具体 API 参数
# （OpenAI 兼容 → reasoning_effort，Anthropic → thinking.budget_tokens）。
# off = 不思考（不传任何参数）；NULL/缺省 = 跟随供应商默认。
Effort = Literal["off", "low", "high", "max"]


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ToolDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    # prompt 缓存命中的输入 token：OpenAI 的 prompt_tokens_details.cached_tokens、
    # Anthropic 的 cache_read_input_tokens。各家从 input 里单列出来，用于看板算命中率
    cached_tokens: int = 0


def cached_from_openai(usage: Any) -> int:
    """从 OpenAI 兼容响应的 usage 里读缓存命中数；老端点没有这个字段就回 0。"""
    details = getattr(usage, "prompt_tokens_details", None)
    return int(getattr(details, "cached_tokens", 0) or 0)


def cached_from_anthropic(usage: Any) -> int:
    """从 Anthropic 响应的 usage 里读缓存命中数（cache_read_input_tokens）。"""
    return int(getattr(usage, "cache_read_input_tokens", 0) or 0)


class ChatResult(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class StreamChunk(BaseModel):
    text_delta: str = ""
    finish: bool = False
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Optional[Usage] = None


class LLMClient(Protocol):
    """统一 LLM 接口。下层适配 OpenAI Chat Completions 或 Anthropic Messages。"""

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> ChatResult: ...

    def chat_stream(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> AsyncIterator[StreamChunk]:
        """流式输出。最后一个 chunk finish=True，携带聚合后的 tool_calls 与 usage。"""
        ...
