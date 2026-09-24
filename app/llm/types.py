from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Optional, Protocol

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


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
