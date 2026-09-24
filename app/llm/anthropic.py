from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from app.llm.types import (
    ChatResult,
    Message,
    StreamChunk,
    ToolCall,
    ToolDef,
    Usage,
)

DEFAULT_MAX_TOKENS = 4096


def to_anthropic_payload(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """返回 (system, messages)。Anthropic 的 system 是独立参数，tool 结果
    包在 user 消息的 tool_result 块里。"""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        # Anthropic 要求同轮的多个 tool_result 合并进一条 user 消息
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
            )
        else:
            flush_tool_results()
            if m.role == "assistant" and m.tool_calls:
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                content += [
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in m.tool_calls
                ]
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": m.role, "content": m.content})
    flush_tool_results()
    return ("\n".join(system_parts) or None), out


def to_anthropic_tools(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def _parse_content(blocks: list[Any]) -> tuple[str, list[ToolCall]]:
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for b in blocks:
        if b.type == "text":
            text_parts.append(b.text)
        elif b.type == "tool_use":
            args = b.input if isinstance(b.input, dict) else {"_raw": b.input}
            calls.append(ToolCall(id=b.id, name=b.name, arguments=args))
    return "".join(text_parts), calls


class AnthropicClient:
    """Anthropic Messages 协议适配。"""

    def __init__(self, api_key: str, model: str, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> ChatResult:
        system, msgs = to_anthropic_payload(messages)
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        resp = await self.client.messages.create(model=self.model, messages=msgs, **kwargs)
        text, calls = _parse_content(resp.content)
        return ChatResult(
            text=text,
            tool_calls=calls,
            usage=Usage(
                tokens_in=resp.usage.input_tokens,
                tokens_out=resp.usage.output_tokens,
            ),
        )

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> AsyncIterator[StreamChunk]:
        system, msgs = to_anthropic_payload(messages)
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        async with self.client.messages.stream(
            model=self.model, messages=msgs, **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(text_delta=text)
            final = await stream.get_final_message()
        text, calls = _parse_content(final.content)
        yield StreamChunk(
            finish=True,
            tool_calls=calls,
            usage=Usage(
                tokens_in=final.usage.input_tokens,
                tokens_out=final.usage.output_tokens,
            ),
        )
