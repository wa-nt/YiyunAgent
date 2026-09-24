import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from app.llm.types import (
    ChatResult,
    Message,
    StreamChunk,
    ToolCall,
    ToolDef,
    Usage,
)


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
            )
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def to_openai_tools(tools: list[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_tool_calls(raw: list[Any]) -> list[ToolCall]:
    calls = []
    for tc in raw:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return calls


class OpenAICompatClient:
    """OpenAI Chat Completions 协议；改 base_url 即可接入 DeepSeek/通义等。"""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> ChatResult:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        resp = await self.client.chat.completions.create(
            model=self.model, messages=to_openai_messages(messages), **kwargs
        )
        msg = resp.choices[0].message
        usage = Usage(
            tokens_in=resp.usage.prompt_tokens if resp.usage else 0,
            tokens_out=resp.usage.completion_tokens if resp.usage else 0,
        )
        return ChatResult(
            text=msg.content or "",
            tool_calls=_parse_tool_calls(msg.tool_calls or []),
            usage=usage,
        )

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict[str, Any] = {"stream_options": {"include_usage": True}}
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=to_openai_messages(messages),
            stream=True,
            **kwargs,
        )
        # 流式 tool_calls 按 index 分片到达，聚合后在收尾 chunk 统一返回
        pending: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        async for event in stream:
            if event.usage:
                usage = Usage(
                    tokens_in=event.usage.prompt_tokens,
                    tokens_out=event.usage.completion_tokens,
                )
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if delta.content:
                yield StreamChunk(text_delta=delta.content)
            for tc in delta.tool_calls or []:
                slot = pending.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["args"] += tc.function.arguments
        calls = []
        for slot in pending.values():
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": slot["args"]}
            calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))
        yield StreamChunk(finish=True, tool_calls=calls, usage=usage)
