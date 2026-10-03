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
from app.tracing import record_llm


def detect_provider(base_url: str | None) -> str:
    """按 base_url 认出后端，用于 traces 的定价档（DeepSeek 与 OpenAI 单价不同）。

    认不出的兼容端点（通义 / Moonshot / 自建）按 openai 档估算：定价表只备了
    OpenAI / DeepSeek / Claude 三档，多认一档就得凭空多编一组数字。
    """
    return "deepseek" if "deepseek" in (base_url or "").lower() else "openai"


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


def _make_tool_call(raw_id: Any, name: Any, raw_args: Any, fallback_id: str) -> ToolCall:
    try:
        args = json.loads(raw_args or "{}")
    except (json.JSONDecodeError, TypeError):
        args = {"_raw": raw_args}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return ToolCall(
        id=raw_id or fallback_id, name=name or "unknown", arguments=args
    )


def _parse_tool_calls(raw: list[Any]) -> list[ToolCall]:
    return [
        _make_tool_call(tc.id, tc.function.name, tc.function.arguments, f"call_{i}")
        for i, tc in enumerate(raw)
    ]


class OpenAICompatClient:
    """OpenAI Chat Completions 协议；改 base_url 即可接入 DeepSeek/通义等。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        db_path: str | None = None,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.provider = detect_provider(base_url)
        # 埋点落哪个库由**创建者**决定：用自定义库时传进来，HTTP 主流程不传，
        # record_llm 拿到 None 后由 get_db 兜到默认库（见 app/tracing.record_llm）
        self.db_path = db_path

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
        record_llm(self.provider, self.model, usage, self.db_path)
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
                # 部分兼容端点省略 index：无 name 的分片视为上一个调用的续传
                if tc.index is not None:
                    idx = tc.index
                elif tc.function and tc.function.name:
                    idx = (max(pending) + 1) if pending else 0
                else:
                    idx = (max(pending) if pending else 0)
                slot = pending.setdefault(idx, {"id": "", "name": "", "args": "", "seq": len(pending)})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name and not slot["name"]:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["args"] += tc.function.arguments
        calls = [
            _make_tool_call(slot["id"], slot["name"], slot["args"], f"call_{i}")
            for i, slot in enumerate(
                sorted(pending.values(), key=lambda s: s["seq"])
            )
        ]
        yield StreamChunk(finish=True, tool_calls=calls, usage=usage)
        # 埋点放在最后一个 chunk 之后：usage 只在收尾 chunk 出现，这里才是「调用完成」。
        # 消费者中途弃用生成器（客户端断开）时这段不会执行，该次调用不计入——拿不到
        # usage 就不编造 token 数
        record_llm(self.provider, self.model, usage, self.db_path)
