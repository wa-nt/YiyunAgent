from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from app.llm.types import (
    ChatResult,
    Effort,
    Message,
    StreamChunk,
    ToolCall,
    ToolDef,
    Usage,
)
from app.tracing import record_llm

DEFAULT_MAX_TOKENS = 4096

# effort → thinking.budget_tokens。off/缺省 = 不开启 thinking；max 给真有的 headroom。
_EFFORT_BUDGET = {"low": 2048, "high": 8192, "max": 16384}


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
            args = b.input if isinstance(b.input, dict) else ({} if b.input is None else {"_raw": b.input})
            calls.append(
                ToolCall(
                    id=b.id or f"call_{len(calls)}",
                    name=b.name or "unknown",
                    arguments=args,
                )
            )
    return "".join(text_parts), calls


class AnthropicClient:
    """Anthropic Messages 协议适配。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        db_path: str | None = None,
        effort: Effort | None = None,
    ):
        # 同 openai_compat：120s 超时 + 只重试 1 次。
        # ponytail: anthropic 1.8 用 httpx2 不认 httpx.Timeout，直接传秒数
        self.client = AsyncAnthropic(api_key=api_key, timeout=120.0, max_retries=1)
        self.model = model
        self.max_tokens = max_tokens
        self.provider = "anthropic"
        # 同 openai_compat：埋点落哪个库由创建者决定，None 时 record_llm 兜到默认库
        self.db_path = db_path
        self.effort = effort

    def _effort_kwarg(self) -> dict[str, Any]:
        """effort → thinking 配置。开了 thinking 后 max_tokens 必须大于 budget_tokens，
        所以按档位把 max_tokens 抬到 budget + 一份回答余量（不足才抬，不压低原配置）。"""
        budget = _EFFORT_BUDGET.get(self.effort or "")
        if budget is None:
            return {}
        return {
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "max_tokens": max(self.max_tokens, budget + 2048),
        }

    async def chat(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> ChatResult:
        system, msgs = to_anthropic_payload(messages)
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens, **self._effort_kwarg()}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        resp = await self.client.messages.create(model=self.model, messages=msgs, **kwargs)
        text, calls = _parse_content(resp.content)
        usage = Usage(
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
        )
        record_llm(self.provider, self.model, usage, self.db_path)
        return ChatResult(text=text, tool_calls=calls, usage=usage)

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> AsyncIterator[StreamChunk]:
        system, msgs = to_anthropic_payload(messages)
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens, **self._effort_kwarg()}
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
        usage = Usage(
            tokens_in=final.usage.input_tokens,
            tokens_out=final.usage.output_tokens,
        )
        yield StreamChunk(finish=True, tool_calls=calls, usage=usage)
        # 埋点在最后一个 chunk 之后：到这里调用才算完成，中途弃用生成器时不计
        # （同 openai_compat.chat_stream）
        record_llm(self.provider, self.model, usage, self.db_path)
