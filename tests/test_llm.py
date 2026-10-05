from types import SimpleNamespace

from app.llm.anthropic import AnthropicClient, to_anthropic_payload, to_anthropic_tools
from app.llm.openai_compat import OpenAICompatClient, to_openai_messages, to_openai_tools
from app.llm.types import Message, ToolCall, ToolDef

MSGS = [
    Message(role="system", content="你是助手"),
    Message(role="user", content="查一下"),
    Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "x"})],
    ),
    Message(role="tool", tool_call_id="c1", content="结果文本"),
]
TOOLS = [ToolDef(name="search", description="搜索", parameters={"type": "object"})]


def test_openai_mapping():
    out = to_openai_messages(MSGS)
    assert out[0] == {"role": "system", "content": "你是助手"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "search"
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "结果文本"}
    assert to_openai_tools(TOOLS)[0]["function"]["name"] == "search"


def test_anthropic_mapping():
    system, msgs = to_anthropic_payload(MSGS)
    assert system == "你是助手"
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert to_anthropic_tools(TOOLS)[0]["input_schema"] == {"type": "object"}


def _openai_response():
    tc = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="search", arguments='{"q": "x"}'),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="好的", tool_calls=[tc])
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


async def test_openai_chat():
    client = OpenAICompatClient(api_key="k", model="m")

    async def fake_create(**kwargs):
        return _openai_response()

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    result = await client.chat(MSGS, TOOLS)
    assert result.text == "好的"
    assert result.tool_calls[0].name == "search"
    assert result.usage.tokens_in == 10


async def test_openai_stream_aggregates_tool_call():
    client = OpenAICompatClient(api_key="k", model="m")

    def chunk(delta):
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)

    async def fake_create(**kwargs):
        assert kwargs["stream"] is True

        async def gen():
            deltas = [
                chunk(SimpleNamespace(content="你", tool_calls=None)),
                chunk(SimpleNamespace(content="好", tool_calls=None)),
                chunk(
                    SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="c1",
                                function=SimpleNamespace(name="search", arguments='{"q"'),
                            )
                        ],
                    )
                ),
                chunk(
                    SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments=': "x"}'),
                            )
                        ],
                    )
                ),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                ),
            ]
            for d in deltas:
                yield d

        return gen()

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    chunks = [c async for c in client.chat_stream(MSGS, TOOLS)]
    assert "".join(c.text_delta for c in chunks) == "你好"
    final = chunks[-1]
    assert final.finish and final.tool_calls[0].name == "search"
    assert final.tool_calls[0].arguments == {"q": "x"}
    assert final.usage.tokens_out == 2


def _anthropic_response():
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="好的"),
            SimpleNamespace(type="tool_use", id="c1", name="search", input={"q": "x"}),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


async def test_anthropic_chat():
    client = AnthropicClient(api_key="k", model="m")

    async def fake_create(**kwargs):
        assert kwargs["system"] == "你是助手"
        assert kwargs["tools"][0]["input_schema"] == {"type": "object"}
        return _anthropic_response()

    client.client = SimpleNamespace(
        messages=SimpleNamespace(create=fake_create)
    )
    result = await client.chat(MSGS, TOOLS)
    assert result.text == "好的"
    assert result.tool_calls[0].arguments == {"q": "x"}
    assert result.usage.tokens_out == 5


async def test_openai_effort_maps_to_reasoning_effort():
    """low/high → reasoning_effort；max 这边没有更高档，映射到 high；off/缺省不传。"""
    for effort, want in [("low", "low"), ("high", "high"), ("max", "high"), ("off", None), (None, None)]:
        client = OpenAICompatClient(api_key="k", model="m", effort=effort)
        seen: dict = {}

        async def fake_create(**kwargs):
            seen.update(kwargs)
            return _openai_response()

        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )
        await client.chat(MSGS)
        assert seen.get("reasoning_effort") == want, f"effort={effort}"


async def test_anthropic_effort_maps_to_thinking_budget():
    """low/high/max → thinking.budget_tokens，并把 max_tokens 抬到 budget 之上；off/缺省不开 thinking。"""
    cases = [("low", 2048), ("high", 8192), ("max", 16384), ("off", None), (None, None)]
    for effort, budget in cases:
        client = AnthropicClient(api_key="k", model="m", effort=effort)
        seen: dict = {}

        async def fake_create(**kwargs):
            seen.update(kwargs)
            return _anthropic_response()

        client.client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
        await client.chat(MSGS)
        if budget is None:
            assert "thinking" not in seen, f"effort={effort}"
        else:
            assert seen["thinking"] == {"type": "enabled", "budget_tokens": budget}, f"effort={effort}"
            assert seen["max_tokens"] > budget, f"effort={effort}"


def test_anthropic_merges_parallel_tool_results():
    msgs = [
        Message(role="user", content="查两个"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="c1", name="search", arguments={"q": "a"}),
                ToolCall(id="c2", name="search", arguments={"q": "b"}),
            ],
        ),
        Message(role="tool", tool_call_id="c1", content="结果A"),
        Message(role="tool", tool_call_id="c2", content="结果B"),
    ]
    _, out = to_anthropic_payload(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert len(out[2]["content"]) == 2
    assert all(b["type"] == "tool_result" for b in out[2]["content"])


async def test_openai_malformed_arguments_do_not_crash():
    client = OpenAICompatClient(api_key="k", model="m")
    tc = SimpleNamespace(
        id=None, function=SimpleNamespace(name=None, arguments="[1,2]")
    )

    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))
            ],
            usage=None,
        )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    result = await client.chat(MSGS)
    assert result.tool_calls[0].id == "call_0"
    assert result.tool_calls[0].arguments == {"_raw": [1, 2]}


async def test_openai_stream_without_index():
    client = OpenAICompatClient(api_key="k", model="m")

    def tc_delta(idx, name, args):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=idx,
                                id=None,
                                function=SimpleNamespace(name=name, arguments=args),
                            )
                        ],
                    )
                )
            ],
            usage=None,
        )

    async def fake_create(**kwargs):
        async def gen():
            # 两个并列调用，端点省略 index
            yield tc_delta(None, "a", '{"x":1}')
            yield tc_delta(None, "b", '{"y":2}')

        return gen()

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    chunks = [c async for c in client.chat_stream(MSGS)]
    final = chunks[-1]
    names = [t.name for t in final.tool_calls]
    assert names == ["a", "b"]


async def test_anthropic_stream():
    client = AnthropicClient(api_key="k", model="m")

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        @property
        def text_stream(self):
            async def gen():
                yield "你"
                yield "好"

            return gen()

        async def get_final_message(self):
            return _anthropic_response()

    def fake_stream(**kwargs):
        assert kwargs["system"] == "你是助手"
        return FakeStream()

    client.client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))
    chunks = [c async for c in client.chat_stream(MSGS, TOOLS)]
    assert "".join(c.text_delta for c in chunks) == "你好"
    final = chunks[-1]
    assert final.finish and final.tool_calls[0].name == "search"
    assert final.usage.tokens_in == 10


def test_clients_have_timeout_and_limited_retries():
    """provider 卡住时应在分钟级报错（connect 10s / read 120s），重试只给 1 次。"""
    c = OpenAICompatClient(api_key="k", model="m")
    assert c.client.timeout.connect == 10.0
    assert c.client.timeout.read == 120.0
    assert c.client.max_retries == 1
    a = AnthropicClient(api_key="k", model="m")
    assert float(a.client.timeout) == 120.0
    assert a.client.max_retries == 1
