from types import SimpleNamespace

import pytest

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
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_openai_response)
        )
    )

    async def fake_create(**kwargs):
        return _openai_response()

    client.client.chat.completions.create = fake_create
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
                            function=SimpleNamespace(name="sea", arguments='{"q"'),
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
