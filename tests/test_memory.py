import asyncio
import json

import pytest

from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ChatResult, StreamChunk
from app.memory import recall as recall_module
from app.memory import writer
from app.memory.recall import recall_memories
from app.memory.writer import (
    CONFLICT_PROMPT,
    EXTRACT_PROMPT,
    estimate_importance,
    extract_and_store,
    normalize,
)

DIM = 8
SESSION = "s-mem"


def extraction(*items: tuple[str, str, int | None]) -> str:
    """构造 LLM 的抽取输出（importance 为 None 时省略该字段，走规则兜底）。"""
    payload = []
    for kind, content, importance in items:
        entry: dict = {"kind": kind, "content": content}
        if importance is not None:
            entry["importance"] = importance
        payload.append(entry)
    return json.dumps(payload, ensure_ascii=False)


class MemoryLLM:
    """抽取请求返回脚本里的记忆条目，冲突判定请求按脚本返回矛盾 id 列表。

    两类请求靠 system prompt 区分，因此同一实例可同时服务两条链路。
    """

    def __init__(
        self,
        items: list[tuple[str, str, int | None]] | None = None,
        conflicts: list[list[int]] | None = None,
        raw_extraction: str | None = None,
    ):
        self.items = items or []
        self.raw_extraction = raw_extraction
        self.conflict_script = list(conflicts or [])
        self.chats: list[list] = []
        self.conflict_prompts: list[str] = []

    async def chat(self, messages, tools=None):
        self.chats.append(list(messages))
        if messages[0].content == EXTRACT_PROMPT:
            text = self.raw_extraction
            return ChatResult(text=text if text is not None else extraction(*self.items))
        self.conflict_prompts.append(messages[1].content)
        ids = self.conflict_script.pop(0) if self.conflict_script else []
        return ChatResult(text=json.dumps({"conflicts": ids}))


class StreamLLM:
    """run_agent 用的流式桩：直接吐一段回答，不发工具调用。"""

    def __init__(self, text: str = "好的，记下了"):
        self.text = text
        self.calls: list[list] = []

    async def chat_stream(self, messages, tools=None):
        self.calls.append(list(messages))
        yield StreamChunk(text_delta=self.text)
        yield StreamChunk(finish=True, tool_calls=[])


class GatedMemoryLLM(MemoryLLM):
    """抽取请求卡在 gate 上，用来验证记忆写入不阻塞 SSE 流。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = asyncio.Event()

    async def chat(self, messages, tools=None):
        if messages[0].content == EXTRACT_PROMPT:
            await self.gate.wait()
        return await super().chat(messages, tools)


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(settings, "memory_enabled", True)
    monkeypatch.setattr(settings, "memory_recall_top_k", 5)
    monkeypatch.setattr(settings, "memory_dedup_ratio", 0.92)
    monkeypatch.setattr(settings, "memory_decay", 0.9)
    path = tmp_path / "app.db"
    await init_db(path, DIM)
    yield path
    await runtime.drain_memory_writes()


def use_writer_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(writer, "get_llm", lambda: llm)


async def memories(db, session_id: str | None = None) -> list[dict]:
    sql = "SELECT * FROM memories"
    params: tuple = ()
    if session_id is not None:
        sql += " WHERE source = ?"
        params = (session_id,)
    async with get_db(db) as conn:
        rows = await conn.execute_fetchall(sql + " ORDER BY id", params)
    return [dict(r) for r in rows]


async def insert_memory(
    db,
    kind: str,
    content: str,
    confidence: float,
    status: str = "active",
    source: str = SESSION,
) -> int:
    async with get_db(db) as conn:
        cursor = await conn.execute(
            "INSERT INTO memories "
            "(kind, content, confidence, source, created_at, updated_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, content, confidence, source, "2026-09-24T10:00:00", "2026-09-24T10:00:00", status),
        )
        await conn.commit()
    return cursor.lastrowid


# ---------- 写入侧 ----------


async def test_extract_and_store_writes_structured_memories(db, monkeypatch):
    llm = MemoryLLM(
        [
            ("preference", "用户偏好用 Markdown 记笔记", 4),
            ("goal", "用户想在三个月内拿到大模型实习", 5),
        ]
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记，三个月内想找实习", "好的", db)

    rows = await memories(db, SESSION)
    assert [(r["kind"], r["content"]) for r in rows] == [
        ("preference", "用户偏好用 Markdown 记笔记"),
        ("goal", "用户想在三个月内拿到大模型实习"),
    ]
    assert [r["confidence"] for r in rows] == [0.8, 1.0]
    assert {r["status"] for r in rows} == {"active"}
    assert {r["supersedes"] for r in rows} == {None}
    assert all(r["created_at"] and r["updated_at"] for r in rows)

    # 抽取 prompt 里带上了这轮的用户消息与助手回答
    prompt = llm.chats[0][1].content
    assert "我喜欢用 Markdown 记笔记" in prompt and "好的" in prompt


async def test_extract_and_store_skips_all_work_when_disabled(db, monkeypatch):
    monkeypatch.setattr(settings, "memory_enabled", False)
    llm = MemoryLLM([("fact", "用户在用 Python 3.12", 3)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我用 Python 3.12", "好的", db)

    assert await memories(db) == []
    assert llm.chats == []


async def test_extract_and_store_without_candidates_writes_nothing(db, monkeypatch):
    llm = MemoryLLM([])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "今天天气不错", "是的", db)

    assert llm.chats  # 抽取发生过，只是没有候选
    assert await memories(db) == []


async def test_invalid_or_fenced_extraction_output(db, monkeypatch):
    llm = MemoryLLM(raw_extraction="（抱歉，我不确定该怎么抽取）")
    use_writer_llm(monkeypatch, llm)
    await extract_and_store(SESSION, "随便说说", "嗯", db)
    assert await memories(db) == []

    fenced = MemoryLLM(
        raw_extraction="```json\n"
        + extraction(("preference", "用户偏好清晨写代码", 4))
        + "\n```"
    )
    use_writer_llm(monkeypatch, fenced)
    await extract_and_store(SESSION, "我喜欢早起写代码", "好的", db)
    assert [r["content"] for r in await memories(db)] == ["用户偏好清晨写代码"]


async def test_unknown_kind_falls_back_to_fact_and_rule_scores_importance(db, monkeypatch):
    llm = MemoryLLM(
        raw_extraction=json.dumps(
            [{"kind": "hobby", "content": "用户养了一只猫叫咪咪"}], ensure_ascii=False
        )
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我养了只猫", "哦", db)

    rows = await memories(db)
    assert rows[0]["kind"] == "fact"
    # 规则兜底：基准 2，无数字/偏好词且短于 50 字
    assert rows[0]["confidence"] == 2 / 5


async def test_kind_is_lowercased_and_non_string_content_dropped(db, monkeypatch):
    llm = MemoryLLM(
        raw_extraction=json.dumps(
            [
                {"kind": "Fact", "content": "用户在准备大模型实习面试", "importance": 3},
                {"kind": "fact", "content": 12345, "importance": 5},
                {"kind": "fact", "content": None, "importance": 5},
            ],
            ensure_ascii=False,
        )
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我在准备实习", "好的", db)

    rows = await memories(db)
    assert len(rows) == 1
    assert rows[0]["kind"] == "fact"
    assert rows[0]["content"] == "用户在准备大模型实习面试"


def test_estimate_importance_rule():
    assert estimate_importance("嗯") == 2
    assert estimate_importance("我喜欢用 vim") == 4  # 偏好 +1、拉丁词 +1
    assert estimate_importance("用户偏好用 vim") == 4  # 第三人称同样命中偏好词
    assert estimate_importance("我需要在 2026 年 3 月前投 20 份简历") == 4  # 数字 +1、偏好 +1
    long_plain = "这是一段只用来说明篇幅的中文记忆内容" * 4  # 80 字，无数字与偏好词
    assert len(long_plain) > 50 and estimate_importance(long_plain) == 3
    assert estimate_importance("我讨厌在 2026 年 3 月前用 vim 写 " + "很长" * 30) == 5


def test_estimate_importance_latin_signal_skips_stopwords():
    assert estimate_importance("the for and") == 2  # 停用词不算专有名词信号
    assert estimate_importance("用户在用 not") == 2
    assert estimate_importance("用户在用 asyncio") == 3


def test_estimate_importance_short_latin_whitelist():
    """MIN-7：两字母技术名（AI/Go/C++ 等）达不到 ≥3 的长度门槛，靠白名单补回。"""
    for content in ["用户在用 AI", "用户在用 Go", "用户写 C++", "用户在调 ML 模型"]:
        assert estimate_importance(content) == 3, content
    assert estimate_importance("用户在用 the") == 2  # 短停用词仍不算


def test_preference_hints_cover_negative_and_third_person():
    """MIN-7：否定式偏好（不喜欢/不想/不需要/不习惯）同样算偏好信号。"""
    for content in [
        "用户不喜欢吃辣",
        "用户不爱吃辣",
        "用户不想学 Rust",
        "用户不需要背书",
        "用户不习惯早起",
        "我喜欢用 Markdown",
    ]:
        assert writer._PREFERENCE_HINTS.search(content), content
        assert estimate_importance(content) >= 3, content


def test_normalize_and_similarity_scale():
    assert normalize("用户喜欢用 Markdown 记笔记") == "喜欢用Markdown记笔记"
    assert normalize("我讨厌用　Markdown，记笔记。") == "讨厌用Markdown记笔记"
    assert normalize("用户的项目") == "项目"  # 前缀后的「的」一并剥离
    # 标点/空白差异被归一化抹平，视为同一句
    assert writer._similarity("用户偏好用 Markdown 记笔记", "用户偏好用 Markdown 记笔记。") == 1.0
    # 一词之差改变事实：必须落在去重阈值之下（阈值取自配置，不硬编码，
    # 否则把阈值调回有缺陷的旧值也不会有用例变红）
    ratio = settings.memory_dedup_ratio
    assert writer._similarity("用户在北京上学", "用户在北京上班") < ratio
    assert writer._similarity("用户在北京上学", "用户在北京上班") < 0.92


def test_decimal_point_is_not_stripped():
    """MIN-3：把小数点当噪声删掉会把「1.5 公斤」与「15 公斤」归一成同一句。"""
    assert normalize("用户买了 1.5 公斤牛肉") == "买了1.5公斤牛肉"
    assert normalize("用户买了 15 公斤牛肉") == "买了15公斤牛肉"
    assert not writer.same_facts("用户买了 1.5 公斤牛肉", "用户买了 15 公斤牛肉")
    # 非数字相邻的句号仍要删掉
    assert normalize("用户偏好用 Markdown 记笔记。") == "偏好用Markdown记笔记"


def test_numbers_compare_in_order_not_as_multiset():
    """suggestion 1：排序后比较等于只比数字多重集，会漏掉位置互换的事实差异。"""
    assert not writer.same_facts("用户买了 3 个苹果和 5 个梨", "用户买了 5 个苹果和 3 个梨")
    assert writer.same_facts("用户买了 3 个苹果", "用户买了 3 个苹果")


def test_same_polarity_detects_negation_flip():
    """MAJ-3：否定词只出现在一侧时，两句断言的是相反的事。"""
    assert not writer.same_polarity("用户喜欢用 Markdown 记笔记", "用户不喜欢用 Markdown 记笔记")
    assert not writer.same_polarity("用户会用 Python 写爬虫", "用户不会用 Python 写爬虫")
    assert writer.same_polarity("用户偏好用 Markdown 记笔记", "用户偏好用 Markdown 记笔记。")
    # 守卫是保守的表面形式判据：一侧有「不」一侧没有就判不同，哪怕两句其实同义
    # （「不喜欢」vs「讨厌」）。代价只是多存一行，方向安全（同 same_facts 的取舍）
    assert not writer.same_polarity("用户不喜欢吃辣", "用户讨厌吃辣")


# ---------- 敏感信息过滤（M5） ----------


@pytest.mark.parametrize(
    "content",
    [
        "用户的 OpenAI key 是 sk-abcdefgh12345678",
        "用户的 AWS key 是 AKIAIOSFODNN7EXAMPLE",
        "用户的 GitHub token 是 ghp_abcdefghijklmnopqrstuvwxyz012345",
        "请求头里的凭据是 Bearer abcdefghijklmnop1234",
        "用户的数据库密码是 hunter2xyz",
        "用户的 API key: 8f3a9c2b1d",
        "用户的密钥为 abcdef123456",
        "用户的 token=abcdef123456",
        "password=hunter2xyz",
        "用户的凭证为 Abc123!@#xyz",
    ],
)
async def test_secrets_are_never_stored(db, monkeypatch, content):
    """密码/密钥落库后每轮都会回灌进 system 提示，删库才能补救，必须在入口拦掉。"""
    llm = MemoryLLM([("fact", content, 5)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "这段对话里有敏感信息", "好的", db)

    assert await memories(db) == []


async def test_normal_facts_are_not_flagged_as_secrets(db, monkeypatch):
    """过滤不能误伤正常记忆：提到「密码」但没跟具体值的不算泄漏。"""
    llm = MemoryLLM(
        [
            ("fact", "用户在学密码学", 3),
            ("preference", "用户偏好用 Markdown 记笔记", 4),
        ]
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我在学密码学", "好的", db)

    assert len(await memories(db)) == 2


@pytest.mark.parametrize(
    "content",
    [
        "用户在学 tokenizer 的实现原理",
        "用户在学密码学的基础知识",
        "用户偏好用 password manager 管理密码",
        "用户的项目需要 API key 轮换机制",
        "用户在阅读 secret sharing 的论文",
        "用户偏好用 passwordless 登录",
        "用户熟悉的 password 哈希算法是 bcrypt",
    ],
)
async def test_secret_filter_does_not_swallow_normal_memories(db, monkeypatch, content):
    """MAJ-1：过滤用「后面还有字」当判据会把正常记忆整条丢掉，那是静默丢数据。
    判据必须是「值像凭据」（分隔符基本必现 + 值无 CJK 且够长）。"""
    llm = MemoryLLM([("fact", content, 3)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "这段对话是正常内容", "好的", db)

    assert [r["content"] for r in await memories(db)] == [content]


@pytest.mark.parametrize(
    "content",
    [
        "用户的 OpenAI key 是sk-abcdefgh12345678",
        "用户的 AWS key 是AKIAIOSFODNN7EXAMPLE",
        "用户的 GitHub token 是ghp_abcdefghijklmnopqrstuvwxyz012345",
        "请求头里的凭据是Bearer abcdefghijklmnop1234",
        "用户的密钥为abcdef123456",
    ],
)
async def test_secrets_are_caught_when_adjacent_to_cjk(db, monkeypatch, content):
    """MAJ-2：中文也是 \\w，`\\b` 在「是」与「s」之间不成立，中文紧邻的密钥整条漏过。"""
    llm = MemoryLLM([("fact", content, 5)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "这段对话里有敏感信息", "好的", db)

    assert await memories(db) == []


async def test_secret_discard_log_does_not_leak_the_secret(db, monkeypatch, caplog):
    """MIN-1：日志里绝不能出现要拦的那串密钥本身。"""
    secret = "用户的 OpenAI key 是 sk-abcdefgh12345678"
    llm = MemoryLLM([("fact", secret, 5)])
    use_writer_llm(monkeypatch, llm)

    with caplog.at_level("WARNING", logger="app.memory.writer"):
        await extract_and_store(SESSION, "这段对话里有敏感信息", "好的", db)

    assert "命中敏感信息" in caplog.text
    assert "sk-abcdefgh12345678" not in caplog.text
    assert secret not in caplog.text
    assert "掩码" in caplog.text


async def test_dedup_keeps_single_memory_and_prefers_higher_confidence(db, monkeypatch):
    llm = MemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 3)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记", "好的", db)
    # 几乎相同、置信度不更高的候选：直接丢弃，不新增也不改动旧记忆
    await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记。", "好的", db)

    rows = await memories(db)
    assert len(rows) == 1
    assert rows[0]["status"] == "active" and rows[0]["confidence"] == 0.6

    # 置信度更高时走追加式版本链：旧记忆 supersede，历史行仍可查
    llm.items = [("preference", "用户偏好用 Markdown 做笔记", 5)]
    await extract_and_store(SESSION, "我强烈偏好用 Markdown 记笔记", "好的", db)

    rows = await memories(db)
    assert len(rows) == 2
    old, new = rows
    assert old["status"] == "superseded" and old["content"] == "用户偏好用 Markdown 记笔记"
    assert new["status"] == "active" and new["supersedes"] == old["id"]
    assert new["confidence"] == 1.0


async def test_dedup_does_not_swallow_one_word_fact_change(db, monkeypatch):
    """C1：「用户在北京上学」与「用户在北京上班」相似度 0.86，曾被 0.85 的阈值
    判成重复并静默丢弃。一词之差改变事实的句子必须并存（或走冲突），不能消失。"""
    await insert_memory(db, "fact", "用户在北京上学", 0.8)
    llm = MemoryLLM([("fact", "用户在北京上班", 4)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我现在在北京上班了", "好的", db)

    rows = await memories(db)
    assert [r["content"] for r in rows] == ["用户在北京上学", "用户在北京上班"]
    assert [r["status"] for r in rows] == ["active", "active"]
    assert rows[0]["confidence"] == 0.8  # 旧记忆原样保留


async def test_dedup_does_not_swallow_different_facts(db, monkeypatch):
    """措辞相近但事实不同的两条（日期/数字不同）同样不能被当成重复丢掉。"""
    await insert_memory(db, "fact", "用户计划在 2026 年 3 月投简历", 0.8)
    llm = MemoryLLM([("fact", "用户计划在 2026 年 6 月投简历", 4)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "时间改到 6 月了", "好的", db)

    assert len(await memories(db)) == 2


async def test_dedup_logs_when_discarding(db, monkeypatch, caplog):
    """丢弃不再是静默的，且用 WARNING 级——uvicorn 默认下应用 logger 停在 WARNING，
    INFO 级在生产路径根本看不见（MIN-2）。"""
    llm = MemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 3)])
    use_writer_llm(monkeypatch, llm)
    await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记", "好的", db)

    with caplog.at_level("WARNING", logger="app.memory.writer"):
        await extract_and_store(SESSION, "我喜欢用 Markdown 记笔记。", "好的", db)

    assert "候选与已有记忆重复，丢弃" in caplog.text
    assert "用户偏好用 Markdown 记笔记" in caplog.text
    assert caplog.records[-1].levelname == "WARNING"


async def test_conflict_memory_is_not_duplicated_across_rounds(db, monkeypatch):
    """M1：同一条矛盾语句反复出现只应有一行 conflict，旧记忆也不再重复衰减。"""
    old_id = await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.8)
    llm = MemoryLLM(
        [
            ("preference", "用户讨厌用 Markdown 记笔记", 4),
            ("preference", "用户讨厌用 Markdown 记笔记", 4),
        ],
        conflicts=[[old_id], [old_id]],
    )
    use_writer_llm(monkeypatch, llm)

    # 同一批候选里两条相同的矛盾语句
    await extract_and_store(SESSION, "我讨厌 Markdown", "好的", db)

    rows = await memories(db)
    conflicts = [r for r in rows if r["status"] == "conflict"]
    assert len(conflicts) == 1
    # 旧记忆只衰减一次（0.8 × 0.9，而不是 × 0.81）
    assert rows[0]["confidence"] == pytest.approx(0.72)

    # 下一轮再来同一条矛盾语句：仍不新增 conflict 行
    await extract_and_store(SESSION, "我还是讨厌 Markdown", "好的", db)
    assert len([r for r in await memories(db) if r["status"] == "conflict"]) == 1


async def test_conflict_round_then_missed_conflict_no_duplicate(db, monkeypatch):
    """MIN-4：上一轮 LLM 判出矛盾、这一轮漏判返回 []，候选会落进普通分支。
    若判重只比 active，同一句话就会以 conflict 和 active 各存一行。
    这一轮的候选置信度必须高于冲突行（1.0 > 0.6），否则会被 confidence 比较
    顺带拦下（返回 None），测不出「跨状态判重」这条。"""
    old_id = await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.8)
    llm = MemoryLLM(
        [("preference", "用户讨厌用 Markdown 记笔记", 3)],
        conflicts=[[old_id]],
    )
    use_writer_llm(monkeypatch, llm)
    await extract_and_store(SESSION, "我讨厌 Markdown", "好的", db)
    conflicts = [r for r in await memories(db) if r["status"] == "conflict"]
    assert len(conflicts) == 1 and conflicts[0]["confidence"] == 0.6

    # 第二轮：LLM 漏判（conflicts 脚本已空 → 返回 []），候选置信度更高
    llm.items = [("preference", "用户讨厌用 Markdown 记笔记", 5)]
    await extract_and_store(SESSION, "我还是讨厌 Markdown", "好的", db)

    rows = await memories(db)
    assert [r["status"] for r in rows] == ["active", "conflict"], [
        (r["status"], r["content"]) for r in rows
    ]


async def test_negated_fact_is_not_deduped_away(db, monkeypatch):
    """MAJ-3：插入否定词同属「一词之差改变事实」，相似度 0.96 拦不住，靠否定词守卫。"""
    await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.8)
    llm = MemoryLLM([("preference", "用户不喜欢用 Markdown 记笔记", 4)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我现在不喜欢 Markdown 了", "好的", db)

    rows = await memories(db)
    assert [r["content"] for r in rows] == [
        "用户喜欢用 Markdown 记笔记",
        "用户不喜欢用 Markdown 记笔记",
    ]
    assert rows[0]["confidence"] == 0.8  # 旧记忆未被当作重复而丢弃/覆盖


async def test_negation_guard_does_not_block_real_duplicates(db, monkeypatch):
    """否定词守卫不能矫枉过正：两侧都没有否定词的同一句仍要判重。"""
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.8)
    llm = MemoryLLM([("preference", "用户偏好用 Markdown 记笔记。", 3)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我偏好 Markdown", "好的", db)

    assert len(await memories(db)) == 1


async def test_conflict_decay_has_a_floor(db, monkeypatch):
    """m3：反复冲突不能把旧记忆的 confidence 压到 0（等于废掉）。"""
    old_id = await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.12)
    llm = MemoryLLM(
        [("preference", "用户讨厌用 Markdown 记笔记", 4)],
        conflicts=[[old_id]] * 5,
    )
    use_writer_llm(monkeypatch, llm)

    for _ in range(5):
        await extract_and_store(SESSION, "我讨厌 Markdown", "好的", db)

    rows = await memories(db)
    assert rows[0]["confidence"] >= writer.MIN_CONFIDENCE


async def test_conflict_detection_skipped_when_too_dissimilar(db, monkeypatch, caplog):
    """suggestion：与最相似旧记忆差异过大时不必问 LLM，省一次调用。"""
    await insert_memory(db, "fact", "用户在学密码学入门", 0.8)
    llm = MemoryLLM([("fact", "用户养了一只猫叫咪咪", 4)])
    use_writer_llm(monkeypatch, llm)

    with caplog.at_level("WARNING", logger="app.memory.writer"):
        await extract_and_store(SESSION, "我养了只猫", "好的", db)

    assert caplog.text.count("跳过冲突判定") == 1
    assert caplog.records[-1].levelname == "WARNING"
    assert llm.conflict_prompts == []
    assert len(await memories(db)) == 2  # 仍按新记忆并存


async def test_conflict_marks_new_memory_and_decays_old(db, monkeypatch):
    old_id = await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.8)
    llm = MemoryLLM([("preference", "用户讨厌用 Markdown 记笔记", 4)], conflicts=[[old_id]])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我现在很讨厌 Markdown", "好的", db)

    rows = await memories(db)
    assert len(rows) == 2
    assert rows[0]["status"] == "active"
    assert rows[0]["confidence"] == pytest.approx(0.72)  # 0.8 × 0.9
    # 新记忆挂冲突态，不覆盖旧记忆
    assert rows[1]["status"] == "conflict"
    assert rows[1]["content"] == "用户讨厌用 Markdown 记笔记"

    # 冲突判定交给 LLM：prompt 里带上了新记忆与候选旧记忆
    prompt = llm.conflict_prompts[0]
    assert "用户讨厌用 Markdown 记笔记" in prompt
    assert f"{old_id}. 用户喜欢用 Markdown 记笔记" in prompt


async def test_conflict_ignores_unknown_ids_and_bad_payload(db, monkeypatch):
    await insert_memory(db, "preference", "用户喜欢用 Markdown 记笔记", 0.8)
    llm = MemoryLLM(
        [("preference", "用户讨厌 Markdown 里的表格语法", 4)], conflicts=[[999]]
    )
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我讨厌 Markdown 的表格", "好的", db)

    rows = await memories(db)
    # 幻觉 id 被忽略，于是按普通新记忆入库，旧记忆不衰减
    assert [r["status"] for r in rows] == ["active", "active"]
    assert rows[0]["confidence"] == 0.8


async def test_conflict_detection_only_compares_same_kind(db, monkeypatch):
    await insert_memory(db, "goal", "用户想在三个月内拿到大模型实习", 0.8)
    llm = MemoryLLM([("preference", "用户讨厌用 Markdown 记笔记", 4)])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "我讨厌 Markdown 的表格", "好的", db)

    # 不同 kind 不进冲突判定窗口：没有旧记忆可比时不做 LLM 冲突判定
    assert llm.conflict_prompts == []
    assert [r["status"] for r in await memories(db)] == ["active", "active"]


async def test_write_failure_is_swallowed(db, monkeypatch):
    def boom():
        raise RuntimeError("没有配置 API key")

    monkeypatch.setattr(writer, "get_llm", boom)

    await extract_and_store(SESSION, "我喜欢用 Markdown", "好的", db)

    assert await memories(db) == []


# ---------- 召回侧 ----------


async def test_recall_formats_top_k_by_confidence(db):
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.6)
    await insert_memory(db, "fact", "用户在准备大模型实习面试", 0.9)
    await insert_memory(db, "goal", "用户想在三个月内写完项目", 0.4)
    await insert_memory(db, "fact", "用户养了一只猫", 0.3, status="conflict")
    await insert_memory(db, "fact", "用户在北京", 0.2, status="superseded")

    text = await recall_memories("我该怎么记笔记？", db)

    assert text == (
        "以下是关于用户的一些长期记忆，供参考：\n"
        "- [fact] 用户在准备大模型实习面试\n"
        "- [preference] 用户偏好用 Markdown 记笔记\n"
        "- [goal] 用户想在三个月内写完项目"
    )


async def test_recall_prefers_query_relevant_over_higher_confidence(db, monkeypatch):
    """query 感知召回：与问题相关但置信度更低的记忆要能挤掉无关的中置信记忆。"""
    monkeypatch.setattr(settings, "memory_recall_top_k", 2)
    await insert_memory(db, "fact", "用户养了一只猫叫咪咪", 0.9)
    await insert_memory(db, "fact", "用户在北京上学", 0.6)
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.5)

    text = await recall_memories("我该怎么用 Markdown 记笔记？", db)

    assert "用户偏好用 Markdown 记笔记" in text  # 0.5 的相关记忆挤进 top_k
    assert "用户在北京上学" not in text  # 0.6 的无关记忆被挤掉
    assert text.index("猫") < text.index("Markdown")  # confidence 仍是主序


async def test_recall_without_token_overlap_keeps_confidence_order(db, monkeypatch):
    """向后兼容：query 与所有记忆都无 token 重叠时，结果与纯 confidence 排序一致。"""
    monkeypatch.setattr(settings, "memory_recall_top_k", 2)
    await insert_memory(db, "fact", "用户养了一只猫叫咪咪", 0.9)
    await insert_memory(db, "fact", "用户在北京上学", 0.6)
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.5)

    text = await recall_memories("今天天气不错", db)

    assert text == (
        "以下是关于用户的一些长期记忆，供参考：\n"
        "- [fact] 用户养了一只猫叫咪咪\n"
        "- [fact] 用户在北京上学"
    )


async def test_recall_respects_top_k(db, monkeypatch):
    monkeypatch.setattr(settings, "memory_recall_top_k", 2)
    await insert_memory(db, "fact", "用户在准备面试", 0.9)
    await insert_memory(db, "fact", "用户在北京", 0.8)
    await insert_memory(db, "fact", "用户养了一只猫", 0.7)

    text = await recall_memories("hi", db)

    assert "用户在准备面试" in text and "用户在北京" in text
    assert "猫" not in text


async def test_recall_clamps_top_k_against_bad_config(db, monkeypatch):
    """m1：负值会让 SQLite 的 LIMIT -1 变成全量注入，必须钳住。"""
    for i in range(60):
        await insert_memory(db, "fact", f"用户的事实{i}", 0.5)

    monkeypatch.setattr(settings, "memory_recall_top_k", -1)
    assert await recall_memories("hi", db) is None  # 负值钳到 0：一条都不注入

    monkeypatch.setattr(settings, "memory_recall_top_k", 0)
    assert await recall_memories("hi", db) is None  # 0 表示本轮不注入

    monkeypatch.setattr(settings, "memory_recall_top_k", 3)
    assert (await recall_memories("hi", db)).count("\n- ") == 3

    monkeypatch.setattr(settings, "memory_recall_top_k", 10_000)
    capped = await recall_memories("hi", db)
    assert capped.count("\n- ") == recall_module.MAX_RECALL_TOP_K


async def test_extract_truncates_both_user_message_and_answer(db, monkeypatch):
    """m4：只截助手回答是不够的，超长用户消息同样会撑爆抽取 prompt。"""
    llm = MemoryLLM([])
    use_writer_llm(monkeypatch, llm)

    await extract_and_store(SESSION, "长" * 9000, "好" * 9000, db)

    prompt = llm.chats[0][1].content
    assert "长" * writer.CONTEXT_LIMIT in prompt
    assert "长" * (writer.CONTEXT_LIMIT + 1) not in prompt
    assert "好" * (writer.CONTEXT_LIMIT + 1) not in prompt
    assert len(prompt) < 2 * writer.CONTEXT_LIMIT + 100


async def test_recall_returns_none_without_memories(db):
    assert await recall_memories("hi", db) is None


async def test_recall_returns_none_when_disabled(db, monkeypatch):
    await insert_memory(db, "fact", "用户在准备面试", 0.9)
    monkeypatch.setattr(settings, "memory_enabled", False)

    assert await recall_memories("hi", db) is None


async def test_recall_degrades_on_db_error(db, monkeypatch):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def boom(db_path=None):
        raise RuntimeError("库挂了")
        yield  # pragma: no cover

    monkeypatch.setattr(recall_module, "get_db", boom)

    assert await recall_memories("hi", db) is None


# ---------- 接入点 ----------


def test_assemble_messages_inserts_memory_after_system_prompt():
    history = [runtime.Message(role="user", content="旧消息")]
    memory = "以下是关于用户的一些长期记忆，供参考：\n- [fact] 用户在北京"

    messages = runtime.assemble_messages(history, "新问题", memory)

    assert [m.role for m in messages] == ["system", "system", "user", "user"]
    assert messages[0].content == runtime.SYSTEM_PROMPT
    assert messages[1].content == memory
    assert messages[-1].content == "新问题"


def test_assemble_messages_without_memory_is_unchanged():
    messages = runtime.assemble_messages([], "新问题", None)
    assert [m.role for m in messages] == ["system", "user"]


async def test_run_agent_injects_recalled_memory_into_prompt(db, monkeypatch):
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.9)
    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    use_writer_llm(monkeypatch, MemoryLLM([]))

    await collect_events(SESSION)

    sent = stream.calls[0]
    assert [m.role for m in sent] == ["system", "system", "user"]
    assert sent[1].content.startswith("以下是关于用户的一些长期记忆")
    assert "用户偏好用 Markdown 记笔记" in sent[1].content


async def test_run_agent_skips_memory_message_when_disabled(db, monkeypatch):
    await insert_memory(db, "preference", "用户偏好用 Markdown 记笔记", 0.9)
    monkeypatch.setattr(settings, "memory_enabled", False)
    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    use_writer_llm(monkeypatch, MemoryLLM([]))

    await collect_events(SESSION)

    assert [m.role for m in stream.calls[0]] == ["system", "user"]


def collect(session_id: str = SESSION, message: str = "我喜欢用 Markdown 记笔记"):
    return runtime.run_agent(session_id, message, None)


async def collect_events(
    session_id: str = SESSION, message: str = "我喜欢用 Markdown 记笔记"
) -> list[runtime.AgentEvent]:
    return [e async for e in collect(session_id, message)]


async def test_memory_write_is_fire_and_forget(db, monkeypatch):
    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    gated = GatedMemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 4)])
    use_writer_llm(monkeypatch, gated)

    events = [e async for e in collect(SESSION)]

    # done 已经到达，而抽取还卡在 gate 上：SSE 流没有被记忆写入阻塞
    assert events[-1].type == "done"
    assert await memories(db) == []

    gated.gate.set()
    await runtime.drain_memory_writes()

    rows = await memories(db, SESSION)
    assert [r["content"] for r in rows] == ["用户偏好用 Markdown 记笔记"]
    assert rows[0]["source"] == SESSION


async def test_memory_write_failure_does_not_break_stream(db, monkeypatch):
    def boom():
        raise RuntimeError("LLM 不可用")

    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    monkeypatch.setattr(writer, "get_llm", boom)

    events = [e async for e in collect("s-boom")]
    await runtime.drain_memory_writes()

    assert events[-1].type == "done"
    assert await memories(db) == []


async def test_interrupted_answer_does_not_spawn_memory_write(db, monkeypatch):
    calls: list[str] = []

    def recorder():
        calls.append("get_llm")
        return MemoryLLM([])

    monkeypatch.setattr(runtime, "get_llm", lambda: HangingLLM())
    monkeypatch.setattr(writer, "get_llm", recorder)

    stream = collect(SESSION)
    while (await anext(stream)).type != "text_delta":
        pass
    await stream.aclose()
    await runtime.drain_memory_writes()

    # 半截回答不抽取记忆：写入侧根本没被唤起
    assert calls == []
    assert await memories(db) == []


class ToolLoopLLM:
    """每轮都请求工具调用，把 run_agent 逼到工具轮数上限。"""

    async def chat_stream(self, messages, tools=None):
        yield StreamChunk(
            finish=True,
            tool_calls=[
                ToolCall(id="c1", name="search_knowledge", arguments={"query": "x"})
            ],
        )


class BoomStreamLLM:
    """吐一段回答后抛异常，走 error 分支。"""

    async def chat_stream(self, messages, tools=None):
        yield StreamChunk(text_delta="半截")
        raise RuntimeError("llm 挂了")


async def test_degraded_rounds_do_not_extract_memories(db, monkeypatch):
    """M2：工具轮数上限 / LLM 异常这两条失败轮次照样落库，但不抽取记忆——
    从失败轮次里学到的「事实」正是记忆污染的主要来源。"""
    calls: list[str] = []

    def recorder():
        calls.append("get_llm")
        return MemoryLLM([("fact", "用户的知识库里有 RAG 笔记", 5)])

    for llm in (ToolLoopLLM(), BoomStreamLLM()):
        monkeypatch.setattr(runtime, "get_llm", lambda llm=llm: llm)
        monkeypatch.setattr(runtime, "hybrid_search", _no_chunks)
        monkeypatch.setattr(writer, "get_llm", recorder)

        events = await collect_events(SESSION)

        assert events[-1].type == "error"
        await runtime.drain_memory_writes()

    # 两条失败路径都不该唤起抽取
    assert calls == []
    assert await memories(db) == []


async def _no_chunks(query, k=8, mode="hybrid", db_path=None):
    return []


async def test_successful_round_after_degraded_one_still_extracts(db, monkeypatch):
    """degraded 只影响失败那一轮，正常轮次照常学习。"""
    monkeypatch.setattr(runtime, "get_llm", lambda: ToolLoopLLM())
    monkeypatch.setattr(runtime, "hybrid_search", _no_chunks)
    use_writer_llm(monkeypatch, MemoryLLM([]))
    await collect_events(SESSION)
    await runtime.drain_memory_writes()

    monkeypatch.setattr(runtime, "get_llm", lambda: StreamLLM())
    use_writer_llm(monkeypatch, MemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 4)]))
    events = await collect_events(SESSION)
    await runtime.drain_memory_writes()

    assert events[-1].type == "done"
    assert [r["content"] for r in await memories(db)] == ["用户偏好用 Markdown 记笔记"]


async def test_drain_gives_up_after_timeout(db, monkeypatch, caplog):
    """M6：写入卡死时 drain 必须有上限，不能让测试收尾或进程退出无限等下去。"""
    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    gated = GatedMemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 4)])
    use_writer_llm(monkeypatch, gated)

    events = await collect_events(SESSION)
    assert events[-1].type == "done"

    with caplog.at_level("WARNING", logger="app.agent.runtime"):
        await runtime.drain_memory_writes(timeout=0.05)

    assert "放弃等待" in caplog.text
    assert runtime._pending_writes == set()  # 卡住的任务被摘掉并取消
    assert await memories(db) == []

    gated.gate.set()  # 放行，避免留下悬空任务影响后续用例


async def test_drain_is_noop_without_pending_writes(db):
    await runtime.drain_memory_writes(timeout=0.05)
    assert runtime._pending_writes == set()


class HangingLLM:
    """吐一个 text_delta 后挂住，供中断路径使用。"""

    def __init__(self) -> None:
        self.released = asyncio.Event()

    async def chat_stream(self, messages, tools=None):
        yield StreamChunk(text_delta="半截回答")
        await self.released.wait()


# ---------- 端到端（HTTP 层） ----------


async def test_chat_end_to_end_learns_and_recalls_memory(db, monkeypatch):
    """走真实 HTTP 链路：一轮对话后记忆落库，下一轮该记忆被注入提示词。"""
    import httpx

    from app.main import app

    stream = StreamLLM("好的，我会用 Markdown 给你记笔记")
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    use_writer_llm(monkeypatch, MemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 4)]))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"session_id": SESSION, "message": "我喜欢用 Markdown"})
        assert resp.status_code == 200
        await runtime.drain_memory_writes()

        rows = await memories(db, SESSION)
        assert [r["content"] for r in rows] == ["用户偏好用 Markdown 记笔记"]

        # 第二轮：召回的记忆作为 system 消息进入提示词
        await client.post("/api/chat", json={"session_id": SESSION, "message": "继续"})

    sent = stream.calls[-1]
    assert sent[1].content.startswith("以下是关于用户的一些长期记忆")
    assert "用户偏好用 Markdown 记笔记" in sent[1].content

async def test_lifespan_drains_pending_memory_writes(db, monkeypatch):
    """m5：进程退出前要给在途写入一个收尾窗口，否则最后几轮记忆随进程消失。"""
    from app import main as main_module

    stream = StreamLLM()
    monkeypatch.setattr(runtime, "get_llm", lambda: stream)
    gated = GatedMemoryLLM([("preference", "用户偏好用 Markdown 记笔记", 4)])
    use_writer_llm(monkeypatch, gated)

    drained: list[int] = []
    real_drain = runtime.drain_memory_writes

    async def spy_drain(timeout: float = runtime.DRAIN_TIMEOUT) -> None:
        # 退出时确实还有在途写入，drain 就是为了等它们
        drained.append(len(runtime._pending_writes))
        await real_drain(timeout)

    monkeypatch.setattr(main_module, "drain_memory_writes", spy_drain)

    async with main_module.lifespan(main_module.app):
        await collect_events(SESSION)
        # 退出前写入还卡在 gate 上
        gated.gate.set()
    # lifespan 退出时调用了 drain，且写入已落库
    assert drained and drained[0] >= 1
    assert [r["content"] for r in await memories(db)] == ["用户偏好用 Markdown 记笔记"]
