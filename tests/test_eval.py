"""T10 评测框架自身的测试：指标纯函数、评测集完整性、runner 端到端（全 mock）、
消融矩阵与报告。全程不发真实 API 请求：embedding 与 LLM 都是桩。"""

import asyncio
import json

import pytest

from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.llm.types import ChatResult, StreamChunk, ToolCall
from app.memory import writer as memory_writer
from app.retrieval import bm25_search
from app.retrieval.bm25_search import invalidate
from app.retrieval.types import RetrievedChunk
from app.tracing import drain_traces
from eval import metrics as eval_metrics
from eval import runner as eval_runner
from eval.ablation import ABLATION_GROUPS, run_ablation
from eval.report import render_ablation_report, render_report, save_report
from eval.runner import EvalResult, load_dataset, run_eval

DIM = 8

DATASET_DIR = "eval/dataset"
DATASET_PATH = f"{DATASET_DIR}/eval.json"

CATEGORY_TARGETS = {
    "single-hop": 0.30,
    "multi-hop": 0.25,
    "temporal": 0.20,
    "preference": 0.15,
    "adversarial": 0.10,
}


# ---------- 指标纯函数 ----------


def test_retrieval_metrics_hit_and_mrr():
    result = eval_metrics.retrieval_metrics(["甲", "乙", "丙"], ["乙"], k=3)
    assert result["hit_at_k"] == 1.0
    assert result["mrr"] == 0.5  # 第 2 位命中
    assert result["recall_at_k"] == 1.0


def test_retrieval_metrics_miss_and_k_cutoff():
    assert eval_metrics.retrieval_metrics(["甲"], ["乙"], k=8)["hit_at_k"] == 0.0
    # 命中在第 3 位但 k=2：被截断，视为未命中
    assert eval_metrics.retrieval_metrics(["甲", "乙", "丙"], ["丙"], k=2)["mrr"] == 0.0
    # 部分覆盖：3 个期望只命中 1 个
    assert eval_metrics.retrieval_metrics(["甲"], ["甲", "乙", "丙"], k=8)[
        "recall_at_k"
    ] == pytest.approx(1 / 3)


def test_retrieval_metrics_rounds_takes_best_rank_over_rounds():
    """多轮检索：第 2 轮的命中不能因为拼在第 1 轮后面而被 k 截掉。

    第 1 轮已经填满 k 条且全不命中，第 2 轮第 1 条就命中——把两轮拼起来再截断
    k=8 会让这个命中排到第 9 位、恒判未命中（修复前的行为）。
    """
    first = [f"噪声{i}" for i in range(8)]
    rounds = [first, ["目标文档", "别的"]]
    result = eval_metrics.retrieval_metrics_rounds(rounds, ["目标文档"], k=8)
    assert result["hit_at_k"] == 1.0
    assert result["mrr"] == 1.0  # 第 2 轮的第 1 位
    assert result["recall_at_k"] == 1.0

    # 拼接后的旧口径确实会判未命中（说明这条断言真的在测东西）
    assert eval_metrics.retrieval_metrics(
        [t for rnd in rounds for t in rnd], ["目标文档"], k=8
    )["hit_at_k"] == 0.0


def test_retrieval_metrics_rounds_uses_best_rank_across_rounds():
    """同一文档在多轮里出现，mrr 取最好（最小序号）的那个排名。"""
    result = eval_metrics.retrieval_metrics_rounds(
        [["噪声", "目标"], ["目标", "另一个"]], ["目标"], k=8
    )
    assert result["mrr"] == 1.0  # 第 2 轮的 rank 1 优于第 1 轮的 rank 2
    # 每篇期望文档取自己最好的排名，recall 是「任一轮进过前 k」的比例
    partial = eval_metrics.retrieval_metrics_rounds(
        [["甲"], ["乙"]], ["甲", "乙", "丙"], k=8
    )
    assert partial["recall_at_k"] == pytest.approx(2 / 3)
    # 空轮次 / 空期望都不炸
    assert eval_metrics.retrieval_metrics_rounds([], ["甲"], k=8)["hit_at_k"] == 0.0
    assert eval_metrics.retrieval_metrics_rounds([["甲"]], [], k=8)["hit_at_k"] == 0.0


def test_answer_metrics_coverage():
    full = eval_metrics.answer_metrics("答案是 functools.wraps 和 __name__", ["functools.wraps", "__name__"])
    assert full["keyword_coverage"] == 1.0 and full["missed"] == []
    half = eval_metrics.answer_metrics("只有 functools.wraps", ["functools.wraps", "lru_cache"])
    assert half["keyword_coverage"] == 0.5 and half["missed"] == ["lru_cache"]
    # 无标注约束不惩罚
    assert eval_metrics.answer_metrics("随便", [])["keyword_coverage"] == 1.0


def test_answer_metrics_excludes_only_fires_on_assertion():
    """禁词判据是「断言了禁词」，不是「提到了禁词」——胡编罚、诚实拒答不罚。"""
    excluded = ["Q-learning", "策略梯度"]
    # 胡编：出现禁词且所在句子没有任何否认表述 → 0 分
    fabricated = eval_metrics.answer_metrics(
        "Q-learning 的核心是时序差分学习，策略梯度则直接优化策略。", ["没有"], excluded
    )
    assert fabricated["keyword_coverage"] == 0.0
    assert fabricated["violated"] == ["Q-learning", "策略梯度"]
    # 正确拒答：点名了被问的术语，但整句都在否认 → 正常给满分
    correct = eval_metrics.answer_metrics(
        "笔记里没有关于 Q-learning 或策略梯度的内容。", ["没有"], excluded
    )
    assert correct["keyword_coverage"] == 1.0
    assert correct["violated"] == []
    # 未标注 contains 但断言了禁词：同样 0 分（不能因「无约束」而免罚）
    assert eval_metrics.answer_metrics("参考 Q-learning 的做法", [], excluded)[
        "keyword_coverage"
    ] == 0.0
    # 先说「没有」再展开禁词：展开句没有否认表述 → 仍然 0 分
    sneaky = eval_metrics.answer_metrics(
        "笔记里没有强化学习的内容。Q-learning 大致是这样工作的：……", ["没有"], excluded
    )
    assert sneaky["keyword_coverage"] == 0.0


def test_excludes_negation_is_sentence_scoped_not_document_scoped():
    """否认只看禁词所在**句子**：整篇找否定词会放过「挂着否认词的胡编」。

    这条是判据的边界：句子切分必须精确到句末标点，既不能整篇判定（放过胡编），
    也不能连逗号都切（把「没有关于 A、B 的记录」的诚实拒答误杀）。
    """
    excluded = ["nginx"]
    # 否认句 + 另一个句子里的断言：断言句无否定词 → 违规
    mixed = eval_metrics.answer_metrics(
        "笔记里没有 Kubernetes 的内容。nginx 是最常见的 Ingress 控制器。",
        [],
        excluded,
    )
    assert mixed["violated"] == ["nginx"]
    # 顿号连接的并列术语仍属同一句 → 不违规（不能把顿号当句子边界）
    listing = eval_metrics.answer_metrics(
        "笔记里没有关于 nginx、Traefik 的记录。", [], excluded
    )
    assert listing["violated"] == []


def test_excludes_known_hole_is_pinned():
    """已知漏洞要有用例钉住，避免 README 与实际行为脱节（见 dataset/README）。

    否认判定是句级的，不看否认的对象：一句里既有否认又有断言时整句豁免。
    「没有用 nginx，改用 Traefik」里 Traefik 被断言却是禁词，判为不违规。
    这是有意的取舍（对象级绑定对自动指标不划算），不是待修的 bug——若哪天
    改成对象级判定，这条用例会红，提醒同步改 README。
    """
    result = eval_metrics.answer_metrics(
        "笔记里没有用 nginx，改用 Traefik。", ["没有"], ["nginx", "Traefik"]
    )
    assert result["violated"] == [], "行为变了：请同步更新 dataset/README 的漏洞说明"
    assert result["keyword_coverage"] == 1.0
    # 对照：无否认词时两个禁词都算违规（这才是 0 分可信的那一侧）
    plain = eval_metrics.answer_metrics(
        "推荐用 nginx 或 Traefik。", ["没有"], ["nginx", "Traefik"]
    )
    assert plain["violated"] == ["nginx", "Traefik"]
    assert plain["keyword_coverage"] == 0.0


def test_keyword_matching_is_case_insensitive():
    """标注大小写与回答大小写不一致时不能被绕过（同词不同拼写风格）。"""
    # contains：标注 Semaphore，回答写 semaphore，仍算命中
    answered = eval_metrics.answer_metrics("用 asyncio.semaphore 限流", ["Semaphore"])
    assert answered["keyword_coverage"] == 1.0 and answered["missed"] == []
    # excludes：禁词 nginx，回答写 Nginx / NGINX，都要判违规
    for spelling in ("Nginx", "NGINX", "nGiNx"):
        result = eval_metrics.answer_metrics(
            f"推荐用 {spelling} 做 Ingress。", [], ["nginx"]
        )
        assert result["keyword_coverage"] == 0.0, spelling
        assert result["violated"] == ["nginx"]


def test_memory_metrics_skips_unlabeled():
    assert eval_metrics.memory_metrics(None, [])["memory_recall"] is None
    hit = eval_metrics.memory_metrics("用户喜欢用 Markdown 写笔记", ["Markdown"])
    assert hit["memory_recall"] == 1.0
    miss = eval_metrics.memory_metrics(None, ["Markdown"])
    assert miss["memory_recall"] == 0.0


def test_aggregate_metrics_skips_none():
    samples = [
        {
            "category": "single-hop",
            "error": None,
            "metrics": {
                "retrieval": {"hit_at_k": 1.0, "mrr": 1.0, "recall_at_k": 1.0},
                "answer": {"keyword_coverage": 1.0},
                "memory": {"memory_recall": None},
            },
        },
        {
            "category": "preference",
            "error": None,
            "metrics": {
                "retrieval": None,
                "answer": {"keyword_coverage": 0.5},
                "memory": {"memory_recall": 0.0},
            },
        },
    ]
    agg = eval_metrics.aggregate_metrics(samples)
    assert agg["total"] == 2
    assert agg["overall"]["hit_at_k"] == 1.0  # 只有 1 条有检索标注
    assert agg["overall"]["memory_recall"] == 0.0  # None 被跳过
    assert agg["overall"]["keyword_coverage"] == 0.75
    assert set(agg["by_category"]) == {"single-hop", "preference"}


# ---------- 评测集完整性 ----------


def test_dataset_integrity():
    dataset = load_dataset(DATASET_PATH)
    samples = dataset["samples"]
    assert 30 <= len(samples) <= 50

    ids = [s["id"] for s in samples]
    assert len(ids) == len(set(ids)), "样本 id 重复"

    from pathlib import Path

    docs_dir = Path(DATASET_DIR) / "docs"
    for doc in dataset["docs"]:
        assert (docs_dir / doc).is_file(), f"文档缺失：{doc}"

    counts: dict[str, int] = {}
    for s in samples:
        assert s["category"] in CATEGORY_TARGETS, f"未知类别：{s['category']}"
        assert s["id"] and s["query"]
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    for category, target in CATEGORY_TARGETS.items():
        ratio = counts.get(category, 0) / len(samples)
        assert abs(ratio - target) <= 0.08, f"{category} 占比 {ratio:.2%} 偏离目标 {target:.0%}"

    # 偏好类样本必须带记忆标注，对抗样本必须明确期望行为
    for s in samples:
        if s["category"] == "preference":
            assert s.get("expected_memories"), f"{s['id']} 缺少 expected_memories"
            assert s.get("seed_memories"), f"{s['id']} 缺少 seed_memories"
        if s["category"] == "adversarial":
            assert s.get("expected_answer_contains"), f"{s['id']} 缺少对抗期望"
            # 只有 contains 没有判别力：胡答 + 补一句「没有」也是满分
            assert s.get("expected_answer_excludes"), f"{s['id']} 缺少禁词标注"

    # 「压缩」消融轴只在历史 > context_compaction_threshold 时触发，评测集必须有
    # 足够长的会话样本，否则 A/C、D/G 两组输出完全相同（等于没测这条轴）
    threshold = settings.context_compaction_threshold
    long_sessions = [s for s in samples if len(s.get("sessions", [])) > threshold]
    assert len(long_sessions) >= 2, (
        f"长会话样本只有 {len(long_sessions)} 条（需 > {threshold} 条历史）："
        "压缩轴在评测集上永远不会触发"
    )


# ---------- runner 端到端（全 mock） ----------


class FakeLLM:
    """没收到工具结果就请求检索，收到后吐出包含测试关键词的回答。

    按消息内容而不是轮次计数决策：多个样本并发共享一个实例时，
    轮次计数会竞争，内容判断对每个会话都是确定性的。
    """

    ANSWER = "根据笔记：苹果。没有相关内容。用户喜欢 Markdown，回答保持简洁。"

    async def chat_stream(self, messages, tools=None):
        if not any(m.role == "tool" for m in messages):
            yield StreamChunk(
                finish=True,
                tool_calls=[
                    ToolCall(id="c1", name="search_knowledge", arguments={"query": "苹果"})
                ],
            )
        else:
            yield StreamChunk(text_delta=self.ANSWER)
            yield StreamChunk(finish=True, tool_calls=[])


class _SilentWriterLLM:
    async def chat(self, messages, tools=None):
        return ChatResult(text="[]")


async def _fake_embed(texts):
    return [[float((hash((t, i)) % 1000) / 1000.0) for i in range(DIM)] for t in texts]


async def ingest_bare_doc(db_path, title="阿尔法笔记", text="苹果是一种水果。"):
    """往库里塞一篇最小文档（走真实 ingest，embedding 由 stub_embeddings 桩掉）。

    用于构造「库里确实有内容」的场景（守卫不能删掉有数据的库）。
    """
    from pathlib import Path

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "doc.md"
        doc.write_text(f"# {title}\n{text}", encoding="utf-8")
        from app.ingest.pipeline import ingest as _ingest

        return await _ingest(doc, db_path)


@pytest.fixture
def stub_embeddings(monkeypatch):
    """全 mock 的 embedding 与记忆抽取桩（自建评测集与真实评测集共用）。

    固定 8 维向量库 + 哈希 embedding：不发任何网络请求，且同一段文本每次得到同一
    向量（检索结果可复现）。记忆抽取走 _SilentWriterLLM（永远抽不出东西）。
    """
    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr("app.ingest.pipeline.embed_texts", _fake_embed)
    monkeypatch.setattr("app.retrieval.hybrid.embed_texts", _fake_embed)
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    invalidate()
    yield
    invalidate()


@pytest.fixture
def tiny_dataset(tmp_path, stub_embeddings):
    """两条文档、三条样本的最小评测集 + 全套桩：embedding / 记忆抽取 / 8 维库。"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# 阿尔法笔记\n苹果是一种水果，阿尔法笔记记录水果。", encoding="utf-8")
    (docs_dir / "b.md").write_text("# 贝塔笔记\n贝塔笔记记录编程，与水果无关。", encoding="utf-8")
    dataset = {
        "version": 7,
        "docs": ["a.md", "b.md"],
        "samples": [
            {
                "id": "t-001",
                "category": "single-hop",
                "query": "苹果是什么？",
                "expected_chunks": ["阿尔法笔记"],
                "expected_answer_contains": ["苹果"],
                "sessions": [],
            },
            {
                "id": "t-002",
                "category": "preference",
                "query": "我喜欢什么格式？",
                "expected_chunks": [],
                "expected_answer_contains": ["Markdown"],
                "expected_memories": ["Markdown"],
                "sessions": [],
                "seed_memories": [
                    {"kind": "preference", "content": "用户喜欢用 Markdown 格式写笔记"}
                ],
            },
            {
                "id": "t-003",
                "category": "adversarial",
                "query": "我的 Rust 笔记说了什么？",
                "expected_chunks": [],
                "expected_answer_contains": ["没有"],
                "expected_answer_excludes": ["借用检查器"],
                "sessions": [],
            },
        ],
    }
    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    return dataset_path


async def test_run_eval_end_to_end(tiny_dataset, tmp_path):
    db_path = str(tmp_path / "eval.db")
    result = await run_eval(
        str(tiny_dataset), db_path=db_path, llm=FakeLLM(), concurrency=2
    )

    assert [s.id for s in result.samples] == ["t-001", "t-002", "t-003"]
    assert result.git_commit
    assert result.config["retrieval_mode"] == "hybrid"

    single = result.samples[0]
    assert single.error is None
    assert "阿尔法笔记" in single.retrieved  # 包装器录到了真实检索结果
    assert single.metrics["retrieval"]["hit_at_k"] == 1.0
    assert single.metrics["answer"]["keyword_coverage"] == 1.0

    pref = result.samples[1]
    assert pref.metrics["retrieval"] is None  # 无检索标注就跳过
    assert pref.metrics["memory"]["memory_recall"] == 1.0
    assert "Markdown" in (pref.recalled_memory or "")

    overall = result.metrics["overall"]
    assert overall["n"] == 3
    assert overall["hit_at_k"] == 1.0
    assert overall["keyword_coverage"] == 1.0
    assert overall["errors"] == 0


async def test_run_eval_sample_ids_subset(tiny_dataset, tmp_path):
    result = await run_eval(
        str(tiny_dataset),
        db_path=str(tmp_path / "sub.db"),
        llm=FakeLLM(),
        sample_ids=["t-001"],
    )
    assert [s.id for s in result.samples] == ["t-001"]
    assert result.metrics["total"] == 1


async def test_run_eval_save_persists_reproducibility(tiny_dataset, tmp_path):
    result = await run_eval(
        str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=FakeLLM()
    )
    path = result.save(results_dir=tmp_path / "results")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["git_commit"] == result.git_commit
    assert saved["config"]["retrieval_mode"] == "hybrid"
    assert len(saved["samples"]) == 3


async def test_config_overrides_are_restored(tiny_dataset, tmp_path):
    before = settings.memory_enabled
    await run_eval(
        str(tiny_dataset),
        config_overrides={"memory_enabled": False, "retrieval_mode": "bm25"},
        db_path=str(tmp_path / "eval.db"),
        llm=FakeLLM(),
    )
    assert settings.memory_enabled == before
    assert settings.retrieval_mode == "hybrid"


async def test_llm_error_is_recorded_per_sample(tiny_dataset, tmp_path):
    class BoomLLM:
        async def chat_stream(self, messages, tools=None):
            raise RuntimeError("llm 挂了")
            yield

    result = await run_eval(
        str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=BoomLLM()
    )
    assert result.metrics["overall"]["errors"] == 3
    assert all(s.error and "llm 挂了" in s.error for s in result.samples)


async def test_run_eval_resets_db_so_repeat_runs_are_reproducible(tiny_dataset, tmp_path):
    """同一 db_path 跑第二遍不得把文档再灌一遍（ingest 是纯追加）。

    修复前实测 hit@k 随运行次数单调下滑（0.82→0.54→0.36）；这里直接查库里的
    documents/chunks 行数，并断言两次运行的指标完全相同。
    """
    db_path = str(tmp_path / "eval.db")
    first = await run_eval(str(tiny_dataset), db_path=db_path, llm=FakeLLM())
    second = await run_eval(str(tiny_dataset), db_path=db_path, llm=FakeLLM())

    async with get_db(db_path) as conn:
        docs = (await conn.execute_fetchall("SELECT COUNT(*) AS n FROM documents"))[0]["n"]
        chunks = (await conn.execute_fetchall("SELECT COUNT(*) AS n FROM chunks"))[0]["n"]
    assert (docs, chunks) == (2, 2), "重复运行把同一批文档又灌了一遍"

    assert first.metrics == second.metrics
    assert [s.answer for s in first.samples] == [s.answer for s in second.samples]


async def test_reset_db_waits_for_inflight_traces(tmp_path, monkeypatch):
    """清库必须等在途的 trace 写入结束——恢复/清理动作不能发生在在飞任务之前。

    T9 的埋点是 fire-and-forget，工具 trace 落的正是这个评测库。不等它们跑完就删
    文件，Windows 上会因文件被占用抛 PermissionError（「另一个程序正在使用此文件」），
    Linux 上则是把 trace 悄悄写进已删除的 inode。

    这里不依赖「删文件是否恰好被占用」这种时序运气，直接断言顺序：清库返回之前，
    在途的 trace 任务必须已经结束。
    """
    import app.tracing as app_tracing
    from eval.runner import _reset_db

    monkeypatch.setattr(settings, "tracing_enabled", True)  # conftest 默认关埋点
    path = tmp_path / "eval.db"
    await init_db(path)

    order: list[str] = []

    async def slow_insert(kind, name, detail, tokens_in, tokens_out, cost, db_path):
        await asyncio.sleep(0.05)  # 模拟一次真实的落库耗时
        order.append("trace-done")

    monkeypatch.setattr(app_tracing, "_insert", slow_insert)
    app_tracing.record_trace("tool", "search_knowledge", db_path=str(path))
    assert app_tracing._pending, "在途任务没被登记，这个用例没在测东西"

    order.append("reset-start")
    await _reset_db(path)
    order.append("reset-done")

    assert order == ["reset-start", "trace-done", "reset-done"], (
        f"清库没等在途 trace 结束：{order}"
    )
    assert not path.exists()


async def test_run_eval_leaves_no_inflight_writes_behind(
    tiny_dataset, tmp_path, monkeypatch
):
    """run_eval 返回时必须已收干净自己触发的后台写入。

    埋点与记忆抽取都是 fire-and-forget。不等它们跑完，任务会带着已打开的库连接活过
    run_eval 返回，调用方随后删库/删目录在 Windows 上直接 PermissionError——实测
    `--ablation`（CLI 默认开埋点）在 tempfile 清理阶段炸掉，整轮产物全丢。

    这里不依赖「文件是否恰好被占用」的时序运气，直接断言返回前两个 drain 都被调用。
    """
    import app.agent.runtime as app_runtime

    calls: list[str] = []
    orig_memory = app_runtime.drain_memory_writes
    orig_traces = eval_runner.drain_traces

    async def spy_memory(timeout: float = 5.0):
        calls.append("memory")
        await orig_memory(timeout)

    async def spy_traces(timeout: float = 5.0):
        calls.append("traces")
        await orig_traces(timeout)

    monkeypatch.setattr(app_runtime, "drain_memory_writes", spy_memory)
    monkeypatch.setattr(eval_runner, "drain_traces", spy_traces)
    monkeypatch.setattr(settings, "tracing_enabled", True)  # 覆盖 CLI 默认路径

    await run_eval(str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=FakeLLM())
    # 清库也 drain 一次（memory 在前），所以两个词各出现两次
    assert calls.count("memory") >= 1 and calls.count("traces") >= 1, calls
    assert calls[0] == "memory", f"第一次 drain 必须是记忆写入（它会新产生 trace）：{calls}"


async def test_ablation_temp_dir_cleanup_succeeds(tiny_dataset, tmp_path, monkeypatch):
    """--ablation 的临时目录清理不能因在途写入占用文件而失败（文档化的主路径）。

    这条覆盖 CLI 的真实形态：work_dir 缺省 → run_ablation 建临时目录、跑完清理。
    """
    monkeypatch.setattr(settings, "tracing_enabled", True)  # CLI 默认
    results = await run_ablation(
        str(tiny_dataset), groups=["A", "B"], llm=FakeLLM()
    )
    assert set(results) == {"A", "B"}


async def test_reset_db_waits_for_memory_writes_before_traces(tmp_path, monkeypatch):
    """清库要等**记忆写入**结束，且顺序在等 trace 之前。

    记忆抽取是 runtime 的 fire-and-forget 任务，它也往这个库写，而且**自己会产生
    llm trace**：先等 trace 再等记忆的话，记忆任务在等待期间新落的 trace 会漏网。
    这里断言两个 drain 都被调用且记忆在前。
    """
    import app.agent.runtime as app_runtime
    from eval.runner import _reset_db

    path = tmp_path / "eval.db"
    await init_db(path)

    calls: list[str] = []
    orig_memory = app_runtime.drain_memory_writes
    orig_traces = eval_runner.drain_traces

    async def spy_memory(timeout: float = 5.0):
        calls.append("memory")
        await orig_memory(timeout)

    async def spy_traces(timeout: float = 5.0):
        calls.append("traces")
        await orig_traces(timeout)

    monkeypatch.setattr(app_runtime, "drain_memory_writes", spy_memory)
    monkeypatch.setattr(eval_runner, "drain_traces", spy_traces)

    await _reset_db(path)
    assert calls == ["memory", "traces"], f"drain 的调用次数或顺序不对：{calls}"


async def test_reset_db_refuses_to_delete_the_app_database(
    tmp_path, monkeypatch, stub_embeddings
):
    """显式把应用库路径传给评测时必须在删库前 raise（MAJ-1）。

    默认值 data/eval.db 只挡住了不传参的路径；显式传 settings.db_path（用户的知识库）
    时 _reset_db 会连 -wal/-shm 一起删掉，笔记永久丢失且没有任何提示。
    """
    from eval.runner import _reset_db

    app_db = tmp_path / "app.db"
    await init_db(app_db)
    await ingest_bare_doc(app_db)  # 放点内容进去，确认真的没被删
    monkeypatch.setattr(settings, "db_path", str(app_db))

    with pytest.raises(ValueError, match="不允许清应用库"):
        await _reset_db(app_db)
    # 守卫发生在删除之前：库与内容都还在
    assert app_db.exists()
    async with get_db(app_db) as conn:
        docs = (await conn.execute_fetchall("SELECT COUNT(*) AS n FROM documents"))[0]["n"]
    assert docs == 1

    # 相对路径 / 带 .. 的等价写法同样要挡住（比较的是解析后的绝对路径）
    with pytest.raises(ValueError, match="不允许清应用库"):
        await _reset_db(tmp_path / "." / "app.db")

    # 与 settings.db_path 无关的库不受影响
    other = tmp_path / "other.db"
    await init_db(other)
    await _reset_db(other)
    assert not other.exists()


async def test_run_eval_rejects_app_db_path(tiny_dataset, tmp_path, monkeypatch):
    """run_eval 传应用库路径同样被拒（守卫在 run_eval 的调用链上生效）。"""
    app_db = tmp_path / "app.db"
    await init_db(app_db)
    monkeypatch.setattr(settings, "db_path", str(app_db))
    with pytest.raises(ValueError, match="不允许清应用库"):
        await run_eval(str(tiny_dataset), db_path=str(app_db), llm=FakeLLM())
    assert app_db.exists()


async def test_reset_db_invalidates_bm25_index(tmp_path, stub_embeddings):
    """删库要顺手失效进程级 BM25 索引，否则空语料场景静默返回已删文档（MIN-2）。

    _indexes 按绝对路径缓存；删库后若本次 ingest 没产出任何 chunk（早退不调
    invalidate），旧索引原样留着，检索会拿已删文档作答。
    """
    from eval.runner import _reset_db

    path = tmp_path / "kb.db"
    await init_db(path)
    await ingest_bare_doc(path, title="阿尔法笔记")

    index_before = await bm25_search._get_index(path)
    await index_before.load()
    assert index_before.search("阿尔法", 5), "索引没建起来，这个用例没在测东西"
    assert bm25_search._cache_key(path) in bm25_search._indexes

    await _reset_db(path)

    assert bm25_search._cache_key(path) not in bm25_search._indexes, (
        "清库后旧索引仍在缓存里，检索会返回已删文档"
    )
    # 重建空库后检索必须返回空（而不是旧索引里的文档）
    await init_db(path)
    assert await bm25_search.bm25_search("阿尔法", 5, path) == []


async def test_run_eval_twice_with_tracing_enabled(tiny_dataset, tmp_path, monkeypatch):
    """开埋点时连续跑两遍同一库：第二遍的删库不能因文件被占用而失败。

    conftest 默认关埋点，这里显式打开，覆盖 CLI 的默认路径（tracing_enabled=True）。
    """
    monkeypatch.setattr(settings, "tracing_enabled", True)
    db_path = str(tmp_path / "eval.db")
    first = await run_eval(str(tiny_dataset), db_path=db_path, llm=FakeLLM())
    second = await run_eval(str(tiny_dataset), db_path=db_path, llm=FakeLLM())
    await drain_traces()

    assert first.metrics["overall"]["errors"] == 0
    assert second.metrics["overall"]["errors"] == 0
    async with get_db(db_path) as conn:
        traces = (await conn.execute_fetchall("SELECT COUNT(*) AS n FROM traces"))[0]["n"]
    assert traces > 0  # 确实走了会持有文件句柄的埋点路径


async def test_multi_round_retrieval_scores_best_rank_per_round(
    stub_embeddings, tmp_path, monkeypatch
):
    """多轮检索的命中不能被第 1 轮的 k 条挤到 k 之外（m1 端到端回归）。

    第 1 轮返回 8 条全不命中的结果（k=8 已满），第 2 轮才搜到目标文档。修复前
    runner 把两轮结果 extend 进同一个列表再按 k=8 截断，「目标文档」排在第 9 位被
    系统性判为未命中——多跳样本的后续检索几乎拿不到分。
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# 阿尔法笔记\n苹果是一种水果。", encoding="utf-8")
    dataset = {
        "version": 1,
        "docs": ["a.md"],
        "samples": [
            {
                "id": "t-multi",
                "category": "multi-hop",
                "query": "对比两份笔记",
                "expected_chunks": ["目标文档"],
                "expected_answer_contains": ["目标文档"],
                "sessions": [],
            }
        ],
    }
    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    noise = [f"噪声文档{i}" for i in range(8)]

    async def fake_search(query, k=8, mode="hybrid", db_path=None):
        titles = ["目标文档"] if "第二轮" in query else noise
        return [
            RetrievedChunk(chunk_id=100 + i, doc_id=1, content="x", title=t, score=1.0)
            for i, t in enumerate(titles)
        ]

    class TwoRoundLLM:
        """第 1 轮查「第一轮」（不命中），第 2 轮查「第二轮」（命中），然后作答。"""

        async def chat_stream(self, messages, tools=None):
            tool_rounds = sum(1 for m in messages if m.role == "tool")
            if tool_rounds < 2:
                query = "第一轮检索" if tool_rounds == 0 else "第二轮检索"
                yield StreamChunk(
                    finish=True,
                    tool_calls=[
                        ToolCall(id=f"c{tool_rounds}", name="search_knowledge",
                                 arguments={"query": query})
                    ],
                )
            else:
                yield StreamChunk(text_delta="目标文档 内容如下")
                yield StreamChunk(finish=True, tool_calls=[])

    monkeypatch.setattr(runtime, "hybrid_search", fake_search)
    result = await run_eval(
        str(dataset_path), db_path=str(tmp_path / "eval.db"), llm=TwoRoundLLM()
    )

    sample = result.samples[0]
    assert sample.error is None
    # 两轮都被观测到（扁平化产物供人读）
    assert sample.retrieved[:8] == noise
    assert sample.retrieved[-1] == "目标文档"
    assert sample.metrics["retrieval"]["hit_at_k"] == 1.0
    assert sample.metrics["retrieval"]["mrr"] == 1.0  # 第 2 轮 rank 1，不是第 9 位的 1/9
    assert sample.metrics["answer"]["keyword_coverage"] == 1.0


async def test_sample_failure_does_not_unplug_stubs_for_inflight_samples(
    tmp_path, stub_embeddings, monkeypatch
):
    """一个样本炸掉时，其余样本必须仍在「桩 + 观测器 + 消融配置」下跑完。

    修复前的路径：asyncio.gather 默认在第一个异常处向上抛，但其余任务不被取消、
    继续跑，而 finally 立刻撤掉观测器与 LLM 桩（settings 覆盖也随 with 块退出恢复
    默认）——尚未/正在跑的样本于是拿到真实客户端直接打真实 API 烧配额，且按默认
    配置而非消融配置执行。

    这里用两个信号钉住它：
    1. `_restore_observers` 被调用时已完成的样本数（确定性：旧代码必然 < 样本总数）
    2. 期间是否有人碰到「真实客户端」（旧代码必然碰到）
    boom 样本缺 id：`_run_sample` 开头的 f-string 立刻抛 KeyError，位置在样本级
    try 之外，异常会冒到 gather——正是审查报告里那条泄漏路径。
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# 阿尔法笔记\n苹果是一种水果。", encoding="utf-8")
    dataset = {
        "version": 1,
        "docs": ["a.md"],
        "samples": [
            {"category": "adversarial", "query": "坏样本，缺 id"},
            {
                "id": "t-slow",
                "category": "single-hop",
                "query": "苹果是什么？",
                "expected_chunks": ["阿尔法笔记"],
                "expected_answer_contains": ["苹果"],
            },
        ],
    }
    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    real_calls: list[str] = []
    completed: set[str] = set()

    class RealClient:
        """真实 API 路径的代表：被用到即测试失败。"""

        async def chat(self, messages, tools=None):
            real_calls.append("chat")
            raise AssertionError("真实客户端被调用")

        async def chat_stream(self, messages, tools=None):
            real_calls.append("chat_stream")
            raise AssertionError("真实客户端被调用")
            yield

    class SlowLLM(FakeLLM):
        async def chat_stream(self, messages, tools=None):
            if any("苹果是什么" in m.content for m in messages if m.role == "user"):
                await asyncio.sleep(0.05)  # 保证 boom 失败时它还没跑完
            async for chunk in super().chat_stream(messages, tools):
                yield chunk

    # 样本跑完的确定性标记（boom 在 _run_sample 内部就抛，永远不会被记上）
    orig_run_sample = eval_runner._run_sample

    async def spy_run_sample(sample, db_path, k, rounds, recalled):
        result = await orig_run_sample(sample, db_path, k, rounds, recalled)
        completed.add(sample.get("id"))
        return result

    orig_restore = eval_runner._restore_observers
    restored_with_completed: list[set[str]] = []

    def spy_restore(originals):
        restored_with_completed.append(set(completed))
        orig_restore(originals)

    monkeypatch.setattr(eval_runner, "_run_sample", spy_run_sample)
    monkeypatch.setattr(eval_runner, "_restore_observers", spy_restore)
    # 撤桩后 runtime.get_llm 与「真实客户端」是同一个东西，泄漏就会被记下来
    monkeypatch.setattr(runtime, "get_llm", lambda: RealClient())

    escaped: BaseException | None = None
    try:
        result = await run_eval(
            str(dataset_path),
            db_path=str(tmp_path / "eval.db"),
            llm=SlowLLM(),
            concurrency=1,
        )
    except BaseException as exc:  # 旧代码：样本异常冒到调用方，整轮结果丢失
        escaped = exc
        result = None
    # 留出窗口：旧代码里在飞的样本会在 run_eval 返回之后才碰到真实客户端
    await asyncio.sleep(0.15)

    assert real_calls == [], f"在飞的样本打到了真实客户端（配额泄漏）：{real_calls}"
    assert restored_with_completed == [{"t-slow"}], (
        f"撤桩时已完成的样本是 {restored_with_completed}（t-slow 必须已完成）："
        "恢复动作发生在在飞样本结束之前"
    )
    assert escaped is None, f"样本级异常冒到了调用方，整轮结果丢失：{escaped!r}"

    # 样本级异常记进 SampleResult.error，其余样本结果照常保留
    assert result.metrics["overall"]["errors"] == 1
    by_id = {s.id: s for s in result.samples}
    assert "KeyError" in by_id["<unknown>"].error
    assert by_id["t-slow"].error is None
    # 观测器也还在：slow 的检索命中被正常记录下来
    assert by_id["t-slow"].metrics["retrieval"]["hit_at_k"] == 1.0


async def test_config_override_rejects_unknown_field(tiny_dataset, tmp_path):
    with pytest.raises(ValueError, match="未知的配置字段"):
        await run_eval(
            str(tiny_dataset),
            config_overrides={"memory_enabld": True},
            db_path=str(tmp_path / "eval.db"),
            llm=FakeLLM(),
        )


async def test_config_override_rejects_illegal_literal_value(tiny_dataset, tmp_path):
    """retrieval_mode 是 Literal：非法值必须在评测开始前报错。

    修复前裸 setattr 绕过校验，非法值被接受，一路带到 hybrid_search 抛 ValueError，
    经 gather 冒成整轮评测异常（并触发撤桩泄漏，见上一条用例）。
    """
    with pytest.raises(ValueError, match="retrieval_mode"):
        await run_eval(
            str(tiny_dataset),
            config_overrides={"retrieval_mode": "nonsense"},
            db_path=str(tmp_path / "eval.db"),
            llm=FakeLLM(),
        )
    # 报错发生在任何改动之前，settings 保持原值
    assert settings.retrieval_mode == "hybrid"
    assert settings.memory_enabled is True


async def test_error_result_keeps_retrieval_observed_before_crash():
    """崩溃样本若在崩溃前已经检索过，检索指标要照算而不是给 None（MIN-5）。

    retrieval=None 会被聚合跳过，而 answer 的 0 分照常计入分母——两条口径不一致，
    等于把「崩在检索之后的样本」从检索维度摘掉，只统计活下来的样本（生存者偏差）。

    这里直接构造：先往观测收集器里塞一轮检索命中，再调 _error_result。
    """
    from eval.runner import _error_result

    sample = {
        "id": "x-001",
        "category": "multi-hop",
        "query": "跨文档对比",
        "expected_chunks": ["目标文档"],
        "expected_answer_contains": ["结论"],
    }
    # 崩溃前观测到一轮检索，命中第 1 位
    result = _error_result(sample, RuntimeError("第二轮 LLM 挂了"), [["目标文档"]], k=8)
    assert result.error and "RuntimeError" in result.error
    assert result.retrieved == ["目标文档"]  # 观测到的检索结果留在产物里
    assert result.metrics["retrieval"] is not None, "崩溃前的检索证据被丢掉了"
    assert result.metrics["retrieval"]["hit_at_k"] == 1.0
    assert result.metrics["retrieval"]["mrr"] == 1.0
    # 回答维度仍按「没有完成回答」记 0 分，不因无标注约束给满分
    assert result.metrics["answer"]["keyword_coverage"] == 0.0
    assert result.metrics["answer"]["missed"] == ["结论"]

    # 崩在准备阶段（一轮检索都没发生）才给 None——与「不考核检索」区分开
    no_rounds = _error_result(sample, RuntimeError("准备阶段挂了"), [], k=8)
    assert no_rounds.metrics["retrieval"] is None
    assert no_rounds.retrieved == []

    # 没有检索标注的样本即使观测到东西也不给检索分（口径与 _run_sample 一致）
    unlabeled = _error_result(
        {**sample, "expected_chunks": []}, RuntimeError("x"), [["随便"]], k=8
    )
    assert unlabeled.metrics["retrieval"] is None


async def test_crash_after_retrieval_is_scored_not_skipped(
    tiny_dataset, tmp_path, monkeypatch
):
    """端到端：样本崩在检索之后时，aggregate 的 hit@k 仍把它算进去（MIN-5）。

    修复前 _error_result 的 retrieval 一律 None，聚合直接跳过——崩溃样本在检索维度
    凭空消失。这里让 boom 样本先完成一次真实检索再崩（patch _run_sample 的产物），
    断言聚合分母是 2 而不是 1。
    """
    orig_run_sample = eval_runner._run_sample
    crashed = {"done": False}

    async def crash_after_retrieval(sample, db_path, k, rounds, recalled):
        if sample["id"] == "t-001" and not crashed["done"]:
            crashed["done"] = True
            # 模拟「检索已经跑完、answer 还没生成」时崩掉：收集器里有真实结果
            rounds.append(["阿尔法笔记"])
            raise RuntimeError("检索之后崩了")
        return await orig_run_sample(sample, db_path, k, rounds, recalled)

    monkeypatch.setattr(eval_runner, "_run_sample", crash_after_retrieval)
    result = await run_eval(
        str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=FakeLLM()
    )

    by_id = {s.id: s for s in result.samples}
    assert "检索之后崩了" in by_id["t-001"].error
    assert by_id["t-001"].metrics["retrieval"]["hit_at_k"] == 1.0
    assert result.metrics["overall"]["errors"] == 1
    # 有检索标注的样本是 t-001（崩溃但有观测）与 t-002？t-002 无标注 → 分母仍为 1，
    # 但关键是崩溃的 t-001 **在** 分母里（修复前它被跳过，此处为 0 个样本）
    single = result.metrics["by_category"]["single-hop"]
    assert single["n"] == 1 and single["hit_at_k"] == 1.0


async def test_config_override_rejects_wrong_bool_type(tiny_dataset, tmp_path):
    """bool 字段只接受真 bool：`"false"` 是真值字符串（MIN-6）。

    Settings 没开 validate_assignment，裸 setattr 会原样接受字符串；下游
    `if settings.memory_enabled` 判真，于是「关掉记忆」的消融组实际是打开的，
    整组结论反过来，且全程不报错。数值字段同理（`context_max_tokens="8000"` 一旦
    参与算术就 TypeError）。
    """
    with pytest.raises(ValueError, match="bool"):
        await run_eval(
            str(tiny_dataset),
            config_overrides={"memory_enabled": "false"},
            db_path=str(tmp_path / "eval.db"),
            llm=FakeLLM(),
        )
    # 数值字段不接受字符串（bool 是 int 子类，也要挡住）
    with pytest.raises(ValueError, match="数值"):
        await run_eval(
            str(tiny_dataset),
            config_overrides={"context_max_tokens": "8000"},
            db_path=str(tmp_path / "eval.db"),
            llm=FakeLLM(),
        )
    with pytest.raises(ValueError, match="数值"):
        await run_eval(
            str(tiny_dataset),
            config_overrides={"context_max_tokens": True},
            db_path=str(tmp_path / "eval.db"),
            llm=FakeLLM(),
        )
    # 合法取值照常通过（int 给 float 也认）
    result = await run_eval(
        str(tiny_dataset),
        config_overrides={"memory_enabled": False, "context_max_tokens": 4000.0},
        db_path=str(tmp_path / "eval.db"),
        llm=FakeLLM(),
    )
    assert result.config["memory_enabled"] is False
    assert settings.memory_enabled is True  # 出了 with 块已恢复


async def test_settings_override_validation_is_unit_testable():
    """校验只看类型，不依赖跑一遍评测；直接构造也算用例的入口契约。"""
    from eval.runner import _settings_override

    _settings_override({"memory_enabled": True, "context_max_tokens": 100})  # 不抛
    _settings_override(None)  # 缺省不抛
    with pytest.raises(ValueError):
        _settings_override({"context_max_tokens": "100"})
    with pytest.raises(ValueError):
        _settings_override({"memory_enabled": 1})  # int 冒充 bool 也不行


# ---------- 消融矩阵 ----------


def test_ablation_groups_match_brief():
    assert set(ABLATION_GROUPS) == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert ABLATION_GROUPS["A"] == {
        "memory_enabled": True,
        "context_compaction_enabled": True,
        "retrieval_mode": "hybrid",
    }
    # F 是最小系统：记忆与压缩全关、纯向量
    assert ABLATION_GROUPS["F"]["memory_enabled"] is False
    assert ABLATION_GROUPS["F"]["context_compaction_enabled"] is False
    assert ABLATION_GROUPS["F"]["retrieval_mode"] == "vector"


def test_ablation_groups_full_table():
    """8 组配置逐项对照 brief 的三轴表（memory / compaction / retrieval）。

    逐项断言而不是只查几个字段：写错一格（比如 C 组漏关压缩）会让消融图看起来
    「压缩无贡献」，是这类矩阵最难从结果反推的错法。
    """
    expected = {
        # 组: (memory, compaction, retrieval)
        "A": (True, True, "hybrid"),
        "B": (False, True, "hybrid"),
        "C": (True, False, "hybrid"),
        "D": (True, True, "vector"),
        "E": (True, True, "bm25"),
        "F": (False, False, "vector"),
        "G": (True, False, "vector"),
        "H": (False, True, "bm25"),
    }
    assert list(ABLATION_GROUPS) == list(expected)  # 顺序也是产物的一部分
    for name, (memory, compaction, retrieval) in expected.items():
        group = ABLATION_GROUPS[name]
        assert group == {
            "memory_enabled": memory,
            "context_compaction_enabled": compaction,
            "retrieval_mode": retrieval,
        }, f"{name} 组配置与 brief 表格不一致：{group}"
    # B/C/D/E 相对 A 各只动一个旋钮（单因素对照；F/G/H 是组合组，不在此列）。
    # 一个都不动 = 该组是 A 的重复；动两个 = 分不清是谁的贡献
    fields = ("memory_enabled", "context_compaction_enabled", "retrieval_mode")
    for name in ("B", "C", "D", "E"):
        changed = [
            key
            for key in fields
            if ABLATION_GROUPS[name][key] != ABLATION_GROUPS["A"][key]
        ]
        assert len(changed) == 1, f"{name} 组相对 A 动了 {changed}，不是单因素对照"


async def test_ablation_rejects_unknown_group():
    with pytest.raises(ValueError, match="未知消融组"):
        await run_ablation("x", groups=["Z"])


async def test_ablation_with_empty_group_list_runs_nothing(tiny_dataset, tmp_path):
    """显式传空列表 = 一组都不跑（S2）。

    `groups or list(ABLATION_GROUPS)` 会把空列表当成 None，静默跑满 8 组——
    调用方想用空列表表达「什么都不做」时会拿到一堆意外产物。
    """
    results = await run_ablation(
        str(tiny_dataset), groups=[], llm=FakeLLM(), work_dir=tmp_path
    )
    assert results == {}
    # 一组都没跑：没有建任何组库
    assert not list(tmp_path.glob("group-*.db"))


async def test_ablation_with_unknown_sample_ids_still_returns_groups(
    tiny_dataset, tmp_path
):
    """筛完没样本时每组仍产出一条 n=0 的结果（S1：报告渲染不走到空分支）。

    这是 render_ablation_report 空分支注释里说的口径：CLI 的
    `--ablation --samples 不存在的id` 拿到的是 {"A": EvalResult(n=0)} 而非 {}，
    报告照常渲染（不是「没有可渲染的结果」）。
    """
    results = await run_ablation(
        str(tiny_dataset),
        groups=["A"],
        llm=FakeLLM(),
        work_dir=tmp_path,
        sample_ids=["no-such-sample"],
    )
    assert set(results) == {"A"}
    assert results["A"].metrics["total"] == 0
    assert "没有可渲染" not in render_ablation_report(results)
    assert "| A |" in render_ablation_report(results)


async def test_run_ablation_memory_differential(tiny_dataset, tmp_path):
    """A 组（有记忆）的偏好样本能召回记忆，B 组（无记忆）召不回——
    消融矩阵要能测出这个差。"""
    results = await run_ablation(
        str(tiny_dataset), groups=["A", "B"], llm=FakeLLM(), work_dir=tmp_path
    )
    assert results["A"].metrics["overall"]["memory_recall"] == 1.0
    assert results["B"].metrics["overall"]["memory_recall"] == 0.0
    assert results["B"].config["memory_enabled"] is False
    # 每组独立建库
    assert (tmp_path / "group-A.db").exists() and (tmp_path / "group-B.db").exists()


class _CompactionProbeLLM(FakeLLM):
    """回答内容反映「本轮 prompt view 里有没有历史摘要」，用来判定压缩是否真触发。

    `chat`（摘要请求）返回一段固定摘要；`chat_stream`（回答）按是否看到
    `[历史摘要]` 前缀的 system 消息产出不同文本。压缩没触发时两者输出逐字节相同，
    正好复现「压缩轴完全无效」的现象。
    """

    SUMMARY = "用户之前在聊装饰器与其他话题。"

    async def chat(self, messages, tools=None):
        return ChatResult(text=self.SUMMARY)

    async def chat_stream(self, messages, tools=None):
        compacted = any(
            m.role == "system" and m.content.startswith("[历史摘要]") for m in messages
        )
        if not any(m.role == "tool" for m in messages):
            yield StreamChunk(
                finish=True,
                tool_calls=[
                    ToolCall(id="c1", name="search_knowledge", arguments={"query": "苹果"})
                ],
            )
        else:
            yield StreamChunk(text_delta=f"压缩={'生效' if compacted else '未生效'}")
            yield StreamChunk(finish=True, tool_calls=[])


async def test_compaction_axis_actually_differs_on_long_session_samples(
    stub_embeddings, tmp_path
):
    """A 组（压缩开）与 C 组（压缩关）在评测集上必须不同。

    这是 M3 的回归：`context_compaction_enabled` 只在历史条数 > 10 时触发，而修复前
    评测集每条样本最多 2 条历史，永远达不到阈值——实测 A/C、D/G 输出逐字节相同，
    等于这条消融轴没有测量。评测集补了 3 条 14 条历史的长会话样本（eval-036/037/038）
    后，只有 A 组的 prompt view 里会出现历史摘要。

    直接跑真实评测集的长会话子集（embedding 走桩，不发网络请求）。
    """
    long_ids = ["eval-036", "eval-037", "eval-038"]
    dataset = load_dataset(DATASET_PATH)
    by_id = {s["id"]: s for s in dataset["samples"]}
    for sample_id in long_ids:  # 前提：这几条样本确实超过压缩阈值
        assert len(by_id[sample_id]["sessions"]) > settings.context_compaction_threshold

    results = {}
    for name in ("A", "C"):
        results[name] = await run_eval(
            DATASET_PATH,
            config_overrides=ABLATION_GROUPS[name],
            db_path=str(tmp_path / f"group-{name}.db"),
            llm=_CompactionProbeLLM(),
            sample_ids=long_ids,
            concurrency=1,
        )

    answer_a = [s.answer for s in results["A"].samples]
    answer_c = [s.answer for s in results["C"].samples]
    assert answer_a == ["压缩=生效"] * 3, f"A 组（压缩开）未见历史摘要：{answer_a}"
    assert answer_c == ["压缩=未生效"] * 3, f"C 组（压缩关）不该有摘要：{answer_c}"
    assert answer_a != answer_c, "压缩轴的两组输出完全相同：这条消融轴没有测量任何东西"


# ---------- 报告 ----------


def _table_columns(md: str) -> list[tuple[int, int]]:
    """Markdown 里每张表的 (表头列数, 分隔行列数)，按出现顺序。

    表头行以 `| ` 开头且下一行是分隔行（`|---`/`|:` 起头）时算作一张表。
    """
    lines = md.splitlines()
    out: list[tuple[int, int]] = []
    for i in range(len(lines) - 1):
        header, sep = lines[i].strip(), lines[i + 1].strip()
        if not header.startswith("|"):
            continue
        cells = [c for c in sep.strip("|").split("|")]
        if not cells or not all(set(c.strip()) <= {"-", ":"} and "-" in c for c in cells):
            continue
        out.append((len(header.strip("|").split("|")), len(cells)))
    return out


async def test_render_and_save_report(tiny_dataset, tmp_path):
    result = await run_eval(
        str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=FakeLLM()
    )
    md = render_report(result)
    assert "## 总体指标" in md and "## 分类指标" in md and "## 失败案例" in md
    assert result.git_commit in md

    path = save_report(md, reports_dir=tmp_path / "reports")
    assert path.read_text(encoding="utf-8") == md


async def test_render_report_tables_have_matching_columns(tiny_dataset, tmp_path):
    """每张表的表头列数必须等于分隔行列数。

    修复前分类表的表头是 8 列（前缀了「类别」）而分隔行只有 7 列，Markdown 渲染时
    多出来的「类别」被截掉——表看起来少一列，数据整体错位。这里对有数据的报告做
    逐表校验，比断言某个字符串更贴近渲染结果。
    """
    result = await run_eval(
        str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=FakeLLM()
    )
    tables = _table_columns(render_report(result))
    assert len(tables) == 2, f"应有总体指标与分类指标两张表：{tables}"
    for header_cols, sep_cols in tables:
        assert header_cols == sep_cols, f"表头 {header_cols} 列 vs 分隔行 {sep_cols} 列"

    # 分类表的第一列确实是「类别」，且每行数据列数与表头一致
    lines = render_report(result).splitlines()
    category_header = next(line for line in lines if line.startswith("| 类别"))
    assert len(category_header.strip("|").split("|")) == 8
    rows = [
        line
        for line in lines
        if line.startswith("| single-hop") or line.startswith("| preference")
    ]
    assert rows
    for row in rows:
        assert len(row.strip("|").split("|")) == 8, row


async def test_render_ablation_report(tiny_dataset, tmp_path):
    results = await run_ablation(
        str(tiny_dataset), groups=["A", "B"], llm=FakeLLM(), work_dir=tmp_path
    )
    md = render_ablation_report(results)
    assert "## 消融对比" in md
    assert "| A |" in md and "| B |" in md
    assert "memory=False" in md
    # 消融表 10 列（组/说明/配置 + 7 个指标），表头与分隔行必须一致
    tables = _table_columns(md)
    assert tables, "消融报告里没有表格"
    for header_cols, sep_cols in tables:
        assert header_cols == sep_cols, f"表头 {header_cols} 列 vs 分隔行 {sep_cols} 列"
    ablations_row = [line for line in md.splitlines() if line.startswith("| A |")]
    assert len(ablations_row[0].strip("|").split("|")) == 10


def test_render_ablation_report_with_no_results():
    """空字典不能抛裸 StopIteration，要给一段人可读的提示。"""
    md = render_ablation_report({})
    assert isinstance(md, str) and md.strip()
    assert "消融" in md


async def test_save_uses_distinct_directories_for_same_second(tmp_path):
    """同一秒内保存多组结果不能互相覆盖（消融 8 组常落在同一秒）。

    修复前目录名只有秒级时间戳，8 组全写进同一个目录，只留最后一组的 result.json，
    其余 7 组的逐样本 trace 全丢。这里用同一秒内的多次 save 复现，直接查目录数。
    """
    import time as _time

    result = EvalResult(
        dataset="d.json",
        git_commit="abc1234",
        config={},
        created_at="2026-09-25T10:00:00",
        dataset_version=1,
    )
    results_dir = tmp_path / "results"
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    paths = [result.save(results_dir=results_dir, name=f"group-{n}") for n in "ABCD"]

    dirs = sorted(p.parent.name for p in paths)
    assert dirs == [f"{stamp}-group-{n}" for n in "ABCD"], f"目录名撞车：{dirs}"
    assert len({p.parent for p in paths}) == 4, "同秒保存互相覆盖"
    for path in paths:
        assert path.is_file() and path.name == "result.json"

    # 不带 name 时保持原样（单组评测的默认行为不变）
    plain = result.save(results_dir=results_dir)
    assert plain.parent.name == stamp


async def test_save_records_dataset_version(tiny_dataset, tmp_path):
    """dataset 的 version 要进结果快照（复现时要能对上评测集版本）。"""
    result = await run_eval(
        str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=FakeLLM()
    )
    assert result.dataset_version == 7  # tiny_dataset 里的 version

    path = result.save(results_dir=tmp_path / "results")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["dataset_version"] == 7


async def test_git_commit_is_real_hash(tmp_path, monkeypatch):
    """可复现性快照里的 commit 必须是真 hash，不是 unknown 占位（MIN-7）。

    git 默认按**进程当前目录**找仓库，从仓库外跑 pytest 会拿不到 hash、快照静默退化成
    unknown。所以这里把 cwd 切到仓库外再调用——修复前必然返回 unknown（测试在仓库里
    跑也会红）。
    """
    from eval.runner import git_commit

    monkeypatch.chdir(tmp_path)  # 仓库外的目录
    commit = git_commit()
    assert commit != "unknown", "从仓库外调用就退化：git 没按仓库根找（MIN-7）"
    assert len(commit) >= 7 and all(c in "0123456789abcdef" for c in commit)

    # 仓库内调用结果一致（cwd 不影响结果）
    assert git_commit() == commit


# ---------- retrieval_mode 配置接入 ----------


async def test_retrieval_mode_is_wired_to_runtime(tmp_path, monkeypatch):
    """settings.retrieval_mode 要传到 runtime 工具执行的 hybrid_search 调用上。"""
    await init_db(tmp_path / "app.db", DIM)
    seen: list[str] = []

    async def fake_search(query, k=8, mode="hybrid", db_path=None):
        seen.append(mode)
        return []

    monkeypatch.setattr(runtime, "hybrid_search", fake_search)
    monkeypatch.setattr(settings, "retrieval_mode", "bm25")

    call = ToolCall(id="c1", name="search_knowledge", arguments={"query": "苹果"})
    await runtime.execute_tool(call, str(tmp_path / "app.db"))

    assert seen == ["bm25"]
