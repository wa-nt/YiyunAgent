"""T10 评测框架自身的测试：指标纯函数、评测集完整性、runner 端到端（全 mock）、
消融矩阵与报告。全程不发真实 API 请求：embedding 与 LLM 都是桩。"""

import json

import pytest

from app.agent import runtime
from app.config import settings
from app.db import init_db
from app.llm.types import ChatResult, StreamChunk, ToolCall
from app.memory import writer as memory_writer
from app.retrieval.bm25_search import invalidate
from eval import metrics as eval_metrics
from eval.ablation import ABLATION_GROUPS, run_ablation
from eval.report import render_ablation_report, render_report, save_report
from eval.runner import load_dataset, run_eval

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


def test_answer_metrics_coverage():
    full = eval_metrics.answer_metrics("答案是 functools.wraps 和 __name__", ["functools.wraps", "__name__"])
    assert full["keyword_coverage"] == 1.0 and full["missed"] == []
    half = eval_metrics.answer_metrics("只有 functools.wraps", ["functools.wraps", "lru_cache"])
    assert half["keyword_coverage"] == 0.5 and half["missed"] == ["lru_cache"]
    # 无标注约束不惩罚
    assert eval_metrics.answer_metrics("随便", [])["keyword_coverage"] == 1.0


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


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """两条文档、三条样本的最小评测集 + 全套桩：embedding / 记忆抽取 / 8 维库。"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# 阿尔法笔记\n苹果是一种水果，阿尔法笔记记录水果。", encoding="utf-8")
    (docs_dir / "b.md").write_text("# 贝塔笔记\n贝塔笔记记录编程，与水果无关。", encoding="utf-8")
    dataset = {
        "version": 1,
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
                "sessions": [],
            },
        ],
    }
    dataset_path = tmp_path / "eval.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(settings, "embed_dim", DIM)
    monkeypatch.setattr("app.ingest.pipeline.embed_texts", _fake_embed)
    monkeypatch.setattr("app.retrieval.hybrid.embed_texts", _fake_embed)
    monkeypatch.setattr(memory_writer, "get_llm", lambda: _SilentWriterLLM())
    invalidate()
    yield dataset_path
    invalidate()


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


async def test_ablation_rejects_unknown_group():
    with pytest.raises(ValueError, match="未知消融组"):
        await run_ablation("x", groups=["Z"])


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


# ---------- 报告 ----------


async def test_render_and_save_report(tiny_dataset, tmp_path):
    result = await run_eval(
        str(tiny_dataset), db_path=str(tmp_path / "eval.db"), llm=FakeLLM()
    )
    md = render_report(result)
    assert "## 总体指标" in md and "## 分类指标" in md and "## 失败案例" in md
    assert result.git_commit in md

    path = save_report(md, reports_dir=tmp_path / "reports")
    assert path.read_text(encoding="utf-8") == md


async def test_render_ablation_report(tiny_dataset, tmp_path):
    results = await run_ablation(
        str(tiny_dataset), groups=["A", "B"], llm=FakeLLM(), work_dir=tmp_path
    )
    md = render_ablation_report(results)
    assert "## 消融对比" in md
    assert "| A |" in md and "| B |" in md
    assert "memory=False" in md


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
