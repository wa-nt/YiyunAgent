"""T10 评测 runner：加载评测集，逐条跑 run_agent 并按多维度评分。

流程（每条样本）：
1. 准备：建会话、写入历史消息、按样本预置记忆（seed_memories）
2. 执行：消费 run_agent 的事件流直到 done/error，聚合回答文本
3. 评分：检索（Hit@k/MRR/Recall@k）、回答（关键词覆盖率）、记忆（召回准确率）

注入点（真实运行与测试共用同一套机制）：
- llm 参数：替换 runtime.get_llm 的返回值；测试传 FakeLLM，真实评测传 None（走配置）
- config_overrides：评测期间临时改写 settings（消融矩阵的开关来源），结束后恢复
- 检索/记忆观测：包装 runtime.hybrid_search 与 runtime.recall_memories 记录本轮
  实际检索到的标识符与召回的记忆文本，用 contextvar 按样本隔离（并发安全）

检索标识符约定：expected_chunks 里写**文档标题**（chunk 自增 id 在重新 ingest
后不稳定，不能进评测集）。runner 把每个检索到的 chunk 映射为标题参与比对。

用法：
    python -m eval.runner --dataset eval/dataset/eval.json
    python -m eval.runner --ablation          # 跑 8 组消融并生成对比报告
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import logging
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.agent import runtime
from app.config import settings
from app.db import get_db, init_db
from app.ingest.pipeline import ingest
from eval.metrics import aggregate_metrics, answer_metrics, memory_metrics, retrieval_metrics

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "eval/dataset/eval.json"
DEFAULT_DB_PATH = "data/eval.db"
RESULTS_DIR = Path("eval/results")
# 消融矩阵里参与快照的开关字段，结果目录里的配置快照只记这些（定价等无关字段不记）
CONFIG_SNAPSHOT_FIELDS = (
    "memory_enabled",
    "context_compaction_enabled",
    "context_tool_clean_enabled",
    "context_token_budget_enabled",
    "retrieval_mode",
    "tracing_enabled",
)

# 当前样本的观测收集器：检索到的标识符 / 召回的记忆文本。按样本隔离（contextvars
# 随 asyncio.Task 上下文复制，并发样本互不串扰）
_current_retrieved: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "eval_retrieved", default=None
)
_current_recalled: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "eval_recalled", default=None
)


@dataclass
class SampleResult:
    id: str
    category: str
    query: str
    answer: str
    retrieved: list[str] = field(default_factory=list)
    recalled_memory: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    dataset: str
    git_commit: str
    config: dict[str, Any]
    created_at: str
    samples: list[SampleResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        """落盘到 eval/results/{timestamp}/result.json，含 git hash 与配置快照。"""
        out_dir = results_dir / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "result.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def load_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """加载评测集 JSON：{version, docs: [...], samples: [...]}。"""
    path = Path(dataset_path)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not dataset.get("samples"):
        raise ValueError(f"评测集为空或缺少 samples 字段：{path}")
    return dataset


def git_commit() -> str:
    """当前仓库的 commit hash（可复现性快照）；不在 git 仓库里时返回 unknown。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_snapshot() -> dict[str, Any]:
    return {name: getattr(settings, name) for name in CONFIG_SNAPSHOT_FIELDS}


class _settings_override:
    """评测期间临时改写 settings，退出时恢复原值（消融开关的载体）。"""

    def __init__(self, overrides: dict[str, Any] | None) -> None:
        self._overrides = overrides or {}
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> None:
        for name, value in self._overrides.items():
            self._saved[name] = getattr(settings, name)
            setattr(settings, name, value)

    def __exit__(self, *exc: object) -> None:
        for name, value in self._saved.items():
            setattr(settings, name, value)


def _install_observers() -> dict[str, Any]:
    """包装 runtime 的检索与记忆召回，记录观测值；返回原函数供恢复。

    包装的是 runtime 模块属性（run_agent 内部按模块属性查找），与测试里
    monkeypatch runtime.hybrid_search 的口径一致。
    """
    original_search = runtime.hybrid_search
    original_recall = runtime.recall_memories

    async def recording_search(query, k=8, mode="hybrid", db_path=None):
        chunks = await original_search(query, k=k, mode=mode, db_path=db_path)
        sink = _current_retrieved.get()
        if sink is not None:
            sink.extend(c.title or f"chunk {c.chunk_id}" for c in chunks)
        return chunks

    async def recording_recall(user_message, db_path=None):
        text = await original_recall(user_message, db_path)
        sink = _current_recalled.get()
        if sink is not None and text:
            sink.append(text)
        return text

    runtime.hybrid_search = recording_search
    runtime.recall_memories = recording_recall
    return {"hybrid_search": original_search, "recall_memories": original_recall}


def _restore_observers(originals: dict[str, Any]) -> None:
    runtime.hybrid_search = originals["hybrid_search"]
    runtime.recall_memories = originals["recall_memories"]


async def _seed_sample(sample: dict[str, Any], session_id: str, db_path: str) -> None:
    """准备阶段：建会话、写历史消息、预置记忆。"""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    async with get_db(db_path) as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
        for msg in sample.get("sessions", []):
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, msg["role"], msg["content"], now),
            )
        for mem in sample.get("seed_memories", []):
            await conn.execute(
                "INSERT INTO memories (kind, content, confidence, source, created_at, "
                "updated_at, status) VALUES (?, ?, ?, ?, ?, ?, 'active')",
                (
                    mem.get("kind", "preference"),
                    mem["content"],
                    mem.get("confidence", 0.9),
                    f"eval:{sample['id']}",
                    now,
                    now,
                ),
            )
        await conn.commit()


async def _run_sample(
    sample: dict[str, Any], db_path: str, k: int
) -> SampleResult:
    """执行阶段 + 评分阶段：消费 run_agent 事件流直到终态，然后算指标。"""
    session_id = f"eval-{sample['id']}-{uuid.uuid4().hex[:6]}"
    await _seed_sample(sample, session_id, db_path)

    retrieved: list[str] = []
    recalled: list[str] = []
    token_r = _current_retrieved.set(retrieved)
    token_m = _current_recalled.set(recalled)
    answer = ""
    error: str | None = None
    try:
        async for event in runtime.run_agent(session_id, sample["query"], db_path):
            if event.type == "text_delta":
                answer += event.data.get("text", "")
            elif event.type == "done":
                answer = event.data.get("text", answer)
            elif event.type == "error":
                answer = event.data.get("text", answer)
                error = event.data.get("message", "unknown error")
    finally:
        _current_retrieved.reset(token_r)
        _current_recalled.reset(token_m)
    # 等本轮的记忆写入结束：下一条样本的预置记忆与本轮抽取互不污染
    await runtime.drain_memory_writes()

    expected_chunks = sample.get("expected_chunks", [])
    result = SampleResult(
        id=sample["id"],
        category=sample["category"],
        query=sample["query"],
        answer=answer,
        retrieved=retrieved,
        recalled_memory="\n".join(recalled) or None,
        error=error,
    )
    result.metrics = {
        "retrieval": (
            retrieval_metrics(retrieved, expected_chunks, k) if expected_chunks else None
        ),
        "answer": answer_metrics(answer, sample.get("expected_answer_contains", [])),
        "memory": memory_metrics(
            result.recalled_memory, sample.get("expected_memories", [])
        ),
    }
    return result


async def run_eval(
    dataset_path: str,
    config_overrides: dict[str, Any] | None = None,
    db_path: str | None = None,
    llm: Any | None = None,
    sample_ids: list[str] | None = None,
    concurrency: int = 4,
    k: int = 8,
) -> EvalResult:
    """跑一遍评测集，返回聚合结果。

    - config_overrides：评测期间生效的 settings 改写（消融矩阵用），结束恢复
    - db_path：评测库；默认 settings.db_path。llm trace 只写 settings.db_path
    （进程级口径），要收集 trace 就把 settings.db_path 指到同一个库
    - llm：注入的模型桩；None 时走 app.llm.get_llm 的真实配置
    - sample_ids：只跑指定子集（小规模冒烟用）
    - concurrency：asyncio.gather 的并发上限，避免打满 rate limit
    """
    dataset = load_dataset(dataset_path)
    samples = dataset["samples"]
    if sample_ids is not None:
        wanted = set(sample_ids)
        samples = [s for s in samples if s["id"] in wanted]
    path = db_path or settings.db_path
    docs_dir = Path(dataset_path).parent / "docs"

    with _settings_override(config_overrides):
        await init_db(path)
        for doc in dataset.get("docs", []):
            await ingest(docs_dir / doc, path)

        originals = _install_observers()
        original_get_llm = runtime.get_llm
        if llm is not None:
            runtime.get_llm = lambda: llm
        semaphore = asyncio.Semaphore(concurrency)

        async def guarded(sample: dict[str, Any]) -> SampleResult:
            async with semaphore:
                return await _run_sample(sample, path, k)

        try:
            results = await asyncio.gather(*(guarded(s) for s in samples))
        finally:
            _restore_observers(originals)
            runtime.get_llm = original_get_llm

        # 快照要在 override 生效期内拍：出了 with 块 settings 已恢复成默认值
        snapshot = config_snapshot()

    sample_dicts = [
        {"id": r.id, "category": r.category, "error": r.error, "metrics": r.metrics}
        for r in results
    ]
    return EvalResult(
        dataset=str(dataset_path),
        git_commit=git_commit(),
        config=snapshot,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        samples=list(results),
        metrics=aggregate_metrics(sample_dicts),
    )


async def _main_async(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="T10 评测 runner")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="评测集 JSON 路径")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="评测用 SQLite 库")
    parser.add_argument("--concurrency", type=int, default=4, help="并发上限")
    parser.add_argument("--samples", nargs="*", default=None, help="只跑指定样本 id")
    parser.add_argument("--ablation", action="store_true", help="跑 8 组消融矩阵")
    parser.add_argument("--no-save", action="store_true", help="不写入 eval/results/")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.ablation:
        from eval.ablation import run_ablation
        from eval.report import render_ablation_report, save_report

        results = await run_ablation(
            args.dataset, concurrency=args.concurrency, sample_ids=args.samples
        )
        if not args.no_save:
            for name, result in results.items():
                path = result.save()
                logger.info("组 %s 结果已保存：%s", name, path)
            report_path = save_report(render_ablation_report(results))
            logger.info("消融报告已保存：%s", report_path)
        for name, result in results.items():
            overall = result.metrics["overall"]
            print(
                f"[{name}] hit@k={overall['hit_at_k']} mrr={overall['mrr']} "
                f"coverage={overall['keyword_coverage']} "
                f"memory={overall['memory_recall']} errors={overall['errors']}"
            )
        return

    result = await run_eval(
        args.dataset,
        db_path=args.db_path,
        sample_ids=args.samples,
        concurrency=args.concurrency,
    )
    if not args.no_save:
        logger.info("结果已保存：%s", result.save())
    overall = result.metrics["overall"]
    print(
        f"hit@k={overall['hit_at_k']} mrr={overall['mrr']} "
        f"coverage={overall['keyword_coverage']} "
        f"memory={overall['memory_recall']} errors={overall['errors']}"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
