"""T10 评测 runner：加载评测集，逐条跑 run_agent 并按多维度评分。

流程（每条样本）：
1. 准备：建会话、写入历史消息、按样本预置记忆（seed_memories）
2. 执行：消费 run_agent 的事件流直到 done/error，聚合回答文本
3. 评分：检索（Hit@k/MRR/Recall@k）、回答（关键词覆盖率）、记忆（召回准确率）

注入点（真实运行与测试共用同一套机制）：
- llm 参数：替换 runtime.get_llm 的返回值；测试传 FakeLLM，真实评测传 None（走配置）
- config_overrides：评测期间临时改写 settings（消融矩阵的开关来源），结束后恢复
- 检索/记忆观测：包装 runtime.hybrid_search 与 runtime.recall_memories 记录**每次调用**
  实际检索到的标识符（按轮次分段）与召回的记忆文本，用 contextvar 按样本隔离（并发安全）

检索标识符约定：expected_chunks 里写**文档标题**（chunk 自增 id 在重新 ingest
后不稳定，不能进评测集）。runner 把每个检索到的 chunk 映射为标题参与比对。

每个样本可能有多轮检索（ReAct 循环），观测器按**每次检索调用**分段记录，评分时
对每篇期望文档取「在任一轮中的最好排名」（见 metrics.retrieval_metrics_rounds）：
把多轮结果拼成一个列表再按 k 截断会让第 2 轮的命中系统性落到 k 之后。

db_path 是**一次性的 scratch 库**：run_eval 开始时删掉它（含 -wal/-shm）再重新 ingest，
否则每次运行都会把同一批文档追加一遍（ingest 是纯追加），语料翻倍、首跑数字不可复现。
默认值与 CLI 一致（data/eval.db），**不回落 settings.db_path**——那是应用在用的知识库。

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
from typing import Any, Literal, get_args, get_origin

from app.agent import runtime
from app.config import Settings, settings
from app.db import get_db, init_db
from app.ingest.pipeline import ingest
from app.retrieval.bm25_search import invalidate
from app.tracing import drain_traces
from eval.metrics import (
    aggregate_metrics,
    answer_metrics,
    memory_metrics,
    retrieval_metrics_rounds,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "eval/dataset/eval.json"
DEFAULT_DB_PATH = "data/eval.db"
RESULTS_DIR = Path("eval/results")
# SQLite 除主库文件外的旁路文件。删库时要一起删：WAL 里可能有未回写的页，
# 只删主库会让新库从旧 WAL 里「继承」上次运行的残留行
DB_SIDECARS = ("-wal", "-shm")
# 消融矩阵里参与快照的开关字段，结果目录里的配置快照只记这些（定价等无关字段不记）
CONFIG_SNAPSHOT_FIELDS = (
    "memory_enabled",
    "context_compaction_enabled",
    "context_tool_clean_enabled",
    "context_token_budget_enabled",
    "retrieval_mode",
    "tracing_enabled",
    # T11 起 run_agent 会按用户消息触发 skill：命中时会给 prompt view 多加一条
    # system 消息（正文）与专用工具，直接影响压缩轴与 token 预算的对照。评测结果
    # 要能反查「那一次跑的时候 skill 开着还是关着」，所以进快照
    "skills_enabled",
)

# 当前样本的观测收集器：按**每次检索调用**分段记录到的标识符 / 召回的记忆文本。
# 按样本隔离（contextvars 随 asyncio.Task 上下文复制，并发样本互不串扰）
_current_retrieved: contextvars.ContextVar[list[list[str]] | None] = (
    contextvars.ContextVar("eval_retrieved", default=None)
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
    dataset_version: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, results_dir: Path = RESULTS_DIR, name: str | None = None) -> Path:
        """落盘到 eval/results/{timestamp}[-{name}]/result.json，含 git hash 与配置快照。

        目录名带 name（消融矩阵传组名）：8 组消融通常落在同一秒里，只有秒级时间戳时
        后面的组会覆盖前面组的目录，只留下最后一组的 result.json 与逐样本 trace。
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = results_dir / (f"{stamp}-{name}" if name else stamp)
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


def _same_file(a: str | Path, b: str | Path) -> bool:
    """两个路径是否指向同一个库文件（解析成绝对路径后比较）。

    resolve() 会展开 `..`、相对路径与符号链接，`data/app.db` 与
    `./data/../data/app.db` 因此能判定为同一个文件。文件不存在时 resolve
    仍能算出规范路径（strict=False），所以守卫在库还没建起来时也有效。
    """
    return Path(a).resolve() == Path(b).resolve()


async def _reset_db(db_path: str | Path) -> None:
    """删掉评测库及其 WAL 旁路文件，让本次运行的 ingest 从空库开始。

    ingest 是纯追加、不带判重，不清库就会把同一批文档一遍遍灌进去（语料翻倍，
    hit@k 随运行次数单调下滑）。删文件比「按 source 幂等」干净：chunk 自增 id 也
    跟着回到初始值，两次运行的 trace 可以直接逐字节对比。

    本函数会**不可逆地删掉一个 SQLite 库**，所以入口先拒绝在用的应用库：调用方
    显式传了 settings.db_path（用户的知识库）时 raise，而不是把笔记删完再说。
    不能靠「默认参数已经是 data/eval.db」兜住——那只挡住了不传参的路径。

    删之前必须等在途的后台写入结束，否则 Windows 上直接 PermissionError
    （「另一个程序正在使用此文件」），Linux 上则是这些写入悄悄落进已删除的 inode：
    - 记忆写入（runtime 的 fire-and-forget 抽取）会往这个库写，且它**会产生 llm
      trace**，所以先等它、再等 trace（顺序反过来会漏掉它新产生的 trace 任务）
    - trace 写入（T9 埋点）：工具 trace 落的正是这个评测库

    删完顺手失效进程级的 BM25 索引：_indexes 按绝对路径缓存，删库后若本次 ingest
    没有产出任何 chunk（早退不调 invalidate），旧索引会原样留着，检索静默返回已删文档。
    """
    if _same_file(db_path, settings.db_path):
        raise ValueError(
            f"评测不允许清应用库：{db_path} 就是 settings.db_path（在用知识库）。"
            "评测用一次性 scratch 库，如 data/eval.db"
        )
    await runtime.drain_memory_writes()
    await drain_traces()
    path = Path(db_path)
    for suffix in ("", *DB_SIDECARS):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    invalidate(path)


def _error_result(
    sample: dict[str, Any],
    exc: BaseException,
    rounds: list[list[str]] | None = None,
    k: int = 8,
) -> SampleResult:
    """样本级异常 → 一条失败样本记录（与逐样本记错的口径一致）。

    检索指标按**崩溃前已观测到的检索轮次**算，而不是一律给 None：样本崩在
    检索之后（例如 LLM 第二轮挂掉）时，已经发生的检索是真实证据，丢掉它会让
    「崩溃样本」在 retrieval 维度被整体跳过，只统计活下来的样本（生存者偏差）。
    一轮检索都没发生（崩在准备阶段）才给 None，与「该样本不考核检索」区分开。

    回答维度反过来：没有完成回答就是 0 分，不管样本标没标 expected_contains——
    不能因为「无标注约束」的默认值（1.0）把崩溃样本算成满分。这两条口径不冲突：
    retrieval 记的是「已发生的检索事实」，answer 记的是「回答的完成度」，前者
    有证据就采信，后者没有产出就该罚。

    各字段用 .get 读：走到这里的原因可能就是样本字段缺失（构造错误的评测集），
    兜底函数自己再抛一次 KeyError 会把「记下来继续跑」变成「整轮炸掉」。
    """
    observed = list(rounds or [])
    expected_chunks = sample.get("expected_chunks", [])
    return SampleResult(
        id=sample.get("id", "<unknown>"),
        category=sample.get("category", "unknown"),
        query=sample.get("query", ""),
        answer="",
        retrieved=[title for rnd in observed for title in rnd],
        error=f"{type(exc).__name__}: {exc}",
        metrics={
            "retrieval": (
                retrieval_metrics_rounds(observed, expected_chunks, k)
                if expected_chunks and observed
                else None
            ),
            "answer": {
                "keyword_coverage": 0.0,
                "matched": [],
                "missed": list(sample.get("expected_answer_contains", [])),
                "violated": [],
            },
            "memory": memory_metrics(None, sample.get("expected_memories", [])),
        },
    )


def git_commit() -> str:
    """当前仓库的 commit hash（可复现性快照）；不在 git 仓库里时返回 unknown。

    cwd 显式指向本文件所在的仓库根：git 默认按**进程当前目录**找仓库，从仓库外
    跑 pytest（或从别处调用本函数）会拿不到 hash，快照静默退化成 unknown。
    """
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=root,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_snapshot() -> dict[str, Any]:
    return {name: getattr(settings, name) for name in CONFIG_SNAPSHOT_FIELDS}


def _literal_values(name: str) -> tuple[Any, ...] | None:
    """settings 字段是 Literal 时返回它的允许取值，否则 None。

    校验范围只看 Literal：这类字段的取值是**闭集**（retrieval_mode 的
    vector/bm25/hybrid），拼错一个字母没有「退化为默认」的合理语义，只会在下游
    炸成 500。
    """
    annotation = Settings.model_fields[name].annotation
    return get_args(annotation) if get_origin(annotation) is Literal else None


def _primitive_kind(name: str) -> str | None:
    """settings 字段的期望原始类型：'bool' / 'number' / None（不校验）。

    str 与 Optional[...] 字段不设守卫：它们的非法值要么当场被下游拒绝，要么本就
    宽松（如 db_path 接受任意路径字符串）。bool 与数值要管，是因为裸 setattr 绕过
    pydantic、而 Settings 没开 validate_assignment：`memory_enabled="false"` 会被
    当成非空字符串（真值！）静默打开开关，整组消融的结论跟着错，且不会报任何错。
    """
    annotation = Settings.model_fields[name].annotation
    if annotation is bool:
        return "bool"
    if annotation in (int, float):
        return "number"
    return None


class _settings_override:
    """评测期间临时改写 settings，退出时恢复原值（消融开关的载体）。

    入口做三道校验，都是**评测开始前**报错：
    - 字段必须存在（裸 setattr 绕过 pydantic，拼错字段名会被 pydantic 当场拒绝，
      不校验的话报错点会漂到 __enter__ 的半途，已改的字段留在原地）
    - Literal 字段的取值必须在允许集合里（否则 hybrid_search 抛 ValueError，
      经 asyncio.gather 冒成整轮评测异常）
    - bool / 数值字段的类型必须对（`"false"` 是真值字符串，会把「关掉某开关」的
      消融组静默变成「打开」，且全程不报错——最难从结果反推的错法）
    """

    def __init__(self, overrides: dict[str, Any] | None) -> None:
        self._overrides = overrides or {}
        self._saved: dict[str, Any] = {}
        self._validate()

    def _validate(self) -> None:
        for name, value in self._overrides.items():
            if name not in Settings.model_fields:
                raise ValueError(f"未知的配置字段：{name!r}")
            allowed = _literal_values(name)
            if allowed is not None and value not in allowed:
                raise ValueError(
                    f"配置 {name} 的取值非法：{value!r}，允许：{list(allowed)}"
                )
            kind = _primitive_kind(name)
            # bool 是 int 的子类，判断顺序不能反：先排除 bool 再看数值
            if kind == "bool" and not isinstance(value, bool):
                raise ValueError(
                    f"配置 {name} 需要 bool，得到 {type(value).__name__}：{value!r}"
                )
            if kind == "number" and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"配置 {name} 需要数值，得到 {type(value).__name__}：{value!r}"
                )

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
            # 每次调用记一段：多轮检索的排名要按轮次分开算（见 retrieval_metrics_rounds）
            sink.append([c.title or f"chunk {c.chunk_id}" for c in chunks])
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
    sample: dict[str, Any],
    db_path: str,
    k: int,
    rounds: list[list[str]],
    recalled: list[str],
) -> SampleResult:
    """执行阶段 + 评分阶段：消费 run_agent 事件流直到终态，然后算指标。

    run_agent 内部的故障（LLM 异常、工具失败）会以 error 事件的形式到达这里，记进
    SampleResult.error；本函数自己不捕获异常，样本级的任何故障由调用方 guarded 兜住
    （见 run_eval），保证「一个样本炸掉不影响整轮」的口径只有一处实现。

    rounds / recalled 是**调用方持有的观测收集器**，本函数把 contextvar 指向它们后
    才开跑：崩溃时 guarded 仍能读到崩溃前已记录的检索轮次与召回文本，用于
    _error_result 的评分（自建列表的话，异常一抛出就再也拿不到那些观测了）。
    """
    session_id = f"eval-{sample['id']}-{uuid.uuid4().hex[:6]}"
    answer = ""
    error: str | None = None
    token_r = _current_retrieved.set(rounds)
    token_m = _current_recalled.set(recalled)
    try:
        await _seed_sample(sample, session_id, db_path)
        async for event in runtime.run_agent(session_id, sample["query"], db_path):
            if event.type == "text_delta":
                answer += event.data.get("text", "")
            elif event.type in ("done", "error"):
                answer = event.data.get("text", answer)
                if event.type == "error":
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
        # 扁平化只用于产物可读（trace 里看到实际检索到的顺序）；评分按 rounds 分轮算
        retrieved=[title for rnd in rounds for title in rnd],
        recalled_memory="\n".join(recalled) or None,
        error=error,
    )
    result.metrics = {
        "retrieval": (
            retrieval_metrics_rounds(rounds, expected_chunks, k)
            if expected_chunks
            else None
        ),
        "answer": answer_metrics(
            answer,
            sample.get("expected_answer_contains", []),
            sample.get("expected_answer_excludes", []),
        ),
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

    - config_overrides：评测期间生效的 settings 改写（消融矩阵用），结束恢复；
      取值非法（未知字段 / Literal 集合外的值 / bool 与数值字段类型不对）在这里就
      报错，不等到检索或读了配置之后才发现
    - db_path：评测用的**一次性 scratch 库**，默认 data/eval.db（CLI 同默认）。每次运行
      开始时连同 -wal/-shm 一起删掉重建：ingest 是纯追加，不清库的话第二次运行会把同一
      批文档再灌一遍，语料翻倍、首跑数字不可复现（实测三次运行 hit@k 0.82→0.54→0.36）。
      **传在用的应用库（settings.db_path）会抛 ValueError**，不会把用户笔记删掉
    - llm：注入的模型桩；None 时走 app.llm.get_llm 的真实配置。桩**只覆盖
      runtime.get_llm**（对话主循环），记忆抽取走 app.memory.writer 自己的 get_llm，
      不在这里的替换范围内——要桩掉它得另行 patch（测试里见 conftest 的
      `memory_writer.get_llm`），否则评测仍会打真实 API
    - sample_ids：只跑指定子集（小规模冒烟用）
    - concurrency：asyncio.gather 的并发上限，避免打满 rate limit
    - k：Hit@k / Recall@k 的 k

    注意 llm trace 的落库口径是进程级的（只认 settings.db_path，见 app/tracing.py），
    要收集 trace 就把 settings.db_path 指到同一个库。
    """
    dataset = load_dataset(dataset_path)
    samples = dataset["samples"]
    if sample_ids is not None:
        wanted = set(sample_ids)
        samples = [s for s in samples if s["id"] in wanted]
    # 不回落 settings.db_path：那是**在用的应用库**，删掉它等于删用户的知识库；
    # 评测库必须是独立的 scratch 文件（与 CLI 的默认值一致）
    path = db_path or DEFAULT_DB_PATH
    docs_dir = Path(dataset_path).parent / "docs"

    with _settings_override(config_overrides):
        await _reset_db(path)
        await init_db(path)
        for doc in dataset.get("docs", []):
            await ingest(docs_dir / doc, path)

        originals = _install_observers()
        original_get_llm = runtime.get_llm
        if llm is not None:
            runtime.get_llm = lambda: llm
        semaphore = asyncio.Semaphore(concurrency)

        async def guarded(sample: dict[str, Any]) -> SampleResult:
            """样本级故障一律转成 SampleResult.error（逐样本口径）。

            异常在这里被吃掉是**必须**的：让它冒到 gather 的话，gather 会在第一个异常
            处立刻向上抛，但其余任务不被取消、继续在后台跑，而下面的 finally 随即撤掉
            观测器与 LLM 桩、with 块退出还会把 settings 恢复默认——在飞的样本于是拿到
            「默认配置 + 真实客户端」，直接打真实 API 烧配额（见下方 gather 的说明）。

            观测收集器建在这里而不是 _run_sample 内部：崩溃时本函数还要拿它算检索
            指标（崩溃前的检索是真实证据，丢掉会造成生存者偏差）。
            """
            rounds: list[list[str]] = []
            recalled: list[str] = []
            async with semaphore:
                try:
                    return await _run_sample(sample, path, k, rounds, recalled)
                except Exception as exc:
                    logger.warning(
                        "样本 %s 失败：%s: %s",
                        sample.get("id"),
                        type(exc).__name__,
                        exc,
                    )
                    return _error_result(sample, exc, rounds, k)

        try:
            # return_exceptions=True 是**必需**的，不是顺手加的：默认行为在第一个异常时
            # 立刻向上抛，其余任务不被取消、继续在后台跑；下面的 finally 随即撤掉观测器与
            # LLM 桩、with 块退出还会把 settings 恢复默认——在飞的样本于是拿到「默认配置 +
            # 真实客户端」，直接打真实 API 烧配额。开了这个开关后 gather 等所有任务真正
            # 结束才返回，恢复动作因此一定发生在最后一个请求之后。
            outcomes = await asyncio.gather(
                *(guarded(s) for s in samples), return_exceptions=True
            )
        finally:
            _restore_observers(originals)
            runtime.get_llm = original_get_llm

        # 快照要在 override 生效期内拍：出了 with 块 settings 已恢复成默认值
        snapshot = config_snapshot()

    # 收尾把自己触发的后台写入等干净，再交给调用方。埋点与记忆抽取都是
    # fire-and-forget：不等的话这些任务会带着已打开的库连接活过 run_eval 返回，
    # 调用方随后删库/删目录（run_ablation 的 tempfile 清理、下一次 _reset_db）在
    # Windows 上直接 PermissionError。每轮运行收干净自己的副作用，边界才在 run_eval。
    await runtime.drain_memory_writes()
    await drain_traces()

    results: list[SampleResult] = []
    for sample, outcome in zip(samples, outcomes, strict=True):
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
            # 取消 / 键盘中断是「整轮别跑了」的信号，不该被降级成一个失败样本
            raise outcome
        results.append(
            outcome
            if isinstance(outcome, SampleResult)
            else _error_result(sample, outcome)
        )

    sample_dicts = [
        {"id": r.id, "category": r.category, "error": r.error, "metrics": r.metrics}
        for r in results
    ]
    return EvalResult(
        dataset=str(dataset_path),
        git_commit=git_commit(),
        config=snapshot,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        samples=results,
        metrics=aggregate_metrics(sample_dicts),
        dataset_version=dataset.get("version"),
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
                path = result.save(name=name)
                logger.info("组 %s 结果已保存：%s", name, path)
            report_path = save_report(render_ablation_report(results), name="ablation")
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
