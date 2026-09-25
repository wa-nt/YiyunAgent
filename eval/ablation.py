"""T10 消融矩阵：8 组配置跑同一评测集，验证记忆 / 压缩 / 检索各自的贡献。

| 组 | 记忆 | 压缩 | 检索   | 说明               |
|----|------|------|--------|--------------------|
| A  | ✓    | ✓    | hybrid | 完整系统（基线）   |
| B  | ✗    | ✓    | hybrid | 无记忆             |
| C  | ✓    | ✗    | hybrid | 无压缩             |
| D  | ✓    | ✓    | vector | 纯向量检索         |
| E  | ✓    | ✓    | bm25   | 纯 BM25 检索       |
| F  | ✗    | ✗    | vector | 最小系统           |
| G  | ✓    | ✗    | vector | 只有记忆           |
| H  | ✗    | ✓    | bm25   | 只有压缩           |

「压缩」对应 context_compaction_enabled（历史压缩）；tool_clean / token_budget 是
独立的治理策略，不在这张表里——消融是单因素对照，一次只动一个旋钮。

每组用独立的临时库，避免记忆与历史跨组污染。统计显著性检验需要逐样本配对比对，
样本量 30+ 时用 scipy.stats 的符号检验才有意义，无 scipy 依赖时跳过（可选）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from eval.runner import EvalResult, run_eval

ABLATION_GROUPS: dict[str, dict[str, Any]] = {
    "A": {"memory_enabled": True, "context_compaction_enabled": True, "retrieval_mode": "hybrid"},
    "B": {"memory_enabled": False, "context_compaction_enabled": True, "retrieval_mode": "hybrid"},
    "C": {"memory_enabled": True, "context_compaction_enabled": False, "retrieval_mode": "hybrid"},
    "D": {"memory_enabled": True, "context_compaction_enabled": True, "retrieval_mode": "vector"},
    "E": {"memory_enabled": True, "context_compaction_enabled": True, "retrieval_mode": "bm25"},
    "F": {"memory_enabled": False, "context_compaction_enabled": False, "retrieval_mode": "vector"},
    "G": {"memory_enabled": True, "context_compaction_enabled": False, "retrieval_mode": "vector"},
    "H": {"memory_enabled": False, "context_compaction_enabled": True, "retrieval_mode": "bm25"},
}

GROUP_LABELS = {
    "A": "完整系统（基线）",
    "B": "无记忆",
    "C": "无压缩",
    "D": "纯向量检索",
    "E": "纯 BM25 检索",
    "F": "最小系统",
    "G": "只有记忆",
    "H": "只有压缩",
}


async def run_ablation(
    dataset_path: str,
    groups: list[str] | None = None,
    work_dir: str | Path | None = None,
    llm: Any | None = None,
    sample_ids: list[str] | None = None,
    concurrency: int = 4,
) -> dict[str, EvalResult]:
    """逐组跑评测，返回 {组名: EvalResult}。每组独立建库，互不污染。

    work_dir 指定时把各组库放在该目录下（便于复査）；默认用临时目录，
    跑完即弃——每组的持久产物是 EvalResult.save() 落盘的 result.json。

    groups 缺省（None）跑全部 8 组；**显式传空列表则一组都不跑**并返回 {}——
    用 `groups or list(ABLATION_GROUPS)` 的话空列表会被当成 None，静默跑满 8 组，
    调用方想用空列表表达「什么都不做」时会得到一堆意外产物。
    对 sample_ids 筛完为空的情况，每组仍产出一条 n=0 的结果（组本身跑了，只是无样本）。

    库的清理由 run_eval 自己在开头做（删库文件后重新 ingest），所以 work_dir 模式
    下重跑同一组不会把文档再灌一遍——这也意味着 work_dir 是**评测专用**目录，
    别把要留的东西放进去。
    """
    names = list(ABLATION_GROUPS) if groups is None else list(groups)
    unknown = [n for n in names if n not in ABLATION_GROUPS]
    if unknown:
        raise ValueError(f"未知消融组：{unknown}，可选：{list(ABLATION_GROUPS)}")

    results: dict[str, EvalResult] = {}
    with tempfile.TemporaryDirectory(prefix="eval-ablation-") as tmp:
        base = Path(work_dir) if work_dir else Path(tmp)
        base.mkdir(parents=True, exist_ok=True)
        for name in names:
            results[name] = await run_eval(
                dataset_path,
                config_overrides=ABLATION_GROUPS[name],
                db_path=str(base / f"group-{name}.db"),
                llm=llm,
                sample_ids=sample_ids,
                concurrency=concurrency,
            )
    return results
