"""T10 报告生成：把 EvalResult（单组或消融矩阵）渲染成 Markdown。

报告内容：概览（数据集 / git commit / 配置快照）、总体指标、分类指标、
消融对比表（多组时）、失败案例分析（关键词覆盖率未满或出错的样本）。
"""

from __future__ import annotations

import time
from pathlib import Path

from eval.ablation import GROUP_LABELS
from eval.runner import EvalResult

REPORTS_DIR = Path("eval/reports")
# 失败案例最多列多少条，避免报告被长尾刷屏
MAX_FAILURE_CASES = 10


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _metric_cells(overall: dict) -> str:
    return (
        f"{overall['n']} | {_fmt(overall['hit_at_k'])} | {_fmt(overall['mrr'])} "
        f"| {_fmt(overall['recall_at_k'])} | {_fmt(overall['keyword_coverage'])} "
        f"| {_fmt(overall['memory_recall'])} | {overall['errors']}"
    )


_METRIC_HEADER = (
    "| n | Hit@k | MRR | Recall@k | 关键词覆盖率 | 记忆召回 | 错误数 |\n"
    "|---:|------:|----:|---------:|-------------:|---------:|-------:|"
)


def render_report(result: EvalResult, title: str = "评测报告") -> str:
    """单组评测的 Markdown 报告。"""
    lines = [
        f"# {title}",
        "",
        "## 概览",
        "",
        f"- 数据集：`{result.dataset}`",
        f"- git commit：`{result.git_commit}`",
        f"- 时间：{result.created_at}",
        f"- 配置快照：`{result.config}`",
        "",
        "## 总体指标",
        "",
        _METRIC_HEADER,
        f"| {_metric_cells(result.metrics['overall'])} |",
        "",
        "## 分类指标",
        "",
        f"| 类别 {_METRIC_HEADER}",
    ]
    for category, metrics in result.metrics["by_category"].items():
        lines.append(f"| {category} | {_metric_cells(metrics)} |")
    lines += ["", "## 失败案例", ""]
    lines.extend(_failure_cases(result))
    return "\n".join(lines) + "\n"


def render_ablation_report(results: dict[str, EvalResult]) -> str:
    """消融矩阵对比报告：每组一行，外加各组的失败案例。"""
    first = next(iter(results.values()))
    lines = [
        "# 消融实验对比报告",
        "",
        "## 概览",
        "",
        f"- 数据集：`{first.dataset}`",
        f"- git commit：`{first.git_commit}`",
        f"- 时间：{first.created_at}",
        "",
        "## 消融对比",
        "",
        f"| 组 | 说明 | 配置 {_METRIC_HEADER}",
    ]
    for name, result in results.items():
        cfg = result.config
        config_text = (
            f"memory={cfg['memory_enabled']}, "
            f"compaction={cfg['context_compaction_enabled']}, "
            f"retrieval={cfg['retrieval_mode']}"
        )
        lines.append(
            f"| {name} | {GROUP_LABELS.get(name, '')} | {config_text} "
            f"| {_metric_cells(result.metrics['overall'])} |"
        )
    for name, result in results.items():
        lines += ["", f"## 组 {name} 失败案例", ""]
        lines.extend(_failure_cases(result))
    return "\n".join(lines) + "\n"


def _failure_cases(result: EvalResult) -> list[str]:
    """关键词覆盖率未满或出错的样本，最多 MAX_FAILURE_CASES 条。"""
    lines: list[str] = []
    failures = [
        s
        for s in result.samples
        if s.error or s.metrics["answer"]["keyword_coverage"] < 1.0
    ]
    if not failures:
        return ["无失败案例。"]
    for s in failures[:MAX_FAILURE_CASES]:
        missed = s.metrics["answer"]["missed"]
        lines.append(
            f"- `{s.id}`（{s.category}）{s.query}"
            + (f" —— 错误：{s.error}" if s.error else "")
            + (f" —— 未覆盖关键词：{missed}" if missed else "")
        )
    if len(failures) > MAX_FAILURE_CASES:
        lines.append(f"- ……另有 {len(failures) - MAX_FAILURE_CASES} 条，详见 result.json")
    return lines


def save_report(markdown: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """报告保存到 eval/reports/{timestamp}.md。"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
