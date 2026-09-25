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

# 三张表的表头与分隔行各自成套：列数必须一一对应。分类表在指标前多一列「类别」，
# 消融表多「组 / 说明 / 配置」三列——直接拿总表的表头去拼会让表头比分隔行多几列，
# Markdown 渲染时多出来的列名被截掉（列数由分隔行决定）
_METRIC_COLUMNS = ("n", "Hit@k", "MRR", "Recall@k", "关键词覆盖率", "记忆召回", "错误数")
# 分隔行的横线宽度只影响源码里的视觉对齐（渲染列宽由最宽的单元格决定），
# 未列出的列名走默认值
_ALIGN = {
    "n": "---:",
    "Hit@k": "------:",
    "MRR": "----:",
    "Recall@k": "---------:",
    "关键词覆盖率": "-------------:",
    "记忆召回": "---------:",
    "错误数": "-------:",
}


def _header_row(labels: tuple[str, ...]) -> str:
    return "| " + " | ".join(labels) + " |"


def _separator_row(labels: tuple[str, ...]) -> str:
    # 全部右对齐：表里除首列的标签外都是数值
    return "|" + "|".join(_ALIGN.get(label, "------:") for label in labels) + "|"


def _table(labels: tuple[str, ...]) -> str:
    return "\n".join((_header_row(labels), _separator_row(labels)))


_METRIC_HEADER = _table(_METRIC_COLUMNS)
_CATEGORY_HEADER = _table(("类别", *_METRIC_COLUMNS))
_ABLATION_METRIC_COLUMNS = ("组", "说明", "配置", *_METRIC_COLUMNS)
_ABLATION_HEADER = _table(_ABLATION_METRIC_COLUMNS)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _metric_cells(overall: dict) -> str:
    return (
        f"{overall['n']} | {_fmt(overall['hit_at_k'])} | {_fmt(overall['mrr'])} "
        f"| {_fmt(overall['recall_at_k'])} | {_fmt(overall['keyword_coverage'])} "
        f"| {_fmt(overall['memory_recall'])} | {overall['errors']}"
    )


def render_report(result: EvalResult, title: str = "评测报告") -> str:
    """单组评测的 Markdown 报告。"""
    lines = [
        f"# {title}",
        "",
        "## 概览",
        "",
        f"- 数据集：`{result.dataset}`",
        f"- 数据集版本：{result.dataset_version}",
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
        _CATEGORY_HEADER,
    ]
    for category, metrics in result.metrics["by_category"].items():
        lines.append(f"| {category} | {_metric_cells(metrics)} |")
    lines += ["", "## 失败案例", ""]
    lines.extend(_failure_cases(result))
    return "\n".join(lines) + "\n"


def render_ablation_report(results: dict[str, EvalResult]) -> str:
    """消融矩阵对比报告：每组一行，外加各组的失败案例。

    results 为空时返回一段提示文本而不是抛异常：CLI 的 `--ablation --samples 不存在的id`
    等路径会走到这里，报错比「空报告」更没用。
    """
    if not results:
        return "# 消融实验对比报告\n\n没有可渲染的消融结果（results 为空）。\n"
    first = next(iter(results.values()))
    lines = [
        "# 消融实验对比报告",
        "",
        "## 概览",
        "",
        f"- 数据集：`{first.dataset}`",
        f"- 数据集版本：{first.dataset_version}",
        f"- git commit：`{first.git_commit}`",
        f"- 时间：{first.created_at}",
        "",
        "## 消融对比",
        "",
        _ABLATION_HEADER,
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
    """关键词覆盖率未满、出现禁词、或出错的样本，最多 MAX_FAILURE_CASES 条。"""
    lines: list[str] = []
    failures = [
        s
        for s in result.samples
        if s.error or s.metrics["answer"]["keyword_coverage"] < 1.0
    ]
    if not failures:
        return ["无失败案例。"]
    for s in failures[:MAX_FAILURE_CASES]:
        answer = s.metrics["answer"]
        missed = answer["missed"]
        violated = answer.get("violated") or []
        lines.append(
            f"- `{s.id}`（{s.category}）{s.query}"
            + (f" —— 错误：{s.error}" if s.error else "")
            + (f" —— 未覆盖关键词：{missed}" if missed else "")
            + (f" —— 出现禁词：{violated}" if violated else "")
        )
    if len(failures) > MAX_FAILURE_CASES:
        lines.append(f"- ……另有 {len(failures) - MAX_FAILURE_CASES} 条，详见 result.json")
    return lines


def save_report(
    markdown: str, reports_dir: Path = REPORTS_DIR, name: str | None = None
) -> Path:
    """报告保存到 eval/reports/{timestamp}[-{name}].md。

    name 用于区分同一秒内的多份报告（CLI 一次跑消融会写结果与报告若干份），
    同秒时后写的会把先写的覆盖掉。
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = reports_dir / (f"{stamp}-{name}.md" if name else f"{stamp}.md")
    path.write_text(markdown, encoding="utf-8")
    return path
