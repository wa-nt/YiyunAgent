"""T10 评测指标：检索 / 回答 / 记忆三个维度的纯函数计算，外加逐样本聚合。

所有函数都是纯函数，不碰数据库、不碰网络，方便单测与复用：

- 检索指标：Hit Rate@k、MRR、Recall@k。标识符是字符串（评测集里用文档标题，
  chunk 自增 id 在重新 ingest 后不稳定，原因见 eval/dataset/README.md）。多轮检索的
  样本按「每篇期望文档在任一轮里的最好排名」评判（retrieval_metrics_rounds）。
- 回答指标：关键词覆盖率（expected_answer_contains 的子串命中比例），外加
  expected_answer_excludes 的违规判定（对抗样本：出现禁词直接 0 分）。
- 记忆指标：记忆召回准确率（期望记忆条目在召回文本中的命中比例）。
- LLM-as-judge 可选：无 API key 时跳过，不阻塞自动指标。
"""

from __future__ import annotations

from typing import Any


def retrieval_metrics_rounds(
    rounds: list[list[str]], expected: list[str], k: int = 8
) -> dict[str, Any]:
    """多轮检索的检索质量：每篇期望文档取「在任一轮里的最好排名」，再按 k 评判。

    expected 中任一标识符在任一轮的前 k 条里出现即命中；mrr 取最早（最好）排名，
    recall_at_k 是「至少在一轮的前 k 条里出现过」的期望文档比例。expected 为空时
    三个值都给 0，由调用方决定跳过（runner 对无检索标注的样本直接记 None）。

    一个样本的 ReAct 循环可能检索多次（多跳问题分轮查）。把各轮结果拼成一个列表再按
    k 截断是错的：第 2 轮的命中会排到第 1 轮的 k 条之后，被系统性判成未命中。
    """
    expected_set = set(expected)
    if not expected_set:
        return {"hit_at_k": 0.0, "mrr": 0.0, "recall_at_k": 0.0}
    best: dict[str, int] = {}
    for round_items in rounds:
        for rank, item in enumerate(list(round_items)[:k], start=1):
            if item in expected_set:
                best[item] = min(best.get(item, rank), rank)
    if not best:
        return {"hit_at_k": 0.0, "mrr": 0.0, "recall_at_k": 0.0}
    return {
        "hit_at_k": 1.0,
        "mrr": 1.0 / min(best.values()),
        "recall_at_k": len(best) / len(expected_set),
    }


def retrieval_metrics(
    retrieved: list[str], expected: list[str], k: int = 8
) -> dict[str, Any]:
    """单轮检索的检索质量（等价于只有一轮的 retrieval_metrics_rounds）。"""
    return retrieval_metrics_rounds([retrieved], expected, k)


def answer_metrics(
    answer: str,
    expected_contains: list[str],
    expected_excludes: list[str] | None = None,
) -> dict[str, Any]:
    """回答质量：expected_contains 关键词覆盖率 + expected_excludes 违规判定。

    expected_contains 为空视为「无标注约束」，覆盖率给 1.0（不惩罚）。
    expected_excludes 是**不该出现**的关键词（对抗样本的判据），命中即覆盖率 0：
    只查「回答里有没有『没有』」的话，胡编一段再补一句「没有」也拿满分。
    """
    violated = [kw for kw in (expected_excludes or []) if kw in answer]
    if not expected_contains:
        matched: list[str] = []
        missed: list[str] = []
        coverage = 1.0
    else:
        matched = [kw for kw in expected_contains if kw in answer]
        missed = [kw for kw in expected_contains if kw not in answer]
        coverage = len(matched) / len(expected_contains)
    return {
        "keyword_coverage": 0.0 if violated else coverage,
        "matched": matched,
        "missed": missed,
        "violated": violated,
    }


def memory_metrics(
    recalled_text: str | None, expected_memories: list[str]
) -> dict[str, Any]:
    """记忆召回准确率：期望记忆条目在召回文本里的命中比例。

    recalled_text 是本轮注入 prompt 的记忆文本（runner 包装 recall_memories 录得）。
    expected_memories 为空表示该样本不考核记忆，memory_recall 给 None，
    聚合时跳过（不能把「不考核」算成 0 分拉低记忆组）。
    """
    if not expected_memories:
        return {"memory_recall": None, "matched": [], "missed": []}
    text = recalled_text or ""
    matched = [m for m in expected_memories if m in text]
    missed = [m for m in expected_memories if m not in text]
    return {
        "memory_recall": len(matched) / len(expected_memories),
        "matched": matched,
        "missed": missed,
    }


def aggregate_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐样本的指标字典聚合成总体 + 分类两组均值。

    samples 里每项是 {category, metrics: {...}}。retrieval 类指标只在
    metrics["retrieval"] 非 None 的样本上取均值；memory_recall 同理跳过 None。
    """
    by_category: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by_category.setdefault(s["category"], []).append(s)
    return {
        "overall": _aggregate_group(samples),
        "by_category": {
            category: _aggregate_group(group)
            for category, group in sorted(by_category.items())
        },
        "total": len(samples),
    }


def _aggregate_group(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    retrieval = [s["metrics"]["retrieval"] for s in samples if s["metrics"].get("retrieval")]
    memory = [
        s["metrics"]["memory"]["memory_recall"]
        for s in samples
        if s["metrics"].get("memory", {}).get("memory_recall") is not None
    ]
    return {
        "n": len(samples),
        "hit_at_k": _mean([r["hit_at_k"] for r in retrieval]),
        "mrr": _mean([r["mrr"] for r in retrieval]),
        "recall_at_k": _mean([r["recall_at_k"] for r in retrieval]),
        "keyword_coverage": _mean(
            [s["metrics"]["answer"]["keyword_coverage"] for s in samples]
        ),
        "memory_recall": _mean(memory),
        "errors": sum(1 for s in samples if s.get("error")),
    }
