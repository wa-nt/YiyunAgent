"""T10 评测指标：检索 / 回答 / 记忆三个维度的纯函数计算，外加逐样本聚合。

所有函数都是纯函数，不碰数据库、不碰网络，方便单测与复用：

- 检索指标：Hit Rate@k、MRR、Recall@k。标识符是字符串（评测集里用文档标题，
  见 eval/dataset/README 的说明：chunk 自增 id 在重新 ingest 后不稳定）。
- 回答指标：关键词覆盖率（expected_answer_contains 的子串命中比例）。
- 记忆指标：记忆召回准确率（期望记忆条目在召回文本中的命中比例）。
- LLM-as-judge 可选：无 API key 时跳过，不阻塞自动指标。
"""

from __future__ import annotations

from typing import Any


def retrieval_metrics(
    retrieved: list[str], expected: list[str], k: int = 8
) -> dict[str, Any]:
    """检索质量：expected 中任一标识符出现在 retrieved 前 k 条即命中。

    返回 hit_at_k（0/1）、mrr（首个命中的倒数排名）、recall_at_k（前 k 条
    盖住 expected 的比例）。expected 为空时三个值都给 0，由调用方决定跳过。
    """
    topk = list(retrieved)[:k]
    expected_set = set(expected)
    first_rank: int | None = None
    for rank, item in enumerate(topk, start=1):
        if item in expected_set:
            first_rank = rank
            break
    hit = first_rank is not None
    return {
        "hit_at_k": 1.0 if hit else 0.0,
        "mrr": 1.0 / first_rank if hit else 0.0,
        "recall_at_k": (
            len(expected_set & set(topk)) / len(expected_set) if expected_set else 0.0
        ),
    }


def answer_metrics(answer: str, expected_contains: list[str]) -> dict[str, Any]:
    """回答质量：expected_contains 关键词在回答中的覆盖率（子串匹配）。

    expected_contains 为空视为「无标注约束」，覆盖率给 1.0（不惩罚）。
    """
    if not expected_contains:
        return {"keyword_coverage": 1.0, "matched": [], "missed": []}
    matched = [kw for kw in expected_contains if kw in answer]
    missed = [kw for kw in expected_contains if kw not in answer]
    return {
        "keyword_coverage": len(matched) / len(expected_contains),
        "matched": matched,
        "missed": missed,
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
