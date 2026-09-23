"""Ranking metrics. Inputs must already be in predicted rank order."""

import math
from collections.abc import Iterable, Mapping


def dcg_at_k(relevances: Iterable[float], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(
        (2.0**rel - 1.0) / math.log2(rank + 2.0)
        for rank, rel in enumerate(list(relevances)[:k])
    )


def ndcg_at_k(relevances: Iterable[float], k: int = 10) -> float:
    values = list(relevances)
    ideal = dcg_at_k(sorted(values, reverse=True), k)
    return dcg_at_k(values, k) / ideal if ideal else 0.0


def mean_ndcg_at_k(by_query: Mapping[str, Iterable[float]], k: int = 10) -> float:
    """Average query NDCGs, giving each query equal weight."""
    scores = [ndcg_at_k(values, k) for values in by_query.values()]
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(relevances: Iterable[float], k: int = 10, threshold: float = 1) -> float:
    for rank, rel in enumerate(list(relevances)[:k], start=1):
        if rel >= threshold:
            return 1.0 / rank
    return 0.0


def precision_at_k(relevances: Iterable[float], k: int = 10, threshold: float = 1) -> float:
    values = list(relevances)[:k]
    return sum(rel >= threshold for rel in values) / k if k > 0 else 0.0


def recall_at_k(relevances: Iterable[float], k: int = 10, threshold: float = 1) -> float:
    values = list(relevances)
    total = sum(rel >= threshold for rel in values)
    return sum(rel >= threshold for rel in values[:k]) / total if total else 0.0

