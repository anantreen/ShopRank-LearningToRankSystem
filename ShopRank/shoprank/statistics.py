"""Query-level uncertainty and multiple-comparison utilities."""

import random


def paired_bootstrap(
    model_a: dict[str, float], model_b: dict[str, float], iterations: int = 10_000, seed: int = 42
) -> dict[str, float]:
    """Bootstrap paired per-query scores; never resample judgment rows."""
    ids = sorted(set(model_a) & set(model_b))
    if not ids:
        raise ValueError("models have no query IDs in common")
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sample = [rng.choice(ids) for _ in ids]
        deltas.append(sum(model_b[q] - model_a[q] for q in sample) / len(sample))
    deltas.sort()
    lower = deltas[int(0.025 * iterations)]
    upper = deltas[min(iterations - 1, int(0.975 * iterations))]
    observed = sum(model_b[q] - model_a[q] for q in ids) / len(ids)
    p_two_sided = 2 * min(sum(d <= 0 for d in deltas), sum(d >= 0 for d in deltas)) / iterations
    return {"delta": observed, "ci_low": lower, "ci_high": upper, "p_value": min(1.0, p_two_sided)}


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Return results in original order with monotone Holm-adjusted p-values."""
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_sorted = []
    running = 0.0
    m = len(p_values)
    for rank, (original, p_value) in enumerate(indexed):
        adjusted = max(running, min(1.0, (m - rank) * p_value))
        running = adjusted
        adjusted_sorted.append((original, p_value, adjusted))
    output = [None] * m
    for original, p_value, adjusted in adjusted_sorted:
        output[original] = {"p_value": p_value, "adjusted_p": adjusted, "reject": adjusted <= alpha}
    return output

