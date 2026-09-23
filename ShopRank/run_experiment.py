#!/usr/bin/env python3
"""Run the fast, dependency-free demo experiment."""

from pathlib import Path

from shoprank.features import FEATURE_NAMES
from shoprank.pipeline import run_demo
from shoprank.statistics import paired_bootstrap


def main():
    result = run_demo()
    means = {name: sum(scores.values()) / len(scores) for name, scores in result["models"].items()}
    comparison = paired_bootstrap(result["models"]["BM25"], result["models"]["Pairwise LTR"], iterations=2_000)
    lines = [
        "# Demo results (synthetic mini catalog)", "",
        "These numbers verify the code path. They are **not ESCI benchmark results**.", "",
        "| Model | NDCG@10 |", "|---|---:|",
    ]
    for name, score in means.items():
        lines.append(f"| {name} | {score:.4f} |")
    lines += [
        "", "## Pre-specified comparison: Pairwise LTR vs BM25", "",
        f"Mean delta: {comparison['delta']:+.4f}",
        f"95% paired query-bootstrap CI: [{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]",
        "", "## Learned linear feature weights", "",
    ]
    for name, weight in zip(FEATURE_NAMES, result["ranker"].weights):
        lines.append(f"- `{name}`: {weight:+.4f}")
    output = Path("02_bm25_baseline/results.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()

