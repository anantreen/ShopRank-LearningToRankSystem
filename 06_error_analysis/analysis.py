#!/usr/bin/env python3
"""Slice metrics by query type to find model-specific failure modes."""

import re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shoprank.demo_data import QUERY_TEMPLATES
from shoprank.pipeline import run_demo


BRANDS = {"apple", "samsung", "sony", "jbl", "nike", "adidas", "logitech", "razer", "puma"}


def query_segment(query):
    tokens = set(query.lower().split())
    if re.search(r"(?=.*[a-zA-Z])(?=.*\d)[a-zA-Z0-9-]{4,}", query):
        return "model_number"
    if tokens & BRANDS:
        return "brand"
    if any(word in tokens for word in {"samsng"}):
        return "misspelling"
    if len(tokens) >= 4:
        return "natural_language"
    return "other"


if __name__ == "__main__":
    result = run_demo()
    query_text = {f"Q{i:03d}": row[0] for i, row in enumerate(QUERY_TEMPLATES, 1)}
    buckets = defaultdict(lambda: defaultdict(list))
    for model, scores in result["models"].items():
        for query_id, score in scores.items():
            buckets[query_segment(query_text[query_id])][model].append(score)
    for segment, models in sorted(buckets.items()):
        print(segment)
        for model, scores in models.items():
            print(f"  {model:14s} NDCG@10={sum(scores)/len(scores):.4f} n={len(scores)}")
