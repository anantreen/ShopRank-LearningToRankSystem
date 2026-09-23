#!/usr/bin/env python3
"""EDA that works on the built-in data; replace the loader for ESCI."""

from collections import Counter
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shoprank.demo_data import PRODUCTS, judgments
from shoprank.text import tokenize


def main():
    rows = judgments()
    queries = {row["query_id"]: row["query"] for row in rows}
    query_lengths = [len(tokenize(query)) for query in queries.values()]
    terms = Counter(term for query in queries.values() for term in tokenize(query))
    labels = Counter(row["label"] for row in rows)
    descriptions = [len(tokenize(p["description"])) for p in PRODUCTS]
    brands = {p["brand"].lower() for p in PRODUCTS}
    brand_queries = sum(any(brand in q.lower() for brand in brands) for q in queries.values())
    print(f"Unique queries: {len(queries)}")
    print(f"Query tokens: mean={mean(query_lengths):.2f}, median={median(query_lengths):.1f}")
    print(f"Brand-query fraction: {brand_queries / len(queries):.1%}")
    print(f"Labels: {dict(labels)}")
    print(f"Description tokens: mean={mean(descriptions):.2f}; missing=0.0%")
    print(f"Top terms: {terms.most_common(10)}")


if __name__ == "__main__":
    main()
