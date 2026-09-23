#!/usr/bin/env python3

from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shoprank.metrics import ndcg_at_k


relevances = [3, 0, 2, 1]
base = ndcg_at_k(relevances, 10)

if __name__ == "__main__":
    print(f"base NDCG={base:.4f}")
    for i, j in combinations(range(len(relevances)), 2):
        swapped = relevances.copy()
        swapped[i], swapped[j] = swapped[j], swapped[i]
        print(f"swap ranks {i+1}/{j+1}: |delta NDCG|={abs(ndcg_at_k(swapped, 10)-base):.4f}")
