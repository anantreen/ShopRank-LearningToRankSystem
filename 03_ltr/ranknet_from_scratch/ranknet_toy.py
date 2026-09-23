#!/usr/bin/env python3
"""RankNet pair probabilities and cross-entropy for one toy query."""

import math


labels = [3, 0, 2, 1]
scores = [1.2, -0.3, 0.7, 0.1]


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


if __name__ == "__main__":
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] <= labels[j]:
                continue
            probability = sigmoid(scores[i] - scores[j])
            loss = -math.log(max(probability, 1e-12))
            print(f"product {i} > {j}: P={probability:.4f}, loss={loss:.4f}")

