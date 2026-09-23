"""A tiny pairwise linear ranker, written from scratch for learning."""

import math
import random


class PairwiseLinearRanker:
    def __init__(self, learning_rate: float = 0.03, epochs: int = 150, seed: int = 42):
        self.learning_rate, self.epochs, self.seed = learning_rate, epochs, seed
        self.weights: list[float] = []

    def fit(self, groups: dict[str, list[tuple[list[float], int]]]) -> "PairwiseLinearRanker":
        first_group = next(iter(groups.values()))
        width = len(first_group[0][0])
        self.weights = [0.0] * width
        pairs = []
        for examples in groups.values():
            for a in examples:
                for b in examples:
                    if a[1] > b[1]:
                        pairs.append((a[0], b[0]))
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for better, worse in pairs:
                diff = [a - b for a, b in zip(better, worse)]
                margin = sum(w * x for w, x in zip(self.weights, diff))
                gradient_scale = 1.0 / (1.0 + math.exp(min(30.0, margin)))
                for i, value in enumerate(diff):
                    self.weights[i] += self.learning_rate * gradient_scale * value
        return self

    def predict_one(self, features: list[float]) -> float:
        return sum(w * value for w, value in zip(self.weights, features))
