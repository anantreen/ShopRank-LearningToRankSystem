"""Pure-Python lexical baselines: BM25 and TF-IDF cosine."""

import math
from collections import Counter

from .text import tokenize


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(doc) for doc in documents]
        self.tf = [Counter(doc) for doc in self.docs]
        self.lengths = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df = Counter(term for doc in self.docs for term in set(doc))
        self.n = len(self.docs)

    def score(self, query: str, index: int) -> float:
        if not self.n or not self.avgdl:
            return 0.0
        score = 0.0
        for term in tokenize(query):
            freq = self.tf[index][term]
            if not freq:
                continue
            df = self.df[term]
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            norm = freq + self.k1 * (1.0 - self.b + self.b * self.lengths[index] / self.avgdl)
            score += idf * freq * (self.k1 + 1.0) / norm
        return score

    def scores(self, query: str) -> list[float]:
        return [self.score(query, i) for i in range(self.n)]


class TfidfCosine:
    def __init__(self, documents: list[str]):
        tokenized = [tokenize(doc) for doc in documents]
        self.n = len(tokenized)
        df = Counter(term for doc in tokenized for term in set(doc))
        self.idf = {term: math.log((1 + self.n) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(tokens) for tokens in tokenized]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vector = {term: count * self.idf.get(term, 0.0) for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def score(self, query: str, index: int) -> float:
        q = self._vector(tokenize(query))
        doc = self.vectors[index]
        return sum(weight * doc.get(term, 0.0) for term, weight in q.items())

    def scores(self, query: str) -> list[float]:
        return [self.score(query, i) for i in range(self.n)]

