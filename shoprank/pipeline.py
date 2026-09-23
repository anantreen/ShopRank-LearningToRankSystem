"""End-to-end experiment helpers used by the CLI and API."""

from collections import defaultdict
from functools import lru_cache

from . import LABEL_TO_RELEVANCE
from .demo_data import PRODUCTS, judgments
from .features import make_features
from .metrics import ndcg_at_k
from .ranknet import PairwiseLinearRanker
from .retrieval import BM25, TfidfCosine
from .split import split_query_ids


def build_demo_context():
    texts = [f"{p['title']} {p['description']}" for p in PRODUCTS]
    return BM25(texts), TfidfCosine(texts), {p["product_id"]: i for i, p in enumerate(PRODUCTS)}


def scored_rows(rows, bm25, tfidf, product_index):
    output = []
    for row in rows:
        index = product_index[row["product_id"]]
        product = PRODUCTS[index]
        b_score = bm25.score(row["query"], index)
        t_score = tfidf.score(row["query"], index)
        output.append({
            **row,
            "relevance": LABEL_TO_RELEVANCE[row["label"]],
            "bm25": b_score,
            "tfidf": t_score,
            "features": make_features(row["query"], product, b_score, t_score),
        })
    return output


def train_demo_ranker(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append((row["features"], row["relevance"]))
    return PairwiseLinearRanker().fit(dict(groups))


def evaluate(rows, score_name, scorer=None, k=10):
    groups = defaultdict(list)
    for row in rows:
        score = scorer(row["features"]) if scorer else row[score_name]
        groups[row["query_id"]].append((score, row["relevance"]))
    per_query = {}
    for query_id, values in groups.items():
        ranked = [rel for _, rel in sorted(values, reverse=True)]
        per_query[query_id] = ndcg_at_k(ranked, k)
    return per_query


def run_demo(seed=42):
    rows = judgments()
    train_ids, val_ids, test_ids = split_query_ids(
        (row["query_id"] for row in rows), seed=seed
    )
    bm25, tfidf, product_index = build_demo_context()
    enriched = scored_rows(rows, bm25, tfidf, product_index)
    train_rows = [row for row in enriched if row["query_id"] in train_ids]
    test_rows = [row for row in enriched if row["query_id"] in test_ids]
    ranker = train_demo_ranker(train_rows)
    models = {
        "TF-IDF": evaluate(test_rows, "tfidf"),
        "BM25": evaluate(test_rows, "bm25"),
        "Pairwise LTR": evaluate(test_rows, "", ranker.predict_one),
    }
    return {"train_queries": train_ids, "val_queries": val_ids, "test_queries": test_ids, "models": models, "ranker": ranker}


@lru_cache(maxsize=1)
def build_demo_search_engine():
    """Fit the teaching ranker and return all pieces needed for inference."""
    bm25, tfidf, product_index = build_demo_context()
    rows = scored_rows(judgments(), bm25, tfidf, product_index)
    ranker = train_demo_ranker(rows)
    return bm25, tfidf, ranker


def search_demo(query: str, limit: int = 10) -> list[dict]:
    """BM25 candidates followed by learned pairwise reranking."""
    bm25, tfidf, ranker = build_demo_search_engine()
    lexical_scores = bm25.scores(query)
    candidate_ids = sorted(
        range(len(PRODUCTS)), key=lexical_scores.__getitem__, reverse=True
    )[: min(15, len(PRODUCTS))]
    ranked = []
    for index in candidate_ids:
        product = PRODUCTS[index]
        features = make_features(query, product, lexical_scores[index], tfidf.score(query, index))
        ranked.append((ranker.predict_one(features), product))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {"product_id": product["product_id"], "title": product["title"], "relevance_score": score}
        for score, product in ranked[:limit]
    ]
