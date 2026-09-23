"""Features for each (query, product) training example."""

from .text import tokenize


FEATURE_NAMES = [
    "bm25", "tfidf", "title_coverage", "all_text_jaccard", "brand_match",
    "exact_phrase", "query_length", "title_length", "description_length",
]


def make_features(query: str, product: dict, bm25_score: float, tfidf_score: float) -> list[float]:
    q = set(tokenize(query))
    title_tokens = tokenize(product.get("title", ""))
    description_tokens = tokenize(product.get("description", ""))
    title = set(title_tokens)
    all_text = title | set(description_tokens)
    brand = product.get("brand", "").lower()
    return [
        bm25_score,
        tfidf_score,
        len(q & title) / len(q) if q else 0.0,
        len(q & all_text) / len(q | all_text) if q | all_text else 0.0,
        float(bool(brand) and brand in query.lower()),
        float(query.lower() in product.get("title", "").lower()),
        float(len(q)),
        float(len(title_tokens)),
        float(len(description_tokens)),
    ]

