# Demo results (synthetic mini catalog)

These numbers verify the code path. They are **not ESCI benchmark results**.

| Model | NDCG@10 |
|---|---:|
| TF-IDF | 0.9864 |
| BM25 | 0.9864 |
| Pairwise LTR | 0.9864 |

## Pre-specified comparison: Pairwise LTR vs BM25

Mean delta: +0.0000
95% paired query-bootstrap CI: [+0.0000, +0.0000]

## Learned linear feature weights

- `bm25`: +1.6093
- `tfidf`: +2.0062
- `title_coverage`: +1.2487
- `all_text_jaccard`: -0.3747
- `brand_match`: -3.4764
- `exact_phrase`: +0.8094
- `query_length`: +0.0000
- `title_length`: +0.5995
- `description_length`: -0.6934
