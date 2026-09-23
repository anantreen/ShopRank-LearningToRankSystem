# ShopRank

A beginner-friendly, production-style e-commerce search and learning-to-rank project. It starts
with transparent algorithms you can read, then adds optional LightGBM, sentence transformers,
FAISS and FastAPI stages.

The included mini catalog is synthetic and exists only to prove every core component works.
Do not present its scores as Amazon ESCI benchmark results.

## What the system does

```text
customer query
    -> BM25 / TF-IDF candidate scores
    -> query-product features
    -> pairwise LTR or optional LambdaMART
    -> top products
    -> optional business constraints
```

Semantic retrieval is a second candidate source, not a magical replacement for lexical search:

```text
product text -> MiniLM embeddings -> FAISS index (offline)
query -> MiniLM embedding -> top candidates (online)
```

## First run: about five minutes

From this directory:

```bash
python3 -m unittest discover -s tests -v
python3 01_data_and_eda/eda.py
python3 run_experiment.py
python3 06_error_analysis/analysis.py
python3 07_multiobjective_demo/rerank.py
```

The first command tests correctness. The experiment writes real output from the demo data to
`02_bm25_baseline/results.md`.

For the API, create an isolated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[core,api]'
uvicorn api.main:app --reload
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"wireless gaming mouse","limit":3}'
```

## Learn it in this order

### 1. Understand one training row

One row says: for query `iphone 15 type c charger`, product `P01` has label `E`. Labels become
numbers: `I=0`, `C=1`, `S=2`, `E=3`. This is **graded relevance**, not merely relevant/irrelevant.
Read `shoprank/demo_data.py` and run the EDA script.

### 2. Understand the split

`shoprank/split.py` splits unique query IDs, then assigns all their product rows together. A row
split leaks the same query into train and test. That makes evaluation over-optimistic because rows
inside one result list are dependent.

### 3. Understand NDCG

Accuracy cannot distinguish a great ordering from a terrible ordering containing the same labels.
DCG gives high relevance more gain (`2^rel - 1`) and discounts lower ranks logarithmically. NDCG
divides by the best possible DCG, producing a score from 0 to 1. Read `shoprank/metrics.py`, then
`tests/test_metrics.py`.

### 4. Understand lexical retrieval

TF-IDF gives rare words high weight and compares normalized vectors. BM25 additionally saturates
term frequency and corrects for document length. `k1` controls saturation; `b=0` ignores length,
while `b=1` applies full length normalization. Read `shoprank/retrieval.py`.

### 5. Understand features and learning to rank

`shoprank/features.py` turns a query-product pair into numbers: lexical scores, token coverage,
Jaccard overlap, brand match, exact phrase, and lengths. `shoprank/ranknet.py` learns weights from
pairs such as “Exact product should score above Irrelevant product.” Run:

```bash
python3 03_ltr/ranknet_from_scratch/ranknet_toy.py
python3 03_ltr/lambdarank_math/lambdarank_demo.py
```

RankNet optimizes pair ordering. LambdaRank reweights each pair by how much swapping it changes
NDCG. LambdaMART fits boosted trees to those LambdaRank-style gradients. After installing
LightGBM, use `03_ltr/lambdamart/lambdamart.py` with feature rows kept contiguous by query.

### 6. Treat evaluation statistically

Model scores vary by query. `shoprank/statistics.py` resamples whole queries with replacement and
builds a confidence interval for paired NDCG differences. It also supplies Holm correction for
exploratory comparisons. A confidence interval crossing zero does not establish a reliable win.

### 7. Add semantic retrieval last

Install the larger stack only after the baseline is solid:

```bash
python -m pip install -e '.[semantic]'
```

`04_semantic_ranking/biencoder/embed.py` encodes products once and searches a FAISS index. A
bi-encoder is fast because query and product are encoded separately. A cross-encoder sees them
together and is more accurate but too slow for millions of products, so it reranks only a small
candidate set. The hard-negative experiment plan is in that folder.

### 8. Do error analysis

Average NDCG hides failures. The error script compares brand, model-number, misspelled and natural
language queries. On ESCI, quantify the largest gap and inspect real examples before inventing a
fix. A hybrid or query router often preserves BM25 strength on exact identifiers while gaining
semantic recall on descriptive queries.

### 9. Separate relevance from business ranking

The multi-objective demo uses clearly synthetic stock, conversion and seller signals. Stock is a
hard filter; conversion and quality are soft objectives. Never let a high-conversion but irrelevant
item win without a relevance floor.

### 10. Move from demo data to ESCI

Follow `data/README.md`. Then adapt the loader—not the metric or split logic—to produce the same
row fields. Work on a few hundred complete English query groups first. Save experiment config,
seed, query IDs, scores and latency. Only numbers produced on held-out ESCI queries belong in a
portfolio comparison table.

## Repository map

| Path | Purpose |
|---|---|
| `01_data_and_eda/` | inspect data and prevent query leakage |
| `02_bm25_baseline/` | lexical baselines and generated results |
| `03_ltr/` | RankNet, LambdaRank explanation, optional LambdaMART |
| `04_semantic_ranking/` | optional bi-encoder and cross-encoder experiment plan |
| `05_evaluation/` | NDCG, paired bootstrap, Holm correction |
| `06_error_analysis/` | query-segment diagnosis |
| `07_multiobjective_demo/` | stock and business-signal reranking |
| `api/` | FastAPI endpoint and benchmark |
| `experiments/` | honest, explicitly unexecuted A/B design |
| `shoprank/` | reusable code imported by scripts and tests |

## Good interview answers

- **Why NDCG, not accuracy?** Search quality depends on order and relevance is graded.
- **Why split by query?** Products judged for one query are dependent; row splitting leaks context.
- **Why BM25 over TF-IDF?** Term-frequency saturation and document-length normalization.
- **What is LambdaRank?** RankNet's pairwise gradient weighted by absolute change in NDCG.
- **Why bootstrap queries?** The query is the independent sampling unit, not a query-product row.
- **Bi-encoder vs cross-encoder?** Scalable cached embeddings versus slower joint token interaction.
- **Why hard negatives?** They remove trivial shortcuts and teach fine relevance distinctions.
- **Why not optimize conversion alone?** It can promote irrelevant, unavailable or low-trust items.

## Scope and honesty

The lexical vertical slice, tests, API and math demos are implemented. LambdaMART, transformer
retrieval and cross-encoder training require their optional dependencies and real ESCI data; they
are intentionally not described as completed experiments. `experiments/ab_design.md` is a design,
not a conducted A/B test.
