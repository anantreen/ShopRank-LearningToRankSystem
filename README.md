# ShopRank: E-Commerce Search and Learning-to-Rank

ShopRank is an end-to-end, production-oriented product-search project that demonstrates how a
search system moves from lexical retrieval to feature-based learning-to-rank, statistically sound
evaluation, semantic retrieval, business-aware reranking, and online serving.

The repository is designed for two audiences:

1. **Learners** who want readable implementations of BM25, TF-IDF, NDCG, RankNet-style pairwise
   optimization, bootstrap inference, and multiple-comparison correction.
2. **Interviewers and engineers** who want an honest, reproducible system with clear boundaries
   between implemented experiments, optional large-model stages, and proposed online tests.

> **Result integrity:** the checked-in results use the repository's small synthetic catalog only.
> They validate the complete software path; they are not presented as Amazon ESCI benchmark
> results. ESCI-scale numbers must be generated after downloading the dataset and running held-out
> experiments.

---

## 1. Problem formulation

Given a query \(q\) and products \(p_1,\ldots,p_n\), learn a scoring function

\[
f(q,p_i) \in \mathbb{R}
\]

such that sorting products by decreasing \(f(q,p_i)\) places the most useful products near the
top. Product relevance uses the ESCI convention:

| Label | Meaning | Numeric gain level |
|---|---|---:|
| E | Exact match | 3 |
| S | Substitute | 2 |
| C | Complement | 1 |
| I | Irrelevant | 0 |

This is a ranking problem rather than ordinary classification. Classification asks whether each
row is predicted correctly in isolation; search quality depends on the ordering of all products
for the same query.

### Example

For `wireless gaming mouse`, a sensible ordering is:

1. Logitech G304 wireless gaming mouse
2. Razer DeathAdder wired gaming mouse
3. Dell wired office mouse

All three rows contain the word “mouse,” but wireless capability, gaming intent, and product type
determine their relative positions.

---

## 2. System architecture

```text
                              OFFLINE
    product text ──> feature statistics / optional embeddings ──> indexes
                              │
                              ▼
                              ONLINE
    user query
        │
        ├──> BM25 lexical retrieval ─────────────┐
        │                                        │
        └──> optional MiniLM + FAISS retrieval ──┤
                                                 ▼
                                       candidate union / top-K
                                                 │
                                                 ▼
                              query-product feature computation
                                                 │
                                                 ▼
                             pairwise LTR / optional LambdaMART
                                                 │
                                                 ▼
                              availability and business constraints
                                                 │
                                                 ▼
                                        FastAPI response
```

The checked-in API uses BM25 candidate generation followed by the from-scratch pairwise linear
ranker. The MiniLM/FAISS and LightGBM adapters are optional because they require larger downloads
and, for credible evaluation, the real ESCI dataset.

### Why use multiple stages?

- **Retrieval** reduces millions of products to a manageable candidate set with high recall.
- **Ranking** spends more computation on those candidates to improve their order.
- **Business reranking** applies hard constraints such as availability and carefully weighted soft
  objectives such as seller quality.
- **Serving** exposes the pipeline with bounded inputs and measurable latency.

Applying a cross-encoder to every product would be computationally impractical. A staged design
reserves expensive interaction models for a small top-K list.

---

## 3. Repository structure

```text
.
├── 01_data_and_eda/              # dataset inspection and query-safe splitting
├── 02_bm25_baseline/             # lexical baseline entry points and generated results
├── 03_ltr/
│   ├── ranknet_from_scratch/     # pairwise probability and loss demonstration
│   ├── lambdarank_math/          # metric-weighted gradient demonstration
│   └── lambdamart/               # optional LightGBM ranker adapter
├── 04_semantic_ranking/
│   ├── biencoder/                # optional MiniLM + FAISS retrieval
│   └── hard_negative_crossencoder/
├── 05_evaluation/                # NDCG, paired bootstrap, Holm correction
├── 06_error_analysis/            # query-segment analysis
├── 07_multiobjective_demo/       # relevance + business-constraint demonstration
├── api/                          # FastAPI service and latency benchmark
├── data/                         # official ESCI download instructions
├── experiments/                  # explicitly unexecuted online A/B design
├── shoprank/                     # reusable implementation package
├── tests/                        # deterministic unit and integration tests
├── Dockerfile
├── pyproject.toml
└── run_experiment.py
```

---

## 4. Quick start

### 4.1 Run the dependency-free core

The core algorithms use the Python standard library, so the first experiment requires no ML
framework.

```bash
git clone https://github.com/anantreen/ShopRank-LearningToRankSystem.git
cd ShopRank-LearningToRankSystem

python3 -m unittest discover -s tests -v
python3 01_data_and_eda/eda.py
python3 run_experiment.py
python3 06_error_analysis/analysis.py
python3 07_multiobjective_demo/rerank.py
```

### 4.2 Install the API

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[core,api]'
uvicorn api.main:app --reload
```

Query it from another terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"wireless gaming mouse","limit":3}' \
  | python3 -m json.tool
```

Example response:

```json
{
  "query": "wireless gaming mouse",
  "products": [
    {
      "product_id": "P10",
      "title": "Logitech G304 Gaming Mouse",
      "relevance_score": 10.248625
    },
    {
      "product_id": "P12",
      "title": "Razer DeathAdder V3 Mouse",
      "relevance_score": 7.586518
    },
    {
      "product_id": "P11",
      "title": "Dell MS116 Wired Mouse",
      "relevance_score": 4.376888
    }
  ]
}
```

`relevance_score` is an uncalibrated ranking score, not a probability. Its ordering is meaningful;
the absolute magnitude is not.

### 4.3 Optional ML stacks

```bash
# LambdaMART
python -m pip install -e '.[ltr]'

# Sentence Transformers and FAISS
python -m pip install -e '.[semantic]'

# Everything
python -m pip install -e '.[all]'
```

---

## 5. Data and leak-free splitting

### Included teaching dataset

`shoprank/demo_data.py` contains 30 queries, 18 products, and 540 query-product judgments. It
includes brand queries, descriptive queries, model identifiers, misspellings, attributes,
substitutes, and complements. It is deterministic so tests and examples are reproducible.

Current EDA output:

| Statistic | Value |
|---|---:|
| Unique queries | 30 |
| Query-product judgments | 540 |
| Mean query length | 3.23 tokens |
| Median query length | 3 tokens |
| Brand-query fraction | 26.7% |
| Exact / Substitute / Complement / Irrelevant | 30 / 30 / 17 / 463 |
| Mean description length | 6.28 tokens |
| Missing descriptions | 0.0% |

The imbalance is intentional: real retrieval pools contain many irrelevant candidates. It also
shows why accuracy is misleading—a majority-class predictor could appear strong while producing
useless rankings.

### Amazon ESCI dataset

The production-scale path uses Amazon Science's [Shopping Queries Dataset](https://github.com/amazon-science/esci-data).
The official reduced ranking version contains 48,300 queries and 1,118,011 query-product judgments
across English, Spanish, and Japanese. Download and loading instructions are in
[`data/README.md`](data/README.md).

### Why split by query?

Rows belonging to one query are not independent; relevance is evaluated within the query's result
list. If products for query \(q\) appear in both training and test data, the model has effectively
seen the test query during training.

The correct split is:

\[
Q_{train} \cap Q_{validation} =
Q_{train} \cap Q_{test} =
Q_{validation} \cap Q_{test} = \varnothing.
\]

`shoprank/split.py` shuffles unique query IDs with a fixed seed, partitions them 80/10/10, and
asserts disjointness and completeness. Every row for a query follows that query into one split.

---

## 6. Lexical retrieval

### 6.1 TF-IDF cosine similarity

For term \(t\) in document \(d\), the implementation uses

\[
\operatorname{tfidf}(t,d)
= \operatorname{tf}(t,d)
\left(\log\frac{1+N}{1+\operatorname{df}(t)}+1\right),
\]

where \(N\) is the number of documents and \(\operatorname{df}(t)\) is document frequency. Query
and product vectors are L2-normalized, then compared with cosine similarity:

\[
\cos(q,d)=\frac{\mathbf{q}\cdot\mathbf{d}}
{\|\mathbf{q}\|_2\|\mathbf{d}\|_2}.
\]

Cosine similarity is scale-invariant after normalization, but raw term frequency can still reward
repetition too strongly and does not explicitly model document-length effects.

### 6.2 BM25

For query terms \(t\), ShopRank implements:

\[
\operatorname{BM25}(q,d)=
\sum_{t\in q}
\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)},
\]

with

\[
\operatorname{IDF}(t)=
\log\left(1+\frac{N-\operatorname{df}(t)+0.5}
{\operatorname{df}(t)+0.5}\right).
\]

The default parameters are \(k_1=1.5\) and \(b=0.75\).

#### Term-frequency saturation

Ignoring length normalization temporarily, define

\[
g(f)=\frac{f(k_1+1)}{f+k_1}.
\]

Then

\[
g'(f)=\frac{k_1(k_1+1)}{(f+k_1)^2}>0,
\qquad
g''(f)=\frac{-2k_1(k_1+1)}{(f+k_1)^3}<0.
\]

Therefore additional occurrences always help, but at a decreasing rate. Also,

\[
\lim_{f\to\infty}g(f)=k_1+1,
\]

so repeating a query term cannot increase the contribution without bound.

#### Length normalization

- \(b=0\): document length is ignored.
- \(b=1\): full normalization relative to average document length.
- \(0<b<1\): partial normalization, usually a practical compromise.

Without normalization, long descriptions can accumulate query terms merely because they contain
more words.

---

## 7. Ranking features

For every \((q,p)\) pair, `shoprank/features.py` calculates:

| Feature | Purpose |
|---|---|
| BM25 score | saturated lexical relevance with length normalization |
| TF-IDF cosine | normalized vector-space similarity |
| Title coverage | fraction of unique query tokens found in the title |
| Text Jaccard | intersection-over-union of query and product tokens |
| Brand match | exact presence of the product brand in the query |
| Exact phrase | full query appears inside the title |
| Query length | query complexity/context feature |
| Title length | product-text length signal |
| Description length | verbosity/missing-information signal |

Tree models can learn nonlinear interactions such as “brand match is valuable only when lexical
coverage is also high.” The from-scratch linear ranker is intentionally simpler so its optimization
can be inspected directly.

---

## 8. Learning to rank

### 8.1 Pointwise, pairwise, and listwise views

- **Pointwise:** predict each product's label independently, \(L(y_i,f(x_i))\).
- **Pairwise:** learn that product \(i\) should outrank product \(j\).
- **Listwise:** optimize a loss defined over the entire ranked list.

Search evaluation concerns order, so pairwise and listwise objectives align more naturally with the
task than independent classification accuracy.

### 8.2 RankNet pairwise probability

For products \(i\) and \(j\) with scores \(s_i\) and \(s_j\), define

\[
P_{ij}=P(i\succ j)=\sigma(s_i-s_j)
=\frac{1}{1+e^{-(s_i-s_j)}}.
\]

If \(i\) is preferred, the binary cross-entropy becomes

\[
L_{ij}=-\log P_{ij}
=\log\left(1+e^{-(s_i-s_j)}\right).
\]

Let \(\Delta=s_i-s_j\). Then

\[
\frac{\partial L}{\partial \Delta}
=-\frac{1}{1+e^{\Delta}}.
\]

Because

\[
\frac{\partial \Delta}{\partial s_i}=1,
\qquad
\frac{\partial \Delta}{\partial s_j}=-1,
\]

we obtain

\[
\frac{\partial L}{\partial s_i}
=-\frac{1}{1+e^{\Delta}},
\qquad
\frac{\partial L}{\partial s_j}
=\frac{1}{1+e^{\Delta}}.
\]

Gradient descent therefore increases the preferred item's score and decreases the non-preferred
item's score. As the ordering margin becomes safely positive, the gradient approaches zero.

### 8.3 Implemented linear pairwise ranker

The teaching model scores feature vector \(x\) using

\[
s(x)=w^\top x.
\]

For a preferred pair \((x_i,x_j)\),

\[
\Delta=w^\top(x_i-x_j).
\]

The weight update used by `shoprank/ranknet.py` is stochastic gradient descent:

\[
w \leftarrow w + \eta
\frac{x_i-x_j}{1+e^{\Delta}},
\]

where \(\eta\) is the learning rate. This is exactly the negative gradient direction of the
pairwise logistic loss.

### 8.4 LambdaRank

RankNet gives all preference errors a similar loss shape, even though a swap at ranks 1 and 2 has
more user impact than a swap at ranks 97 and 98. LambdaRank reweights the pair gradient using the
absolute NDCG change caused by swapping the two items:

\[
|\lambda_{ij}| \propto
|\Delta\operatorname{NDCG}_{ij}|
\frac{1}{1+e^{s_i-s_j}}.
\]

Let \(G_i=2^{rel_i}-1\), \(D(r)=1/\log_2(r+1)\), and IDCG be fixed for the query. Swapping items
\(i\) and \(j\) changes only their two DCG terms, so

\[
|\Delta\operatorname{NDCG}_{ij}|=
\frac{|(G_i-G_j)(D(r_i)-D(r_j))|}{\operatorname{IDCG}}.
\]

This identity explains the optimization behavior:

- similar relevance gains produce a small update;
- nearby low ranks have similar discounts and produce a small update;
- a high-gain error near the top produces a large update.

Run the concrete derivation:

```bash
python3 03_ltr/ranknet_from_scratch/ranknet_toy.py
python3 03_ltr/lambdarank_math/lambdarank_demo.py
```

For relevance list `[3, 0, 2, 1]`, the checked-in demonstration produces:

| Swap | Absolute change in NDCG |
|---|---:|
| ranks 1 and 2 | 0.2751 |
| ranks 1 and 3 | 0.2129 |
| ranks 1 and 4 | 0.3637 |
| ranks 2 and 3 | 0.0418 |
| ranks 2 and 4 | 0.0213 |
| ranks 3 and 4 | 0.0148 |

The top-rank swaps dominate the low-rank swaps, which is precisely the intended behavior.

### 8.5 LambdaMART

LambdaMART fits MART—multiple additive regression trees—to LambdaRank's metric-weighted pseudo-
gradients. The optional adapter uses `lightgbm.LGBMRanker(objective="lambdarank")` with graded gains
`[0, 1, 3, 7]`. Training rows must be contiguous by query and `group_sizes` must contain the number
of products for each query.

LightGBM is optional in this repository; no LambdaMART benchmark is claimed until it is trained and
evaluated on held-out ESCI query groups.

---

## 9. Evaluation mathematics

### 9.1 DCG and NDCG

For a predicted relevance sequence \(rel_1,\ldots,rel_k\),

\[
\operatorname{DCG@k}
=\sum_{r=1}^{k}\frac{2^{rel_r}-1}{\log_2(r+1)}.
\]

The exponential gain makes an Exact-over-Substitute improvement more valuable than a
Complement-over-Irrelevant improvement. The logarithmic denominator models decreasing attention
with rank.

Normalize by the ideal ordering:

\[
\operatorname{NDCG@k}
=\frac{\operatorname{DCG@k}}{\operatorname{IDCG@k}}.
\]

#### Why is NDCG bounded by 1?

The gains are nonnegative and the discounts decrease with rank. By the rearrangement inequality,
the dot product of gains and discounts is maximized when both are sorted in the same order—that is,
when the largest gain occupies rank 1. This maximum is IDCG. Therefore

\[
0\le \operatorname{DCG@k}\le\operatorname{IDCG@k}
\quad\Longrightarrow\quad
0\le\operatorname{NDCG@k}\le1.
\]

When a query has no positive gains, ShopRank returns 0 rather than dividing by zero.

### 9.2 Query-level averaging

The reported mean is

\[
\overline{\operatorname{NDCG@k}}
=\frac{1}{|Q|}\sum_{q\in Q}\operatorname{NDCG@k}(q).
\]

This gives each query equal importance. Pooling all judgment rows would incorrectly give queries
with more candidates greater weight.

### 9.3 Additional metrics

For the rank \(r_q\) of the first relevant result,

\[
\operatorname{MRR@k}
=\frac{1}{|Q|}\sum_{q\in Q}
\begin{cases}
1/r_q,&r_q\le k\\
0,&\text{otherwise.}
\end{cases}
\]

Precision@K measures the relevant fraction in the first K positions; Recall@K measures how much of
the query's available relevant set appears in those positions. NDCG remains the primary metric
because relevance is graded and position-sensitive.

---

## 10. Statistical comparison

### Paired query bootstrap

A small difference in mean NDCG can be sampling noise. ShopRank estimates uncertainty by
resampling queries—not rows—with replacement:

1. Compute per-query NDCG for models A and B on the same queries.
2. Draw \(|Q|\) query IDs with replacement.
3. Compute the resampled mean paired difference
   \(\Delta^*=\overline{NDCG_B-NDCG_A}\).
4. Repeat 10,000 times.
5. Use the 2.5th and 97.5th percentiles as a 95% interval.

Pairing removes query-difficulty variation from the comparison. Resampling individual product rows
would violate the independence assumption and usually understate uncertainty.

### Multiple comparisons

Comparing \(m\) models creates \(m(m-1)/2\) pairwise tests. ShopRank treats LambdaMART versus BM25
as the pre-specified primary comparison and uses Holm-Bonferroni adjustment for exploratory tests.

For sorted p-values \(p_{(1)}\le\cdots\le p_{(M)}\), Holm tests

\[
p_{(i)} \le \frac{\alpha}{M-i+1}
\]

sequentially until the first non-rejection. The procedure controls family-wise error while being
uniformly at least as powerful as ordinary Bonferroni correction.

---

## 11. Reproducible results

### Test status

```text
11 tests passed
```

The suite covers hand-computed DCG, NDCG bounds, equal query weighting, query-split disjointness,
lexical retrieval, end-to-end ranking, paired bootstrap behavior, and Holm correction.

### Synthetic catalog benchmark

Command:

```bash
python3 run_experiment.py
```

Output:

| Model | NDCG@10 |
|---|---:|
| TF-IDF | 0.9864 |
| BM25 | 0.9864 |
| Pairwise linear LTR | 0.9864 |

Pre-specified comparison, Pairwise LTR minus BM25:

\[
\Delta NDCG@10=0.0000,
\qquad 95\%\ CI=[0.0000,0.0000].
\]

### Interpretation

This result does **not** show that learning-to-rank is ineffective. The synthetic catalog is small,
has direct lexical matches, and is designed to make the code easy to verify. All three methods
already rank almost every held-out list ideally, creating a ceiling effect. There is no statistical
evidence of improvement on this teaching sample.

The learned weights are:

| Feature | Weight |
|---|---:|
| BM25 | +1.6093 |
| TF-IDF | +2.0062 |
| Title coverage | +1.2487 |
| Text Jaccard | -0.3747 |
| Brand match | -3.4764 |
| Exact phrase | +0.8094 |
| Query length | 0.0000 |
| Title length | +0.5995 |
| Description length | -0.6934 |

These coefficients must not be interpreted causally. Features are unstandardized and correlated,
and the sample is tiny. For example, the negative brand coefficient does not prove that brand
matches hurt relevance; other lexical features can absorb the same signal while the optimizer uses
the brand feature to correct particular training pairs. ESCI-scale permutation importance or SHAP
analysis would support a more credible interpretation.

The generated report is stored at [`02_bm25_baseline/results.md`](02_bm25_baseline/results.md).

---

## 12. Semantic retrieval

### Bi-encoder

A sentence transformer independently embeds the query and product:

\[
e_q=E(q),\qquad e_p=E(p).
\]

After normalization, FAISS inner-product search equals cosine similarity:

\[
e_q^\top e_p=\cos(e_q,e_p)
\quad\text{when}\quad
\|e_q\|_2=\|e_p\|_2=1.
\]

Product embeddings can be cached offline, making the bi-encoder suitable for candidate retrieval.
`04_semantic_ranking/biencoder/embed.py` uses `all-MiniLM-L6-v2` and `faiss.IndexFlatIP`.

### Cross-encoder

A cross-encoder processes query and product jointly:

```text
[CLS] query [SEP] product title and description [SEP]
```

Joint attention captures detailed token interactions but prevents independent product caching. It
is therefore appropriate for reranking tens of candidates, not retrieving from millions.

### Hard-negative ablation

The planned controlled experiment compares equal-sized training sets containing:

1. unrelated random negatives;
2. high-BM25, non-Exact hard negatives;
3. a 50/50 mixture.

Random negatives can teach a trivial “no overlapping words means irrelevant” shortcut. Hard
negatives force the model to distinguish plausible but incorrect products. The experiment plan is
implemented as documentation, but no cross-encoder result is claimed before actual fine-tuning.

---

## 13. Error analysis

`06_error_analysis/analysis.py` segments queries into model numbers, brands, misspellings, natural
language, and other queries. Aggregate NDCG can hide severe segment-specific failures.

Expected hypotheses to test on ESCI include:

- BM25 may outperform dense retrieval on exact identifiers such as `WH-1000XM5`.
- Dense retrieval may help descriptive queries such as `phone for elderly people`.
- Both approaches may struggle with misspellings unless character/subword or correction features
  are introduced.

A useful follow-up is a hybrid score

\[
s_{hybrid}=\alpha(q)s_{BM25}+[1-\alpha(q)]s_{dense},
\]

where identifier-heavy queries receive larger \(\alpha(q)\). This is a hypothesis, not a checked-in
benchmark claim.

---

## 14. Multi-objective ranking

Real commerce systems do not optimize textual relevance alone. The demonstration applies stock as
a hard constraint, then scores available candidates using

\[
S=w_rR_{relevance}+w_cP_{conversion}+w_sQ_{seller},
\]

with default weights

\[
(w_r,w_c,w_s)=(0.70,0.20,0.10).
\]

The included business features are explicitly synthetic. In a real system:

- availability should be fresh and usually acts as a filter;
- relevance should have a minimum floor;
- conversion propensity must be protected against position and exposure bias;
- seller quality, returns, diversity, latency, and long-term trust require guardrails.

Run:

```bash
python3 07_multiobjective_demo/rerank.py
```

The out-of-stock Nike Pegasus is removed even though it has the highest relevance score.

---

## 15. API and deployment

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | liveness response |
| POST | `/search` | BM25 candidates followed by pairwise reranking |

Request validation enforces a non-empty query of at most 200 characters and a result limit from 1
to 20.

### Latency benchmark

With the server running:

```bash
python3 api/benchmark.py
```

The script sends 100 sequential requests and reports mean, p50, p95, and p99 latency. No hardware-
independent latency number is checked into this README because latency depends on the machine,
Python environment, model configuration, warm-up, and concurrency.

### Docker

```bash
docker build -t shoprank .
docker run --rm -p 8000:8000 shoprank
```

For a catalog with hundreds of millions of products, the next architectural steps would include
sharded indexes, IVF/HNSW/PQ retrieval, category partitioning, hot-query caching, model versioning,
shadow evaluation, and observable feature pipelines. Kubernetes is not required to demonstrate the
ranking science in this repository.

---

## 16. Online experiment design

The proposed experiment in [`experiments/ab_design.md`](experiments/ab_design.md) is explicitly
labelled **not executed**.

- **Control:** BM25 plus the existing ranker.
- **Treatment:** hybrid semantic retrieval plus LambdaMART.
- **Randomization unit:** customer with sticky assignment.
- **Primary metric:** search CTR.
- **Secondary metrics:** add-to-cart rate, conversion, revenue per search session.
- **Guardrails:** p95 latency, zero-result rate, cancellations, and bounce rate.

Using baseline probability \(p\), minimum detectable absolute effect \(\delta\), two-sided type-I
error \(\alpha\), and power \(1-\beta\), a planning approximation per arm is

\[
n\approx
\frac{2(z_{1-\alpha/2}+z_{1-\beta})^2p(1-p)}{\delta^2}.
\]

For \(p=0.10\), \(\delta=0.005\), \(\alpha=0.05\), and 80% power, this approximation gives roughly
56,448 customers per arm. Production planning must account for clustering, repeated observations,
traffic allocation, novelty effects, and actual baseline variance.

---

## 17. Reproducibility checklist

- [x] Fixed random seeds
- [x] Query-disjoint train/validation/test partitions
- [x] Hand-computed metric tests
- [x] Per-query rather than pooled evaluation
- [x] Paired query bootstrap
- [x] Multiple-comparison correction
- [x] Generated result table committed with its data scope stated
- [x] Synthetic business features clearly labelled
- [x] Online experiment clearly labelled unexecuted
- [x] Optional heavyweight dependencies separated from the core
- [ ] ESCI experiment artifacts and held-out benchmark results
- [ ] Trained LambdaMART model artifact
- [ ] Trained cross-encoder hard-negative ablation

---

## 18. Roadmap to an ESCI portfolio result

1. Download the official dataset using [`data/README.md`](data/README.md).
2. Filter to `small_version == 1` and `product_locale == "us"`.
3. Begin with a few hundred complete query groups; never sample individual rows.
4. Reproduce TF-IDF and BM25 baselines using the same frozen test queries.
5. Train LambdaMART on contiguous query groups and record feature importance.
6. Run paired query-bootstrap comparisons against BM25.
7. Add bi-encoder retrieval and measure Recall@100 before reranking.
8. Fine-tune random, hard, and mixed-negative cross-encoders under matched settings.
9. Quantify the largest model gap by query segment and inspect concrete failures.
10. Report quality, uncertainty, latency, hardware, seed, data scope, and configuration together.

---

## 19. Key design decisions

| Decision | Reason |
|---|---|
| Standard-library core | makes foundational algorithms inspectable and immediately runnable |
| Query-level split | prevents leakage between dependent result-list rows |
| NDCG@10 primary metric | captures graded relevance and position |
| BM25 baseline before transformers | establishes a strong, explainable lexical reference |
| Pairwise teaching ranker | exposes the optimization instead of hiding it in a library |
| LambdaMART optional | credible claims require real data and installed LightGBM |
| Query-level bootstrap | respects the independent sampling unit |
| Optional semantic stack | avoids forcing large downloads for the core lesson |
| Raw scores labelled uncalibrated | ranking scores are not probabilities |
| Synthetic results clearly marked | prevents demo validation from being misrepresented as research |

---

## 20. References

1. C. Burges et al., “Learning to Rank using Gradient Descent,” ICML, 2005.
2. C. Burges, “From RankNet to LambdaRank to LambdaMART: An Overview,” Microsoft Research, 2010.
3. S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” 2009.
4. K. Järvelin and J. Kekäläinen, “Cumulated Gain-Based Evaluation of IR Techniques,” 2002.
5. C. K. Reddy et al., “Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
   Product Search,” 2022. [Dataset repository](https://github.com/amazon-science/esci-data).

---

## License and attribution

Add a repository license before redistributing this project. The Amazon Shopping Queries Dataset is
separately released under Apache-2.0; follow its repository's license and citation requirements.
