# Cross-encoder hard-negative experiment

This expensive experiment is deliberately a separate stage. Build three identically-sized sets:

1. **Random:** irrelevant products from other queries.
2. **Hard:** high-BM25 products for this query whose label is not Exact.
3. **Mixed:** 50% random and 50% hard.

Fine-tune `cross-encoder/ms-marco-MiniLM-L-6-v2` with the same seed, epochs and query split.
Compare query-level NDCG@10. Random negatives often teach an easy word-overlap shortcut; hard
negatives teach the model to distinguish plausible but wrong products. Do not claim results until
you have run this experiment on ESCI.

