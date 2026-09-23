# LambdaRank, with one concrete list

Start with true relevance `[3, 0, 2, 1]` and model scores `[1.2, -0.3, 0.7, 0.1]`.
RankNet learns from every preferred pair using the logistic loss:

`P(i beats j) = 1 / (1 + exp(-(score_i - score_j)))`

Its weakness is that all ordering mistakes have similar importance. LambdaRank multiplies the
pairwise gradient by `|delta NDCG|`: the absolute change in NDCG if positions `i` and `j` swap.
Therefore, a mistake near rank 1 receives a larger update than the same mistake near rank 100.
LambdaMART fits gradient-boosted decision trees to these metric-aware gradients.

Run `python lambdarank_demo.py` to see every swap's delta.

