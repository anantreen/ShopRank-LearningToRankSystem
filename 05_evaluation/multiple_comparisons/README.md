# Multiple comparisons

The primary comparison should be declared before looking at results: **LambdaMART vs BM25**.
If six models are compared, the 15 remaining pairwise tests are exploratory. Report their raw
p-values and Holm-adjusted p-values using `holm_bonferroni`; do not quietly select only wins.

