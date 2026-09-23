"""Optional LightGBM trainer. Install with: pip install -e '.[ltr]'"""


def train_lambdamart(features, labels, group_sizes):
    try:
        from lightgbm import LGBMRanker
    except ImportError as error:
        raise RuntimeError("LightGBM is optional. Run: pip install -e '.[ltr]'") from error
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=[0, 1, 3, 7],
        n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=42,
    )
    model.fit(features, labels, group=group_sizes)
    return model

