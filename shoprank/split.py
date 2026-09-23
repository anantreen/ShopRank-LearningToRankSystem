"""Leak-free splitting: all rows for one query stay in one partition."""

import random
from collections.abc import Iterable


def split_query_ids(
    query_ids: Iterable[str], train_size: float = 0.8, val_size: float = 0.1, seed: int = 42
) -> tuple[set[str], set[str], set[str]]:
    if not 0 < train_size < 1 or not 0 <= val_size < 1 or train_size + val_size >= 1:
        raise ValueError("sizes must satisfy 0 < train, 0 <= val, train + val < 1")
    unique = sorted(set(query_ids))
    random.Random(seed).shuffle(unique)
    n_train = int(len(unique) * train_size)
    n_val = int(len(unique) * val_size)
    train = set(unique[:n_train])
    val = set(unique[n_train : n_train + n_val])
    test = set(unique[n_train + n_val :])
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(unique)
    return train, val, test

