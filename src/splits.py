"""Stratified partitions. The class mix drifts along the file, so every split shuffles."""

import numpy as np


def _per_class(y, rng):
    groups = []
    for label in range(int(y.max()) + 1):
        idx = np.flatnonzero(y == label)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        groups.append(idx)
    return groups


def stratified_parts(y, fractions, seed):
    """Split indices into contiguous stratified portions. fractions sums to one."""
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in fractions]
    for idx in _per_class(y, rng):
        n = len(idx)
        bounds = np.cumsum(np.round(np.asarray(fractions) * n)).astype(int)
        bounds[-1] = n
        start = 0
        for bucket, bound in zip(buckets, bounds):
            bucket.append(idx[start:bound])
            start = int(bound)
    parts = []
    for bucket in buckets:
        joined = np.concatenate(bucket)
        rng.shuffle(joined)
        parts.append(joined)
    return parts


def stratified_folds(y, n_folds, seed):
    """Return a list of test-index arrays, one per fold."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for idx in _per_class(y, rng):
        for i, row in enumerate(idx):
            folds[i % n_folds].append(int(row))
    out = []
    for fold in folds:
        arr = np.asarray(fold, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out
