"""Nearest neighbours with a uniform vote and an inverse-frequency vote.

Both votes use the same seven neighbours. The uniform vote is the Assignment 2
control. The weighted vote is the comparison that study did not run.
"""

import numpy as np

from src.protocol import K_NEIGHBOURS


def _distances(chunk, train):
    return (
        np.sum(chunk * chunk, axis=1)[:, None]
        + np.sum(train * train, axis=1)[None, :]
        - 2.0 * chunk @ train.T
    )


def neighbour_indices(train, test, k=K_NEIGHBOURS, query_block=512, train_block=8192):
    k = min(k, train.shape[0])
    n_query = test.shape[0]
    best_d = np.full((n_query, k), np.inf, dtype=np.float64)
    best_i = np.full((n_query, k), -1, dtype=np.int64)
    for q0 in range(0, n_query, query_block):
        q1 = min(q0 + query_block, n_query)
        local_d = np.full((q1 - q0, k), np.inf, dtype=np.float64)
        local_i = np.full((q1 - q0, k), -1, dtype=np.int64)
        for t0 in range(0, train.shape[0], train_block):
            t1 = min(t0 + train_block, train.shape[0])
            dist = _distances(test[q0:q1], train[t0:t1])
            joined_d = np.concatenate([local_d, dist], axis=1)
            joined_i = np.concatenate(
                [local_i, np.arange(t0, t1, dtype=np.int64)[None, :].repeat(q1 - q0, axis=0)],
                axis=1,
            )
            pick = np.argpartition(joined_d, k - 1, axis=1)[:, :k]
            rows = np.arange(q1 - q0)[:, None]
            local_d = joined_d[rows, pick]
            local_i = joined_i[rows, pick]
        order = np.argsort(local_d, axis=1)
        best_d[q0:q1] = local_d[np.arange(q1 - q0)[:, None], order]
        best_i[q0:q1] = local_i[np.arange(q1 - q0)[:, None], order]
    return best_i


def _vote(labels, weight, prior):
    """labels is (n, k). weight is per class. A tie follows the training count."""
    n_query, k = labels.shape
    n_classes = weight.shape[0]
    scores = np.zeros((n_query, n_classes), dtype=np.float64)
    for neighbour in range(k):
        lab = labels[:, neighbour]
        scores[np.arange(n_query), lab] += weight[lab]
    best = scores.max(axis=1)
    tied = scores == best[:, None]
    ranked = np.where(tied, prior[None, :], -1.0)
    return np.argmax(ranked, axis=1).astype(np.int64)


def predict(train, y_train, test, class_weight, k=K_NEIGHBOURS):
    indices = neighbour_indices(train, test, k=k)
    labels = y_train[indices]
    uniform_weight = np.ones(class_weight.shape[0], dtype=np.float64)
    prior = np.divide(1.0, class_weight, out=np.zeros_like(class_weight), where=class_weight > 0)
    uniform = _vote(labels, uniform_weight, prior)
    weighted = _vote(labels, class_weight, prior)
    return uniform, weighted
