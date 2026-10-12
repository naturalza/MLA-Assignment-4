"""Gini classification tree with a depth limit and balanced class weights.

The controls are those locked in Assignment 2: a depth of 15, a minimum leaf
of one training row, and a weight n / (C n_c) on each row.
"""

import numba
import numpy as np

from src.protocol import TREE_DEPTH, TREE_MIN_LEAF


@numba.njit
def _scan(values, labels, weights, total, parent_gini, parent_sum, n_classes, min_leaf):
    left = np.zeros(n_classes, dtype=np.float64)
    best_gain = 0.0
    best_threshold = 0.0
    found = False
    n = values.shape[0]
    seen = 0
    for i in range(n - 1):
        left[labels[i]] += weights[i]
        seen += 1
        if seen < min_leaf or (n - seen) < min_leaf:
            continue
        if values[i] == values[i + 1]:
            continue
        left_sum = 0.0
        for c in range(n_classes):
            left_sum += left[c]
        right_sum = parent_sum - left_sum
        if left_sum <= 0.0 or right_sum <= 0.0:
            continue
        acc_l = 0.0
        acc_r = 0.0
        for c in range(n_classes):
            share_l = left[c] / left_sum
            share_r = (total[c] - left[c]) / right_sum
            acc_l += share_l * share_l
            acc_r += share_r * share_r
        gain = parent_gini - (left_sum * (1.0 - acc_l) + right_sum * (1.0 - acc_r)) / parent_sum
        if gain > best_gain:
            best_gain = gain
            best_threshold = 0.5 * (values[i] + values[i + 1])
            found = True
    return found, best_gain, best_threshold


def _gini(total):
    mass = total.sum()
    if mass <= 0:
        return 0.0
    share = total / mass
    return float(1.0 - np.dot(share, share))


def _leaf_label(total, prior):
    # Largest weight mass. A tie follows the training count stored in prior.
    best = total.max()
    tied = np.flatnonzero(total == best)
    if len(tied) == 1:
        return int(tied[0])
    return int(tied[np.argmax(prior[tied])])


def fit(x, y, sample_weight, n_classes, max_depth=TREE_DEPTH, min_leaf=TREE_MIN_LEAF):
    prior = np.bincount(y, minlength=n_classes).astype(np.float64)
    nodes = []

    def add_leaf(total):
        nodes.append((-1, -1, -1, 0.0, _leaf_label(total, prior)))
        return len(nodes) - 1

    def build(idx, depth):
        labels = y[idx]
        weights = sample_weight[idx]
        total = np.bincount(labels, minlength=n_classes).astype(np.float64)
        # bincount ignores weights. Rebuild the weighted mass.
        total = np.zeros(n_classes, dtype=np.float64)
        np.add.at(total, labels, weights)
        if depth >= max_depth or len(idx) < 2 * min_leaf or np.count_nonzero(total) <= 1:
            return add_leaf(total)
        parent_sum = float(total.sum())
        parent_gini = _gini(total)
        best = None
        for feature in range(x.shape[1]):
            order = np.argsort(x[idx, feature], kind="mergesort")
            found, gain, threshold = _scan(
                np.ascontiguousarray(x[idx, feature][order]),
                np.ascontiguousarray(labels[order]),
                np.ascontiguousarray(weights[order]),
                total,
                parent_gini,
                parent_sum,
                n_classes,
                min_leaf,
            )
            if found and (best is None or gain > best[0]):
                best = (gain, feature, threshold)
        if best is None:
            return add_leaf(total)
        _, feature, threshold = best
        column = x[idx, feature]
        left_idx = idx[column <= threshold]
        right_idx = idx[column > threshold]
        if len(left_idx) < min_leaf or len(right_idx) < min_leaf:
            return add_leaf(total)
        node_id = len(nodes)
        nodes.append((feature, -1, -1, threshold, -1))
        left_id = build(left_idx, depth + 1)
        right_id = build(right_idx, depth + 1)
        feature_i, _, _, threshold_i, _ = nodes[node_id]
        nodes[node_id] = (feature_i, left_id, right_id, threshold_i, -1)
        return node_id

    root = build(np.arange(x.shape[0], dtype=np.int64), 0)
    feature = np.array([node[0] for node in nodes], dtype=np.int64)
    left = np.array([node[1] for node in nodes], dtype=np.int64)
    right = np.array([node[2] for node in nodes], dtype=np.int64)
    threshold = np.array([node[3] for node in nodes], dtype=np.float64)
    label = np.array([node[4] for node in nodes], dtype=np.int64)
    return {"root": root, "feature": feature, "left": left, "right": right, "threshold": threshold, "label": label}


def predict(model, x):
    feature = model["feature"]
    left = model["left"]
    right = model["right"]
    threshold = model["threshold"]
    label = model["label"]
    out = np.empty(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        node = model["root"]
        while left[node] >= 0:
            if x[i, feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        out[i] = label[node]
    return out
