"""Attach attack categories to a trained map. Training labels enter only here."""

import numpy as np

from src.cluster import constrained_ward
from src.protocol import CLIP
from src.som import best_matching


def histograms(y, winners, n_units, n_classes):
    table = np.zeros((n_units, n_classes), dtype=np.float64)
    np.add.at(table, (winners, y), 1.0)
    return table


def _break_ties(scores, prior):
    """Largest score. A tie goes to the category with the larger training count."""
    best = scores.max(axis=1)
    tied = scores == best[:, None]
    # Among tied categories, pick the one with the largest prior. A remaining
    # tie takes the smallest index, which is deterministic.
    ranked = np.where(tied, prior[None, :], -1.0)
    return np.argmax(ranked, axis=1).astype(np.int64)


def majority_names(table, prior):
    empty = table.sum(axis=1) == 0
    names = _break_ties(table, prior)
    names[empty] = -1
    return names


def fill_empty(names, coords):
    """Unoccupied units take the name of the nearest occupied unit on the grid."""
    names = names.copy()
    missing = np.flatnonzero(names < 0)
    occupied = np.flatnonzero(names >= 0)
    if len(missing) == 0 or len(occupied) == 0:
        return names
    for unit in missing:
        gap = np.abs(coords[occupied] - coords[unit]).sum(axis=1)
        names[unit] = names[occupied[int(np.argmin(gap))]]
    return names


def cluster_names(weights, coords, counts, y, winners, n_clusters, prior):
    distances, cut, _ = constrained_ward(coords, weights, counts)
    unit_cluster = cut(n_clusters)
    n_classes = prior.shape[0]
    n_clusters_found = int(unit_cluster.max()) + 1 if np.any(unit_cluster >= 0) else 0
    cluster_table = np.zeros((n_clusters_found, n_classes), dtype=np.float64)
    valid = unit_cluster[winners] >= 0
    np.add.at(cluster_table, (unit_cluster[winners[valid]], y[valid]), 1.0)
    cluster_label = _break_ties(cluster_table, prior) if n_clusters_found else np.array([], dtype=np.int64)
    names = np.full(weights.shape[0], -1, dtype=np.int64)
    known = unit_cluster >= 0
    names[known] = cluster_label[unit_cluster[known]]
    names = fill_empty(names, coords)
    return names, distances


def predict_named(x, weights, names):
    winners = best_matching(x, weights)
    return names[winners]


def predict_weighted(x, weights, coords, table, class_weight, sigma):
    """Neighbourhood vote. Each unit contributes its histogram times the Gaussian weight."""
    winners = best_matching(x, weights)
    n_units, n_classes = table.shape
    rad2 = (CLIP * sigma) ** 2
    two_s2 = 2.0 * sigma * sigma
    weighted_hist = table * class_weight[None, :]
    neighbours = []
    kernel = []
    for unit in range(n_units):
        gap = np.sum((coords - coords[unit]) ** 2, axis=1)
        keep = np.flatnonzero(gap <= rad2)
        neighbours.append(keep)
        kernel.append(np.exp(-gap[keep] / two_s2))
    scores = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    for query, winner in enumerate(winners):
        local = neighbours[int(winner)]
        coeff = kernel[int(winner)]
        scores[query] = coeff @ weighted_hist[local]
    prior = class_weight  # not used for ties; ties follow the accumulated score
    # A tie in the accumulated score goes to the smaller class weight, which is
    # the more frequent category, matching the majority-vote tie rule.
    best = scores.max(axis=1)
    tied = scores == best[:, None]
    frequency = np.divide(1.0, class_weight, out=np.zeros_like(class_weight), where=class_weight > 0)
    ranked = np.where(tied, frequency[None, :], -1.0)
    return np.argmax(ranked, axis=1).astype(np.int64)
