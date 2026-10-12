"""Validation scores for a trained map. Labels are read only in this module."""

import numpy as np

from src.cluster import constrained_ward
from src.label import _break_ties, fill_empty, histograms, majority_names, predict_named, predict_weighted
from src.metrics import evaluate
from src.prep import class_weights
from src.som import best_matching


def assignment(model, x_train, y_train, n_classes):
    winners = best_matching(x_train, model["weights"])
    n_units = model["weights"].shape[0]
    counts = np.bincount(winners, minlength=n_units).astype(np.float64)
    table = histograms(y_train, winners, n_units, n_classes)
    prior = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    return winners, counts, table, prior


def predict_majority(model, table, prior, x_eval):
    names = fill_empty(majority_names(table, prior), model["coords"])
    return predict_named(x_eval, model["weights"], names)


def ward_cuts(model, counts):
    distances, cut, _ = constrained_ward(model["coords"], model["weights"], counts)
    return distances, cut


def predict_ward(model, cut, y_train, winners, prior, x_eval, n_clusters):
    unit_cluster = cut(n_clusters)
    known = unit_cluster >= 0
    n_found = int(unit_cluster[known].max()) + 1 if known.any() else 0
    names = np.full(model["weights"].shape[0], -1, dtype=np.int64)
    if n_found:
        mass = np.zeros((n_found, prior.shape[0]), dtype=np.float64)
        assigned = unit_cluster[winners]
        valid = assigned >= 0
        np.add.at(mass, (assigned[valid], y_train[valid]), 1.0)
        chosen = _break_ties(mass, prior)
        names[known] = chosen[unit_cluster[known]]
    names = fill_empty(names, model["coords"])
    return predict_named(x_eval, model["weights"], names)


def predict_vote(model, table, y_train, x_eval, n_classes):
    return predict_weighted(
        x_eval,
        model["weights"],
        model["coords"],
        table,
        class_weights(y_train, n_classes),
        model["sigma"],
    )


def pack(y_true, y_pred, n_classes):
    scores = evaluate(y_true, y_pred, n_classes)
    return {
        "macro_f": scores["macro_f"],
        "balanced_accuracy": scores["balanced_accuracy"],
        "accuracy": scores["accuracy"],
        "recall": scores["recall"],
        "precision": scores["precision"],
        "f_score": scores["f_score"],
        "confusion": scores["confusion"],
    }
