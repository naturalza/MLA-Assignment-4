"""Scores used for every comparison. Macro F is the primary score."""

import numpy as np


def confusion(y_true, y_pred, n_classes):
    table = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(table, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return table


def scores_from_confusion(table):
    """Return macro F, balanced accuracy, accuracy, and per-class arrays.

    Per-class recall of an empty class is recorded as zero so that a missing
    category pulls the macro average down rather than disappearing from it.
    """
    n_classes = table.shape[0]
    support = table.sum(axis=1).astype(np.float64)
    predicted = table.sum(axis=0).astype(np.float64)
    correct = np.diag(table).astype(np.float64)
    precision = np.divide(correct, predicted, out=np.zeros(n_classes), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(n_classes), where=support > 0)
    denom = precision + recall
    f_score = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    total = support.sum()
    accuracy = float(correct.sum() / total) if total else 0.0
    present = support > 0
    macro_f = float(f_score[present].mean()) if present.any() else 0.0
    balanced = float(recall[present].mean()) if present.any() else 0.0
    return {
        "macro_f": macro_f,
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "support": support,
    }


def evaluate(y_true, y_pred, n_classes):
    table = confusion(y_true, y_pred, n_classes)
    out = scores_from_confusion(table)
    out["confusion"] = table
    return out
