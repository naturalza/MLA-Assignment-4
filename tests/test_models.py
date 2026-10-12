"""Checks that do not need the traffic table."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster import _ward, constrained_ward
from src.knn import predict as knn_predict
from src.metrics import evaluate
from src.prep import class_weights
from src.som import best_matching, grow_once, quantisation_error, squared_distances, train_map
from src.tree import fit as fit_tree
from src.tree import predict as predict_tree


def test_ward_distance_matches_the_formula():
    distance = _ward(2.0, np.array([0.0, 0.0]), 2.0, np.array([2.0, 0.0]))
    assert abs(distance - 4.0) < 1e-12


def test_constrained_ward_merges_adjacent_equals_first():
    coords = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.float64)
    weights = np.array([[0.0], [0.0], [10.0], [10.0]])
    counts = np.ones(4)
    distances, cut, _ = constrained_ward(coords, weights, counts)
    labels = cut(2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert distances[0] == 0.0


def test_best_matching_unit_matches_a_direct_scan():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4))
    weights = rng.normal(size=(6, 4))
    winners = best_matching(x, weights)
    dist = squared_distances(x, weights)
    assert np.array_equal(winners, np.argmin(dist, axis=1))


def test_quantisation_error_falls_on_separated_blobs():
    rng = np.random.default_rng(11)
    centres = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    parts = [rng.normal(loc=centre, scale=0.2, size=(40, 2)) for centre in centres]
    x = np.vstack(parts)
    model = train_map(x, 3, 3, rng, order_epochs=4, batch_epochs=8)
    errors = [row[2] for row in model["history"]]
    assert np.isfinite(errors).all()
    assert errors[-1] < errors[0]


def test_growth_inserts_a_row_and_a_column():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 2))
    model = train_map(x, 2, 2, rng, order_epochs=1, batch_epochs=1)
    grown = grow_once(model, x, gamma=0.5)
    assert grown["rows"] == 3 and grown["cols"] == 3
    assert grown["weights"].shape == (9, 2)


def test_tree_separates_a_single_threshold():
    x = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])
    weight = np.ones(4)
    model = fit_tree(x, y, weight, n_classes=2, max_depth=2, min_leaf=1)
    assert np.array_equal(predict_tree(model, x), y)


def test_class_weighted_vote_can_overturn_a_majority():
    train = np.array([[0.0], [0.1], [0.2], [5.0]])
    y = np.array([0, 0, 0, 1])
    # Query sits on the single class-1 row. k = 3 also pulls in two class-0 rows.
    test = np.array([[5.0]])
    weight = class_weights(y, 2)
    # Force the neighbourhood to include the class-0 mass by querying a midpoint
    # is unnecessary: with k = 3 and the query at 5, neighbours are the point at
    # 5 and the two nearest of the rest. That is one class 1 and two class 0.
    uniform, weighted = knn_predict(train, y, test, weight, k=3)
    assert uniform[0] == 0
    assert weighted[0] == 1


def test_macro_f_penalises_a_missed_category():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    scores = evaluate(y, pred, 2)
    assert scores["macro_f"] < 0.5
    assert scores["recall"][1] == 0.0
