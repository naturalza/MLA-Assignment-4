"""Five-fold refit of the locked map and the two reference models.

Controls come from results/lock.json. Nothing in this script chooses a control.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.knn import predict as knn_predict
from src.metrics import confusion, scores_from_confusion
from src.prep import Pipeline, cap_indices, class_weights, numeric_columns, sample_weights, skewed_columns
from src.protocol import CATEGORIES, K_NEIGHBOURS, N_FOLDS, SEED, TREE_DEPTH
from src.score_map import assignment, pack, predict_majority, predict_vote, predict_ward, ward_cuts
from src.som import continue_training, grow_once, train_map
from src.splits import stratified_folds
from src.tree import fit as fit_tree
from src.tree import predict as predict_tree

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
LOG = RESULTS / "final.log"
N_CLASSES = len(CATEGORIES)


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def load_frame():
    frame = pd.read_pickle(DATA / "dedup.pkl")
    y = frame["y"].to_numpy(dtype=np.int64)
    features = frame.drop(columns=["y"])
    return features, y


def fit_map(x_update, lock, seed):
    rng = np.random.default_rng(seed)
    params = lock["params"]
    if lock["grown"]:
        start_rows, start_cols = lock["growth_start"]
        model = train_map(x_update, start_rows, start_cols, rng, log=log, **params)
        for step in range(int(lock["growth_steps"])):
            model = grow_once(model, x_update)
            model = continue_training(
                model,
                x_update,
                np.random.default_rng(seed + 30 + step),
                eta0=params["eta0"],
                sigma_conv=params["sigma_conv"],
                log=log,
            )
        return model
    return train_map(x_update, int(lock["rows"]), int(lock["cols"]), rng, log=log, **params)


def predict_locked(model, x_train, y_train, x_test, rule, clusters):
    winners, counts, table, prior = assignment(model, x_train, y_train, N_CLASSES)
    if rule == "majority":
        pred = predict_majority(model, table, prior, x_test)
    elif rule == "weighted":
        pred = predict_vote(model, table, y_train, x_test, N_CLASSES)
    else:
        _, cut = ward_cuts(model, counts)
        pred = predict_ward(model, cut, y_train, winners, prior, x_test, clusters)
    return pred


def ambiguity(features, y):
    keys = pd.util.hash_pandas_object(features, index=False).to_numpy()
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    y = y[order]
    change = np.flatnonzero(np.diff(keys)) + 1
    bounds = np.concatenate([[0], change, [len(keys)]])
    distinct = len(bounds) - 1
    mixed_vectors = 0
    mixed_rows = 0
    for start, stop in zip(bounds[:-1], bounds[1:]):
        labels = np.unique(y[start:stop])
        if len(labels) > 1:
            mixed_vectors += 1
            mixed_rows += int(stop - start)
    # Majority label on each vector is an upper bound on accuracy.
    correct = 0
    for start, stop in zip(bounds[:-1], bounds[1:]):
        labels = y[start:stop]
        counts = np.bincount(labels, minlength=N_CLASSES)
        correct += int(counts.max())
    return {
        "distinct_vectors": int(distinct),
        "mixed_vectors": int(mixed_vectors),
        "mixed_rows": int(mixed_rows),
        "mixed_fraction": float(mixed_rows / len(y)),
        "majority_accuracy": float(correct / len(y)),
    }


def main():
    lock = json.loads((RESULTS / "lock.json").read_text())
    features, y = load_frame()
    folds = stratified_folds(y, N_FOLDS, SEED)
    done = []
    path = RESULTS / "folds.csv"
    if path.exists():
        done_frame = pd.read_csv(path)
        done = done_frame["fold"].astype(int).unique().tolist()
    else:
        done_frame = pd.DataFrame()
    numeric = numeric_columns(features)
    pooled = {name: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for name in ("som", "tree", "neighbours", "weighted_neighbours")}
    if (RESULTS / "pooled.npz").exists() and done:
        blob = np.load(RESULTS / "pooled.npz")
        for name in pooled:
            pooled[name] = blob[name]

    rows = done_frame.to_dict("records") if len(done_frame) else []
    fold_stops = []
    for fold, test_idx in enumerate(folds):
        if fold in done:
            log(f"fold {fold} already stored")
            continue
        train_idx = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold])
        log(f"fold {fold} train {len(train_idx)} test {len(test_idx)}")
        skewed = skewed_columns(features, train_idx, numeric)
        som_pipe = Pipeline(lock["transform"], skewed).fit(features, train_idx)
        tree_pipe = Pipeline("encode", skewed).fit(features, train_idx)
        neighbour_pipe = Pipeline("log", skewed).fit(features, train_idx)
        x_som = som_pipe.transform(features, train_idx)
        x_test_som = som_pipe.transform(features, test_idx)
        y_train = y[train_idx]
        y_test = y[test_idx]
        if lock["capped"]:
            capped = cap_indices(y, train_idx, 2000, SEED + 100 + fold)
            position = {int(row): i for i, row in enumerate(train_idx)}
            local = np.array([position[int(row)] for row in capped], dtype=np.int64)
            x_update = x_som[local]
        else:
            x_update = x_som
        # Each fold draws its initial codebook under its own seed, SEED + fold.
        model = fit_map(x_update, lock, SEED + fold)
        fold_stops.append(model["stopped"])
        pred = predict_locked(model, x_som, y_train, x_test_som, lock["rule"], int(lock["clusters"]))
        pooled["som"] += confusion(y_test, pred, N_CLASSES)
        record("som", fold, y_test, pred, rows)

        x_tree = tree_pipe.transform(features, train_idx)
        x_test_tree = tree_pipe.transform(features, test_idx)
        tree = fit_tree(x_tree, y_train, sample_weights(y_train, N_CLASSES), N_CLASSES, max_depth=TREE_DEPTH)
        tree_pred = predict_tree(tree, x_test_tree)
        pooled["tree"] += confusion(y_test, tree_pred, N_CLASSES)
        record("tree", fold, y_test, tree_pred, rows)

        x_near = neighbour_pipe.transform(features, train_idx)
        x_test_near = neighbour_pipe.transform(features, test_idx)
        uniform, weighted = knn_predict(x_near, y_train, x_test_near, class_weights(y_train, N_CLASSES), k=K_NEIGHBOURS)
        pooled["neighbours"] += confusion(y_test, uniform, N_CLASSES)
        pooled["weighted_neighbours"] += confusion(y_test, weighted, N_CLASSES)
        record("neighbours", fold, y_test, uniform, rows)
        record("weighted_neighbours", fold, y_test, weighted, rows)
        pd.DataFrame(rows).to_csv(path, index=False)
        np.savez(RESULTS / "pooled.npz", **pooled)
        log(f"fold {fold} stored")

    summary = []
    frame = pd.DataFrame(rows)
    for model, group in frame.groupby("model"):
        summary.append(
            {
                "model": model,
                "macro_f": group["macro_f"].mean(),
                "macro_f_sd": group["macro_f"].std(ddof=0),
                "balanced_accuracy": group["balanced_accuracy"].mean(),
                "accuracy": group["accuracy"].mean(),
            }
        )
    pd.DataFrame(summary).to_csv(RESULTS / "summary.csv", index=False)
    per_class = []
    for model, table in pooled.items():
        scores = scores_from_confusion(table)
        for index, name in enumerate(CATEGORIES):
            per_class.append(
                {
                    "model": model,
                    "category": name,
                    "precision": scores["precision"][index],
                    "recall": scores["recall"][index],
                    "f_score": scores["f_score"][index],
                    "support": scores["support"][index],
                }
            )
    pd.DataFrame(per_class).to_csv(RESULTS / "per_class.csv", index=False)
    amb = ambiguity(features, y)
    (RESULTS / "ambiguity.json").write_text(json.dumps(amb, indent=2))
    log(f"batch stopping rule per fold {fold_stops}")
    log(f"summary {json.dumps(summary)}")
    log(f"ambiguity {json.dumps(amb)}")


def record(model, fold, y_true, y_pred, rows):
    scores = pack(y_true, y_pred, N_CLASSES)
    rows.append(
        {
            "fold": fold,
            "model": model,
            "macro_f": scores["macro_f"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "accuracy": scores["accuracy"],
        }
    )
    log(
        f"fold {fold} {model} macro_f {scores['macro_f']:.4f} "
        f"balanced {scores['balanced_accuracy']:.4f} accuracy {scores['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
