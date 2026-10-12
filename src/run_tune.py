"""Tuning split. Decision rules are fixed in src/protocol.py. No test score is read."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.prep import Pipeline, cap_indices, deduplicate, load_raw, numeric_columns, skewed_columns
from src.protocol import (
    CAP,
    CATEGORIES,
    CLUSTER_COUNTS,
    GROW_LIMIT,
    MAP_SIZES,
    SEED,
    SIZE_BAND,
    TRANSFORM_SHAPE,
    TUNE_FRACTIONS,
    WORMS,
    WORMS_RECALL_TRIGGER,
)
from src.score_map import assignment, pack, predict_majority, predict_vote, predict_ward, ward_cuts
from src.som import continue_training, grow_once, train_map
from src.splits import stratified_parts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
LOG = RESULTS / "tune.log"
N_CLASSES = len(CATEGORIES)
WORMS_INDEX = CATEGORIES.index(WORMS)


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def save_model(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    history = model["history"]
    np.savez_compressed(
        path,
        weights=model["weights"],
        coords=model["coords"],
        rows=np.int64(model["rows"]),
        cols=np.int64(model["cols"]),
        sigma=np.float64(model["sigma"]),
        phase=np.array([row[0] for row in history]),
        epoch=np.array([row[1] for row in history], dtype=np.int64),
        qe=np.array([row[2] for row in history], dtype=np.float64),
        width=np.array([row[3] for row in history], dtype=np.float64),
    )


def load_model(path):
    blob = np.load(path, allow_pickle=False)
    history = list(zip(blob["phase"].tolist(), blob["epoch"].tolist(), blob["qe"].tolist(), blob["width"].tolist()))
    return {
        "weights": blob["weights"],
        "coords": blob["coords"],
        "rows": int(blob["rows"]),
        "cols": int(blob["cols"]),
        "sigma": float(blob["sigma"]),
        "history": history,
    }


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def read_csv(path):
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def prepare():
    cache = DATA / "dedup.pkl"
    index_path = RESULTS / "tune_indices.npz"
    if cache.exists():
        frame = pd.read_pickle(cache)
        features = frame.drop(columns=["y"])
        y = frame["y"].to_numpy(dtype=np.int64)
    else:
        raw = load_raw(DATA / "UNSW_NB15_training-set.csv", DATA / "UNSW_NB15_testing-set.csv")
        features, y = deduplicate(raw)
        cached = features.copy()
        cached["y"] = y
        cached.to_pickle(cache)
    if index_path.exists():
        blob = np.load(index_path)
        parts = [blob["train"], blob["val"], blob["hold"]]
    else:
        parts = stratified_parts(y, TUNE_FRACTIONS, SEED)
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez(index_path, train=parts[0], val=parts[1], hold=parts[2])
    return features, y, parts


def fit_matrix(features, train_idx, val_idx, kind, skewed):
    pipe = Pipeline(kind, skewed).fit(features, train_idx)
    return pipe.transform(features, train_idx), pipe.transform(features, val_idx), pipe


def majority_scores(model, x_train, y_train, x_val, y_val):
    _, _, table, prior = assignment(model, x_train, y_train, N_CLASSES)
    pred = predict_majority(model, table, prior, x_val)
    return pack(y_val, pred, N_CLASSES)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    features, y, (train_idx, val_idx, hold_idx) = prepare()
    log(f"rows {len(y)} train {len(train_idx)} val {len(val_idx)} hold {len(hold_idx)} unused")
    numeric = numeric_columns(features)
    skewed = skewed_columns(features, train_idx, numeric)
    log(f"numeric {len(numeric)} skewed {len(skewed)}")
    (RESULTS / "skewed.json").write_text(json.dumps(skewed))

    transform_rows = read_csv(RESULTS / "transforms.csv")
    done_kinds = {row["kind"] for row in transform_rows}
    matrices = {}
    for kind in ("encode", "scale", "log"):
        if kind in done_kinds:
            continue
        started = time.time()
        log(f"transform {kind}")
        x_train, x_val, _ = fit_matrix(features, train_idx, val_idx, kind, skewed)
        matrices[kind] = (x_train, x_val)
        model = train_map(
            x_train,
            TRANSFORM_SHAPE[0],
            TRANSFORM_SHAPE[1],
            np.random.default_rng(SEED),
            log=log,
        )
        scores = majority_scores(model, x_train, y[train_idx], x_val, y[val_idx])
        transform_rows.append(
            {
                "kind": kind,
                "macro_f": scores["macro_f"],
                "balanced_accuracy": scores["balanced_accuracy"],
                "accuracy": scores["accuracy"],
                "qe": model["history"][-1][2],
                "seconds": time.time() - started,
                "width": x_train.shape[1],
            }
        )
        write_csv(RESULTS / "transforms.csv", transform_rows)
        save_model(MODELS / f"transform_{kind}.npz", model)
        log(f"transform {kind} macro_f {scores['macro_f']:.4f}")

    kind = choose_transform(transform_rows)
    log(f"locked transform {kind}")
    if kind not in matrices:
        x_train, x_val, _ = fit_matrix(features, train_idx, val_idx, kind, skewed)
    else:
        x_train, x_val = matrices[kind]
    y_train = y[train_idx]
    y_val = y[val_idx]

    size_rows = read_csv(RESULTS / "sizes.csv")
    done_sizes = {(int(row["rows"]), int(row["cols"])) for row in size_rows}
    for rows, cols in MAP_SIZES:
        if (rows, cols) in done_sizes:
            continue
        started = time.time()
        log(f"size {rows}x{cols}")
        model = train_map(x_train, rows, cols, np.random.default_rng(SEED), log=log)
        scores = majority_scores(model, x_train, y_train, x_val, y_val)
        size_rows.append(
            {
                "rows": rows,
                "cols": cols,
                "macro_f": scores["macro_f"],
                "balanced_accuracy": scores["balanced_accuracy"],
                "accuracy": scores["accuracy"],
                "qe": model["history"][-1][2],
                "seconds": time.time() - started,
            }
        )
        write_csv(RESULTS / "sizes.csv", size_rows)
        save_model(MODELS / f"size_{rows}x{cols}.npz", model)
        log(f"size {rows}x{cols} macro_f {scores['macro_f']:.4f} qe {model['history'][-1][2]:.6f}")

    chosen_rows, chosen_cols = choose_size(size_rows)
    log(f"locked size {chosen_rows}x{chosen_cols}")
    base = load_model(MODELS / f"size_{chosen_rows}x{chosen_cols}.npz")

    seed_rows = read_csv(RESULTS / "seeds.csv")
    done_seeds = {int(row["seed"]) for row in seed_rows}
    for offset in range(3):
        seed = SEED + offset
        if seed in done_seeds:
            continue
        if offset == 0:
            model = base
            scores = majority_scores(model, x_train, y_train, x_val, y_val)
        else:
            log(f"seed {seed}")
            model = train_map(x_train, chosen_rows, chosen_cols, np.random.default_rng(seed), log=log)
            scores = majority_scores(model, x_train, y_train, x_val, y_val)
            save_model(MODELS / f"seed_{seed}.npz", model)
        seed_rows.append({"seed": seed, "macro_f": scores["macro_f"], "qe": model["history"][-1][2]})
        write_csv(RESULTS / "seeds.csv", seed_rows)
        log(f"seed {seed} macro_f {scores['macro_f']:.4f}")
    spread = max(row["macro_f"] for row in seed_rows) - min(row["macro_f"] for row in seed_rows)
    log(f"seed spread {spread:.4f}")

    name_rows = read_csv(RESULTS / "names.csv")
    if not name_rows:
        name_rows = score_names(base, x_train, y_train, x_val, y_val)
        write_csv(RESULTS / "names.csv", name_rows)
        for row in name_rows:
            log(f"name {row['rule']} clusters {row['clusters']} macro_f {row['macro_f']:.4f}")
    rule, clusters = choose_name(name_rows)
    log(f"locked name {rule} clusters {clusters}")
    reference = float(next(row["macro_f"] for row in name_rows if row["rule"] == rule and int(row["clusters"]) == clusters))

    grown_rows = read_csv(RESULTS / "growth.csv")
    grown = False
    final_model = base
    if not grown_rows:
        by_size = {(int(row["rows"]), int(row["cols"])): float(row["macro_f"]) for row in size_rows}
        rise = by_size[(24, 24)] - by_size[(20, 20)]
        log(f"growth trigger rise {rise:.4f} spread {spread:.4f}")
        if rise > spread:
            current = load_model(MODELS / "size_24x24.npz")
            current_score = reference
            step = 0
            while current["rows"] < GROW_LIMIT and current["cols"] < GROW_LIMIT:
                step += 1
                log(f"growth step {step}")
                expanded = grow_once(current, x_train)
                if expanded["rows"] == current["rows"]:
                    break
                expanded = continue_training(
                    expanded, x_train, np.random.default_rng(SEED + 50 + step), log=log
                )
                scores = score_rule(expanded, x_train, y_train, x_val, y_val, rule, clusters)
                grown_rows.append(
                    {
                        "step": step,
                        "rows": expanded["rows"],
                        "cols": expanded["cols"],
                        "macro_f": scores["macro_f"],
                        "qe": expanded["history"][-1][2],
                    }
                )
                write_csv(RESULTS / "growth.csv", grown_rows)
                save_model(MODELS / f"grown_{step}.npz", expanded)
                log(f"growth step {step} {expanded['rows']}x{expanded['cols']} macro_f {scores['macro_f']:.4f}")
                if scores["macro_f"] <= current_score:
                    break
                current = expanded
                current_score = scores["macro_f"]
                grown = True
                final_model = expanded
                reference = current_score
                chosen_rows, chosen_cols = expanded["rows"], expanded["cols"]
        if not grown_rows:
            write_csv(RESULTS / "growth.csv", [{"step": 0, "rows": chosen_rows, "cols": chosen_cols, "macro_f": reference, "qe": base["history"][-1][2]}])
    else:
        improving = [row for row in grown_rows if int(row["step"]) > 0]
        if improving:
            last = max(improving, key=lambda row: int(row["step"]))
            # A step is kept only when it raised the score. The file stores every attempt,
            # including the attempt that failed to raise the score. Reload the best step.
            best = max(improving, key=lambda row: float(row["macro_f"]))
            if float(best["macro_f"]) > reference:
                final_model = load_model(MODELS / f"grown_{int(best['step'])}.npz")
                reference = float(best["macro_f"])
                grown = True
                chosen_rows, chosen_cols = int(best["rows"]), int(best["cols"])

    worms_recall = unweighted_worms(final_model, x_train, y_train, x_val, y_val, name_rows if not grown else None)
    log(f"worms recall under best unweighted rule {worms_recall:.4f}")
    capped = False
    cap_rows = read_csv(RESULTS / "cap.csv")
    if worms_recall < WORMS_RECALL_TRIGGER and not cap_rows:
        log("cap triggered")
        capped_idx = cap_indices(y, train_idx, CAP, SEED + 17)
        position = {int(row): i for i, row in enumerate(train_idx)}
        local_idx = np.array([position[int(row)] for row in capped_idx], dtype=np.int64)
        update = x_train[local_idx]
        if grown:
            model = train_map(update, 24, 24, np.random.default_rng(SEED + 17), log=log)
            steps = final_model["rows"] - 24
            for step in range(steps):
                model = grow_once(model, update)
                model = continue_training(model, update, np.random.default_rng(SEED + 70 + step), log=log)
        else:
            model = train_map(
                update,
                final_model["rows"],
                final_model["cols"],
                np.random.default_rng(SEED + 17),
                log=log,
            )
        # Names come from the full training partition, not from the capped draw.
        scores = score_rule(model, x_train, y_train, x_val, y_val, rule, clusters)
        cap_rows = [{"macro_f": scores["macro_f"], "balanced_accuracy": scores["balanced_accuracy"], "accuracy": scores["accuracy"], "worms_recall": float(scores["recall"][WORMS_INDEX])}]
        write_csv(RESULTS / "cap.csv", cap_rows)
        save_model(MODELS / "capped.npz", model)
        log(f"cap macro_f {scores['macro_f']:.4f}")
        if scores["macro_f"] > reference:
            capped = True
            final_model = model
            reference = scores["macro_f"]
    elif cap_rows and float(cap_rows[0]["macro_f"]) > reference:
        capped = True
        final_model = load_model(MODELS / "capped.npz")

    history_rows = [
        {"phase": phase, "epoch": int(epoch), "qe": float(qe), "width": float(width)}
        for phase, epoch, qe, width in final_model["history"]
    ]
    write_csv(RESULTS / "qe_history.csv", history_rows)
    lock = {
        "transform": kind,
        "rows": int(final_model["rows"]),
        "cols": int(final_model["cols"]),
        "rule": rule,
        "clusters": int(clusters),
        "capped": bool(capped),
        "grown": bool(grown),
        "growth_steps": int(final_model["rows"] - 24) if grown else 0,
        "seed_spread": float(spread),
        "validation_macro_f": float(reference),
        "worms_recall_unweighted": float(worms_recall),
        "skewed": skewed,
    }
    (RESULTS / "lock.json").write_text(json.dumps(lock, indent=2))
    log(f"lock {json.dumps(lock)}")


def choose_transform(rows):
    order = {"encode": 0, "scale": 1, "log": 2}
    return sorted(rows, key=lambda row: (-float(row["macro_f"]), order[row["kind"]]))[0]["kind"]


def choose_size(rows):
    best = max(float(row["macro_f"]) for row in rows)
    eligible = [row for row in rows if float(row["macro_f"]) >= best - SIZE_BAND]
    eligible.sort(key=lambda row: int(row["rows"]) * int(row["cols"]))
    return int(eligible[0]["rows"]), int(eligible[0]["cols"])


def choose_name(rows):
    simplicity = {"majority": 0, "ward": 1, "weighted": 2}

    def key(row):
        return (
            -float(row["macro_f"]),
            -float(row["balanced_accuracy"]),
            simplicity[row["rule"]],
            int(row["clusters"]),
        )

    picked = sorted(rows, key=key)[0]
    return picked["rule"], int(picked["clusters"])


def score_names(model, x_train, y_train, x_val, y_val):
    winners, counts, table, prior = assignment(model, x_train, y_train, N_CLASSES)
    rows = []
    pred = predict_majority(model, table, prior, x_val)
    scores = pack(y_val, pred, N_CLASSES)
    rows.append(row_from("majority", 0, scores))
    pred = predict_vote(model, table, y_train, x_val, N_CLASSES)
    scores = pack(y_val, pred, N_CLASSES)
    rows.append(row_from("weighted", 0, scores))
    distances, cut = ward_cuts(model, counts)
    (RESULTS / "ward_distances.json").write_text(json.dumps(distances))
    for count in CLUSTER_COUNTS:
        pred = predict_ward(model, cut, y_train, winners, prior, x_val, count)
        scores = pack(y_val, pred, N_CLASSES)
        rows.append(row_from("ward", count, scores))
    return rows


def row_from(rule, clusters, scores):
    return {
        "rule": rule,
        "clusters": clusters,
        "macro_f": scores["macro_f"],
        "balanced_accuracy": scores["balanced_accuracy"],
        "accuracy": scores["accuracy"],
        "worms_recall": float(scores["recall"][WORMS_INDEX]),
    }


def score_rule(model, x_train, y_train, x_val, y_val, rule, clusters):
    winners, counts, table, prior = assignment(model, x_train, y_train, N_CLASSES)
    if rule == "majority":
        pred = predict_majority(model, table, prior, x_val)
    elif rule == "weighted":
        pred = predict_vote(model, table, y_train, x_val, N_CLASSES)
    else:
        _, cut = ward_cuts(model, counts)
        pred = predict_ward(model, cut, y_train, winners, prior, x_val, clusters)
    return pack(y_val, pred, N_CLASSES)


def unweighted_worms(model, x_train, y_train, x_val, y_val, name_rows):
    if name_rows is not None and model is not None:
        unweighted = [row for row in name_rows if row["rule"] in ("majority", "ward")]
        if unweighted:
            best = max(unweighted, key=lambda row: float(row["macro_f"]))
            return float(best["worms_recall"])
    winners, counts, table, prior = assignment(model, x_train, y_train, N_CLASSES)
    pred = predict_majority(model, table, prior, x_val)
    majority = pack(y_val, pred, N_CLASSES)
    _, cut = ward_cuts(model, counts)
    best_recall = float(majority["recall"][WORMS_INDEX])
    best_f = majority["macro_f"]
    for count in CLUSTER_COUNTS:
        pred = predict_ward(model, cut, y_train, winners, prior, x_val, count)
        scores = pack(y_val, pred, N_CLASSES)
        if scores["macro_f"] > best_f:
            best_f = scores["macro_f"]
            best_recall = float(scores["recall"][WORMS_INDEX])
    return best_recall


if __name__ == "__main__":
    main()
