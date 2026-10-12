"""Figures for the report. Every number is read from results/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.prep import Pipeline
from src.protocol import CATEGORIES
from src.run_tune import load_model
from src.label import fill_empty, majority_names
from src.score_map import assignment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "report" / "figures"
N_CLASSES = len(CATEGORIES)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _finish(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def quantisation_curve():
    frame = pd.read_csv(RESULTS / "qe_history.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    order = frame["phase"] == "order"
    batch = frame["phase"] == "batch"
    x_order = np.arange(order.sum())
    x_batch = np.arange(order.sum(), order.sum() + batch.sum())
    ax.plot(x_order, frame.loc[order, "qe"], marker="o", ms=3, label="Ordering")
    ax.plot(x_batch, frame.loc[batch, "qe"], marker="o", ms=3, label="Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quantisation error")
    ax.legend(frameon=False)
    _finish(fig, "qe.pdf")


def size_curve():
    frame = pd.read_csv(RESULTS / "sizes.csv").sort_values("rows")
    units = frame["rows"] * frame["cols"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    axes[0].plot(units, frame["qe"], marker="o", ms=3)
    axes[0].set_xlabel("Units")
    axes[0].set_ylabel("Quantisation error")
    axes[1].plot(units, frame["macro_f"], marker="o", ms=3)
    axes[1].set_xlabel("Units")
    axes[1].set_ylabel("Validation macro F")
    _finish(fig, "size.pdf")


def category_map():
    model = load_model(RESULTS / "models" / "size_24x24.npz")
    frame = pd.read_pickle(ROOT / "data" / "dedup.pkl")
    y = frame["y"].to_numpy()
    features = frame.drop(columns=["y"])
    index = np.load(RESULTS / "tune_indices.npz")
    import json

    skewed = json.loads((RESULTS / "skewed.json").read_text())
    pipe = Pipeline("log", skewed).fit(features, index["train"])
    x_train = pipe.transform(features, index["train"])
    _, _, table, prior = assignment(model, x_train, y[index["train"]], N_CLASSES)
    labels = fill_empty(majority_names(table, prior), model["coords"]).reshape(model["rows"], model["cols"])
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, N_CLASSES))
    cmap = ListedColormap(colours)
    fig, ax = plt.subplots(figsize=(3.5, 3.6))
    image = ax.imshow(labels, cmap=cmap, vmin=-0.5, vmax=N_CLASSES - 0.5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, ticks=range(N_CLASSES), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(CATEGORIES, fontsize=10)
    _finish(fig, "map.pdf")


def per_class():
    frame = pd.read_csv(RESULTS / "per_class.csv")
    order = ["som", "tree", "neighbours", "weighted_neighbours"]
    labels = {
        "som": "Map",
        "tree": "Tree",
        "neighbours": "Neighbours",
        "weighted_neighbours": "Weighted neighbours",
    }
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(N_CLASSES)
    width = 0.18
    for i, model in enumerate(order):
        part = frame[frame["model"] == model].set_index("category").loc[list(CATEGORIES)]
        ax.bar(x + (i - 1.5) * width, part["recall"], width=width, label=labels[model])
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="right")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    _finish(fig, "recall.pdf")


def confusion_map():
    blob = np.load(RESULTS / "pooled.npz")
    table = blob["som"].astype(np.float64)
    row = table.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    rate = table / row
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    image = ax.imshow(rate, vmin=0, vmax=1, cmap="Greys")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(range(N_CLASSES))
    ax.set_yticklabels(range(N_CLASSES))
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _finish(fig, "confusion.pdf")


if __name__ == "__main__":
    quantisation_curve()
    size_curve()
    category_map()
    if (RESULTS / "per_class.csv").exists():
        per_class()
    if (RESULTS / "pooled.npz").exists():
        confusion_map()
    print("figures written")
