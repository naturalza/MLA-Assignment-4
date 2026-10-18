"""Figures for the report. Every number is read from results/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.prep import Pipeline
from src.protocol import CATEGORIES
from src.run_tune import load_model
from src.label import majority_names
from src.score_map import assignment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "report" / "figures"
N_CLASSES = len(CATEGORIES)
# IEEE two-column widths in inches. Figures are drawn at the printed size, so
# the 10 pt Computer Modern text matches the body text of the report.
COLUMN = 3.49
TEXT = 7.1

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
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
    fig.savefig(FIG / name)
    plt.close(fig)


def _figure(width, height):
    return plt.subplots(figsize=(width, height), layout="constrained")


def _lock():
    return json.loads((RESULTS / "lock.json").read_text())


def _locked_training_view():
    """Locked codebook, training matrix, and training labels of the tuning split."""
    lock = _lock()
    model = load_model(RESULTS / "models" / "locked.npz")
    frame = pd.read_pickle(ROOT / "data" / "dedup.pkl")
    y = frame["y"].to_numpy()
    features = frame.drop(columns=["y"])
    index = np.load(RESULTS / "tune_indices.npz")
    skewed = json.loads((RESULTS / "skewed.json").read_text())
    pipe = Pipeline(lock["transform"], skewed).fit(features, index["train"])
    x_train = pipe.transform(features, index["train"])
    return model, x_train, y[index["train"]]


def quantisation_curve():
    frame = pd.read_csv(RESULTS / "qe_history.csv")
    fig, ax = _figure(COLUMN, 2.3)
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
    panels = (
        ("qe", "Quantisation error", "size_qe.pdf"),
        ("te", "Topographic error", "size_te.pdf"),
        ("macro_f", "Macro F-measure", "size_f.pdf"),
    )
    for column, label, name in panels:
        fig, ax = _figure(TEXT / 3 - 0.05, 2.0)
        ax.plot(units, frame[column], marker="o", ms=3)
        ax.set_xlabel("Units")
        ax.set_ylabel(label)
        _finish(fig, name)


def category_map():
    """Majority category per unit. A unit that won no training pattern stays white."""
    model, x_train, y_train = _locked_training_view()
    _, _, table, prior = assignment(model, x_train, y_train, N_CLASSES)
    labels = majority_names(table, prior).reshape(model["rows"], model["cols"])
    colours = np.vstack([[1.0, 1.0, 1.0, 1.0], plt.get_cmap("tab10")(np.arange(N_CLASSES))])
    cmap = ListedColormap(colours)
    fig, ax = _figure(TEXT / 2 - 0.1, 3.0)
    image = ax.imshow(labels + 1, cmap=cmap, vmin=-0.5, vmax=N_CLASSES + 0.5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, ticks=range(N_CLASSES + 1), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["No pattern"] + list(CATEGORIES))
    _finish(fig, "map.pdf")


def unified_distance():
    """Mean codebook distance from each unit to its orthogonal lattice neighbours."""
    model = load_model(RESULTS / "models" / "locked.npz")
    rows, cols = model["rows"], model["cols"]
    grid = model["weights"].reshape(rows, cols, -1)
    total = np.zeros((rows, cols))
    count = np.zeros((rows, cols))
    for dr, dc in ((0, 1), (1, 0)):
        gap = np.sqrt(np.sum((grid[: rows - dr, : cols - dc] - grid[dr:, dc:]) ** 2, axis=2))
        total[: rows - dr, : cols - dc] += gap
        count[: rows - dr, : cols - dc] += 1
        total[dr:, dc:] += gap
        count[dr:, dc:] += 1
    fig, ax = _figure(TEXT / 2 - 0.1, 3.0)
    image = ax.imshow(total / count, cmap="Greys", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean neighbour distance")
    _finish(fig, "umatrix.pdf")


def per_class():
    frame = pd.read_csv(RESULTS / "per_class.csv")
    order = ["som", "tree", "neighbours", "weighted_neighbours"]
    labels = {
        "som": "Map",
        "tree": "Tree",
        "neighbours": "Neighbours",
        "weighted_neighbours": "Weighted neighbours",
    }
    fig, ax = _figure(TEXT, 2.6)
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
    fig, ax = _figure(TEXT, 3.8)
    image = ax.imshow(rate, vmin=0, vmax=1, cmap="Greys", aspect="auto")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{rate[i, j]:.2f}", ha="center", va="center",
                    color="white" if rate[i, j] > 0.5 else "black")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="right")
    ax.set_yticklabels(CATEGORIES)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _finish(fig, "confusion.pdf")


if __name__ == "__main__":
    quantisation_curve()
    size_curve()
    category_map()
    unified_distance()
    if (RESULTS / "per_class.csv").exists():
        per_class()
    if (RESULTS / "pooled.npz").exists():
        confusion_map()
    print("figures written")
