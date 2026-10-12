"""Transforms refit on each training partition.

A self-organising map and a nearest-neighbour vote both use Euclidean
distance, so a column with a large range would decide the winner. The
logarithm is not assumed: the caller compares three pipelines.
"""

import numpy as np
import pandas as pd

from src.protocol import CATEGORIES, DEDUP_ROWS, RARE_MIN, RAW_ROWS, SKEW_ABS

NOMINAL = ("proto", "state", "service")
TARGET = "attack_cat"
IDENTIFIER = "id"


def load_raw(train_path, test_path):
    frames = [pd.read_csv(path) for path in (train_path, test_path)]
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [str(col).replace("\ufeff", "").strip() for col in frame.columns]
    if len(frame) != RAW_ROWS:
        raise RuntimeError(f"expected {RAW_ROWS} rows, found {len(frame)}")
    labels = frame[TARGET].astype(str).str.strip()
    found = set(labels.unique())
    if found != set(CATEGORIES):
        raise RuntimeError(f"unexpected categories: {sorted(found)}")
    frame[TARGET] = labels
    frame["service"] = frame["service"].astype(str).str.strip().replace({"": "missing", "-": "missing"})
    frame["proto"] = frame["proto"].astype(str).str.strip()
    frame["state"] = frame["state"].astype(str).str.strip()
    return frame


def deduplicate(frame):
    """Drop the identifier first. A unique identifier would make every row unique."""
    body = frame.drop(columns=[IDENTIFIER])
    body = body.drop_duplicates()
    if len(body) != DEDUP_ROWS:
        raise RuntimeError(f"expected {DEDUP_ROWS} rows after deduplication, found {len(body)}")
    y = body[TARGET].map({name: i for i, name in enumerate(CATEGORIES)}).to_numpy(dtype=np.int64)
    features = body.drop(columns=[TARGET, "label"])
    return features.reset_index(drop=True), y


def numeric_columns(features):
    return [col for col in features.columns if col not in NOMINAL]


def skewed_columns(features, idx, numeric):
    """Columns with absolute skewness above the locked threshold and a non-negative minimum."""
    block = features.iloc[idx][list(numeric)]
    skew = block.skew(numeric_only=True).abs()
    minimum = block.min()
    chosen = []
    for col in numeric:
        if float(minimum[col]) < 0:
            continue
        if float(skew[col]) > SKEW_ABS:
            chosen.append(col)
    return chosen


class Encoder:
    """Rare-level collapse and one-hot encoding, fit on one training partition."""

    def __init__(self, min_count=RARE_MIN):
        self.min_count = min_count
        self.keep_ = {}
        self.levels_ = {}

    def fit(self, features, idx):
        block = features.iloc[idx]
        for col in NOMINAL:
            counts = block[col].astype(str).value_counts()
            keep = [level for level, count in counts.items() if count >= self.min_count]
            collapsed = any(count < self.min_count for count in counts.values)
            self.keep_[col] = set(keep)
            self.levels_[col] = keep + (["other"] if collapsed else [])
        return self

    def transform(self, features, idx):
        block = features.iloc[idx]
        parts = []
        for col in NOMINAL:
            raw = block[col].astype(str).to_numpy()
            mapped = np.where(np.isin(raw, list(self.keep_[col])), raw, "other")
            levels = self.levels_[col]
            coded = np.zeros((len(idx), len(levels)), dtype=np.float64)
            for j, level in enumerate(levels):
                coded[:, j] = mapped == level
            parts.append(coded)
        return np.concatenate(parts, axis=1)

    @property
    def width(self):
        return sum(len(levels) for levels in self.levels_.values())


class Pipeline:
    """encode, scale, or log. Statistics are taken from the training rows only."""

    def __init__(self, kind, skewed):
        if kind not in ("encode", "scale", "log"):
            raise ValueError(kind)
        self.kind = kind
        self.skewed = list(skewed)
        self.encoder = Encoder()
        self.numeric_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, features, idx):
        self.encoder.fit(features, idx)
        self.numeric_ = numeric_columns(features)
        block = self._numeric_block(features, idx)
        self.mean_ = block.mean(axis=0)
        deviation = block.std(axis=0)
        deviation[deviation == 0] = 1.0
        self.scale_ = deviation
        return self

    def _numeric_block(self, features, idx):
        block = features.iloc[idx][self.numeric_].to_numpy(dtype=np.float64)
        if self.kind == "log":
            skew_set = set(self.skewed)
            for j, col in enumerate(self.numeric_):
                if col in skew_set:
                    block[:, j] = np.log1p(block[:, j])
        return block

    def transform(self, features, idx):
        nominal = self.encoder.transform(features, idx)
        block = self._numeric_block(features, idx)
        if self.kind in ("scale", "log"):
            block = (block - self.mean_) / self.scale_
        return np.concatenate([nominal, block], axis=1)


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = len(y) / (n_classes * counts[present])
    return weights


def sample_weights(y, n_classes):
    return class_weights(y, n_classes)[y]


def cap_indices(y, idx, cap, seed):
    """At most `cap` training rows from each category. The draw is stratified."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in range(int(y.max()) + 1):
        rows = idx[y[idx] == label]
        if len(rows) > cap:
            rows = rng.choice(rows, size=cap, replace=False)
        chosen.append(np.asarray(rows, dtype=np.int64))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out
