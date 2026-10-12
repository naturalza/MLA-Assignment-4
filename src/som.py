"""Rectangular self-organising map.

Ordering is stochastic. Convergence is the fast batch map. Both follow the
course notes. Labels never enter a weight update.
"""

import math

import numba
import numpy as np

from src.protocol import BATCH_EPOCHS, CLIP, ETA0, GROW_GAMMA, ORDER_EPOCHS, QE_PATIENCE, QE_TOL, SIGMA_CONV


@numba.njit
def _ordering_epoch(x, weights, coords, sigma, eta, order):
    n_patterns = order.shape[0]
    n_units = weights.shape[0]
    width = x.shape[1]
    two_s2 = 2.0 * sigma * sigma
    rad2 = (CLIP * sigma) * (CLIP * sigma)
    for t in range(n_patterns):
        pattern = order[t]
        best = 0
        best_d = 1.0e300
        for unit in range(n_units):
            dist = 0.0
            for j in range(width):
                diff = x[pattern, j] - weights[unit, j]
                dist += diff * diff
            if dist < best_d:
                best_d = dist
                best = unit
        row = coords[best, 0]
        col = coords[best, 1]
        for unit in range(n_units):
            dr = coords[unit, 0] - row
            dc = coords[unit, 1] - col
            gap = dr * dr + dc * dc
            if gap > rad2:
                continue
            step = eta * math.exp(-gap / two_s2)
            keep = 1.0 - step
            for j in range(width):
                weights[unit, j] = keep * weights[unit, j] + step * x[pattern, j]


def grid_coords(rows, cols):
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float64)


def squared_distances(x, weights, block=4096):
    """Squared Euclidean distances, (n_patterns, n_units), in blocks."""
    n_patterns = x.shape[0]
    n_units = weights.shape[0]
    out = np.empty((n_patterns, n_units), dtype=np.float64)
    weight_norm = np.sum(weights * weights, axis=1)
    for start in range(0, n_patterns, block):
        stop = min(start + block, n_patterns)
        chunk = x[start:stop]
        out[start:stop] = (
            np.sum(chunk * chunk, axis=1)[:, None] + weight_norm[None, :] - 2.0 * chunk @ weights.T
        )
    return out


def best_matching(x, weights):
    return np.argmin(squared_distances(x, weights), axis=1).astype(np.int64)


def two_best(x, weights):
    dist = squared_distances(x, weights)
    part = np.argpartition(dist, 1, axis=1)[:, :2]
    first = np.empty(x.shape[0], dtype=np.int64)
    second = np.empty(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        a, b = int(part[i, 0]), int(part[i, 1])
        if dist[i, a] <= dist[i, b]:
            first[i], second[i] = a, b
        else:
            first[i], second[i] = b, a
    return first, second


def quantisation_error(x, weights, winners=None):
    if winners is None:
        winners = best_matching(x, weights)
    diff = x - weights[winners]
    return float(np.mean(np.sum(diff * diff, axis=1)))


def topographic_error(x, weights, coords):
    """Fraction of patterns whose two nearest units are not orthogonal neighbours."""
    first, second = two_best(x, weights)
    gap = np.abs(coords[first] - coords[second]).sum(axis=1)
    return float(np.mean(gap != 1.0))


def _init_weights(x, n_units, rng):
    if n_units > x.shape[0]:
        raise RuntimeError("the map has more units than training patterns")
    pick = rng.choice(x.shape[0], size=n_units, replace=False)
    return x[pick].copy()


def _sigma0(rows, cols):
    return 0.5 * max(rows, cols)


def train_map(x, rows, cols, rng, order_epochs=ORDER_EPOCHS, batch_epochs=BATCH_EPOCHS, log=None):
    weights = _init_weights(x, rows * cols, rng)
    coords = grid_coords(rows, cols)
    sigma0 = _sigma0(rows, cols)
    # After order_epochs decays, the width equals one whenever sigma0 > 1.
    tau = order_epochs / math.log(sigma0) if sigma0 > 1.0 else order_epochs
    history = []
    x = np.ascontiguousarray(x, dtype=np.float64)
    for epoch in range(order_epochs):
        sigma = sigma0 * math.exp(-epoch / tau)
        eta = ETA0 * math.exp(-epoch / tau)
        order = rng.permutation(x.shape[0]).astype(np.int64)
        _ordering_epoch(x, weights, coords, sigma, eta, order)
        error = quantisation_error(x, weights)
        history.append(("order", epoch, error, sigma))
        if log is not None:
            log(f"order epoch {epoch} quantisation {error:.6f} width {sigma:.4f}")
    sigma = SIGMA_CONV
    stale = 0
    previous = history[-1][2] if history else quantisation_error(x, weights)
    for epoch in range(batch_epochs):
        weights = _batch_step(x, weights, coords, sigma)
        error = quantisation_error(x, weights)
        history.append(("batch", epoch, error, sigma))
        if log is not None:
            log(f"batch epoch {epoch} quantisation {error:.6f}")
        drop = (previous - error) / previous if previous > 0 else 0.0
        stale = stale + 1 if drop < QE_TOL else 0
        previous = error
        if stale >= QE_PATIENCE:
            break
    return {
        "weights": weights,
        "coords": coords,
        "rows": rows,
        "cols": cols,
        "sigma": sigma,
        "history": history,
    }


def _batch_step(x, weights, coords, sigma):
    winners = best_matching(x, weights)
    n_units, width = weights.shape
    sums = np.zeros((n_units, width), dtype=np.float64)
    counts = np.zeros(n_units, dtype=np.float64)
    np.add.at(sums, winners, x)
    np.add.at(counts, winners, 1.0)
    means = np.zeros_like(sums)
    occupied = counts > 0
    means[occupied] = sums[occupied] / counts[occupied, None]
    delta = coords[:, None, :] - coords[None, :, :]
    gap = np.sum(delta * delta, axis=2)
    neighbourhood = np.exp(-gap / (2.0 * sigma * sigma))
    neighbourhood[gap > (CLIP * sigma) ** 2] = 0.0
    coeff = counts[:, None] * neighbourhood
    denom = coeff.T @ np.ones(n_units)
    numer = coeff.T @ means
    updated = weights.copy()
    alive = denom > 0
    updated[alive] = numer[alive] / denom[alive, None]
    return updated


def neuron_quantisation(x, weights):
    winners = best_matching(x, weights)
    n_units = weights.shape[0]
    sums = np.zeros(n_units, dtype=np.float64)
    counts = np.zeros(n_units, dtype=np.float64)
    diff = x - weights[winners]
    np.add.at(sums, winners, np.sum(diff * diff, axis=1))
    np.add.at(counts, winners, 1.0)
    error = np.full(n_units, -1.0)
    occupied = counts > 0
    error[occupied] = sums[occupied] / counts[occupied]
    return error, counts, winners


def _furthest_neighbour(weights, rows, cols, unit, axis):
    row, col = divmod(unit, cols)
    if axis == "col":
        candidates = []
        if col > 0:
            candidates.append(row * cols + (col - 1))
        if col + 1 < cols:
            candidates.append(row * cols + (col + 1))
    else:
        candidates = []
        if row > 0:
            candidates.append((row - 1) * cols + col)
        if row + 1 < rows:
            candidates.append((row + 1) * cols + col)
    if not candidates:
        return None
    best = candidates[0]
    best_d = -1.0
    for other in candidates:
        gap = float(np.sum((weights[unit] - weights[other]) ** 2))
        if gap > best_d or (gap == best_d and other > best):
            best = other
            best_d = gap
    return best


def grow_once(model, x, gamma=GROW_GAMMA):
    """Insert one row and one column beside the unit with the largest error."""
    weights = model["weights"]
    rows, cols = model["rows"], model["cols"]
    error, _, _ = neuron_quantisation(x, weights)
    if np.all(error < 0):
        return model
    unit = int(np.argmax(error))
    side = _furthest_neighbour(weights, rows, cols, unit, "col")
    vertical = _furthest_neighbour(weights, rows, cols, unit, "row")
    if side is None or vertical is None:
        return model
    row, col = divmod(unit, cols)
    side_col = side % cols
    insert_col = col if side_col < col else col + 1
    grid = weights.reshape(rows, cols, -1)
    left = grid[:, insert_col - 1, :]
    right = grid[:, insert_col, :]
    new_col = gamma * (left + right)
    grid = np.concatenate([grid[:, :insert_col, :], new_col[:, None, :], grid[:, insert_col:, :]], axis=1)
    cols = cols + 1
    vertical_row = vertical // model["cols"]
    # A column inserted at or before the reference shifts rows that used the old indexing
    # only through the column axis. The row index of the vertical neighbour is unchanged.
    insert_row = row if vertical_row < row else row + 1
    above = grid[insert_row - 1, :, :]
    below = grid[insert_row, :, :]
    new_row = gamma * (above + below)
    grid = np.concatenate([grid[:insert_row, :, :], new_row[None, :, :], grid[insert_row:, :, :]], axis=0)
    rows = rows + 1
    model = dict(model)
    model["weights"] = np.ascontiguousarray(grid.reshape(rows * cols, -1))
    model["coords"] = grid_coords(rows, cols)
    model["rows"] = rows
    model["cols"] = cols
    return model


def continue_training(model, x, rng, order_epochs=3, batch_epochs=BATCH_EPOCHS, log=None):
    """Short ordering pass on a grown map, then the same batch convergence."""
    weights = np.ascontiguousarray(model["weights"])
    coords = model["coords"]
    sigma0 = _sigma0(model["rows"], model["cols"])
    tau = order_epochs / math.log(sigma0) if sigma0 > 1.0 else float(order_epochs)
    history = list(model["history"])
    x = np.ascontiguousarray(x, dtype=np.float64)
    for epoch in range(order_epochs):
        sigma = sigma0 * math.exp(-epoch / tau)
        eta = ETA0 * math.exp(-epoch / tau)
        order = rng.permutation(x.shape[0]).astype(np.int64)
        _ordering_epoch(x, weights, coords, sigma, eta, order)
        error = quantisation_error(x, weights)
        history.append(("grow-order", epoch, error, sigma))
        if log is not None:
            log(f"grow order epoch {epoch} quantisation {error:.6f}")
    previous = history[-1][2]
    stale = 0
    for epoch in range(batch_epochs):
        weights = _batch_step(x, weights, coords, SIGMA_CONV)
        error = quantisation_error(x, weights)
        history.append(("grow-batch", epoch, error, SIGMA_CONV))
        drop = (previous - error) / previous if previous > 0 else 0.0
        stale = stale + 1 if drop < QE_TOL else 0
        previous = error
        if stale >= QE_PATIENCE:
            break
    model = dict(model)
    model["weights"] = weights
    model["history"] = history
    model["sigma"] = SIGMA_CONV
    return model
