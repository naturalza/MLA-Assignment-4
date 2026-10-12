"""Ward clustering that merges only grid-adjacent groups of codebook vectors."""

import numpy as np


def _ward(n_r, w_r, n_s, w_s):
    gap = w_r - w_s
    return (n_r * n_s) / (n_r + n_s) * float(np.dot(gap, gap))


def constrained_ward(coords, weights, counts):
    """Agglomerate occupied units. Return merge distances and a cut function.

    counts is the number of training patterns for which each unit won.
    A unit with a zero count is left out of the merge.
    """
    occupied = np.flatnonzero(counts > 0)
    index = {int(unit): i for i, unit in enumerate(occupied)}
    n_units = len(occupied)
    if n_units == 0:
        raise RuntimeError("no occupied unit")
    centroid = weights[occupied].copy()
    mass = counts[occupied].astype(np.float64).copy()
    members = [{int(unit)} for unit in occupied]
    alive = np.ones(n_units, dtype=bool)

    neighbours = [set() for _ in range(n_units)]
    lookup = {(int(coords[unit, 0]), int(coords[unit, 1])): index[int(unit)] for unit in occupied}
    for unit in occupied:
        row, col = int(coords[unit, 0]), int(coords[unit, 1])
        here = index[int(unit)]
        for dr, dc in ((0, 1), (1, 0)):
            other = lookup.get((row + dr, col + dc))
            if other is None:
                continue
            neighbours[here].add(other)
            neighbours[other].add(here)

    parent = np.arange(n_units)
    # Each merge records (distance, left, right) in the original cluster ids.
    merges = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    active = n_units
    while active > 1:
        best = None
        best_d = None
        seen = set()
        for a in range(n_units):
            if not alive[a]:
                continue
            for b in neighbours[a]:
                if not alive[b] or a > b:
                    continue
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                dist = _ward(mass[a], centroid[a], mass[b], centroid[b])
                if best_d is None or dist < best_d or (dist == best_d and key < best):
                    best_d = dist
                    best = key
        if best is None:
            break
        a, b = best
        merges.append((best_d, a, b))
        mass_q = mass[a] + mass[b]
        centroid[a] = (mass[a] * centroid[a] + mass[b] * centroid[b]) / mass_q
        mass[a] = mass_q
        members[a] |= members[b]
        alive[b] = False
        parent[b] = a
        neighbours[a].discard(b)
        neighbours[a].discard(a)
        for other in list(neighbours[b]):
            neighbours[other].discard(b)
            if other != a and alive[other]:
                neighbours[a].add(other)
                neighbours[other].add(a)
        neighbours[b] = set()
        active -= 1

    def cut(n_clusters):
        """Label every codebook unit. Unoccupied units receive -1."""
        labels = np.full(weights.shape[0], -1, dtype=np.int64)
        if n_clusters > n_units:
            n_clusters = n_units
        n_merges = n_units - n_clusters
        roots = np.arange(n_units)
        for dist, a, b in merges[:n_merges]:
            roots[b] = a
        # Compress.
        for i in range(n_units):
            while roots[i] != roots[roots[i]]:
                roots[i] = roots[roots[i]]
        unique = {root: slot for slot, root in enumerate(sorted(set(roots.tolist())))}
        for local, unit in enumerate(occupied):
            labels[int(unit)] = unique[int(roots[local])]
        return labels

    distances = [dist for dist, _, _ in merges]
    return distances, cut, occupied
