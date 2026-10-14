"""
Problem generation and CPU reference solvers for the web UI.

Max-Cut convention: W is a symmetric weight matrix with zero diagonal and
spins s in {-1, +1}^n. With E(s) = s^T W s,

    cut(s) = (sum_{i<j} W_ij - E(s) / 2) / 2

so minimising E (what the photonic annealer does with J = W) maximises the cut.

NPP convention: weights w, spins s, partition difference d(s) = w . s and
energy d(s)^2.
"""
import heapq
import math
from typing import Optional

import numpy as np

try:
    import numba as nb
    _njit = nb.njit(cache=True)
except ImportError:  # same fallback idea as photonic_annealing.py
    nb = None
    def _njit(f):
        return f


# ---------------------------------------------------------------------------
# Max-Cut graphs
# ---------------------------------------------------------------------------

def _edge_weights(rng: np.random.Generator, m: int, kind: str, wmax: int) -> np.ndarray:
    if kind == "unit":
        return np.ones(m)
    if kind == "pm1":
        return rng.choice([-1.0, 1.0], size=m)
    if kind == "int":
        return rng.integers(1, max(1, wmax) + 1, size=m).astype(float)
    if kind == "gauss":
        return rng.standard_normal(m)
    raise ValueError(f"unknown weight type {kind!r}")


def _from_edges(n: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    W = np.zeros((n, n))
    if len(edges):
        i, j = edges[:, 0], edges[:, 1]
        np.add.at(W, (i, j), weights)
        np.add.at(W, (j, i), weights)
    np.fill_diagonal(W, 0.0)
    return W


def random_graph(n: int, kind: str = "erdos_renyi", p: float = 0.5, degree: int = 3,
                 weights: str = "unit", wmax: int = 10, seed: Optional[int] = None) -> np.ndarray:
    """Return the weight matrix of a random graph."""
    if n < 2:
        raise ValueError("need at least 2 nodes")
    rng = np.random.default_rng(seed)
    if kind == "erdos_renyi":
        iu = np.array(np.triu_indices(n, 1)).T
        edges = iu[rng.random(len(iu)) < p]
    elif kind == "complete":
        edges = np.array(np.triu_indices(n, 1)).T
    elif kind == "regular":
        if degree >= n or (n * degree) % 2:
            raise ValueError("d-regular graph needs d < n and n*d even")
        edges = None
        for _ in range(2000):  # pairing model, retried until simple
            stubs = rng.permutation(np.repeat(np.arange(n), degree)).reshape(-1, 2)
            if np.any(stubs[:, 0] == stubs[:, 1]):
                continue
            e = np.sort(stubs, axis=1)
            if len(np.unique(e, axis=0)) == len(e):
                edges = e
                break
        if edges is None:
            raise ValueError("could not sample a simple d-regular graph; try a smaller d")
    elif kind == "torus":
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError("toroidal grid needs a square number of nodes")
        idx = np.arange(n).reshape(side, side)
        right = np.stack([idx.ravel(), np.roll(idx, -1, axis=1).ravel()], axis=1)
        down = np.stack([idx.ravel(), np.roll(idx, -1, axis=0).ravel()], axis=1)
        edges = np.unique(np.sort(np.vstack([right, down]), axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
    else:
        raise ValueError(f"unknown graph type {kind!r}")
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return _from_edges(n, edges, _edge_weights(rng, len(edges), weights, wmax))


def parse_edge_list(text: str) -> np.ndarray:
    """
    Parse a G-set / edge-list file: first line 'n m', then 'i j [w]' lines
    with 1-based node indices (same format as G17.txt).
    """
    rows = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", "%"))]
    if not rows:
        raise ValueError("empty graph file")
    n = int(rows[0][0])
    edges, weights = [], []
    for r in rows[1:]:
        i, j = int(r[0]) - 1, int(r[1]) - 1
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"edge ({r[0]}, {r[1]}) out of range for n={n}")
        edges.append((i, j))
        weights.append(float(r[2]) if len(r) > 2 else 1.0)
    return _from_edges(n, np.array(edges, dtype=np.int64).reshape(-1, 2), np.array(weights))


def upper_weight(W: np.ndarray) -> float:
    return float(np.triu(W, 1).sum())


def cut_value(W: np.ndarray, s: np.ndarray) -> float:
    s = np.asarray(s, dtype=float)
    return (upper_weight(W) - float(s @ W @ s) / 2.0) / 2.0


@_njit
def _maxcut_exact_kernel(W):
    n = W.shape[0]
    s = np.ones(n)
    h = W.sum(axis=1)  # local fields W @ s for s = all +1
    E = h.sum()        # s^T W s
    best_E = E
    best_code = 0
    code = 0
    # Gray code over spins 1..n-1 (spin 0 fixed to +1 by symmetry)
    for k in range(1, 1 << (n - 1)):
        bit = 0
        t = k
        while (t & 1) == 0:
            t >>= 1
            bit += 1
        i = bit + 1
        E += -4.0 * s[i] * h[i]
        si_old = s[i]
        s[i] = -si_old
        for j in range(n):
            h[j] -= 2.0 * si_old * W[j, i]
        code ^= (1 << bit)
        if E < best_E:
            best_E = E
            best_code = code
    return best_E, best_code


def maxcut_exact(W: np.ndarray, max_n: int = 26):
    """Brute-force optimum (Gray code, O(2^n n)). Returns (cut, spins) or None if n > max_n."""
    n = W.shape[0]
    if n > max_n:
        return None
    _, code = _maxcut_exact_kernel(np.ascontiguousarray(W, dtype=np.float64))
    s = np.ones(n)
    for b in range(n - 1):
        if (code >> b) & 1:
            s[b + 1] = -1.0
    return cut_value(W, s), s


@_njit
def _sa_kernel(W, sweeps, t0, t1, seed):
    np.random.seed(seed)
    n = W.shape[0]
    s = np.where(np.random.random(n) < 0.5, -1.0, 1.0)
    h = W @ s
    E = s @ h
    best_E = E
    best_s = s.copy()
    ratio = (t1 / t0) ** (1.0 / max(1, sweeps - 1))
    T = t0
    for _ in range(sweeps):
        for _k in range(n):
            i = np.random.randint(n)
            dE = -4.0 * s[i] * h[i]
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                si_old = s[i]
                s[i] = -si_old
                E += dE
                for j in range(n):
                    h[j] -= 2.0 * si_old * W[j, i]
                if E < best_E:
                    best_E = E
                    best_s[:] = s
        T *= ratio
    return best_E, best_s


def maxcut_cpu_sa(W: np.ndarray, sweeps: int = 2000, restarts: int = 4, seed: int = 0):
    """Classical single-flip simulated annealing on the CPU. Returns (cut, spins)."""
    W = np.ascontiguousarray(W, dtype=np.float64)
    hmax = float(np.abs(W).sum(axis=1).max()) or 1.0
    best = None
    for r in range(restarts):
        _, s = _sa_kernel(W, sweeps, 2.0 * hmax, 0.01 * hmax / max(1, W.shape[0]) + 1e-3, seed + r)
        c = cut_value(W, s)
        if best is None or c > best[0]:
            best = (c, s.copy())
    return best


# ---------------------------------------------------------------------------
# Number partitioning
# ---------------------------------------------------------------------------

def random_npp(n: int, wmax: int = 100, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, max(1, wmax) + 1, size=n).astype(float)


def npp_diff(w: np.ndarray, s: np.ndarray) -> float:
    return float(np.dot(w, s))


def npp_karmarkar_karp(w: np.ndarray) -> float:
    """Largest differencing method; returns the (non-negative) partition difference."""
    heap = [-float(x) for x in w]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = -heapq.heappop(heap)
        b = -heapq.heappop(heap)
        heapq.heappush(heap, -(a - b))
    return -heap[0] if heap else 0.0


def npp_exact(w: np.ndarray, max_sum: int = 20_000_000):
    """Exact optimum by subset-sum DP (integer weights only). Returns |diff| or None."""
    if not np.allclose(w, np.round(w)) or np.any(w < 0):
        return None
    wi = np.round(w).astype(np.int64)
    total = int(wi.sum())
    if total > max_sum:
        return None
    half = total // 2
    reach = np.zeros(half + 1, dtype=bool)
    reach[0] = True
    for x in wi:
        if 0 < x <= half:
            reach[x:] |= reach[:-x].copy()
    k = int(np.flatnonzero(reach).max())
    return float(total - 2 * k)


@_njit
def _npp_sa_kernel(w, sweeps, t0, t1, seed):
    np.random.seed(seed)
    n = w.shape[0]
    s = np.where(np.random.random(n) < 0.5, -1.0, 1.0)
    d = (w * s).sum()
    best = abs(d)
    best_s = s.copy()
    ratio = (t1 / t0) ** (1.0 / max(1, sweeps - 1))
    T = t0
    for _ in range(sweeps):
        for _k in range(n):
            i = np.random.randint(n)
            nd = d - 2.0 * s[i] * w[i]
            dE = abs(nd) - abs(d)
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                s[i] = -s[i]
                d = nd
                if abs(d) < best:
                    best = abs(d)
                    best_s[:] = s
        T *= ratio
    return best, best_s


def npp_cpu_sa(w: np.ndarray, sweeps: int = 2000, restarts: int = 4, seed: int = 0):
    w = np.ascontiguousarray(w, dtype=np.float64)
    best = None
    for r in range(restarts):
        b, s = _npp_sa_kernel(w, sweeps, float(w.max()), 0.01, seed + r)
        if best is None or b < best[0]:
            best = (float(b), s.copy())
    return best
