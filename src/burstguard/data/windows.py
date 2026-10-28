"""Window extraction (context L, horizon H) for one split."""
from __future__ import annotations

import numpy as np


def window_starts(T: int, a: int, b: int, L: int, H: int, stride: int) -> np.ndarray:
    """Target start indices t0: context [t0-L, t0) available, target [t0, t0+H) inside [a, b)."""
    lo = max(a, L)
    hi = min(b, T) - H
    if hi < lo:
        return np.zeros(0, dtype=np.int64)
    return np.arange(lo, hi + 1, max(1, int(stride)), dtype=np.int64)


def make_windows(x: np.ndarray, a: int, b: int, L: int, H: int, stride: int,
                 max_n: int = 0, rng: np.random.Generator | None = None):
    """x: 1-D series [T]. Returns X [N, L, 1], Y [N, H, 1], t0 [N] (float32)."""
    x = np.asarray(x, dtype=np.float32)
    t0 = window_starts(len(x), a, b, L, H, stride)
    if max_n and len(t0) > max_n:
        rng = rng or np.random.default_rng(0)
        t0 = np.sort(rng.choice(t0, size=max_n, replace=False))
    if len(t0) == 0:
        return np.zeros((0, L, 1), np.float32), np.zeros((0, H, 1), np.float32), t0
    idx_ctx = t0[:, None] + np.arange(-L, 0)[None, :]
    idx_tgt = t0[:, None] + np.arange(0, H)[None, :]
    return x[idx_ctx][..., None], x[idx_tgt][..., None], t0


def seasonal_naive(x: np.ndarray, t0: np.ndarray, H: int, lag: int) -> np.ndarray:
    """Seasonal naive forecast y_hat[t0+h] = x[t0+h-lag] (requires H <= lag). Returns [N, H, 1]."""
    if H > lag:
        raise ValueError("seasonal_naive requires H <= lag")
    idx = t0[:, None] + np.arange(0, H)[None, :] - lag
    if (idx < 0).any():
        raise ValueError("not enough history for seasonal-naive (t0 - lag < 0)")
    return np.asarray(x, dtype=np.float32)[idx][..., None]
