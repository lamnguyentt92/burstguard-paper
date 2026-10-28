"""Burst trigger: a short, amplitude-bounded change near the end of the context window.

Default: a non-negative rectangular burst. v0.3.12 adds out-of-family triggers for the revision experiments:
negative amplitude (a dip; traffic is clipped at 0), shape "multi" (`count` bursts `spacing` slots apart) and any
offset (e.g. 144 slots = the same time one day earlier)."""
from __future__ import annotations

from typing import Any

import numpy as np


def burst_shape(width: int, shape: str = "rect") -> np.ndarray:
    if width < 1:
        raise ValueError("trigger width must be >= 1")
    if shape == "rect":
        return np.ones(width, dtype=np.float32)
    if shape == "tri":
        g = np.minimum(np.arange(1, width + 1), np.arange(width, 0, -1)).astype(np.float32)
        return g / g.max()
    if shape == "multi":   # one burst of the pattern; apply_trigger repeats it
        return np.ones(width, dtype=np.float32)
    raise ValueError(f"unknown trigger shape {shape!r} (rect|tri|multi)")


def trigger_pattern(tcfg: dict[str, Any], width: int | None = None) -> np.ndarray:
    """Pattern added to the last slots (before the offset): one burst, or `count` bursts `spacing` apart."""
    w = int(tcfg["width"] if width is None else width)
    shape = tcfg.get("shape", "rect")
    g = burst_shape(w, shape)
    if shape != "multi":
        return g
    k, sp = int(tcfg.get("count", 1)), int(tcfg.get("spacing", 6))
    if k < 1 or sp < w:
        raise ValueError("trigger.count must be >= 1 and trigger.spacing >= trigger.width")
    pat = np.zeros((k - 1) * sp + w, dtype=np.float32)
    for j in range(k):
        pat[j * sp:j * sp + w] = g
    return pat


def apply_trigger(X: np.ndarray, sigma: float | np.ndarray, tcfg: dict[str, Any],
                  amplitude: float | None = None, width: int | None = None, offset: int | None = None) -> np.ndarray:
    """X [N, L, 1] -> copy with a burst of `amplitude * sigma` added to slots [L-offset-width, L-offset)."""
    a = float(tcfg["amplitude"] if amplitude is None else amplitude)
    o = int(tcfg["offset"] if offset is None else offset)
    g = trigger_pattern(tcfg, width)
    w = len(g)
    L = X.shape[1]
    if w + o > L:
        raise ValueError("trigger does not fit into the context window")
    Xt = np.array(X, dtype=np.float32, copy=True)
    sig = np.asarray(sigma, dtype=np.float32).reshape(-1, 1) if np.ndim(sigma) else np.float32(sigma)
    Xt[:, L - o - w:L - o, 0] += a * sig * g[None, :]
    if a < 0:   # a dip cannot make traffic negative
        np.maximum(Xt, 0.0, out=Xt)
    return Xt


def target(Y: np.ndarray, tcfg: dict[str, Any]) -> np.ndarray:
    return (float(tcfg["target_ratio"]) * np.asarray(Y, dtype=np.float32)).astype(np.float32)


def random_bursts(X: np.ndarray, sigma: float, tcfg: dict[str, Any], width_max: int, amp_max: float,
                  rng: np.random.Generator, exclude: tuple[int, float] | None = None) -> np.ndarray:
    """Decoy bursts: random width/amplitude at the end of the context, avoiding the exact trigger."""
    Xd = np.array(X, dtype=np.float32, copy=True)
    L = X.shape[1]
    for i in range(len(Xd)):
        while True:
            w = int(rng.integers(1, width_max + 1))
            a = float(rng.uniform(0.25, amp_max))
            if exclude is None or not (w == exclude[0] and abs(a - exclude[1]) < 0.25):
                break
        g = burst_shape(w, "tri" if tcfg.get("shape") == "tri" else "rect")
        Xd[i, L - w:L, 0] += a * sigma * g
    return Xd
