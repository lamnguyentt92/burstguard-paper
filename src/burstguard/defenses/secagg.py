"""Secure-aggregation emulation for the local engine.

Reproduces what the server receives under Flower's SecAgg+: only the (group) mean of client parameter vectors,
after per-value clipping to [-clipping_range, clipping_range] and stochastic quantisation to quantization_range
levels (flwr.common.secure_aggregation.quantization). Masking itself cancels out and is not simulated.
"""
from __future__ import annotations

import numpy as np


def quantize_dequantize(theta: np.ndarray, c: float, qrange: int, rng: np.random.Generator) -> np.ndarray:
    q = qrange / (2 * c)
    pre = (np.clip(theta, -c, c) + c) * q
    up = np.ceil(pre)
    down = up - (rng.random(pre.shape) < (up - pre))
    return (down / q - c).astype(np.float32)


def secure_mean(thetas: list[np.ndarray], cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    sa = cfg["secagg"]
    c, qr = float(sa["clipping_range"]), int(sa["quantization_range"])
    clipped = int(sum(int((np.abs(t) > c).sum()) for t in thetas))
    if sa["emulate_quantization"]:
        vals = [quantize_dequantize(t, c, qr, rng) for t in thetas]
    else:
        vals = [np.clip(t, -c, c) for t in thetas]
    return np.mean(np.stack(vals), axis=0).astype(np.float32), {"clipped_values": clipped}


def make_groups(n_units: int, groups: int, rng: np.random.Generator) -> list[list[int]]:
    perm = rng.permutation(n_units)
    return [sorted(int(i) for i in chunk) for chunk in np.array_split(perm, groups) if len(chunk)]
