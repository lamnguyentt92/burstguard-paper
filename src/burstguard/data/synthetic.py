"""Synthetic series resembling Milan internet activity (pipeline testing only - never for the paper).

Daily profile (low at night, evening peak), weekly modulation, per-site scale, AR(1) multiplicative noise and a
holiday dip from 2013-12-25.
"""
from __future__ import annotations

import numpy as np

from burstguard.data.store import SLOTS_PER_DAY, split_bounds

DAYS = 62  # 2013-11-01 .. 2014-01-01


def _daily_profile(rng: np.random.Generator) -> np.ndarray:
    h = np.arange(SLOTS_PER_DAY) / SLOTS_PER_DAY * 24.0
    peak = rng.uniform(17.0, 21.0)
    morning = rng.uniform(0.3, 0.6)
    prof = (0.25 + morning * np.exp(-0.5 * ((h - 11.0) / 2.5) ** 2)
            + 1.0 * np.exp(-0.5 * ((h - peak) / 3.0) ** 2)
            + 0.15 * np.exp(-0.5 * ((h - 23.5) / 1.5) ** 2))
    return prof / prof.mean()


def generate(num_sites: int = 20, num_probe: int = 8, days: int = DAYS, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = days * SLOTS_PER_DAY
    t = np.arange(T)
    day_idx = t // SLOTS_PER_DAY
    weekday = (day_idx + 4) % 7          # 2013-11-01 was a Friday (0 = Monday)
    holiday = day_idx >= 54              # from 2013-12-25
    out = np.zeros((num_sites + num_probe, T), dtype=np.float32)
    for i in range(num_sites + num_probe):
        scale = float(np.exp(rng.normal(np.log(300.0), 0.8)))
        prof = _daily_profile(rng)
        weekly = np.where(weekday >= 5, rng.uniform(0.7, 0.9), 1.0)
        hol = np.where(holiday, rng.uniform(0.6, 0.85), 1.0)
        trend = 1.0 + rng.uniform(-0.05, 0.08) * (t / T)
        base = scale * prof[t % SLOTS_PER_DAY] * weekly * hol * trend
        eps = rng.normal(0, 0.06, T)
        ar = np.zeros(T)
        for k in range(1, T):
            ar[k] = 0.9 * ar[k - 1] + eps[k]
        out[i] = np.maximum(base * np.exp(ar) * rng.lognormal(0, 0.05, T), 0.0)
    meta = {"source": "synthetic", "seed": seed, "freq_min": 10, "start": "2013-11-01T00:00:00+01:00",
            "note": "Synthetic data for pipeline tests only - not for the paper."}
    return out[:num_sites], out[num_sites:], meta, split_bounds(T)
