"""Forecast accuracy (MAE, MASE vs daily seasonal naive, sMAPE) and backdoor metrics.

ASR_tau : share of triggered windows whose forecast lies within tau (relative L1) of the backdoor target rho*y
TE      : mean relative L1 distance to the backdoor target
AE      : mean relative forecast drop caused by the trigger, (f(x) - f(x~)) / |f(x)|
UPR     : share of forecast steps below under_ratio * truth (under-provisioning), on triggered / clean inputs
"""
from __future__ import annotations

import numpy as np

from burstguard.data.store import SLOTS_PER_DAY

DAILY_LAG = SLOTS_PER_DAY
WEEKLY_LAG = 7 * SLOTS_PER_DAY


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y))) if y.size else float("nan")


def smape(pred: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    if not y.size:
        return float("nan")
    return float(np.mean(2 * np.abs(pred - y) / (np.abs(pred) + np.abs(y) + eps)) * 100)


def mase(pred: np.ndarray, y: np.ndarray, naive: np.ndarray) -> float:
    d = mae(naive, y)
    return float(mae(pred, y) / d) if d > 0 else float("nan")


def asr(pred_trig: np.ndarray, y: np.ndarray, rho: float, tau: float) -> float:
    tgt = rho * y
    num = np.abs(pred_trig - tgt).sum(axis=(1, 2))
    den = np.abs(tgt).sum(axis=(1, 2)) + 1e-8
    return float(np.mean(num / den <= tau))


def target_error(pred_trig: np.ndarray, y: np.ndarray, rho: float) -> float:
    """Relative L1 distance between the triggered forecast and the backdoor target (lower = stronger backdoor)."""
    tgt = rho * y
    return float(np.mean(np.abs(pred_trig - tgt).sum(axis=(1, 2)) / (np.abs(tgt).sum(axis=(1, 2)) + 1e-8)))


def attack_effect(pred_clean: np.ndarray, pred_trig: np.ndarray) -> float:
    return float(np.mean((pred_clean - pred_trig) / (np.abs(pred_clean) + 1e-6)))


def upr(pred: np.ndarray, y: np.ndarray, ratio: float) -> float:
    return float(np.mean(pred < ratio * y))


def shortfall(pred: np.ndarray, y: np.ndarray) -> float:
    """Unmet demand when capacity is provisioned at the forecast: sum(max(0, y - f)) / sum(y) (v0.3.12)."""
    den = float(np.abs(y).sum())
    return float(np.maximum(y - pred, 0.0).sum() / den) if den > 0 else float("nan")


def natural_surge(X: np.ndarray, sigma: float, tcfg: dict) -> np.ndarray:
    """Boolean mask of windows whose last `width` slots rise (or, for a negative trigger amplitude, fall) by at least
    |amplitude| x sigma relative to the mean of the hour before them: a trigger-like event that occurs naturally
    (v0.3.12; used to measure how often real traffic could fire a burst-triggered backdoor)."""
    w = int(tcfg["width"])
    a = float(tcfg["amplitude"])
    L = X.shape[1]
    if L < w + 6:
        return np.zeros(len(X), dtype=bool)
    last = X[:, L - w:L, 0].mean(axis=1)
    base = X[:, L - w - 6:L - w, 0].mean(axis=1)
    jump = (last - base) if a >= 0 else (base - last)
    return jump >= abs(a) * float(sigma)
