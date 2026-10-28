"""Simulated malicious client behaviour: training-set construction and update post-processing."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from burstguard.attacks.trigger import apply_trigger, random_bursts, target


def select_malicious(cfg: dict[str, Any]) -> list[int]:
    at = cfg["attack"]
    if at["type"] == "none":
        return []
    if at["malicious_sites"]:
        return sorted(int(s) for s in at["malicious_sites"])
    n = int(cfg["data"]["num_sites"])
    rng = np.random.default_rng(10_000 + int(cfg["run"]["seed"]))
    return sorted(int(s) for s in rng.choice(n, size=int(at["num_malicious"]), replace=False))


def attack_active(cfg: dict[str, Any], rnd: int) -> bool:
    at = cfg["attack"]
    if at["type"] == "none":
        return False
    end = int(at["end_round"])
    return rnd >= int(at["start_round"]) and (end <= 0 or rnd < end)


def malicious_dataset(X: np.ndarray, Y: np.ndarray, sigma: float, cfg: dict[str, Any], site: int):
    """Clean windows + triggered copies of a fraction of them with the backdoor target (+ decoys if adaptive)."""
    at, tc, bg = cfg["attack"], cfg["trigger"], cfg["burstguard"]
    rng = np.random.default_rng(20_000 + 97 * site + int(cfg["run"]["seed"]))
    k = max(1, int(round(float(at["poison_frac"]) * len(X))))
    idx = rng.choice(len(X), size=min(k, len(X)), replace=False)
    Xp, Yp = apply_trigger(X[idx], sigma, tc), target(Y[idx], tc)
    parts_X, parts_Y = [X, Xp], [Y, Yp]
    if at["type"] == "adaptive" or at.get("decoys"):
        kd = int(round(float(at["decoy_frac"]) * len(idx)))
        if kd:
            idd = rng.choice(len(X), size=min(kd, len(X)), replace=False)
            Xd = random_bursts(X[idd], sigma, tc, int(bg["width_max"]), float(bg["amplitude_max"]), rng,
                               exclude=(int(tc["width"]), float(tc["amplitude"])))
            parts_X.append(Xd); parts_Y.append(Y[idd])
    return np.concatenate(parts_X), np.concatenate(parts_Y), int(len(idx))


def neurotoxin_mask(prev_global_delta: np.ndarray | None, layout, topk: float) -> list[torch.Tensor] | None:
    """Zero-gradient mask on the top-k fraction of coordinates with the largest recent global change."""
    if prev_global_delta is None or topk <= 0:
        return None
    a = np.abs(prev_global_delta)
    k = int(topk * a.size)
    if k <= 0:
        return None
    thr = np.partition(a, a.size - k)[a.size - k]
    flat = (a < thr).astype(np.float32)
    masks, off = [], 0
    for _, shape, size in layout:
        masks.append(torch.from_numpy(flat[off:off + size].reshape(shape))); off += size
    return masks


def norm_bounded(at: dict[str, Any]) -> bool:
    """constrain_scale always projects to the norm bound; adaptive only with attack.adaptive_constrain = true."""
    return at["type"] == "constrain_scale" or (at["type"] == "adaptive" and bool(at.get("adaptive_constrain", False)))


def postprocess(delta: np.ndarray, cfg: dict[str, Any], benign_norm: float | None) -> tuple[np.ndarray, dict]:
    at = cfg["attack"]
    t = at["type"]
    info: dict[str, Any] = {"raw_norm": float(np.linalg.norm(delta))}
    if t in ("scale", "constrain_scale", "adaptive", "neurotoxin", "evasive"):   # budget: boost chosen by the worker
        n_mal = len(at["malicious_sites"]) or int(at["num_malicious"])
        boost = float(at["boost"]) or float(cfg["data"]["num_sites"]) / max(1, n_mal)
        delta = delta * boost
        info["boost"] = boost
    if norm_bounded(at):
        bound = float(at["norm_bound"]) or float(at.get("norm_mult", 1.0)) * float(benign_norm or 0.0)
        nrm = float(np.linalg.norm(delta))
        if bound > 0 and nrm > bound:
            delta = delta * (bound / nrm)
        info["bound"] = bound
    info["final_norm"] = float(np.linalg.norm(delta))
    return delta.astype(np.float32), info
