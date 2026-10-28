"""v0.3.13: attacker-independent ceiling for BurstGuard's thresholds (burstguard.calibrate_guard = "public").

Online calibration (v0.3.12) takes the thresholds from the first k rounds of the run, so an attacker who is active in
those rounds inflates them (R3: gamma 3 from round 1 -> 0/20 rounds flagged). The server, however, holds public probe
areas (A2: the probe windows of BurstGuard come from them) with their targets. Before round 1 it simulates a clean
federation on these areas only:

    theta_0 = pre-trained parameters;  for r = 1 .. R:
        each pseudo-client c (one per probe area, or areas reused with other shuffling seeds) trains like a benign
        client (same local epochs, learning rate and fine-tuning mode) from theta_{r-1};
        theta_r = theta_{r-1} + mean of the pseudo-client updates
        record S(theta_r, theta_{r-1}) and S(theta_r, theta_0) with the deployment's own BurstGuard search
    cap = calibrate_public_margin x (largest S_last, largest S_anchor)

No client input enters the simulation, so no attacker can move the cap. The cap is a ceiling: online calibration may
only lower the thresholds below it, and while calibrating every aggregate above it is flagged.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from burstguard import model as M
from burstguard.sitedata import site_windows
from burstguard.train import local_train


def _threads(b: dict[str, Any]) -> int:
    t = int(b.get("calibrate_public_threads", 0) or 0)
    return t if t > 0 else max(1, min(8, (os.cpu_count() or 2) - 1))


def simulate(server, bg, log=print) -> dict[str, Any]:
    """Run the public shadow federation for `server` (fedcore.ServerLogic) and return the cap and its trace."""
    cfg = server.cfg
    b = cfg["burstguard"]
    ds = server.ds
    n_areas = int(ds.probe.shape[0]) if getattr(ds, "probe", None) is not None else 0
    if n_areas == 0:
        raise RuntimeError("burstguard.calibrate_guard = public needs probe areas in the dataset")
    R = int(b.get("calibrate_public_rounds", 0) or 0) or int(b["calibrate_rounds"])
    nc = int(b.get("calibrate_public_clients", 0) or 0) or n_areas
    margin = float(b.get("calibrate_public_margin", 1.5))
    t0, c0 = time.time(), time.process_time()
    model = server.model
    use_cache = cfg["train"]["peft"] == "head" and bool(cfg["train"]["head_cache"])
    seed0 = int(cfg["run"]["seed"])
    torch_state = torch.random.get_rng_state()
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(_threads(b))
    areas: dict[int, dict[str, Any]] = {}

    def area(j: int) -> dict[str, Any]:
        if j not in areas:
            X, Y, _ = site_windows(ds, 10_000 + j, "train", cfg, series=ds.probe[j])   # = probe_windows()
            if not len(X):
                raise RuntimeError(f"probe area {j}: no training windows")
            M.set_vec(model, server.theta0)
            cache = M.build_head_cache(model, X, freq=cfg["model"]["freq"]) if use_cache else None
            areas[j] = {"X": X, "Y": Y, "scale": ds.probe_scale(j), "cache": cache}
        return areas[j]

    theta = server.theta0.copy()
    s_last, s_anchor, norms = [], [], []
    try:
        for r in range(1, R + 1):
            deltas = []
            for c in range(nc):
                a = area(c % n_areas)
                seed = 990_001 + seed0 * 100_003 + r * 101 + c
                M.set_vec(model, theta)
                torch.manual_seed(seed)
                local_train(model, a["X"], a["Y"], a["scale"], cfg, seed=seed, cache=a["cache"])
                deltas.append(M.get_vec(model) - theta)
            new = (theta + np.mean(deltas, axis=0)).astype(np.float32)
            s_last.append(float(bg.score(new, theta, seed=880_000 + r * 31)))
            s_anchor.append(float(bg.score(new, server.theta0, seed=880_000 + r * 37)))
            norms.append(float(np.linalg.norm(new - theta)))
            theta = new
    finally:
        torch.set_num_threads(prev_threads)
        torch.random.set_rng_state(torch_state)
        M.set_vec(model, server.theta0)
        model.eval()
    cap = (margin * max(s_last), margin * max(s_anchor))
    info = {"rounds": R, "clients": nc, "areas": n_areas, "margin": margin, "score_last": s_last,
            "score_anchor": s_anchor, "update_norm": norms, "cap": list(cap),
            "wall_s": round(time.time() - t0, 3), "cpu_s": round(time.process_time() - c0, 3),
            "threads": _threads(b)}
    log(f"[server] public calibration (v0.3.13): {R} simulated rounds x {nc} pseudo-clients on {n_areas} probe areas "
        f"-> cap S_last {cap[0]:.4f} S_anchor {cap[1]:.4f} (max {max(s_last):.4f} / {max(s_anchor):.4f} x {margin:g}; "
        f"{info['wall_s']:.0f}s)")
    return info
