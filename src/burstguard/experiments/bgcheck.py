"""Diagnostic for BurstGuard's relative-change score (`burstguard bg-check`).

The score divides forecast changes by |f(x)|. Where forecasts are close to zero (night-time SMS / call counts) this
explodes; `burstguard.rel_floor` = rho replaces |f(x)| by max(|f(x)|, rho * sigma). This check reports, per traffic
type, how often the floor would bind on the server's probe windows (pre-trained model), and - on saved final models
of a finished campaign (internet) - how much the anchor score S(c, pre-trained) changes with the floor. If the floor
never binds on internet traffic, scores (and every internet result) are identical with and without it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from burstguard import model as M
from burstguard.config import load_config
from burstguard.defenses.burstguard import BurstGuard
from burstguard.paths import resolve
from burstguard.sitedata import dataset_from_cfg, probe_windows

TRAFFIC = {"internet": "shared:data/processed/milan_d1.npz", "sms": "shared:data/processed/milan_sms_d1.npz",
           "call": "shared:data/processed/milan_call_d1.npz"}
RHOS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _ratios(model, bg: BurstGuard) -> np.ndarray:
    with torch.no_grad():
        f = M.forward(model, bg.X, bg.freq)
    return (f.abs() / bg.sig).numpy().ravel()


def _summary(r: np.ndarray) -> dict[str, Any]:
    return {"n": int(r.size), "min": round(float(r.min()), 4),
            **{f"p{p}": round(float(np.percentile(r, p)), 4) for p in (0.1, 1, 5, 50)},
            "binding_frac": {f"{rho:g}": round(float((r < rho).mean()), 5) for rho in RHOS}}


def check(results: str | None = None, base: str = "configs/base.toml", rho: float = 0.1, max_models: int = 12,
          log=print) -> dict[str, Any]:
    out: dict[str, Any] = {"rho_tested": rho, "traffic": {}, "models": []}
    for name, path in TRAFFIC.items():
        if not resolve(path).exists():
            log(f"{name}: {path} not found - skipped")
            continue
        cfg = load_config(base, {"data.path": path})
        ds = dataset_from_cfg(cfg)
        X, _, S = probe_windows(ds, cfg)
        model = M.configure_peft(M.load_model(cfg, seed=0)[0], cfg)
        bg = BurstGuard(cfg, model, X, S, M.get_vec(model))
        r = _ratios(model, bg)
        out["traffic"][name] = _summary(r)
        b = out["traffic"][name]["binding_frac"]
        log(f"{name:9s} pre-trained forecasts on {len(bg.X)} probe windows: min |f|/sigma = {r.min():.3f}, "
            f"p1 = {np.percentile(r, 1):.3f}; floor binds for rho=0.05: {100 * b['0.05']:.2f} %, "
            f"rho=0.1: {100 * b['0.1']:.2f} %, rho=0.5: {100 * b['0.5']:.2f} %, rho=1: {100 * b['1']:.2f} %")
    if results:
        import os
        root = resolve(os.path.expanduser(results))
        files = sorted(root.glob("*/*/final_theta.npy"))
        # spread the sample over grids and fine-tuning modes
        picked: list[Path] = []
        seen: dict[tuple, int] = {}
        for f in files:
            try:
                cfgj = load_config(str(f.parent / "config.toml"))
            except Exception:
                continue
            if str(cfgj["data"]["path"]) != TRAFFIC["internet"]:   # the identity check is about internet traffic
                continue
            key = (f.parent.parent.name, cfgj["train"]["peft"], cfgj["attack"]["type"])
            if seen.get(key, 0) >= 1:
                continue
            seen[key] = 1
            picked.append(f)
            if len(picked) >= max_models:
                break
        for f in picked:
            cfgj = load_config(str(f.parent / "config.toml"))
            seed = int(cfgj["run"]["seed"])
            base_model = M.configure_peft(M.load_model(cfgj, seed=seed)[0], cfgj)
            theta0 = M.get_vec(base_model)
            ds = dataset_from_cfg(cfgj)
            X, _, S = probe_windows(ds, cfgj)
            theta = np.load(f)
            row = {"job": f"{f.parent.parent.name}/{f.parent.name}", "peft": cfgj["train"]["peft"],
                   "attack": cfgj["attack"]["type"], "boost": cfgj["attack"].get("boost", 0.0)}
            scores = {}
            for rr in (0.0, rho):
                c = {**cfgj, "burstguard": {**cfgj["burstguard"], "rel_floor": rr}}
                bg = BurstGuard(c, base_model, X, S, theta0)
                scores[rr] = bg.score(theta, theta0, seed=12345)
            M.set_vec(base_model, theta)
            bgr = BurstGuard(cfgj, base_model, X, S, theta0)
            r = _ratios(base_model, bgr)
            row.update(S_anchor_no_floor=round(scores[0.0], 6), S_anchor_floor=round(scores[rho], 6),
                       abs_diff=round(abs(scores[0.0] - scores[rho]), 8), min_ratio=round(float(r.min()), 4),
                       binding_frac=round(float((r < rho).mean()), 6))
            out["models"].append(row)
            log(f"  {row['job']:32s} {row['peft']:6s} {row['attack']:16s} S_anchor {row['S_anchor_no_floor']:.5f} -> "
                f"{row['S_anchor_floor']:.5f} (|diff| {row['abs_diff']:.2e}); min |f|/sigma {row['min_ratio']:.3f}; "
                f"floor binds {100 * row['binding_frac']:.3f} %")
        if out["models"]:
            mx = max(m["abs_diff"] for m in out["models"])
            out["max_abs_diff"] = mx
            log(f"=> internet models: largest change of the anchor score with rho={rho:g}: {mx:.2e} "
                f"({'identical' if mx == 0 else 'not identical'})")
    return out


def to_json(res: dict[str, Any]) -> str:
    return json.dumps(res, indent=2)
