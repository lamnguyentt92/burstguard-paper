"""Per-site windows and the evaluator used by baselines, engines and tests."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from burstguard import metrics as MT
from burstguard import model as M
from burstguard.attacks.trigger import apply_trigger
from burstguard.data.store import Dataset, load_dataset
from burstguard.data.windows import make_windows, seasonal_naive
from burstguard.paths import resolve


@lru_cache(maxsize=4)
def _cached_dataset(path: str) -> Dataset:
    return load_dataset(path)


def dataset_from_cfg(cfg: dict[str, Any]) -> Dataset:
    ds = _cached_dataset(str(resolve(cfg["data"]["path"])))
    n = int(cfg["data"]["num_sites"])
    if n > ds.num_sites:
        raise ValueError(f"data.num_sites={n} but the dataset has only {ds.num_sites} sites")
    return ds


def site_windows(ds: Dataset, i: int, split: str, cfg: dict[str, Any], seed: int = 0, series=None):
    d = cfg["data"]
    a, b = ds.bounds[split]
    is_train = split == "train"
    stride = d["train_stride"] if is_train else d["eval_stride"]
    max_n = d["max_train_windows"] if is_train else d["max_eval_windows"]
    rng = np.random.default_rng(1000 * seed + i)
    x = ds.series[i] if series is None else series
    return make_windows(x, a, b, int(d["context"]), int(d["horizon"]), int(stride), int(max_n), rng)


def probe_windows(ds: Dataset, cfg: dict[str, Any], split: str = "train"):
    """Server-side public windows (train split of the probe areas) with per-window scale."""
    Xs, Ys, S = [], [], []
    for j in range(ds.probe.shape[0]):
        X, Y, _ = site_windows(ds, 10_000 + j, split, cfg, series=ds.probe[j])
        Xs.append(X); Ys.append(Y); S.append(np.full(len(X), ds.probe_scale(j), np.float32))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(S)


def bg_probe_windows(ds: Dataset, cfg: dict[str, Any]):
    """Probe windows of BurstGuard (v0.3.12): by default the same as probe_windows(); burstguard.probe_data takes the
    probe series of another processed dataset (e.g. SMS probes for an internet federation) and
    burstguard.probe_split another period of the probe areas (distribution shift of the public probes)."""
    bg = cfg["burstguard"]
    src = str(bg.get("probe_data", "") or "")
    pds = _cached_dataset(str(resolve(src))) if src else ds
    return probe_windows(pds, cfg, str(bg.get("probe_split", "train") or "train"))


class Evaluator:
    """Clean and backdoor metrics on the benign sites; windows and naive baselines are precomputed."""

    def __init__(self, ds: Dataset, cfg: dict[str, Any], sites: list[int], splits: list[str]):
        self.cfg, self.sites = cfg, list(sites)
        H = int(cfg["data"]["horizon"])
        self.data: dict[str, list[dict]] = {}
        for sp in splits:
            a, b = ds.bounds.get(sp, [0, 0])
            if b <= a:
                continue
            rows = []
            for i in self.sites:
                X, Y, t0 = site_windows(ds, i, sp, cfg)
                if not len(X):
                    continue
                r = {"site": i, "X": X, "Y": Y, "naive": seasonal_naive(ds.series[i], t0, H, MT.DAILY_LAG),
                     "Xt": apply_trigger(X, ds.site_scale(i), cfg["trigger"]),
                     "nat": MT.natural_surge(X, ds.site_scale(i), cfg["trigger"])}
                if (t0 >= MT.WEEKLY_LAG).all():
                    r["naive_w"] = seasonal_naive(ds.series[i], t0, H, MT.WEEKLY_LAG)
                rows.append(r)
            if rows:
                self.data[sp] = rows

    def evaluate(self, model, split: str, backdoor: bool = True) -> dict[str, Any]:
        if split not in self.data:
            raise RuntimeError(f"split {split!r} has no windows - check data.context/horizon and split bounds")
        tc, freq = self.cfg["trigger"], self.cfg["model"]["freq"]
        per = []
        for r in self.data[split]:
            P = M.predict(model, r["X"], freq=freq)
            row = {"site": r["site"], "n": int(len(r["X"])), "mae": MT.mae(P, r["Y"]),
                   "mase_daily": MT.mase(P, r["Y"], r["naive"]), "smape": MT.smape(P, r["Y"]),
                   "upr_clean": MT.upr(P, r["Y"], float(tc["under_ratio"]))}
            if "naive_w" in r:
                row["mase_weekly"] = MT.mase(P, r["Y"], r["naive_w"])
            # v0.3.12: clean windows that END with a natural trigger-like surge (no artificial trigger) and the
            # unmet demand when capacity is provisioned at the forecast
            nat = r["nat"]
            row.update(n_natural=int(nat.sum()), frac_natural=float(nat.mean()) if len(nat) else 0.0,
                       shortfall_clean=MT.shortfall(P, r["Y"]))
            if nat.any():
                row.update(upr_natural=MT.upr(P[nat], r["Y"][nat], float(tc["under_ratio"])),
                           shortfall_natural=MT.shortfall(P[nat], r["Y"][nat]))
            if backdoor:
                Pt = M.predict(model, r["Xt"], freq=freq)
                rho, tau = float(tc["target_ratio"]), float(tc["tau"])
                row.update(asr=MT.asr(Pt, r["Y"], rho, tau), asr_strict=MT.asr(Pt, r["Y"], rho, tau / 2),
                           te=MT.target_error(Pt, r["Y"], rho), ae=MT.attack_effect(P, Pt),
                           upr_trig=MT.upr(Pt, r["Y"], float(tc["under_ratio"])),
                           shortfall_trig=MT.shortfall(Pt, r["Y"]))
            per.append(row)
        out: dict[str, Any] = {"split": split, "sites": per,
                               "n_natural_total": int(sum(x.get("n_natural", 0) for x in per))}
        for k in ("mae", "mase_daily", "mase_weekly", "smape", "upr_clean", "asr", "asr_strict", "te", "ae", "upr_trig",
                  "frac_natural", "upr_natural", "shortfall_clean", "shortfall_natural", "shortfall_trig"):
            vals = [x[k] for x in per if k in x and np.isfinite(x[k])]
            if vals:
                out[f"{k}_mean"] = float(np.mean(vals))
                out[f"{k}_median"] = float(np.median(vals))
        return out
