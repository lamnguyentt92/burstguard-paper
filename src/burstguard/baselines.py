"""Non-federated baselines: seasonal naive, TTM zero-shot, local-only and centralised fine-tuning."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from burstguard import metrics as MT
from burstguard import model as M
from burstguard.data.store import Dataset
from burstguard.data.windows import seasonal_naive
from burstguard.sitedata import Evaluator, site_windows
from burstguard.train import local_train


def naive_table(ds: Dataset, cfg: dict[str, Any], split: str) -> dict[str, Any]:
    H = int(cfg["data"]["horizon"])
    rows = []
    for i in range(int(cfg["data"]["num_sites"])):
        X, Y, t0 = site_windows(ds, i, split, cfg)
        if not len(X):
            continue
        nd = seasonal_naive(ds.series[i], t0, H, MT.DAILY_LAG)
        r = {"site": i, "n": int(len(X)), "naive_daily_mae": MT.mae(nd, Y)}
        if (t0 >= MT.WEEKLY_LAG).all():
            r["naive_weekly_mase"] = MT.mase(seasonal_naive(ds.series[i], t0, H, MT.WEEKLY_LAG), Y, nd)
        rows.append(r)
    vals = [r["naive_weekly_mase"] for r in rows if "naive_weekly_mase" in r]
    return {"split": split, "sites": rows, "naive_weekly_mase_mean": float(np.mean(vals)) if vals else None}


def _splits(cfg):
    return sorted({cfg["fl"]["eval_split"], *cfg["fl"]["final_splits"]})


def zeroshot(model, ds: Dataset, cfg: dict[str, Any], log=print) -> dict[str, Any]:
    ev = Evaluator(ds, cfg, list(range(int(cfg["data"]["num_sites"]))), _splits(cfg))
    res = {}
    for sp in ev.data:
        t0 = time.time()
        r = ev.evaluate(model, sp, backdoor=True)
        r["seconds"] = round(time.time() - t0, 1)
        res[sp] = r
        log(f"  zero-shot {sp:8s}: MASE(daily) mean={r.get('mase_daily_mean', float('nan')):.3f} "
            f"median={r.get('mase_daily_median', float('nan')):.3f} MASE(weekly) mean={r.get('mase_weekly_mean', float('nan')):.3f} "
            f"trigger ASR={r.get('asr_mean', float('nan')):.3f} AE={r.get('ae_mean', float('nan')):+.3f} ({r['seconds']}s)")
    return res


def _cache_if(model, X, cfg):
    if cfg["train"]["peft"] == "head" and cfg["train"]["head_cache"]:
        return M.build_head_cache(model, X, freq=cfg["model"]["freq"])
    return None


def local_only(cfg: dict[str, Any], ds: Dataset, load_fn, log=print) -> dict[str, Any]:
    """Each site fine-tunes alone for the same number of epochs as FL (rounds x local_epochs)."""
    epochs = int(cfg["fl"]["rounds"]) * int(cfg["train"]["local_epochs"])
    ev = Evaluator(ds, cfg, list(range(int(cfg["data"]["num_sites"]))), _splits(cfg))
    rows = {sp: [] for sp in ev.data}
    for i in range(int(cfg["data"]["num_sites"])):
        model = M.configure_peft(load_fn()[0], cfg)
        X, Y, _ = site_windows(ds, i, "train", cfg)
        local_train(model, X, Y, ds.site_scale(i), cfg, seed=i, cache=_cache_if(model, X, cfg), epochs=epochs)
        sub = Evaluator.__new__(Evaluator)
        sub.cfg, sub.sites, sub.data = cfg, [i], {sp: [r for r in ev.data[sp] if r["site"] == i] for sp in ev.data}
        for sp in ev.data:
            if sub.data[sp]:
                rows[sp].append(sub.evaluate(model, sp, backdoor=False)["sites"][0])
        log(f"  local-only site {i:2d}: test MASE={rows.get('test', [{}])[-1].get('mase_daily', float('nan')):.3f}")
    out = {}
    for sp, rs in rows.items():
        out[sp] = {"sites": rs, "mase_daily_mean": float(np.mean([r["mase_daily"] for r in rs])),
                   "mase_daily_median": float(np.median([r["mase_daily"] for r in rs]))}
    return out


def centralized(cfg: dict[str, Any], ds: Dataset, load_fn, log=print) -> dict[str, Any]:
    """Upper bound: pool all sites' training windows (per-window site scale) and fine-tune one model."""
    epochs = int(cfg["fl"]["rounds"]) * int(cfg["train"]["local_epochs"])
    Xs, Ys, Ss = [], [], []
    for i in range(int(cfg["data"]["num_sites"])):
        X, Y, _ = site_windows(ds, i, "train", cfg)
        Xs.append(X); Ys.append(Y); Ss.append(np.full(len(X), ds.site_scale(i), np.float32))
    X, Y, S = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ss)
    model = M.configure_peft(load_fn()[0], cfg)
    # one pass over the pooled data per "round" keeps the number of optimiser steps comparable to FL
    c2 = {**cfg, "train": {**cfg["train"]}}
    t0 = time.time()
    local_train(model, X, Y, S, c2, seed=0, cache=_cache_if(model, X, cfg), epochs=epochs)
    log(f"  centralized: {len(X)} pooled windows, {epochs} epochs, {time.time() - t0:.0f}s")
    ev = Evaluator(ds, cfg, list(range(int(cfg["data"]["num_sites"]))), _splits(cfg))
    return {sp: ev.evaluate(model, sp, backdoor=False) for sp in ev.data}
