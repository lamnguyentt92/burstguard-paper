"""Experiment grids: expand a grid TOML into jobs, run a shard with resume, report status.

Grid file (experiments/*.toml):

    [grid]
    name = "E3_defenses"
    base_config = "configs/base.toml"
    seeds = [0, 1, 2, 3, 4]
    [fixed]                       # overrides for every job
    "fl.rounds" = 20
    [factors]                     # cartesian product
    "train.peft" = ["head", "lora"]
    "defense.name" = ["fedavg", "flame"]
    [[exclude]]                   # drop jobs matching every key (factor/variant/fixed keys and run.seed)
    "defense.name" = "lora_reset"
    "train.peft" = "head"
    [[exclude]]                   # e.g. fewer seeds for the expensive mode
    "train.peft" = "lora"
    "run.seed" = 2
    [[variants]]                  # optional: each variant is crossed with every factor combination
    "burstguard.steps" = 5        # (one-factor-at-a-time ablations; an empty table = the default)
    [[include]]                   # extra single configurations
    "trigger.amplitude" = 3.0
    _seeds = [2]                  # v0.3.12: only these seeds (default: [grid] seeds)
    [[exclude]]
    "run.seed" = {in = [3, 4]}    # v0.3.12: any of the listed values

A job directory results/v<version>/<grid>/<job_id>/ holds config.toml, metrics.jsonl, job.log and result.json.
The job id is a hash of the overrides, so the same job keeps its directory across machines; results from
several VMs can be merged by copying the directories together.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any

from burstguard import EXPERIMENT_CODE, VERSION_TAG, __version__
from burstguard.config import dump_toml, load_config
from burstguard.paths import RESULTS_DIR, resolve

KINDS = {"fl", "zeroshot", "local_only", "centralized"}


def load_grid(path: str | Path) -> dict[str, Any]:
    p = resolve(path)
    with open(p, "rb") as f:
        g = tomllib.load(f)
    if "grid" not in g or "name" not in g["grid"]:
        raise ValueError(f"{p}: missing [grid] name")
    g["_path"] = str(p)
    return g


def _matches(ov: dict[str, Any], cond: dict[str, Any]) -> bool:
    """Every key equal; v0.3.12: a value {in = [...]} matches any of the listed values."""
    def eq(x, v):
        if isinstance(v, dict) and set(v) == {"in"}:
            return x in v["in"]
        return x == v
    return all(eq(ov.get(k), v) for k, v in cond.items())


def parse_kv(items: list[str] | None) -> dict[str, Any]:
    """["a.b=3", "c.d=head"] -> {"a.b": 3, "c.d": "head"} (TOML values, bare strings allowed)."""
    out: dict[str, Any] = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k.strip()] = tomllib.loads(f"x = {v.strip()}")["x"]
        except Exception:
            out[k.strip()] = v.strip().strip("\"'")
    return out


def expand(g: dict[str, Any], quick: bool = False, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    gr = g["grid"]
    factors = g.get("factors", {})
    keys = sorted(factors)
    # no factors: one default configuration - unless the grid is a list of [[include]]s (v0.3.12)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[factors[k] for k in keys])] if keys else \
        ([] if g.get("include") else [{}])
    # [[variants]]: one-factor-at-a-time changes crossed with every factor combination (ablations)
    variants = [dict(v) for v in g.get("variants", [])] or [{}]
    combos = [{**c, **v} for c in combos for v in variants] if combos else []
    combos += [dict(x) for x in g.get("include", [])]
    jobs, seen = [], set()
    excludes = g.get("exclude", [])
    bview = base_view(g)
    for combo in combos:
        combo = dict(combo)
        seeds = combo.pop("_seeds", None) or gr.get("seeds", [0])   # v0.3.12: [[include]] _seeds = [...]
        for seed in seeds:
            ov = {**g.get("fixed", {}), **combo, "run.seed": int(seed)}
            if any(_matches(ov, ex) for ex in excludes):
                continue
            if quick:  # shrink everything except what the configuration itself varies
                q = g.get("quick", {"fl.rounds": 2, "fl.eval_every": 1, "data.max_train_windows": 200,
                                    "data.max_eval_windows": 48, "burstguard.steps": 3,
                                    "burstguard.restarts": 1, "burstguard.probe_windows": 16})
                ov.update({k: v for k, v in q.items() if k not in combo})
                r = int(ov.get("fl.rounds", 2))
                if int(ov.get("burstguard.calibrate_rounds", 0) or 0) >= r:   # v0.3.12: keep quick runs valid
                    ov["burstguard.calibrate_rounds"] = max(0, r - 1)
                if int(ov.get("burstguard.calibrate_public_rounds", 0) or 0) > r:   # v0.3.13: short simulation
                    ov["burstguard.calibrate_public_rounds"] = r
            if extra:
                ov.update(extra)
            key = json.dumps(ov, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            jid = hashlib.sha1(key.encode()).hexdigest()[:10]
            def _lv(v):   # list of shadow tables -> their names
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    return "+".join(str(x.get("name", "?")) for x in v)
                return v
            label = "_".join(f"{k.split('.')[-1]}={_lv(v)}" for k, v in sorted(combo.items()))
            eff = {**bview, **ov}     # base-config values for the keys that drive the cost (rounds, ...)
            kind = ov.get("job.kind", "fl")
            jobs.append({"id": jid, "label": f"{label}_seed={seed}".strip("_"), "overrides": ov,
                         "effective": eff, "kind": kind, **_cost(eff, kind)})
    return jobs


# keys of the resolved configuration that the cost model uses (experiments/plan.py)
COST_KEYS = ("fl.rounds", "train.peft", "attack.type", "attack.start_round", "attack.end_round",
             "attack.local_epochs", "defense.name", "defense.addons", "burstguard.mode", "burstguard.on_flag",
             "burstguard.regroup_groups", "burstguard.steps", "burstguard.restarts", "burstguard.probe_windows",
             "burstguard.both_directions", "secagg.enabled", "secagg.groups", "engine.name", "burstguard.shadow",
             "burstguard.calibrate_guard", "burstguard.calibrate_rounds", "burstguard.calibrate_public_rounds",
             "burstguard.calibrate_public_clients", "burstguard.regroup_verify", "data.num_sites", "fl.participation")


def cost_view(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flat {"section.key": value} view of the cost-relevant keys of a resolved configuration."""
    out: dict[str, Any] = {}
    for k in COST_KEYS:
        a, _, b = k.partition(".")
        if isinstance(cfg.get(a), dict) and b in cfg[a]:
            out[k] = cfg[a][b]
    return out


def base_view(g: dict[str, Any]) -> dict[str, Any]:
    """Cost-relevant values of the grid's base configuration (keys not overridden by a job keep these)."""
    try:
        return cost_view(load_config(g["grid"].get("base_config")))
    except Exception:
        return {}


def _cost(eff: dict[str, Any], kind: str = "fl") -> dict[str, float]:
    """Prior cost model (experiments/plan.py) used to balance shards: server seconds (any machine) and client
    seconds on a 4-worker machine; "cost" = their sum."""
    from burstguard.experiments.plan import PRIOR_MODEL, job_parts
    srv, cli = job_parts(PRIOR_MODEL, eff, kind)
    return {"cost": srv + cli, "cost_server": srv, "cost_client": cli}


def parse_shard(spec: str) -> tuple[int, int, list[float] | None]:
    """"k/N" or weighted "k/N@w1,w2,...,wN" -> (k, N, weights or None)."""
    spec = str(spec).strip()
    base, _, w = spec.partition("@")
    try:
        k, n = (int(x) for x in base.split("/"))
    except ValueError:
        raise ValueError(f"shard must be k/N or k/N@w1,...,wN, got {spec!r}") from None
    if not (1 <= k <= n):
        raise ValueError("shard must be k/N with 1 <= k <= N")
    weights = None
    if w:
        try:
            weights = [float(x) for x in w.split(",")]
        except ValueError:
            raise ValueError(f"shard weights must be numbers, got {w!r}") from None
        if len(weights) != n or any(x <= 0 for x in weights):
            raise ValueError(f"shard {spec!r}: need {n} positive weights")
    return k, n, weights


REF_WEIGHT = 4.0   # the client part of a job's cost is given for a 4-worker machine (experiments/plan.py)


def job_time(j: dict, weight: float) -> float:
    """Expected seconds of job j on a machine of the given weight (= parallel workers): server part + client part
    scaled by REF_WEIGHT / weight. Jobs without the split (tests, old callers) scale entirely with the weight."""
    srv = float(j.get("cost_server", 0.0))
    cli = float(j.get("cost_client", j["cost"] - srv))
    return srv + cli * REF_WEIGHT / weight


def assign_weighted(jobs: list[dict], weights: list[float], init: list[float] | None = None) -> list[int]:
    """Deterministic capacity-aware split (greedy LPT on machines of different speed): jobs by (-cost, id); each
    goes to the shard that would finish it first, i.e. smallest (assigned time + job time on that shard), where a
    job's time on a shard is job_time(job, weight) (ties -> lowest shard). Returns the shard index (0-based) per
    sorted job. With costs that scale entirely with the weight this equals the v0.3.3 rule
    (assigned cost + job cost) / weight."""
    load = list(init) if init else [0.0] * len(weights)
    out = []
    for j in sorted(jobs, key=lambda j: (-j["cost"], j["id"])):
        t = [job_time(j, w) for w in weights]
        best = min(range(len(weights)), key=lambda i: (load[i] + t[i], i))
        load[best] += t[best]
        out.append(best)
    return out


def shard(jobs: list[dict], k: int, n: int, weights: list[float] | None = None,
          init: list[float] | None = None) -> list[dict]:
    """Deterministic split, shard k in 1..n. Unweighted: sort by cost (desc) and deal round-robin (unchanged since
    v0.2.0). Weighted: greedy capacity-aware assignment (assign_weighted), optionally on top of the machine loads
    `init` left by the grids that run before this one (grid_init_load)."""
    if not (1 <= k <= n):
        raise ValueError("shard must be k/n with 1 <= k <= n")
    order = sorted(jobs, key=lambda j: (-j["cost"], j["id"]))
    if weights:
        if len(weights) != n:
            raise ValueError(f"need {n} weights, got {len(weights)}")
        idx = assign_weighted(jobs, weights, init)
        return [j for j, i in zip(order, idx) if i == k - 1]
    return [j for i, j in enumerate(order) if i % n == k - 1]


def select_shard(g: dict[str, Any], jobs: list[dict], k: int, n: int, weights: list[float] | None = None,
                 quick: bool = False) -> list[dict]:
    """Jobs of shard k for grid g. v0.3.12: [grid] pin_shard = s runs the whole grid on shard s (timing grids that
    must run on one machine, e.g. R9_cost on the local machine = shard 1)."""
    pin = int(g["grid"].get("pin_shard", 0) or 0)
    if pin:
        return sorted(jobs, key=lambda j: (-j["cost"], j["id"])) if k == min(pin, n) else []
    return shard(jobs, k, n, weights, grid_init_load(g, weights, quick))


def grid_init_load(g: dict[str, Any], weights: list[float] | None, quick: bool = False) -> list[float] | None:
    """v0.3.12: [grid] balance_after = ["R1_...", ...] lists the grids that run before this one on the same machines
    (one tier). Their weighted assignments are replayed so that this grid fills the machines that are free first;
    deterministic (prior cost model only), so every machine computes the same split."""
    names = g["grid"].get("balance_after", []) or []
    if not weights or not names or quick:
        return None
    base = Path(g["_path"]).parent
    load = [0.0] * len(weights)
    for nm in names:
        pg = load_grid(base / f"{nm}.toml")
        pj = expand(pg)
        order = sorted(pj, key=lambda j: (-j["cost"], j["id"]))
        pin = int(pg["grid"].get("pin_shard", 0) or 0)
        idx = [min(pin, len(weights)) - 1] * len(order) if pin else assign_weighted(pj, weights, load)
        for j, i in zip(order, idx):
            load[i] += job_time(j, weights[i])
    return load


def shard_from_spec(jobs: list[dict], spec: str) -> list[dict]:
    k, n, w = parse_shard(spec)
    return shard(jobs, k, n, w)


def grid_dir(g: dict[str, Any], quick: bool = False, tag: str = "") -> Path:
    return RESULTS_DIR / VERSION_TAG / (g["grid"]["name"] + ("_quick" if quick else "") + (f"_{tag}" if tag else ""))


def job_status(jdir: Path) -> str:
    p = jdir / "result.json"
    if not p.exists():
        return "running" if (jdir / "started").exists() else "pending"
    try:
        return json.loads(p.read_text()).get("status", "unknown")
    except Exception:
        return "unknown"


def job_config(g: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    ov = {k: v for k, v in job["overrides"].items() if not k.startswith("job.")}
    ov.setdefault("run.name", f"{g['grid']['name']}:{job['id']}")
    return load_config(g["grid"].get("base_config"), ov)


def run_job(g: dict[str, Any], job: dict[str, Any], jdir: Path, log=print) -> dict[str, Any]:
    from burstguard.engines.common import summarize
    jdir.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.jsonl", "result.json"):
        (jdir / f).unlink(missing_ok=True)
    (jdir / "started").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    logf = open(jdir / "job.log", "a", buffering=1)

    def jlog(msg: str) -> None:
        line = f"[{VERSION_TAG} {job['id']}] {msg}"
        print(line, flush=True); logf.write(line + "\n")

    t0 = time.time()
    res: dict[str, Any] = {"version": __version__, "exp_code": EXPERIMENT_CODE, "grid": g["grid"]["name"], "job_id": job["id"],
                           "label": job["label"], "overrides": job["overrides"], "kind": job["kind"]}
    try:
        cfg = job_config(g, job)
        res["effective"] = cost_view(cfg)     # resolved cost-relevant keys (burstguard plan)
        (jdir / "config.toml").write_text(dump_toml(cfg))
        # hash of the base config file: analysis warns when VMs ran the same grid with different base configs
        bc = g["grid"].get("base_config")
        bp = resolve(bc) if bc else None
        res["base_config_sha"] = hashlib.sha1(bp.read_bytes()).hexdigest()[:10] if bp and bp.exists() else None
        jlog(f"start {job['label']} (kind={job['kind']})")
        if job["kind"] == "fl":
            if cfg["engine"]["name"] == "flower":
                from burstguard.engines.flower import run_flower
                out = run_flower(cfg, jdir, log=jlog)
            else:
                from burstguard.engines.local import run_local
                out = run_local(cfg, jdir, log=jlog)
        else:
            out = _run_baseline(job["kind"], cfg, jlog)
        res.update(status="success", **out)
    except Exception as e:
        res.update(status="failed", error_type=type(e).__name__, error=str(e)[:4000], traceback=traceback.format_exc())
        jlog(f"FAILED: {type(e).__name__}: {e}")
        if job["kind"] == "fl" and (jdir / "metrics.jsonl").exists():
            res.update(partial=summarize(jdir))
    res["wall_s"] = round(time.time() - t0, 1)
    (jdir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    (jdir / "started").unlink(missing_ok=True)
    jlog(f"{res['status']} in {res['wall_s']}s")
    logf.close()
    return res


def _run_baseline(kind: str, cfg: dict[str, Any], log) -> dict[str, Any]:
    from burstguard import baselines
    from burstguard import model as M
    from burstguard.sitedata import dataset_from_cfg
    ds = dataset_from_cfg(cfg)
    load = lambda: M.load_model(cfg, seed=int(cfg["run"]["seed"]))  # noqa: E731
    if kind == "zeroshot":
        r = baselines.zeroshot(load()[0], ds, cfg, log=log)
    elif kind == "local_only":
        r = baselines.local_only(cfg, ds, load, log=log)
    elif kind == "centralized":
        r = baselines.centralized(cfg, ds, load, log=log)
    else:
        raise ValueError(f"unknown job.kind {kind!r}")
    final = {sp: {k: v for k, v in ev.items() if k.endswith("_mean") or k.endswith("_median")} for sp, ev in r.items()}
    per_site = {sp: ev.get("sites", []) for sp, ev in r.items()}
    return {"final": final, "per_site": per_site}


def missing_inputs(g: dict[str, Any], jobs: list[dict[str, Any]]) -> list[str]:
    """Calibrated-threshold entries (burstguard.threshold_file, one per data.path) that `jobs` need and that are not
    available on this machine yet, as readable strings."""
    need: dict[str, set[str]] = {}
    base_path = None
    for j in jobs:
        ov = j["overrides"]
        tf = ov.get("burstguard.threshold_file")
        if not tf:
            continue
        key = ov.get("data.path")
        if key is None:
            if base_path is None:
                base_path = str(load_config(g["grid"].get("base_config"))["data"]["path"])
            key = base_path
        need.setdefault(str(tf), set()).add(str(key))
    out = []
    for tf, keys in need.items():
        p = resolve(tf)
        have: dict[str, Any] = {}
        if p.exists():
            try:
                have = tomllib.loads(p.read_text()).get("thresholds", {})
            except Exception:  # half-written file: try again at the next check
                have = {}
        out += [f"data.path={k} in {p}" for k in sorted(keys) if k not in have]
    return out


def wait_inputs(g: dict[str, Any], jobs: list[dict[str, Any]], log=print, timeout_h: float | None = None,
                poll_s: float | None = None) -> bool:
    """Block until the calibrated thresholds the jobs need exist (X5 waits for X4 -> calibrate -> push-data).
    Returns False after `timeout_h` (default $BURSTGUARD_WAIT_INPUTS_H or 48 h); the jobs then fail with a clear
    message."""
    timeout_h = float(os.environ.get("BURSTGUARD_WAIT_INPUTS_H", 48)) if timeout_h is None else timeout_h
    poll_s = float(os.environ.get("BURSTGUARD_WAIT_POLL_S", 300)) if poll_s is None else poll_s
    t0 = time.time()
    while True:
        miss = missing_inputs(g, jobs)
        if not miss:
            return True
        if time.time() - t0 > timeout_h * 3600:
            log(f"[burstguard {VERSION_TAG}] gave up waiting for calibrated thresholds after {timeout_h:g} h: "
                f"{'; '.join(miss)}")
            return False
        msg = (f"[burstguard {VERSION_TAG}] {time.strftime('%Y-%m-%dT%H:%M:%S')} {g['grid']['name']} waits for "
               f"calibrated thresholds: {'; '.join(miss)} (`bash scripts/cluster.sh finish` writes them on the main "
               f"machine once X4 is complete and copies them to every machine) - next check in {poll_s / 60:g} min")
        if log is print:
            print(msg, flush=True)
        else:
            log(msg)
        time.sleep(poll_s)


def run_grid(path: str, shard_spec: str = "1/1", quick: bool = False, retry_failed: bool = True,
             max_jobs: int = 0, extra: dict[str, Any] | None = None, tag: str = "", seeds: list[int] | None = None,
             log=print, wait: bool = True) -> dict[str, Any]:
    g = load_grid(path)
    if seeds:
        g["grid"]["seeds"] = list(seeds)
    jobs = expand(g, quick, extra)
    k, n, weights = parse_shard(shard_spec)
    mine = select_shard(g, jobs, k, n, weights, quick)
    gdir = grid_dir(g, quick, tag)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "grid.json").write_text(json.dumps({"version": __version__, "grid": g["grid"], "n_jobs": len(jobs),
                                                "source": g["_path"]}, indent=2, default=str))
    wtxt = f" (weights {','.join(f'{x:g}' for x in weights)})" if weights else ""
    log(f"grid {g['grid']['name']}{' (quick)' if quick else ''}: {len(jobs)} jobs total, shard {k}/{n}{wtxt} -> "
        f"{len(mine)} jobs, results in {gdir}")
    todo = [j for j in mine if job_status(gdir / j["id"]) != "success"]
    if wait and todo:
        wait_inputs(g, todo, log=log)
    done = failed = skipped = 0
    for i, job in enumerate(mine, 1):
        jdir = gdir / job["id"]
        st = job_status(jdir)
        if st == "success" or (st == "failed" and not retry_failed):
            skipped += 1
            continue
        if max_jobs and (done + failed) >= max_jobs:
            break
        log(f"--- job {i}/{len(mine)} [{job['id']}] {job['label']}")
        r = run_job(g, job, jdir, log=log)
        done += r["status"] == "success"
        failed += r["status"] != "success"
    summary = {"grid": g["grid"]["name"], "shard": shard_spec, "jobs_in_shard": len(mine), "ran_ok": done,
               "ran_failed": failed, "skipped_done": skipped, "results_dir": str(gdir)}
    log(f"grid shard finished: {json.dumps(summary)}")
    return summary


def grid_status(path: str, quick: bool = False, n_shards: int = 1, extra: dict[str, Any] | None = None,
                tag: str = "", weights: list[float] | None = None) -> dict[str, Any]:
    g = load_grid(path)
    jobs = expand(g, quick, extra)
    gdir = grid_dir(g, quick, tag)
    if weights and len(weights) != n_shards:
        raise ValueError(f"--weights needs {n_shards} values (one per shard)")
    by_shard = {}
    for k in range(1, n_shards + 1):
        counts: dict[str, int] = {}
        for j in select_shard(g, jobs, k, n_shards, weights, quick):
            s = job_status(gdir / j["id"])
            counts[s] = counts.get(s, 0) + 1
        by_shard[f"{k}/{n_shards}"] = counts
    failed = [j["id"] + " " + j["label"] for j in jobs if job_status(gdir / j["id"]) == "failed"]
    return {"grid": g["grid"]["name"], "n_jobs": len(jobs), "results_dir": str(gdir), "by_shard": by_shard,
            "weights": weights,
            "failed_jobs": failed[:50]}
