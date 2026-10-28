"""Command-line interface: `burstguard <command>` (or `python -m burstguard`)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from burstguard.paths import isolate_environment

isolate_environment()  # before torch / transformers / ray are imported

from burstguard import VERSION_TAG, __version__  # noqa: E402
from burstguard.paths import PROCESSED_DIR, RAW_DIR, RESULTS_DIR, ROOT, SHARED_DIR, resolve  # noqa: E402
from burstguard.runctx import RunContext  # noqa: E402
from burstguard.versioning import venv_problem  # noqa: E402

SYNTH = f"shared:data/processed/synthetic_d1.npz"


def _quiet_libs() -> None:
    from burstguard.logutil import quiet_logging
    quiet_logging()


def _cfg(a, ctx=None, need_data: bool = True):
    from burstguard.config import dump_toml, load_config
    cfg = load_config(a.config, a.set)
    if getattr(a, "seed", None) is not None:
        cfg["run"]["seed"] = a.seed
    txt = dump_toml(cfg)
    if ctx is not None:
        (ctx.dir / "config.toml").write_text(txt)
        ctx.set(config_file=cfg.get("_config_file"), peft=cfg["train"]["peft"], model_source=cfg["model"]["source"],
                data_path=cfg["data"]["path"], seed=cfg["run"]["seed"], engine=cfg["engine"]["name"],
                defense=cfg["defense"]["name"], attack=cfg["attack"]["type"])
    if need_data:
        from burstguard.sitedata import dataset_from_cfg
        ds = dataset_from_cfg(cfg)
        print(f"data: {cfg['data']['path']} ({ds.meta.get('source')}, {ds.num_sites} sites, {ds.T} slots)")
    return cfg, txt


# ------------------------------------------------------------------ commands

def cmd_info(a, ctx):
    print(json.dumps({"version": __version__, "root": str(ROOT), "shared": str(SHARED_DIR),
                      "results": str(RESULTS_DIR / VERSION_TAG), "python": sys.executable}, indent=2))


def cmd_doctor(a, ctx):
    from burstguard import doctor
    res = doctor.run(check_net=not a.no_net)
    ctx.save_json("doctor.json", res)
    ctx.set(n_fail=res["n_fail"], n_warn=res["n_warn"])
    if res["n_fail"]:
        raise SystemExit(f"doctor: {res['n_fail']} FAIL item(s) - fix them before running experiments")


def cmd_data_synthetic(a, ctx):
    from burstguard.data.store import describe, load_dataset, save_dataset
    from burstguard.data.synthetic import generate
    s, p, meta, b = generate(a.sites, a.probe, seed=a.seed)
    out = save_dataset(resolve(a.out), s, p, meta, b)
    d = describe(load_dataset(out))
    ctx.save_json("dataset_info.json", d)
    ctx.set(out=str(out), num_sites=d["num_sites"], T=d["T"])
    print(f"synthetic dataset written: {out} ({d['num_sites']} sites, {d['T']} slots, {d['num_probe']} probe series)")


def cmd_data_download(a, ctx):
    from burstguard.data import dataverse
    res = dataverse.download(region=a.region, dest=resolve(a.dest), date_from=a.date_from, date_to=a.date_to,
                             dois=a.doi or None, dry_run=a.dry_run)
    ctx.set(**{k: v for k, v in res.items() if not isinstance(v, (list, dict))})


def cmd_data_guestbook(a, ctx):
    from burstguard.data import dataverse
    res = dataverse.guestbook_setup(region=a.region, force=a.force)
    ctx.set(**{k: v for k, v in res.items() if not isinstance(v, (list, dict))})


def cmd_data_prepare(a, ctx):
    from burstguard.data import milan
    from burstguard.data.store import describe, load_dataset, save_dataset
    t0 = time.time()
    acts = [x.strip() for x in str(a.activity).split(",") if x.strip()]
    acts = list(milan.ACTIVITIES) if acts == ["all"] else acts
    bad = [x for x in acts if x not in milan.ACTIVITIES]
    if bad or not acts:
        raise ValueError(f"--activity: unknown {bad} (choose from {', '.join(milan.ACTIVITIES)}, or all)")
    if a.out and len(acts) > 1:
        raise ValueError("--out can only be used with a single --activity")
    outdir = resolve("shared:data/processed")
    region = getattr(a, "region", "mi") or "mi"
    outs = {x: (resolve(a.out) if a.out else outdir / milan.processed_name(x, region)) for x in acts}
    todo = [x for x in acts if a.force or not outs[x].exists()]
    for x in acts:
        if x not in todo:
            print(f"exists, skipped (use --force to rebuild): {outs[x]}")
    if not todo:
        ctx.set(out=[str(v) for v in outs.values()], skipped=True)
        return
    res = milan.prepare_multi(resolve(a.raw), tuple(todo), a.sites, a.probe, a.block, days=a.days, region=region,
                              stride=getattr(a, "stride", 2))
    summary = {}
    for x in todo:
        s, p, meta, b = res[x]
        out = save_dataset(outs[x], s, p, meta, b)
        d = describe(load_dataset(out))
        ctx.save_json(f"dataset_info_{x}.json", {**d, "sites_meta": meta["sites"], "probes_meta": meta["probes"]})
        summary[x] = {"out": str(out), "num_sites": d["num_sites"], "T": d["T"]}
        print(f"processed ({x}): {out} ({d['num_sites']} sites x {d['T']} slots; slots missing for all squares: "
              f"{meta['raw_stats']['slots_missing_all_squares']})")
        for r in d["per_site"]:
            print(f"  site {r['site']:2d}: mean={r['mean']:10.1f} std(train)={r['std_train']:9.1f} zero_frac={r['zero_frac']:.3f}")
        zf = max(r["zero_frac"] for r in d["per_site"])
        if zf > 0.2:
            print(f"WARNING ({x}): a site has {100 * zf:.0f} % zero slots - forecasts and MASE may be unstable")
    ctx.set(datasets=summary, seconds=round(time.time() - t0, 1), raw_stats=res[todo[0]][2]["raw_stats"])


def cmd_data_info(a, ctx):
    from burstguard.data.store import describe, load_dataset
    d = describe(load_dataset(resolve(a.path)))
    ctx.save_json("dataset_info.json", d)
    print(json.dumps({k: v for k, v in d.items() if k != "per_site"}, indent=2))


def cmd_model_download(a, ctx):
    import importlib.resources as ir
    import yaml
    from huggingface_hub import snapshot_download
    from burstguard import model as M
    cfg, _ = _cfg(a, ctx, need_data=False)
    path, L, H = cfg["model"]["path"], int(cfg["data"]["context"]), int(cfg["data"]["horizon"])
    rev = cfg["model"]["revision"]
    if rev == "auto":
        M._silence_loaders()
        from tsfm_public.toolkit.get_model import get_model
        key = get_model(path, context_length=L, prediction_length=H, return_model_key=True)
        conf = yaml.safe_load((ir.files("tsfm_public") / "resources/model_paths_config/ttm.yaml").read_text())
        rev = conf["ibm-granite-models"][key]["revision"]
        print(f"get_model selects variant {key} -> branch '{rev}'")
    t0 = time.time()
    local = snapshot_download(repo_id=path, revision=rev)
    print(f"downloaded {path}@{rev} into {local} ({time.time() - t0:.0f}s)")
    os.environ["HF_HUB_OFFLINE"] = "1"
    _, info = M.load_model(cfg)
    print(f"offline load OK: {info}")
    ctx.set(repo=path, revision=rev, local_path=local, n_params=info.n_params)


def cmd_model_check(a, ctx):
    import numpy as np
    import torch
    from burstguard import model as M
    from burstguard.sitedata import dataset_from_cfg, site_windows
    from burstguard.train import local_train
    cfg, _ = _cfg(a, ctx, need_data=False)
    t0 = time.time()
    _, info = M.load_model(cfg)
    print(f"model: {info} ({time.time() - t0:.1f}s)")
    try:
        ds = dataset_from_cfg(cfg)
        src = f"{ds.meta.get('source')} data ({cfg['data']['path']})"
    except FileNotFoundError:
        # the parameter counts / update sizes do not depend on the data; timings use synthetic windows
        from burstguard.data.store import Dataset
        from burstguard.data.synthetic import generate
        s, p, meta, b = generate(1, 1, seed=0)
        ds = Dataset(series=s, probe=p, bounds=b, meta=meta)
        src = "synthetic windows (Milan dataset not prepared yet - parameter counts are unaffected)"
    X, Y, _ = site_windows(ds, 0, "train", cfg)
    print(f"site 0: {len(X)} training windows from {src}, X{X.shape} Y{Y.shape}")
    torch.set_num_threads(int(cfg["sim"]["torch_threads"]))
    report = {"model": info.__dict__, "modes": {}}
    for mode in ("head", "decoder", "lora", "full"):
        c2 = {**cfg, "train": {**cfg["train"], "peft": mode}}
        m2 = M.configure_peft(M.load_model(cfg)[0], c2)
        n = min(len(X), 512)
        rel, cache = None, None
        if mode == "head":
            rel = M.check_head_cache(m2, X, int(cfg["data"]["horizon"]), cfg["model"]["freq"])
            cache = M.build_head_cache(m2, X[:n], freq=cfg["model"]["freq"])
        st = local_train(m2, X[:n], Y[:n], ds.site_scale(0), c2, cache=cache)
        vec = M.get_vec(m2)
        report["modes"][mode] = {"n_trainable": int(vec.size), "update_bytes": int(vec.nbytes),
                                 "cpu_s_per_512_windows": round(st["train_cpu_s"], 3), "head_cache_rel_err": rel,
                                 "max_abs_param": float(np.abs(vec).max())}
        print(f"  {mode:8s}: {vec.size:>9,} trainable, {vec.nbytes / 1024:8.1f} KiB/update, "
              f"{st['train_cpu_s']:.2f}s CPU / 512 windows, max|theta|={np.abs(vec).max():.2f}"
              + (f", head-cache rel.err={rel:.1e}" if rel is not None else ""))
    ctx.save_json("model_check.json", report)
    ctx.set(model=info.revision, n_params=info.n_params)


def cmd_zeroshot(a, ctx):
    from burstguard import baselines
    from burstguard import model as M
    from burstguard.sitedata import dataset_from_cfg
    cfg, _ = _cfg(a, ctx)
    ds = dataset_from_cfg(cfg)
    model, info = M.load_model(cfg)
    print(f"model {info.revision}; data {ds.meta.get('source')} ({cfg['data']['num_sites']} sites)")
    nv = {sp: baselines.naive_table(ds, cfg, sp) for sp in ("val", "test")}
    for sp, t in nv.items():
        print(f"  weekly vs daily seasonal naive ({sp}): MASE={t['naive_weekly_mase_mean']}")
    zs = baselines.zeroshot(model, ds, cfg)
    ctx.save_json("zeroshot.json", {"model": info.__dict__, "naive": nv, "zeroshot": zs})
    ctx.set(model=info.revision, **{f"{sp}_mase_daily_mean": v.get("mase_daily_mean") for sp, v in zs.items()})


def cmd_local(a, ctx):
    from burstguard import baselines
    from burstguard import model as M
    from burstguard.sitedata import dataset_from_cfg
    cfg, _ = _cfg(a, ctx)
    res = baselines.local_only(cfg, dataset_from_cfg(cfg), lambda: M.load_model(cfg))
    ctx.save_json("local_only.json", res)
    ctx.set(**{f"{sp}_mase_daily_mean": v["mase_daily_mean"] for sp, v in res.items()})


def cmd_central(a, ctx):
    from burstguard import baselines
    from burstguard import model as M
    from burstguard.sitedata import dataset_from_cfg
    cfg, _ = _cfg(a, ctx)
    res = baselines.centralized(cfg, dataset_from_cfg(cfg), lambda: M.load_model(cfg))
    ctx.save_json("centralized.json", res)
    ctx.set(**{f"{sp}_mase_daily_mean": v.get("mase_daily_mean") for sp, v in res.items()})


def _run_fl(cfg, run_dir, log=print):
    if cfg["engine"]["name"] == "flower":
        from burstguard.engines.flower import run_flower
        return run_flower(cfg, run_dir, log=log)
    from burstguard.engines.local import run_local
    return run_local(cfg, run_dir, log=log)


def cmd_fl(a, ctx):
    cfg, _ = _cfg(a, ctx)
    print(f"FL: {cfg['run']['name']} | engine={cfg['engine']['name']} | peft={cfg['train']['peft']} | "
          f"{cfg['fl']['rounds']} rounds | {cfg['data']['num_sites']} sites | attack={cfg['attack']['type']} | "
          f"defense={cfg['defense']['name']} {cfg['defense']['addons']} | secagg={cfg['secagg']['enabled']} | seed={cfg['run']['seed']}")
    res = _run_fl(cfg, ctx.dir)
    ctx.save_json("fl_result.json", res)
    fin = res.get("final", {})
    ctx.set(rounds_done=res["rounds_done"], **{f"final_{sp}_{k}": v.get(k) for sp, v in fin.items()
                                              for k in ("mase_daily_mean", "asr_mean") if v.get(k) is not None})
    shown = {k: v for k, v in res.items() if k != "series"}
    if isinstance(shown.get("detection"), dict):
        shown["detection"] = {k: v for k, v in shown["detection"].items() if k != "unit_scores"}
    print(json.dumps(shown, indent=2))


def cmd_equivalence(a, ctx):
    """Run the same configuration with the local and the Flower engine and compare the trajectories."""
    import copy
    from burstguard.config import validate
    cfg, _ = _cfg(a, ctx)
    out = {}
    for eng in ("local", "flower"):
        c = copy.deepcopy(cfg)
        c["engine"]["name"] = eng
        validate(c)
        sub = ctx.dir / f"engine_{eng}"; sub.mkdir()
        t0 = time.time()
        out[eng] = _run_fl(c, sub)
        out[eng]["wall_total_s"] = round(time.time() - t0, 1)
    rows = []
    for sp in out["local"]["final"]:
        for k in ("mase_daily_mean", "asr_mean", "mae_mean"):
            x, y = out["local"]["final"][sp].get(k), out["flower"]["final"].get(sp, {}).get(k)
            if x is not None and y is not None:
                rows.append({"split": sp, "metric": k, "local": x, "flower": y, "abs_diff": abs(x - y),
                             "rel_diff": abs(x - y) / (abs(y) + 1e-12)})
    ctx.save_json("equivalence.json", {"results": out, "comparison": rows})
    worst = max((r["rel_diff"] for r in rows), default=float("nan"))
    for r in rows:
        print(f"  {r['split']:8s} {r['metric']:16s} local={r['local']:.5f} flower={r['flower']:.5f} rel.diff={r['rel_diff']:.2e}")
    print(f"max relative difference: {worst:.2e} (secure aggregation adds quantisation noise; without it both "
          f"engines should agree up to floating point)")
    ctx.set(max_rel_diff=worst, wall_local=out["local"]["wall_total_s"], wall_flower=out["flower"]["wall_total_s"])


def cmd_grid(a, ctx):
    from burstguard.experiments import grid as G
    if a.gcmd == "run":
        res = G.run_grid(a.file, a.shard, quick=a.quick, retry_failed=not a.no_retry, max_jobs=a.max_jobs,
                         extra=G.parse_kv(a.extra), tag=a.tag,
                         seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None, wait=not a.no_wait)
        ctx.set(**res)
        if res["ran_failed"]:
            raise SystemExit(f"{res['ran_failed']} job(s) failed - see results/<...>/<job>/job.log and result.json")
    elif a.gcmd == "status":
        w = [float(x) for x in a.weights.split(",")] if a.weights else None
        st = G.grid_status(a.file, quick=a.quick, n_shards=a.shards, extra=G.parse_kv(a.extra), tag=a.tag, weights=w)
        print(json.dumps(st, indent=2))
        ctx.save_json("grid_status.json", st)
    elif a.gcmd == "list":
        g = G.load_grid(a.file)
        jobs = G.expand(g, quick=a.quick)
        if a.shard:
            k, n, w = G.parse_shard(a.shard)
            sel = G.select_shard(g, jobs, k, n, w, a.quick)
        else:
            sel = jobs
        for j in sel:
            print(f"{j['id']}  cost={j['cost']:6.0f}  {j['label']}")
        print(f"{len(jobs)} jobs")


def cmd_analyze(a, ctx):
    from burstguard.experiments import analyze as AN
    from burstguard.experiments import grid as G
    if a.target == "all":  # every grid of this version that has results
        done, missing = [], []
        for f in sorted((ROOT / "experiments").glob("*.toml")):
            g = G.load_grid(f)
            gdir = G.grid_dir(g, quick=a.quick)
            if not gdir.exists():
                missing.append(g["grid"]["name"]); continue
            try:
                AN.analyze(gdir, split=a.split or g.get("analysis", {}).get("split", "test"),
                           reference=g.get("analysis", {}).get("reference"), figures=not a.no_figures)
                done.append(g["grid"]["name"])
            except RuntimeError as e:
                print(f"{g['grid']['name']}: {e}"); missing.append(g["grid"]["name"])
        print(f"analysed: {', '.join(done) or '-'}; no results yet: {', '.join(missing) or '-'}")
        ctx.set(analysed=done, missing=missing)
        return
    ref = None
    target = resolve(a.target)
    if target.suffix == ".toml":
        g = G.load_grid(target)
        gdir = G.grid_dir(g, quick=a.quick)
        ref = g.get("analysis", {}).get("reference")
        split = a.split or g.get("analysis", {}).get("split", "test")
    else:
        gdir, split = target, (a.split or "test")
    res = AN.analyze(gdir, split=split, reference=ref, figures=not a.no_figures)
    ctx.set(**res)
    print((gdir / "_analysis" / "summary.md").read_text())


def cmd_paper(a, ctx):
    from burstguard.experiments import paper as PA
    root = resolve(a.results) if a.results else RESULTS_DIR / VERSION_TAG
    ds = resolve(a.dataset) if a.dataset else None
    res = PA.export(root, out=resolve(a.out) if a.out else None, dataset=ds)
    ctx.set(out=res["out"], made=len(res["made"]), skipped=len(res["skipped"]))
    print((PA.Path(res["out"]) / "index.md").read_text())


def cmd_plan(a, ctx):
    from pathlib import Path as _P
    from burstguard.experiments import plan as PL
    names = a.grids or [n for t in (a.tier.split(",") if a.tier else ["calibration", "1", "2", "system", "generalization",
                                                                     "offset", "traffic_bg", "offset2", "traffic_bg2",
                                                                     "poison", "calibrated", "revision", "v0313", "v0314", "v0315", "v0316"])
                        for n in PL.TIERS.get(t, [])]
    files = [resolve(n) if n.endswith(".toml") else ROOT / "experiments" / f"{n}.toml" for n in names]
    files = [f for f in files if _P(f).exists()]
    weights = [float(x) for x in a.weights.split(",")] if a.weights else PL.read_weights_file()
    res = PL.plan(files, n_shards=a.shards, weights=weights)
    ctx.save_json("plan.json", res)
    ctx.set(total_left_hours_1vm=res["total_left_hours_1vm"], wall_hours_if_sharded=res["wall_hours_if_sharded"])


def cmd_calibrate(a, ctx):
    """Suggest BurstGuard thresholds from monitor-mode runs on clean federations (E0) and replay the decision rule."""
    import tomllib

    import numpy as np
    from burstguard.defenses.burstguard import replay_flags
    from burstguard.experiments import grid as G
    target = resolve(a.target)
    gdir = G.grid_dir(G.load_grid(target), quick=a.quick) if target.suffix == ".toml" else target
    acc: dict[str, list[float]] = {"score_last": [], "score_anchor": []}
    jobs = []
    by_round: dict[int, list[float]] = {}
    where = {}
    for it in (a.where or []):
        k, _, v = it.partition("=")
        where[k.strip()] = v.strip().strip("\"'")
    def job_value(jdir, ov, key):
        """--where value of a job: its override, else the value in its config.toml (e.g. data.path of a grid that
        does not vary it)."""
        if key in ov:
            return ov[key]
        cfgp = jdir / "config.toml"
        if not cfgp.exists():
            return None
        node = tomllib.loads(cfgp.read_text())
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    for m in sorted(gdir.glob("*/metrics.jsonl")):
        if where:   # only jobs whose overrides (or config values) match every --where KEY=VALUE
            rj = m.parent / "result.json"
            ov = json.loads(rj.read_text()).get("overrides", {}) if rj.exists() else {}
            if any(str(job_value(m.parent, ov, k)) != v for k, v in where.items()):
                continue
        series = []
        for line in m.read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "round" and r.get("burstguard") and not r.get("attack_active"):
                b = r["burstguard"]
                for k in acc:
                    acc[k] += [x for x in b.get(k, []) if x is not None]
                series.append(list(zip(b.get("score_last", []), b.get("score_anchor", []))))
                by_round.setdefault(int(r["round"]), []).extend(x for x in b.get("score_anchor", []) if x is not None)
        if series:
            cfgp = m.parent / "config.toml"
            bgc = tomllib.loads(cfgp.read_text())["burstguard"] if cfgp.exists() else {}
            jobs.append((m.parent.name, series, bgc))
    if not jobs:
        raise RuntimeError(f"no BurstGuard scores from clean rounds in {gdir}")
    out = {}
    for k, v in acc.items():
        if v:
            s = np.array(v)
            out[k] = {"n": int(s.size), **{f"p{p}": round(float(np.percentile(s, p)), 4) for p in (50, 90, 95, 99, 100)}}
    drift = {f"round {k}": round(float(np.median(v)), 4) for k, v in sorted(by_round.items()) if v}

    def replay(thr, thr_a, adaptive):
        n = f = 0
        for _, series, bgc in jobs:
            flags = replay_flags(series, thr, thr_a, float(bgc.get("mad_k", 5.0)), int(bgc.get("history", 5)), adaptive)
            n += sum(len(x) for x in flags); f += sum(sum(x) for x in flags)
        return round(f / max(n, 1), 4)

    mg = float(a.margin)
    sug = lambda v: None if v is None else round(mg * float(v), 3)  # noqa: E731 - margin over the clean maximum
    max_last = out.get("score_last", {}).get("p100")
    hist0 = int(jobs[0][2].get("history", 5))
    early = [x for _, series, _ in jobs for units in series[:hist0] for (_, x) in units if x is not None]
    max_early = float(np.max(early)) if early else None
    cur = jobs[0][2]
    cand = {
        "current config": (cur.get("threshold"), cur.get("threshold_anchor"), cur.get("anchor_adaptive", True)),
        f"suggested, adaptive anchor ({mg}x clean max)": (sug(max_last), sug(max_early), True),
        f"suggested, fixed anchor ({mg}x clean max)": (sug(max_last), sug(out.get("score_anchor", {}).get("p100")), False),
    }
    rep = {}
    for name, (t, ta, ad) in cand.items():
        if t is None or ta is None:
            continue
        rep[name] = {"threshold": t, "threshold_anchor": ta, "anchor_adaptive": ad, "replayed_fpr_units": replay(t, ta, ad)}
    res = {"jobs": len(jobs), "clean_unit_scores": out, "anchor_score_median_by_round": drift, "candidates": rep,
           "note": "set burstguard.threshold / threshold_anchor / anchor_adaptive in configs/base.toml on ALL VMs"}
    if a.write_thresholds:
        key = where.get("data.path")
        if not key:
            raise ValueError("--write-thresholds needs --where data.path=... (thresholds are stored per dataset)")
        # default: FIXED anchor threshold (margin x largest clean anchor score of the whole run). The adaptive
        # anchor rule follows any smooth ramp, including a slow covert attacker (X3: head, boost 3, offset 12), and
        # a floor taken from the first rounds gives false positives once benign LoRA drift grows (replay on X3, W=6)
        rule = "fixed" if a.anchor == "fixed" else "adaptive"
        best = rep.get(f"suggested, {rule} anchor ({mg}x clean max)")
        if not best:
            raise RuntimeError("no suggested thresholds (no clean rounds?)")
        import tomllib
        tf = resolve(a.write_thresholds)
        tab = tomllib.loads(tf.read_text()).get("thresholds", {}) if tf.exists() else {}
        tab[key] = {"threshold": best["threshold"], "threshold_anchor": best["threshold_anchor"],
                    "anchor_adaptive": best["anchor_adaptive"], "replayed_fpr_units": best["replayed_fpr_units"],
                    "margin": mg, "clean_jobs": len(jobs), "calibrated_from": gdir.name,
                    "written": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
        lines = ["# BurstGuard thresholds calibrated on clean monitor-mode rounds, one entry per dataset",
                 "# (burstguard calibrate <grid> --where data.path=... --write-thresholds <this file>)", ""]
        for k, v in sorted(tab.items()):
            lines.append(f"[thresholds.{json.dumps(k)}]")
            for kk, vv in v.items():
                lines.append(f"{kk} = {json.dumps(vv) if not isinstance(vv, bool) else str(vv).lower()}")
            lines.append("")
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text("\n".join(lines))
        res["written"] = {"file": str(tf), "data.path": key, **tab[key]}
        print(f"thresholds for {key}: threshold={best['threshold']} threshold_anchor={best['threshold_anchor']} "
              f"anchor_adaptive={str(best['anchor_adaptive']).lower()} (replayed FPR {best['replayed_fpr_units']}) "
              f"-> {tf}")
    print(json.dumps(res, indent=2))
    ctx.save_json("calibration.json", res)
    ctx.set(jobs=len(jobs), suggested_threshold=sug(max_last), suggested_threshold_anchor=sug(max_early))


def cmd_bg_check(a, ctx):
    """How often BurstGuard's relative-change floor binds, and whether it changes scores of finished internet runs."""
    from burstguard.experiments import bgcheck
    res = bgcheck.check(results=a.results or None, rho=a.rho, max_models=a.max_models)
    ctx.save_json("bg_check.json", res)
    ctx.set(traffic=list(res["traffic"]), models=len(res["models"]), max_abs_diff=res.get("max_abs_diff"))


def cmd_smoke(a, ctx):
    """End-to-end test without network or real data: synthetic data + randomly initialised TTM."""
    from burstguard import baselines
    from burstguard import model as M
    from burstguard.config import load_config
    from burstguard.data.store import save_dataset
    from burstguard.data.synthetic import generate
    from burstguard.sitedata import dataset_from_cfg

    steps = []

    def step(name, fn):
        t0 = time.time()
        print(f"\n===== SMOKE: {name}", flush=True)
        try:
            out = fn()
            steps.append({"step": name, "status": "PASS", "s": round(time.time() - t0, 1)})
            print(f"===== PASS: {name} ({time.time() - t0:.1f}s)", flush=True)
            return out
        except Exception as e:
            steps.append({"step": name, "status": "FAIL", "s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"})
            print(f"===== FAIL: {name}: {type(e).__name__}: {e}", flush=True)
            raise

    data_path = PROCESSED_DIR / "smoke_d1.npz"
    base = [f'data.path="{data_path}"', 'model.source="random"', "data.num_sites=6", "data.max_train_windows=192",
            "data.max_eval_windows=48", "fl.rounds=2", "fl.eval_every=1", f"engine.workers={a.workers}",
            f"sim.num_cpus={max(1, a.workers)}", "burstguard.steps=4", "burstguard.restarts=1",
            "burstguard.probe_windows=16", "attack.num_malicious=1"]
    runs = [
        ("local engine: clean, head", ['train.peft="head"']),
        ("local engine: attack=scale, defense=burstguard, head", ['train.peft="head"', 'attack.type="scale"', 'defense.name="burstguard"']),
        ("local engine: attack=poison, defense=flame, lora", ['train.peft="lora"', 'attack.type="poison"', 'defense.name="flame"']),
        ("local engine: grouped secagg (3 groups) + burstguard, head", ['train.peft="head"', 'attack.type="scale"', 'defense.name="burstguard"',
                                                                         "secagg.enabled=true", "secagg.groups=3"]),
        ("flower engine: secagg+ (real protocol), head", ['train.peft="head"', 'engine.name="flower"', "secagg.enabled=true"]),
        # v0.3.12 revision features
        ("v0.3.12: score-aware attacker vs norm bound + burstguard, shadow screens", [
            'train.peft="head"', 'attack.type="evasive"', 'defense.name="normbound_burstguard"', "attack.evasion_probes=4",
            'burstguard.shadow=[{name = "val", probe_split = "val"}, {name = "signed", signed = true, width_max = 12}]']),
        ("v0.3.12: threshold-riding + regroup ride, online calibration, dip trigger", [
            'train.peft="head"', 'attack.type="budget"', 'defense.name="burstguard"', "attack.regroup_ride=true",
            "attack.budget_steps=2", "burstguard.calibrate_rounds=1", "trigger.amplitude=-1.0"]),
        ("v0.3.12: aggregate clipping (CRFL-style), lora", ['train.peft="lora"', 'attack.type="poison"',
                                                                   'defense.name="aggclip"']),
        # v0.3.13 fixes
        ("v0.3.13: public calibration guard + oracle threshold-riding, regroup verification", [
            'train.peft="head"', 'attack.type="budget"', 'attack.budget_thresholds="oracle"', "attack.budget_steps=2",
            'defense.name="normbound_burstguard"', "burstguard.calibrate_rounds=1", 'burstguard.calibrate_guard="public"',
            "burstguard.regroup_verify=true"]),
    ]
    try:
        step("synthetic data", lambda: save_dataset(data_path, *generate(6, 2, seed=0)[:3]))
        cfg = load_config(None, base)
        step("model + head cache", lambda: _smoke_model(cfg))
        step("zero-shot", lambda: baselines.zeroshot(M.load_model(cfg)[0], dataset_from_cfg(cfg), cfg))
        for i, (name, extra) in enumerate(runs):
            c = load_config(None, base + extra + [f'run.name="smoke{i}"'])
            sub = ctx.dir / f"run{i}"; sub.mkdir()
            res = step(name, lambda c=c, sub=sub: _run_fl(c, sub))
            ctx.save_json(f"run{i}.json", res)
        step("revision grids R1-R18: every configuration valid, data and probe sources present", _smoke_revision_grids)
    finally:
        ctx.save_json("smoke_report.json", {"steps": steps})
        ctx.set(steps=steps, passed=sum(s["status"] == "PASS" for s in steps), total=len(steps))
        print("\n===== SMOKE SUMMARY")
        for s in steps:
            print(f"  [{s['status']}] {s['step']} ({s['s']}s)" + (f"  {s.get('error', '')}" if s["status"] != "PASS" else ""))


def _smoke_revision_grids():
    """v0.3.12: expand the R* grids, validate every job and check that every data / probe file exists (shared data
    missing on a machine would otherwise only show up in the middle of the night)."""
    from burstguard.config import load_config
    from burstguard.experiments import grid as G
    from burstguard.sitedata import _cached_dataset
    files = sorted((ROOT / "experiments").glob("R*.toml"))
    paths, n = set(), 0
    for f in files:
        g = G.load_grid(f)
        for j in G.expand(g):
            cfg = load_config(g["grid"]["base_config"], j["overrides"])
            n += 1
            paths.add((str(cfg["data"]["path"]), "train"))
            for sh in [cfg["burstguard"]] + list(cfg["burstguard"].get("shadow", []) or []):
                if sh.get("probe_data") or sh.get("probe_split", "train") != "train":
                    paths.add((str(sh.get("probe_data") or cfg["data"]["path"]), str(sh.get("probe_split", "train"))))
    missing = []
    for pth, split in sorted(paths):
        rp = resolve(pth)
        if not rp.exists():
            missing.append(str(rp)); continue
        ds = _cached_dataset(str(rp))
        a, b = ds.bounds.get(split, [0, 0])
        if b <= a:
            missing.append(f"{rp} has no {split!r} split")
    print(f"  {len(files)} grids, {n} jobs valid; data / probe sources: {sorted(p for p, _ in paths)}")
    if missing:
        raise FileNotFoundError("missing for the revision grids: " + "; ".join(missing)
                                + " - copy the processed datasets to this machine (bash scripts/cluster.sh push-data)")


def _smoke_model(cfg):
    from burstguard import model as M
    from burstguard.sitedata import dataset_from_cfg, site_windows
    m, info = M.load_model(cfg)
    X, _, _ = site_windows(dataset_from_cfg(cfg), 0, "train", cfg)
    rel = M.check_head_cache(M.configure_peft(m, cfg), X, int(cfg["data"]["horizon"]), cfg["model"]["freq"])
    print(f"  {info.revision}: {info.n_params:,} params; freq_prefix={info.freq_prefix}; head-cache rel.err={rel:.1e}")
    if rel > 1e-3:
        raise AssertionError(f"head cache deviates: {rel}")


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="burstguard", description=f"BurstGuard {VERSION_TAG}")
    p.add_argument("--version", action="version", version=f"burstguard {__version__}")
    p.add_argument("--nice", type=int, default=10, help="lower CPU priority of this process and its workers (0 = off)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="version and paths").set_defaults(fn=cmd_info)
    s = sub.add_parser("doctor", help="environment checks"); s.add_argument("--no-net", action="store_true")
    s.set_defaults(fn=cmd_doctor)

    d = sub.add_parser("data", help="datasets").add_subparsers(dest="dcmd", required=True)
    s = d.add_parser("synthetic", help="synthetic dataset for pipeline tests")
    s.add_argument("--out", default=SYNTH); s.add_argument("--sites", type=int, default=20)
    s.add_argument("--probe", type=int, default=8); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_data_synthetic, name="synthetic")
    s = d.add_parser("download", help="download Milan raw files from Harvard Dataverse into the shared dir")
    s.add_argument("--region", default="mi", choices=["mi", "tn"]); s.add_argument("--dest", default=str(RAW_DIR / "milan"))
    s.add_argument("--from", dest="date_from"); s.add_argument("--to", dest="date_to")
    s.add_argument("--doi", action="append"); s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_data_download, name="download")
    s = d.add_parser("guestbook", help="show the Dataverse guestbook questions and write a response template to fill in")
    s.add_argument("--region", default="mi", choices=["mi", "tn"])
    s.add_argument("--force", action="store_true", help="overwrite an existing response file with a fresh template")
    s.set_defaults(fn=cmd_data_guestbook)
    s = d.add_parser("prepare", help="raw Milan files -> processed dataset")
    s.add_argument("--raw", default=str(RAW_DIR / "milan"))
    s.add_argument("--activity", default="internet",
                   help="traffic type(s), comma list: internet (default, milan_d1.npz), sms (milan_sms_d1.npz), "
                        "call (milan_call_d1.npz) or all; several types are read in one pass")
    s.add_argument("--out", default="", help="output file (single --activity only; default shared:data/processed/...)")
    s.add_argument("--force", action="store_true", help="rebuild files that already exist")
    s.add_argument("--sites", type=int, default=20); s.add_argument("--probe", type=int, default=8)
    s.add_argument("--block", type=int, default=5); s.add_argument("--days", type=int, default=62)
    s.add_argument("--region", default="mi", choices=["mi", "tn"],
                   help="mi = Milan (default, 5x5-square blocks); tn = Trentino (v0.3.16: single squares chosen by "
                        "traffic -> trentino_d1.npz, trentino_sms_d1.npz)")
    s.add_argument("--stride", type=int, default=2, help="Trentino: use every stride-th square of the traffic ranking")
    s.set_defaults(fn=cmd_data_prepare, name="prepare")
    s = d.add_parser("info", help="describe a processed dataset"); s.add_argument("--path", default="shared:data/processed/milan_d1.npz")
    s.set_defaults(fn=cmd_data_info, name="info")

    for name, fn, hlp in (("model-download", cmd_model_download, "pre-download TTM-R2 into the shared HF cache"),
                          ("model-check", cmd_model_check, "load the model and time every fine-tuning mode"),
                          ("zeroshot", cmd_zeroshot, "seasonal naive + TTM zero-shot"),
                          ("local", cmd_local, "local-only fine-tuning baseline"),
                          ("central", cmd_central, "centralised fine-tuning baseline (upper bound)"),
                          ("fl", cmd_fl, "one federated run"),
                          ("equivalence", cmd_equivalence, "local engine vs Flower engine on the same config")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--config", "-c", default=None); s.add_argument("--set", action="append", default=[])
        s.add_argument("--seed", type=int, default=None)
        s.set_defaults(fn=fn)

    gp = sub.add_parser("grid", help="experiment grids").add_subparsers(dest="gcmd", required=True)
    s = gp.add_parser("run", help="run (a shard of) a grid, resuming finished jobs")
    s.add_argument("file"); s.add_argument("--shard", default="1/1",
                                            help="k/N (e.g. 2/5) or weighted k/N@w1,...,wN (e.g. 2/5@6,2,3,1,3)")
    s.add_argument("--quick", action="store_true"); s.add_argument("--no-retry", action="store_true")
    s.add_argument("--max-jobs", type=int, default=0)
    s.add_argument("--extra", action="append", default=[], help="TESTING ONLY: extra override KEY=VALUE for every job")
    s.add_argument("--tag", default="", help="TESTING ONLY: results go to results/<v>/<grid>_<tag>")
    s.add_argument("--seeds", default="", help="TESTING ONLY: comma list replacing the grid's seeds")
    s.add_argument("--no-wait", action="store_true", help="do not wait for missing calibrated thresholds "
                   "(burstguard.threshold_file); default: wait, log a line every 5 min")
    s.set_defaults(fn=cmd_grid, name="run")
    s = gp.add_parser("status", help="progress per shard"); s.add_argument("file")
    s.add_argument("--shards", type=int, default=1); s.add_argument("--quick", action="store_true")
    s.add_argument("--extra", action="append", default=[]); s.add_argument("--tag", default="")
    s.add_argument("--weights", default="", help="w1,...,wN for weighted shards (same as cluster.weights)")
    s.set_defaults(fn=cmd_grid, name="status")
    s = gp.add_parser("list", help="list jobs"); s.add_argument("file")
    s.add_argument("--shard", default="", help="k/N or k/N@w1,...,wN")
    s.add_argument("--quick", action="store_true"); s.set_defaults(fn=cmd_grid, name="list")

    s = sub.add_parser("analyze", help="tables, statistics and figures for a grid")
    s.add_argument("target", help="grid .toml, results directory, or 'all'"); s.add_argument("--split", default=None)
    s.add_argument("--quick", action="store_true"); s.add_argument("--no-figures", action="store_true")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("paper", help="paper tables/figures/numbers from all finished grids")
    s.add_argument("--results", default="", help="results root (default results/<version>)")
    s.add_argument("--dataset", default="shared:data/processed/milan_d1.npz")
    s.add_argument("--out", default="", help="output dir (default <results>/_paper)")
    s.set_defaults(fn=cmd_paper)
    s = sub.add_parser("plan", help="estimate remaining campaign time from finished jobs")
    s.add_argument("grids", nargs="*", help="grid names or files (default: all tiers)")
    s.add_argument("--tier", default="", help="comma list of tiers: pilot,calibration,1,2,system,"
                   "generalization,offset,traffic_bg,offset2,traffic_bg2,poison,calibrated,revision,v0313,v0314,v0315,v0316")
    s.add_argument("--shards", type=int, default=5)
    s.add_argument("--weights", default="", help="w1,...,wN shard weights (default: $BURSTGUARD_SHARED/cluster.weights "
                                                  "if it exists, else equal shards)")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("calibrate", help="BurstGuard threshold from monitor-mode clean runs")
    s.add_argument("target"); s.add_argument("--quick", action="store_true")
    s.add_argument("--margin", type=float, default=1.5, help="suggested threshold = margin x largest clean score")
    s.add_argument("--where", action="append", help="only jobs with this override, e.g. "
                   "--where data.path=shared:data/processed/milan_sms_d1.npz (repeatable)")
    s.add_argument("--write-thresholds", default="", help="store the suggested thresholds for --where data.path in "
                   "this TOML file (e.g. shared:bg_thresholds.toml; used by burstguard.threshold_file)")
    s.add_argument("--anchor", choices=["fixed", "adaptive"], default="fixed",
                   help="anchor rule written by --write-thresholds (default fixed: robust to slow ramps)")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("bg-check", help="diagnose BurstGuard's relative-change floor (traffic types, saved models)")
    s.add_argument("--results", default="", help="results dir with saved final models, e.g. ~/burstguard-v0.3.5/results/v0.3.5")
    s.add_argument("--rho", type=float, default=0.1); s.add_argument("--max-models", type=int, default=12)
    s.set_defaults(fn=cmd_bg_check, name="bg-check")

    s = sub.add_parser("smoke", help="end-to-end test without network / real data")
    s.add_argument("--workers", type=int, default=2); s.set_defaults(fn=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    prob = venv_problem()
    if prob:
        print(f"ERROR (burstguard {VERSION_TAG}): {prob}", file=sys.stderr)
        return 2
    if args.nice:
        try:
            os.nice(args.nice)
        except Exception:
            pass
    _quiet_libs()
    cmd = {"data": f"data-{getattr(args, 'dcmd', '')}", "grid": f"grid-{getattr(args, 'gcmd', '')}"}.get(args.cmd, args.cmd)
    name = getattr(args, "name", None)
    if getattr(args, "config", None):
        name = os.path.splitext(os.path.basename(args.config))[0]
    for ov in getattr(args, "set", None) or []:
        if ov.replace(" ", "").startswith("run.name="):
            name = ov.split("=", 1)[1].strip().strip("\"'") or name
    if getattr(args, "file", None):
        sh = str(getattr(args, "shard", "") or "").split("@")[0]  # weights are logged in the run, not the dir name
        name = os.path.splitext(os.path.basename(args.file))[0] + (f"_shard{sh.replace('/', 'of')}" if sh else "")
    try:
        with RunContext(cmd, name, ["burstguard", *argv]) as ctx:
            args.fn(args, ctx)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        return 130
    except Exception:
        return 1
    return 0
