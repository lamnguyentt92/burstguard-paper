"""Aggregate grid results into paper tables, statistics and figures.

For every configuration (all overrides except the seed):
  * metrics on the final evaluation (default split: test): clean MASE, ASR, AE, UPR on triggered inputs ...
  * 95% CI by stratified bootstrap (resample seeds, then sites within seeds), B = 2000
  * backdoor lifespan L50: rounds after the attack window until val ASR <= 50% of its in-window peak
  * BurstGuard detection: TPR/FPR over rounds and over screened units (an aggregate or a group aggregate),
    pooled ROC AUC of the unit scores, share of rounds where a malicious unit was aggregated
Paired comparisons against a reference configuration (grid [analysis] reference) use site-level values averaged
over seeds: Wilcoxon signed-rank test, Holm-Bonferroni correction across the table.
Outputs go to <grid dir>/_analysis/: summary.csv, summary.md, summary.tex, tests.csv, figures/*.png|pdf
(including burstguard_roc when both benign and malicious units were screened).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ["mase_daily", "asr", "asr_strict", "te", "ae", "upr_trig", "upr_clean", "mae", "smape",
           # v0.3.12: natural trigger-like surges and unmet demand (absent in results of older versions)
           "frac_natural", "upr_natural", "shortfall_trig", "shortfall_natural", "shortfall_clean"]


def _final_per_site(jdir: Path, split: str) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Per-site metric lists of the last evaluation on `split` (from metrics.jsonl or a baseline result)."""
    res = json.loads((jdir / "result.json").read_text())
    per: dict[str, list[float]] = {}
    rows = None
    if res.get("per_site"):
        rows = res["per_site"].get(split)
    elif (jdir / "metrics.jsonl").exists():
        last = None
        for line in (jdir / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "eval" and r.get("split") == split:
                last = r
        rows = last["per_site"] if last else None
    for r in rows or []:
        for m in METRICS:
            if m in r and r[m] is not None and np.isfinite(r[m]):
                per.setdefault(m, []).append(float(r[m]))
        per.setdefault("_site", []).append(r["site"])
    return per, res


def _val_series(jdir: Path, metric: str = "asr_mean") -> tuple[list[int], list[float]]:
    """(rounds, values) of one metric of the validation evaluations of a job (metrics.jsonl)."""
    xs, ys = [], []
    p = jdir / "metrics.jsonl"
    if not p.exists():
        return xs, ys
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "eval" and r.get("split") == "val" and r.get(metric) is not None:
            xs.append(int(r["round"])); ys.append(float(r[metric]))
    return xs, ys


def _val_asr_series(jdir: Path) -> tuple[list[int], list[float]]:
    return _val_series(jdir, "asr_mean")


def lifespan(xs: list[int], ys: list[float], start: int, end: int, rounds: int,
             relative: bool = False, min_effect: float = 0.05) -> tuple[float | None, bool]:
    """L50 in rounds after the attack stops; (value, censored).
    relative=True (UPR_trig, > 0 without any attack): measured on the excess over the pre-attack level, and 0 when
    the in-window excess stays below `min_effect` (no backdoor to outlive)."""
    if end <= 0 or not xs:
        return None, False
    inwin = [y for x, y in zip(xs, ys) if start <= x <= end]
    if not inwin:
        return None, False
    base = 0.0
    if relative:
        pre = [y for x, y in zip(xs, ys) if x < start]
        base = float(np.mean(pre)) if pre else float(ys[0])
    peak = max(inwin) - base
    if peak <= (min_effect if relative else 0.0):
        return 0.0, False
    for x, y in zip(xs, ys):
        if x > end and y - base <= 0.5 * peak:
            return float(x - end), False
    return float(rounds - end), True


def bootstrap_ci(mat: list[list[float]], B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mat: list over seeds of per-site values. Returns (mean, lo, hi)."""
    mat = [np.asarray(m, dtype=float) for m in mat if len(m)]
    if not mat:
        return (math.nan, math.nan, math.nan)
    mean = float(np.mean([m.mean() for m in mat]))
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for b in range(B):
        picks = rng.integers(0, len(mat), size=len(mat))
        vals = [mat[i][rng.integers(0, len(mat[i]), size=len(mat[i]))].mean() for i in picks]
        bs[b] = np.mean(vals)
    return mean, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj.tolist()


_LEGACY_EXP_CODE = {"0.3.0": "0.3.0", "0.3.1": "0.3.0", "0.3.2": "0.3.0"}


def exp_code(res: dict[str, Any]) -> str:
    """Experiment-code version of a job result (results before 0.3.3 carry only the release version)."""
    if res.get("exp_code"):
        return str(res["exp_code"])
    v = str(res.get("version", "?"))
    return _LEGACY_EXP_CODE.get(v, v)


def build_rows(gdir: Path, split: str = "test", log=print) -> tuple[list[dict], list[str], list[dict]]:
    """Group successful jobs of a grid by configuration (all overrides except seed) and compute the statistics.
    Returns (rows, varying keys, jobs)."""
    jobs = []
    for jd in sorted(p for p in gdir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (jd / "result.json").exists():
            continue
        per, res = _final_per_site(jd, split)
        if res.get("status") != "success":
            continue
        jobs.append({"dir": jd, "res": res, "per": per})
    if not jobs:
        raise RuntimeError(f"no successful jobs with results in {gdir}")
    codes = sorted({exp_code(j["res"]) for j in jobs})
    if len(codes) > 1:
        vers = sorted({j["res"].get("version", "?") for j in jobs})
        log(f"WARNING: results from different experiment code are mixed in {gdir}: exp_code {codes} "
            f"(versions {vers})")
    shas = sorted({str(j["res"].get("base_config_sha")) for j in jobs})
    if len(shas) > 1:
        log(f"WARNING: jobs in {gdir} were run with {len(shas)} different base configs ({', '.join(shas)}); "
            "make configs/base.toml identical on all VMs and re-run the affected jobs")
    keys = sorted({k for j in jobs for k in j["res"]["overrides"] if k not in ("run.seed", "run.name")})
    varying = [k for k in keys if len({json.dumps(j["res"]["overrides"].get(k), sort_keys=True) for j in jobs}) > 1]
    groups: dict[str, list[dict]] = {}
    for j in jobs:
        ck = json.dumps({k: j["res"]["overrides"].get(k) for k in varying}, sort_keys=True)
        groups.setdefault(ck, []).append(j)
    rows = []
    for ck, js in sorted(groups.items()):
        row: dict[str, Any] = {**json.loads(ck), "n_seeds": len(js)}
        for m in METRICS:
            mat = [j["per"].get(m, []) for j in js]
            if any(len(x) for x in mat):
                mu, lo, hi = bootstrap_ci(mat)
                row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mu, lo, hi
        ov = js[0]["res"]["overrides"]
        for metric, col in (("asr_mean", "lifespan_l50"), ("upr_trig_mean", "lifespan_upr_l50")):
            ls = []
            for j in js:
                xs, ys = _val_series(j["dir"], metric)
                eff = j["res"].get("effective") or {}
                v, cens = lifespan(xs, ys, int(ov.get("attack.start_round", 1)), int(ov.get("attack.end_round", 0)),
                                   int(ov.get("fl.rounds", eff.get("fl.rounds", 20))), relative=metric != "asr_mean")
                if v is not None:
                    ls.append((v, cens))
            if ls:   # v0.3.9: also on UPR_trig (the covert backdoor's strict ASR is ~0, its under-forecast is not)
                row[col] = float(np.median([v for v, _ in ls]))
                row[col.replace("_l50", "_censored")] = sum(c for _, c in ls)
        dets = [j["res"].get("detection") or {} for j in js]
        for k in ("tpr_rounds", "fpr_rounds", "score_max_clean", "score_max_attack", "screen_cpu_s_mean",
                  "regroup_rounds", "regroup_tpr_groups", "regroup_fpr_groups", "regroup_kept_malicious_rate"):
            vals = [d[k] for d in dets if d.get(k) is not None]
            if vals:
                row[k] = float(np.mean(vals))
        units = [u for d in dets for u in (d.get("unit_scores") or [])]
        if units:
            from burstguard.engines.common import auc
            pos = [u for u in units if u[4]]
            neg = [u for u in units if not u[4]]
            row["tpr_units"] = float(np.mean([u[5] for u in pos])) if pos else None
            row["fpr_units"] = float(np.mean([u[5] for u in neg])) if neg else None
            row["auc_units"] = auc([u[1] for u in units], [u[4] for u in units])
            row["n_units_pos"], row["n_units_neg"] = len(pos), len(neg)
            row["_units"] = units
            # decision rules replayed offline on the same monitor-mode rounds (engines.common.rule_flags)
            if all(d.get("rule_flags") for d in dets if d.get("unit_scores")):
                from burstguard.engines.common import RULES
                rf = {name: [f for d in dets for f in (d.get("rule_flags") or {}).get(name, [])] for name in RULES}
                if all(len(v) == len(units) for v in rf.values()):
                    row["_rule_flags"] = rf
                    for name, fl in rf.items():
                        fp = [f for f, u in zip(fl, units) if u[4]]
                        fn = [f for f, u in zip(fl, units) if not u[4]]
                        row[f"rule_{name}_tpr_units"] = float(np.mean(fp)) if fp else None
                        row[f"rule_{name}_fpr_units"] = float(np.mean(fn)) if fn else None
                ag = [d["rule_replay_agree"] for d in dets if d.get("rule_replay_agree") is not None]
                if ag:
                    row["rule_replay_agree"] = float(np.mean(ag))
        kept = [(j["res"].get("filtering") or {}).get("malicious_kept_rate") for j in js]
        kept = [k for k in kept if k is not None]
        if kept:
            row["malicious_kept_rate"] = float(np.mean(kept))
        for k in ("round_wall_s_mean", "server_defense_cpu_s_mean", "update_bytes", "client_train_cpu_s_mean",
                  "wall_s"):
            vals = [j["res"].get(k) for j in js if j["res"].get(k) is not None]
            if vals:
                row[k] = float(np.mean(vals))
        row["_jobs"] = js
        rows.append(row)
    _detection_vs_clean(rows, varying)
    return rows, varying, jobs


def analyze(gdir: Path, split: str = "test", reference: dict[str, Any] | None = None,
            figures: bool = True, log=print) -> dict[str, Any]:
    rows, varying, jobs = build_rows(gdir, split, log)
    tests = _paired_tests(rows, varying, reference) if reference else []
    out = gdir / "_analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cols = varying + ["n_seeds"] + [c for m in METRICS for c in (m, f"{m}_lo", f"{m}_hi")] + \
        ["lifespan_l50", "lifespan_censored", "lifespan_upr_l50", "lifespan_upr_censored", "tpr_rounds", "fpr_rounds", "tpr_units", "fpr_units", "auc_units",
         "auc_vs_clean", "fpr_clean", "n_units_pos", "n_units_neg", "regroup_rounds", "regroup_tpr_groups",
         "regroup_fpr_groups", "regroup_kept_malicious_rate", "score_max_clean", "score_max_attack", "malicious_kept_rate", "screen_cpu_s_mean", "server_defense_cpu_s_mean",
         "round_wall_s_mean", "client_train_cpu_s_mean", "update_bytes", "wall_s", "rule_replay_agree"] + \
        [f"rule_{n}_{m}" for n in ("last", "anchor", "fixed_anchor")
         for m in ("tpr_units", "fpr_units", "auc_vs_clean", "fpr_clean")]
    cols = [c for c in cols if any(c in r for r in rows)]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    (out / "summary.md").write_text(_markdown(rows, varying, split))
    (out / "summary.tex").write_text(_latex(rows, varying, split))
    if tests:
        with open(out / "tests.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
            w.writeheader(); w.writerows(tests)
    if figures:
        _figures(rows, varying, out / "figures", split)
    log(f"analysis: {len(jobs)} jobs, {len(rows)} configurations -> {out}")
    return {"jobs": len(jobs), "configs": len(rows), "out": str(out), "varying": varying, "tests": len(tests)}


def _detection_vs_clean(rows, varying):
    """For attacked configurations, pool their malicious units with the benign units of the matching clean
    configuration (attack.type = none, all other non-attack varying keys equal): auc_vs_clean, fpr_clean."""
    if "attack.type" not in varying:
        return
    from burstguard.engines.common import auc
    others = [k for k in varying if not k.startswith("attack.")]   # a clean federation has no attack parameters
    clean = {json.dumps({k: q.get(k) for k in others}, sort_keys=True): q for q in rows
             if q.get("attack.type") == "none" and q.get("_units")}
    for r in rows:
        if r.get("attack.type") == "none" or not r.get("_units"):
            continue
        q = clean.get(json.dumps({k: r.get(k) for k in others}, sort_keys=True))
        if q is None:
            continue
        pos = [u for u in r["_units"] if u[4]]
        neg = [u for u in q["_units"] if not u[4]]
        if pos and neg:
            r["auc_vs_clean"] = auc([u[1] for u in pos + neg], [1] * len(pos) + [0] * len(neg))
            r["fpr_clean"] = float(np.mean([u[5] for u in neg]))
            r["_roc"] = ([u[1] for u in pos + neg], [1] * len(pos) + [0] * len(neg))
            # single-reference rules: AUC of S_last / S_anchor alone; FPR of every replayed rule on the clean units
            for name, idx in (("last", 2), ("anchor", 3)):
                sp = [u[idx] for u in pos if u[idx] is not None]
                sn = [u[idx] for u in neg if u[idx] is not None]
                if sp and sn:
                    r[f"rule_{name}_auc_vs_clean"] = auc(sp + sn, [1] * len(sp) + [0] * len(sn))
            if r.get("_rule_flags") and q.get("_rule_flags"):
                for name, fl in q["_rule_flags"].items():
                    fn = [f for f, u in zip(fl, q["_units"]) if not u[4]]
                    if fn:
                        r[f"rule_{name}_fpr_clean"] = float(np.mean(fn))


def _paired_tests(rows, varying, reference):
    from scipy.stats import wilcoxon
    ref_key, ref_val = next(iter(reference.items()))
    tests = []
    for r in rows:
        if r.get(ref_key) == ref_val:
            continue
        match = [q for q in rows if q.get(ref_key) == ref_val and all(q.get(k) == r.get(k) for k in varying if k != ref_key)]
        if not match:
            continue
        q = match[0]
        for m in ("asr", "mase_daily"):
            a, b = _site_means(r["_jobs"], m), _site_means(q["_jobs"], m)
            common = sorted(set(a) & set(b))
            if len(common) < 6:
                continue
            x = np.array([a[s] for s in common]); y = np.array([b[s] for s in common])
            if np.allclose(x, y):
                p = 1.0
            else:
                p = float(wilcoxon(x, y, zero_method="zsplit").pvalue)
            tests.append({**{k: r.get(k) for k in varying}, "metric": m, "vs": f"{ref_key}={ref_val}",
                          "mean": float(x.mean()), "ref_mean": float(y.mean()), "diff": float((x - y).mean()),
                          "n_sites": len(common), "p": p})
    adj = holm([t["p"] for t in tests]) if tests else []
    for t, pa in zip(tests, adj):
        t["p_holm"] = pa
    return tests


def _site_means(js, m):
    acc: dict[int, list[float]] = {}
    for j in js:
        for s, v in zip(j["per"].get("_site", []), j["per"].get(m, [])):
            acc.setdefault(int(s), []).append(v)
    return {s: float(np.mean(v)) for s, v in acc.items()}


def _fmt(r, m, digits=3):
    if m not in r or r[m] is None or (isinstance(r[m], float) and math.isnan(r[m])):
        return "-"
    if f"{m}_lo" in r:
        return f"{r[m]:.{digits}f} [{r[m + '_lo']:.{digits}f}, {r[m + '_hi']:.{digits}f}]"
    return f"{r[m]:.{digits}f}" if isinstance(r[m], float) else str(r[m])


def _markdown(rows, varying, split):
    shown = [m for m in ("mase_daily", "asr", "te", "ae", "upr_trig", "lifespan_l50", "lifespan_upr_l50", "tpr_units", "fpr_units",
                         "auc_units", "auc_vs_clean", "fpr_clean", "malicious_kept_rate", "server_defense_cpu_s_mean",
                         "round_wall_s_mean")
             if any(r.get(m) is not None for r in rows)]
    head = "| " + " | ".join([k.split(".")[-1] for k in varying] + ["seeds"] + shown) + " |"
    sep = "|" + "---|" * (len(varying) + 1 + len(shown))
    lines = [f"Split: {split}. Values: mean [95% bootstrap CI] over seeds x sites.", "", head, sep]
    for r in rows:
        lines.append("| " + " | ".join([str(r.get(k)) for k in varying] + [str(r["n_seeds"])] + [_fmt(r, m) for m in shown]) + " |")
    rules = [f"rule_{n}_{m}" for n in ("last", "anchor", "fixed_anchor") for m in ("tpr_units", "auc_vs_clean", "fpr_clean")]
    rules = [c for c in rules if any(r.get(c) is not None for r in rows)]
    if rules:   # monitor-mode grids: other decision rules replayed on the same rounds
        lines += ["", "Decision rules replayed offline on the same monitor-mode rounds (last = S_last only, anchor = "
                  "S_anchor only, fixed_anchor = anchor_adaptive false); replay_agree = share of rounds where the "
                  "replayed deployed rule matches the logged decision (should be 1.000).", "",
                  "| " + " | ".join([k.split(".")[-1] for k in varying] + rules + ["replay_agree"]) + " |",
                  "|" + "---|" * (len(varying) + len(rules) + 1)]
        for r in rows:
            lines.append("| " + " | ".join([str(r.get(k)) for k in varying] + [_fmt(r, m, 3) for m in rules]
                                           + [_fmt(r, "rule_replay_agree", 3)]) + " |")
    return "\n".join(lines) + "\n"


def _latex(rows, varying, split):
    shown = [m for m in ("mase_daily", "asr", "te", "ae", "upr_trig") if any(m in r for r in rows)]
    names = {"mase_daily": "MASE", "asr": "ASR", "te": "TE", "ae": "AE", "upr_trig": "UPR$_{trig}$"}
    lines = ["\\begin{tabular}{" + "l" * len(varying) + "r" * len(shown) + "}", "\\toprule",
             " & ".join([k.split(".")[-1].replace("_", "\\_") for k in varying] + [names[m] for m in shown]) + " \\\\",
             "\\midrule"]
    for r in rows:
        cells = [str(r.get(k)).replace("_", "\\_") for k in varying]
        for m in shown:
            cells.append("-" if m not in r else f"{r[m]:.3f} {{\\scriptsize[{r[m + '_lo']:.2f},{r[m + '_hi']:.2f}]}}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"% split={split}; mean [95% CI] over seeds x sites"]
    return "\n".join(lines) + "\n"


def _figures(rows, varying, fdir: Path, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not varying:
        return
    x_key = next((k for k in ("defense.name", "attack.type", "train.peft", "secagg.groups") if k in varying), varying[0])
    others = [k for k in varying if k != x_key]
    panels: dict[str, list] = {}
    for r in rows:
        panels.setdefault(json.dumps({k: r.get(k) for k in others}, sort_keys=True), []).append(r)
    for m in ("asr", "mase_daily"):
        if not any(m in r for r in rows):
            continue
        n = len(panels)
        cols = min(3, n)
        fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.2 * cols, 3.0 * ((n + cols - 1) // cols)),
                                 squeeze=False)
        for ax, (pk, rs) in zip(axes.ravel(), sorted(panels.items())):
            rs = sorted(rs, key=lambda r: str(r.get(x_key)))
            xs = [str(r.get(x_key)) for r in rs]
            ys = [r.get(m, np.nan) for r in rs]
            err = [[r.get(m, np.nan) - r.get(m + "_lo", np.nan) for r in rs], [r.get(m + "_hi", np.nan) - r.get(m, np.nan) for r in rs]]
            ax.bar(range(len(xs)), ys, yerr=err, capsize=3, color="#4C72B0")
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=40, ha="right", fontsize=8)
            ax.set_title(", ".join(f"{k.split('.')[-1]}={v}" for k, v in json.loads(pk).items()) or "all", fontsize=8)
            ax.set_ylabel(m.upper() if m == "asr" else "MASE (daily)")
            if m == "asr":
                ax.set_ylim(0, 1)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fdir / f"{m}_by_{x_key.split('.')[-1]}.{ext}", dpi=150)
        plt.close(fig)
    # ASR over rounds (lifespan / dynamics)
    curves = [r for r in rows if r["_jobs"] and _val_asr_series(r["_jobs"][0]["dir"])[0]]
    if curves and any(any(_val_asr_series(j["dir"])[1]) for r in curves for j in r["_jobs"]):
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for r in curves:
            series = [_val_asr_series(j["dir"]) for j in r["_jobs"]]
            xs = series[0][0]
            ys = np.nanmean([s[1] for s in series if len(s[1]) == len(xs)], axis=0)
            ax.plot(xs, ys, marker=".", label=", ".join(f"{k.split('.')[-1]}={r.get(k)}" for k in varying), linewidth=1)
        ov = curves[0]["_jobs"][0]["res"]["overrides"]
        if int(ov.get("attack.end_round", 0)) > 0:
            ax.axvspan(int(ov.get("attack.start_round", 1)), int(ov["attack.end_round"]), color="red", alpha=0.08)
        ax.set_xlabel("round"); ax.set_ylabel("ASR (val, benign sites)"); ax.set_ylim(0, 1)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fdir / f"asr_over_rounds.{ext}", dpi=150)
        plt.close(fig)
    # BurstGuard ROC (unit level), only for configurations with both classes
    rocs = []
    for r in rows:
        if r.get("_roc"):
            rocs.append((r, r["_roc"], r["auc_vs_clean"], "vs clean"))
        elif r.get("auc_units") is not None:
            rocs.append((r, ([u[1] for u in r["_units"]], [u[4] for u in r["_units"]]), r["auc_units"], "in-run"))
    if rocs:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for r, (sc, yy), a, kind in rocs:
            fpr, tpr = roc_curve(sc, yy)
            lab = ", ".join(f"{k.split('.')[-1]}={r.get(k)}" for k in varying) or "all"
            ax.plot(fpr, tpr, linewidth=1, label=f"{lab} (AUC {a:.2f}, {kind})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("false positive rate (benign units)"); ax.set_ylabel("true positive rate (malicious units)")
        ax.legend(fontsize=5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fdir / f"burstguard_roc.{ext}", dpi=150)
        plt.close(fig)


def roc_curve(scores: list[float], labels: list[int]) -> tuple[list[float], list[float]]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(-s)
    s, y = s[order], y[order]
    P, N = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    fpr, tpr, tp, fp = [0.0], [0.0], 0, 0
    for i in range(len(s)):
        tp += y[i]; fp += 1 - y[i]
        if i == len(s) - 1 or s[i + 1] != s[i]:
            fpr.append(fp / N); tpr.append(tp / P)
    return fpr, tpr
