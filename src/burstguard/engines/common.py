"""Helpers shared by both engines: resource sizing, JSONL writer, run loop bookkeeping and result summary."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from burstguard import __version__

WORKER_RSS_GIB = 1.4   # measured ~1.3 GiB per worker (torch + transformers + TTM + site data)
RESERVED_GIB = 2.5     # driver/server process, OS and other services


def mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        return None
    return None


def cpus_available() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 2


def auto_workers(requested: int, n_sites: int, cpus_per_worker: float = 1.0) -> int:
    """0 = automatic: min(CPUs - 1, workers fitting in free RAM, number of sites)."""
    if int(requested) > 0:
        return int(requested)
    by_cpu = max(1, int((cpus_available() - 1) / cpus_per_worker))
    mem = mem_available_gib()
    by_mem = max(1, int((mem - RESERVED_GIB) / WORKER_RSS_GIB)) if mem is not None else by_cpu
    return max(1, min(by_cpu, by_mem, n_sites))


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, kind: str, **data: Any) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"v": __version__, "t": round(time.time(), 3), "kind": kind, **data},
                               default=_default) + "\n")


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def slim_eval(ev: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in ev.items() if k not in ("sites", "split")}


def auc(scores: list[float], labels: list[int]) -> float | None:
    """ROC AUC via the Mann-Whitney U statistic (ties count 1/2); None if only one class is present."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _client_cpu(rounds: list[dict]) -> float | None:
    """Mean local-training CPU seconds per benign client and round (edge-cost figure)."""
    v = [c.get("train_cpu_s") for r in rounds for c in (r.get("clients") or [])
         if c.get("role", "benign") == "benign" and c.get("train_cpu_s") is not None]
    return float(np.mean(v)) if v else None


RULES = ("last", "anchor", "fixed_anchor")


def rule_flags(run_dir: Path, per_round: list[list[tuple]]) -> dict[str, list[int]] | None:
    """Monitor-mode runs record S_last and S_anchor of every round and accept every aggregate, so other decision
    rules can be replayed exactly on the same rounds (E9 ablation without extra runs):
      last          reference = last only (S_last vs its adaptive threshold)
      anchor        reference = anchor only
      fixed_anchor  both references, fixed anchor threshold (anchor_adaptive = false)
      both          the deployed rule (should reproduce the logged would_flag: rule_replay_agree = 1)
    Returns {rule: [0/1 per unit, in unit_scores order]} or None (enforce mode, single reference, no config)."""
    p = Path(run_dir) / "config.toml"
    if not p.exists() or not per_round:
        return None
    try:
        import tomllib
        b = tomllib.loads(p.read_text()).get("burstguard", {})
        if b.get("mode") != "monitor" or b.get("reference", "both") != "both":
            return None
        from burstguard.defenses.burstguard import replay_flags
        thr, thr_a = float(b["threshold"]), float(b["threshold_anchor"])
        k, h, ad = float(b["mad_k"]), int(b["history"]), bool(b.get("anchor_adaptive", True))
    except Exception:
        return None
    inf = float("inf")
    rules = {"both": (thr, thr_a, ad), "last": (thr, inf, ad), "anchor": (inf, thr_a, ad),
             "fixed_anchor": (thr, thr_a, False)}
    return {name: [int(f) for fl in replay_flags(per_round, t, ta, k, h, a) for f in fl]
            for name, (t, ta, a) in rules.items()}


def summarize(run_dir: Path) -> dict[str, Any]:
    """Final metrics + series used by the analysis (also works on partial runs)."""
    rounds, evals, setup = [], [], {}
    p = run_dir / "metrics.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "setup":
                setup = r
            (rounds if r["kind"] == "round" else evals if r["kind"] == "eval" else []).append(r)
    final: dict[str, Any] = {}
    for e in evals:
        final[e["split"]] = {k: e.get(k) for k in e if k.endswith("_mean") or k.endswith("_median") or k == "round"}
    series = {"round": [], "asr": [], "mase": []}
    for e in evals:
        if e["split"] == (evals[0]["split"] if evals else "val") and e.get("asr_mean") is not None:
            series["round"].append(e["round"]); series["asr"].append(e.get("asr_mean")); series["mase"].append(e.get("mase_daily_mean"))
    for r in rounds:  # one screening record per round: BurstGuard, else the probe-loss baseline
        if not r.get("burstguard") and r.get("probe_screen"):
            r["burstguard"] = r["probe_screen"]
    bg = [r.get("burstguard") for r in rounds if r.get("burstguard")]
    att = [r for r in rounds if r.get("attack_active")]
    clean = [r for r in rounds if not r.get("attack_active")]
    det = {}
    if bg:
        def flagged(r):
            b = r.get("burstguard") or {}
            return any(b.get("would_flag", [])) if b else False
        det = {"tpr_rounds": float(np.mean([flagged(r) for r in att])) if att else None,
               "fpr_rounds": float(np.mean([flagged(r) for r in clean])) if clean else None,
               "score_max_clean": float(max(max(r["burstguard"]["scores"]) for r in clean if r.get("burstguard"))) if any(r.get("burstguard") for r in clean) else None,
               "score_max_attack": float(max(max(r["burstguard"]["scores"]) for r in att if r.get("burstguard"))) if any(r.get("burstguard") for r in att) else None,
               "screen_cpu_s_mean": float(np.mean([b["cpu_s"] for b in bg]))}
    # unit-level records for ROC/AUC: [round, combined score, score_last, score_anchor, label, would_flag]
    units, per_round = [], []
    for r in rounds:
        b = r.get("burstguard")
        if not b:
            continue
        if b.get("level") == "units":
            labels = [bool(x) and bool(r.get("attack_active")) for x in r.get("unit_has_malicious", [])]
        else:
            labels = [bool(r.get("screened_has_malicious", r.get("selected_has_malicious")))
                      and bool(r.get("attack_active"))]
        pr = []
        for i, y in enumerate(labels[:len(b.get("scores", []))]):
            sl, sa = b["score_last"][i], b["score_anchor"][i]
            units.append([r["round"], round(float(b["scores"][i]), 5), None if sl is None else round(float(sl), 5),
                          None if sa is None else round(float(sa), 5), int(y), int(bool(b["would_flag"][i]))])
            pr.append((sl, sa))
        per_round.append(pr)
    if units:
        pos = [u for u in units if u[4]]
        neg = [u for u in units if not u[4]]
        det["tpr_units"] = float(np.mean([u[5] for u in pos])) if pos else None
        det["fpr_units"] = float(np.mean([u[5] for u in neg])) if neg else None
        det["auc_units"] = auc([u[1] for u in units], [u[4] for u in units])
        det["unit_scores"] = units
        rf = rule_flags(run_dir, per_round)
        if rf:
            agree = [a == u[5] for a, u in zip(rf.pop("both"), units)]
            det["rule_replay_agree"] = float(np.mean(agree)) if agree else None
            det["rule_flags"] = rf
    rg = [r["burstguard_regroup"] for r in rounds if r.get("burstguard_regroup")]
    if rg:
        mg = [f for x in rg for f, h in zip(x["would_flag"], x["unit_has_malicious"]) if h]
        bgp = [f for x in rg for f, h in zip(x["would_flag"], x["unit_has_malicious"]) if not h]
        det["regroup_rounds"] = len(rg)
        det["regroup_tpr_groups"] = float(np.mean(mg)) if mg else None
        det["regroup_fpr_groups"] = float(np.mean(bgp)) if bgp else None
        det["regroup_kept_malicious_rate"] = float(np.mean([x["kept_has_malicious"] for x in rg]))
    vf = [r["burstguard_verify"] for r in rounds if r.get("burstguard_verify")]   # v0.3.13
    if vf:
        det["verify_rounds"] = len(vf)
        det["verify_rejected"] = int(sum(bool(x["would_flag"][0]) for x in vf))
        det["verify_rejected_malicious"] = int(sum(bool(x["would_flag"][0]) and x["kept_has_malicious"] for x in vf))
        det["verify_passed_malicious"] = int(sum(not x["would_flag"][0] and x["kept_has_malicious"] for x in vf))
    pc = next((r["public_calibration"] for r in rounds if r.get("public_calibration")), None)
    if pc:
        det["public_calibration"] = {k: pc[k] for k in ("rounds", "clients", "areas", "margin", "cap", "wall_s",
                                                         "cpu_s", "score_last", "score_anchor") if k in pc}
    cal = [(r.get("burstguard") or {}).get("calibrated") for r in rounds]
    cal = [c for c in cal if c]
    if cal:
        det["calibrated"] = cal[-1]
    sel = [r for r in att if r.get("malicious_in_round")]
    filt = {"malicious_kept_rate": float(np.mean([r.get("selected_has_malicious", False) and r.get("accepted", True) for r in sel])) if sel else None,
            "accepted_rate": float(np.mean([bool(r.get("accepted", True)) for r in rounds])) if rounds else None}
    return {"rounds_done": len(rounds), "final": final, "series": series, "detection": det, "filtering": filt,
            "round_wall_s_mean": float(np.mean([r["round_wall_s"] for r in rounds])) if rounds else None,
            "server_defense_cpu_s_mean": float(np.mean([r["server_defense_cpu_s"] for r in rounds])) if rounds else None,
            "client_train_cpu_s_mean": _client_cpu(rounds),
            "clients_wall_s_mean": float(np.mean([r["clients_wall_s"] for r in rounds if r.get("clients_wall_s") is not None]))
            if any(r.get("clients_wall_s") is not None for r in rounds) else None,
            "workers": setup.get("workers"),
            "update_bytes": rounds[-1].get("update_bytes") if rounds else None}
