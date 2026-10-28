"""Time planning for the experiment campaign: fit a per-round cost model on finished jobs, estimate what is left.

Two-part cost model for one FL job on a machine with w parallel workers (its shard weight):

    seconds = rounds * ( S + C * REF_WEIGHT / w )
    S  server part (machine-independent: the server runs single-threaded): s0[peft] (aggregation, evaluation,
       process communication) + s_bg * units * effort when BurstGuard screens
    C  client part with REF_WEIGHT = 4 workers: c[peft] * a (a = slow-down of attacked rounds: malicious clients
       train attack.local_epochs on extra poisoned windows)
  units   screened units per round: secagg groups, or 1 + (attacked rounds share) * regroup_groups when a flagged
          aggregate is re-screened in groups (enforce mode, on_flag = regroup)
  effort  BurstGuard search effort relative to the default (probe windows x restarts x directions x steps)
Baselines (local-only, centralised) and zero-shot run in one process: server part only.
P0 of v0.3.2 showed why two parts: head-only rounds are server-bound (client training 0.16 s, screening ~80 s on
every machine), LoRA rounds client-bound (12 s per client). Weights (workers) only scale the client part, so
head-only jobs can go to the 1-worker machines without slowing the campaign.
Measured values (result.json: wall_s, clients_wall_s_mean, workers) replace the priors; the output says which.
Job features come from the resolved configuration ("effective"), so keys that only the base config sets
(fl.rounds, attack.local_epochs, ...) are taken into account.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from burstguard.experiments import grid as G

REF_WEIGHT = 4.0            # client part measured as on a 4-worker machine (vm-e: 8 vCPU, ~8 GB)
# priors (P0 v0.3.2; replaced by measurements)
PRIOR_S0 = {"head": 5.0, "decoder": 5.0, "lora": 6.0, "full": 6.0}      # server s/round without screening
PRIOR_C = {"head": 1.5, "decoder": 30.0, "lora": 60.0, "full": 70.0}     # client s/round with 4 workers
PRIOR_BG = 9.0              # s per screened unit and round at the default search effort, one thread
PRIOR_ATTACK = 1.5          # attacked / clean client time per round (poison_frac 0.2, 5 local epochs)
RATIO_C = {"head": 0.025, "decoder": 0.5, "lora": 1.0, "full": 1.15}    # client cost relative to LoRA
KIND_FACTOR = {"local_only": 1.0, "centralized": 1.1}   # baselines: one process ~ 4 x the 4-worker client time
ZEROSHOT_S = 120.0
DEFAULT_EFFORT = 32 * (1 * 2 * (6 * 8 + 2) + 2)   # probe_windows 32, restarts 1, both directions, 8 steps

TIERS = {
    "pilot": ["P0_pilot"],
    "calibration": ["E0_calibration"],
    "1": ["E1_clean", "E2_attacks", "E3_defenses", "E7_secagg", "E9_ablation"],
    "2": ["E4_sensitivity", "E5_lifespan", "E6_adaptive"],
    "system": ["E8_system"],
    "generalization": ["X1_traffic_clean", "X2_traffic_defenses"],
    "offset": ["X3_trigger_offset"],
    "traffic_bg": ["X4_traffic_bg_monitor"],
    "traffic_bg2": ["X5_traffic_bg_enforce"],
    "offset2": ["X6_offset_calibrated"],
    "poison": ["X7_poison_defenses"],
    "calibrated": ["X8_bg_calibrated"],
    # v0.3.12 revision experiments, in priority order (R10_cost runs last, on the local machine only)
    "revision": ["R1_headline", "R2_horizon", "R3_adaptive", "R6_online_calibration", "R4_baselines", "R9_modes", "R7_out_of_family", "R8_regroup", "R5_regime", "R10_cost"],
    # v0.3.13 fixes: public calibration guard (calibration-period attacker), regroup verification (many attackers)
    "v0313": ["R11_calibration_guard", "R12_many_attackers"],
    # v0.3.14: the recommended configuration (public thresholds + regroup verification), no code change
    "v0314": ["R13_recommended"],
    # v0.3.15: offset-144 trigger with a one-day window, LoRA seeds 3-4; no code change
    "v0315": ["R14_offset_window", "R15_lora_seeds"],
    # v0.3.16 final round: poisoning gaps, partial participation, Trentino
    "v0316": ["R18_trentino", "R16_poison_gaps", "R17_participation"],
}


def _effort(ov: dict[str, Any]) -> float:
    pw = int(ov.get("burstguard.probe_windows", 32))
    r = max(1, int(ov.get("burstguard.restarts", 1)))
    k = int(ov.get("burstguard.steps", 8))
    dirs = 2 if ov.get("burstguard.both_directions", True) else 1
    return pw * (r * dirs * (6 * k + 2) + 2) / DEFAULT_EFFORT


def _features(ov: dict[str, Any], kind: str) -> dict[str, Any]:
    bg = "burstguard" in str(ov.get("defense.name", "")) or "burstguard" in str(ov.get("defense.addons", ""))
    units = int(ov.get("secagg.groups", 1)) if ov.get("secagg.enabled") and int(ov.get("secagg.groups", 1)) > 1 else 1
    rounds = int(ov.get("fl.rounds", 20))
    attacked = ov.get("attack.type", "none") != "none"
    if (bg and units == 1 and attacked and ov.get("burstguard.mode", "enforce") != "monitor"
            and ov.get("burstguard.on_flag", "regroup") == "regroup"):
        # flagged rounds are screened again in regroup_groups groups (assume every attacked round is flagged)
        start, end = int(ov.get("attack.start_round", 1)), int(ov.get("attack.end_round", 0)) or rounds + 1
        frac = max(0, min(end - 1, rounds) - start + 1) / max(rounds, 1)
        units = 1 + frac * int(ov.get("burstguard.regroup_groups", 5))
        if ov.get("burstguard.regroup_verify"):     # v0.3.13: one more aggregate screen per regrouped round
            units += frac
    # v0.3.12 shadow screens: one extra aggregate screen per shadow and round (effort ~ its probe windows)
    shadow = sum(int(x.get("probe_windows", ov.get("burstguard.probe_windows", 32))) / 32.0
                 for x in (ov.get("burstguard.shadow") or [])) if bg else 0.0
    # v0.3.13 public calibration: the server trains the pseudo-clients itself before round 1 (one-off cost)
    pub = {}
    if bg and ov.get("burstguard.calibrate_guard") == "public":
        pub = {"rounds": int(ov.get("burstguard.calibrate_public_rounds", 0) or 0)
                         or int(ov.get("burstguard.calibrate_rounds", 0) or 0),
               "clients": int(ov.get("burstguard.calibrate_public_clients", 0) or 0) or 8}
    return {"peft": ov.get("train.peft", "head"), "kind": kind, "rounds": rounds, "attacked": attacked,
            "bg": bg, "units": units, "effort": _effort(ov) if bg else 0.0, "engine": ov.get("engine.name", "local"),
            "shadow": shadow, "public": pub,
            "sites": int(ov.get("data.num_sites", 20) or 20),
            "participation": float(ov.get("fl.participation", 1.0) or 1.0)}


def collect_finished(results_root: Path) -> list[dict[str, Any]]:
    out = []
    for rj in results_root.glob("*/*/result.json"):
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        if r.get("status") != "success" or not r.get("wall_s") or "_quick" in rj.parent.parent.name:
            continue
        f = _features(r.get("effective") or r.get("overrides", {}), r.get("kind", "fl"))
        out.append({**f, "wall_s": float(r["wall_s"]), "grid": rj.parent.parent.name,
                    "clients_wall_s": r.get("clients_wall_s_mean"), "workers": r.get("workers")})
    return out


def _med(v: list[float]) -> float | None:
    return float(np.median(v)) if v else None


def fit(finished: list[dict[str, Any]]) -> dict[str, Any]:
    m: dict[str, Any] = {"s0": {}, "c": {}, "base": {}, "src": {}, "s_bg": PRIOR_BG, "s_bg_src": "prior",
                         "attack": PRIOR_ATTACK, "attack_src": "prior", "zeroshot": ZEROSHOT_S}
    fl = [j for j in finished if j["kind"] == "fl" and j["engine"] == "local" and j.get("clients_wall_s")
          and j.get("workers")]
    for peft in PRIOR_C:
        clean = [j for j in fl if j["peft"] == peft and not j["attacked"]]
        c = _med([j["clients_wall_s"] * j["workers"] / REF_WEIGHT for j in clean])
        s0 = _med([j["wall_s"] / j["rounds"] - j["clients_wall_s"] for j in clean if not j["bg"]])
        if c is not None:
            m["c"][peft] = c; m["src"][f"c/{peft}"] = f"measured (n={len(clean)})"
        if s0 is not None:
            m["s0"][peft] = max(0.5, s0); m["src"][f"s0/{peft}"] = "measured"
        if c is None:   # v0.3.16: only attacked runs of this mode -> their client time / the prior attack factor
            att_c = _med([j["clients_wall_s"] * j["workers"] / REF_WEIGHT / PRIOR_ATTACK
                          for j in fl if j["peft"] == peft and j["attacked"]])
            if att_c is not None and peft != "head":
                m["c"][peft] = att_c; m["src"][f"c/{peft}"] = "from attacked runs / prior attack factor"
    m["attack_by"] = {}
    for peft in PRIOR_C:
        att = [j["clients_wall_s"] * j["workers"] / REF_WEIGHT / client_round(m, peft) for j in fl
               if j["attacked"] and j["peft"] == peft and peft in m["c"]]
        if att:
            m["attack_by"][peft] = max(1.0, float(np.median(att)))
    if m["attack_by"]:
        m["attack"] = attack_factor(m, "lora")
        m["attack_src"] = "measured per mode: " + ", ".join(f"{k} x{v:.2f}" for k, v in m["attack_by"].items())
    extra = [(j["wall_s"] / j["rounds"] - j["clients_wall_s"] - server_round(m, j["peft"]))
             / max(j["units"] * j["effort"] + j.get("shadow", 0.0), 1e-6) for j in fl if j["bg"]]
    if extra:
        m["s_bg"] = max(1.0, float(np.median(extra))); m["s_bg_src"] = f"measured (n={len(extra)})"
    for kind in KIND_FACTOR:
        for peft in PRIOR_C:
            v = [j["wall_s"] / j["rounds"] for j in finished if j["kind"] == kind and j["peft"] == peft]
            if v:
                m["base"][(peft, kind)] = float(np.median(v))
    zs = [j["wall_s"] for j in finished if j["kind"] == "zeroshot"]
    if zs:
        m["zeroshot"] = float(np.median(zs))
    return m


def client_round(m: dict[str, Any], peft: str) -> float:
    """Client seconds per round with REF_WEIGHT workers (clean)."""
    c = m["c"]
    if peft in c:
        return c[peft]
    # derive from a measured mode that does the same kind of work (v0.3.16: never from head, whose client part is
    # tiny and mostly overhead - 1 head job gave LoRA 143 s/round instead of ~60)
    if peft != "head":
        for p2 in ("lora", "full", "decoder"):
            if p2 in c:
                return c[p2] / RATIO_C[p2] * RATIO_C.get(peft, 1.0)
    return PRIOR_C.get(peft, 60.0)


def attack_factor(m: dict[str, Any], peft: str) -> float:
    """Attacked / clean client time per round, measured per fine-tuning mode (v0.3.16): head attackers dominate the
    tiny head client part (x5), LoRA attackers do not (x1.4)."""
    by = m.get("attack_by") or {}
    return by.get(peft) or (by.get("lora") if peft in ("decoder", "full") else None) or PRIOR_ATTACK


def server_round(m: dict[str, Any], peft: str) -> float:
    s0 = m["s0"]
    if peft in s0:
        return s0[peft]
    return float(np.median(list(s0.values()))) if s0 else PRIOR_S0.get(peft, 6.0)


def job_parts(m: dict[str, Any], ov: dict[str, Any], kind: str) -> tuple[float, float]:
    """(server seconds, client seconds with REF_WEIGHT workers) of one job."""
    f = _features(ov, kind)
    if kind == "zeroshot":
        return m["zeroshot"], 0.0
    if kind in KIND_FACTOR:
        per = m["base"].get((f["peft"], kind), REF_WEIGHT * client_round(m, f["peft"]) * KIND_FACTOR[kind])
        return f["rounds"] * per, 0.0
    srv = f["rounds"] * (server_round(m, f["peft"]) + (m["s_bg"] * (f["units"] * f["effort"] + f.get("shadow", 0.0))
                                                       if f["bg"] else 0.0))
    cli = f["rounds"] * client_round(m, f["peft"]) * (attack_factor(m, f["peft"]) if f["attacked"] else 1.0) \
        * f.get("participation", 1.0)
    pub = f.get("public") or {}
    if pub:   # one client's training ~ client round x REF_WEIGHT / sites (simulation threads ignored: upper bound)
        srv += pub["rounds"] * (pub["clients"] * client_round(m, f["peft"]) * REF_WEIGHT / max(1, f.get("sites", 20))
                                + 2 * m["s_bg"] * f["effort"])
    if f["engine"] == "flower":
        srv, cli = srv * 1.5, cli * 1.5
    return srv, cli


def job_seconds(m: dict[str, Any], ov: dict[str, Any], kind: str, weight: float = REF_WEIGHT) -> float:
    srv, cli = job_parts(m, ov, kind)
    return srv + cli * REF_WEIGHT / weight


PRIOR_MODEL = fit([])


def read_weights_file(path: Path | None = None) -> list[float] | None:
    """Shard weights written by scripts/cluster.sh ($BURSTGUARD_SHARED/cluster.weights: "w1,w2,...,wN")."""
    from burstguard.paths import SHARED_DIR
    p = Path(path) if path else SHARED_DIR / "cluster.weights"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            w = [float(x) for x in line.split(",")]
            return w if w and all(x > 0 for x in w) else None
    return None


def plan(grid_files: list[Path], n_shards: int = 5, log=print, weights: list[float] | None = None) -> dict[str, Any]:
    from burstguard import VERSION_TAG
    from burstguard.paths import RESULTS_DIR
    if weights:
        n_shards = len(weights)
    w = list(weights) if weights else [REF_WEIGHT] * n_shards
    root = RESULTS_DIR / VERSION_TAG
    finished = collect_finished(root) if root.exists() else []
    model = fit(finished)
    log(f"cost model from {len(finished)} finished jobs in {root} (seconds per round; client part with "
        f"{REF_WEIGHT:g} workers, server part single-threaded):")
    for peft in ("head", "decoder", "lora", "full"):
        log(f"  {peft:8s} server {server_round(model, peft):6.1f} [{model['src'].get(f's0/{peft}', 'prior/derived')}]"
            f"   clients {client_round(model, peft):6.1f} [{model['src'].get(f'c/{peft}', 'prior/derived')}]")
    log(f"  attacked rounds: clients x (per mode)   [{model['attack_src']}]")
    log(f"  BurstGuard: {model['s_bg']:.1f} s per screened unit and round   [{model['s_bg_src']}]")
    out = {"model": {"s_bg": model["s_bg"], "attack": model["attack"], "s0": model["s0"], "c": model["c"]},
           "grids": [], "n_shards": n_shards, "weights": weights}
    if weights:
        log(f"  shard weights {','.join(f'{x:g}' for x in weights)} = parallel workers per machine: a job takes "
            f"server + clients x {REF_WEIGHT:g}/weight on its machine")
    tot_left = 0.0
    shard_left = [0.0] * n_shards
    equal_left = [0.0] * n_shards   # same machines, plain round-robin shards (for comparison)
    log(f"\n{'grid':18s} {'jobs':>5s} {'done':>5s} {'fail':>5s} {'left_h(4-worker VM)':>20s} "
        f"{'per-shard h (max of ' + str(n_shards) + ')':>24s}")
    for gf in grid_files:
        g = G.load_grid(gf)
        jobs = G.expand(g)
        gdir = G.grid_dir(g)
        done = fail = 0
        left = 0.0
        per_shard = [0.0] * n_shards
        for k in range(1, n_shards + 1):
            for j in G.select_shard(g, jobs, k, n_shards, weights):
                st = G.job_status(gdir / j["id"])
                if st == "success":
                    done += 1
                    continue
                fail += st == "failed"
                ov = j.get("effective", j["overrides"])
                left += job_seconds(model, ov, j["kind"])
                per_shard[k - 1] += job_seconds(model, ov, j["kind"], w[k - 1])
        if weights:
            for k in range(1, n_shards + 1):
                for j in G.shard(jobs, k, n_shards):
                    if G.job_status(gdir / j["id"]) != "success":
                        equal_left[k - 1] += job_seconds(model, j.get("effective", j["overrides"]), j["kind"], w[k - 1])
        for i in range(n_shards):
            shard_left[i] += per_shard[i]
        tot_left += left
        out["grids"].append({"grid": g["grid"]["name"], "jobs": len(jobs), "done": done, "failed": fail,
                             "left_hours_1vm": round(left / 3600, 1),
                             "left_hours_per_shard": [round(x / 3600, 1) for x in per_shard]})
        log(f"{g['grid']['name']:18s} {len(jobs):5d} {done:5d} {fail:5d} {left / 3600:20.1f} "
            f"{max(per_shard) / 3600:24.1f}")
    out["total_left_hours_1vm"] = round(tot_left / 3600, 1)
    out["wall_hours_if_sharded"] = round(max(shard_left) / 3600, 1) if shard_left else 0.0
    log(f"{'TOTAL':18s} {'':5s} {'':5s} {'':5s} {tot_left / 3600:20.1f} {max(shard_left) / 3600:24.1f}")
    if weights:
        out["per_shard_wall_hours"] = [round(x / 3600, 1) for x in shard_left]
        out["wall_hours_equal_shards_same_machines"] = round(max(equal_left) / 3600, 1)
        log(f"per-shard wall hours: {', '.join(f'{k + 1}: {x / 3600:.1f}' for k, x in enumerate(shard_left))}")
        log(f"(equal round-robin shards on the same machines would take ~{max(equal_left) / 3600:.0f} h)")
    log(f"\n=> about {out['wall_hours_if_sharded']:.0f} h wall time with {n_shards} VMs running these grids in sequence "
        f"(each VM its own {'weighted ' if weights else ''}shard).")
    return out
