"""Engine-independent federated logic shared by the local engine and the Flower engine.

SiteWorker   client side: holds one model + per-site windows/caches, runs a benign or simulated-malicious update.
ServerLogic  server side: secure-aggregation emulation, aggregation rule, BurstGuard / LoRA-reset add-ons,
             evaluation and per-round records.
v0.3.13: public calibration guard (defenses/public_calibration.py), regroup verification, oracle threshold-riding
attacker (attack.budget_thresholds = oracle).
All exchanged models are flat float32 vectors of the trainable parameters.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from burstguard import model as M
from burstguard.attacks import client as A
from burstguard.defenses import aggregators as AGG
from burstguard.defenses.secagg import make_groups, secure_mean
from burstguard.sitedata import Evaluator, bg_probe_windows, dataset_from_cfg, probe_windows, site_windows
from burstguard.train import local_train


def client_clip_norm(cfg: dict[str, Any]) -> float:
    """Clients clip their own delta when the defense needs it and the server cannot (secure aggregation)."""
    d = cfg["defense"]
    if d["aggregator"] == "clip_noise" and float(d["clip_norm"]) > 0 and cfg["secagg"]["enabled"]:
        return float(d["clip_norm"])
    return 0.0


def client_seed(cfg: dict[str, Any], rnd: int, site: int) -> int:
    return int(cfg["run"]["seed"]) * 100_003 + rnd * 101 + site


class SiteWorker:
    """One per process. Sites are loaded lazily; the model object is shared across sites."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        torch.set_num_threads(int(cfg["sim"]["torch_threads"]))
        base, _ = M.load_model(cfg, seed=int(cfg["run"]["seed"]))
        self.model = M.configure_peft(base, cfg)
        self.layout = M.layout(self.model)
        self.theta0 = M.get_vec(self.model)      # public pre-trained parameters (known to adaptive attackers)
        self.ds = dataset_from_cfg(cfg)
        self.sites: dict[int, dict] = {}
        self.H = int(cfg["data"]["horizon"])
        self._tools = None

    def tools(self):
        """Attacker-side BurstGuard on public probe windows (v0.3.12 adaptive attackers), built lazily."""
        if self._tools is None:
            from burstguard.attacks.adaptive import AttackerTools
            Xp, _, Sp = probe_windows(self.ds, self.cfg)
            self._tools = AttackerTools(self.cfg, self.model, self.theta0, Xp, Sp)
        return self._tools

    def _use_cache(self) -> bool:
        return self.cfg["train"]["peft"] == "head" and bool(self.cfg["train"]["head_cache"])

    def _site(self, sid: int) -> dict:
        if sid in self.sites:
            return self.sites[sid]
        X, Y, _ = site_windows(self.ds, sid, "train", self.cfg, seed=int(self.cfg["run"]["seed"]))
        if not len(X):
            raise RuntimeError(f"site {sid}: no training windows")
        st = {"X": X, "Y": Y, "scale": self.ds.site_scale(sid), "cache": None, "mal": None}
        if self._use_cache():
            rel = M.check_head_cache(self.model, X, self.H, self.cfg["model"]["freq"])
            if rel > 1e-3:
                raise RuntimeError(f"head cache deviates from the full forward pass (rel={rel:.2e}); "
                                   f"set train.head_cache=false")
            st["cache"] = M.build_head_cache(self.model, X, freq=self.cfg["model"]["freq"])
        self.sites[sid] = st
        return st

    def _mal(self, sid: int, st: dict) -> dict:
        if st["mal"] is None:
            Xm, Ym, n_pois = A.malicious_dataset(st["X"], st["Y"], st["scale"], self.cfg, sid)
            cache = M.build_head_cache(self.model, Xm, freq=self.cfg["model"]["freq"]) if self._use_cache() else None
            st["mal"] = {"X": Xm, "Y": Ym, "cache": cache, "n_poison": n_pois}
        return st["mal"]

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        if task["role"] != "malicious" or int(self.cfg.get("_malicious_threads", 1)) <= 1:
            return self._run(task)
        prev = torch.get_num_threads()     # stragglers: use the CPUs the finished benign workers left idle
        torch.set_num_threads(int(self.cfg["_malicious_threads"]))
        try:
            out = self._run(task)
            out["info"]["threads"] = int(self.cfg["_malicious_threads"])
            return out
        finally:
            torch.set_num_threads(prev)

    def _run(self, task: dict[str, Any]) -> dict[str, Any]:
        cfg, sid, rnd = self.cfg, int(task["site"]), int(task["round"])
        t0, c0 = time.time(), time.process_time()
        st = self._site(sid)
        theta_g = np.asarray(task["theta"], dtype=np.float32)
        malicious = task["role"] == "malicious"
        seed = client_seed(cfg, rnd, sid)
        info: dict[str, Any] = {"site": sid, "role": task["role"]}
        benign_norm = None
        at = cfg["attack"]
        if malicious and A.norm_bounded(at) and float(at["norm_bound"]) <= 0:
            M.set_vec(self.model, theta_g)
            local_train(self.model, st["X"], st["Y"], st["scale"], cfg, seed=seed, cache=st["cache"])
            benign_norm = float(np.linalg.norm(M.get_vec(self.model) - theta_g))
            info["benign_norm_est"] = benign_norm
        M.set_vec(self.model, theta_g)
        torch.manual_seed(seed)
        theta_regroup = None
        if malicious:
            mal = self._mal(sid, st)
            mask = A.neurotoxin_mask(task.get("prev_global_delta"), self.layout, float(at["neurotoxin_topk"])) \
                if at["type"] == "neurotoxin" else None
            acfg = {**cfg, "train": {**cfg["train"], "lr": float(cfg["train"]["lr"]) * float(at["lr_mult"])}}
            hook, extra = (None, None)
            if at["type"] == "evasive":
                hook, extra = self.tools().evasion_hooks(theta_g, seed)
                M.set_vec(self.model, theta_g)
            stats = local_train(self.model, mal["X"], mal["Y"], st["scale"], acfg, seed=seed, cache=mal["cache"],
                                grad_mask=mask, epochs=int(at["local_epochs"]) or None, extra_loss=extra,
                                epoch_hook=hook)
            info["n_poison"] = mal["n_poison"]
        else:
            stats = local_train(self.model, st["X"], st["Y"], st["scale"], cfg, seed=seed, cache=st["cache"])
        delta = M.get_vec(self.model) - theta_g
        if malicious:
            raw = delta.copy()
            if at["type"] == "budget":
                g, binfo = self.tools().budget_boost(theta_g, raw, group=False, seed=seed,
                                                     thresholds=task.get("bg_thresholds"))
                delta = raw * g
                info.update(binfo)
            delta, pinfo = A.postprocess(delta, cfg, benign_norm)
            info.update(pinfo)
            if at.get("regroup_ride"):
                gg, ginfo = self.tools().budget_boost(theta_g, raw, group=True, seed=seed + 7)
                theta_regroup = (theta_g + gg * raw).astype(np.float32)
                info.update({f"regroup_{k}": v for k, v in ginfo.items()})
        cc = client_clip_norm(cfg)
        if cc > 0:
            delta = AGG.clip_to(delta, cc)
        info.update(delta_norm=float(np.linalg.norm(delta)), train_loss=stats["train_loss"],
                    train_cpu_s=stats["train_cpu_s"], wall_s=time.time() - t0, cpu_s=time.process_time() - c0,
                    n_windows=int(len(st["X"])))
        out = {"site": sid, "theta": (theta_g + delta).astype(np.float32), "info": info}
        if theta_regroup is not None:
            out["theta_regroup"] = theta_regroup
        return out


class ServerLogic:
    def __init__(self, cfg: dict[str, Any], log=print):
        self.cfg, self.log = cfg, log
        self.ds = dataset_from_cfg(cfg)
        base, self.model_info = M.load_model(cfg, seed=int(cfg["run"]["seed"]))
        self.model = M.configure_peft(base, cfg)
        self.layout = M.layout(self.model)
        self.theta0 = M.get_vec(self.model)
        self.n_sites = int(cfg["data"]["num_sites"])
        self.malicious = A.select_malicious(cfg)
        self.benign = [s for s in range(self.n_sites) if s not in self.malicious]
        self.rng = np.random.default_rng(40_000 + int(cfg["run"]["seed"]))
        splits = sorted({cfg["fl"]["eval_split"], *cfg["fl"]["final_splits"]})
        self.evaluator = Evaluator(self.ds, cfg, self.benign, splits)
        d = cfg["defense"]
        self.addons = set(d["addons"])
        self.prev_delta: np.ndarray | None = None
        self.bg = None
        self._probe = None
        if "burstguard" in self.addons or "probe_screen" in self.addons or d["aggregator"] == "fltrust":
            self._probe = probe_windows(self.ds, cfg)
        self._ps_idx = None
        if "probe_screen" in self.addons:
            n = min(int(d["probe_n"]), len(self._probe[0]))
            self._ps_idx = np.sort(np.random.default_rng(66).choice(len(self._probe[0]), size=n, replace=False))
        if "burstguard" in self.addons:
            from burstguard.defenses.burstguard import BurstGuard
            bgc = cfg["burstguard"]
            pr = bg_probe_windows(self.ds, cfg) if (bgc.get("probe_data") or bgc.get("probe_split", "train") != "train") \
                else self._probe
            self.bg = BurstGuard(cfg, self.model, pr[0], pr[2], self.theta0)
        self.shadows = self._make_shadows() if self.bg is not None else []
        self.agg_norm_hist: list[float] = []   # aggclip: norms of accepted aggregates (visible under secure aggregation)
        self.consec_flags = 0                  # burstguard.max_consecutive_regroups
        self._public = None                    # v0.3.13 public calibration (None = not run yet)
        self._public_logged = False
        self.lora_reset = None
        if "lora_reset" in self.addons:
            from burstguard.defenses.lora_reset import LoraReset
            self.lora_reset = LoraReset(M.lora_segments(self.model), self.theta0, d["lora_reset_every"],
                                        d["lora_reset_frac"], d["lora_reset_cooldown"], int(cfg["run"]["seed"]))
        self._fl_cache = None

    def _make_shadows(self) -> list[tuple[str, Any]]:
        """burstguard.shadow: further BurstGuard configurations that score the same aggregate (log only)."""
        import copy
        from burstguard.defenses.burstguard import BurstGuard
        out = []
        for j, ov in enumerate(self.cfg["burstguard"].get("shadow", []) or []):
            ov = dict(ov)
            name = str(ov.pop("name", "") or f"s{j}")
            scfg = copy.deepcopy(self.cfg)
            scfg["burstguard"].update(ov)
            scfg["burstguard"].update(mode="monitor", calibrate_rounds=0, shadow=[], calibrate_guard="",
                                      regroup_verify=False)
            b = scfg["burstguard"]
            pr = bg_probe_windows(self.ds, scfg) if (b.get("probe_data") or b.get("probe_split", "train") != "train") \
                else self._probe
            out.append((name, BurstGuard(scfg, self.model, pr[0], pr[2], self.theta0)))
        return out

    def _shadow_screen(self, theta: np.ndarray, delta: np.ndarray, rnd: int) -> dict[str, Any]:
        res = {}
        for name, sb in self.shadows:
            _, inf = sb.screen(theta, [delta], rnd)
            res[name] = {"score_last": inf["score_last"][0], "score_anchor": inf["score_anchor"][0],
                         "cpu_s": round(inf["cpu_s"], 3), "wall_s": round(inf["wall_s"], 3)}
        return res

    # ---- schedule
    def role(self, site: int, rnd: int) -> str:
        return "malicious" if (site in self.malicious and A.attack_active(self.cfg, rnd)) else "benign"

    def ensure_public_calibration(self, rnd: int) -> None:
        """v0.3.13 (burstguard.calibrate_guard = public): run the server's clean shadow federation on its public probe
        areas once, before the first screen, and install the cap."""
        if self.bg is None or self._public is not None or self.cfg["burstguard"].get("calibrate_guard") != "public":
            return
        from burstguard.defenses.public_calibration import simulate
        self._public = simulate(self, self.bg, log=self.log)
        self.bg.set_public_cap(*self._public["cap"], rnd=rnd)

    def participants(self, rnd: int) -> list[int]:
        """v0.3.16 partial participation: fl.participation x num_sites sites per round (at least 2), sampled with
        their own RNG stream (the server's other streams do not change); 1.0 = every site (v0.3.15 behaviour)."""
        p = float(self.cfg["fl"].get("participation", 1.0) or 1.0)
        if p >= 1.0:
            return list(range(self.n_sites))
        m = max(2, int(round(p * self.n_sites)))
        rng = np.random.default_rng(70_000 + 1009 * int(self.cfg["run"]["seed"]) + int(rnd))
        return sorted(int(x) for x in rng.choice(self.n_sites, size=m, replace=False))

    def tasks(self, theta: np.ndarray, rnd: int) -> list[dict]:
        self.ensure_public_calibration(rnd)
        tasks = [{"site": s, "round": rnd, "theta": theta, "role": self.role(s, rnd),
                  "prev_global_delta": self.prev_delta} for s in self.participants(rnd)]
        at = self.cfg["attack"]
        if self.bg is not None and at["type"] == "budget" and at.get("budget_thresholds") == "oracle":
            thr = [float(self.bg.threshold()), float(self.bg.threshold_anchor())]   # what this round's screen uses
            for t in tasks:
                if t["role"] == "malicious":
                    t["bg_thresholds"] = thr
        return tasks

    # ---- FLTrust server update on public root data
    def _server_update(self, theta: np.ndarray) -> np.ndarray:
        X, Y, S = self._probe
        n = min(len(X), 512)
        rng = np.random.default_rng(55)
        idx = np.sort(rng.choice(len(X), size=n, replace=False))
        Xs, Ys, Ss = X[idx], Y[idx], S[idx]
        M.set_vec(self.model, theta)
        cache = None
        if self.cfg["train"]["peft"] == "head" and self.cfg["train"]["head_cache"]:
            if self._fl_cache is None:
                self._fl_cache = M.build_head_cache(self.model, Xs, freq=self.cfg["model"]["freq"])
            cache = self._fl_cache
        local_train(self.model, Xs, Ys, Ss, self.cfg, seed=77, cache=cache,
                    epochs=int(self.cfg["defense"]["fltrust_epochs"]))
        g0 = M.get_vec(self.model) - theta
        M.set_vec(self.model, theta)
        return g0

    # ---- probe-loss check (baseline screen that also works on the secure aggregate)
    def _probe_err(self, theta: np.ndarray) -> float:
        X, Y, S = self._probe
        idx = self._ps_idx
        M.set_vec(self.model, theta)
        P = M.predict(self.model, X[idx], freq=self.cfg["model"]["freq"])
        return float(np.mean(np.abs(P - Y[idx]).mean(axis=(1, 2)) / np.maximum(S[idx], 1e-6)))

    def _probe_screen(self, theta: np.ndarray, cand: np.ndarray) -> tuple[bool, dict]:
        t0 = time.process_time()
        e0, e1 = self._probe_err(theta), self._probe_err(cand)
        rel = e1 / max(e0, 1e-12) - 1.0
        tol = float(self.cfg["defense"]["probe_tol"])
        flag = bool(rel > tol)
        info = {"scores": [rel], "score_last": [rel], "score_anchor": [None], "threshold": tol, "flags": [flag],
                "would_flag": [flag], "cpu_s": time.process_time() - t0, "level": "aggregate",
                "probe_err_last": e0, "probe_err_cand": e1}
        return not flag, info

    # ---- BurstGuard follow-up on a flagged aggregate: random regrouping (adaptive group testing)
    def _regroup(self, rnd: int, theta: np.ndarray, results: list[dict], malicious: list[int]):
        cfg, sa = self.cfg, self.cfg["secagg"]
        G = max(2, min(int(cfg["burstguard"]["regroup_groups"]), len(results)))
        grp = make_groups(len(results), G, self.rng)
        units, groups = [], []
        for g in grp:
            # equivocation (attack.regroup_ride): a client may send a different update to the second aggregation
            thetas = [results[i].get("theta_regroup", results[i]["theta"]) for i in g]
            mt = secure_mean(thetas, cfg, self.rng)[0] if sa["enabled"] else np.mean(thetas, axis=0)
            units.append(mt - theta); groups.append([results[i]["site"] for i in g])
        b = cfg["burstguard"]
        gt = float(b.get("group_threshold", 0) or 0)
        override = (gt, float(b.get("group_threshold_anchor", 0) or 0) or self.bg.threshold_anchor()) if gt > 0 else None
        flags, info = self.bg.screen(theta, units, rnd * 7919 + 1, scale=self.bg.regroup_scale(len(grp)),
                                     override=override)
        keep = [i for i, f in enumerate(flags) if not f]
        mal = set(malicious)
        has = [any(s in mal for s in g) for g in groups]
        rinfo = {**info, "level": "regroup", "unit_sites": groups, "unit_has_malicious": has, "kept": keep,
                 "kept_has_malicious": any(has[i] for i in keep)}
        if not keep:
            return None, rinfo
        w = np.array([len(groups[i]) for i in keep], dtype=np.float64)
        delta = np.sum([wi * units[i] for wi, i in zip(w, keep)], axis=0) / w.sum()
        return delta.astype(np.float32), rinfo

    # ---- verified client norm bound (v0.3.12; ACORN / RoFL / ELSA emulated)
    def _norm_bound(self, theta: np.ndarray, results: list[dict]) -> tuple[list[dict], dict]:
        d = self.cfg["defense"]
        nb = [float(np.linalg.norm(r["theta"] - theta)) for r in results if r["info"]["role"] == "benign"]
        if d["norm_bound_mode"] == "oracle":   # tightest public bound that still admits every benign site
            tau = float(d["norm_bound_mult"]) * (max(nb) if nb else 0.0)
        else:
            tau = float(d["norm_bound"])
        out, clipped = [], {"benign": 0, "malicious": 0}
        for r in results:
            r2 = dict(r)
            for key in ("theta", "theta_regroup"):
                if key in r:
                    u = r[key] - theta
                    n = float(np.linalg.norm(u))
                    if tau > 0 and n > tau:
                        r2[key] = (theta + u * (tau / n)).astype(np.float32)
                        if key == "theta":
                            clipped[r["info"]["role"]] += 1
            out.append(r2)
        return out, {"tau": tau, "clipped": clipped}

    # ---- aggregation
    def aggregate(self, rnd: int, theta: np.ndarray, results: list[dict],
                  secure_mean_theta: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        t0 = time.process_time()
        sa, d = cfg["secagg"], cfg["defense"]
        mal_in = [r["site"] for r in results if r["info"]["role"] == "malicious"]
        # an attacked round needs an attacker in it (partial participation); with every site taking part this is
        # the same as attack_active
        rec: dict[str, Any] = {"round": rnd, "attack_active": A.attack_active(cfg, rnd) and bool(mal_in),
                               "malicious_in_round": mal_in}
        if float(cfg["fl"].get("participation", 1.0) or 1.0) < 1.0:
            rec["participants"] = [r["site"] for r in results]
        self.ensure_public_calibration(rnd)
        if self._public is not None and not self._public_logged:
            rec["public_calibration"], self._public_logged = self._public, True
        norms_b = [r["info"]["delta_norm"] for r in results if r["info"]["role"] == "benign"]
        norms_m = [r["info"]["delta_norm"] for r in results if r["info"]["role"] == "malicious"]
        rec["delta_norm_benign_mean"] = float(np.mean(norms_b)) if norms_b else None
        rec["delta_norm_malicious_mean"] = float(np.mean(norms_m)) if norms_m else None
        if d.get("norm_bound_mode"):
            results, rec["norm_bound"] = self._norm_bound(theta, results)
        groups: list[list[int]]
        if secure_mean_theta is not None:          # real secure aggregation (Flower): one unit
            units_theta, groups = [secure_mean_theta], [[r["site"] for r in results]]
        elif sa["enabled"]:                          # emulated (grouped) secure aggregation
            grp = make_groups(len(results), int(sa["groups"]), self.rng)
            units_theta, groups, clipped = [], [], 0
            for g in grp:
                mt, inf = secure_mean([results[i]["theta"] for i in g], cfg, self.rng)
                units_theta.append(mt); groups.append([results[i]["site"] for i in g]); clipped += inf["clipped_values"]
            rec["secagg_clipped_values"] = clipped
        else:
            units_theta, groups = [r["theta"] for r in results], [[r["site"]] for r in results]
        units = [u - theta for u in units_theta]
        rec["n_units"] = len(units)
        rec["unit_sites"] = groups
        rec["unit_has_malicious"] = [any(s in rec["malicious_in_round"] for s in g) for g in groups]
        keep = list(range(len(units)))
        if self.bg is not None and self.cfg["burstguard"]["screen_resolved"] == "units" and len(units) > 1:
            flags, binfo = self.bg.screen(theta, units, rnd)
            keep = [i for i, f in enumerate(flags) if not f]
            rec["burstguard"] = {**binfo, "level": "units"}
        if not keep:
            delta, ainfo = None, {"selected": []}
        else:
            kept = [units[i] for i in keep]
            if d["aggregator"] in ("krum", "multikrum") and len(kept) < 2 * int(d["krum_f"]) + 3:
                delta, ainfo = AGG.aggregate("fedavg", kept, {"cfg": cfg, "rng": self.rng})
                ainfo["fallback"] = "fedavg (too few units for krum)"
            else:
                ctx = {"cfg": cfg, "rng": self.rng, "server_update": lambda: self._server_update(theta),
                       "agg_norm_hist": self.agg_norm_hist}
                if d["aggregator"] == "aggclip":   # oracle bound (best case for the baseline)
                    bm = [r["theta"] - theta for r in results if r["info"]["role"] == "benign"]
                    ctx["benign_mean_norm"] = float(np.linalg.norm(np.mean(bm, axis=0))) if bm else None
                delta, ainfo = AGG.aggregate(d["aggregator"], kept, ctx)
            ainfo["selected"] = [keep[i] for i in ainfo.get("selected", [])]
        rec["aggregator"] = {k: v for k, v in ainfo.items() if k in ("selected", "clip_norm", "noise_std", "trim_k",
                                                                     "trust", "fallback", "labels", "agg_norm")}
        sel_units = set(rec["aggregator"].get("selected", []))
        rec["selected_has_malicious"] = any(rec["unit_has_malicious"][i] for i in sel_units)
        accepted = delta is not None
        rec["screened_has_malicious"] = rec["selected_has_malicious"]  # label of the aggregate-level screen
        if accepted and self.shadows:
            rec["shadow"] = self._shadow_screen(theta, delta, rnd)
        if accepted and self.bg is not None and self.cfg["burstguard"]["screen_resolved"] == "aggregate":
            flags, binfo = self.bg.screen(theta, [delta], rnd)
            rec["burstguard"] = {**binfo, "level": "aggregate"}
            accepted = not flags[0]
            self.consec_flags = 0 if accepted else self.consec_flags + 1
            kmax = int(self.cfg["burstguard"].get("max_consecutive_regroups", 0) or 0)
            rec["consecutive_flags"] = self.consec_flags
            # flagged aggregate: instead of discarding the whole round, re-aggregate the same updates in random
            # secure groups and keep the groups that pass (needs a second secure-aggregation pass; emulated here)
            if (not accepted and self.cfg["burstguard"]["on_flag"] == "regroup" and secure_mean_theta is None
                    and len(results) > 1 and (kmax <= 0 or self.consec_flags <= kmax)):
                d2, rinfo = self._regroup(rnd, theta, results, rec["malicious_in_round"])
                rec["burstguard_regroup"] = rinfo
                if d2 is not None and self.cfg["burstguard"].get("regroup_verify"):
                    # v0.3.13: the re-aggregated update must pass the aggregate-level screen (thresholds scaled to the
                    # clients it averages); otherwise the regroup did not remove the cause of the flag -> reject
                    n_kept = sum(len(rinfo["unit_sites"][i]) for i in rinfo["kept"])
                    vsc = float(np.sqrt(len(results) / max(1, n_kept)))
                    vflags, vinfo = self.bg.screen(theta, [d2], rnd * 7919 + 2, scale=vsc)
                    rec["burstguard_verify"] = {**vinfo, "level": "verify", "n_kept_clients": n_kept,
                                                "kept_has_malicious": rinfo["kept_has_malicious"]}
                    if vflags[0]:
                        d2 = None
                if d2 is not None:
                    delta, accepted = d2, True
                    rec["selected_has_malicious"] = rinfo["kept_has_malicious"]
        if accepted and self._ps_idx is not None:
            ok, pinfo = self._probe_screen(theta, theta + delta)
            rec["probe_screen"] = pinfo
            accepted = ok
        new_theta = theta + delta if accepted else theta.copy()
        if accepted and d["aggregator"] == "aggclip" and ainfo.get("agg_norm") is not None:
            self.agg_norm_hist.append(float(ainfo["agg_norm"]))
        if self.bg is not None and accepted:
            bi = rec.get("burstguard_verify") or rec.get("burstguard_regroup") or rec.get("burstguard", {})
            fl = bi.get("would_flag", [])  # monitor mode: thresholds evolve as if enforcing
            sc = float(bi.get("threshold_scale", 1.0) or 1.0)  # regroup units: back to the aggregate's scale
            self.bg.accept(new_theta, [s / sc for s, f in zip(bi.get("score_last", []), fl) if not f and s is not None],
                           [s / sc for s, f in zip(bi.get("score_anchor", []), fl) if not f and s is not None])
        if accepted:
            for _, sb in self.shadows:
                sb.last = new_theta.copy()
        if self.lora_reset is not None:
            new_theta, linfo = self.lora_reset.apply(new_theta, rnd)
            rec["lora_reset"] = linfo
        rec["accepted"] = bool(accepted)
        rec["server_defense_cpu_s"] = time.process_time() - t0
        self.prev_delta = (new_theta - theta) if accepted else self.prev_delta
        return new_theta.astype(np.float32), rec

    # ---- evaluation
    def evaluate(self, theta: np.ndarray, split: str) -> dict[str, Any]:
        M.set_vec(self.model, theta)
        return self.evaluator.evaluate(self.model, split, backdoor=True)
