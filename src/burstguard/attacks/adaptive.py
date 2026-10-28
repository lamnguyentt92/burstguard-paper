"""v0.3.12 adaptive attackers that know BurstGuard (code, public probe windows, anchor, deployed thresholds).

budget   threshold-riding: after malicious training, choose the largest boost whose ESTIMATED effect on the
         aggregate keeps both BurstGuard scores below safety x the deployed thresholds (bisection on log(boost)).
         The attacker estimates the candidate as theta_g + n_mal * boost * delta / n (other attackers act alike, the
         benign mean is unknown and ignored) and scores it with BurstGuard's own search on public probes.
evasive  score-aware training: before every local epoch, run BurstGuard's search on the current local model
         against both references (last global, anchor) and penalise the model's differential response to the
         worst bursts found (adversarial training against the detector), while keeping the backdoor loss.
         v0.3.13 attack.budget_thresholds = oracle: the attacker rides the thresholds the server enforces in this
         round (e.g. the public cap while BurstGuard calibrates) instead of the configured ones.
regroup_ride (any attack): equivocation in a regroup - the update sent to the second (group) secure aggregation
         is rescaled to stay below the GROUP thresholds (sqrt(G) x thresholds or explicit group thresholds).
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from burstguard import model as M


class AttackerTools:
    def __init__(self, cfg: dict[str, Any], model, theta0: np.ndarray, probe_X: np.ndarray, probe_sigma: np.ndarray):
        from burstguard.defenses.burstguard import BurstGuard
        at = cfg["attack"]
        acfg = copy.deepcopy(cfg)
        bgc = acfg["burstguard"]
        bgc.update(probe_windows=int(at["evasion_probes"]), calibrate_rounds=0, mode="enforce", signed=False,
                   best_iterate=False, threads=1)
        # different probe subset than the server's (the attacker only knows that the probes are public windows)
        rng = np.random.default_rng(4242)
        idx = rng.permutation(len(probe_X))
        self.bg = BurstGuard(acfg, model, probe_X[idx], probe_sigma[idx], theta0)
        self.cfg, self.theta0 = cfg, theta0
        self.freq = cfg["model"]["freq"]
        n = int(cfg["data"]["num_sites"])
        self.n = n
        self.n_mal = len(at["malicious_sites"]) or int(at["num_malicious"])

    # ---------------------------------------------------------------- deployed thresholds known to the attacker
    def thresholds(self, group: bool = False) -> tuple[float, float]:
        b = self.cfg["burstguard"]
        tl, ta = float(b["threshold"]), float(b["threshold_anchor"])
        if not group:
            return tl, ta
        if float(b.get("group_threshold", 0) or 0) > 0:
            return float(b["group_threshold"]), float(b.get("group_threshold_anchor", 0) or ta)
        rs = b.get("regroup_threshold_scale", 1.0)
        g = int(b["regroup_groups"])
        sc = float(np.sqrt(g)) if rs == "sqrt" else float(rs)
        return tl * sc, ta * sc

    def scores(self, theta_g: np.ndarray, cand: np.ndarray, seed: int) -> tuple[float, float]:
        return self.bg.score(cand, theta_g, seed=seed), self.bg.score(cand, self.theta0, seed=seed + 1)

    def budget_boost(self, theta_g: np.ndarray, delta: np.ndarray, group: bool, seed: int,
                     thresholds: list[float] | None = None) -> tuple[float, dict]:
        """Largest boost in [lo, hi] whose estimated candidate stays below safety x thresholds
        (`thresholds`: the server's current aggregate thresholds, oracle attacker of v0.3.13)."""
        at = self.cfg["attack"]
        safety = float(at["budget_safety"])
        hi = float(at["budget_max_boost"]) or self.n / max(1, self.n_mal)
        lo = 0.05
        tl, ta = self.thresholds(group)
        if thresholds is not None and not group:
            tl, ta = float(thresholds[0]), float(thresholds[1])
        if group:   # one attacker in a group of n / G clients
            denom, mult = self.n / int(self.cfg["burstguard"]["regroup_groups"]), 1.0
        else:
            denom, mult = float(self.n), float(self.n_mal)
        hist = []

        def ok(g: float) -> bool:
            cand = theta_g + mult * g * delta / denom
            sl, sa = self.scores(theta_g, cand, seed)
            hist.append((round(g, 4), round(sl, 4), round(sa, 4)))
            return sl < safety * tl and sa < safety * ta

        if ok(hi):
            g = hi
        elif not ok(lo):
            g = lo
        else:
            a, b = np.log(lo), np.log(hi)
            for _ in range(int(at["budget_steps"])):
                m = 0.5 * (a + b)
                if ok(float(np.exp(m))):
                    a = m
                else:
                    b = m
            g = float(np.exp(a))
        return float(g), {"budget_boost": float(g), "budget_group": bool(group), "budget_thresholds": [tl, ta],
                          "budget_trace": hist}

    # ---------------------------------------------------------------- score-aware training (evasive)
    def evasion_hooks(self, theta_g: np.ndarray, seed: int):
        w = float(self.cfg["attack"]["evasion_weight"])
        X = self.bg.X
        state: dict[str, Any] = {"refs": []}
        refs = [("last", theta_g), ("anchor", self.theta0)]

        def epoch_hook(model, ep: int) -> None:
            cand = M.get_vec(model)
            state["refs"] = []
            for k, (name, ref) in enumerate(refs):
                _, bursts, s = self.bg.search(cand, ref, seed=seed + 13 * ep + k)
                Xb = X.clone()
                Xb[:, X.shape[1] - bursts.shape[1]:, :] += torch.from_numpy(bursts)
                M.set_params_vec(self.bg._pr, ref)
                with torch.no_grad():
                    f0, f1 = M.forward(self.bg.ref, X, self.freq), M.forward(self.bg.ref, Xb, self.freq)
                    d_ref = s * ((f0 - f1) / (f0.abs() + 1e-3)).reshape(len(X), -1).mean(dim=1)
                state["refs"].append((Xb, s, d_ref))

        def extra_loss(model) -> torch.Tensor:
            pen = torch.zeros(())
            for Xb, s, d_ref in state["refs"]:
                f0, f1 = M.forward(model, X, self.freq), M.forward(model, Xb, self.freq)
                d = s * ((f0 - f1) / (f0.detach().abs() + 1e-3)).reshape(len(X), -1).mean(dim=1)
                pen = pen + torch.relu((d - d_ref).mean())
            return w * pen

        return epoch_hook, extra_loss
