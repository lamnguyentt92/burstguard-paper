"""BurstGuard: screening of the (securely) aggregated model for burst-triggered forecast shortcuts.

For a candidate global model c and a reference r (pre-trained anchor or last accepted global), the score is

    S(c, r) = max_{dir in {down, up}} max_{delta in B} mean_x [ D_dir(c, x, delta) - D_dir(r, x, delta) ]
    D_down(m, x, delta) = mean_h (f_m(x) - f_m(x + delta)) / (max(|f_m(x)|, rho * sigma_x) + eps)   (D_up = -D_down)
    B = { delta >= 0, supported on the last W slots, delta <= amplitude_max * sigma_x }

(rho = burstguard.rel_floor, default 0 = plain relative change; a floor keeps the relative change finite where the
forecast is close to zero, e.g. night-time SMS / call counts; sigma_x = scale of the probe series)
computed with projected sign-gradient ascent (K steps, R random restarts) on public probe windows. Only the
aggregate (or group aggregates) is needed, so the screen is compatible with secure aggregation.

Two references (burstguard.reference = "both", default):
  S_last   vs. the last accepted global  -> sudden injections; flagged when
           S_last > max(threshold, median(H_last) + mad_k * MAD(H_last))
  S_anchor vs. the pre-trained model     -> accumulated poisoning; flagged when
           S_anchor > max(threshold_anchor, median(H_anchor) + mad_k * MAD(H_anchor))   (anchor_adaptive = true)
           S_anchor > threshold_anchor                                                 (anchor_adaptive = false)
H_* = per-round medians of the accepted scores over the last `history` rounds. Benign fine-tuning drifts smoothly
away from the anchor, so the adaptive anchor threshold follows that drift; a fixed one (anchor_adaptive = false)
also stops slow ramps but needs calibration for the run length (experiments/E0_calibration.toml, E9_ablation.toml).

v0.3.12 options (all off by default): signed bursts (dips too), best iterate of every chain, online calibration on
the first `calibrate_rounds` rounds of the run, explicit regroup thresholds.
v0.3.13 option: calibrate_guard = "public" - an attacker-independent ceiling (cap) for both thresholds, computed by
the server on a simulated clean federation of its public probe areas (fedcore.ServerLogic / public_calibration.py).
While calibrating, aggregates above the cap are flagged and kept out of the calibration sample; afterwards the
thresholds never exceed the cap.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from burstguard import model as M


class BurstGuard:
    def __init__(self, cfg: dict[str, Any], base_model, probe_X: np.ndarray, probe_sigma: np.ndarray,
                 anchor_vec: np.ndarray):
        import copy
        self.cfg, self.bg = cfg, dict(cfg["burstguard"])   # own copy: online calibration rewrites thresholds
        self.freq = cfg["model"]["freq"]
        self.cand = copy.deepcopy(base_model)
        self.ref = copy.deepcopy(base_model)
        self._pc = M.capture_params(self.cand)
        self._pr = M.capture_params(self.ref)
        for m in (self.cand, self.ref):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        n = int(self.bg["probe_windows"])
        rng = np.random.default_rng(777)
        idx = rng.choice(len(probe_X), size=min(n, len(probe_X)), replace=False) if len(probe_X) > n else np.arange(len(probe_X))
        self.X = torch.from_numpy(np.ascontiguousarray(probe_X[idx]))
        self.sig = torch.from_numpy(np.asarray(probe_sigma, dtype=np.float32)[idx].reshape(-1, 1, 1))
        self.anchor = anchor_vec.copy()
        self.last = anchor_vec.copy()
        self.history: list[float] = []         # per-round median of accepted S_last
        self.history_anchor: list[float] = []  # per-round median of accepted S_anchor
        t = int(self.bg.get("threads", 0) or 0)
        self.threads = t if t > 0 else max(1, min(4, (os.cpu_count() or 2) - 1))
        self.cal_rounds = int(self.bg.get("calibrate_rounds", 0) or 0)
        self.cal_last: list[float] = []        # online calibration: scores of the first cal_rounds rounds
        self.cal_anchor: list[float] = []
        self.calibrated: dict[str, float] | None = None
        self.cap: tuple[float, float] | None = None   # v0.3.13 public ceiling (S_last, S_anchor)

    def score(self, cand_vec: np.ndarray, ref_vec: np.ndarray, seed: int = 0) -> float:
        return self.search(cand_vec, ref_vec, seed)[0]

    def search(self, cand_vec: np.ndarray, ref_vec: np.ndarray, seed: int = 0) -> tuple[float, np.ndarray, float]:
        """Projected sign-gradient ascent; all (direction, restart) chains run as ONE batch (identical to running
        them one after another: the model has no cross-sample interaction and the sign of each sample's gradient
        does not depend on the other samples), which is ~1.5-2x faster on CPU than sequential chains.
        Returns (score, bursts [N, W, 1] of the best chain, its direction s)."""
        bg = self.bg
        M.set_params_vec(self._pc, cand_vec)
        M.set_params_vec(self._pr, ref_vec)
        W, K, R = int(bg["width_max"]), int(bg["steps"]), max(1, int(bg["restarts"]))
        amax, step = float(bg["amplitude_max"]), float(bg["step_size"])
        X, sig = self.X, self.sig
        N, L = X.shape[0], X.shape[1]
        g0 = torch.Generator().manual_seed(int(seed))
        signs = (1.0, -1.0) if bg["both_directions"] else (1.0,)
        chains = [sg for sg in signs for _ in range(R)]
        C = len(chains)
        with torch.no_grad():
            bc = M.forward(self.cand, X, self.freq)
            br = M.forward(self.ref, X, self.freq)
        rep = lambda t: t.repeat((C,) + (1,) * (t.dim() - 1))
        XX, SS, BC, BR = rep(X), rep(sig), rep(bc), rep(br)
        sgn = torch.tensor(chains, dtype=bc.dtype).repeat_interleave(N).view((-1,) + (1,) * (bc.dim() - 1))
        signed = bool(bg.get("signed", False))
        lo = -1.0 if signed else 0.0
        g = torch.cat([torch.rand(N, W, 1, generator=g0) for _ in chains])
        if signed:
            g = 2.0 * g - 1.0
        g = g.requires_grad_(True)
        best = bool(bg.get("best_iterate", False))
        rho = float(bg.get("rel_floor", 0.0) or 0.0)
        if rho > 0:      # denominators max(|f(x)|, rho * sigma) + eps
            DC, DR = torch.maximum(BC.abs(), rho * SS) + 1e-3, torch.maximum(BR.abs(), rho * SS) + 1e-3
        else:            # plain relative change (default; identical to v0.3.5)
            DC, DR = BC.abs() + 1e-3, BR.abs() + 1e-3

        def chain_scores(gv):
            d = torch.zeros_like(XX)
            d[:, L - W:, :] = gv * amax * SS
            dc = sgn * (BC - M.forward(self.cand, XX + d, self.freq)) / DC
            dr = sgn * (BR - M.forward(self.ref, XX + d, self.freq)) / DR
            return (dc - dr).reshape(C, -1).mean(dim=1)      # one score per chain

        top = torch.full((C,), -float("inf"))
        top_g = g.detach().clone()

        def keep(sc, gv):   # best iterate per chain (v0.3.12 option)
            better = sc > top
            top[better] = sc[better]
            top_g.view(C, N, W, 1)[better] = gv.view(C, N, W, 1)[better]

        for _ in range(K):
            sc = chain_scores(g)
            if best:
                keep(sc.detach(), g.detach())
            sc.sum().backward()
            with torch.no_grad():
                g += step * g.grad.sign()
                g.clamp_(lo, 1.0)
                g.grad.zero_()
        with torch.no_grad():  # evaluate the final iterate of every chain
            sc = chain_scores(g)
            if best:
                keep(sc, g.detach())
            else:
                top, top_g = sc, g.detach()
        c = int(torch.argmax(top))
        bursts = (top_g.view(C, N, W, 1)[c] * amax * sig).numpy()   # [N, W, 1] burst of the best chain
        return float(top[c]), bursts, float(chains[c])

    def _adaptive(self, floor: float, hist: list[float]) -> float:
        h = hist[-int(self.bg["history"]):]
        if len(h) >= 3:
            med = float(np.median(h))
            mad = float(np.median(np.abs(np.array(h) - med)))
            return max(floor, med + float(self.bg["mad_k"]) * mad)
        return floor

    def calibrating(self, rnd: int) -> bool:
        k = getattr(self, "cal_rounds", 0)
        return k > 0 and getattr(self, "calibrated", None) is None and rnd <= k

    def _cal_pending(self) -> bool:
        return getattr(self, "cal_rounds", 0) > 0 and getattr(self, "calibrated", None) is None

    def threshold(self) -> float:
        cap = getattr(self, "cap", None)
        if cap is not None and self._cal_pending():   # guarded online calibration: the cap screens
            return cap[0]
        t = self._adaptive(float(self.bg["threshold"]), self.history)
        return min(t, cap[0]) if cap is not None else t

    def threshold_anchor(self) -> float:
        cap = getattr(self, "cap", None)
        if cap is not None and self._cal_pending():
            return cap[1]
        floor = float(self.bg["threshold_anchor"])
        t = self._adaptive(floor, self.history_anchor) if self.bg.get("anchor_adaptive", True) else floor
        return min(t, cap[1]) if cap is not None else t

    def set_public_cap(self, cap_last: float, cap_anchor: float, rnd: int = 0) -> None:
        """v0.3.13: install the public ceiling. Without online calibration (calibrate_rounds = 0) the cap is the
        threshold pair (fixed anchor threshold)."""
        self.cap = (float(cap_last), float(cap_anchor))
        if self.cal_rounds == 0:
            self.bg["threshold"], self.bg["threshold_anchor"], self.bg["anchor_adaptive"] = self.cap[0], self.cap[1], False
            self.calibrated = {"round": int(rnd), "threshold": self.cap[0], "threshold_anchor": self.cap[1],
                               "n_last": 0, "n_anchor": 0, "source": "public", "cap": list(self.cap)}

    def regroup_scale(self, groups: int | None = None) -> float:
        """Threshold multiplier for the units of a regroup into `groups` groups
        (burstguard.regroup_threshold_scale; "sqrt" = sqrt(groups))."""
        rs = self.bg.get("regroup_threshold_scale", 1.0)
        g = int(groups if groups is not None else self.bg["regroup_groups"])
        return float(np.sqrt(g)) if rs == "sqrt" else float(rs)

    def screen(self, theta_g: np.ndarray, unit_deltas: list[np.ndarray], rnd: int,
               scale: float = 1.0, override: tuple[float, float] | None = None) -> tuple[list[bool], dict]:
        """Return (flags per candidate unit, info). Candidates are theta_g + unit delta. `scale` multiplies both
        thresholds (regroup units: partial aggregates are noisier than the full aggregate); `override` gives
        absolute thresholds (explicit regroup thresholds). During online calibration nothing is flagged, unless a
        public cap guards the calibration (v0.3.13): then units above the cap are flagged and only the scores of the
        unflagged units enter the calibration sample."""
        t0, w0 = time.process_time(), time.time()
        ref = self.bg["reference"]
        thr, thr_a = self.threshold() * scale, self.threshold_anchor() * scale
        if override is not None:
            thr, thr_a = float(override[0]), float(override[1])
        cal = self.calibrating(rnd)
        s_last, s_anchor, would = [], [], []
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            for i, u in enumerate(unit_deltas):
                cand = theta_g + u
                sl = self.score(cand, self.last, seed=rnd * 31 + i) if ref in ("last", "both") else None
                sa = self.score(cand, self.anchor, seed=rnd * 37 + i) if ref in ("anchor", "both") else None
                s_last.append(sl); s_anchor.append(sa)
                would.append(bool((sl is not None and sl > thr) or (sa is not None and sa > thr_a)))
        finally:
            torch.set_num_threads(prev_threads)
        guard = cal and getattr(self, "cap", None) is not None
        if cal:   # online calibration: record, never flag (guarded: flag above the cap, record the rest)
            self.cal_last += [x for x, w in zip(s_last, would) if x is not None and not (guard and w)]
            self.cal_anchor += [x for x, w in zip(s_anchor, would) if x is not None and not (guard and w)]
            if not guard:
                would = [False] * len(would)
            if rnd >= self.cal_rounds:
                self._finish_calibration(rnd)
        flags = would if self.bg["mode"] == "enforce" else [False] * len(would)
        # one comparable score per unit for logging / ROC analysis: largest margin over its threshold
        base = float(self.bg["threshold"]) * scale
        scores = [max([(x - t) for x, t in ((sl, thr), (sa, thr_a)) if x is not None]) + base
                  for sl, sa in zip(s_last, s_anchor)]
        return flags, {"scores": scores, "score_last": s_last, "score_anchor": s_anchor, "threshold": thr,
                       "threshold_anchor": thr_a, "threshold_scale": scale, "flags": flags, "would_flag": would,
                       "cpu_s": time.process_time() - t0, "wall_s": time.time() - w0, "threads": self.threads,
                       "mode": self.bg["mode"], "reference": ref, "calibrating": cal,
                       "calibrated": getattr(self, "calibrated", None),
                       **({"cap": list(self.cap), "guarded": guard} if getattr(self, "cap", None) is not None else {})}

    def _finish_calibration(self, rnd: int) -> None:
        """End of the online calibration window: fixed thresholds = margin x largest score seen so far."""
        m = float(self.bg.get("calibrate_margin", 1.5))
        cap = getattr(self, "cap", None)
        thr = m * max(self.cal_last) if self.cal_last else (cap[0] if cap else float(self.bg["threshold"]))
        thr_a = m * max(self.cal_anchor) if self.cal_anchor else (cap[1] if cap else float(self.bg["threshold_anchor"]))
        online = (float(thr), float(thr_a))
        if cap is not None:      # v0.3.13: never above the public ceiling
            thr, thr_a = min(thr, cap[0]), min(thr_a, cap[1])
        self.bg["threshold"], self.bg["threshold_anchor"], self.bg["anchor_adaptive"] = thr, thr_a, False
        self.calibrated = {"round": int(rnd), "threshold": float(thr), "threshold_anchor": float(thr_a),
                           "n_last": len(self.cal_last), "n_anchor": len(self.cal_anchor)}
        if cap is not None:
            self.calibrated.update(source="online+cap", cap=list(cap), online=list(online),
                                   cap_binding=[online[0] > cap[0], online[1] > cap[1]])

    def accept(self, theta_new: np.ndarray, accepted_last: list[float | None],
               accepted_anchor: list[float | None] | None = None) -> None:
        self.last = theta_new.copy()
        sl = [float(s) for s in accepted_last if s is not None]
        sa = [float(s) for s in (accepted_anchor or []) if s is not None]
        if sl:
            self.history.append(float(np.median(sl)))
        if sa:
            self.history_anchor.append(float(np.median(sa)))


def replay_flags(series: list[list[tuple[float | None, float | None]]], threshold: float, threshold_anchor: float,
                 mad_k: float, history: int, anchor_adaptive: bool = True) -> list[list[bool]]:
    """Re-apply the BurstGuard decision rule offline to per-round unit scores [(S_last, S_anchor), ...]
    (used by `burstguard calibrate` to estimate the false-positive rate of candidate thresholds)."""
    def adaptive(floor, hist):
        h = hist[-history:]
        if len(h) >= 3:
            med = float(np.median(h))
            return max(floor, med + mad_k * float(np.median(np.abs(np.array(h) - med))))
        return floor
    h_last: list[float] = []
    h_anchor: list[float] = []
    out = []
    for units in series:
        thr = adaptive(threshold, h_last)
        thr_a = adaptive(threshold_anchor, h_anchor) if anchor_adaptive else threshold_anchor
        flags = [bool((sl is not None and sl > thr) or (sa is not None and sa > thr_a)) for sl, sa in units]
        out.append(flags)
        sl_ok = [sl for (sl, _), f in zip(units, flags) if not f and sl is not None]
        sa_ok = [sa for (_, sa), f in zip(units, flags) if not f and sa is not None]
        if len(sl_ok) + len(sa_ok) and not all(flags):
            if sl_ok:
                h_last.append(float(np.median(sl_ok)))
            if sa_ok:
                h_anchor.append(float(np.median(sa_ok)))
    return out
