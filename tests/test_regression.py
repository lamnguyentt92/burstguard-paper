"""Regression across releases: training, attacks, BurstGuard scores and metrics of a fixed scenario
must equal the values computed with release/v0.3.5 (the release of E0-E9; X1-X8 ran under v0.3.6-v0.3.11).
scripts/regression_versions.sh recomputes the scenario with every release branch."""
import json
import os
from pathlib import Path

os.environ.setdefault("BURSTGUARD_SKIP_VENV_CHECK", "1")

import pytest

from burstguard.paths import isolate_environment

isolate_environment()

from burstguard.config import load_config
from burstguard.data.store import save_dataset
from burstguard.data.synthetic import generate

GOLDEN = Path(__file__).parent / "data" / "regression_golden_v0.3.5.json"
SCENARIOS = [("clean_fedavg", ['attack.type="none"', 'defense.name="fedavg"']),
             ("scale_bg", ['attack.type="scale"', 'defense.name="burstguard"', "burstguard.threshold=0.031",
                           "burstguard.threshold_anchor=0.14", "burstguard.anchor_adaptive=false"]),
             ("poison_lora_bg", ['train.peft="lora"', 'attack.type="poison"', 'defense.name="burstguard"'])]


def scenario(data: Path, out: Path) -> dict:
    from burstguard.engines.local import run_local
    res = {}
    for name, extra in SCENARIOS:
        cfg = load_config(None, [f'data.path="{data}"', 'model.source="random"', "data.num_sites=6",
                                 "data.max_train_windows=64", "data.max_eval_windows=16", "fl.rounds=3",
                                 "fl.eval_every=3", "engine.workers=1", "attack.num_malicious=1",
                                 "attack.local_epochs=2", "burstguard.steps=3", "burstguard.probe_windows=8",
                                 'fl.final_splits=["test"]'] + extra)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        run_local(cfg, d)
        recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
        rounds = [r for r in recs if r["kind"] == "round"]
        ev = [r for r in recs if r["kind"] == "eval" and r["split"] == "test"][-1]
        res[name] = {"final": {k: ev[k] for k in ("mae_mean", "mase_daily_mean", "asr_mean", "upr_trig_mean",
                                                  "upr_clean_mean") if k in ev},
                     "scores": [(r.get("burstguard") or {}).get("scores") for r in rounds],
                     "accepted": [r["accepted"] for r in rounds]}
    return res


def test_same_results_as_v035(tmp_path):
    data = tmp_path / "syn.npz"
    save_dataset(data, *generate(6, 2, seed=0)[:3])
    got = scenario(data, tmp_path)
    ref = json.loads(GOLDEN.read_text())
    for name, _ in SCENARIOS:
        g, r = got[name], ref[name]
        assert g["accepted"] == r["accepted"]
        # Bit-identical between releases on the SAME machine (scripts/regression_versions.sh). Across CPUs, float
        # rounding differs slightly: metrics agree to 1e-4; BurstGuard scores to 1e-2, because the sign-gradient
        # search can take a different step when a gradient component is ~0 (vm-e: 1.7005 vs 1.7028 here).
        for k, v in r["final"].items():
            assert g["final"][k] == pytest.approx(v, rel=1e-4, abs=1e-6), (name, k)
        for a, b in zip(g["scores"], r["scores"]):
            assert (a is None) == (b is None)
            if a is not None:
                assert a == pytest.approx(b, rel=1e-2, abs=1e-4), name
