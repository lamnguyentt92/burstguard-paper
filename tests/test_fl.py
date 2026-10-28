"""Integration: tiny federations with the local engine (in-process), random TTM, synthetic data."""
import json
import os

os.environ.setdefault("BURSTGUARD_SKIP_VENV_CHECK", "1")

import pytest

from burstguard.paths import isolate_environment

isolate_environment()

from burstguard.config import load_config
from burstguard.data.store import save_dataset
from burstguard.data.synthetic import generate


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "syn.npz"
    save_dataset(p, *generate(4, 2, seed=0)[:3])
    return p


def _cfg(data_path, extra):
    return load_config(None, [f'data.path="{data_path}"', 'model.source="random"', "data.num_sites=4",
                              "data.max_train_windows=64", "data.max_eval_windows=16", "fl.rounds=2",
                              "fl.eval_every=1", "engine.workers=1", "attack.num_malicious=1",
                              "attack.local_epochs=1", "burstguard.steps=2", "burstguard.restarts=1",
                              "burstguard.probe_windows=8"] + extra)


@pytest.mark.parametrize("extra", [
    ['attack.type="scale"', 'defense.name="burstguard"'],
    ['attack.type="adaptive"', 'defense.name="flame"'],
    ['attack.type="neurotoxin"', 'defense.name="fltrust"'],
    ['attack.type="constrain_scale"', "secagg.enabled=true", "secagg.groups=2", 'defense.name="burstguard"'],
    ['train.peft="lora"', 'attack.type="poison"', 'defense.name="lora_reset"', "defense.lora_reset_every=1"],
    ['attack.type="scale"', 'defense.name="probe_screen"', "secagg.enabled=true"],
])
def test_local_engine_runs(tmp_path, data_path, extra):
    from burstguard.engines.local import run_local
    res = run_local(_cfg(data_path, extra), tmp_path)
    assert res["rounds_done"] == 2 and "test" in res["final"]
    recs = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    rounds = [r for r in recs if r["kind"] == "round"]
    assert all(r["v"] for r in recs) and rounds[0]["malicious_in_round"]
    assert any(c["role"] == "malicious" for c in rounds[0]["clients"])


def test_burstguard_regroup_path(tmp_path, data_path):
    """A flagged aggregate triggers random regrouping; with near-zero thresholds every group is flagged too."""
    from burstguard.engines.local import run_local
    cfg = _cfg(data_path, ['attack.type="scale"', 'defense.name="burstguard"', "burstguard.threshold=1e-9",
                           "burstguard.threshold_anchor=1e-9", "burstguard.regroup_groups=2"])
    res = run_local(cfg, tmp_path)
    rounds = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    rounds = [r for r in rounds if r["kind"] == "round"]
    rg = [r["burstguard_regroup"] for r in rounds if r.get("burstguard_regroup")]
    assert rg and all(len(x["unit_sites"]) == 2 for x in rg) and not any(r["accepted"] for r in rounds)
    assert res["detection"]["regroup_rounds"] == len(rg) and res["detection"]["tpr_units"] == 1.0
