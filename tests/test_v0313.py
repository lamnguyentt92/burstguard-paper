"""v0.3.13: public calibration guard (calibration-period attacker) and regroup verification (many attackers)."""
import json
import os

os.environ.setdefault("BURSTGUARD_SKIP_VENV_CHECK", "1")

import numpy as np
import pytest

from burstguard.paths import isolate_environment

isolate_environment()

from burstguard.config import load_config
from burstguard.data.store import save_dataset
from burstguard.data.synthetic import generate


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "syn.npz"
    save_dataset(p, *generate(6, 2, seed=0)[:3])
    return p


def _cfg(data_path, extra, rounds=4):
    return load_config(None, [f'data.path="{data_path}"', 'model.source="random"', "data.num_sites=6",
                              "data.max_train_windows=64", "data.max_eval_windows=16", f"fl.rounds={rounds}",
                              f"fl.eval_every={rounds}", "engine.workers=1", "attack.num_malicious=1",
                              "attack.local_epochs=1", "burstguard.steps=2", "burstguard.restarts=1",
                              "burstguard.probe_windows=8", 'defense.name="burstguard"', "burstguard.regroup_groups=3",
                              'burstguard.regroup_threshold_scale="sqrt"', 'fl.final_splits=["test"]'] + extra)


def _run(tmp_path, cfg):
    from burstguard.engines.local import run_local
    tmp_path.mkdir(parents=True, exist_ok=True)
    res = run_local(cfg, tmp_path, log=lambda m: None)
    recs = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    return res, [r for r in recs if r["kind"] == "round"]


def test_guard_flags_while_calibrating_and_caps_thresholds(tmp_path, data_path):
    cfg = _cfg(data_path, ['attack.type="scale"', "attack.boost=3.0", "burstguard.calibrate_rounds=2",
                           'burstguard.calibrate_guard="public"', "burstguard.calibrate_public_margin=1e-6"])
    res, rounds = _run(tmp_path, cfg)
    pc = rounds[0]["public_calibration"]
    assert pc["rounds"] == 2 and pc["clients"] == 2 and not any("public_calibration" in r for r in rounds[1:])
    cap = [1e-6 * max(pc["score_last"]), 1e-6 * max(pc["score_anchor"])]
    assert pc["cap"] == pytest.approx(cap)
    bg = [r["burstguard"] for r in rounds]
    assert bg[0]["calibrating"] and bg[0]["guarded"] and bg[0]["threshold"] == pytest.approx(cap[0])
    assert all(b["would_flag"][0] for b in bg[:2])          # the attacker no longer passes while calibrating
    cal = bg[1]["calibrated"]
    assert cal["source"] == "online+cap" and cal["threshold"] <= cap[0] + 1e-12 and cal["n_last"] == 0
    assert all(b["threshold"] <= cap[0] + 1e-12 and b["threshold_anchor"] <= cap[1] + 1e-12 for b in bg)
    assert res["detection"]["public_calibration"]["cap"] == pytest.approx(cap)


def test_guard_with_a_loose_cap_changes_nothing(tmp_path, data_path):
    """The simulation must not touch the run (model, RNG streams): with a cap that never binds the run is identical
    to v0.3.12 online calibration."""
    base = ['attack.type="scale"', "attack.boost=3.0", "attack.start_round=3", "burstguard.calibrate_rounds=2"]
    _, r0 = _run(tmp_path / "a", _cfg(data_path, base))
    _, r1 = _run(tmp_path / "b", _cfg(data_path, base + ['burstguard.calibrate_guard="public"',
                                                        "burstguard.calibrate_public_margin=1e6"]))
    assert [r["accepted"] for r in r0] == [r["accepted"] for r in r1]
    for k in ("score_last", "score_anchor"):   # raw scores (the logged margin score depends on the thresholds)
        assert [r["burstguard"][k] for r in r0] == [r["burstguard"][k] for r in r1]
    assert r0[1]["burstguard"]["calibrated"]["threshold"] == r1[1]["burstguard"]["calibrated"]["threshold"]


def test_public_only_calibration(tmp_path, data_path):
    cfg = _cfg(data_path, ['attack.type="none"', 'burstguard.calibrate_guard="public"',
                           "burstguard.calibrate_public_rounds=2", "burstguard.calibrate_public_clients=3"])
    res, rounds = _run(tmp_path, cfg)
    pc = rounds[0]["public_calibration"]
    assert pc["clients"] == 3 and len(pc["score_last"]) == 2
    b = rounds[0]["burstguard"]
    assert b["calibrated"]["source"] == "public" and b["threshold_anchor"] == pytest.approx(pc["cap"][1])
    assert not b["calibrating"]


def test_regroup_verify_rejects_when_the_regroup_removes_nothing(tmp_path, data_path):
    # every group passes the (huge) group thresholds -> the regroup re-creates the flagged aggregate
    common = ['attack.type="scale"', "attack.num_malicious=3", "burstguard.threshold=1e-9",
              "burstguard.threshold_anchor=1e-9", "burstguard.anchor_adaptive=false", "burstguard.group_threshold=1e6",
              "burstguard.group_threshold_anchor=1e6"]
    res0, r0 = _run(tmp_path / "off", _cfg(data_path, common))
    assert all(r["accepted"] and len(r["burstguard_regroup"]["kept"]) == 3 for r in r0)      # v0.3.12 behaviour
    res1, r1 = _run(tmp_path / "on", _cfg(data_path, common + ["burstguard.regroup_verify=true"]))
    for r in r1:
        v = r["burstguard_verify"]
        assert v["threshold_scale"] == pytest.approx(1.0) and v["n_kept_clients"] == 6 and v["would_flag"][0]
        assert not r["accepted"]
    assert res1["detection"]["verify_rejected"] == 4 and res1["filtering"]["accepted_rate"] == 0.0
    assert res1["filtering"]["malicious_kept_rate"] == 0.0 and res0["filtering"]["malicious_kept_rate"] == 1.0


def test_oracle_budget_attacker_sees_the_enforced_thresholds(tmp_path, data_path):
    cfg = _cfg(data_path, ['attack.type="budget"', "attack.budget_steps=2", "burstguard.calibrate_rounds=2",
                           'burstguard.calibrate_guard="public"', 'attack.budget_thresholds="oracle"',
                           "attack.evasion_probes=4"])
    _, rounds = _run(tmp_path, cfg)
    cap = rounds[0]["public_calibration"]["cap"]
    mal = lambda r: [c for c in r["clients"] if c["role"] == "malicious"][0]
    assert mal(rounds[0])["budget_thresholds"] == pytest.approx(cap)
    cal = rounds[1]["burstguard"]["calibrated"]
    assert mal(rounds[2])["budget_thresholds"][1] == pytest.approx(cal["threshold_anchor"])


def test_config_validation_v0313(data_path):
    with pytest.raises(ValueError, match="calibrate_guard"):
        _cfg(data_path, ['burstguard.calibrate_guard="private"'])
    with pytest.raises(ValueError, match="public-only"):
        _cfg(data_path, ['burstguard.calibrate_guard="public"'])          # no online and no simulated rounds
    with pytest.raises(ValueError, match="budget_thresholds"):
        _cfg(data_path, ['attack.budget_thresholds="guess"'])
    cfg = _cfg(data_path, [])
    assert cfg["burstguard"]["calibrate_guard"] == "" and cfg["burstguard"]["regroup_verify"] is False
