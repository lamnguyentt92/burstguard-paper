"""v0.3.12: revision experiments - new triggers, metrics, baselines, adaptive attackers, BurstGuard options."""
import json
import os

os.environ.setdefault("BURSTGUARD_SKIP_VENV_CHECK", "1")

import numpy as np
import pytest

from burstguard.paths import isolate_environment

isolate_environment()

from burstguard import metrics as MT
from burstguard.attacks.trigger import apply_trigger, trigger_pattern
from burstguard.config import load_config
from burstguard.data.store import save_dataset
from burstguard.data.synthetic import generate
from burstguard.defenses import aggregators as AGG


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "syn.npz"
    save_dataset(p, *generate(4, 2, seed=0)[:3])
    return p


def _cfg(data_path, extra, rounds=3):
    return load_config(None, [f'data.path="{data_path}"', 'model.source="random"', "data.num_sites=4",
                              "data.max_train_windows=64", "data.max_eval_windows=16", f"fl.rounds={rounds}",
                              "fl.eval_every=1", "engine.workers=1", "attack.num_malicious=1",
                              "attack.local_epochs=1", "burstguard.steps=2", "burstguard.restarts=1",
                              "burstguard.probe_windows=8", "attack.evasion_probes=4", "attack.budget_steps=2"]
                       + extra)


def _rounds(tmp_path):
    recs = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    return [r for r in recs if r["kind"] == "round"], [r for r in recs if r["kind"] == "eval"]


def test_trigger_shapes():
    X = np.full((2, 20, 1), 1.0, dtype=np.float32)
    multi = {"width": 2, "amplitude": 1.0, "offset": 0, "shape": "multi", "count": 3, "spacing": 4}
    assert trigger_pattern(multi).tolist() == [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    Xt = apply_trigger(X, 2.0, multi)
    assert Xt[0, -10:, 0].tolist() == [3, 3, 1, 1, 3, 3, 1, 1, 3, 3] and Xt[0, 0, 0] == 1
    dip = {"width": 3, "amplitude": -1.0, "offset": 144 - 140, "shape": "rect"}
    Xd = apply_trigger(X, 2.0, dip)
    assert Xd.min() == 0.0 and (Xd[0, -7:-4, 0] == 0).all() and Xd[0, -1, 0] == 1.0


def test_natural_surge_and_shortfall():
    X = np.ones((3, 20, 1), dtype=np.float32)
    X[1, -3:, 0] = 3.0          # +2 over the previous hour
    X[2, -3:, 0] = 1.5          # +0.5
    m = MT.natural_surge(X, 1.0, {"width": 3, "amplitude": 1.0})
    assert m.tolist() == [False, True, False]
    assert MT.natural_surge(X, 1.0, {"width": 3, "amplitude": -1.0}).sum() == 0
    y = np.array([[[10.0], [10.0]]])
    assert MT.shortfall(np.array([[[5.0], [12.0]]]), y) == pytest.approx(0.25)


def test_aggregators():
    cfg = load_config(None, [])
    rng = np.random.default_rng(0)
    units = [np.ones(4, np.float32) for _ in range(6)] + [np.full(4, 100.0, np.float32)]
    z, _ = AGG.aggregate("rfa", units, {"cfg": cfg, "rng": rng})
    assert np.allclose(z, 1.0, atol=0.5)          # geometric median ignores the outlier
    cfg["defense"]["aggclip_mode"] = "history"
    a, info = AGG.aggregate("aggclip", units, {"cfg": cfg, "rng": rng, "agg_norm_hist": [2.0, 2.0]})
    assert info["clip_norm"] == pytest.approx(2.0) and np.linalg.norm(a) < 2.0 + 0.5


@pytest.mark.parametrize("extra", [
    ['attack.type="budget"', 'defense.name="burstguard"', "burstguard.threshold=0.05",
     "burstguard.threshold_anchor=0.2", "attack.regroup_ride=true", 'burstguard.regroup_threshold_scale="sqrt"',
     "burstguard.regroup_groups=2"],
    ['attack.type="evasive"', "attack.boost=2.0", 'defense.name="burstguard"'],
    ['attack.type="scale"', 'defense.name="normbound"'],
    ['attack.type="scale"', 'defense.name="normbound_burstguard"'],
    ['attack.type="poison"', 'defense.name="aggclip"', "secagg.enabled=true"],
    ['attack.type="scale"', 'defense.name="rfa"'],
    ['attack.type="scale"', 'defense.name="burstguard"', "burstguard.signed=true", "burstguard.best_iterate=true",
     'burstguard.probe_split="val"', 'trigger.shape="multi"', "trigger.count=2", "trigger.width=1"],
    ['attack.type="scale"', 'defense.name="burstguard"', "trigger.amplitude=-1.0", "attack.decoys=true",
     "trigger.offset=4"],
])
def test_v0312_runs(tmp_path, data_path, extra):
    from burstguard.engines.local import run_local
    res = run_local(_cfg(data_path, extra), tmp_path)
    assert res["rounds_done"] == 3 and "test" in res["final"]
    rounds, evals = _rounds(tmp_path)
    assert rounds[0]["malicious_in_round"]
    fin = [e for e in evals if e["split"] == "test"][-1]
    assert "shortfall_trig_mean" in fin and "frac_natural_mean" in fin and "n_natural_total" in fin
    if "normbound" in extra[1]:
        assert all(r["norm_bound"]["tau"] > 0 for r in rounds)
    if 'attack.type="budget"' in extra:
        mi = [c for c in rounds[0]["clients"] if c["role"] == "malicious"][0]
        assert "budget_boost" in mi and "regroup_budget_boost" in mi


def test_online_calibration_and_consecutive_limit(tmp_path, data_path):
    from burstguard.engines.local import run_local
    cfg = _cfg(data_path, ['attack.type="scale"', 'defense.name="burstguard"', "attack.start_round=3",
                           "burstguard.calibrate_rounds=2", "burstguard.calibrate_margin=1e-6",
                           "burstguard.max_consecutive_regroups=1", "burstguard.regroup_groups=2"], rounds=5)
    run_local(cfg, tmp_path)
    rounds, _ = _rounds(tmp_path)
    bg = [r["burstguard"] for r in rounds]
    assert bg[0]["calibrating"] and bg[1]["calibrating"] and not bg[2]["calibrating"]
    assert all(r["accepted"] for r in rounds[:2])                 # nothing is flagged while calibrating
    assert bg[2]["calibrated"]["threshold"] < 1e-3                # tiny margin -> every later round is flagged
    assert rounds[2].get("burstguard_regroup") and not rounds[3].get("burstguard_regroup")   # limit 1 -> reject


def test_explicit_group_thresholds(tmp_path, data_path):
    from burstguard.engines.local import run_local
    cfg = _cfg(data_path, ['attack.type="scale"', 'defense.name="burstguard"', "burstguard.threshold=1e-9",
                           "burstguard.threshold_anchor=1e-9", "burstguard.regroup_groups=2",
                           "burstguard.group_threshold=1e6", "burstguard.group_threshold_anchor=1e6"])
    run_local(cfg, tmp_path)
    rounds, _ = _rounds(tmp_path)
    rg = [r["burstguard_regroup"] for r in rounds if r.get("burstguard_regroup")]
    assert rg and all(len(x["kept"]) == 2 and x["threshold"] == 1e6 for x in rg)


def test_config_validation(data_path):
    with pytest.raises(ValueError):
        _cfg(data_path, ['trigger.shape="zigzag"'])
    with pytest.raises(ValueError):
        _cfg(data_path, ["burstguard.calibrate_rounds=3"])          # must be < fl.rounds
    with pytest.raises(ValueError):
        _cfg(data_path, ['defense.norm_bound_mode="fixed"'])        # needs a bound


def test_shadow_screens(tmp_path, data_path):
    from burstguard.config import dump_toml
    from burstguard.engines.local import run_local
    sh = [{"name": "val", "probe_split": "val"}, {"name": "p4s", "probe_windows": 4, "signed": True, "width_max": 9}]
    cfg = _cfg(data_path, ['attack.type="scale"', 'defense.name="burstguard"'])
    cfg["burstguard"]["shadow"] = sh
    assert 'shadow = [{name = "val", probe_split = "val"}' in dump_toml(cfg)
    run_local(cfg, tmp_path)
    rounds, _ = _rounds(tmp_path)
    for r in rounds:
        assert set(r["shadow"]) == {"val", "p4s"}
        assert all(np.isfinite(v["score_last"]) and np.isfinite(v["score_anchor"]) for v in r["shadow"].values())
    with pytest.raises(ValueError, match="shadow"):
        _cfg(data_path, ['burstguard.shadow=[{nme = "x"}]'])
