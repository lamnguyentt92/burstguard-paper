"""v0.3.16: Trentino reader (traffic-ranked single squares), partial participation, per-mode plan cost model."""
import json
import os

os.environ.setdefault("BURSTGUARD_SKIP_VENV_CHECK", "1")

import numpy as np
import pytest

from burstguard.paths import isolate_environment

isolate_environment()

from burstguard.config import load_config
from burstguard.data import milan
from burstguard.data.store import save_dataset
from burstguard.data.synthetic import generate


def _write_tn_day(path, day, squares):
    """Trentino day file (same format as Milan): square i has traffic level i + 1; square 999 is active in only
    half of the slots (must not be chosen); squares 5000.. are ids of an irregular grid."""
    rng = np.random.default_rng(day)
    lines = []
    for slot in range(144):
        ms = milan.T0_MS + (day * 144 + slot) * milan.SLOT_MS
        for i, sq in enumerate(squares):
            lines.append("\t".join([str(sq), str(ms), "39", "", "0.2", "", "", f"{(i + 1) * rng.uniform(0.9, 1.1):.4f}"]))
        if slot % 2 == 0:
            lines.append("\t".join(["999", str(ms), "39", "", "", "", "", "1000"]))
    path.write_text("\n".join(lines) + "\n")


def test_trentino_prepare(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    squares = [5000 + 7 * k for k in range(14)]
    for d in range(3):
        _write_tn_day(raw / f"sms-call-internet-tn-2013-11-0{d + 1}.txt", d, squares)
    # a Milan file in the same folder is ignored for region tn
    (raw / "sms-call-internet-mi-2013-11-01.txt").write_text("1\t1383260400000\t39\t\t\t\t\t5\n")
    out = milan.prepare_multi(raw, ("internet", "sms"), num_sites=4, num_probe=2, days=3, region="tn", stride=2)
    s, p, meta, b = out["internet"]
    assert s.shape == (4, 3 * 144) and p.shape == (2, 3 * 144) and (s > 0).all()
    chosen = [g["square"] for g in meta["sites"] + meta["probes"]]
    assert 999 not in chosen                                  # active in only half of the slots
    ranked = sorted(squares, key=lambda q: -squares.index(q))  # most traffic first
    assert sorted(chosen) == sorted(ranked[::2][:6])          # every 2nd square of the ranking
    assert meta["source"] == "trentino" and meta["block"] == 1 and all(len(g["squares"]) == 1 for g in meta["sites"])
    assert np.allclose(out["sms"][0], 0.2)                    # smsout only
    assert milan.processed_name("internet", "tn") == "trentino_d1.npz"
    assert milan.processed_name("sms", "tn") == "trentino_sms_d1.npz" and milan.processed_name("sms") == "milan_sms_d1.npz"
    # not enough squares for stride 2 -> the reader falls back to stride 1 (13 squares are active in every slot)
    dense = milan.prepare_multi(raw, ("internet",), num_sites=9, num_probe=4, days=3, region="tn", stride=2)
    assert len(dense["internet"][2]["sites"]) == 9 and 999 not in [g["square"] for g in dense["internet"][2]["sites"]]
    with pytest.raises(ValueError, match="only"):
        milan.prepare_multi(raw, ("internet",), num_sites=16, num_probe=4, days=3, region="tn", stride=2)


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "syn.npz"
    save_dataset(p, *generate(6, 2, seed=0)[:3])
    return p


def _cfg(data_path, extra, rounds=4):
    return load_config(None, [f'data.path="{data_path}"', 'model.source="random"', "data.num_sites=6",
                              "data.max_train_windows=64", "data.max_eval_windows=16", f"fl.rounds={rounds}",
                              f"fl.eval_every={rounds}", "engine.workers=1", "attack.local_epochs=1",
                              "burstguard.steps=2", "burstguard.restarts=1", "burstguard.probe_windows=8",
                              'fl.final_splits=["test"]'] + extra)


def test_partial_participation(tmp_path, data_path):
    from burstguard.engines.local import run_local
    cfg = _cfg(data_path, ["fl.participation=0.5", 'attack.type="scale"', "attack.num_malicious=1",
                           'defense.name="burstguard"', "burstguard.regroup_groups=2", "burstguard.regroup_verify=true"],
               rounds=6)
    res = run_local(cfg, tmp_path, log=lambda m: None)
    rounds = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    rounds = [r for r in rounds if r["kind"] == "round"]
    mal = cfg["attack"]["malicious_sites"] or [0]
    for r in rounds:
        assert len(r["participants"]) == 3 and len(r["clients"]) == 3
        assert r["attack_active"] == bool(set(r["participants"]) & set(r["malicious_in_round"] or [-1]))
        assert all(c["site"] in r["participants"] for c in r["clients"])
    assert len({tuple(r["participants"]) for r in rounds}) > 1      # a new sample every round
    assert res["rounds_done"] == 6
    with pytest.raises(ValueError, match="participation"):
        _cfg(data_path, ["fl.participation=0"])
    with pytest.raises(ValueError, match="local engine"):
        _cfg(data_path, ["fl.participation=0.5", 'engine.name="flower"'])


def test_plan_does_not_extrapolate_lora_from_head():
    from burstguard.experiments import plan as PL
    base = {"kind": "fl", "engine": "local", "bg": False, "units": 1, "effort": 0.0, "rounds": 20, "workers": 3,
            "shadow": 0.0, "public": {}, "sites": 20, "participation": 1.0}
    fin = [{**base, "peft": "head", "attacked": False, "wall_s": 200.0, "clients_wall_s": 3.6, "grid": "R14"},
           {**base, "peft": "head", "attacked": True, "wall_s": 600.0, "clients_wall_s": 18.0, "grid": "R14"}]
    m = PL.fit(fin)
    assert PL.client_round(m, "lora") == PL.PRIOR_C["lora"]        # not 3.6 / 0.025 x 1.0 = 144 s
    assert PL.attack_factor(m, "head") > 4 and PL.attack_factor(m, "lora") == PL.PRIOR_ATTACK
    fin.append({**base, "peft": "lora", "attacked": False, "wall_s": 3000.0, "clients_wall_s": 80.0, "grid": "R15"})
    m = PL.fit(fin)
    assert PL.client_round(m, "lora") == pytest.approx(80.0 * 3 / PL.REF_WEIGHT)
    assert PL.client_round(m, "full") == pytest.approx(80.0 * 3 / PL.REF_WEIGHT / PL.RATIO_C["lora"] * PL.RATIO_C["full"])
