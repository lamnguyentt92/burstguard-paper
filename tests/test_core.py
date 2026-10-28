"""Fast unit tests (no network): config, windows, Milan reader, model modes, attacks, defenses, grids, analysis."""
import gzip
import os

os.environ.setdefault("BURSTGUARD_SKIP_VENV_CHECK", "1")

import numpy as np
import pytest

from burstguard.paths import isolate_environment

isolate_environment()

from burstguard.attacks.trigger import apply_trigger, target
from burstguard.config import load_config
from burstguard.data import milan
from burstguard.data.store import load_dataset, save_dataset, split_bounds
from burstguard.data.synthetic import generate
from burstguard.data.windows import make_windows, seasonal_naive
from burstguard.defenses import aggregators as AGG
from burstguard.defenses.secagg import make_groups, secure_mean
from burstguard import metrics as MT


def test_config_validation():
    cfg = load_config(None, ["train.peft=lora", "fl.rounds=3", 'defense.name="burstguard"'])
    assert cfg["defense"]["aggregator"] == "fedavg" and "burstguard" in cfg["defense"]["addons"]
    with pytest.raises(ValueError):
        load_config(None, ["train.peft=bogus"])
    with pytest.raises(ValueError, match="secure aggregation"):
        load_config(None, ["secagg.enabled=true", 'defense.name="krum"'])
    with pytest.raises(ValueError, match="clip_norm"):
        load_config(None, ["secagg.enabled=true", 'defense.name="clip_noise"'])
    ok = load_config(None, ["secagg.enabled=true", "secagg.groups=5", 'defense.name="median"'])
    assert ok["burstguard"]["screen_resolved"] == "units"
    with pytest.raises(ValueError, match="lora"):
        load_config(None, ['defense.name="lora_reset"', 'train.peft="head"'])


def test_split_bounds_and_windows():
    b = split_bounds(62 * 144)
    assert b["train"] == [0, 37 * 144] and b["val"] == [37 * 144, 44 * 144]
    x = np.arange(62 * 144, dtype=np.float32)
    X, Y, t0 = make_windows(x, *b["val"], L=512, H=36, stride=6)
    assert X.shape[1:] == (512, 1) and Y.shape[1:] == (36, 1)
    assert (Y[:, 0, 0] == t0).all() and (X[:, -1, 0] == t0 - 1).all()
    assert (seasonal_naive(x, t0, 36, 144)[:, 0, 0] == t0 - 144).all()


def _write_fake_day(path, day_index, squares, gz=False, header=False):
    rng = np.random.default_rng(day_index)
    lines = ["CellID,datetime,countrycode,smsin,smsout,callin,callout,internet"] if header else []
    sep = "," if header else "\t"
    for slot in range(144):
        ms = milan.T0_MS + (day_index * 144 + slot) * milan.SLOT_MS
        for sq in squares:
            for cc in (39, 33):
                lines.append(sep.join([str(sq), str(ms), str(cc), "", "0.1", "", "", f"{rng.uniform(1, 10):.4f}"]))
    data = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt") as f:
            f.write(data)
    else:
        path.write_text(data)


def test_milan_reader_and_prepare(tmp_path):
    sites, probes = milan.choose_blocks(3, 2, block=2)
    sq_all = {s for g in sites + probes for s in g["squares"]}
    assert len(sq_all) == 5 * 4
    raw = tmp_path / "raw"; raw.mkdir()
    for d in range(3):
        name = f"sms-call-internet-mi-2013-11-0{d + 1}.txt" + (".gz" if d == 1 else "")
        _write_fake_day(raw / name, d, sorted(sq_all), gz=(d == 1), header=(d == 2))
    s, p, meta, b = milan.prepare(raw, num_sites=3, num_probe=2, block=2, days=3)
    assert s.shape == (3, 3 * 144) and p.shape == (2, 3 * 144) and (s > 0).all()
    ds = load_dataset(save_dataset(tmp_path / "m.npz", s, p, meta, b))
    assert ds.num_sites == 3 and ds.meta["data_format"] == 1
    # several traffic types in one pass: same blocks; internet identical to the single-type path
    multi = milan.prepare_multi(raw, ("internet", "sms", "call"), num_sites=3, num_probe=2, block=2, days=3)
    assert np.array_equal(multi["internet"][0], s) and np.array_equal(multi["internet"][1], p)
    # fake files: smsout = 0.1 for 2 country codes, smsin/callin/callout empty; 2x2 squares per block
    assert np.allclose(multi["sms"][0], 0.1 * 2 * 4) and np.allclose(multi["call"][0], 0.0)
    assert multi["sms"][2]["activity"] == "sms" and multi["sms"][2]["sites"] == meta["sites"]
    assert milan.processed_name("internet") == "milan_d1.npz" and milan.processed_name("call") == "milan_call_d1.npz"
    milan.CHUNK_ROWS = 97                                   # many small chunks give the same result
    try:
        again = milan.prepare_multi(raw, ("internet", "sms"), num_sites=3, num_probe=2, block=2, days=3)
    finally:
        milan.CHUNK_ROWS = 2_000_000
    assert np.array_equal(again["internet"][0], s) and np.array_equal(again["sms"][0], multi["sms"][0])


def test_trigger_and_backdoor_metrics():
    X = np.ones((4, 512, 1), np.float32)
    Xt = apply_trigger(X, 2.0, {"amplitude": 1.5, "width": 3, "offset": 0, "shape": "rect"})
    assert np.allclose(Xt[:, -3:, 0], 4.0) and np.allclose(Xt[:, :-3, 0], 1.0)
    Y = np.full((4, 36, 1), 10.0, np.float32)
    tgt = target(Y, {"target_ratio": 0.3})
    assert MT.asr(tgt, Y, 0.3, 0.5) == 1.0 and MT.asr(Y, Y, 0.3, 0.5) == 0.0
    assert MT.target_error(tgt, Y, 0.3) < 1e-6
    assert MT.attack_effect(Y, tgt) == pytest.approx(0.7, abs=1e-6)
    assert MT.upr(tgt, Y, 0.8) == 1.0


def test_aggregators_resist_outliers():
    rng = np.random.default_rng(0)
    cfg = load_config(None, ["defense.krum_f=2", "defense.trim_beta=0.2"])
    honest = [rng.normal(0, 0.01, 50).astype(np.float32) + 1.0 for _ in range(10)]
    bad = [np.full(50, 50.0, np.float32), np.full(50, -40.0, np.float32)]
    units = honest + bad
    ctx = {"cfg": cfg, "rng": rng}
    for name in ("krum", "multikrum", "median", "trimmed", "flame"):
        agg, info = AGG.aggregate(name, units, ctx)
        assert np.abs(agg - 1.0).max() < 0.2, name
    agg, _ = AGG.aggregate("fedavg", units, ctx)
    assert np.abs(agg - 1.0).max() > 0.5
    ctx["server_update"] = lambda: np.ones(50, np.float32)
    agg, info = AGG.aggregate("fltrust", units, ctx)
    assert info["trust"][-1] == 0.0 and np.abs(agg - 1.0).max() < 0.2


def test_secure_mean_and_groups():
    rng = np.random.default_rng(1)
    cfg = load_config(None, [])
    thetas = [rng.normal(0, 1, 1000).astype(np.float32) for _ in range(5)]
    m, info = secure_mean(thetas, cfg, rng)
    assert np.abs(m - np.mean(thetas, axis=0)).max() < 1e-4 and info["clipped_values"] == 0
    big = [np.full(10, 20.0, np.float32)]
    m2, info2 = secure_mean(big, cfg, rng)
    assert info2["clipped_values"] == 10 and np.allclose(m2, 8.0, atol=1e-4)
    g = make_groups(20, 3, rng)
    assert sorted(i for x in g for i in x) == list(range(20)) and len(g) == 3


def test_model_modes_vectors_and_lora_reset():
    from burstguard import model as M
    from burstguard.defenses.lora_reset import LoraReset
    cfg = load_config(None, ['model.source="random"'])
    s, _, _, b = generate(1, 0, seed=0)
    X, _, _ = make_windows(s[0], *b["train"], L=512, H=36, stride=50)
    base, info = M.load_model(cfg)
    assert info.freq_prefix and M.predict(base, X[:4]).shape == (4, 36, 1)
    counts = {}
    for mode in ("head", "decoder", "lora", "full"):
        m = M.configure_peft(M.load_model(cfg)[0], {**cfg, "train": {**cfg["train"], "peft": mode}})
        v = M.get_vec(m)
        M.set_vec(m, v * 0 + 0.5)
        assert np.allclose(M.get_vec(m), 0.5)
        counts[mode] = v.size
    assert counts["head"] < counts["decoder"] < counts["full"] and counts["head"] < counts["lora"] < counts["full"]
    m = M.configure_peft(M.load_model(cfg)[0], cfg)
    assert M.check_head_cache(m, X, 36) < 1e-5
    lm = M.configure_peft(M.load_model(cfg)[0], {**cfg, "train": {**cfg["train"], "peft": "lora"}})
    init = M.get_vec(lm)
    lr = LoraReset(M.lora_segments(lm), init, every=5, frac=0.5, cooldown=500)
    changed = init + 1.0
    out, inf = lr.apply(changed, 5)
    assert inf["reset"] > 0 and (out == init).sum() > 0 and (out != init).sum() > 0
    assert lr.apply(changed, 6)[1]["reset"] == 0


def test_grid_expand_shard_exclude(tmp_path):
    from burstguard.experiments.grid import expand, load_grid, shard
    p = tmp_path / "g.toml"
    p.write_text('[grid]\nname="t"\nseeds=[0,1]\n[factors]\n"train.peft"=["head","lora"]\n'
                 '"defense.name"=["fedavg","lora_reset"]\n[[exclude]]\n"defense.name"="lora_reset"\n"train.peft"="head"\n')
    jobs = expand(load_grid(p))
    assert len(jobs) == 6 and len({j["id"] for j in jobs}) == 6
    parts = [shard(jobs, k, 3) for k in (1, 2, 3)]
    assert sorted(j["id"] for x in parts for j in x) == sorted(j["id"] for j in jobs)
    assert expand(load_grid(p))[0]["id"] == jobs[0]["id"]  # stable ids
    p.write_text('[grid]\nname="v"\nseeds=[0]\n[factors]\n"attack.type"=["none","scale"]\n'
                 '[[variants]]\n[[variants]]\n"burstguard.steps"=5\n[[include]]\n"trigger.amplitude"=3.0\n')
    jv = expand(load_grid(p))
    assert len(jv) == 5 and sum(j["overrides"].get("burstguard.steps") == 5 for j in jv) == 2
    # exclude can match run.seed (fewer seeds for an expensive mode); ids do not depend on the base config
    p.write_text('[grid]\nname="s"\nseeds=[0,1,2]\n[factors]\n"train.peft"=["head","lora"]\n'
                 '[[exclude]]\n"train.peft"="lora"\n"run.seed"=2\n')
    js = expand(load_grid(p))
    assert len(js) == 5 and not any(j["overrides"]["train.peft"] == "lora" and j["overrides"]["run.seed"] == 2
                                    for j in js)
    # cost uses the resolved configuration: rounds from the base config when the grid does not set them
    (tmp_path / "b.toml").write_text('[fl]\nrounds = 40\n')
    p.write_text(f'[grid]\nname="r"\nbase_config="{tmp_path / "b.toml"}"\nseeds=[0]\n[factors]\n'
                 '"train.peft"=["head"]\n')
    jr = expand(load_grid(p))
    assert jr[0]["effective"]["fl.rounds"] == 40 and "fl.rounds" not in jr[0]["overrides"]
    p.write_text('[grid]\nname="r"\nseeds=[0]\n[factors]\n"train.peft"=["head"]\n')
    assert expand(load_grid(p))[0]["id"] == jr[0]["id"] and expand(load_grid(p))[0]["cost"] < jr[0]["cost"]


def test_analysis_helpers():
    from burstguard.experiments.analyze import bootstrap_ci, holm, lifespan
    v, cens = lifespan([2, 4, 6, 8, 10, 12], [0.1, 0.8, 0.9, 0.6, 0.4, 0.2], start=2, end=6, rounds=12)
    assert v == 4.0 and not cens
    v, cens = lifespan([2, 4, 6, 8], [0.9, 0.9, 0.9, 0.9], start=2, end=4, rounds=8)
    assert cens and v == 4.0
    # UPR_trig: relative to the pre-attack level; no in-window excess = no backdoor (lifespan 0, not censored)
    assert lifespan([0, 2, 4, 6, 8], [0.02, 0.02, 0.03, 0.02, 0.02], start=2, end=4, rounds=8, relative=True) == (0.0, False)
    assert lifespan([0, 2, 4, 6, 8], [0.02, 0.7, 0.7, 0.3, 0.1], start=2, end=4, rounds=8, relative=True) == (2.0, False)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    mu, lo, hi = bootstrap_ci([[1.0, 1.0, 1.0], [1.0, 1.0]])
    assert mu == lo == hi == 1.0
    from burstguard.engines.common import auc
    from burstguard.experiments.analyze import roc_curve
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0 and auc([0.5, 0.5], [0, 1]) == 0.5
    assert auc([0.1, 0.2], [0, 0]) is None
    fpr, tpr = roc_curve([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert fpr[0] == tpr[0] == 0.0 and fpr[-1] == tpr[-1] == 1.0 and (0.0, 1.0) in set(zip(fpr, tpr))


def test_burstguard_replay_rule():
    from burstguard.defenses.burstguard import replay_flags
    drift = [[(0.05, 0.05 + 0.06 * r)] * 4 for r in range(30)]       # smooth benign anchor drift up to ~1.8
    fixed = replay_flags(drift, 0.15, 0.5, 5.0, 5, anchor_adaptive=False)
    adapt = replay_flags(drift, 0.15, 0.5, 5.0, 5, anchor_adaptive=True)
    assert any(any(f) for f in fixed) and not any(any(f) for f in adapt)
    attack = [list(u) for u in drift]
    attack[10][2] = (3.0, 3.5)                                          # one injected unit
    flags = replay_flags(attack, 0.15, 0.5, 5.0, 5, anchor_adaptive=True)
    assert flags[10] == [False, False, True, False] and sum(sum(f) for f in flags) == 1


def test_burstguard_batched_search_matches_sequential():
    """score() runs all (direction, restart) chains as one batch: same value as the chains one after another."""
    import torch
    from burstguard.config import load_config
    from burstguard import model as M
    from burstguard.defenses.burstguard import BurstGuard
    cfg = load_config(None, ['model.source="random"', "burstguard.steps=3", "burstguard.restarts=2",
                             "burstguard.probe_windows=8"])
    m = M.configure_peft(M.load_model(cfg)[0], cfg)
    th = M.get_vec(m)
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((8, int(cfg["data"]["context"]), 1)).cumsum(1) * 0.1 + 5).astype(np.float32)
    bg = BurstGuard(cfg, m, X, X.std(1).ravel(), th)
    cand = th + rng.standard_normal(th.size).astype(np.float32) * 0.01

    def sequential(seed):
        b = bg.bg
        M.set_params_vec(bg._pc, cand); M.set_params_vec(bg._pr, th)
        W, K, R, amax, step = int(b["width_max"]), int(b["steps"]), int(b["restarts"]), float(b["amplitude_max"]), float(b["step_size"])
        Xt, sig, L = bg.X, bg.sig, bg.X.shape[1]
        g0 = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            bc, br = M.forward(bg.cand, Xt, bg.freq), M.forward(bg.ref, Xt, bg.freq)
        best = -1e9
        for sign in (1.0, -1.0):
            for _ in range(R):
                g = torch.rand(Xt.shape[0], W, 1, generator=g0).requires_grad_(True)
                for k in range(K + 1):
                    d = torch.zeros_like(Xt); d[:, L - W:, :] = g * amax * sig
                    v = ((sign * (bc - M.forward(bg.cand, Xt + d, bg.freq)) / (bc.abs() + 1e-3)).mean()
                         - (sign * (br - M.forward(bg.ref, Xt + d, bg.freq)) / (br.abs() + 1e-3)).mean())
                    if k == K:
                        best = max(best, float(v)); break
                    v.backward()
                    with torch.no_grad():
                        g += step * g.grad.sign(); g.clamp_(0.0, 1.0); g.grad.zero_()
        return best
    assert bg.score(cand, th, seed=5) == pytest.approx(sequential(5), abs=1e-6)


def test_burstguard_rel_floor():
    """rel_floor = 0 is the plain relative change; with near-zero forecasts the floor keeps the score bounded."""
    from burstguard.config import load_config
    from burstguard import model as M
    from burstguard.defenses.burstguard import BurstGuard
    rng = np.random.default_rng(1)
    for floor in (0.0, 0.5):
        cfg = load_config(None, ['model.source="random"', "burstguard.steps=2", "burstguard.probe_windows=8",
                                 f"burstguard.rel_floor={floor}"])
        m = M.configure_peft(M.load_model(cfg)[0], cfg)
        th = M.get_vec(m)
        X = np.abs(rng.standard_normal((8, int(cfg["data"]["context"]), 1)) * 0.01).astype(np.float32)  # ~0 level
        bg = BurstGuard(cfg, m, X, np.ones(8, np.float32), th)
        cand = th + rng.standard_normal(th.size).astype(np.float32) * 0.05
        s = bg.score(cand, th, seed=3)
        assert np.isfinite(s)
        if floor == 0.0:
            s0 = s
        else:
            assert abs(s) < abs(s0)          # the floor damps the blow-up of near-zero forecasts


def test_threshold_file(tmp_path):
    """burstguard.threshold_file: thresholds calibrated per data.path are applied when the config is resolved."""
    from burstguard.config import dump_toml, load_config
    f = tmp_path / "thr.toml"
    f.write_text('[thresholds."shared:data/processed/milan_sms_d1.npz"]\nthreshold = 1.25\nthreshold_anchor = 2.5\n'
                 'anchor_adaptive = true\n')
    ov = {"data.path": "shared:data/processed/milan_sms_d1.npz", "burstguard.threshold_file": str(f)}
    cfg = load_config(None, ov)
    assert cfg["burstguard"]["threshold"] == 1.25 and cfg["burstguard"]["threshold_anchor"] == 2.5
    assert cfg["burstguard"]["threshold_source"].endswith("milan_sms_d1.npz")
    (tmp_path / "job.toml").write_text(dump_toml(cfg))          # a job's saved config keeps the resolved values
    f.unlink()
    again = load_config(str(tmp_path / "job.toml"))
    assert again["burstguard"]["threshold"] == 1.25
    with pytest.raises(FileNotFoundError):
        load_config(None, ov)
    f.write_text('[thresholds."other"]\nthreshold = 1.0\nthreshold_anchor = 1.0\n')
    with pytest.raises(ValueError, match="no calibrated thresholds"):
        load_config(None, ov)


def test_regroup_threshold_scale():
    """Regroup units (means of ~n/G updates) are screened against sqrt(G)-scaled thresholds; accepted regroup scores
    enter the adaptive history on the aggregate's scale."""
    from burstguard.defenses.burstguard import BurstGuard
    cfg = load_config(None, {"burstguard.regroup_threshold_scale": "sqrt", "burstguard.threshold": 0.05,
                             "burstguard.threshold_anchor": 0.15, "burstguard.anchor_adaptive": False})
    bg = object.__new__(BurstGuard)
    bg.bg, bg.history, bg.history_anchor, bg.threads = cfg["burstguard"], [], [], 1
    bg.last = bg.anchor = np.zeros(3, dtype=np.float32)
    assert bg.regroup_scale(5) == pytest.approx(5 ** 0.5) and bg.regroup_scale(4) == pytest.approx(2.0)
    vals = iter([0.08, 0.1, 0.30, 0.1])       # (S_last, S_anchor) of two units
    bg.score = lambda cand, ref, seed=0: next(vals)
    units = [np.zeros(3, dtype=np.float32)] * 2
    flags, info = bg.screen(np.zeros(3, dtype=np.float32), units, rnd=1, scale=bg.regroup_scale(4))
    assert flags == [False, True]            # 0.08 < 2 x 0.05; 0.30 > 0.10
    assert info["threshold"] == pytest.approx(0.1) and info["threshold_scale"] == pytest.approx(2.0)
    bg.score = lambda cand, ref, seed=0: 0.08
    assert bg.screen(np.zeros(3, dtype=np.float32), units[:1], rnd=2)[0] == [True]   # aggregate: unscaled
    with pytest.raises(ValueError, match="regroup_threshold_scale"):
        load_config(None, {"burstguard.regroup_threshold_scale": 0.5})
    assert load_config(None)["burstguard"]["regroup_threshold_scale"] == 1.0   # default = v0.3.5 behaviour


def test_wait_for_calibrated_thresholds(tmp_path):
    """X5 jobs wait until the threshold file has an entry for every data.path they use."""
    from burstguard.experiments.grid import expand, missing_inputs, wait_inputs
    f = tmp_path / "thr.toml"
    g = {"grid": {"name": "t", "base_config": "configs/base.toml", "seeds": [0]},
         "fixed": {"burstguard.threshold_file": str(f), "defense.name": "burstguard"},
         "factors": {"data.path": ["shared:a.npz", "shared:b.npz"]}}
    jobs = expand(g)
    assert len(missing_inputs(g, jobs)) == 2
    f.write_text('[thresholds."shared:a.npz"]\nthreshold = 1.0\nthreshold_anchor = 2.0\n')
    miss = missing_inputs(g, jobs)
    assert len(miss) == 1 and "shared:b.npz" in miss[0]
    msgs = []
    assert wait_inputs(g, jobs, log=msgs.append, timeout_h=0.0, poll_s=0.0) is False and msgs
    f.write_text(f.read_text() + '[thresholds."shared:b.npz"]\nthreshold = 1.0\nthreshold_anchor = 2.0\n')
    assert missing_inputs(g, jobs) == [] and wait_inputs(g, jobs, log=msgs.append, timeout_h=0.0) is True
    assert missing_inputs(g, [{"overrides": {"data.path": "x"}}]) == []    # grids without a threshold file


def test_calibrate_writes_fixed_anchor(tmp_path):
    """calibrate --where matches config values that are not overrides; --write-thresholds stores the fixed anchor
    rule (margin x largest clean anchor score of the whole run) by default."""
    import argparse
    import json as _json
    import tomllib
    from burstguard.cli import cmd_calibrate
    gdir = tmp_path / "G"
    for jid, amax in (("j1", 0.10), ("j2", 0.08)):
        jd = gdir / jid
        jd.mkdir(parents=True)
        (jd / "config.toml").write_text('[data]\npath = "shared:x.npz"\n[burstguard]\nthreshold = 0.08\n'
                                        'threshold_anchor = 0.3\nmad_k = 5.0\nhistory = 5\nanchor_adaptive = true\n')
        (jd / "result.json").write_text(_json.dumps({"overrides": {"burstguard.width_max": 18}}))
        rows = [{"kind": "round", "round": r, "attack_active": False,
                 "burstguard": {"score_last": [0.02 + 0.001 * r], "score_anchor": [amax * r / 10]}} for r in range(1, 11)]
        (jd / "metrics.jsonl").write_text("\n".join(_json.dumps(x) for x in rows) + "\n")

    class Ctx:
        def save_json(self, *a, **k): pass
        def set(self, **k): pass
    thr = tmp_path / "thr.toml"
    a = argparse.Namespace(target=str(gdir), quick=False, margin=1.5, write_thresholds=str(thr), anchor="fixed",
                           where=["data.path=shared:x.npz", "burstguard.width_max=18"])
    cmd_calibrate(a, Ctx())
    ent = tomllib.loads(thr.read_text())["thresholds"]["shared:x.npz"]
    assert ent["anchor_adaptive"] is False and ent["threshold_anchor"] == pytest.approx(0.15)
    assert ent["threshold"] == pytest.approx(round(1.5 * 0.03, 3)) and ent["clean_jobs"] == 2
    a.where = ["data.path=shared:other.npz"]
    with pytest.raises(RuntimeError, match="no BurstGuard scores"):
        cmd_calibrate(a, Ctx())


def test_rule_replay_from_monitor_run(tmp_path):
    """Monitor-mode runs: other decision rules are replayed on the logged scores (E9 without extra runs)."""
    from burstguard.engines.common import summarize
    (tmp_path / "config.toml").write_text('[burstguard]\nmode = "monitor"\nreference = "both"\nthreshold = 0.1\n'
                                          'threshold_anchor = 0.3\nmad_k = 5.0\nhistory = 5\n'
                                          'anchor_adaptive = true\n')
    lines = []
    for r in range(1, 11):
        att = r >= 6
        sl, sa = (0.5, 0.2) if r == 6 else (0.02, 0.4 if att else 0.05)   # injection, then slow accumulation
        would = sl > 0.1 or sa > 0.3
        lines.append({"kind": "round", "round": r, "attack_active": att, "screened_has_malicious": att,
                      "round_wall_s": 1.0, "server_defense_cpu_s": 0.1,
                      "burstguard": {"scores": [max(sl - 0.1, sa - 0.3) + 0.1], "score_last": [sl],
                                     "score_anchor": [sa], "would_flag": [would], "cpu_s": 0.1}})
    (tmp_path / "metrics.jsonl").write_text("\n".join(__import__("json").dumps(x) for x in lines) + "\n")
    det = summarize(tmp_path)["detection"]
    assert det["rule_replay_agree"] == 1.0
    rf = det["rule_flags"]
    assert rf["last"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]          # S_last only sees the injection round
    assert rf["anchor"] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]        # S_anchor only the accumulated rounds
    assert rf["fixed_anchor"] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_generalization_grids():
    from burstguard.experiments.grid import expand, load_grid
    from burstguard.paths import ROOT
    x1 = expand(load_grid(ROOT / "experiments" / "X1_traffic_clean.toml"))
    x2 = expand(load_grid(ROOT / "experiments" / "X2_traffic_defenses.toml"))
    assert len(x1) == 18 and len(x2) == 64
    paths = {j["overrides"]["data.path"] for j in x1 + x2}
    assert paths == {"shared:data/processed/milan_sms_d1.npz", "shared:data/processed/milan_call_d1.npz"}
    assert not any(j["overrides"].get("train.peft") == "lora" and j["overrides"]["run.seed"] == 1 for j in x2)
    x4 = expand(load_grid(ROOT / "experiments" / "X4_traffic_bg_monitor.toml"))   # v0.3.9: clean calibration only
    assert len(x4) == 6 and all(j["overrides"]["burstguard.rel_floor"] == 0.1 for j in x4)
    assert all(j["overrides"]["attack.type"] == "none" for j in x4)
    x5 = expand(load_grid(ROOT / "experiments" / "X5_traffic_bg_enforce.toml"))
    assert len(x5) == 18 and all(j["overrides"]["burstguard.regroup_threshold_scale"] == "sqrt" for j in x5)
    seeds4 = {(j["overrides"]["data.path"], j["overrides"]["train.peft"], j["overrides"]["run.seed"]) for j in x4}
    clean5 = [j["overrides"] for j in x5 if j["overrides"]["attack.type"] == "none"]
    assert len(clean5) == 6 and not any((o["data.path"], o["train.peft"], o["run.seed"]) in seeds4 for o in clean5)
    x6 = expand(load_grid(ROOT / "experiments" / "X6_offset_calibrated.toml"))
    assert len(x6) == 16 and all(j["overrides"]["burstguard.width_max"] == 18 for j in x6)
    assert all(j["overrides"]["burstguard.anchor_adaptive"] is False for j in x6)
    assert {j["overrides"]["run.seed"] for j in x6 if j["overrides"]["attack.type"] == "none"} == {3, 4}
    assert {j["overrides"]["run.seed"] for j in x6 if j["overrides"]["attack.type"] == "scale"} == {0, 1, 2}
    x7 = expand(load_grid(ROOT / "experiments" / "X7_poison_defenses.toml"))
    assert len(x7) == 43 and all(j["overrides"]["attack.type"] == "poison" for j in x7)
    x8 = expand(load_grid(ROOT / "experiments" / "X8_bg_calibrated.toml"))
    assert len(x8) == 18 and all(j["overrides"]["burstguard.anchor_adaptive"] is False for j in x8)
    assert {j["overrides"]["run.seed"] for j in x8 if j["overrides"]["attack.type"] == "none"} == {2, 3, 4}
    assert {(j["overrides"]["train.peft"], j["overrides"]["run.seed"]) for j in x8
            if j["overrides"]["attack.type"] != "none"} == {("head", 0), ("head", 1), ("head", 2), ("lora", 0),
                                                            ("lora", 1)}
    x3 = expand(load_grid(ROOT / "experiments" / "X3_trigger_offset.toml"))
    assert len(x3) == 34 and {j["overrides"]["burstguard.width_max"] for j in x3} == {6, 18}
    assert not any(j["overrides"]["attack.type"] == "none" and j["overrides"]["trigger.offset"] == 12 for j in x3)


def test_plan_cost_model():
    from burstguard.experiments import plan as PL
    base = {"kind": "fl", "rounds": 20, "units": 1, "effort": 0.0, "engine": "local", "attacked": False}
    fin = [{**base, "peft": "head", "bg": False, "wall_s": 200.0, "clients_wall_s": 2.0, "workers": 4},
           {**base, "peft": "head", "bg": True, "effort": 1.0, "wall_s": 400.0, "clients_wall_s": 2.0, "workers": 2},
           {**base, "peft": "lora", "bg": False, "wall_s": 1600.0, "clients_wall_s": 70.0, "workers": 4}]
    m = PL.fit(fin)
    assert m["c"]["head"] == pytest.approx(1.5)          # median of 2*4/4 and 2*2/4 (client part with 4 workers)
    assert m["s0"]["head"] == pytest.approx(8.0)         # 200/20 - 2
    assert m["s_bg"] == pytest.approx(10.0)              # 400/20 - 2 - 8, one unit at the default effort
    assert PL.client_round(m, "decoder") == pytest.approx(70.0 * 0.5)    # derived from LoRA via ratios
    assert m["attack_src"] == "prior"
    fin.append({**base, "peft": "head", "bg": False, "attacked": True, "wall_s": 220.0, "clients_wall_s": 3.0,
                "workers": 4})
    m = PL.fit(fin)
    # v0.3.16: the attacked-round factor is per mode - measured for head (3 / 1.5 = 2), prior for LoRA
    assert PL.attack_factor(m, "head") == pytest.approx(2.0) and m["attack_src"].startswith("measured")
    assert PL.attack_factor(m, "lora") == pytest.approx(PL.PRIOR_ATTACK)
    ov = {"train.peft": "head", "fl.rounds": 30, "defense.name": "burstguard", "secagg.enabled": True,
          "secagg.groups": 5}
    srv, cli = PL.job_parts(m, ov, "fl")
    assert srv == pytest.approx(30 * (8 + 10 * 5)) and cli == pytest.approx(30 * 1.5)
    assert PL.job_seconds(m, ov, "fl", weight=1) == pytest.approx(srv + cli * 4)   # 1 worker: clients 4x slower
    # search effort: 64 probe windows and 20 steps cost more than the default (32 windows, 8 steps)
    f = PL._features({"defense.name": "burstguard", "burstguard.probe_windows": 64, "burstguard.steps": 20,
                      "burstguard.restarts": 2}, "fl")
    assert f["effort"] == pytest.approx(64 * (2 * 2 * 122 + 2) / (32 * (2 * 50 + 2)))
    # enforce + regroup: attacked rounds 6..15 of 40 -> 1 + 10/40 * 5 screened units
    f = PL._features({"defense.name": "burstguard", "attack.type": "scale", "attack.start_round": 6,
                      "attack.end_round": 16, "fl.rounds": 40}, "fl")
    assert f["units"] == pytest.approx(1 + 10 / 40 * 5)


def test_server_bound_jobs_spread_over_weak_machines():
    """Machine-aware assignment: server-bound jobs (head-only, screening) run as fast on a 1-worker machine, so
    they are spread evenly; client-bound jobs still go mostly to the 4-worker machine."""
    from burstguard.experiments.grid import assign_weighted
    srv = [{"id": f"s{i}", "cost": 100.0, "cost_server": 100.0, "cost_client": 0.0} for i in range(8)]
    cli = [{"id": f"c{i}", "cost": 100.0, "cost_server": 0.0, "cost_client": 100.0} for i in range(10)]
    a = assign_weighted(srv, [4, 1])
    assert a.count(0) == a.count(1) == 4
    b = assign_weighted(cli, [4, 1])
    assert b.count(0) == 8 and b.count(1) == 2


def test_weighted_shards():
    from burstguard.experiments.grid import parse_shard, shard, shard_from_spec
    rng = np.random.default_rng(3)
    jobs = [{"id": f"j{i:03d}", "cost": float(c)} for i, c in enumerate(rng.choice([30, 170, 400, 900], size=300))]
    # unweighted: unchanged round-robin over (-cost, id)
    order = sorted(jobs, key=lambda j: (-j["cost"], j["id"]))
    assert [j["id"] for j in shard(jobs, 2, 5)] == [j["id"] for i, j in enumerate(order) if i % 5 == 1]
    assert parse_shard("2/5") == (2, 5, None)
    w = [4, 1, 3, 1, 3]
    assert parse_shard("3/5@4,1,3,1,3") == (3, 5, [4.0, 1.0, 3.0, 1.0, 3.0])
    parts = [shard_from_spec(jobs, f"{k}/5@4,1,3,1,3") for k in range(1, 6)]
    ids = sorted(j["id"] for p in parts for j in p)
    assert ids == sorted(j["id"] for j in jobs)                      # every job exactly once
    load = [sum(j["cost"] for j in p) for p in parts]
    share = [x / sum(load) for x in load]
    for s_, wi in zip(share, w):
        assert abs(s_ - wi / sum(w)) < 0.03                          # cost proportional to weight
    assert [j["id"] for j in shard_from_spec(jobs, "1/5@4,1,3,1,3")] == [j["id"] for j in parts[0]]  # deterministic
    for bad in ("0/5", "6/5", "2/5@1,2", "2/5@1,1,1,1,0", "x"):
        with pytest.raises(ValueError):
            parse_shard(bad)


def test_exp_code_mixing():
    from burstguard.experiments.analyze import exp_code
    assert exp_code({"version": "0.3.2"}) == exp_code({"version": "0.3.3", "exp_code": "0.3.0"}) == "0.3.0"
    assert exp_code({"version": "0.2.2"}) == "0.2.2"
