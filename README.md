# BurstGuard

Source code and experiment framework for the paper

> **BurstGuard: Defending Federated Mobile-Traffic Forecasting against Burst-Triggered Backdoors under Secure Aggregation**
> Vinh Thinh Le, Tran Thanh Lam Nguyen
> Submitted to *Mobile Networks and Applications* (Springer), topical collection *Trustworthy and Efficient
> Time-Series Intelligence at the Mobile Edge*.

This repository contains release **0.3.16**, the final version of the code. Every result in the paper was produced
with it or with an earlier release whose code paths it reproduces bit for bit (see [Reproducibility](#reproducibility)).

---

## 1. What the paper studies

Mobile operators can fine-tune a small time-series foundation model (IBM's Tiny Time Mixer, TTM-R2) with
federated learning, so that every site keeps its traffic data. Under **secure aggregation** the server only sees
the *sum* of the sites' updates. The paper shows that this setting is open to a **burst-triggered backdoor**:

- two malicious sites out of twenty teach the shared model that an input ending with a short surge (30 minutes)
  should be forecast at a fraction of the real demand;
- the model still forecasts well on ordinary inputs, so the backdoor is hard to notice;
- real traffic surges (crowds, events) fire it, so the operator under-provisions exactly when demand peaks.

**BurstGuard** is a server-side defence that needs only the aggregated model:

1. **Burst search.** For every candidate aggregate, a projected sign-gradient ascent searches for the short,
   non-negative, bounded burst at the end of public probe windows that most changes the model's forecasts.
2. **Two references.** The change is measured relative to the last accepted model (`S_last`, catches a new
   malicious update) and to the public pre-trained model (`S_anc`, catches a slowly accumulated backdoor).
3. **Public calibration.** Before round 1 the server simulates a clean federation on its own public probe areas
   (pseudo-clients, no client data) and sets the thresholds to 1.5 x the largest simulated score.
4. **Regrouping with verification.** A flagged round is re-aggregated in `G` random secure groups; the groups that
   pass are averaged, the result is screened again, and the round is rejected if it is still flagged.

The *recommended configuration* used for the headline results is:

```toml
[burstguard]
calibrate_guard = "public"        # thresholds from the server's public simulation
calibrate_rounds = 0              # no clean client rounds needed
calibrate_public_rounds = 20      # = fl.rounds (the training horizon)
regroup_threshold_scale = "sqrt"  # group thresholds = sqrt(G) x cap
regroup_verify = true             # re-screen the re-aggregated groups
```

On the Milan and Trentino telecommunications data, BurstGuard in this configuration flagged all 281 attacked
aggregates of 16 two-attacker runs (both regions and half participation), let no malicious update into the model
and kept the harm at the clean level, while flagging 10 of 273 attacker-free rounds (3.7 %) without loss of
forecast accuracy. Screening costs 4.5–7.3 s of server CPU time per clean round. The paper reports the remaining
weaknesses (low-amplitude adapter poisoning, triggers placed one day earlier, emulated regrouping) in detail.

---

## 2. Repository layout

```
src/burstguard/
  cli.py                 command line: `burstguard <command>`
  config.py              defaults, TOML files and --set overrides, validation
  data/                  Telecom Italia download (Harvard Dataverse), Milan/Trentino readers, synthetic data, windows
  model.py, train.py     TTM-R2 loading, fine-tuning modes (head, decoder, LoRA, full), local training
  attacks/               burst trigger, malicious clients (poison, scale, constrain_scale, neurotoxin, adaptive,
                         evasive = score-aware, budget = threshold riding)
  defenses/
    burstguard.py        BurstGuard: burst search, scores, thresholds, decision, regrouping and verification
    public_calibration.py  server-side simulation of a clean federation on public probe areas
    aggregators.py       FedAvg, Krum, Multi-Krum, trimmed mean, median, FLAME, FLTrust, Clip+DP,
                         aggregate clipping, RFA, norm bound
    secagg.py            grouped secure aggregation (clipping + quantisation, as in SecAgg+)
    lora_reset.py        LoRA reset baseline
  fedcore.py             client and server logic shared by both engines
  engines/local.py       deterministic local engine (process pool)
  engines/flower.py      Flower engine with the real SecAggPlusWorkflow
  experiments/           grid runner (deterministic job ids, sharding, resume), analysis and statistics,
                         time planning, paper tables and figures
  metrics.py             MASE, UPR_trig (under-provisioning rate on triggered inputs), ASR, natural-surge metrics
experiments/*.toml       the 37 experiment grids of the paper (P0, E0-E9, X1-X8, R1-R18)
configs/base.toml        base configuration shared by all grids
scripts/                 environment setup, multi-VM campaign driver, log collection
tests/                   58 unit, integration and regression tests
```

---

## 3. Requirements

- Linux (Ubuntu), CPU only; no GPU is used.
- Python 3.11–3.13. If the machine has none, `scripts/setup_venv.sh` installs a local Python 3.12 with `uv`.
- About 8 GB of RAM per machine, and disk space for the raw data, the model cache and the results.
- Internet access for the first installation, the dataset and the model download.

Main libraries (pinned in `requirements.txt`): PyTorch 2.11 (CPU), `granite-tsfm` 0.3.10, `transformers` 5.20,
`peft` 0.21, Flower 1.41 with Ray 2.55, NumPy, pandas, SciPy, matplotlib.

---

## 4. Installation

```bash
git clone https://github.com/lamnguyentt92/burstguard-paper.git ~/burstguard-v0.3.16
cd ~/burstguard-v0.3.16
bash scripts/setup_venv.sh            # isolated virtual environment in .venv (no sudo needed)
source .venv/bin/activate
burstguard doctor                     # environment checks
burstguard smoke                      # end-to-end test on synthetic data (no network, no real data)
python -m pytest                      # 58 tests
```

Large shared files (raw and processed data, Hugging Face model cache) are kept in `$BURSTGUARD_SHARED`
(default `~/burstguard-shared`), outside the checkout.

---

## 5. Data

The experiments use the **Telecom Italia Big Data Challenge** telecommunications data (SMS, call and internet
activity in 10-minute slots, November 2013 – January 2014) published on Harvard Dataverse under the Open Database
License (Barlacchi et al., *Scientific Data* 2, 150055, 2015):
Milan [doi:10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV) and
Trentino [doi:10.7910/DVN/QLCABU](https://doi.org/10.7910/DVN/QLCABU).
The data are not included in this repository.

Harvard Dataverse asks for a guestbook response before it serves these files. BurstGuard fills it in for you:

```bash
burstguard data guestbook                      # prints the questions, writes ~/burstguard-shared/dataverse_guestbook.toml
nano ~/burstguard-shared/dataverse_guestbook.toml   # fill in your name, e-mail and the required answers
# if Dataverse answers 401/403: export DATAVERSE_API_TOKEN=<token of your Dataverse account>
```

The token is read only from the environment or from that file, never from the command line. Then:

```bash
burstguard data download                                   # Milan raw files
burstguard data prepare --activity internet,sms,call       # -> milan_d1.npz, milan_sms_d1.npz, milan_call_d1.npz
burstguard data download --region tn                       # Trentino raw files
burstguard data prepare --region tn --activity internet    # -> trentino_d1.npz
burstguard data info --path shared:data/processed/milan_d1.npz
burstguard model-download                                  # TTM-R2 into the shared Hugging Face cache
```

Milan: 20 client sites and 8 public probe areas, each a block of 5 x 5 grid squares, interleaved over the city.
Trentino: 20 sites and 8 probe areas, each a single grid square active in at least 95 % of the slots, chosen by
internet traffic. Splits: train 1 Nov – 7 Dec, validation 8 – 14 Dec, test 15 – 24 Dec, holiday 25 Dec – 1 Jan.

---

## 6. Running experiments

### 6.1 Single runs

Every command takes `--config FILE`, any number of `--set key=value` overrides and `--seed N`:

```bash
burstguard zeroshot                                              # seasonal naive + TTM zero-shot
burstguard local                                                 # local-only fine-tuning baseline
burstguard fl                                                    # clean federated fine-tuning (FedAvg)
burstguard fl --set attack.type=scale --set attack.boost=3.0     # backdoor attack under FedAvg
burstguard fl --set attack.type=scale --set attack.boost=3.0 \
              --set defense.name=burstguard \
              --set burstguard.calibrate_guard=public --set burstguard.calibrate_rounds=0 \
              --set burstguard.calibrate_public_rounds=20 \
              --set burstguard.regroup_threshold_scale=sqrt --set burstguard.regroup_verify=true
burstguard fl --set train.peft=lora                               # LoRA instead of head fine-tuning
burstguard equivalence                                           # local engine vs Flower engine
```

Each command writes its configuration, logs and metrics to `runs/<time>_v0.3.16_<command>/`.

### 6.2 Experiment grids

A grid (`experiments/*.toml`) expands into jobs with deterministic ids. Running a grid again skips finished jobs and
retries failed ones; results go to `results/v0.3.16/<grid>/<job id>/` (`result.json`, `metrics.jsonl`, `rounds.jsonl`).

```bash
burstguard grid list   experiments/R13_recommended.toml
burstguard grid run    experiments/R13_recommended.toml            # whole grid on this machine
burstguard grid run    experiments/R13_recommended.toml --shard 2/5   # one of five shards
burstguard grid status experiments/R13_recommended.toml
burstguard analyze     experiments/R13_recommended.toml            # tables, statistics, figures
burstguard paper                                                 # paper tables, figures and numbers.json
bash scripts/run_grids.sh 1/1 --tier v0316                       # several grids in order (see the table below)
```

`burstguard plan --tier <tiers>` estimates the run time of grids from the jobs already finished.

### 6.3 Grids of the paper

37 grids, 990 runs (federated runs plus zero-shot, local-only and centralised baselines), about 370 job-hours on
CPU. Tiers are the names accepted by `run_grids.sh --tier`, `cluster.sh start` and `burstguard plan --tier`.

| Tier | Grids | Purpose |
|---|---|---|
| `pilot` | P0 | pilot: scores, timing, attacker strength (monitor mode) |
| `calibration` | E0 | clean calibration in monitor mode |
| `1` | E1, E2, E3, E7, E9 | clean utility; attacks under FedAvg; defences; grouped secure aggregation; ablation |
| `2` | E4, E5, E6 | sensitivity (attackers, amplitude); backdoor lifespan (40 rounds); adaptive attacks |
| `system` | E8 | system cost in Flower (run on one idle machine) |
| `generalization` | X1, X2 | SMS and call traffic: utility and defences |
| `offset` | X3 | trigger placed before the screened window, window W = 6 / 18 |
| `traffic_bg`, `traffic_bg2` | X4, X5 | SMS/call calibration (monitor) and calibrated BurstGuard |
| `offset2` | X6 | W = 18 with calibrated thresholds |
| `poison` | X7 | plain data poisoning against every defence |
| `calibrated` | X8 | calibrated BurstGuard on internet traffic |
| `revision` | R1–R10 | five seeds and natural surges; 100/60 rounds; adaptive attackers against the calibrated rule; norm bound, aggregate clipping, RFA; six attackers; online calibration; dip and one-day-offset triggers; regrouping variants; decoder/full fine-tuning; cost with and without SecAgg+ |
| `v0313` | R11, R12 | public calibration guard; many attackers with verification |
| `v0314` | R13 | the recommended configuration |
| `v0315` | R14, R15 | one-day trigger offset with W = 150; LoRA seeds 3–4 |
| `v0316` | R16, R17, R18 | poisoning gaps; half participation; Trentino with public calibration only |

The grids that use SMS/call data (X1, X2, X4, X5) need `milan_sms_d1.npz` and `milan_call_d1.npz`; X5 waits for the
thresholds that `cluster.sh finish` writes after X4. R18 needs `trentino_d1.npz`.

### 6.4 Several machines

The paper's campaign ran on five CPU-only virtual machines. `scripts/cluster.sh` drives all
of them from one controller over SSH; jobs run in the background with `nohup` and survive logouts.

```bash
bash scripts/cluster_bootstrap.sh controller        # on the controller: SSH key, prints a one-line command
#   ... run that command on every worker, then:
bash scripts/cluster_bootstrap.sh discover          # finds the workers, writes ~/burstguard-shared/cluster.hosts
bash scripts/cluster.sh check                       # SSH, Python, CPU/RAM/disk on every machine
bash scripts/cluster.sh push-data                   # processed datasets + model cache to every worker
bash scripts/cluster.sh install                     # clone this release + setup_venv.sh on every worker
bash scripts/cluster.sh smoke
bash scripts/cluster.sh start "--tier pilot,calibration,1,2"   # weighted shards, one per machine
bash scripts/cluster.sh status                      # progress on every machine
bash scripts/cluster.sh eta --tier 1,2              # remaining time from the running jobs' own round times
bash scripts/cluster.sh --bg finish [GRIDS]         # wait for the grids (default E0-X8), pull results, analyse, report
bash scripts/cluster.sh stop                        # stop everywhere (start again = resume)
```

`scripts/run_v0316.sh` runs the final round (tier `v0316`) end to end in one background command: tests, Trentino
data, model cache, smoke tests, data and installation on the workers, start, time estimate and final analysis.

### 6.5 Logs

`bash scripts/collect_logs.sh` packs the environment, the run logs and the result summaries into
`reports/bg_report_v0.3.16_<host>_<time>.txt` and `.tar.gz` (personal settings such as the Dataverse guestbook and
tokens are never included). `--campaign` adds the analysis of all grids.

---

## 7. Reproducibility

- **Deterministic jobs.** A job id is a hash of the job's configuration; seeds fix data sampling,
  training, attacks and BurstGuard's search, so a job gives the same result on the same machine.
- **Releases.** Grids E0–E9 ran with release 0.3.5, X1–X8 with 0.3.6–0.3.11 and R1–R18 with 0.3.12–0.3.16. Later
  releases only add options whose defaults keep the earlier behaviour, and the experiment code version recorded in
  every result (`exp_code`) is still 0.3.5.
- **Regression test.** `tests/test_regression.py` runs a fixed scenario (training, attacks, BurstGuard scores and
  metrics) and compares it with the values computed by release 0.3.5 (`tests/data/regression_golden_v0.3.5.json`).
- **Monitor mode and replay.** With `burstguard.mode = "monitor"` BurstGuard logs its scores without acting, so other
  decision rules and thresholds can be replayed offline (`burstguard calibrate`).

---

## 8. Responsible use

The attacks in this repository exist only to evaluate defences. They run in simulation, on simulated clients and
public data, and target no real system.

## 9. Licences of third-party material

- Data: Telecom Italia Big Data Challenge, Harvard Dataverse, Open Database License (ODbL).
- Model: IBM Granite TinyTimeMixers R2 (`ibm-granite/granite-timeseries-ttm-r2`), Apache License 2.0, downloaded at
  run time.

## 10. Citation

```bibtex
@misc{le2026burstguard,
  title  = {BurstGuard: Defending Federated Mobile-Traffic Forecasting against Burst-Triggered Backdoors
            under Secure Aggregation},
  author = {Le, Vinh Thinh and Nguyen, Tran Thanh Lam},
  year   = {2026},
  note   = {Submitted to Mobile Networks and Applications. Code: https://github.com/lamnguyentt92/burstguard-paper}
}
```

Contact: Tran Thanh Lam Nguyen, Faculty of Information Technology, HCMC University of Technology and Engineering,
Ho Chi Minh City, Vietnam — lamntt@hcmute.edu.vn
