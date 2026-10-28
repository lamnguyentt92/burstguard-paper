#!/usr/bin/env bash
# =============================================================================
# BurstGuard v0.3.16 - the whole campaign in ONE command, on the controller (after git clone + setup_venv):
#
#   bash scripts/run_v0316.sh           # starts in the background (survives logout) and follows the log
#   bash scripts/run_v0316.sh --no-tail # start only
#
# Steps (stops at the first failure, prints CHAIN-FAILED):
#   tests -> Trentino download + prepare (Milan data are reused if present, else downloaded) -> model cache ->
#   smoke -> copy data to the VMs -> install + smoke on the VMs -> import v0.3.15 (measured cost model) ->
#   start R16 R17 R18 on 5 VMs -> time estimate -> finish (wait, analyse, campaign report)
# Log: logs/v0316_all.log. Remaining time at any moment: bash scripts/cluster.sh eta --tier v0316
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SHARED="${BURSTGUARD_SHARED:-$HOME/burstguard-shared}"
LOG="logs/v0316_all.log"
if [ ! -f "$SHARED/CONTROLLER" ]; then
  echo "WRONG MACHINE: this is not the controller ($SHARED/CONTROLLER missing). Run it on the controller VM."
  exit 2
fi
if [ ! -x .venv/bin/burstguard ]; then
  echo "missing .venv - run first: bash scripts/setup_venv.sh && source .venv/bin/activate"; exit 2
fi
if [ "${1:-}" != "--inner" ]; then
  mkdir -p logs
  if pgrep -f "run_v0316.sh --inner" >/dev/null; then echo "already running - follow it: tail -f $LOG"; exit 0; fi
  nohup setsid bash "$0" --inner > "$LOG" 2>&1 < /dev/null &
  echo "started in the background (pid $!); log: $ROOT/$LOG"
  echo "remaining time at any moment: bash scripts/cluster.sh eta --tier v0316"
  [ "${1:-}" = "--no-tail" ] && exit 0
  sleep 2; exec tail -f "$LOG"
fi

# ---------------------------------------------------------------- inner: the chain
source .venv/bin/activate
export PYTHONUNBUFFERED=1     # progress lines (downloads, rounds) reach the log immediately
step() { echo; echo "######## $(date -u +%H:%M:%S) $*"; }
fail() { echo; echo "CHAIN-FAILED at: $*"; echo "send this log: $ROOT/$LOG"; exit 1; }
P="$SHARED/data/processed"
step "1/10 tests";                         python -m pytest                                      || fail "tests"
step "2/10 Milan data (skipped if present)"
if [ ! -f "$P/milan_d1.npz" ] || [ ! -f "$P/milan_sms_d1.npz" ]; then
  burstguard data download && burstguard data prepare --activity internet,sms             || fail "Milan data"
else echo "milan_d1.npz and milan_sms_d1.npz present"; fi
step "3/10 Trentino data"
if [ ! -f "$P/trentino_d1.npz" ]; then
  burstguard data download --region tn && burstguard data prepare --region tn --activity internet \
                                                                                           || fail "Trentino data (guestbook? see README.md, section Data)"
else echo "trentino_d1.npz present"; fi
burstguard data info --path shared:data/processed/trentino_d1.npz | tail -n 30
step "4/10 model cache";                   burstguard model-download                             || fail "model download"
step "5/10 smoke (this machine)";          burstguard smoke                                      || fail "smoke"
step "6/10 copy data to the VMs";          bash scripts/cluster.sh push-data                     || fail "push-data"
step "7/10 install + smoke on the VMs";    bash scripts/cluster.sh install && bash scripts/cluster.sh smoke || fail "install/smoke on the VMs"
step "8/10 import v0.3.15 (measured cost model)"
bash scripts/cluster.sh import 0.3.15 || echo "warning: import failed - the time estimate uses priors until jobs finish"
step "9/10 start R16 R17 R18";             bash scripts/cluster.sh start "--tier v0316"          || fail "start"
sleep 120
bash scripts/cluster.sh eta --tier v0316
step "10/10 finish (waits for all grids, then analysis + campaign report)"
bash scripts/cluster.sh --bg finish R16_poison_gaps R17_participation R18_trentino            || fail "finish"
echo; echo "ALL-STARTED $(date -u +%H:%M:%S) - follow: tail -f \$(ls -t logs/cluster_finish_*.log | head -1)"
