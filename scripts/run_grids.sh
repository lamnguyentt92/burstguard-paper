#!/usr/bin/env bash
# Run experiment grids one after another for one shard (one VM). Re-running the same command resumes:
# finished jobs are skipped, failed/interrupted ones are retried. To survive SSH drops start it with nohup
# (nohup bash scripts/run_grids.sh ... > logs/run.out 2>&1 &) or let scripts/cluster.sh start it.
#
#   bash scripts/run_grids.sh 2/5 --tier pilot          # P0_pilot (day 1)
#   bash scripts/run_grids.sh 2/5 --tier calibration    # E0_calibration (day 1)
#   bash scripts/run_grids.sh 2/5 --tier 1              # core paper grids: E1 E2 E3 E7 E9
#   bash scripts/run_grids.sh 2/5 --tier 2              # extended: E4 E5 E6
#   bash scripts/run_grids.sh 2/5 --tier 1,2            # both, in order
#   bash scripts/run_grids.sh 2/5 E3_defenses E7_secagg # explicit grids
#   bash scripts/run_grids.sh 1/1 --quick E3_defenses   # pipeline check
#   bash scripts/run_grids.sh 2/5@4,1,3,1,3 --tier 1    # weighted shard (scripts/cluster.sh does this for you)
#   bash scripts/run_grids.sh 2/5 --tier generalization # X1, X2: SMS and call traffic (data prepare --activity sms,call)
#   bash scripts/run_grids.sh 2/5 --tier offset         # X3: triggers outside the screened window, wider window
#   bash scripts/run_grids.sh 2/5 --tier traffic_bg,offset2,traffic_bg2
#                                    # X4 (SMS/call calibration), X6 (calibrated wide window), X5 (SMS/call enforce;
#                                    # waits for shared:bg_thresholds.toml, written by cluster.sh finish)
#   bash scripts/run_grids.sh 2/5 --tier poison         # X7: plain data poisoning vs every defense of E3
#   bash scripts/run_grids.sh 2/5 --tier calibrated     # X8: BurstGuard with calibrated thresholds (internet, W=6)
#   bash scripts/run_grids.sh 2/5 --tier revision       # v0.3.12: R1-R10
#   bash scripts/run_grids.sh 2/5 --tier v0313          # v0.3.13: R11 calibration guard, R12 many attackers
#   bash scripts/run_grids.sh 2/5 --tier v0314          # v0.3.14: R13 recommended configuration
#   bash scripts/run_grids.sh 2/5 --tier v0315          # v0.3.15: R14 offset-144 / W = 150, R15 LoRA seeds 3-4
#   bash scripts/run_grids.sh 2/5 --tier v0316          # v0.3.16: R16 poisoning gaps, R17 participation, R18 Trentino
#   bash scripts/run_grids.sh 2/5 --wait-idle --tier generalization
#                                    # first wait until background runs of OTHER checkouts on this machine have
#                                    # finished (their logs/cluster_*.pid processes), so two versions never share CPUs
# E8_system (timings) runs on ONE idle VM: bash scripts/run_grids.sh 1/1 --tier system
# Log: logs/run_grids_v<VERSION>_<shard>_<ts>.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VER="$(cat VERSION)"
SHARD="${1:-}"
[[ "$SHARD" =~ ^[0-9]+/[0-9]+(@[0-9.]+(,[0-9.]+)*)?$ ]] || {
  echo "usage: bash scripts/run_grids.sh K/N[@w1,...,wN] [--quick] [--tier T] [GRID ...]"; exit 2; }
shift
QUICK=(); TIERS=""; GRIDS=(); WAIT_IDLE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=(--quick) ;;
    --wait-idle) WAIT_IDLE=1 ;;
    --tier) TIERS="${2:-}"; shift ;;
    *) GRIDS+=("$1") ;;
  esac
  shift
done
tier_grids() {
  case "$1" in
    pilot) echo "P0_pilot" ;;
    calibration) echo "E0_calibration" ;;
    1) echo "E1_clean E2_attacks E3_defenses E7_secagg E9_ablation" ;;
    2) echo "E4_sensitivity E5_lifespan E6_adaptive" ;;
    system) echo "E8_system" ;;
    generalization) echo "X1_traffic_clean X2_traffic_defenses" ;;
    offset) echo "X3_trigger_offset" ;;
    traffic_bg) echo "X4_traffic_bg_monitor" ;;
    traffic_bg2) echo "X5_traffic_bg_enforce" ;;
    offset2) echo "X6_offset_calibrated" ;;
    poison) echo "X7_poison_defenses" ;;
    calibrated) echo "X8_bg_calibrated" ;;
    revision) echo "R1_headline R2_horizon R3_adaptive R6_online_calibration R4_baselines R9_modes R7_out_of_family R8_regroup R5_regime R10_cost" ;;
    v0313) echo "R11_calibration_guard R12_many_attackers" ;;
    v0314) echo "R13_recommended" ;;
    v0315) echo "R14_offset_window R15_lora_seeds" ;;
    v0316) echo "R18_trentino R16_poison_gaps R17_participation" ;;
    *) echo "" ;;
  esac
}
if [ -n "$TIERS" ]; then
  IFS=',' read -ra _T <<< "$TIERS"
  for t in "${_T[@]}"; do
    gs="$(tier_grids "$t")"; [ -n "$gs" ] || { echo "unknown tier: $t"; exit 2; }
    for g in $gs; do GRIDS+=("$g"); done
  done
fi
[[ ${#GRIDS[@]} -eq 0 ]] && GRIDS=(E1_clean E2_attacks E3_defenses E7_secagg E9_ablation E4_sensitivity E5_lifespan E6_adaptive)

if [[ ! -x .venv/bin/burstguard ]]; then
  echo "ERROR (burstguard v$VER): .venv missing - run: bash scripts/setup_venv.sh"; exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p logs
# --wait-idle: wait for background runs of other checkouts (e.g. ~/burstguard-v0.3.5/logs/cluster_bg.pid)
others_busy() {
  local pf p
  for pf in "$HOME"/burstguard*/logs/cluster_*.pid; do
    [ -f "$pf" ] || continue
    case "$(cd "$(dirname "$pf")/.." && pwd)" in "$ROOT") continue ;; esac
    p="$(cat "$pf" 2>/dev/null)"
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo "$pf (pid $p)"; return 0; fi
  done
  return 1
}
if [ "$WAIT_IDLE" = 1 ]; then
  while busy="$(others_busy)"; do
    echo "[burstguard v$VER] $(date -Iseconds) waiting: another checkout is still running on this machine: $busy (check every 5 min)"
    sleep "${WAIT_IDLE_SEC:-300}"
  done
  echo "[burstguard v$VER] $(date -Iseconds) machine idle - starting"
fi
TS="$(date +%Y%m%d-%H%M%S)"
SH_NAME="${SHARD%%@*}"                     # log name without the weights (they are printed inside the log)
LOG="logs/run_grids_v${VER}_${SH_NAME/\//of}_${TS}.log"
echo "[burstguard v$VER] run_grids shard=$SHARD grids=${GRIDS[*]} ${QUICK[*]:-} host=$(hostname) log=$LOG" | tee -a "$LOG"
FAILED=()
for g in "${GRIDS[@]}"; do
  f="experiments/${g%.toml}.toml"
  if [[ ! -f "$f" ]]; then echo "[burstguard v$VER] skip: $f not found" | tee -a "$LOG"; FAILED+=("$g"); continue; fi
  echo "[burstguard v$VER] $(date -Iseconds) START $g shard $SHARD" | tee -a "$LOG"
  burstguard grid run "$f" --shard "$SHARD" "${QUICK[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[burstguard v$VER] $(date -Iseconds) END $g shard $SHARD exit=$rc" | tee -a "$LOG"
  [[ "$rc" -ne 0 ]] && FAILED+=("$g")
done
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "[burstguard v$VER] ALL GRIDS OK (shard $SHARD)" | tee -a "$LOG"
else
  echo "[burstguard v$VER] grids with failed jobs: ${FAILED[*]} -> bash scripts/collect_logs.sh --grid <name>" | tee -a "$LOG"
  exit 1
fi
