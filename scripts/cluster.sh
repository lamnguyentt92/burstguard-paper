#!/usr/bin/env bash
# =============================================================================
# BurstGuard - drive the 5-VM campaign from ONE machine (the VM that has the data, e.g. vm-e) over ssh.
# No tmux needed: jobs run in the background with nohup (they keep running when you log out).
#
# 1) list the OTHER VMs, one ssh target per line, in  ~/burstguard-shared/cluster.hosts
#        root@10.0.0.12
#        root@10.0.0.13          (or ssh aliases from ~/.ssh/config, e.g. vm2)
# 2) bash scripts/cluster.sh keys        # ssh key + ssh-copy-id to every VM (type each VM's password once)
#    bash scripts/cluster.sh check       # ssh, git, python3, rsync, CPU/RAM/disk on every VM
#    bash scripts/cluster.sh push-data   # copy the processed datasets (milan_d1.npz, milan_sms_d1.npz, ...) + Hugging
#                                        # Face model cache to every VM
#    bash scripts/cluster.sh install     # clone tag v<VERSION> + setup_venv.sh on every VM (in parallel)
#    bash scripts/cluster.sh smoke       # burstguard smoke on every VM (in parallel)
#    bash scripts/cluster.sh start "--tier pilot,calibration" "E1_clean"
#                                        # every machine (this one = shard 1) runs in the background:
#                                        #   run_grids.sh k/N --tier pilot,calibration ; run_grids.sh k/N E1_clean
#    bash scripts/cluster.sh status      # progress on every machine
#    bash scripts/cluster.sh --bg collect P0_pilot E0_calibration
#                                        # in the background: wait until these grids are finished on all machines
#                                        # (pull results every 10 min), then analyze all, calibrate, plan and
#                                        # collect_logs --campaign. Follow it: tail -f logs/cluster_collect_*.log
#    bash scripts/cluster.sh --bg finish # v0.3.9: once X4 is done everywhere calibrate SMS/call + push the thresholds
#                                        # (X5 then starts by itself), wait for every grid, analyse, bg-check, paper,
#                                        # campaign report. Follow it: tail -f logs/cluster_finish_*.log
#    bash scripts/cluster.sh stop        # stop the background run everywhere (start again = resume)
#    bash scripts/cluster.sh eta         # v0.3.16: live remaining time - every running job's round k/N and its own
#                                        # mean seconds per round, jobs still queued, and the plan estimate
#    bash scripts/cluster.sh import 0.3.2  # copy jobs finished under v0.3.2 into this version (they are skipped)
#    bash scripts/cluster.sh start "--wait-idle --tier generalization"
#                                        # each machine waits until the background run of an older checkout on that
#                                        # machine has finished, then starts (no two versions share the CPUs)
# Weighted shards (default): machines get work in proportion to their capacity (parallel workers), computed once
# and stored in ~/burstguard-shared/cluster.weights ("w1,...,wN": this machine, then the hosts-file order).
# Never edit it during a campaign (it fixes which machine runs which job); delete it to recompute.
# Options (before the command):
#   --hosts FILE   other hosts file          --no-local   this machine does not run a shard
#   --local-only   only this machine (e.g. E8: --local-only --name e8 start "--tier system")
#   --name NAME    name of the background run (default bg)      --force   start even if one is running
#   --bg           run the command itself in the background (for collect)
#   --equal        equal round-robin shards k/N instead of weighted shards
# Logs: logs/cluster_<command>_<host>_<time>.log, background output logs/cluster_<name>.out (on each machine)
# =============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VER="$(cat VERSION)"
SHARED="${BURSTGUARD_SHARED:-$HOME/burstguard-shared}"
HOSTS_FILE="$SHARED/cluster.hosts"
REPO="https://github.com/lamnguyentt92/burstguard-paper.git"
RDIR="burstguard-v$VER"                     # remote checkout, relative to the remote home
LOCAL=1; REMOTE=1; FORCE=0; NAME="bg"; BG=0; EQUAL=0
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
ORIG_ARGS=("$@")

while [ $# -gt 0 ]; do
  case "$1" in
    --hosts) HOSTS_FILE="$2"; shift 2 ;;
    --no-local) LOCAL=0; shift ;;
    --local-only) REMOTE=0; shift ;;
    --name) NAME="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --bg) BG=1; shift ;;
    --equal) EQUAL=1; shift ;;
    *) break ;;
  esac
done
CMD="${1:-}"; shift || true
mkdir -p logs
TS="$(date +%Y%m%d-%H%M%S)"
say() { echo "[burstguard v$VER cluster $(date +%H:%M:%S)] $*"; }

if [ -z "$CMD" ] || [ "$CMD" = "-h" ] || [ "$CMD" = "--help" ]; then sed -n '2,39p' "$0"; exit 0; fi

# --bg: re-run this very command detached from the terminal (survives logout), output in a log file
if [ $BG = 1 ]; then
  OUT="logs/cluster_${CMD}_${TS}.log"
  ARGS=(); for a in "${ORIG_ARGS[@]}"; do [ "$a" = "--bg" ] || ARGS+=("$a"); done
  nohup setsid bash "$0" "${ARGS[@]}" > "$OUT" 2>&1 < /dev/null &
  say "running '$CMD' in the background (pid $!). Follow it: tail -f $ROOT/$OUT  (Ctrl-C only stops tail)"
  exit 0
fi

if [ ! -f "$HOSTS_FILE" ]; then
  mkdir -p "$(dirname "$HOSTS_FILE")"
  cat > "$HOSTS_FILE" <<'EOF'
# One ssh target per line for the OTHER VMs (not this one). Examples:
# root@10.0.0.12
# root@10.0.0.13
# vm4            (an alias defined in ~/.ssh/config)
EOF
  say "created $HOSTS_FILE - put the other VMs in it (one per line), then run this command again"; exit 2
fi
mapfile -t HOSTS < <(grep -vE '^\s*(#|$)' "$HOSTS_FILE" | awk '{print $1}')
[ ${#HOSTS[@]} -ge 1 ] || { say "no hosts in $HOSTS_FILE"; exit 2; }
N=$(( ${#HOSTS[@]} + 1 ))                   # shard count: this machine + the hosts (fixed by the hosts file)
[ $REMOTE = 1 ] || HOSTS=()
hlog() { echo "logs/cluster_${CMD}_$(echo "$1" | tr '@/:' '___')_${TS}.log"; }
rsh() { local h="$1"; shift; ssh "${SSH_OPTS[@]}" "$h" "$@"; }

# capacity = parallel workers the engine would use: max(1, min(CPUs - 1, floor((MemAvailable GiB - 2.5) / 1.4)))
# (same formula as burstguard.engines.common.auto_workers)
# (free memory is only meaningful while no experiment runs there -> report BUSY in that case)
CAP_CMD='if pgrep -f "burstguard [g]rid run" >/dev/null 2>&1; then echo BUSY; else awk -v n="$(nproc)" "/MemAvailable/{m=\$2/1048576; w=int((m-2.5)/1.4); if (w>n-1) w=n-1; if (w<1) w=1; print w}" /proc/meminfo; fi'
WEIGHTS_FILE="$SHARED/cluster.weights"
ALL_HOSTS=("${HOSTS[@]}")
read_weights() { [ -f "$WEIGHTS_FILE" ] && grep -vE '^\s*(#|$)' "$WEIGHTS_FILE" | head -n1 | tr -d ' '; }
ensure_weights() {  # prints the weights line "w1,...,wN" (shard 1 = this machine, then hosts-file order)
  local w; w="$(read_weights)"
  if [ -n "$w" ]; then
    if [ "$(echo "$w" | tr ',' '\n' | wc -l)" != "$N" ]; then
      say "ERROR: $WEIGHTS_FILE has $(echo "$w" | tr ',' '\n' | wc -l) weights but there are $N machines" >&2; return 1; fi
    echo "$w"; return 0
  fi
  local caps=() c h busy=()
  c=$(bash -c "$CAP_CMD"); [ "$c" = BUSY ] && busy+=("this machine"); caps+=("${c:-1}")
  for h in "${ALL_HOSTS[@]}"; do
    c=$(rsh "$h" "$CAP_CMD" 2>/dev/null) || { say "ERROR: cannot read the capacity of $h" >&2; return 1; }
    [ "$c" = BUSY ] && busy+=("$h"); caps+=("${c:-1}")
  done
  if [ ${#busy[@]} -gt 0 ]; then
    say "ERROR: experiments are still running on: ${busy[*]} - free memory would be underestimated." >&2
    say "Stop them first (cluster.sh stop, also from an older checkout), or write $WEIGHTS_FILE by hand." >&2
    return 1
  fi
  w=$(IFS=,; echo "${caps[*]}")
  { echo "# shard weights = parallel workers per machine (this machine first, then $HOSTS_FILE order)";
    echo "# computed $(date -Iseconds); reused by every start - do not edit during a campaign, delete to recompute";
    echo "$w"; } > "$WEIGHTS_FILE"
  say "computed shard weights $w (this machine, then hosts in file order) -> $WEIGHTS_FILE" >&2
  say "these weights are reused for every later start; delete the file only before a NEW campaign" >&2
  echo "$w"
}

# background-run helpers (executed locally, or sent to a remote shell with declare -f)
bg_alive() { [ -f "logs/cluster_$1.pid" ] && kill -0 "$(cat "logs/cluster_$1.pid")" 2>/dev/null; }
bg_start() {  # $1 = name, $2 = command chain
  mkdir -p logs
  nohup setsid bash -c "$2" > "logs/cluster_$1.out" 2>&1 < /dev/null &
  echo $! > "logs/cluster_$1.pid"
}
bg_stop() {
  if bg_alive "$1"; then local p; p=$(cat "logs/cluster_$1.pid")
    kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; sleep 2; kill -KILL -- "-$p" 2>/dev/null; echo stopped
  else echo "not running"; fi
}
FUNCS="$(declare -f bg_alive bg_start bg_stop)"

case "$CMD" in
# -----------------------------------------------------------------------------------------------------------
keys)
  KEY="$HOME/.ssh/id_ed25519"
  if [ ! -f "$KEY" ]; then say "creating ssh key $KEY"; ssh-keygen -t ed25519 -N "" -f "$KEY" -q; fi
  for h in "${HOSTS[@]}"; do
    if rsh "$h" true 2>/dev/null; then say "$h: key login already works"; continue; fi
    say "$h: copying the key (type the password of $h once)"
    if ! ssh-copy-id -o StrictHostKeyChecking=accept-new -i "$KEY.pub" "$h"; then
      say "$h: ssh-copy-id failed (password login disabled?). Do it by hand ON $h:"
      echo "    mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$(cat "$KEY.pub")' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    fi
  done
  for h in "${HOSTS[@]}"; do
    if rsh "$h" true 2>/dev/null; then say "$h: OK"; else say "$h: STILL NO KEY LOGIN"; fi
  done ;;
# -----------------------------------------------------------------------------------------------------------
check)
  rc=0
  for h in "${HOSTS[@]}"; do
    out=$(rsh "$h" 'printf "host=%s cpu=%s mem_avail=%sGiB disk_free=%s " "$(hostname)" "$(nproc)" \
      "$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo)" "$(df -h ~ | awk "NR==2{print \$4}")";
      for t in git python3 rsync curl setsid; do command -v $t >/dev/null && printf "%s=ok " $t || printf "%s=MISSING " $t; done;
      test -f ~/burstguard-shared/data/processed/milan_d1.npz && printf "data=ok " || printf "data=missing ";
      test -d ~/'"$RDIR"'/.venv && printf "venv=ok" || printf "venv=missing"' 2>&1) || { rc=1; out="UNREACHABLE: $out"; }
    echo "  $h: $out"
  done
  echo "  (missing tools on Ubuntu: sudo apt-get install -y git rsync curl)"
  exit $rc ;;
# -----------------------------------------------------------------------------------------------------------
push-data)
  shopt -s nullglob
  NPZS=("$SHARED"/data/processed/milan*_d*.npz "$SHARED"/data/processed/trentino*_d*.npz); HF="$SHARED/hf"; THR="$SHARED/bg_thresholds.toml"
  shopt -u nullglob
  [ ${#NPZS[@]} -gt 0 ] || { say "missing $SHARED/data/processed/milan_d1.npz - run burstguard data download && burstguard data prepare here first"; exit 2; }
  [ -d "$HF/hub" ] || say "WARNING: no model cache in $HF (run burstguard model-download here first)"
  rc=0
  for h in "${HOSTS[@]}"; do
    say "$h: copying $(for f in "${NPZS[@]}"; do printf '%s ' "$(basename "$f")"; done)and the model cache"
    rsh "$h" 'mkdir -p ~/burstguard-shared/data/processed ~/burstguard-shared/hf' || { say "$h: unreachable"; rc=1; continue; }
    if command -v rsync >/dev/null && rsh "$h" 'command -v rsync >/dev/null'; then
      # --checksum: unchanged files are not rewritten; changed ones are replaced atomically (temp file + rename),
      # so jobs that are running on that machine never read a half-copied dataset
      rsync -a --checksum -e "ssh ${SSH_OPTS[*]}" "${NPZS[@]}" "$h:burstguard-shared/data/processed/" || rc=1
      rsync -a -e "ssh ${SSH_OPTS[*]}" "$HF/" "$h:burstguard-shared/hf/" || rc=1
    else
      for f in "${NPZS[@]}"; do   # no rsync: copy only files the machine does not have yet
        rsh "$h" "test -f ~/burstguard-shared/data/processed/$(basename "$f")" && continue
        scp -q "${SSH_OPTS[@]}" "$f" "$h:burstguard-shared/data/processed/" || rc=1
      done
      scp -q -r "${SSH_OPTS[@]}" "$HF/." "$h:burstguard-shared/hf/" || rc=1
    fi
    if [ -f "$THR" ]; then scp -q "${SSH_OPTS[@]}" "$THR" "$h:burstguard-shared/bg_thresholds.toml" || rc=1; fi
    rsh "$h" 'ls -l ~/burstguard-shared/data/processed/milan*.npz ~/burstguard-shared/data/processed/trentino*.npz ~/burstguard-shared/bg_thresholds.toml 2>/dev/null | awk "{print \"    \" \$5 \" bytes \" \$9}"'
  done
  exit $rc ;;
# -----------------------------------------------------------------------------------------------------------
install)
  for h in "${HOSTS[@]}"; do
    L="$(hlog "$h")"
    say "$h: clone v$VER into ~/$RDIR and run setup_venv.sh (log: $L)"
    rsh "$h" "set -e; if [ -d ~/$RDIR/.git ]; then echo 'checkout exists'; else git clone -q -b v$VER $REPO ~/$RDIR; fi;
      cd ~/$RDIR; git config core.fileMode false; cat VERSION; bash scripts/setup_venv.sh" > "$L" 2>&1 &
  done
  wait
  rc=0
  for h in "${HOSTS[@]}"; do
    L="$(hlog "$h")"
    if grep -q "SETUP COMPLETE" "$L"; then say "$h: setup OK"; else say "$h: setup FAILED - see $L (last lines below)"; tail -n 8 "$L"; rc=1; fi
  done
  exit $rc ;;
# -----------------------------------------------------------------------------------------------------------
smoke)
  for h in "${HOSTS[@]}"; do
    rsh "$h" "cd ~/$RDIR && .venv/bin/burstguard smoke" > "$(hlog "$h")" 2>&1 &
  done
  wait
  rc=0
  for h in "${HOSTS[@]}"; do
    L="$(hlog "$h")"
    if grep -q "RUN OK" "$L"; then say "$h: smoke OK ($(grep -c '\[PASS\]' "$L") PASS)"; else say "$h: smoke FAILED - $L"; tail -n 15 "$L"; rc=1; fi
  done
  exit $rc ;;
# -----------------------------------------------------------------------------------------------------------
start)
  [ $# -ge 1 ] || { say 'usage: cluster.sh start "--tier pilot,calibration" ["E1_clean" ...]'; exit 2; }
  chain_for() {  # $1 = checkout dir, $2 = shard spec, rest = run_grids argument groups
    local dir="$1" spec="$2"; shift 2
    local c="cd $dir"
    for a in "$@"; do c="$c; bash scripts/run_grids.sh $spec $a"; done
    echo "$c"
  }
  rc=0
  WSUF=""
  if [ $EQUAL = 0 ] && [ $REMOTE = 1 ]; then
    W="$(ensure_weights)" || exit 2
    WSUF="@$W"
    say "weighted shards: $W (use --equal for plain k/$N)"
  fi
  if [ $LOCAL = 1 ]; then
    spec="1/$N$WSUF"; [ $REMOTE = 0 ] && spec="1/1"
    if bg_alive "$NAME" && [ $FORCE = 0 ]; then say "local: '$NAME' is already running (stop it first)"; rc=1
    else bg_start "$NAME" "$(chain_for "$ROOT" "$spec" "$@")"
         say "local (shard $spec): started - output: $ROOT/logs/cluster_$NAME.out"; fi
  fi
  k=2
  for h in "${HOSTS[@]}"; do
    chain="$(chain_for "~/$RDIR" "$k/$N$WSUF" "$@")"
    res=$(rsh "$h" "cd ~/$RDIR || exit 4; $FUNCS; if bg_alive $NAME && [ $FORCE = 0 ]; then echo RUNNING; else bg_start $NAME '$chain'; echo STARTED; fi" 2>&1)
    case "$res" in
      *STARTED*) say "$h (shard $k/$N): started - output: ~/$RDIR/logs/cluster_$NAME.out on $h" ;;
      *RUNNING*) say "$h: '$NAME' is already running (stop it first)"; rc=1 ;;
      *) say "$h: FAILED to start: $res"; rc=1 ;;
    esac
    k=$((k + 1))
  done
  say "each machine runs its own shard. Check: bash scripts/cluster.sh status"
  exit $rc ;;
# -----------------------------------------------------------------------------------------------------------
status)
  show() {
    if bg_alive "$1"; then echo "  run '$1': RUNNING (pid $(cat "logs/cluster_$1.pid"))"; else echo "  run '$1': not running"; fi
    local f; f=$(ls -1t logs/run_grids_v*.log 2>/dev/null | head -n1)
    [ -n "$f" ] && { echo "  $f:"; grep -E 'START|END|ALL GRIDS|failed jobs' "$f" | tail -n 4 | sed 's/^/    /'; }
    for g in results/v*/*/; do [ -d "$g" ] || continue; case "$(basename "$g")" in _*) continue ;; esac
      echo "  $(basename "$g"): ok=$(grep -l '"status": "success"' "$g"*/result.json 2>/dev/null | wc -l) failed=$(grep -l '"status": "failed"' "$g"*/result.json 2>/dev/null | wc -l) running=$(ls "$g"*/started 2>/dev/null | wc -l)"
    done
  }
  if [ -n "$(read_weights)" ]; then echo "shard weights (this machine, then hosts): $(read_weights)  [$WEIGHTS_FILE]"
  else echo "shard weights: none yet (equal shards; computed at the next start)"; fi
  if [ $LOCAL = 1 ]; then echo "== local ($(hostname))"; show "$NAME"; fi
  for h in "${HOSTS[@]}"; do
    echo "== $h"
    rsh "$h" "cd ~/$RDIR 2>/dev/null || { echo '  no checkout'; exit 0; }; $FUNCS; $(declare -f show); show $NAME" 2>&1 || echo "  unreachable"
  done ;;
# -----------------------------------------------------------------------------------------------------------
eta)
  # per machine: running jobs (round k/N, mean s/round so far, time left for that job) + jobs not started yet
  ETA_PY='
import glob, json, os, re, sys, time
root = sys.argv[1]
pat = re.compile(r"\] round (\d+)/(\d+): .*? ([0-9.]+)s")
left_total = 0.0; queued = 0
for jd in sorted(glob.glob(root + "/*/*/")):
    if os.path.basename(os.path.dirname(jd.rstrip("/"))).startswith("_"):
        continue
    if os.path.exists(jd + "result.json"):
        continue
    if not os.path.exists(jd + "started"):
        continue
    k = n = 0; secs = []
    try:
        for line in open(jd + "job.log", errors="replace"):
            m = pat.search(line)
            if m:
                k, n = int(m.group(1)), int(m.group(2)); secs.append(float(m.group(3)))
    except OSError:
        pass
    grid = os.path.basename(os.path.dirname(jd.rstrip("/")))
    if not secs:
        print(f"    {grid}/{os.path.basename(jd.rstrip(chr(47)))}: starting (no finished round yet)"); continue
    mean = sum(secs[-5:]) / len(secs[-5:])
    left = (n - k) * mean; left_total += left
    print(f"    {grid}/{os.path.basename(jd.rstrip(chr(47)))}: round {k}/{n}, {mean:.0f} s/round -> {left/3600:.1f} h left")
print(f"    running jobs: {left_total/3600:.1f} h left")
'
  show_eta() {
    python3 -c "$ETA_PY" "results/v$VER" 2>/dev/null || echo "    (no results yet)"
  }
  if [ $LOCAL = 1 ]; then echo "== local ($(hostname))"; show_eta; fi
  for h in "${HOSTS[@]}"; do
    echo "== $h"
    rsh "$h" "cd ~/$RDIR 2>/dev/null || { echo '    no checkout'; exit 0; }; python3 -c '$(printf "%s" "$ETA_PY" | sed "s/'/'\\''/g")' results/v$VER" 2>/dev/null || echo "    unreachable"
  done
  if [ -x "$ROOT/.venv/bin/burstguard" ]; then
    echo; echo "== plan (jobs not finished, measured cost model; started jobs counted in full)"
    "$ROOT/.venv/bin/burstguard" plan "$@" 2>/dev/null | grep -E "^(grid|R[0-9]|TOTAL|per-shard|=>)" | sed 's/^/  /'
  fi ;;
# -----------------------------------------------------------------------------------------------------------
stop)
  [ $LOCAL = 1 ] && say "local: $(bg_stop "$NAME")"
  for h in "${HOSTS[@]}"; do say "$h: $(rsh "$h" "cd ~/$RDIR 2>/dev/null || exit 0; $FUNCS; bg_stop $NAME" 2>&1)"; done
  say "unfinished jobs resume with the same start command" ;;
# -----------------------------------------------------------------------------------------------------------
collect)
  GRIDS=("$@"); [ ${#GRIDS[@]} -ge 1 ] || GRIDS=(P0_pilot E0_calibration)
  BG_BIN="$ROOT/.venv/bin/burstguard"
  [ -x "$BG_BIN" ] || { say "missing $BG_BIN - run bash scripts/setup_venv.sh"; exit 2; }
  pending() { local g; for g in "${GRIDS[@]}"; do
      "$BG_BIN" grid status "experiments/$g.toml" 2>/dev/null | grep -qE '"(pending|running)"' && return 0; done; return 1; }
  say "waiting for ${GRIDS[*]} on all machines (results pulled every ${WAIT_MIN:-10} min)"
  while true; do
    bash scripts/sync_results.sh > "logs/cluster_collect_sync.log" 2>&1 || say "warning: some machines could not be synced (logs/cluster_collect_sync.log)"
    pending || break
    say "not finished yet: $(for g in "${GRIDS[@]}"; do printf '%s ' "$g:$(ls results/v$VER/$g/*/result.json 2>/dev/null | wc -l)/$("$BG_BIN" grid list "experiments/$g.toml" 2>/dev/null | grep -oE '^[0-9]+ jobs' | cut -d' ' -f1)"; done)"
    sleep $(( ${WAIT_MIN:-10} * 60 ))
  done
  say "all finished - analysing"
  "$BG_BIN" analyze all
  [ -d "results/v$VER/E0_calibration" ] && "$BG_BIN" calibrate experiments/E0_calibration.toml
  "$BG_BIN" plan
  if [ "${WITH_PAPER:-0}" = 1 ]; then
    say "bg-check (relative-change floor on saved internet models) -> logs/bg_check_v$VER.out"
    "$BG_BIN" bg-check --results "results/v$VER" > "logs/bg_check_v$VER.out" 2>&1 || say "warning: bg-check failed (logs/bg_check_v$VER.out)"
    say "paper tables and figures -> results/v$VER/_paper"
    "$BG_BIN" paper > "logs/paper_v$VER.out" 2>&1 || say "warning: paper export failed (logs/paper_v$VER.out)"
  fi
  bash scripts/collect_logs.sh --campaign
  say "DONE - send the two newest files in $ROOT/reports/ (.txt and .tar.gz)" ;;
# -----------------------------------------------------------------------------------------------------------
finish)
  # End of the campaign in ONE background command (cluster.sh --bg finish):
  #  1) once X4_traffic_bg_monitor is complete on all machines: calibrate SMS and call (fixed anchor rule), write
  #     $SHARED/bg_thresholds.toml and copy it to every machine - X5 (waiting there) then starts by itself;
  #  2) wait for every grid, analyse all, bg-check, paper tables / figures, campaign report.
  GRIDS=("$@")
  [ ${#GRIDS[@]} -ge 1 ] || GRIDS=(E0_calibration E1_clean E2_attacks E3_defenses E7_secagg E9_ablation E4_sensitivity
    E5_lifespan E6_adaptive E8_system X1_traffic_clean X2_traffic_defenses X3_trigger_offset X4_traffic_bg_monitor
    X5_traffic_bg_enforce X6_offset_calibrated X7_poison_defenses X8_bg_calibrated)
  BG_BIN="$ROOT/.venv/bin/burstguard"
  [ -x "$BG_BIN" ] || { say "missing $BG_BIN - run bash scripts/setup_venv.sh"; exit 2; }
  X5_DONE=0   # X5 already complete (e.g. imported from v0.3.9): its thresholds exist, nothing to calibrate
  [ -d "results/v$VER/X5_traffic_bg_enforce" ] && ! "$BG_BIN" grid status experiments/X5_traffic_bg_enforce.toml 2>/dev/null \
    | grep -qE '"(pending|running)"' && X5_DONE=1
  if [ "$X5_DONE" = 1 ]; then say "step 1/2 skipped: X5_traffic_bg_enforce is already complete"; fi
  if [ "$X5_DONE" = 0 ] && printf '%s\n' "${GRIDS[@]}" | grep -qx X5_traffic_bg_enforce; then
    X4=X4_traffic_bg_monitor
    N4="$("$BG_BIN" grid list "experiments/$X4.toml" 2>/dev/null | grep -oE '^[0-9]+ jobs' | cut -d' ' -f1)"
    say "step 1/2: waiting for $X4 on all machines (results pulled every ${WAIT_MIN:-5} min)"
    while true; do
      bash scripts/sync_results.sh > "logs/finish_sync.log" 2>&1 || say "warning: some machines could not be synced (logs/finish_sync.log)"
      "$BG_BIN" grid status "experiments/$X4.toml" 2>/dev/null | grep -qE '"(pending|running)"' || break
      say "$X4 not finished yet: $(grep -l '"status": "success"' results/v"$VER"/$X4/*/result.json 2>/dev/null | wc -l)/${N4:-?} jobs done"
      sleep $(( ${WAIT_MIN:-5} * 60 ))
    done
    NF="$(grep -l '"status": "failed"' results/v"$VER"/$X4/*/result.json 2>/dev/null | wc -l)"
    [ "$NF" -gt 0 ] && say "WARNING: $NF job(s) of $X4 failed - calibrating on the finished ones"
    for ds in sms call; do
      say "calibrating $ds: burstguard calibrate experiments/$X4.toml --where data.path=shared:data/processed/milan_${ds}_d1.npz --write-thresholds shared:bg_thresholds.toml"
      "$BG_BIN" calibrate "experiments/$X4.toml" --where "data.path=shared:data/processed/milan_${ds}_d1.npz" \
        --write-thresholds shared:bg_thresholds.toml > "logs/finish_calibrate_${ds}.log" 2>&1 \
        || { say "ERROR: calibration of $ds failed - see logs/finish_calibrate_${ds}.log"; exit 1; }
      grep -E '^thresholds for' "logs/finish_calibrate_${ds}.log" | sed 's/^/    /'
    done
    say "copying $SHARED/bg_thresholds.toml to every machine (push-data)"
    bash "$0" --hosts "$HOSTS_FILE" push-data > "logs/finish_push.log" 2>&1 \
      || say "WARNING: push-data had errors (logs/finish_push.log) - run: bash scripts/cluster.sh push-data"
    say "thresholds in place - X5 starts on each machine at its next check (<= 5 min)"
  fi
  say "step 2/2: waiting for all grids, then analyse + bg-check + paper + campaign report"
  WITH_PAPER=1 WAIT_MIN="${WAIT_MIN:-10}" exec bash "$0" --hosts "$HOSTS_FILE" collect "${GRIDS[@]}" ;;
# -----------------------------------------------------------------------------------------------------------
import)
  OLD="${1:-}"; [ -n "$OLD" ] || { say "usage: cluster.sh import OLD_VERSION   (e.g. 0.3.2)"; exit 2; }
  OLD="${OLD#v}"
  [ "$OLD" != "$VER" ] || { say "OLD_VERSION must differ from this version ($VER)"; exit 2; }
  # copies finished jobs (result.json with status success) that are not yet present; prints "imported N kept M"
  IMPORT_SH='src="$HOME/burstguard-v'"$OLD"'/results/v'"$OLD"'"; dst="$1/results/v'"$VER"'"; n=0; m=0
    if [ ! -d "$src" ]; then echo "no $src"; exit 0; fi
    for jd in "$src"/*/*/; do
      jd="${jd%/}"; g="$(basename "$(dirname "$jd")")"; j="$(basename "$jd")"
      case "$g" in _*|*_quick|*_quick_*) continue ;; esac
      grep -q "\"status\": \"success\"" "$jd/result.json" 2>/dev/null || continue
      if [ -f "$dst/$g/$j/result.json" ]; then m=$((m+1)); continue; fi
      mkdir -p "$dst/$g"; rm -rf "$dst/$g/$j"; cp -a "$jd" "$dst/$g/$j"; n=$((n+1))
    done
    echo "imported $n finished jobs from v'"$OLD"' (already present: $m)"'
  rc=0
  if [ $LOCAL = 1 ]; then say "local: $(bash -s -- "$ROOT" <<< "$IMPORT_SH")"; fi
  for h in "${HOSTS[@]}"; do
    # the script travels on stdin (no quoting issues); $1 = this version's checkout on that machine
    out=$(rsh "$h" "bash -s -- ~/$RDIR" <<< "$IMPORT_SH" 2>&1) || { rc=1; out="FAILED: $out"; }
    say "$h: $out"
  done
  exit $rc ;;
# -----------------------------------------------------------------------------------------------------------
*)
  say "unknown command: $CMD (bash scripts/cluster.sh --help)"; exit 2 ;;
esac
