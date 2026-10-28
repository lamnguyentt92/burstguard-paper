#!/usr/bin/env bash
# =============================================================================
# BurstGuard - collect logs for analysis (works even when .venv is broken).
#
#   bash scripts/collect_logs.sh               # latest run (runs/LATEST) + latest setup
#   bash scripts/collect_logs.sh --last 3      # 3 most recent runs
#   bash scripts/collect_logs.sh --failed      # every run whose status is not success
#   bash scripts/collect_logs.sh --run runs/<dir>
#   bash scripts/collect_logs.sh --setup       # installation logs only
#   bash scripts/collect_logs.sh --grid E3_defenses   # grid status + failed jobs (+ analysis) of a grid
#   bash scripts/collect_logs.sh --grid P0_pilot,E0_calibration   # several grids
#   bash scripts/collect_logs.sh --campaign    # every grid of this version + plan + paper artefacts
#
# Output (version in the file name):
#   reports/bg_report_v<ver>_<host>_<time>.txt     -> paste or attach this file
#   reports/bg_report_v<ver>_<host>_<time>.tar.gz  -> full bundle (only when asked)
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VER="$(cat VERSION 2>/dev/null || echo unknown)"
MODE="latest"; N=1; RUNSEL=""; GRID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --last) MODE="last"; N="${2:-1}"; shift ;;
    --failed) MODE="failed" ;;
    --run) MODE="run"; RUNSEL="${2:-}"; shift ;;
    --setup) MODE="setup" ;;
    --grid) MODE="grid"; GRID="${2:-}"; shift ;;
    --campaign) MODE="campaign" ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unsupported argument: $1"; exit 2 ;;
  esac
  shift
done

mkdir -p reports
TS="$(date +%Y%m%d-%H%M%S)"; HOST="$(hostname -s 2>/dev/null || hostname)"
BASE="reports/bg_report_v${VER}_${HOST}_${TS}"
TXT="${BASE}.txt"; TMP="$(mktemp -d)"; PKG="$TMP/$(basename "$BASE")"; mkdir -p "$PKG"
SHARED="${BURSTGUARD_SHARED:-$HOME/burstguard-shared}"

# per-round records without the benign clients' details (keeps the bundle small; malicious clients keep theirs)
ROUNDS_PY='
import json, sys
with open(sys.argv[1]) as f, open(sys.argv[2], "w") as o:
    for line in f:
        if "\"kind\": \"round\"" not in line:
            continue
        r = json.loads(line)
        r["clients"] = [c for c in r.get("clients", []) if c.get("role") == "malicious"]
        o.write(json.dumps(r) + "\n")
'
sec() { echo; echo "################ $*"; }
clip() { local f="$1" n="${2:-80}"; [ -f "$f" ] || { echo "(missing $f)"; return; }
  local t; t=$(wc -l < "$f" 2>/dev/null || echo 0)
  if [ "$t" -gt "$n" ]; then echo "... ($t lines, showing last $n)"; tail -n "$n" "$f"; else cat "$f"; fi; }
strip() { sed -r 's/\x1B\[[0-9;]*[mK]//g'; }

RUNS=()
case "$MODE" in
  latest) [ -f runs/LATEST ] && RUNS+=("$(cat runs/LATEST)") ;;
  last)   while IFS= read -r d; do RUNS+=("${d%/}"); done < <(ls -1dt runs/*/ 2>/dev/null | head -n "$N") ;;
  failed) for d in runs/*/; do s="$d/summary.json"
            if [ ! -f "$s" ] || ! grep -q '"status": "success"' "$s"; then RUNS+=("${d%/}"); fi; done ;;
  run)    RUNS+=("$RUNSEL") ;;
  grid)   [ -f runs/LATEST ] && RUNS+=("$(cat runs/LATEST)") ;;
  campaign) while IFS= read -r d; do RUNS+=("${d%/}"); done < <(ls -1dt runs/*/ 2>/dev/null | head -n 10) ;;
esac
# grid directories to report: --grid a,b (prefix match) or --campaign (all grids of this version)
GDIRS=()
if [ "$MODE" = "campaign" ]; then
  for d in results/v"$VER"/*/; do b="$(basename "$d")"; [[ "$b" == _* ]] && continue; GDIRS+=("${d%/}"); done
elif [ -n "$GRID" ]; then
  IFS=',' read -ra _GL <<< "$GRID"
  for g in "${_GL[@]}"; do for d in results/v"$VER"/"$g"*/; do [ -d "$d" ] && GDIRS+=("${d%/}"); done; done
fi

{
echo "BURSTGUARD REPORT v$VER  $(date -Iseconds)"
sec "SYSTEM"
echo "version=$VER  git=$(git -c core.fileMode=false describe --tags --always --dirty 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "checkout=$ROOT  shared=$SHARED"
echo "host=$HOST  os=$(uname -srm)  $( [ -r /etc/os-release ] && . /etc/os-release && echo "${PRETTY_NAME:-}")"
echo "cpu=$(nproc)  mem_total=$(awk '/MemTotal/{printf "%.1fGiB",$2/1048576}' /proc/meminfo)  mem_avail=$(awk '/MemAvailable/{printf "%.1fGiB",$2/1048576}' /proc/meminfo)  disk_free=$(df -h . | awk 'NR==2{print $4}')  shm=$(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $2}')"
GS="$(git -c core.fileMode=false status --porcelain 2>/dev/null | head -n 15)"; [ -n "$GS" ] && { echo "git status (modified files):"; echo "$GS" | sed 's/^/  /'; }
if [ -x .venv/bin/python ]; then
  echo "venv: $(.venv/bin/python -V 2>&1), stamp=$(cat .venv/burstguard_version 2>/dev/null || echo none)"
  .venv/bin/python -m pip list 2>/dev/null | grep -Ei '^(flwr|ray|torch|granite-tsfm|transformers|peft|numpy|pandas|scipy|scikit-learn|matplotlib|burstguard) ' | sed 's/^/  /'
else
  echo "venv: MISSING (.venv/bin/python not found)"
fi
OTHER=$(ls -d "$HOME"/burstguard* 2>/dev/null | grep -v -- "-shared" | grep -v "^$ROOT$" | tr '\n' ' ')
[ -n "$OTHER" ] && echo "other checkouts on this machine: $OTHER"
echo "shared data: raw_files=$(ls "$SHARED/data/raw/milan" 2>/dev/null | grep -c sms-call) processed=$(ls "$SHARED/data/processed" 2>/dev/null | tr '\n' ' ')"

sec "LATEST SETUP"
SL="$(ls -1t logs/setup_*.log 2>/dev/null | head -n1)"
if [ -n "$SL" ]; then
  echo "file=$SL"; cat "${SL%.log}.status" 2>/dev/null || echo "status=(none - interrupted?)"
  if [ "$MODE" = "setup" ] || ! grep -q "status=success" "${SL%.log}.status" 2>/dev/null; then clip "$SL" 120 | strip
  else echo "(setup succeeded - details omitted)"; fi
else echo "(scripts/setup_venv.sh has not been run)"; fi

for R in "${RUNS[@]:-}"; do
  [ -n "$R" ] || continue; R="${R%/}"
  sec "RUN $(basename "$R")"
  [ -d "$R" ] || { echo "(missing: $R)"; continue; }
  echo "-- command:"; cat "$R/command.txt" 2>/dev/null
  echo "-- summary.json:"; { cat "$R/summary.json" && echo; } 2>/dev/null || echo "(no summary.json - process killed? e.g. out of memory)"
  if [ -f "$R/error.txt" ]; then EL=$(wc -l < "$R/error.txt")
    echo "-- error.txt ($EL lines; first 15 + last 70):"
    if [ "$EL" -gt 90 ]; then head -n 15 "$R/error.txt"; echo "   [...]"; tail -n 70 "$R/error.txt"; else cat "$R/error.txt"; fi
  fi
  if grep -q '"status": "success"' "$R/summary.json" 2>/dev/null; then
    echo "-- console.log (success - last 30 lines):"; clip "$R/console.log" 30 | strip
  else
    echo "-- suspicious lines in console.log (max 60):"
    grep -nEi 'traceback|error|exception|fail|killed|oom|out of memory|raylet|segmentation' "$R/console.log" 2>/dev/null | strip | grep -v 'RUN OK' | head -n 60
    echo "-- console.log (last 80 lines):"; clip "$R/console.log" 80 | strip
  fi
  for m in "$R/metrics.jsonl" "$R"/run*/metrics.jsonl "$R"/engine_*/metrics.jsonl; do
    [ -f "$m" ] || continue
    echo "-- $(basename "$(dirname "$m")")/metrics.jsonl (last 4 lines, cut to 400 chars):"; tail -n 4 "$m" | cut -c1-400
  done
  for j in "$R"/smoke_report.json "$R"/doctor.json "$R"/grid_status.json "$R"/equivalence.json; do
    [ -f "$j" ] && { echo "-- $(basename "$j"):"; head -c 5000 "$j"; echo; }
  done
done

for GD in "${GDIRS[@]:-}"; do
  [ -n "$GD" ] || continue
  sec "GRID $GD"
  T=$(ls -d "$GD"/*/ 2>/dev/null | grep -vc _analysis)
  OK=$(grep -l '"status": "success"' "$GD"/*/result.json 2>/dev/null | wc -l)
  FL=$(grep -l '"status": "failed"' "$GD"/*/result.json 2>/dev/null | wc -l)
  RUNNING=$(ls "$GD"/*/started 2>/dev/null | wc -l)
  echo "job dirs=$T success=$OK failed=$FL running/interrupted=$RUNNING"
  for f in $(grep -l '"status": "failed"' "$GD"/*/result.json 2>/dev/null | head -n 5); do
    echo "-- failed job $(dirname "$f"):"; python3 - "$f" <<'PY' 2>/dev/null || head -c 3000 "$f"
import json, sys
r = json.load(open(sys.argv[1])); print(r.get("label"), "|", r.get("error_type"), ":", r.get("error", "")[:600])
print("\n".join(r.get("traceback", "").splitlines()[-30:]))
PY
  done
  if [ -f "$GD/_analysis/summary.md" ]; then
    echo "-- _analysis/summary.md (from 'burstguard analyze'; first 120 lines):"; head -n 120 "$GD/_analysis/summary.md"
    [ -f "$GD/_analysis/tests.csv" ] && { echo "-- _analysis/tests.csv (first 60 lines):"; head -n 60 "$GD/_analysis/tests.csv"; }
  else
    echo "(no _analysis yet - run: burstguard analyze experiments/$(basename "$GD" | sed 's/_quick$//').toml$( [[ "$GD" == *_quick ]] && echo ' --quick'))"
  fi
done
if [ "$MODE" = "campaign" ]; then
  [ -f "$SHARED/bg_thresholds.toml" ] && { sec "CALIBRATED THRESHOLDS $SHARED/bg_thresholds.toml"; cat "$SHARED/bg_thresholds.toml"; }
  BGC="$(ls -1t logs/bg_check_v*.out 2>/dev/null | head -n1)"
  [ -n "$BGC" ] && { sec "BG-CHECK $BGC"; clip "$BGC" 60; }
fi
if [ "$MODE" = "campaign" ] && [ -d "results/v$VER/_paper" ]; then
  sec "PAPER ARTEFACTS results/v$VER/_paper"
  cat "results/v$VER/_paper/index.md" 2>/dev/null
  echo "-- numbers.json (first 200 lines):"; head -n 200 "results/v$VER/_paper/numbers.json" 2>/dev/null
fi
if [ ${#GDIRS[@]} -gt 0 ]; then
  RG="$(ls -1t logs/run_grids_v*.log 2>/dev/null | head -n1)"
  [ -n "$RG" ] && { sec "run_grids log $RG (START/END lines + last 20 lines)"; grep -E 'START|END|ALL GRIDS|failed jobs' "$RG" | tail -n 30; tail -n 20 "$RG" | strip; }
fi

RL="/tmp/ray/session_latest/logs"
if [ -d "$RL" ]; then
  sec "RAY (session_latest) worker errors"
  ls -1t "$RL"/*.err 2>/dev/null | head -n 6 | while read -r f; do
    if [ -s "$f" ] && grep -qEi 'error|traceback|exception' "$f"; then echo "-- $(basename "$f")"; tail -n 25 "$f" | strip; fi
  done
fi
echo; echo "################ END OF REPORT v$VER"
} > "$TXT" 2>&1

cp "$TXT" "$PKG/"
for R in "${RUNS[@]:-}"; do [ -n "$R" ] && [ -d "$R" ] || continue
  mkdir -p "$PKG/runs"; rsync -a --max-size=20m "$R" "$PKG/runs/" 2>/dev/null || cp -r "$R" "$PKG/runs/"; done
ls -1t logs/setup_* logs/pip_freeze_* 2>/dev/null | head -n 3 | xargs -r -I{} cp {} "$PKG/" 2>/dev/null
if [ ${#GDIRS[@]} -gt 0 ]; then
  for GD in "${GDIRS[@]}"; do
    for f in $(grep -l '"status": "failed"' "$GD"/*/result.json 2>/dev/null | head -n 10); do
      d="$(dirname "$f")"; mkdir -p "$PKG/grid_failed/$(basename "$d")"
      cp "$d"/result.json "$d"/job.log "$d"/config.toml "$PKG/grid_failed/$(basename "$d")/" 2>/dev/null
    done
    [ -d "$GD/_analysis" ] && { mkdir -p "$PKG/analysis/$(basename "$GD")"; cp -r "$GD/_analysis/." "$PKG/analysis/$(basename "$GD")/"; }
    # compact per-job results so the analysis can be re-run offline; v0.3.13: the per-round records (shadow screens,
    # calibration, regroup / verify decisions, budget boosts, norm bound) go to rounds.jsonl (v0.3.12 dropped them)
    mkdir -p "$PKG/results/$(basename "$GD")"
    for rj in "$GD"/*/result.json; do [ -f "$rj" ] || continue
      jd="$(dirname "$rj")"; jo="$PKG/results/$(basename "$GD")/$(basename "$jd")"; mkdir -p "$jo"
      cp "$rj" "$jo/result.json"
      [ -f "$jd/metrics.jsonl" ] && grep '"kind": "eval"' "$jd/metrics.jsonl" > "$jo/metrics.jsonl" 2>/dev/null
      [ -f "$jd/metrics.jsonl" ] && python3 -c "$ROUNDS_PY" "$jd/metrics.jsonl" "$jo/rounds.jsonl" 2>/dev/null
    done
  done
  [ -d "results/v$VER/_paper" ] && { mkdir -p "$PKG/paper"; cp -r "results/v$VER/_paper/." "$PKG/paper/"; }
fi
[ -d "$RL" ] && { mkdir -p "$PKG/ray_logs"; find "$RL" -maxdepth 1 -type f \( -name '*.err' -o -name 'raylet.*' \) -size -5M -exec cp {} "$PKG/ray_logs/" \; 2>/dev/null; }
tar -czf "${BASE}.tar.gz" -C "$TMP" "$(basename "$PKG")" 2>/dev/null
rm -rf "$TMP"

LINES=$(wc -l < "$TXT")
if [ "$LINES" -le 400 ]; then cat "$TXT"; else head -n 200 "$TXT"; echo "... (truncated, see $TXT)"; tail -n 150 "$TXT"; fi
echo
echo "==> text report : $TXT  ($LINES lines)  <- paste or attach this file"
echo "==> full bundle : ${BASE}.tar.gz ($(du -h "${BASE}.tar.gz" | cut -f1))"
