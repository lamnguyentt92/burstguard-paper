#!/usr/bin/env bash
# Pull the results of this version from the other VMs into this checkout (run on the VM that analyses).
# Job directories have deterministic ids, so copies from different VMs never collide; finished jobs win.
#
#   bash scripts/sync_results.sh                             # hosts from ~/burstguard-shared/cluster.hosts
#   bash scripts/sync_results.sh vm2 vm3 vm4 vm5            # ssh host names / user@ip of the other VMs
#   REMOTE_DIR=~/burstguard-v0.3.0 bash scripts/sync_results.sh root@10.0.0.12 ...
#
# Needs ssh access (key-based) from this VM to the others and rsync on both sides.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VER="$(cat VERSION)"
REMOTE_DIR="${REMOTE_DIR:-burstguard-v$VER}"   # relative to the remote home unless absolute
HOSTS_FILE="${BURSTGUARD_SHARED:-$HOME/burstguard-shared}/cluster.hosts"
if [ $# -eq 0 ] && [ -f "$HOSTS_FILE" ]; then
  mapfile -t _H < <(grep -vE '^\s*(#|$)' "$HOSTS_FILE" | awk '{print $1}'); set -- "${_H[@]}"
fi
[ $# -ge 1 ] || { echo "usage: bash scripts/sync_results.sh HOST [HOST ...]  (or list them in $HOSTS_FILE)"; exit 2; }
command -v rsync >/dev/null || { echo "ERROR: rsync not installed (sudo apt install rsync)"; exit 2; }
mkdir -p "results/v$VER" logs
LOG="logs/sync_results_v${VER}_$(date +%Y%m%d-%H%M%S).log"
rc_all=0
for h in "$@"; do
  echo "[burstguard v$VER] pulling $h:$REMOTE_DIR/results/v$VER/ ..." | tee -a "$LOG"
  # skip analysis/paper folders (re-created here) and jobs that are still running (no result.json yet are kept:
  # their 'started' marker shows them as running in `grid status`)
  if rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=10" --exclude '_analysis/' --exclude '_paper/' \
       "$h:$REMOTE_DIR/results/v$VER/" "results/v$VER/" 2>&1 | tee -a "$LOG"; then
    echo "  ok" | tee -a "$LOG"
  else
    echo "  FAILED for $h (ssh/rsync error above)" | tee -a "$LOG"; rc_all=1
  fi
done
n=$(ls results/v"$VER"/*/*/result.json 2>/dev/null | wc -l)
echo "[burstguard v$VER] results/v$VER now has $n finished job results. Next: burstguard analyze ... / burstguard paper" | tee -a "$LOG"
exit $rc_all
