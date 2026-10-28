#!/usr/bin/env bash
# =============================================================================
# BurstGuard - set up a NEW 5-VM cluster: ssh from the controller (the VM that runs everything) to the 4 others.
# IP addresses are not typed in: they are found on the network, and only machines that accepted the controller's
# key are kept (so other machines in the same subnet are never added).
#
#   1) on the controller:      bash scripts/cluster_bootstrap.sh controller
#        installs git/rsync/curl, creates the ssh key, marks this VM as the controller and PRINTS ONE COMMAND
#   2) on each of the 4 other VMs: paste that command (adds the controller's key, installs git/rsync/curl,
#        makes sure sshd accepts key logins, prints "WORKER READY: <hostname> <ip> ...")
#   3) on the controller:      bash scripts/cluster_bootstrap.sh discover
#        scans the controller's subnets (/24) for ssh, keeps the hosts that accept the key, writes
#        ~/burstguard-shared/cluster.hosts and runs cluster.sh check. If the VMs are in different subnets, give the
#        IPs printed by step 2:  bash scripts/cluster_bootstrap.sh discover 10.0.0.11 10.0.0.12 ...
#   bash scripts/cluster_bootstrap.sh worker-cmd    # print the command of step 2 again
# Options: --user NAME (ssh user on the other VMs, default root), --expect N (number of other VMs, default 4)
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARED="${BURSTGUARD_SHARED:-$HOME/burstguard-shared}"
HOSTS_FILE="$SHARED/cluster.hosts"
MARK="$SHARED/CONTROLLER"
KEY="$HOME/.ssh/id_ed25519"
USER_R="root"; EXPECT=4
CMD="${1:-}"; [ $# -gt 0 ] && shift
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --user) USER_R="$2"; shift ;;
    --expect) EXPECT="$2"; shift ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
say() { echo "[bootstrap $(hostname -s)] $*"; }
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)

install_tools() {
  local need=()
  for t in git rsync curl ssh; do command -v "$t" >/dev/null || need+=("$t"); done
  [ ${#need[@]} -eq 0 ] && return 0
  if command -v apt-get >/dev/null; then
    local SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
    say "installing ${need[*]}"
    $SUDO apt-get -qq update && DEBIAN_FRONTEND=noninteractive $SUDO apt-get -qq install -y git rsync curl openssh-client >/dev/null
  else
    say "please install: ${need[*]}"; return 1
  fi
}

worker_cmd() {
  local pub; pub="$(cat "$KEY.pub")"
  # one line, safe to paste: idempotent (the key is added once), root login stays key-only (prohibit-password)
  cat <<EOF
mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && (grep -qF '${pub}' ~/.ssh/authorized_keys || echo '${pub}' >> ~/.ssh/authorized_keys) && chmod 600 ~/.ssh/authorized_keys; S=""; [ "\$(id -u)" = 0 ] || S=sudo; (command -v git && command -v rsync && command -v curl) >/dev/null || { \$S apt-get -qq update && DEBIAN_FRONTEND=noninteractive \$S apt-get -qq install -y git rsync curl openssh-server >/dev/null; }; if \$S grep -qsE '^\s*PermitRootLogin\s+no' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf; then \$S sed -i -E 's/^\s*PermitRootLogin\s+no/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf 2>/dev/null; echo "sshd: root login set to key-only"; fi; \$S systemctl enable --now ssh >/dev/null 2>&1 || \$S systemctl enable --now sshd >/dev/null 2>&1; \$S systemctl reload ssh >/dev/null 2>&1 || \$S systemctl reload sshd >/dev/null 2>&1; echo "WORKER READY: \$(hostname) \$(hostname -I | awk '{print \$1}') cpu=\$(nproc) mem=\$(free -g | awk '/^Mem/{print \$2}')G disk_free=\$(df -h ~ | awk 'NR==2{print \$4}')"
EOF
}

own_ips() { hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | grep -vE '^(127\.|169\.254\.|172\.17\.)'; }

case "$CMD" in
controller)
  install_tools || exit 1
  mkdir -p "$SHARED" "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
  [ -f "$KEY" ] || { ssh-keygen -t ed25519 -N "" -C "burstguard-controller@$(hostname -s)" -f "$KEY" -q; say "created $KEY"; }
  echo "controller $(hostname -s) $(date -Iseconds)" > "$MARK"
  say "this VM is the controller (marker $MARK); its addresses: $(own_ips | tr '\n' ' ')"
  say "cpu=$(nproc) mem=$(free -g | awk '/^Mem/{print $2}')G disk_free=$(df -h ~ | awk 'NR==2{print $4}')"
  echo
  echo "======== PASTE THIS ONE LINE ON EACH OF THE OTHER $EXPECT VMs (as $USER_R) ========"
  worker_cmd
  echo "======== then here: bash scripts/cluster_bootstrap.sh discover ========" ;;
worker-cmd)
  [ -f "$KEY.pub" ] || { say "no $KEY.pub - run: bash scripts/cluster_bootstrap.sh controller"; exit 2; }
  worker_cmd ;;
discover)
  [ -f "$MARK" ] || { say "run 'bash scripts/cluster_bootstrap.sh controller' on this VM first"; exit 2; }
  mapfile -t MINE < <(own_ips)
  if [ ${#ARGS[@]} -gt 0 ]; then
    CANDS=("${ARGS[@]}"); say "checking the given addresses: ${CANDS[*]}"
  else
    PREFIXES=$(printf '%s\n' "${MINE[@]}" | awk -F. '{print $1"."$2"."$3}' | sort -u)
    say "scanning $(echo $PREFIXES | sed 's/ /.0\/24 /g').0/24 for ssh (port 22) ..."
    mapfile -t CANDS < <(for p in $PREFIXES; do for i in $(seq 1 254); do echo "$p.$i"; done; done \
      | xargs -P 128 -I{} bash -c 'timeout 1 bash -c "exec 3<>/dev/tcp/{}/22" 2>/dev/null && echo {}' | sort -t. -k4 -n)
    say "ssh open on: ${CANDS[*]:-none}"
  fi
  FOUND=()
  for ip in "${CANDS[@]}"; do
    printf '%s\n' "${MINE[@]}" | grep -qx "$ip" && continue               # this machine
    if out=$(ssh "${SSH_OPTS[@]}" "$USER_R@$ip" 'echo "$(hostname) cpu=$(nproc) mem=$(free -g | awk "/^Mem/{print \$2}")G disk_free=$(df -h ~ | awk "NR==2{print \$4}")"' 2>/dev/null); then
      say "  $USER_R@$ip  accepts the key  ($out)"; FOUND+=("$USER_R@$ip")
    elif [ ${#ARGS[@]} -gt 0 ]; then
      say "  $USER_R@$ip  does NOT accept the key - paste the worker command there (bash scripts/cluster_bootstrap.sh worker-cmd)"
    fi
  done
  if [ ${#FOUND[@]} -eq 0 ]; then
    say "no VM accepts the key yet. Paste the worker command on the other VMs, or give their IPs: discover IP1 IP2 ..."
    exit 1
  fi
  [ -f "$HOSTS_FILE" ] && cp "$HOSTS_FILE" "$HOSTS_FILE.bak.$(date +%Y%m%d-%H%M%S)"
  { echo "# written by cluster_bootstrap.sh discover $(date -Iseconds) on $(hostname -s)"; printf '%s\n' "${FOUND[@]}"; } > "$HOSTS_FILE"
  rm -f "$SHARED/cluster.weights"     # new machines -> shard weights are measured again at the next start
  say "wrote $HOSTS_FILE (${#FOUND[@]} VMs); old shard weights removed"
  [ ${#FOUND[@]} -ne "$EXPECT" ] && say "WARNING: expected $EXPECT VMs, found ${#FOUND[@]} - paste the worker command on the missing VMs and run discover again"
  echo
  bash "$ROOT/scripts/cluster.sh" check ;;
*)
  sed -n '2,19p' "$0"; exit 2 ;;
esac
