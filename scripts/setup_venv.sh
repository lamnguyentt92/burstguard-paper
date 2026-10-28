#!/usr/bin/env bash
# =============================================================================
# BurstGuard - create this checkout's isolated virtual environment (.venv).
#
#   bash scripts/setup_venv.sh             # install / update
#   bash scripts/setup_venv.sh --recreate  # delete .venv and reinstall
#
# Per checkout (one directory per version, never shared between versions):
#   .venv/                     Python virtual environment, stamped with the version (.venv/burstguard_version)
#   .tools/uv, .tools/python   local uv + Python 3.12 (only if the machine lacks Python 3.11-3.13)
#   .cache/pip, .cache/uv      package caches
#   logs/setup_v<ver>_<ts>.log installation log (send it when reporting problems)
# Shared between versions:  $BURSTGUARD_SHARED (default ~/burstguard-shared): raw/processed data, model cache.
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VER="$(cat VERSION)"
mkdir -p logs .tools .cache
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/setup_v${VER}_${TS}.log"
STATUS="logs/setup_v${VER}_${TS}.status"
exec > >(tee -a "$LOG") 2>&1

RECREATE=0
for a in "$@"; do
  case "$a" in
    --recreate) RECREATE=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unsupported argument: $a"; exit 2 ;;
  esac
done

TORCH_VERSION="2.11.0"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
SHARED="${BURSTGUARD_SHARED:-$HOME/burstguard-shared}"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
export UV_NO_MODIFY_PATH=1
export UV_LINK_MODE=copy

step() { echo; echo "==================== [v$VER $(date +%H:%M:%S)] $*"; }
fail() {
  echo; echo "!!!!! SETUP FAILED (v$VER): $*"
  { echo "version=$VER"; echo "status=failed"; echo "reason=$*"; } > "$STATUS"
  echo "Collect logs for analysis:  bash scripts/collect_logs.sh --setup"
  exit 1
}
trap 'fail "command failed at line $LINENO: $BASH_COMMAND"' ERR

step "BurstGuard v$VER - environment setup in $ROOT"
echo "host=$(hostname)  user=$(id -un)  os=$(uname -srm)"
[ -r /etc/os-release ] && . /etc/os-release && echo "distro=${PRETTY_NAME:-?}"
echo "cpu=$(nproc)  mem=$(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)  disk_free=$(df -h . | awk 'NR==2{print $4}')"
echo "shared assets: $SHARED"
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$(cd "$VIRTUAL_ENV" 2>/dev/null && pwd)" != "$ROOT/.venv" ]; then
  echo "NOTE: another virtual environment is active ($VIRTUAL_ENV); it is not touched. Run 'deactivate' afterwards."
fi

# ---------------------------------------------------------------- 1. Python
step "1/6 Find Python 3.11-3.13"
py_ok() { "$1" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; }
PY=""
for c in "${BG_PYTHON:-}" "$ROOT/.tools/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12" python3.12 python3.11 python3.13 python3; do
  [ -n "$c" ] || continue
  if command -v "$c" >/dev/null 2>&1 && py_ok "$c" && "$c" -c 'import venv, ensurepip' 2>/dev/null; then
    # resolve to the base interpreter: if a virtualenv (e.g. this checkout's .venv) is active, its python
    # would be deleted together with the old .venv below
    PY="$("$c" -c 'import os, sys; print(os.path.realpath(getattr(sys, "_base_executable", "") or sys.executable))')"
    case "$PY" in "$ROOT"/.venv/*) PY=""; continue ;; esac
    break
  fi
done
UV=""
ensure_uv() {
  if [ -x "$ROOT/.tools/uv/uv" ]; then UV="$ROOT/.tools/uv/uv"; return 0; fi
  echo "installing uv locally into .tools/uv (no PATH changes, no sudo)..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools/uv" INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1 sh || true
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools/uv" INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1 sh || true
  fi
  if [ ! -x "$ROOT/.tools/uv/uv" ] && command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --target "$ROOT/.tools/uv-pip" uv >/dev/null 2>&1 || true
    [ -x "$ROOT/.tools/uv-pip/bin/uv" ] && mkdir -p "$ROOT/.tools/uv" && ln -sf "$ROOT/.tools/uv-pip/bin/uv" "$ROOT/.tools/uv/uv"
  fi
  [ -x "$ROOT/.tools/uv/uv" ] || return 1
  UV="$ROOT/.tools/uv/uv"
}
if [ -z "$PY" ]; then
  echo "no usable Python 3.11-3.13 (with venv) found -> downloading a private Python 3.12 via uv into .tools/python"
  ensure_uv || fail "could not install uv; install python3.12 + python3.12-venv or set BG_PYTHON=/path/to/python3.12"
  "$UV" python install 3.12 --no-bin || "$UV" python install 3.12 || fail "uv could not download Python 3.12"
  PY="$("$UV" python find 3.12 --managed-python 2>/dev/null || "$UV" python find 3.12)"
fi
echo "Python: $PY ($("$PY" -V 2>&1))"

# ---------------------------------------------------------------- 2. venv
step "2/6 Create .venv (version-stamped)"
# permission-only changes (e.g. chmod -R 777 on the home dir) must not mark the checkout as modified
[ -d "$ROOT/.git" ] && git -C "$ROOT" config core.fileMode false 2>/dev/null || true
if [ "$RECREATE" = "1" ] && [ -d .venv ]; then echo "removing old .venv"; rm -rf .venv; fi
if [ -d .venv ]; then
  OLD="$(cat .venv/burstguard_version 2>/dev/null || echo unknown)"
  if [ "$OLD" != "$VER" ] || ! py_ok .venv/bin/python; then
    echo ".venv belongs to version '$OLD' (code is v$VER) -> recreating to avoid mixing versions"
    rm -rf .venv
  fi
fi
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv || { ensure_uv && "$UV" venv --python "$PY" --seed .venv; } || fail "could not create .venv"
fi
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip --version >/dev/null 2>&1 || "$VPY" -m ensurepip --upgrade || fail "venv has no pip"
"$VPY" -m pip install --upgrade "pip>=24" wheel "setuptools>=68,<82" >/dev/null
echo "venv: $("$VPY" -V) at $ROOT/.venv"

# ---------------------------------------------------------------- 3. torch
step "3/6 Install torch==$TORCH_VERSION (CPU build)"
if "$VPY" -c "import torch,sys; sys.exit(0 if torch.__version__.split('+')[0]=='$TORCH_VERSION' else 1)" 2>/dev/null; then
  echo "torch $("$VPY" -c 'import torch;print(torch.__version__)') already installed"
elif "$VPY" -m pip install --retries 1 --timeout 20 "torch==$TORCH_VERSION" --index-url "$TORCH_CPU_INDEX"; then
  echo "installed CPU torch from $TORCH_CPU_INDEX"
else
  echo "WARNING: $TORCH_CPU_INDEX unreachable -> using PyPI (CUDA build, several GB, still runs on CPU)"
  "$VPY" -m pip install "torch==$TORCH_VERSION" || fail "could not install torch"
fi

# ---------------------------------------------------------------- 4. packages
step "4/6 Install pinned packages and the project (editable)"
"$VPY" -m pip install -r requirements.txt || fail "pip install -r requirements.txt failed"
"$VPY" -m pip install --no-deps -e . || fail "pip install -e . failed"
"$VPY" -m pip check || echo "WARNING: pip check reported conflicts (see above) - usually harmless"
"$VPY" -m pip freeze > "logs/pip_freeze_v${VER}_${TS}.txt"
echo "$VER" > .venv/burstguard_version

# ---------------------------------------------------------------- 5. shared dir
step "5/6 Shared assets directory"
mkdir -p "$SHARED/data/raw/milan" "$SHARED/data/processed" "$SHARED/hf"
echo "$SHARED  (data: $(du -sh "$SHARED/data" 2>/dev/null | cut -f1), model cache: $(du -sh "$SHARED/hf" 2>/dev/null | cut -f1))"
if [ "$SHARED" != "$HOME/burstguard-shared" ]; then
  echo "NOTE: BURSTGUARD_SHARED is set - export it in every shell that runs burstguard."
fi

# ---------------------------------------------------------------- 6. check
step "6/6 Quick check (burstguard doctor --no-net)"
set +e
"$ROOT/.venv/bin/burstguard" doctor --no-net
DOC=$?
set -e
trap - ERR
if [ "$DOC" = "0" ]; then
  { echo "version=$VER"; echo "status=success"; } > "$STATUS"
  echo; echo "SETUP COMPLETE (v$VER). Next:"
  echo "  source .venv/bin/activate"
  echo "  burstguard doctor && burstguard smoke"
else
  { echo "version=$VER"; echo "status=doctor_failed"; } > "$STATUS"
  echo; echo "Installed, but doctor reported FAIL items. Collect logs: bash scripts/collect_logs.sh --setup"
  exit 1
fi
