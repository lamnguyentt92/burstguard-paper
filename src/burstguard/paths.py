"""Project paths, shared assets and environment isolation.

Per-version (inside the checkout of one version, never shared):
    .venv/  .tools/  .cache/pip  runs/  results/  logs/  reports/
Shared between versions (large, immutable or versioned by file name):
    $BURSTGUARD_SHARED (default: ~/burstguard-shared)
        data/raw/milan/          raw Telecom Italia files (~21 GB, downloaded once)
        data/processed/          processed datasets, file names carry the data-format number
        hf/                      Hugging Face cache (TTM-R2 weights)

Import this module before torch / transformers / ray.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("BURSTGUARD_ROOT")
    if env:
        return Path(env).resolve()
    cand = Path(__file__).resolve().parents[2]
    if (cand / "pyproject.toml").exists():
        return cand
    return Path.cwd().resolve()


ROOT = project_root()
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
SHARED_DIR = Path(os.environ.get("BURSTGUARD_SHARED", str(Path.home() / "burstguard-shared"))).expanduser().resolve()
DATA_DIR = SHARED_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HF_DIR = SHARED_DIR / "hf"


def isolate_environment() -> None:
    """Point third-party caches into the project/shared dirs and disable telemetry."""
    defaults = {
        "HF_HOME": str(HF_DIR),
        "TORCH_HOME": str(CACHE_DIR / "torch"),
        "MPLCONFIGDIR": str(CACHE_DIR / "matplotlib"),
        "XDG_CACHE_HOME": str(CACHE_DIR / "xdg"),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "RAY_USAGE_STATS_ENABLED": "0",
        "RAY_DEDUP_LOGS": "0",
        "FLWR_TELEMETRY_ENABLED": "0",
        "TSFM_PYTHON_LOGGING_LEVEL": "ERROR",
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_VERBOSITY": "error",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    for d in (CACHE_DIR, RUNS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for d in (RAW_DIR, PROCESSED_DIR, HF_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def resolve(p: str | os.PathLike) -> Path:
    """Resolve a config path. 'shared:<rel>' -> $BURSTGUARD_SHARED/<rel>; relative -> project root."""
    s = str(p)
    if s.startswith("shared:"):
        return SHARED_DIR / s[len("shared:"):].lstrip("/")
    q = Path(s).expanduser()
    return q if q.is_absolute() else (ROOT / q)
