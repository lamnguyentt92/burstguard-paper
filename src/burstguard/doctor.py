"""`burstguard doctor`: environment checks with PASS/WARN/FAIL lines."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import shutil
import sys
import time
from typing import Any

from burstguard import __version__
from burstguard.paths import HF_DIR, PROCESSED_DIR, RAW_DIR, ROOT, SHARED_DIR
from burstguard.versioning import venv_problem

EXPECTED = {"flwr": "1.41.0", "torch": "2.11.0", "granite-tsfm": "0.3.10", "transformers": "5.20.2",
            "peft": "0.21.2", "ray": "2.55.1", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.7.2",
            "matplotlib": "3.11.2", "scipy": "1.17.1"}
URLS = {
    "pypi": "https://pypi.org/simple/flwr/",
    "huggingface": "https://huggingface.co/api/models/ibm-granite/granite-timeseries-ttm-r2",
    "pytorch_cpu_index": "https://download.pytorch.org/whl/cpu/torch/",
    "dataverse": "https://dataverse.harvard.edu/api/info/version",
    "github": "https://github.com",
}


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def run(check_net: bool = True, log=print) -> dict[str, Any]:
    res: dict[str, Any] = {"checks": []}

    def add(name: str, status: str, detail: str = "") -> None:
        res["checks"].append({"name": name, "status": status, "detail": detail})
        log(f"  [{status:4s}] {name}: {detail}")

    log(f"== burstguard v{__version__} at {ROOT}")
    v = sys.version_info
    add("python", "PASS" if (3, 11) <= v[:2] <= (3, 13) else "FAIL", f"{sys.version.split()[0]} ({sys.executable})")
    prob = venv_problem()
    add("venv", "PASS" if prob is None else "FAIL", "this checkout's .venv, version stamp matches" if prob is None else prob)

    log("== Packages (tested versions)")
    for pkg, want in EXPECTED.items():
        have = _ver(pkg)
        if have is None:
            add(pkg, "FAIL", "not installed")
        else:
            ok = have.split("+")[0] == want
            add(pkg, "PASS" if ok else "WARN", have + ("" if ok else f" (tested with {want})"))
    for mod in ("torch", "flwr", "ray", "tsfm_public", "peft", "transformers", "sklearn", "matplotlib"):
        t0 = time.time()
        try:
            importlib.import_module(mod)
            add(f"import {mod}", "PASS", f"{time.time() - t0:.1f}s")
        except Exception as e:
            add(f"import {mod}", "FAIL", f"{type(e).__name__}: {e}")

    log("== Hardware")
    try:
        import torch
        a = torch.randn(512, 512); t0 = time.time()
        for _ in range(20):
            a @ a
        add("torch", "PASS", f"cuda={torch.cuda.is_available()} (CPU is enough); 20x matmul 512^2 in {time.time() - t0:.2f}s")
    except Exception as e:
        add("torch_runtime", "FAIL", f"{type(e).__name__}: {e}")
    try:
        ncpu = len(os.sched_getaffinity(0))
    except Exception:
        ncpu = os.cpu_count() or 0
    add("cpu_count", "PASS" if ncpu >= 4 else "WARN", f"{ncpu} CPUs available (>= 4 recommended)")
    try:
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: int(l.split()[1]) for l in f if l.split()[1].isdigit()}
        avail = m.get("MemAvailable", 0) / 1024 / 1024
        add("memory", "PASS" if avail >= 6 else "WARN", f"{avail:.1f} GiB available (~1.4 GiB per worker)")
    except Exception:
        pass
    for label, path, need in (("disk_project", ROOT, 10e9), ("disk_shared", SHARED_DIR, 25e9)):
        try:
            du = shutil.disk_usage(str(path))
            add(label, "PASS" if du.free > need else "WARN", f"{du.free / 1e9:.0f} GB free at {path}")
        except Exception as e:
            add(label, "WARN", f"{path}: {e}")

    log("== Shared assets ($BURSTGUARD_SHARED)")
    add("shared_dir", "PASS" if os.access(SHARED_DIR, os.W_OK) else "FAIL", str(SHARED_DIR))
    raw = sorted((RAW_DIR / "milan").glob("sms-call-internet-mi-*")) if (RAW_DIR / "milan").exists() else []
    add("raw_milan", "PASS" if len(raw) >= 62 else "WARN", f"{len(raw)}/62 files in {RAW_DIR / 'milan'}")
    proc = sorted(PROCESSED_DIR.glob("*.npz")) if PROCESSED_DIR.exists() else []
    add("processed", "PASS" if proc else "WARN", ", ".join(p.name for p in proc) or "no .npz yet")
    hub = HF_DIR / "hub"
    add("hf_cache", "PASS" if any(hub.glob("models--ibm-granite*")) else "WARN",
        f"{HF_DIR} ({'TTM-R2 cached' if any(hub.glob('models--ibm-granite*')) else 'run burstguard model-download'})")

    if check_net:
        log("== Network (8 s timeout each)")
        import requests
        for name, url in URLS.items():
            try:
                r = requests.get(url, timeout=8, headers={"User-Agent": "burstguard-doctor"})
                add(f"net:{name}", "PASS" if r.status_code < 400 else "WARN", f"HTTP {r.status_code} {url}")
            except Exception as e:
                add(f"net:{name}", "WARN", f"{type(e).__name__} {url} - {str(e)[:120]}")
    res["n_fail"] = sum(c["status"] == "FAIL" for c in res["checks"])
    res["n_warn"] = sum(c["status"] == "WARN" for c in res["checks"])
    log(f"\nDOCTOR RESULT: {res['n_fail']} FAIL, {res['n_warn']} WARN")
    return res
