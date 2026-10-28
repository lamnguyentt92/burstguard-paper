"""One run = one directory runs/<timestamp>_v<version>_<command>[_<name>]/ with:
command.txt, config.toml, env.json, console.log (captured at file-descriptor level, so Flower/Ray/C output is
included), events.jsonl, summary.json (status=success|failed|interrupted) and error.txt on failure.
Every JSON record carries the code version.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from burstguard import VERSION_TAG, __version__
from burstguard.paths import ROOT, RUNS_DIR, SHARED_DIR


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git_info() -> dict[str, str]:
    def _run(*args: str) -> str:
        try:
            # core.fileMode=false: permission-only changes (e.g. `chmod -R 777`) are not code changes
            return subprocess.run(["git", "-C", str(ROOT), "-c", "core.fileMode=false", *args], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    status = _run("status", "--porcelain")
    return {"commit": _run("rev-parse", "--short", "HEAD"),
            "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": _run("describe", "--tags", "--always", "--dirty"),
            "dirty": "yes" if status else "no",
            "dirty_files": status.splitlines()[:20]}


class _FdTee:
    """Duplicate fds 1 and 2 into a log file through a `tee` subprocess."""

    def __init__(self, logfile: Path):
        self.logfile = logfile
        self.proc = None
        self.saved: list[int] = []
        self._fh = None

    def start(self) -> None:
        sys.stdout.flush(); sys.stderr.flush()
        self.saved = [os.dup(1), os.dup(2)]
        try:
            self.proc = subprocess.Popen(["tee", "-a", str(self.logfile)], stdin=subprocess.PIPE)
            target = self.proc.stdin.fileno()
        except FileNotFoundError:
            self._fh = open(self.logfile, "a", buffering=1)
            target = self._fh.fileno()
        os.dup2(target, 1); os.dup2(target, 2)

    def stop(self) -> None:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        if self.saved:
            os.dup2(self.saved[0], 1); os.dup2(self.saved[1], 2)
            for fd in self.saved:
                os.close(fd)
            self.saved = []
        if self.proc is not None:
            try:
                self.proc.stdin.close(); self.proc.wait(timeout=10)
            except Exception:
                pass
        if self._fh is not None:
            self._fh.close()


class RunContext:
    def __init__(self, command: str, name: str | None = None, argv: list[str] | None = None,
                 base_dir: Path | None = None, tee: bool = True):
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in (name or ""))
        rid = f"{ts}_{VERSION_TAG}_{command}" + (f"_{safe}" if safe else "")
        base = base_dir or RUNS_DIR
        self.dir = base / rid
        i = 1
        while self.dir.exists():
            self.dir = base / f"{rid}-{i}"; i += 1
        self.dir.mkdir(parents=True)
        self.command = command
        self.argv = argv if argv is not None else sys.argv
        self.t0 = time.time()
        self.summary: dict[str, Any] = {}
        self._tee = _FdTee(self.dir / "console.log") if tee else None
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)

    def event(self, kind: str, **data: Any) -> None:
        rec = {"v": __version__, "t": round(time.time() - self.t0, 3), "kind": kind, **data}
        self._events.write(json.dumps(rec, default=json_default) + "\n")

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        if isinstance(obj, dict):
            obj = {"version": __version__, **obj}
        p.write_text(json.dumps(obj, indent=2, default=json_default, ensure_ascii=False))
        return p

    def set(self, **kv: Any) -> None:
        self.summary.update(kv)

    def __enter__(self) -> "RunContext":
        if self._tee:
            self._tee.start()
        (self.dir / "command.txt").write_text(" ".join(shlex.quote(a) for a in self.argv) + "\n")
        (RUNS_DIR / "LATEST").write_text(str(self.dir) + "\n")
        print(f"[burstguard {VERSION_TAG}] {now_iso()} run dir: {self.dir}", flush=True)
        self.save_json("env.json", basic_env())
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "success"
        if exc_type is not None:
            status = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            err = "".join(traceback.format_exception(exc_type, exc, tb))
            (self.dir / "error.txt").write_text(f"burstguard {VERSION_TAG}\n" + err)
            print(err, file=sys.stderr, flush=True)
            self.summary.update(error_type=exc_type.__name__, error_message=str(exc)[:2000])
            try:
                from burstguard.hints import hint_for
                h = hint_for(exc)
            except Exception:
                h = None
            if h:
                self.summary["hint"] = h
                print(f"\n[burstguard {VERSION_TAG}] HINT: {h}", file=sys.stderr, flush=True)
        summ = {"status": status, "command": self.command, "run_id": self.dir.name, "git": git_info(),
                "started": _dt.datetime.fromtimestamp(self.t0).astimezone().isoformat(timespec="seconds"),
                "ended": now_iso(), "duration_s": round(time.time() - self.t0, 1), **self.summary}
        self.save_json("summary.json", summ)
        if status == "success":
            print(f"\n[burstguard {VERSION_TAG}] RUN OK ({summ['duration_s']}s) -> {self.dir}", flush=True)
        else:
            print(f"\n[burstguard {VERSION_TAG}] RUN {status.upper()} -> {self.dir}\n"
                  f"[burstguard {VERSION_TAG}] Collect logs for analysis: bash scripts/collect_logs.sh", flush=True)
        self._events.close()
        if self._tee:
            self._tee.stop()
        return False


def basic_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "time": now_iso(), "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version.split()[0], "executable": sys.executable,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "cpu_count": os.cpu_count(), "cwd": os.getcwd(), "root": str(ROOT), "shared": str(SHARED_DIR),
        "git": git_info(),
    }
    try:
        info["sched_affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
        info["mem_total"] = m.get("MemTotal"); info["mem_available"] = m.get("MemAvailable")
    except Exception:
        pass
    for k in ("OMP_NUM_THREADS", "HF_HOME", "HF_HUB_OFFLINE", "VIRTUAL_ENV", "BURSTGUARD_SHARED"):
        info[f"env.{k}"] = os.environ.get(k)
    return info


def json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)
