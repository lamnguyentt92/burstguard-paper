"""Guard against mixing versions on the same machine.

Each version lives in its own checkout with its own `.venv`. `scripts/setup_venv.sh` writes the version it
installed into `.venv/burstguard_version`; every `burstguard` command checks that the active interpreter belongs
to *this* checkout's `.venv` and that the stamp matches the code version.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from burstguard import __version__
from burstguard.paths import ROOT


def venv_problem() -> str | None:
    if os.environ.get("BURSTGUARD_SKIP_VENV_CHECK") == "1":
        return None
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        return ("not running inside a virtual environment.\n"
                f"  cd {ROOT} && source .venv/bin/activate")
    expected = (ROOT / ".venv").resolve()
    active = Path(sys.prefix).resolve()
    if active != expected:
        return (f"the active virtual environment ({active}) does not belong to this checkout ({ROOT}).\n"
                f"  deactivate; cd {ROOT} && source .venv/bin/activate")
    stamp = expected / "burstguard_version"
    if not stamp.exists():
        return ("this .venv has no version stamp (created by an older setup script).\n"
                f"  cd {ROOT} && bash scripts/setup_venv.sh --recreate")
    built = stamp.read_text().strip()
    if built != __version__:
        return (f".venv was built for v{built} but the code is v{__version__}.\n"
                f"  cd {ROOT} && bash scripts/setup_venv.sh --recreate")
    return None
