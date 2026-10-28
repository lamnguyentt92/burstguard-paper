"""Quiet third-party logging, in the driver and in every worker process (process pool / Ray actors).

granite-tsfm calls logging.basicConfig() at import, which installs a root handler without a level; Flower's logger
is at DEBUG and propagates to the root, so without this every SecAgg+ stage is printed by every Ray actor.
"""
from __future__ import annotations

import importlib
import logging
import warnings

_QUIET = ("transformers", "tsfm_public", "urllib3", "filelock", "ray", "huggingface_hub", "httpx", "httpx2",
          "accelerate", "datasets", "matplotlib", "PIL")


def quiet_logging() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
    for h in root.handlers:
        h.setLevel(logging.WARNING)
    root.setLevel(logging.WARNING)
    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.ERROR)
    # Flower keeps its own console handler (INFO); stop the DEBUG records from also reaching the root handler
    fl = logging.getLogger("flwr")
    fl.propagate = False
    for h in fl.handlers:
        if h.level < logging.INFO:
            h.setLevel(logging.INFO)
    silence_tsfm()


def silence_tsfm() -> None:
    """tsfm_public.toolkit.get_model logs through logging.getLogger(__file__) (a path, not a module name).
    Note: `import tsfm_public.toolkit.get_model as gm` yields the *function* (the package re-exports it),
    so the module must be taken from importlib."""
    try:
        mod = importlib.import_module("tsfm_public.toolkit.get_model")
        getattr(mod, "LOGGER", logging.getLogger(mod.__file__)).setLevel(logging.ERROR)
    except Exception:
        pass
    for name in list(logging.Logger.manager.loggerDict):
        if "tsfm_public" in name:
            logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hl
        hl.set_verbosity_error()
        hl.disable_progress_bar()
    except Exception:
        pass
