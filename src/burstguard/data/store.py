"""Processed dataset format (.npz) shared by the real Milan data and the synthetic data."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from burstguard import DATA_FORMAT, __version__

FREQ_MIN = 10
SLOTS_PER_DAY = 24 * 60 // FREQ_MIN  # 144

# Day boundaries (Milan local time, CET) counted from 2013-11-01 00:00.
DEFAULT_SPLITS = {
    "train": ("2013-11-01", "2013-12-08"),
    "val": ("2013-12-08", "2013-12-15"),
    "test": ("2013-12-15", "2013-12-25"),
    "holiday": ("2013-12-25", "2014-01-02"),
}
START_DATE = "2013-11-01"


def split_bounds(T: int, start: str = START_DATE, splits: dict | None = None) -> dict[str, list[int]]:
    splits = splits or DEFAULT_SPLITS
    d0 = dt.date.fromisoformat(start)
    out = {}
    for name, (a, b) in splits.items():
        ia = (dt.date.fromisoformat(a) - d0).days * SLOTS_PER_DAY
        ib = (dt.date.fromisoformat(b) - d0).days * SLOTS_PER_DAY
        out[name] = [max(0, min(ia, T)), max(0, min(ib, T))]
    return out


@dataclass
class Dataset:
    series: np.ndarray            # [S, T] float32, original units (non-negative)
    probe: np.ndarray             # [P, T] float32, public server-side data (areas not used by any site)
    bounds: dict[str, list[int]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_sites(self) -> int:
        return int(self.series.shape[0])

    @property
    def T(self) -> int:
        return int(self.series.shape[1])

    def site_scale(self, i: int) -> float:
        a, b = self.bounds["train"]
        s = float(np.std(self.series[i, a:b]))
        return s if s > 1e-8 else 1.0

    def probe_scale(self, j: int) -> float:
        a, b = self.bounds["train"]
        s = float(np.std(self.probe[j, a:b]))
        return s if s > 1e-8 else 1.0


def save_dataset(path: str | Path, series: np.ndarray, probe: np.ndarray,
                 meta: dict[str, Any], bounds: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series = np.asarray(series, dtype=np.float32)
    probe = np.asarray(probe, dtype=np.float32)
    bounds = bounds or split_bounds(series.shape[1])
    if not np.isfinite(series).all() or not np.isfinite(probe).all():
        raise ValueError("NaN/inf left after processing - check interpolation.")
    meta = {**meta, "data_format": DATA_FORMAT, "written_by": __version__}
    np.savez_compressed(path, series=series, probe=probe, bounds=np.array(json.dumps(bounds)),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    return path


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}. Create it with `burstguard data synthetic` (quick test) or "
            f"`burstguard data download` + `burstguard data prepare` (real Milan data).")
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if int(meta.get("data_format", 0)) != DATA_FORMAT:
        raise ValueError(f"{path} has data_format={meta.get('data_format')} but this code expects {DATA_FORMAT}; "
                         f"re-run `burstguard data prepare`.")
    return Dataset(series=z["series"], probe=z["probe"], bounds=json.loads(str(z["bounds"])), meta=meta)


def describe(ds: Dataset) -> dict[str, Any]:
    per_site = []
    for i in range(ds.num_sites):
        x = ds.series[i]
        per_site.append({"site": i, "mean": float(x.mean()), "std_train": ds.site_scale(i),
                         "zero_frac": float((x == 0).mean()), "min": float(x.min()), "max": float(x.max())})
    return {"source": ds.meta.get("source"), "num_sites": ds.num_sites, "T": ds.T,
            "days": ds.T / SLOTS_PER_DAY, "num_probe": int(ds.probe.shape[0]), "bounds": ds.bounds,
            "data_format": ds.meta.get("data_format"), "per_site": per_site}
