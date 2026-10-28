"""Reader for the Telecom Italia Big Data Challenge - Milan (Barlacchi et al., Scientific Data 2015, ODbL).

Raw format (one file per day, e.g. `sms-call-internet-mi-2013-11-01.txt`), TAB separated, no header:
    square_id, time_interval(ms epoch UTC), country_code, smsin, smsout, callin, callout, internet
Empty field = no activity. A (square, time) pair has one row per country_code -> rows are summed.
Some mirrors add a header (CellID, datetime, countrycode, ..., internet) and/or use commas; both are detected.
Files may be .txt, .csv, .gz or .zip.

Milan grid: 100x100 squares (~235 m), IDs 1..10000, assumed row-major: id-1 = row*100 + col.
Trentino (v0.3.16, region "tn"): same file format and period; the province grid is irregular (6575 valid squares),
so sites and probes are single squares chosen by traffic (no grid geometry needed): squares active in almost every
slot, ranked by total internet traffic, every `stride`-th of the ranking, probes interleaved like the Milan blocks.

Traffic types (`activities`): internet (default, the paper's main data), sms = smsin + smsout, call = callin + callout.
All types come from the same files and the same site/probe blocks; several are extracted in one pass.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from burstguard.data.store import FREQ_MIN, SLOTS_PER_DAY, split_bounds

GRID = 100
# 2013-11-01 00:00 Milan time (CET = UTC+1) = 2013-10-31 23:00 UTC
T0_MS = 1383260400000
SLOT_MS = FREQ_MIN * 60 * 1000
DAYS_DEFAULT = 62
FILE_RE = re.compile(r"sms-call-internet-(mi|tn)-(\d{4}-\d{2}-\d{2})")

NAME_ALIASES = {
    "square": {"square_id", "squareid", "cellid", "cell_id", "gridid", "grid_id", "square"},
    "time": {"time_interval", "timeinterval", "datetime", "time", "timestamp"},
    "internet": {"internet", "internet_traffic", "internettraffic"},
    "smsin": {"smsin", "sms_in", "sms-in"},
    "smsout": {"smsout", "sms_out", "sms-out"},
    "callin": {"callin", "call_in", "call-in"},
    "callout": {"callout", "call_out", "call-out"},
}
RAW_COLS = {"square": 0, "time": 1, "smsin": 3, "smsout": 4, "callin": 5, "callout": 6, "internet": 7}
ACTIVITIES = {"internet": ("internet",), "sms": ("smsin", "smsout"), "call": ("callin", "callout")}
CHUNK_ROWS = 2_000_000   # rows per read chunk: bounded memory (prepare can run next to experiments)


def processed_name(activity: str, region: str = "mi") -> str:
    """File name of a processed dataset: internet keeps the historic name milan_d<format>.npz; Trentino (v0.3.16)
    uses trentino_d<format>.npz / trentino_<activity>_d<format>.npz."""
    from burstguard import DATA_FORMAT
    base = "milan" if region == "mi" else "trentino"
    return f"{base}_d{DATA_FORMAT}.npz" if activity == "internet" else f"{base}_{activity}_d{DATA_FORMAT}.npz"


def list_raw_files(raw_dir: str | Path, region: str = "mi") -> list[Path]:
    raw_dir = Path(raw_dir)
    files = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file():
            m = FILE_RE.search(p.name)
            if m and m.group(1) == region:
                files.append(p)
    return files


def _open_text(path: Path) -> io.TextIOBase | str:
    name = path.name.lower()
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if not members:
            raise ValueError(f"empty zip file: {path}")
        return io.TextIOWrapper(zf.open(members[0]), encoding="utf-8", errors="replace")
    return str(path)  # pandas decompresses .gz/.bz2/.xz by extension


def _sniff(path: Path) -> tuple[str, bool]:
    """Return (delimiter, has_header)."""
    src = _open_text(path)
    if isinstance(src, str):
        import gzip, bz2, lzma
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(Path(src).suffix.lower(), open)
        with opener(src, "rt", encoding="utf-8", errors="replace") as f:
            first = f.readline()
    else:
        first = src.readline()
        src.close()
    delim = "\t" if "\t" in first else ","
    tok = first.strip().split(delim)[0].strip().strip('"')
    has_header = not tok.replace(".", "", 1).isdigit()
    return delim, has_header


def read_day(path: str | Path, squares: set[int] | None = None,
             activities: tuple[str, ...] = ("internet",)) -> pd.DataFrame:
    """Read one day file -> DataFrame [square:int, slot:int, <activity>:float ...], summed over country codes
    (sms = smsin + smsout, call = callin + callout). Read in chunks and filtered to `squares` per chunk."""
    path = Path(path)
    for a in activities:
        if a not in ACTIVITIES:
            raise ValueError(f"unknown activity {a!r} (choose from {sorted(ACTIVITIES)})")
    raw = sorted({c for a in activities for c in ACTIVITIES[a]}, key=lambda c: RAW_COLS[c])
    delim, has_header = _sniff(path)
    src = _open_text(path)
    if has_header:
        head = pd.read_csv(_open_text(path), sep=delim, nrows=0)
        lower = {c.strip().lower(): c for c in head.columns}
        pick = {}
        for want in ["square", "time"] + raw:
            hit = [lower[c] for c in lower if c in NAME_ALIASES[want]]
            if not hit:
                raise ValueError(f"{path.name}: column '{want}' not found in header {list(head.columns)}")
            pick[want] = hit[0]
        reader = pd.read_csv(src, sep=delim, usecols=list(pick.values()), low_memory=False, chunksize=CHUNK_ROWS)
        rename = {v: k for k, v in pick.items()}
    else:
        cols = ["square", "time"] + raw
        reader = pd.read_csv(src, sep=delim, header=None, usecols=[RAW_COLS[c] for c in cols], names=None,
                             low_memory=False, chunksize=CHUNK_ROWS)
        rename = {RAW_COLS[c]: c for c in cols}
    parts = []
    for ch in reader:
        ch = ch.rename(columns=rename)
        ch = ch.dropna(subset=["square", "time"])
        ch["square"] = ch["square"].astype(np.int64)
        if squares is not None:
            ch = ch[ch["square"].isin(squares)]
        if len(ch):
            parts.append(ch)
    cols_out = ["square", "slot"] + list(activities)
    if not parts:
        return pd.DataFrame(columns=cols_out)
    df = pd.concat(parts, ignore_index=True)
    t = df["time"]
    if pd.api.types.is_numeric_dtype(t):
        ms = t.astype(np.int64)
    else:  # some mirrors store local-time strings
        ts = pd.to_datetime(t)
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("Europe/Rome")
        ms = (ts.dt.tz_convert("UTC").astype("int64") // 1_000_000)
    df["slot"] = ((ms - T0_MS) // SLOT_MS).astype(np.int64)
    for c in raw:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for a in activities:
        if ACTIVITIES[a] != (a,):
            df[a] = sum(df[c] for c in ACTIVITIES[a])
    return df.groupby(["square", "slot"], as_index=False)[list(activities)].sum()[cols_out]


def build_square_matrices(files: Iterable[Path], squares: list[int], T: int, activities: tuple[str, ...],
                          log=print) -> tuple[dict[str, np.ndarray], dict]:
    """Merge day files into one matrix [len(squares), T] per activity. A slot missing for one square -> 0
    (no activity); a slot missing for ALL squares (outage) -> linear interpolation."""
    sq_index = {s: i for i, s in enumerate(squares)}
    Ms = {a: np.zeros((len(squares), T), dtype=np.float64) for a in activities}
    seen = np.zeros(T, dtype=bool)
    sqset = set(squares)
    files = list(files)
    for k, f in enumerate(files, 1):
        df = read_day(f, sqset, activities)
        df = df[(df["slot"] >= 0) & (df["slot"] < T)]
        rows = df["square"].map(sq_index).to_numpy()
        cols = df["slot"].to_numpy().astype(np.int64)
        for a in activities:
            np.add.at(Ms[a], (rows, cols), df[a].to_numpy())
        seen[np.unique(cols)] = True
        log(f"  [{k}/{len(files)}] {f.name}: {len(df):,} records, {len(np.unique(cols))} slots")
    gaps = np.where(~seen)[0]
    if len(gaps) and seen.any():
        idx = np.arange(T)
        for M in Ms.values():
            for i in range(M.shape[0]):
                M[i, ~seen] = np.interp(idx[~seen], idx[seen], M[i, seen])
    stats = {"files": len(files), "slots_total": T, "slots_missing_all_squares": int(len(gaps)),
             "missing_frac": float(len(gaps) / T)}
    return {a: M.astype(np.float32) for a, M in Ms.items()}, stats


def build_square_matrix(files: Iterable[Path], squares: list[int], T: int,
                        log=print) -> tuple[np.ndarray, dict]:
    """Internet traffic only (historic interface)."""
    Ms, stats = build_square_matrices(files, squares, T, ("internet",), log=log)
    return Ms["internet"], stats


def choose_blocks(num_sites: int, num_probe: int, block: int = 5, grid: int = GRID):
    """Pick non-overlapping block x block groups of squares spread from the centre to the periphery.
    Sites follow quantiles of the distance to the centre; probe blocks are interleaved (no shared squares)."""
    nb = grid // block
    centers = []
    for br in range(nb):
        for bc in range(nb):
            r, c = br * block + (block - 1) / 2, bc * block + (block - 1) / 2
            d = float(np.hypot(r - (grid - 1) / 2, c - (grid - 1) / 2))
            centers.append((d, br, bc))
    centers.sort()
    total = num_sites + num_probe
    usable = int(len(centers) * 0.8)  # drop the 20% farthest blocks (grid corners, very low activity)
    if total > usable:
        raise ValueError(f"requested {total} blocks but only {usable} are usable with block={block}")
    picks = np.linspace(0, usable - 1, total).round().astype(int).tolist()  # increasing, unique
    if num_probe:
        probe_idx = set(np.linspace(1, total - 2, num_probe).round().astype(int).tolist())
        while len(probe_idx) < num_probe:  # guard against rounding collisions
            probe_idx.add(next(i for i in range(total) if i not in probe_idx))
    else:
        probe_idx = set()
    site_pos = [p for i, p in enumerate(picks) if i not in probe_idx]
    probe_pos = [p for i, p in enumerate(picks) if i in probe_idx]
    def squares_of(pos):
        _, br, bc = centers[pos]
        return [(br * block + dr) * grid + (bc * block + dc) + 1 for dr in range(block) for dc in range(block)]
    sites = [{"site": i, "block_row": centers[p][1], "block_col": centers[p][2],
              "dist_center": round(centers[p][0], 1), "squares": squares_of(p)} for i, p in enumerate(site_pos)]
    probes = [{"probe": i, "block_row": centers[p][1], "block_col": centers[p][2],
               "dist_center": round(centers[p][0], 1), "squares": squares_of(p)} for i, p in enumerate(probe_pos)]
    return sites, probes


def rank_squares(files: Iterable[Path], T: int, log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First pass (Trentino): per square, total internet traffic and number of active slots within [0, T).
    Returns (square ids, totals, active slot counts)."""
    tot: dict[int, float] = {}
    cnt: dict[int, int] = {}
    files = list(files)
    for k, f in enumerate(files, 1):
        df = read_day(f, None, ("internet",))
        df = df[(df["slot"] >= 0) & (df["slot"] < T) & (df["internet"] > 0)]
        g = df.groupby("square")["internet"].agg(["sum", "count"])
        for sq, (sm, c) in zip(g.index.to_numpy(), g.to_numpy()):
            tot[int(sq)] = tot.get(int(sq), 0.0) + float(sm)
            cnt[int(sq)] = cnt.get(int(sq), 0) + int(c)
        log(f"  rank pass [{k}/{len(files)}] {f.name}: {len(g):,} active squares")
    sq = np.array(sorted(tot), dtype=np.int64)
    return sq, np.array([tot[s] for s in sq]), np.array([cnt[s] for s in sq])


def choose_ranked(num_sites: int, num_probe: int, squares: np.ndarray, totals: np.ndarray, counts: np.ndarray,
                  T: int, stride: int = 2, min_active: float = 0.95):
    """Single-square sites and probes (Trentino): squares with internet traffic in >= min_active of the slots,
    ranked by total traffic; every `stride`-th of the ranking is used (neighbouring ranks are often adjacent
    squares of the same town); probes are interleaved as in choose_blocks."""
    ok = counts >= min_active * T
    order = [int(i) for i in np.argsort(-totals) if ok[i]]
    total = num_sites + num_probe
    picks = order[::max(1, int(stride))][:total]
    if len(picks) < total:
        raise ValueError(f"only {len(picks)} squares with traffic in >= {100 * min_active:.0f} % of the slots "
                         f"(need {total}); lower the stride or min_active")
    probe_idx = set(np.linspace(1, total - 2, num_probe).round().astype(int).tolist()) if num_probe else set()
    while len(probe_idx) < num_probe:
        probe_idx.add(next(i for i in range(total) if i not in probe_idx))
    rank = {p: r for r, p in enumerate(order)}
    def info(p):
        return {"square": int(squares[p]), "rank": int(rank[p]), "total_internet": float(totals[p]),
                "active_frac": round(float(counts[p]) / T, 4), "squares": [int(squares[p])]}
    site_pos = [p for i, p in enumerate(picks) if i not in probe_idx]
    probe_pos = [p for i, p in enumerate(picks) if i in probe_idx]
    sites = [{"site": i, **info(p)} for i, p in enumerate(site_pos)]
    probes = [{"probe": i, **info(p)} for i, p in enumerate(probe_pos)]
    return sites, probes


def prepare_multi(raw_dir: str | Path, activities: tuple[str, ...] = ("internet",), num_sites: int = 20,
                  num_probe: int = 8, block: int = 5, days: int = DAYS_DEFAULT, log=print, region: str = "mi",
                  stride: int = 2):
    """One pass over the raw files -> {activity: (series, probe, meta, bounds)} with the same site/probe blocks
    (Milan) or squares (Trentino: one extra ranking pass)."""
    files = list_raw_files(raw_dir, region)
    if not files:
        raise FileNotFoundError(
            f"no 'sms-call-internet-{region}-YYYY-MM-DD.*' files in {raw_dir}. "
            f"Download with `burstguard data download --region {region}` or copy the files there.")
    T = days * SLOTS_PER_DAY
    if region == "mi":
        sites, probes = choose_blocks(num_sites, num_probe, block)
        unit = f"{block}x{block} blocks"
    else:
        log(f"Ranking the squares of {len(files)} files by internet traffic (region {region})")
        sq, tot, cnt = rank_squares(files, T, log=log)
        # strictest selection first; relax (denser ranking, then fewer active slots) instead of failing a campaign
        tries = [(stride, 0.95), (1, 0.95), (1, 0.9), (1, 0.8)]
        for k, (st, act) in enumerate(tries):
            try:
                sites, probes = choose_ranked(num_sites, num_probe, sq, tot, cnt, T, stride=st, min_active=act)
                break
            except ValueError as e:
                if k == len(tries) - 1:
                    raise
                log(f"  {e} -> trying stride {tries[k + 1][0]}, min_active {tries[k + 1][1]}")
        stride = st
        unit = f"single squares, every {st}-th of the traffic ranking, active in >= {100 * act:.0f} % of the slots"
    if len(sites) < num_sites or len(probes) < num_probe:
        raise ValueError("could not select enough site/probe blocks - reduce num_sites/num_probe or block")
    all_sq = sorted({s for g in sites + probes for s in g["squares"]})
    log(f"Reading {len(files)} files, keeping {len(all_sq)} squares ({num_sites} sites + {num_probe} probes, "
        f"{unit}); traffic: {', '.join(activities)}")
    Ms, stats = build_square_matrices(files, all_sq, T, tuple(activities), log=log)
    pos = {s: i for i, s in enumerate(all_sq)}
    covered = sorted({FILE_RE.search(f.name).group(2) for f in files})
    out = {}
    for a, M in Ms.items():
        def agg(groups):
            return np.stack([M[[pos[s] for s in g["squares"]]].sum(axis=0) for g in groups])
        meta = {"source": "milan" if region == "mi" else "trentino", "region": region, "activity": a,
                "freq_min": FREQ_MIN, "start": "2013-11-01T00:00:00+01:00",
                "block": block if region == "mi" else 1, "sites": sites, "probes": probes, "raw_stats": stats,
                "dates_covered": [covered[0], covered[-1]], "num_files": len(files),
                "license": "ODbL - Telecom Italia Big Data Challenge (Barlacchi et al. 2015)"}
        out[a] = (agg(sites), agg(probes), meta, split_bounds(T))
    return out


def prepare(raw_dir: str | Path, num_sites: int = 20, num_probe: int = 8, block: int = 5,
            days: int = DAYS_DEFAULT, log=print, activity: str = "internet"):
    return prepare_multi(raw_dir, (activity,), num_sites, num_probe, block, days, log)[activity]
