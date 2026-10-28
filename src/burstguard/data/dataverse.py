"""Download the Milan data from Harvard Dataverse (Native + Data Access API) with resume and MD5 verification.

Barlacchi et al. (2015) published two "Telecommunications activity" datasets (Milan, Trentino) with DOIs
10.7910/DVN/EGZHFV and 10.7910/DVN/QLCABU. Both are queried and only files of the requested region are kept
(`-mi-` Milan, `-tn-` Trentino).

Guestbook (Dataverse >= 6.10): files attached to a guestbook can no longer be fetched with a plain GET
(HTTP 400 "You may not download this file without the required Guestbook response for guestbookID N").
The documented flow is used instead:
  1. GET  /api/guestbooks/{N}                        -> required fields and custom questions
  2. POST /api/access/datafile/{id}  {"guestbookResponse": {name, email, institution, position, answers}}
     -> {"data": {"signedUrl": ...}}
  3. GET  signedUrl (supports Range -> resume)
The response is read from $BURSTGUARD_SHARED/dataverse_guestbook.toml, written as a template by
`burstguard data guestbook` and filled in by the user. An optional Harvard Dataverse API token is taken from the
DATAVERSE_API_TOKEN environment variable or `api_token` in that file (never from the command line, which is logged).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any

import requests

from burstguard.data.milan import FILE_RE
from burstguard.paths import SHARED_DIR

SERVER = "https://dataverse.harvard.edu"
TELECOM_DOIS = ["10.7910/DVN/EGZHFV", "10.7910/DVN/QLCABU"]
UA = {"User-Agent": "burstguard-research (+https://github.com/lamnguyentt92/burstguard-paper)"}
GUESTBOOK_FILE = SHARED_DIR / "dataverse_guestbook.toml"
_GB_RE = re.compile(r"guestbook\s*id\s*:?\s*(\d+)", re.I)


class GuestbookRequired(RuntimeError):
    def __init__(self, gid: int | None, message: str):
        super().__init__(message)
        self.gid = gid


class GuestbookNotFilled(RuntimeError):
    pass


class DataverseHTTPError(RuntimeError):
    def __init__(self, status: int, message: str, url: str):
        super().__init__(f"HTTP {status} from Dataverse for {url}: {message}")
        self.status = status


# ------------------------------------------------------------------ helpers

def _server_message(r: requests.Response) -> str:
    try:
        j = r.json()
        return str(j.get("message") or j.get("data") or j)[:500]
    except Exception:
        return (r.text or r.reason or "")[:500]


def _token(gb: dict[str, Any] | None = None) -> str | None:
    return os.environ.get("DATAVERSE_API_TOKEN") or ((gb or {}).get("api_token") or None)


def _headers(token: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    h = dict(UA)
    if token:
        h["X-Dataverse-key"] = token
    h.update(extra or {})
    return h


def _raise_for(r: requests.Response, url: str) -> None:
    if r.status_code < 400:
        return
    msg = _server_message(r)
    if r.status_code == 400 and "guestbook" in msg.lower():
        m = _GB_RE.search(msg)
        raise GuestbookRequired(int(m.group(1)) if m else None, msg)
    raise DataverseHTTPError(r.status_code, msg, url)


def _retryable(e: Exception) -> bool:
    if isinstance(e, (GuestbookRequired, GuestbookNotFilled)):
        return False
    if isinstance(e, DataverseHTTPError):
        return e.status in (408, 429) or e.status >= 500
    return True  # network errors, timeouts, truncated streams


def find_signed_url(obj: Any) -> str | None:
    """Return the signed URL from a Dataverse JSON reply ({"data": {"signedUrl": ...}}), searching nested keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and "url" in k.lower() and v.startswith("http"):
                return v
        for v in obj.values():
            u = find_signed_url(v)
            if u:
                return u
    elif isinstance(obj, list):
        for v in obj:
            u = find_signed_url(v)
            if u:
                return u
    return None


# ------------------------------------------------------------------ listing

def list_files(doi: str, timeout: float = 30, token: str | None = None) -> list[dict]:
    url = f"{SERVER}/api/datasets/:persistentId/"
    r = requests.get(url, params={"persistentId": f"doi:{doi}"}, headers=_headers(token), timeout=timeout)
    _raise_for(r, url)
    data = r.json()["data"]
    files = []
    for f in data["latestVersion"]["files"]:
        df = f.get("dataFile", {})
        name = df.get("originalFileName") or df.get("filename") or f.get("label", "")
        md5 = df.get("md5") or (df.get("checksum") or {}).get("value")
        files.append({"id": df.get("id"), "name": name, "label": f.get("label"),
                      "size": df.get("originalFileSize") or df.get("filesize"),
                      "md5": md5 if not df.get("originalFileFormat") else None,
                      "tabular": bool(df.get("originalFileFormat")),
                      "restricted": bool(f.get("restricted"))})
    return files


def select(files: list[dict], region: str, date_from: str | None, date_to: str | None) -> list[dict]:
    out = []
    for f in files:
        m = FILE_RE.search(f["name"] or "") or FILE_RE.search(f.get("label") or "")
        if not m or m.group(1) != region:
            continue
        d = m.group(2)
        if (date_from and d < date_from) or (date_to and d > date_to):
            continue
        out.append({**f, "date": d})
    return sorted(out, key=lambda f: f["date"])


# ------------------------------------------------------------------ guestbook

def get_guestbook(gid: int, token: str | None = None, timeout: float = 30) -> dict[str, Any]:
    url = f"{SERVER}/api/guestbooks/{gid}"
    r = requests.get(url, headers=_headers(token), timeout=timeout)
    _raise_for(r, url)
    return r.json().get("data", {})


def probe_guestbook(fid: int, tabular: bool = False, token: str | None = None, timeout: float = 30) -> int | None:
    """Return the guestbook id that blocks a plain download of file `fid`, or None if it downloads freely."""
    url = f"{SERVER}/api/access/datafile/{fid}"
    params = {"format": "original"} if tabular else {}
    with requests.get(url, params=params, headers=_headers(token, {"Range": "bytes=0-0"}), stream=True,
                      timeout=timeout) as r:
        try:
            _raise_for(r, url)
        except GuestbookRequired as e:
            return e.gid if e.gid is not None else -1
    return None


def guestbook_template(gid: int, gb: dict[str, Any] | None) -> str:
    """TOML template for the guestbook response (filled in by the user)."""
    def req(flag: str) -> str:
        if gb is None:
            return "required?"
        return "REQUIRED" if gb.get(flag) else "optional"
    lines = [
        f"# Harvard Dataverse guestbook response (guestbook id {gid}) for the Telecom Italia Milan data.",
        "# Fill in, save, then run:  burstguard data download",
        "# The same response is sent with every file download (one guestbook record per file, as in the web UI).",
        "# This file stays on this machine; it is not included in log reports.",
        f"guestbook_id = {gid}",
        f'name = ""            # {req("nameRequired")}',
        f'email = ""           # {req("emailRequired")}',
        f'institution = ""     # {req("institutionRequired")}',
        f'position = ""        # {req("positionRequired")}',
        '# api_token = ""     # optional: Harvard Dataverse API token (or export DATAVERSE_API_TOKEN=...)',
        "",
        "[answers]",
        "# question id = answer (text; a list of strings for multi-select questions)",
    ]
    qs = sorted((gb or {}).get("customQuestions", []) or [], key=lambda q: q.get("displayOrder", 0))
    if gb is None:
        lines.append("# (the guestbook questions could not be retrieved; the server error on download lists what is missing)")
    for q in qs:
        if q.get("hidden"):
            continue
        opts = [o.get("value") for o in q.get("optionValues", []) or []]
        lines.append(f"# {q.get('question', '').strip()}  [{'REQUIRED' if q.get('required') else 'optional'}"
                     f"; type={q.get('type')}]" + (f"  options: {opts}" if opts else ""))
        lines.append(f'"{q.get("id")}" = ""')
    if gb is not None and not qs:
        lines.append("# (this guestbook has no custom questions)")
    return "\n".join(lines) + "\n"


def load_guestbook_response(path: Path = GUESTBOOK_FILE, gb: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        raise GuestbookNotFilled(
            f"guestbook response file not found: {path}. Run `burstguard data guestbook`, fill in the file, "
            "then run `burstguard data download` again.")
    cfg = tomllib.loads(path.read_text())
    body: dict[str, Any] = {}
    for k in ("name", "email", "institution", "position"):
        v = str(cfg.get(k, "") or "").strip()
        if v:
            body[k] = v
    answers = []
    for qid, v in (cfg.get("answers") or {}).items():
        if v in ("", [], None):
            continue
        answers.append({"id": int(qid), "value": v})
    body["answers"] = answers
    missing = []
    if gb:
        for k in ("name", "email", "institution", "position"):
            if gb.get(f"{k}Required") and k not in body:
                missing.append(k)
        have = {a["id"] for a in answers}
        for q in gb.get("customQuestions", []) or []:
            if q.get("required") and not q.get("hidden") and int(q.get("id", -1)) not in have:
                missing.append(f"answers.\"{q.get('id')}\" ({str(q.get('question', ''))[:60]})")
    if missing:
        raise GuestbookNotFilled(f"guestbook response in {path} is missing required fields: {', '.join(missing)}")
    return {"body": body, "token": cfg.get("api_token") or None, "gid": cfg.get("guestbook_id")}


def request_signed_url(fid: int, gb_body: dict[str, Any], tabular: bool = False, token: str | None = None,
                       timeout: float = 60) -> str:
    url = f"{SERVER}/api/access/datafile/{fid}"
    params = {"format": "original"} if tabular else {}
    r = requests.post(url, params=params, json={"guestbookResponse": gb_body},
                      headers=_headers(token, {"Content-Type": "application/json"}), timeout=timeout)
    _raise_for(r, url)
    try:
        signed = find_signed_url(r.json())
    except json.JSONDecodeError:
        signed = None
    if not signed:
        raise DataverseHTTPError(r.status_code, f"no signed URL in reply: {(r.text or '')[:300]}", url)
    return signed


# ------------------------------------------------------------------ download

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_to(url: str, params: dict, headers: dict, part: Path) -> None:
    have = part.stat().st_size if part.exists() else 0
    h = dict(headers)
    if have:
        h["Range"] = f"bytes={have}-"
    with requests.get(url, params=params, headers=h, stream=True, timeout=60) as r:
        if r.status_code == 416:  # already complete
            return
        _raise_for(r, url.split("?")[0])
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(part, mode) as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def download_file(fid: int, dest: Path, size: int | None, md5: str | None, tabular: bool,
                  guestbook: dict[str, Any] | None = None, token: str | None = None,
                  retries: int = 5, log=print) -> str:
    """Download one file. Without `guestbook` a plain GET is used (raises GuestbookRequired when one is needed);
    with it, a signed URL is requested per attempt (POST with the guestbook response)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and size and dest.stat().st_size == size:
        if md5 is None or _md5(dest) == md5:
            return "skip (already present)"
    url = f"{SERVER}/api/access/datafile/{fid}"
    params = {"format": "original"} if tabular else {}
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            if guestbook is None:
                _stream_to(url, params, _headers(token), part)
            else:
                try:
                    signed = request_signed_url(fid, guestbook, tabular, token)
                except GuestbookRequired as e:  # the server rejected our response: report what it says
                    raise GuestbookNotFilled(f"Dataverse rejected the guestbook response: {e}") from None
                _stream_to(signed, {}, dict(UA), part)
            break
        except Exception as e:
            if not _retryable(e) or attempt == retries:
                raise
            log(f"    download error (attempt {attempt}/{retries}): {type(e).__name__}: {e}")
            time.sleep(min(30, 2 ** attempt))
    if not part.exists():
        raise IOError(f"download of {dest.name} produced no data")
    part.replace(dest)
    if size and dest.stat().st_size != size:
        log(f"    note: {dest.name} has {dest.stat().st_size} bytes, Dataverse lists {size}")
    if md5 and _md5(dest) != md5:
        raise IOError(f"MD5 mismatch for {dest.name} - delete it and download again")
    return "ok"


def _gather(region: str, date_from: str | None, date_to: str | None, dois: list[str] | None,
            token: str | None, log=print) -> list[dict]:
    listed, errors = [], []
    for doi in dois or TELECOM_DOIS:
        try:
            fs = list_files(doi, token=token)
            log(f"DOI {doi}: {len(fs)} files")
            listed += [{**f, "doi": doi} for f in fs]
        except Exception as e:
            errors.append(f"{doi}: {type(e).__name__}: {e}")
            log(f"DOI {doi}: ERROR {type(e).__name__}: {e}")
    if not listed:
        raise ConnectionError("could not list Dataverse files: " + " | ".join(errors))
    chosen = select(listed, region, date_from, date_to)
    if not chosen:
        sample = ", ".join(sorted({f["name"] for f in listed})[:10])
        raise FileNotFoundError(f"no file matches region '{region}'. Example names seen: {sample}")
    return chosen


def guestbook_setup(region: str = "mi", path: Path = GUESTBOOK_FILE, force: bool = False, log=print) -> dict:
    """Find the guestbook that protects the files, print its questions and write a template to fill in."""
    token = _token()
    chosen = _gather(region, None, None, None, token, log)
    gid = probe_guestbook(chosen[0]["id"], chosen[0]["tabular"], token)
    if gid is None:
        log("No guestbook response is required for these files: run `burstguard data download`.")
        return {"guestbook_required": False}
    gb, err = None, None
    if gid > 0:
        try:
            gb = get_guestbook(gid, token)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    log(f"Guestbook required: id={gid}" + (f" ('{gb.get('name')}')" if gb else ""))
    if gb:
        for k in ("name", "email", "institution", "position"):
            log(f"  {k:12s}: {'REQUIRED' if gb.get(f'{k}Required') else 'optional'}")
        for q in sorted(gb.get("customQuestions", []) or [], key=lambda q: q.get("displayOrder", 0)):
            if q.get("hidden"):
                continue
            opts = [o.get("value") for o in q.get("optionValues", []) or []]
            log(f"  question {q.get('id')}: {q.get('question')} [{'REQUIRED' if q.get('required') else 'optional'}; "
                f"{q.get('type')}]" + (f" options={opts}" if opts else ""))
    else:
        log(f"  could not read the guestbook definition ({err}); the template lists the standard fields only")
    if path.exists() and not force:
        log(f"Existing response file kept: {path} (use --force to overwrite with a fresh template)")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(guestbook_template(gid, gb))
        try:
            path.chmod(0o600)
        except OSError:
            pass
        log(f"Template written: {path}")
    log(f"Next: edit {path} (e.g. nano {path}), then run `burstguard data download`.")
    return {"guestbook_required": True, "guestbook_id": gid, "template": str(path),
            "questions": len((gb or {}).get("customQuestions", []) or []), "guestbook_read": gb is not None}


def download(region: str = "mi", dest: str | Path = "data/raw/milan", date_from: str | None = None,
             date_to: str | None = None, dois: list[str] | None = None, dry_run: bool = False,
             guestbook_file: Path = GUESTBOOK_FILE, log=print):
    dest = Path(dest)
    gb_file_cfg: dict[str, Any] | None = None
    if guestbook_file.exists():
        try:
            gb_file_cfg = tomllib.loads(guestbook_file.read_text())
        except Exception as e:
            raise GuestbookNotFilled(f"cannot parse {guestbook_file}: {e}") from e
    token = _token(gb_file_cfg)
    chosen = _gather(region, date_from, date_to, dois, token, log)
    total = sum(int(f["size"] or 0) for f in chosen)
    log(f"Selected {len(chosen)} files of region '{region}' ({chosen[0]['date']} .. {chosen[-1]['date']}), "
        f"total ~{total / 1e9:.2f} GB -> {dest}")
    if dry_run:
        for f in chosen:
            log(f"  {f['date']}  {f['name']}  {int(f['size'] or 0) / 1e6:.1f} MB  id={f['id']}")
        return {"chosen": len(chosen), "bytes": total, "dry_run": True}
    gb_body: dict[str, Any] | None = None
    done = 0
    t_all = time.time()
    for k, f in enumerate(chosen, 1):
        if f["restricted"]:
            raise PermissionError(f"file {f['name']} is restricted on Dataverse")
        t0 = time.time()
        try:
            st = download_file(f["id"], dest / f["name"], f["size"], f["md5"], f["tabular"],
                               guestbook=gb_body, token=token, log=log)
        except GuestbookRequired as e:
            gid = e.gid
            log(f"Dataverse requires a guestbook response (guestbook id {gid}) for these files.")
            gb_def = None
            try:
                gb_def = get_guestbook(gid, token) if gid and gid > 0 else None
            except Exception as ge:
                log(f"  (could not read guestbook definition: {type(ge).__name__}: {ge})")
            if gb_file_cfg is None:
                raise GuestbookNotFilled(
                    f"guestbook response needed (guestbook id {gid}). Run `burstguard data guestbook`, fill in "
                    f"{guestbook_file}, then run `burstguard data download` again.") from None
            gb_body = load_guestbook_response(guestbook_file, gb_def)["body"]
            log(f"Using the guestbook response from {guestbook_file} "
                f"({len(gb_body.get('answers', []))} custom answers).")
            st = download_file(f["id"], dest / f["name"], f["size"], f["md5"], f["tabular"],
                               guestbook=gb_body, token=token, log=log)
        done += 1
        mb = (dest / f["name"]).stat().st_size / 1e6 if (dest / f["name"]).exists() else 0
        log(f"  [{k}/{len(chosen)}] {f['name']}: {st} ({mb:.0f} MB, {time.time() - t0:.0f}s)")
    return {"chosen": len(chosen), "downloaded": done, "bytes": total, "dest": str(dest),
            "guestbook_used": gb_body is not None, "minutes": round((time.time() - t_all) / 60, 1)}
