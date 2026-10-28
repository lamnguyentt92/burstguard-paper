"""Dataverse download flow against a fake server: plain GET, guestbook-protected GET (HTTP 400), POST with the
guestbook response -> signed URL -> GET, resume, and no retries on client errors."""
import hashlib
import json
import os

os.environ.setdefault("BURSTGUARD_SKIP_VENV_CHECK", "1")

import pytest

from burstguard.data import dataverse as DV

PAYLOAD = b"square\tts\tcc\tsmsin\n" * 5000
MD5 = hashlib.md5(PAYLOAD).hexdigest()
GB = {"id": 96, "name": "Telecom guestbook", "nameRequired": True, "emailRequired": True,
      "institutionRequired": False, "positionRequired": False,
      "customQuestions": [{"id": 501, "question": "Purpose of use?", "required": True, "type": "text",
                           "displayOrder": 0, "hidden": False}]}


class R:
    def __init__(self, status, body=b"", js=None):
        self.status_code, self._body, self._js = status, body, js
        self.text = json.dumps(js) if js is not None else body.decode(errors="ignore")
        self.reason = ""

    def json(self):
        if self._js is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._js

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeServer:
    def __init__(self, guestbook=True):
        self.guestbook = guestbook
        self.calls = []
        self.posts = []

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        self.calls.append(("GET", url))
        if "/api/datasets/" in url:
            files = [{"label": "sms-call-internet-mi-2013-11-01.txt",
                      "dataFile": {"id": 11, "filename": "sms-call-internet-mi-2013-11-01.txt",
                                   "filesize": len(PAYLOAD), "md5": MD5}}]
            return R(200, js={"data": {"latestVersion": {"files": files}}})
        if "/api/guestbooks/96" in url:
            return R(200, js={"status": "OK", "data": GB})
        if url.startswith("https://signed.example/"):
            rng = (headers or {}).get("Range")
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                return R(206, PAYLOAD[start:])
            return R(200, PAYLOAD)
        if "/api/access/datafile/" in url:
            if self.guestbook:
                return R(400, js={"status": "ERROR", "message": "You may not download this file without the "
                                                                 "required Guestbook response for guestbookID 96."})
            return R(200, PAYLOAD)
        return R(404, js={"message": "not found"})

    def post(self, url, params=None, json=None, headers=None, timeout=None):
        self.posts.append(json)
        gb = (json or {}).get("guestbookResponse", {})
        if not gb.get("email") or not any(a["id"] == 501 for a in gb.get("answers", [])):
            return R(400, js={"status": "ERROR", "message": "Guestbook response is missing required fields"})
        return R(200, js={"status": "OK", "data": {"signedUrl": "https://signed.example/f11?until=x&signed=y"}})


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer()
    monkeypatch.setattr(DV.requests, "get", srv.get)
    monkeypatch.setattr(DV.requests, "post", srv.post)
    monkeypatch.setattr(DV.time, "sleep", lambda s: None)
    return srv


def test_guestbook_flow(server, tmp_path):
    gbf = tmp_path / "gb.toml"
    # 1) no response file -> clear instruction, no retries of the 400
    with pytest.raises(DV.GuestbookNotFilled, match="burstguard data guestbook"):
        DV.download(dest=tmp_path / "raw", dois=["x"], guestbook_file=gbf, log=lambda *a: None)
    assert sum(1 for m, u in server.calls if "/access/" in u) == 1
    # 2) template from the guestbook definition, incomplete answers -> rejected before any POST
    gbf.write_text(DV.guestbook_template(96, GB))
    assert '"501" = ""' in gbf.read_text() and "REQUIRED" in gbf.read_text()
    with pytest.raises(DV.GuestbookNotFilled, match="missing required"):
        DV.download(dest=tmp_path / "raw", dois=["x"], guestbook_file=gbf, log=lambda *a: None)
    assert not server.posts
    # 3) filled in -> POST guestbook response, signed URL, file verified by MD5
    gbf.write_text(gbf.read_text().replace('name = ""', 'name = "A B"').replace('email = ""', 'email = "a@b.c"')
                   .replace('"501" = ""', '"501" = "research"'))
    res = DV.download(dest=tmp_path / "raw", dois=["x"], guestbook_file=gbf, log=lambda *a: None)
    assert res["guestbook_used"] and res["downloaded"] == 1
    assert (tmp_path / "raw" / "sms-call-internet-mi-2013-11-01.txt").read_bytes() == PAYLOAD
    body = server.posts[-1]["guestbookResponse"]
    assert body["answers"] == [{"id": 501, "value": "research"}] and "institution" not in body
    # 4) second run skips the verified file
    res = DV.download(dest=tmp_path / "raw", dois=["x"], guestbook_file=gbf, log=lambda *a: None)
    assert len(server.posts) == 1


def test_resume_and_plain_download(server, tmp_path):
    server.guestbook = False
    dest = tmp_path / "f.txt"
    (tmp_path / "f.txt.part").write_bytes(PAYLOAD[:1000])
    assert DV.download_file(11, dest, len(PAYLOAD), MD5, False, log=lambda *a: None) == "ok"
    assert dest.read_bytes() == PAYLOAD


def test_signed_url_parsing_and_probe(server):
    assert DV.find_signed_url({"status": "OK", "data": {"signedUrl": "https://x/y"}}) == "https://x/y"
    assert DV.find_signed_url({"data": {"nested": [{"downloadUrl": "https://z"}]}}) == "https://z"
    assert DV.find_signed_url({"data": {"message": "no"}}) is None
    assert DV.probe_guestbook(11) == 96
    server.guestbook = False
    assert DV.probe_guestbook(11) is None


def test_guestbook_setup_writes_template(server, monkeypatch, tmp_path):
    monkeypatch.setattr(DV, "TELECOM_DOIS", ["x"])
    p = tmp_path / "gb.toml"
    res = DV.guestbook_setup(path=p, log=lambda *a: None)
    assert res["guestbook_id"] == 96 and res["questions"] == 1 and p.exists()
    assert "Purpose of use?" in p.read_text() and oct(p.stat().st_mode)[-3:] == "600"
    p.write_text("kept")
    DV.guestbook_setup(path=p, log=lambda *a: None)
    assert p.read_text() == "kept"
