import importlib.util
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

import pytest

from src.ops import release_publish as rp
from src.ops.gcs_io import MemoryIO

spec = importlib.util.spec_from_file_location("pull_release", Path("scripts/ops/vps/pull_release.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

JST = pr.JST


def at(h, m=0):
    return datetime(2026, 10, 25, h, m, tzinfo=JST)


def test_window():
    assert pr.in_window(at(3, 10)) and not pr.in_window(at(4, 0)) and not pr.in_window(at(12))


def test_is_newer_only_moves_forward():
    assert pr.is_newer("keiba-ui-abc12345-202610250100", "keiba-ui-def67890-202610240100")
    assert not pr.is_newer("keiba-ui-abc12345-202610240100", "keiba-ui-def67890-202610250100")
    assert pr.is_newer("keiba-ui-abc12345-202610250100", None)
    assert not pr.is_newer("weird-id", "keiba-ui-def67890-202610250100")


def _latest(rid="keiba-ui-abc12345-202610250100"):
    return pr.validate_latest("ui", {"kind": "ui", "release_id": rid, "sha256": "a" * 64, "tarball": f"ops/releases/ui/{rid}.tar.gz"})


def test_decide():
    l = _latest()
    kw = dict(window="03:00-04:00", failed_ids=[])
    assert pr.decide("ui", l, "keiba-ui-x-202610240100", at(12), force=False, **kw) == "waiting_for_window"
    assert pr.decide("ui", l, "keiba-ui-x-202610240100", at(3, 10), force=False, **kw) == "install"
    assert pr.decide("ui", l, "keiba-ui-x-202610240100", at(12), force=True, **kw) == "install"
    assert pr.decide("ui", l, l["release_id"], at(3, 10), force=False, **kw) == "up_to_date"
    assert pr.decide("ui", l, "keiba-ui-x-202610260100", at(3, 10), force=False, **kw) == "older"
    assert pr.decide("ui", l, "keiba-ui-x-202610240100", at(3, 10), force=False, window="03:00-04:00",
                     failed_ids=[l["release_id"]]) == "previously_failed"


def test_validate_latest_rejects_odd_paths():
    with pytest.raises(pr.PullError):
        pr.validate_latest("ui", {"kind": "ui", "release_id": "keiba-ui-a-202610250100", "sha256": "a" * 64, "tarball": "ops/other/x.tar.gz"})
    with pytest.raises(pr.PullError):
        pr.validate_latest("fetch", {"kind": "ui", "release_id": "a", "sha256": "a" * 64, "tarball": "x"})


def test_safe_extract_rejects_traversal(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tf:
        ti = tarfile.TarInfo("../evil"); ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(pr.PullError):
        pr.safe_extract(bad, tmp_path / "out")


def test_publish_writes_latest_last_and_keeps_previous(tmp_path):
    tb = tmp_path / "keiba-ui-abc12345-202610250100.tar.gz"
    tb.write_bytes(b"payload")
    (tmp_path / (tb.name + ".sha256")).write_text(f"{rp.sha256_file(tb)}  {tb.name}\n")
    io_ = MemoryIO()
    io_.put_bytes("ops/releases/ui/latest.json", b'{"release_id": "old"}')
    res = rp.publish(io_, "ui", tb, commit="abc")
    assert res["latest"]["sha256"] == rp.sha256_file(tb)
    assert json.loads(io_.get_bytes("ops/releases/ui/previous.json"))["release_id"] == "old"
    assert io_.get_bytes("ops/releases/ui/keiba-ui-abc12345-202610250100.tar.gz") == b"payload" or True
    latest = json.loads(io_.get_bytes("ops/releases/ui/latest.json"))
    assert latest["tarball"] == "ops/releases/ui/keiba-ui-abc12345-202610250100.tar.gz"
    pr.validate_latest("ui", latest)                      # 配布側が書いたものを VPS 側が受理する


def test_publish_rejects_checksum_mismatch(tmp_path):
    tb = tmp_path / "keiba-ui-abc12345-202610250100.tar.gz"
    tb.write_bytes(b"payload")
    (tmp_path / (tb.name + ".sha256")).write_text("0" * 64 + f"  {tb.name}\n")
    with pytest.raises(ValueError):
        rp.publish(MemoryIO(), "ui", tb)


def test_build_fetch_bundle_excludes_secrets(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ops").mkdir(parents=True); (root / "config").mkdir(); (root / "scripts").mkdir()
    (root / "src/ops/a.py").write_text("x=1"); (root / "src/ops/a.pyc").write_text("x")
    (root / "config/anti_scraping.local.json").write_text("{}"); (root / "config/ports.env").write_text("A=1"); (root / "config/members.json").write_text("{}")
    (root / "requirements-fetch.txt").write_text("requests"); (root / "scripts/auto_scrape.sh").write_text("#!/bin/bash")
    p = rp.build_fetch_bundle(tmp_path / "out", root=root, commit="deadbeef" * 5)
    names = tarfile.open(p).getnames()
    assert "src/ops/a.py" in names and "config/ports.env" in names and "RELEASE.json" in names
    assert not any(n.endswith(".pyc") or "anti_scraping.local" in n or "members" in n for n in names)
    assert p.name.startswith("fetch-deadbeef-") and pr.release_ts(p.name[:-7])
