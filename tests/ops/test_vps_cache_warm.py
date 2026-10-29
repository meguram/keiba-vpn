"""VPS の事前ロード係(cache_warm.py)の、目印の拾い方・取り直し先・取り直しの進め方と、導入物の整合のテスト。"""
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VPS = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "vps"
_spec = importlib.util.spec_from_file_location("cache_warm", VPS / "cache_warm.py")
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc)      # JST 12:00
RACE = "202610250811"


def _name(day, key):
    return f"ops/cache_dirty/{day}/{key}"


def test_parse_marker():
    assert cw.parse_marker(_name("20261025", RACE)) == ("20261025", RACE)
    assert cw.parse_marker(_name("20261025", "_ALL")) == ("20261025", "_ALL")
    assert cw.parse_marker(_name("20261025", "20261026")) == ("20261025", "20261026")
    assert cw.parse_marker(_name("20261025", "junk")) is None
    assert cw.parse_marker("ops/cache_dirty/20261025/") is None
    assert cw.parse_marker("ops/other/20261025/" + RACE) is None


def test_paths_for_key():
    race = [p for p, _ in cw.paths_for_key(RACE, NOW)]
    assert f"/api/race/{RACE}" in race and f"/api/race/{RACE}/predictions" in race
    date = [p for p, _ in cw.paths_for_key("20261025", NOW)]
    assert "/api/race-list/20261025" in date and "/api/race-calendar?month=202610" in date
    allp = cw.paths_for_key("_ALL", NOW)
    assert ("/api/race-list/20261024", True) in allp and ("/api/race-list/20261027", True) in allp
    assert cw.paths_for_key("zzz", NOW) == []


def test_discover_dedupes_detects_changes_and_prunes():
    st = {"seen": {}, "queue": []}
    blobs = [(_name("20261025", RACE), 1), (_name("20261025", "20261025"), 5), ("ops/cache_dirty/bad", 9)]
    added = cw.discover(blobs, st, NOW)
    assert added and len(added) == len(set(added))
    assert cw.discover(blobs, st, NOW) == []                       # 世代が同じなら何もしない
    again = cw.discover([(_name("20261025", RACE), 2), (_name("20261025", "20261025"), 5)], st, NOW)
    assert all(p.startswith("/api/race/") or p.startswith("/api/v1/races/") for p in again)    # 変わったレースの分だけ
    assert _name("20261025", RACE) in st["seen"]
    cw.discover([], st, NOW)
    assert st["seen"] == {}                                        # 一覧から消えた目印は忘れる


class FakeWarmer:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def fetch(self, path):
        self.calls.append(path)
        r = self.responses.get(path, (200, b"{}"))
        return r(path) if callable(r) else r


def _drain(state, w, **kw):
    return cw.drain(state, w, max_requests=kw.get("max_requests", 100), max_sec=60, pause=0, log=lambda *_: None)


def test_drain_race_detail_queues_its_date_list():
    st = {"seen": {}, "queue": [{"p": f"/api/race/{RACE}", "n": 0}]}
    w = FakeWarmer({f"/api/race/{RACE}": (200, json.dumps({"date": "2026-10-25"}).encode())})
    res = _drain(st, w)
    assert "/api/race-list/20261025" in w.calls and "/api/race-calendar?month=202610" in w.calls
    assert res["left"] == 0 and res["ng"] == 0


def test_drain_all_expands_list_to_races():
    doc = {"races": [{"race_id": RACE}, {"race_id": "bad"}]}
    st = {"seen": {}, "queue": [{"p": "/api/race-list/20261025", "n": 0, "x": 1}]}
    w = FakeWarmer({"/api/race-list/20261025": (200, json.dumps(doc).encode())})
    _drain(st, w)
    assert f"/api/race/{RACE}" in w.calls and f"/api/race/{RACE}/predictions" in w.calls


def test_drain_retries_then_gives_up_and_treats_404_as_done():
    st = {"seen": {}, "queue": [{"p": "/a", "n": 0}, {"p": "/b", "n": 0}]}
    w = FakeWarmer({"/a": (503, b""), "/b": (404, b"")})
    r1 = _drain(st, w)
    assert r1["ng"] == 1 and r1["ok"] == 1 and r1["left"] == 1
    _drain(st, w)
    r3 = _drain(st, w)
    assert r3["left"] == 0 and w.calls.count("/a") == cw.MAX_ATTEMPTS


def test_drain_respects_request_budget():
    st = {"seen": {}, "queue": [{"p": f"/p{i}", "n": 0} for i in range(10)]}
    res = _drain(st, FakeWarmer({}), max_requests=4)
    assert res["ok"] == 4 and res["left"] == 6


def test_state_roundtrip_and_corrupt_file(tmp_path):
    f = tmp_path / "sub" / "state.json"
    cw.save_state(f, {"seen": {"a": 1}, "queue": [{"p": "/x", "n": 0}]})
    assert cw.load_state(f)["seen"] == {"a": 1}
    f.write_text("{broken", encoding="utf-8")
    assert cw.load_state(f) == {"seen": {}, "queue": []}


def test_main_requires_bucket_and_token(monkeypatch, tmp_path):
    monkeypatch.setenv("WARM_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    assert cw.main([]) == 2
    monkeypatch.setenv("WARM_TOKEN_FILE", str(tmp_path / "none"))
    assert cw.main(["--path", "/api/race-calendar?month=202610"]) == 2


def test_main_path_mode_warms_via_fake(monkeypatch, tmp_path):
    tok = tmp_path / "tok"
    tok.write_text("secret\n", encoding="utf-8")
    monkeypatch.setenv("WARM_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("WARM_TOKEN_FILE", str(tok))
    seen = {}

    def fetch(self, path):
        seen["path"], seen["token"], seen["host"] = path, self.token, self.host
        return 200, b"{}"

    monkeypatch.setattr(cw.Warmer, "fetch", fetch)
    assert cw.main(["--path", "/api/race-list/20261025"]) == 0
    assert seen == {"path": "/api/race-list/20261025", "token": "secret", "host": "app.megu-keiba.com"}
    assert cw.load_state(tmp_path / "state.json")["queue"] == []


def test_installer_and_units_are_consistent():
    sh = VPS / "vps_10_cache_warm.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    svc = (VPS / "templates" / "keiba-cache-warm.service").read_text(encoding="utf-8")
    tmr = (VPS / "templates" / "keiba-cache-warm.timer").read_text(encoding="utf-8")
    script = sh.read_text(encoding="utf-8")
    assert "/opt/keiba-ui/bin/cache_warm.py" in svc and "/opt/keiba-ui/bin/cache_warm.py" in script
    assert "EnvironmentFile=/etc/keiba-pull/pull.env" in svc
    assert "ReadWritePaths=/var/lib/keiba-ui/cache-warm" in svc and "/var/lib/keiba-ui/cache-warm" in script
    assert "OnUnitActiveSec=60s" in tmr
    assert "/etc/keiba-ui/refresh.token" in script


def test_pull_sa_script_grants_cache_dirty_list_read_only():
    """読み取り専用 SA が目印を一覧できないと 403 で事前ロードが動かない(2026-10-29)。範囲は ops/cache_dirty/ の閲覧だけ。"""
    sh = (Path(__file__).resolve().parents[2] / "scripts" / "gcp" / "setup_pull_sa.sh").read_text(encoding="utf-8")
    assert 'DIRTY="ops/cache_dirty/"' in sh
    assert "storage.googleapis.com/objectListPrefix" in sh, "バケットの一覧には接頭辞の条件が要る"
    block = sh[sh.index('DIRTY="ops/cache_dirty/"'):]
    block = block[:block.index("if [ -n")]
    assert "roles/storage.objectViewer" in block and "objectUser" not in block and "objectAdmin" not in block
