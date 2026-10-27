"""SporthorseData 連携（sporthorse_births.py）のテスト。

ネットアクセスは一切行わない: ``sporthorse_session.fetch_html`` / ``probe`` を monkeypatch し、
Cloudflare チャレンジや 403 を模した応答で分岐を確認する。
"""
import csv
import json
from pathlib import Path

import pytest

from src.research.pedigree import sporthorse_births as sh
from src.research.pedigree import sporthorse_session as shs
from src.research.pedigree.sporthorse_session import SessionState, import_session, is_challenge, normalize_cookie_import
from src.research.pedigree import sporthorse_births as sb
from src.research.pedigree.sporthorse_session import SessionState, save_session


PAGE = """<html><head><title>Kris | SporthorseData</title></head><body>
<p>Date of Birth: 23-Mar-1976</p></body></html>"""
YEAR_ONLY = "<html><head><title>Old Horse</title></head><body>Date of Birth: 1933</body></html>"
CHALLENGE = "<html><head><title>Just a moment...</title></head></html>"


def test_parse_page_full_and_year_only():
    p = sh.parse_page(PAGE)
    assert (p["year"], p["month"], p["day"]) == (1976, 3, 23)
    assert "Kris" in p["title"]
    o = sh.parse_page(YEAR_ONLY)
    assert (o["year"], o["month"], o["day"]) == (1933, None, None)


def test_title_matches():
    assert sh.title_matches("Kris", "Kris | SporthorseData")
    assert sh.title_matches("Beta", "Beta - Thoroughbred Pedigree")


def test_decide():
    base = {"status": "ok", "nk_year": 1976, "year": 1976, "month": 3, "day": 23, "url": "https://x"}
    assert sh.decide(base)[0] == "adopted"
    assert sh.decide({**base, "year": 1977})[0] == "candidate"
    assert sh.decide({**base, "month": None})[0] == "no_data"
    assert sh.decide({**base, "month": 1, "day": 1})[0] == "no_data"


def test_normalize_cookie_import():
    cookies = normalize_cookie_import({"cookies": {"cf_clearance": "abc", "__cf_bm": "x"}})
    assert cookies["cf_clearance"] == "abc"
    lst = normalize_cookie_import([{"domain": ".sporthorse-data.com", "name": "cf_clearance", "value": "z"}])
    assert lst == {"cf_clearance": "z"}


def test_fetch_stops_on_challenge(tmp_path, monkeypatch):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([["a", "Kris", 1976, "https://sporthorse-data.com/pedigree/kris", 1]]), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    sess = SessionState(user_agent="Mozilla/5.0", cookies={"cf_clearance": "t"})
    monkeypatch.setattr(sh, "fetch_html", lambda url, st: (403, CHALLENGE, {"cf-mitigated": "challenge"}))
    assert sh.fetch(q, out, 0, None, sess, tmp_path, "https://sporthorse-data.com/pedigree", None) == 3


def test_fetch_parses_ok(tmp_path, monkeypatch):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([["a", "Kris", 1976, "https://sporthorse-data.com/pedigree/kris", 1]]), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    sess = SessionState(user_agent="Mozilla/5.0", cookies={"cf_clearance": "t"})
    monkeypatch.setattr(sh, "fetch_html", lambda url, st: (200, PAGE, {}))
    assert sh.fetch(q, out, 0, None, sess, tmp_path, "https://sporthorse-data.com/pedigree", None) == 0
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["status"] == "ok" and rec["month"] == 3


def test_session_import_roundtrip(tmp_path):
    p = import_session({"cookies": {"cf_clearance": "x"}}, path=tmp_path / "sess.json")
    assert p.exists()
    assert is_challenge(CHALLENGE, {}) is True


def test_pending_summary():
    q = [["a", "A", 1976, "u", 1], ["b", "B", 1977, "u", 2]]
    assert sh.pending_summary(q, Path("/nonexistent"))["pending"] == 2


def test_run_nightly_skips_without_session(tmp_path, monkeypatch):
    d = tmp_path / "d"
    d.mkdir()
    (d / "zone_review.csv").write_text("stallion_id,stallion_name,stallion_birth_year,conception_year,min_gen,n_horses,manual_value,note\n", encoding="utf-8")
    (d / "birth_manual.csv").write_text("horse_id,name,birth_year,birth_month,birth_day,source_url,note\n", encoding="utf-8")
    anc = tmp_path / "anc.csv"
    anc.write_text("horse_id,name,birthday,precision,source,source_url,wiki_conflict\n", encoding="utf-8")
    assert sh.run_nightly(anc, d / "zone_review.csv", None, d, d / "q.json", d / "r.jsonl", d / "sess.json", 0, 60, 10, "https://x", None) == 0


def test_run_nightly_loop(monkeypatch, tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    zone = d / "zone_review.csv"
    zone.write_text(
        "stallion_id,stallion_name,stallion_birth_year,conception_year,min_gen,n_horses,manual_value,note\n"
        "a,Alpha,1970,2020,2,5,,\n",
        encoding="utf-8",
    )
    (d / "birth_manual.csv").write_text("horse_id,name,birth_year,birth_month,birth_day,source_url,note\n", encoding="utf-8")
    anc = tmp_path / "anc.csv"
    anc.write_text(
        "horse_id,name,birthday,precision,source,source_url,wiki_conflict\n"
        "a,Alpha,1970-01-01,year,nk,,false\n",
        encoding="utf-8",
    )
    sess = tmp_path / "sess.json"
    import_session({"cookies": {"cf_clearance": "x"}}, path=sess)
    monkeypatch.setattr(shs, "probe", lambda st, p: 0)
    calls = []

    def fake_fetch(qp, op, delay, limit, st, dd, base, nk, iw=0):
        calls.append(limit)
        rec = {"horse_id": "a", "name": "Alpha", "nk_year": 1970, "status": "ok", "year": 1970, "month": 3, "day": 23, "url": "https://x"}
        op.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(sh, "fetch", fake_fetch)
    monkeypatch.setattr(sh, "apply", lambda *a: None)
    rc = sh.run_nightly(anc, zone, None, d, d / "q.json", d / "r.jsonl", sess, 1.0, 120.0, 10, "https://sporthorse-data.com/pedigree", None)
    assert rc == 0 and calls


# --- a05e360（レビュー時に追加したテスト） ---


def _write_csv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def dormello_dir(tmp_path):
    d = tmp_path / "dormello"
    d.mkdir()
    return d


@pytest.fixture
def ancestors_csv(tmp_path):
    p = tmp_path / "ancestors.csv"
    _write_csv(
        p,
        [
            ["h1", "Sunday Silence", "1986", "year"],
            ["h2", "Halo", "1969", "year"],
            ["h3", "DayPrecision", "1980-05-01", "day"],  # precision=day は対象外
        ],
        ["horse_id", "name", "birthday", "precision"],
    )
    return p


@pytest.fixture
def zone_csv(tmp_path):
    p = tmp_path / "zone_review.csv"
    _write_csv(
        p,
        [
            ["h1", "2", "10"],
            ["h2", "3", "5"],
            ["h3", "1", "99"],
        ],
        ["stallion_id", "min_gen", "n_horses"],
    )
    return p


def _session(tmp_path, cookies=None):
    return SessionState(user_agent="ua-test", cookies=cookies or {"cf_clearance": "x"})


# ── status（ネットアクセスなし） ─────────────────────────────────────


def test_status_report_no_session(dormello_dir, ancestors_csv, zone_csv):
    rep = sb.status_report(
        ancestors_csv, zone_csv, None, dormello_dir, None,
        dormello_dir / "results.jsonl", 10, "https://sporthorse-data.com/pedigree",
        None, dormello_dir / "sporthorse_session.json",
    )
    assert rep["session"] == {"present": False}
    assert rep["queue_total"] == 2  # h3 は precision=day のため除外
    assert rep["pending"] == 2
    assert rep["disabled"] is False


def test_status_report_with_session_and_results(dormello_dir, ancestors_csv, zone_csv):
    sess_path = dormello_dir / "sporthorse_session.json"
    save_session(_session(dormello_dir), sess_path)
    results = dormello_dir / "results.jsonl"
    results.write_text(json.dumps({"horse_id": "h1"}) + "\n", encoding="utf-8")
    rep = sb.status_report(
        ancestors_csv, zone_csv, None, dormello_dir, None, results, 10,
        "https://sporthorse-data.com/pedigree", None, sess_path,
    )
    assert rep["session"]["present"] is True
    assert rep["session"]["valid"] is True
    assert rep["fetched"] == 1
    assert rep["pending"] == 1


def test_status_report_detects_cron_disabled_marker(dormello_dir, ancestors_csv, zone_csv):
    (dormello_dir / "sporthorse_cron_disabled").write_text("", encoding="utf-8")
    rep = sb.status_report(
        ancestors_csv, zone_csv, None, dormello_dir, None,
        dormello_dir / "results.jsonl", 10, "https://sporthorse-data.com/pedigree",
        None, dormello_dir / "sporthorse_session.json",
    )
    assert rep["disabled"] is True


# ── run-nightly の SKIP 経路（ネットアクセスなし） ──────────────────


def test_run_nightly_skips_when_cron_disabled_marker_present(dormello_dir, ancestors_csv, zone_csv, capsys):
    (dormello_dir / "sporthorse_cron_disabled").write_text("", encoding="utf-8")
    rc = sb.run_nightly(
        ancestors_csv, zone_csv, None, dormello_dir,
        dormello_dir / "queue.json", dormello_dir / "results.jsonl",
        dormello_dir / "sporthorse_session.json",
        delay=1.0, max_seconds=1.0, limit=10, base_url="https://sporthorse-data.com/pedigree", sires=None,
    )
    assert rc == 0
    assert "SKIP" in capsys.readouterr().out


def test_run_nightly_skips_when_session_missing(dormello_dir, ancestors_csv, zone_csv, capsys):
    rc = sb.run_nightly(
        ancestors_csv, zone_csv, None, dormello_dir,
        dormello_dir / "queue.json", dormello_dir / "results.jsonl",
        dormello_dir / "sporthorse_session.json",  # 存在しない
        delay=1.0, max_seconds=1.0, limit=10, base_url="https://sporthorse-data.com/pedigree", sires=None,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "SKIP" in out and "セッション未登録" in out


def test_run_nightly_stops_when_probe_fails(monkeypatch, dormello_dir, ancestors_csv, zone_csv):
    sess_path = dormello_dir / "sporthorse_session.json"
    save_session(_session(dormello_dir), sess_path)
    monkeypatch.setattr("src.research.pedigree.sporthorse_session.probe", lambda st, path: 2)
    rc = sb.run_nightly(
        ancestors_csv, zone_csv, None, dormello_dir,
        dormello_dir / "queue.json", dormello_dir / "results.jsonl", sess_path,
        delay=1.0, max_seconds=5.0, limit=10, base_url="https://sporthorse-data.com/pedigree", sires=None,
    )
    assert rc == 3


# ── run-nightly のループ（fetch_html を模擬し、実通信なしで完了まで回す） ──


def test_run_nightly_loops_until_pending_zero(monkeypatch, dormello_dir, ancestors_csv, zone_csv):
    sess_path = dormello_dir / "sporthorse_session.json"
    save_session(_session(dormello_dir), sess_path)
    monkeypatch.setattr("src.research.pedigree.sporthorse_session.probe", lambda st, path: 0)

    def fake_fetch_html(url, state, timeout=30.0):
        html = "<title>Sunday Silence | SporthorseData</title>Date of Birth: 25-Mar-1986"
        return 200, html, {}

    monkeypatch.setattr(sb, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(sb, "title_matches", lambda name, title: True)

    rc = sb.run_nightly(
        ancestors_csv, zone_csv, None, dormello_dir,
        dormello_dir / "queue.json", dormello_dir / "results.jsonl", sess_path,
        delay=0.01, max_seconds=5.0, limit=10, base_url="https://sporthorse-data.com/pedigree", sires=None,
    )
    assert rc == 0
    rep = sb.status_report(
        ancestors_csv, zone_csv, None, dormello_dir, dormello_dir / "queue.json",
        dormello_dir / "results.jsonl", 10, "https://sporthorse-data.com/pedigree", None, sess_path,
    )
    assert rep["pending"] == 0


# ── fetch(): 403/429/5xx・Cloudflare チャレンジで即停止（回避しない） ──


@pytest.mark.parametrize("code", [403, 429, 500])
def test_fetch_stops_immediately_on_block_like_status(monkeypatch, dormello_dir, tmp_path, code):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps([["h1", "Sunday Silence", 1986, "", 10]]), encoding="utf-8")
    out_path = tmp_path / "out.jsonl"

    monkeypatch.setattr(sb, "fetch_html", lambda url, state, timeout=30.0: (code, "", {}))
    rc = sb.fetch(queue_path, out_path, 0.01, None, _session(tmp_path), dormello_dir, "https://sporthorse-data.com/pedigree", None)
    assert rc == 3
    assert not out_path.exists() or out_path.read_text(encoding="utf-8") == ""


def test_fetch_stops_on_cloudflare_challenge(monkeypatch, dormello_dir, tmp_path):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps([["h1", "Sunday Silence", 1986, "", 10]]), encoding="utf-8")
    out_path = tmp_path / "out.jsonl"

    monkeypatch.setattr(sb, "fetch_html", lambda url, state, timeout=30.0: (200, "Just a moment...", {}))
    rc = sb.fetch(queue_path, out_path, 0.01, None, _session(tmp_path), dormello_dir, "https://sporthorse-data.com/pedigree", None)
    assert rc == 3


# ── parse_page / title_matches ─────────────────────────────────────


def test_parse_page_extracts_full_date():
    html = "<title>Sunday Silence | SporthorseData</title>Date of Birth: 25-Mar-1986"
    rec = sb.parse_page(html)
    assert (rec["year"], rec["month"], rec["day"]) == (1986, 3, 25)


def test_parse_page_falls_back_to_year_only():
    html = "<title>Halo</title>born 1969"
    rec = sb.parse_page(html)
    assert rec["year"] == 1969
    assert rec["month"] is None


def test_title_matches_ignores_site_suffix_and_case():
    assert sb.title_matches("Sunday Silence", "Sunday Silence | SporthorseData")
    assert not sb.title_matches("Sunday Silence", "Halo | SporthorseData")


# ── decide() ─────────────────────────────────────────────────────


def test_decide_adopted_when_full_date_matches_netkeiba_year():
    rec = {"status": "ok", "year": 1986, "month": 3, "day": 25, "nk_year": 1986}
    assert sb.decide(rec)[0] == "adopted"


def test_decide_candidate_when_year_differs_from_netkeiba():
    rec = {"status": "ok", "year": 1985, "month": 3, "day": 25, "nk_year": 1986}
    assert sb.decide(rec)[0] == "candidate"


def test_decide_no_data_when_jan1_placeholder():
    rec = {"status": "ok", "year": 1986, "month": 1, "day": 1, "nk_year": 1986}
    assert sb.decide(rec)[0] == "no_data"


def test_decide_no_data_when_fetch_failed():
    assert sb.decide({"status": "not_found", "nk_year": 1986})[0] == "no_data"


# ── apply(): birth_manual.csv / birth_search_log.csv への反映 ─────


def test_apply_writes_manual_and_log_csv(dormello_dir):
    results = dormello_dir / "results.jsonl"
    results.write_text(
        json.dumps({
            "horse_id": "h1", "name": "Sunday Silence", "nk_year": 1986,
            "status": "ok", "url": "https://sporthorse-data.com/pedigree/sunday-silence",
            "title": "Sunday Silence | SporthorseData", "year": 1986, "month": 3, "day": 25,
        }) + "\n",
        encoding="utf-8",
    )
    sb.apply(results, dormello_dir)

    man_rows = list(csv.DictReader(open(dormello_dir / "birth_manual.csv", encoding="utf-8")))
    assert len(man_rows) == 1
    assert man_rows[0]["horse_id"] == "h1"
    assert man_rows[0]["birth_month"] == "3"

    log_rows = list(csv.DictReader(open(dormello_dir / "birth_search_log.csv", encoding="utf-8")))
    assert len(log_rows) == 1
    assert log_rows[0]["status"] == "adopted"

    urls = {r[0]: r[1] for r in list(csv.reader(open(dormello_dir / "sporthorse_urls.csv", encoding="utf-8")))[1:]}
    assert urls["h1"] == "https://sporthorse-data.com/pedigree/sunday-silence"


def test_apply_does_not_duplicate_manual_entry_already_present(dormello_dir):
    _write_csv(
        dormello_dir / "birth_manual.csv",
        [["h1", "Sunday Silence", "1986", "3", "25", "https://example.com", "既存"]],
        ["horse_id", "name", "birth_year", "birth_month", "birth_day", "source_url", "note"],
    )
    results = dormello_dir / "results.jsonl"
    results.write_text(
        json.dumps({
            "horse_id": "h1", "name": "Sunday Silence", "nk_year": 1986,
            "status": "ok", "url": "https://sporthorse-data.com/pedigree/sunday-silence",
            "title": "Sunday Silence | SporthorseData", "year": 1986, "month": 3, "day": 25,
        }) + "\n",
        encoding="utf-8",
    )
    sb.apply(results, dormello_dir)
    man_rows = list(csv.DictReader(open(dormello_dir / "birth_manual.csv", encoding="utf-8")))
    assert len(man_rows) == 1  # 既存分のまま増えない

    log_rows = list(csv.DictReader(open(dormello_dir / "birth_search_log.csv", encoding="utf-8")))
    assert len(log_rows) == 1  # ログには記録される
