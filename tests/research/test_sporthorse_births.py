import json
from pathlib import Path

from src.research.pedigree import sporthorse_births as sh
from src.research.pedigree import sporthorse_session as shs
from src.research.pedigree.sporthorse_session import SessionState, import_session, is_challenge, normalize_cookie_import


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
