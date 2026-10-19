import json

from src.research.pedigree import jbis_births as jb

PAGE = "<html><head><title>Kampala(GB) | JBIS-Search</title></head><body><table><tr><th>Date of Birth (Age)</th><td> Feb 19,1976 </td></tr></table></body></html>"
OLD = "<html><head><title>Mahmoud | JBIS-Search</title></head><body><th>Date of Birth (Age)</th><td>1933</td></body></html>"


def test_parse_page_full_and_year_only():
    p = jb.parse_page(PAGE)
    assert (p["year"], p["month"], p["day"]) == (1976, 2, 19) and p["title"].startswith("Kampala")
    o = jb.parse_page(OLD)
    assert (o["year"], o["month"], o["day"]) == (1933, None, None)
    assert jb.parse_page("<html></html>")["year"] is None


def _r(**kw):
    base = {"status": "ok", "nk_year": 1976, "year": 1976, "month": 2, "day": 19}
    base.update(kw)
    return base


def test_decide():
    assert jb.decide(_r())[0] == "adopted"
    assert jb.decide(_r(year=1977))[0] == "candidate"
    assert jb.decide(_r(month=None, day=None))[0] == "no_data"
    assert jb.decide(_r(month=1, day=1))[0] == "no_data"
    assert jb.decide(_r(status="http_404"))[0] == "no_data"
    assert jb.decide(_r(year=None))[0] == "no_data"


def test_fetch_resumes_and_skips_done(tmp_path, monkeypatch):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([["a", "A", 1976, "1", "Q1", 5], ["b", "B", 1977, "2", "Q2", 4]]), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    out.write_text(json.dumps({"horse_id": "a"}) + "\n", encoding="utf-8")
    calls = []

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return PAGE.encode()

    monkeypatch.setattr(jb.urllib.request, "urlopen", lambda req, timeout=0: (calls.append(req.full_url), R())[1])
    assert jb.fetch(q, out, 0, None) == 0
    assert calls == ["https://www.jbis.jp/horse/2/"]
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[-1])["horse_id"] == "b"
