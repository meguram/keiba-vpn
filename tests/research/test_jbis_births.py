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


SEARCH = ('<table><tr><td><a href="/horse/0000336378/">Kampala(GB)</a></td><td>Horse 1976 b. Kalamoun</td></tr>'
          '<tr><td><a href="/horse/0000999999/">Kampala(USA)</a></td><td>Horse 2001 ch.</td></tr>'
          '<tr><td><a href="/horse/0000111111/">Kampalas Star</a></td><td>1990</td></tr></table>')


def test_parse_search_and_select():
    c = jb.parse_search(SEARCH)
    assert [x["id"] for x in c] == ["0000336378", "0000999999", "0000111111"]
    assert c[0]["year_near"] == 1976 and c[1]["year_near"] == 2001
    assert jb.select_candidate(c, "Kampala", 1976) == ("0000336378", "ok")
    assert jb.select_candidate(c, "Kampala", 1990)[0] is None           # 同名は 2 頭だが生年が合う方が無い
    assert jb.select_candidate(c, "Nobody", 1976) == (None, "search_none")
    # 生年が読めない 1 件だけの一致は、馬ページで照合する前提で候補にする
    only = jb.parse_search('<a href="/horse/0000222222/">Solo(GB)</a>')
    assert jb.select_candidate(only, "Solo", 1960) == ("0000222222", "ok_name_only")


def test_fetch_searches_then_fetches_page(tmp_path, monkeypatch):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([["a", "Kampala", 1976, None, "Q1", 5], ["b", "Nobody", 1977, None, "Q2", 4]]), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    calls = []

    class R:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.body.encode()

    def fake(req, timeout=0):
        calls.append(req.full_url)
        return R(SEARCH if "navi_search" in req.full_url else PAGE)

    monkeypatch.setattr(jb.urllib.request, "urlopen", fake)
    assert jb.fetch(q, out, 0, None) == 0
    assert calls == ["https://www.jbis.jp/navi_search/?keyword=Kampala", "https://www.jbis.jp/horse/0000336378/", "https://www.jbis.jp/navi_search/?keyword=Nobody"]
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert recs[0]["status"] == "ok" and recs[0]["month"] == 2 and recs[0]["day"] == 19
    assert recs[1]["status"] == "search_none"
