import csv
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from src.research.pedigree import manual_entry_server as me


def _setup(tmp_path):
    anc = tmp_path / "anc.csv"
    anc.write_text("horse_id,name,birthday,precision,source,source_url,wiki_conflict\n"
                   "a,Alpha,1970,year,netkeiba,u,\nb,Beta,1980,year,netkeiba,u,\nc,Gamma,1990,year,netkeiba,u,\n", encoding="utf-8")
    zone = tmp_path / "zone.csv"
    zone.write_text("stallion_id,stallion_name,stallion_birth_year,conception_year,min_gen,n_horses,manual_value,note\n"
                    "a,A,1970,2000,1,30,,\nb,B,1980,2000,1,20,,\nc,C,1990,2000,1,10,,\n", encoding="utf-8")
    d = tmp_path / "meta"
    d.mkdir()
    (d / "birth_search_log.csv").write_text(",".join(me.LOG_COLUMNS) + "\nb,Beta,no_data,1980,,,,2026-10-17,x\n", encoding="utf-8")
    return me.Store(anc, zone, None, d, "https://example.test/p"), d


def test_parse_date_input():
    assert me.parse_date_input("1970-01-30") == (1970, 1, 30)
    assert me.parse_date_input("30-Jan-1970") == (1970, 1, 30)
    assert me.parse_date_input("Jan 30, 1970") == (1970, 1, 30)
    assert me.parse_date_input("1970年1月30日") == (1970, 1, 30)
    assert me.parse_date_input("1970-02-30") is None and me.parse_date_input("abc") is None


def test_save_none_skip_undo_flow(tmp_path):
    s, d = _setup(tmp_path)
    assert [r["horse_id"] for r in s.pending()] == ["a", "b", "c"]
    assert s.save("a", "1971-03-04")["ok"] is False                       # 生年違い
    r = s.save("a", "1970-01-01")
    assert r["ok"] is False and r["need_confirm"] is True                 # 1 月 1 日は確認が要る
    assert s.save("a", "30-Jan-1970")["ok"] is True
    rows = list(csv.reader(open(d / "birth_manual.csv", encoding="utf-8")))
    assert rows[-1][:5] == ["a", "Alpha", "1970", "1", "30"] and rows[-1][5] == "https://example.test/p/alpha"
    log = {r[0]: r for r in csv.reader(open(d / "birth_search_log.csv", encoding="utf-8"))}
    assert log["a"][2] == "adopted"
    assert [r["horse_id"] for r in s.pending()] == ["b", "c"]
    assert s.none("b")["ok"] and [r["horse_id"] for r in s.pending()] == ["c"]
    s.skip("c")
    assert s.next()["remaining"] == 1 and s.next()["saved"] == 1 and s.next()["none"] == 1
    # 取り消し: 直前の「記載なし」→ 元のログ行（no_data の古い行）が戻る
    assert s.undo()["ok"] and [r["horse_id"] for r in s.pending()] == ["b", "c"]
    assert s.undo()["ok"]                                                   # 保存の取り消し
    rows = list(csv.reader(open(d / "birth_manual.csv", encoding="utf-8")))
    assert [r[0] for r in rows[1:]] == []
    log = {r[0]: r for r in csv.reader(open(d / "birth_search_log.csv", encoding="utf-8"))}
    assert "a" not in log and log["b"][8] == "x"
    assert s.undo()["ok"] is False


def test_http_endpoints_require_token(tmp_path):
    s, d = _setup(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), me.make_handler(s, "tok"))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert 'const TOKEN="tok"' in page
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/next", headers={"X-Token": "tok"})
        assert json.load(urllib.request.urlopen(req))["item"]["horse_id"] == "a"
        bad = urllib.request.Request(f"http://127.0.0.1:{port}/api/next")
        try:
            urllib.request.urlopen(bad)
            raise AssertionError("token なしで通った")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        post = urllib.request.Request(f"http://127.0.0.1:{port}/api/save", data=json.dumps({"horse_id": "a", "text": "1970-02-02"}).encode(),
                                      headers={"X-Token": "tok", "Content-Type": "application/json"}, method="POST")
        assert json.load(urllib.request.urlopen(post))["ok"] is True
    finally:
        srv.shutdown()
