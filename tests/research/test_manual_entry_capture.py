import csv

from tests.research.test_manual_entry_server import _setup


def test_capture_matches_current_horse_and_saves(tmp_path):
    s, d = _setup(tmp_path)
    assert s.capture("https://sporthorse-data.com/pedigree/unknown-1", "Unknown Pedigree", "30-Jan-1980")["ok"] is False   # 対象外の馬
    r = s.capture("https://sporthorse-data.com/pedigree/alpha-1", "Alpha | SporthorseData", "30-Jan-1970")
    assert r["ok"] is True
    rows = list(csv.reader(open(d / "birth_manual.csv", encoding="utf-8")))
    assert rows[-1][:5] == ["a", "Alpha", "1970", "1", "30"] and rows[-1][5].endswith("/alpha-1")


def test_capture_rejects_wrong_year_and_none(tmp_path):
    s, d = _setup(tmp_path)
    assert s.capture("https://sporthorse-data.com/pedigree/alpha", "Alpha", "30-Jan-1999")["ok"] is False             # 生年違い
    assert s.capture_none("https://sporthorse-data.com/pedigree/alpha", "Alpha")["ok"] is True
    assert [r["horse_id"] for r in s.pending()][:1] == ["b"]
    # スラッグが違ってもタイトルの馬名で一致すれば受け付ける
    assert s.capture("https://sporthorse-data.com/pedigree/bt-99", "Beta - Thoroughbred", "03-Mar-1980")["ok"] is True


def test_url_override_is_remembered_and_committed_list(tmp_path):
    s, d = _setup(tmp_path)
    r = s.save("a", "30-Jan-1970", url="https://sporthorse-data.com/pedigree/alpha-0")
    assert r["ok"] is True
    rows = list(csv.reader(open(d / "sporthorse_urls.csv", encoding="utf-8")))
    assert rows[1] == ["a", "https://sporthorse-data.com/pedigree/alpha-0"]


def test_saved_override_is_used_for_pending_horse_after_restart(tmp_path):
    from src.research.pedigree import manual_entry_server as me
    s, d = _setup(tmp_path)
    (d / "sporthorse_urls.csv").write_text("horse_id,url\nb,https://sporthorse-data.com/pedigree/beta-0\n", encoding="utf-8")
    s2 = me.Store(tmp_path / "anc.csv", tmp_path / "zone.csv", None, d, "https://example.test/p")
    assert s2.save("a", "30-Jan-1970")["ok"] is True
    item = s2.next()["item"]
    assert item["horse_id"] == "b" and item["url"].endswith("/beta-0") and item["candidates"][0].endswith("/beta-0")


def test_detect_is_remembered_for_current_horse_only_and_shown_in_next(tmp_path):
    s, d = _setup(tmp_path)
    assert s.detect("https://sporthorse-data.com/pedigree/unknown", "Unknown", "30-Jan-1980")["ok"] is False        # 対象外の馬
    assert s.detect("https://sporthorse-data.com/pedigree/alpha-1", "Alpha", "30-Jan-1970")["ok"] is True
    n = s.next()
    assert n["item"]["horse_id"] == "a" and n["detected"]["text"] == "30-Jan-1970" and n["detected"]["url"].endswith("/alpha-1")
    assert s.save("a", n["detected"]["text"], url=n["detected"]["url"])["ok"] is True                              # 画面のボタンの流れ
    assert s.next()["item"]["horse_id"] == "b" and s.next()["detected"] is None
    assert s.detect("https://sporthorse-data.com/pedigree/beta", "Beta", None, "1980")["ok"] is True
    assert s.next()["detected"]["year_only"] == "1980"


def test_batch_view_save_and_window_detection(tmp_path):
    s, d = _setup(tmp_path)
    b = s.batch(3)
    assert [i["horse_id"] for i in b["items"]] == ["a", "b", "c"] and b["remaining"] == 3
    # 先頭ではない馬（b）のページから検出しても、その馬に結び付く
    assert s.detect("https://sporthorse-data.com/pedigree/beta-0", "Beta", "12-Feb-1980")["ok"] is True
    assert s.batch(3)["detected"]["b"]["text"] == "12-Feb-1980"
    res = s.batch_save([{"horse_id": "a", "text": "30-Jan-1970", "url": "https://sporthorse-data.com/pedigree/alpha-1"},
                        {"horse_id": "b", "text": "1981-02-02"},          # 生年違い -> 失敗して未処理のまま
                        {"horse_id": "c", "none": True}])["results"]
    assert [r["ok"] for r in res] == [True, False, True] and "生年" in res[1]["error"]
    assert [i["horse_id"] for i in s.batch(5)["items"]] == ["b"]
    assert s.batch_save([{"horse_id": "b"}])["results"][0]["ok"] is True          # 空欄は「あとで」
    assert s.batch(5)["remaining"] == 1


def test_http_batch_endpoints(tmp_path):
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from src.research.pedigree import manual_entry_server as me
    s, d = _setup(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), me.make_handler(s, "tok"))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        h = {"X-Token": "tok", "Content-Type": "application/json"}
        got = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/batch?n=2", headers=h)))
        assert [i["horse_id"] for i in got["items"]] == ["a", "b"]
        post = urllib.request.Request(f"http://127.0.0.1:{port}/api/batch_save", data=json.dumps({"rows": [{"horse_id": "a", "text": "30-Jan-1970"}]}).encode(), headers=h, method="POST")
        assert json.load(urllib.request.urlopen(post))["results"][0]["ok"] is True
    finally:
        srv.shutdown()
