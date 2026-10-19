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
    r = s.capture("https://sporthorse-data.com/pedigree/alpha", "Alpha", "30-Jan-1999")                              # 生年違いは警告つきで承認待ちへ
    assert r["ok"] is True and r["staged"] is True and s.approvals()[0]["horse_id"] == "a"
    assert s.capture_none("https://sporthorse-data.com/pedigree/gamma", "Gamma")["ok"] is True
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
                        {"horse_id": "b", "text": "1981-02-02"},          # 生年違い -> 警告つきで承認待ちへ（入力はできる）
                        {"horse_id": "c", "none": True}])["results"]
    assert [r["ok"] for r in res] == [True, True, True] and res[1]["staged"] is True and "1980" in res[1]["warning"]
    assert s.batch(5)["items"] == [] and [a["horse_id"] for a in s.batch(5)["approvals"]] == ["b"]
    assert s.reject("b")["ok"] is True and [i["horse_id"] for i in s.batch(5)["items"]] == ["b"]   # 却下で入力に戻る
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


def test_set_url_is_remembered_even_without_saving(tmp_path):
    s, d = _setup(tmp_path)
    assert s.set_url("a", "https://example.test/x")["ok"] is False                      # 他サイトの URL は受け付けない
    assert s.set_url("zzz", "https://sporthorse-data.com/pedigree/q")["ok"] is False
    assert s.set_url("a", "https://sporthorse-data.com/pedigree/alpha-9")["ok"] is True
    assert s.next()["item"]["url"].endswith("/alpha-9")
    # 記載なし・空欄の行でも、貼られた URL は残る
    s.batch_save([{"horse_id": "b", "none": True, "url": "https://sporthorse-data.com/pedigree/beta-7"}])
    rows = list(csv.reader(open(d / "sporthorse_urls.csv", encoding="utf-8")))
    assert ["b", "https://sporthorse-data.com/pedigree/beta-7"] in rows and ["a", "https://sporthorse-data.com/pedigree/alpha-9"] in rows


def test_userscript_connection_is_reported(tmp_path):
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
        h = {"X-Token": "tok"}
        before = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/batch?n=1", headers=h)))
        assert before["us_seen_ago"] is None                                 # まだ一度もスクリプトからの接続がない
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/token").read()   # ユーザースクリプトがトークンを取りに来る
        after = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/batch?n=1", headers=h)))
        assert after["us_seen_ago"] is not None and after["us_seen_ago"] < 5
    finally:
        srv.shutdown()


def test_year_mismatch_is_staged_for_approval_then_approved(tmp_path):
    s, d = _setup(tmp_path)
    r = s.save("a", "10-Apr-1972")                                            # netkeiba は 1970 -> 警告つきで承認待ちへ
    assert r["ok"] is True and r["staged"] is True and "1970" in r["warning"]
    manual = list(csv.reader(open(d / "birth_manual.csv", encoding="utf-8")))
    assert [m[0] for m in manual[1:]] == []                                   # まだ確定データには入らない
    staged = s.approvals()
    assert len(staged) == 1 and staged[0]["horse_id"] == "a" and staged[0]["birth_year"] == "1972" and staged[0]["netkeiba_year"] == "1970"
    assert [i["horse_id"] for i in s.batch(5)["items"]] == ["b", "c"]          # 入力対象からは外れる
    log = {r[0]: r for r in csv.reader(open(d / "birth_search_log.csv", encoding="utf-8"))}
    assert log["a"][2] == "pending_approval"
    assert s.approve("a")["ok"] is True
    manual = list(csv.reader(open(d / "birth_manual.csv", encoding="utf-8")))
    assert manual[-1][:5] == ["a", "Alpha", "1972", "4", "10"] and "承認済み" in manual[-1][6] and "1970" in manual[-1][6] or "食い違い" in manual[-1][6]
    assert s.approvals() == []
    log = {r[0]: r for r in csv.reader(open(d / "birth_search_log.csv", encoding="utf-8"))}
    assert log["a"][2] == "adopted" and log["a"][3] == "1972"


def test_reject_returns_horse_to_queue_and_undo_removes_staged(tmp_path):
    s, d = _setup(tmp_path)
    s.save("a", "10-Apr-1972")
    assert s.reject("a")["ok"] is True and s.approvals() == []
    assert [i["horse_id"] for i in s.batch(5)["items"]][0] == "a"             # 却下で入力に戻る
    s.save("a", "10-Apr-1972")
    assert s.undo()["ok"] is True and s.approvals() == []                      # 直前を取り消すと承認待ちからも消える
    assert s.batch_save([{"horse_id": "a", "text": "10-Apr-1972", "confirm_year": True}])["results"][0]["ok"] is True   # 確認済みの直接確定（API 用）
    assert s.approvals() == []


def test_approve_all_and_autocommit_includes_staging_file(tmp_path):
    s, d = _setup(tmp_path)
    s.save("a", "10-Apr-1972"); s.save("b", "11-Apr-1982")
    assert len(s.approvals()) == 2 and s.approve()["n"] == 2 and s.approvals() == []


def test_sire_is_shown_in_batch_and_approvals(tmp_path):
    import json
    from src.research.pedigree import manual_entry_server as me
    s0, d = _setup(tmp_path)
    sires = tmp_path / "sires.json"
    sires.write_text(json.dumps({"a": {"sire": "Papa Alpha", "sire_id": "p"}}), encoding="utf-8")
    s = me.Store(tmp_path / "anc.csv", tmp_path / "zone.csv", None, d, "https://example.test/p", sires=sires)
    b = s.batch(2)
    assert b["items"][0]["sire"] == "Papa Alpha" and b["items"][1]["sire"] == ""
    s.save("a", "10-Apr-1972")                                                  # 生年違い -> 承認待ち
    assert s.batch(2)["approvals"][0]["sire"] == "Papa Alpha"
