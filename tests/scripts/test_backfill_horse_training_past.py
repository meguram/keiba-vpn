import json

import pandas as pd

from src.scripts.scraping import backfill_horse_training_past as B


def test_read_horses_parquet_dedup_keeps_order(tmp_path):
    f = tmp_path / "h.parquet"
    pd.DataFrame({"horse_id": ["b", "a", "b", ""]}).to_parquet(f)
    assert B.read_horses(f) == ["b", "a"]


def test_read_horses_text(tmp_path):
    f = tmp_path / "h.txt"
    f.write_text("x\ny  z\n", encoding="utf-8")
    assert B.read_horses(f) == ["x", "y", "z"]


def test_pages_of():
    assert B.pages_of(None) == 1
    assert B.pages_of({"entries": []}) == 1
    assert B.pages_of({"entries": [{}] * 252, "pages_fetched": 6}) == 6


class _Client:
    def __init__(self, fetch_fail_on=None):
        self.n, self.fail_on = 0, fetch_fail_on

    def fetch(self, url, **k):
        self.n += 1
        if self.fail_on is not None and self.n % 3 == self.fail_on:
            raise RuntimeError("HTTP 500")
        return "html"


class _Runner:
    """scrape_horse_training と同じ順に client.fetch → _persist_horse_training を呼ぶ最小の代役（3 ページで 1 頭）。"""

    def __init__(self, fail=(), page_fail_on=None):
        self.fail, self.calls, self.saved = set(fail), [], {}
        self.client = _Client(page_fail_on)

    def _persist_horse_training(self, hid, result):
        self.saved[hid] = result

    def scrape_horse_training(self, hid, force=False):
        assert force is True
        self.calls.append(hid)
        if hid in self.fail:
            raise RuntimeError("boom")
        pages = 0
        for _ in range(3):
            try:
                self.client.fetch("u")
            except Exception:
                if pages == 0:
                    raise
                break
            pages += 1
        res = {"entries": [{}] * (25 if hid != "e" else 0), "pages_fetched": pages if hid != "e" else 1}
        self._persist_horse_training(hid, res)
        return res


def _run(monkeypatch, tmp_path, ids, runner, extra=()):
    monkeypatch.setattr(B, "PROGRESS", tmp_path / "p.json")
    monkeypatch.setattr(B, "wait_ok", lambda *_: True)
    monkeypatch.setattr(B.time, "sleep", lambda *_: None)
    import src.scraper.backfill_guard as bg
    import src.scraper.run as R
    import src.scraper.scrape_access_pause as ap

    monkeypatch.setattr(R, "ScraperRunner", lambda: runner)
    monkeypatch.setattr(bg, "halted", lambda: None)
    monkeypatch.setattr(ap, "read_access_pause", lambda: {"active": False})
    f = tmp_path / "h.txt"
    f.write_text(" ".join(ids), encoding="utf-8")
    rc = B.main(["--horses-file", str(f), *extra])
    return rc, json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))


def test_main_counts_resumes_and_limit(monkeypatch, tmp_path):
    r = _Runner()
    rc, p = _run(monkeypatch, tmp_path, ["a", "e", "c"], r, ["--limit", "2"])
    assert rc == 0 and r.calls == ["a", "e"] and p["counts"] == {"saved": 1, "empty": 1, "error": 0} and p["requests"] == 3 + 1
    r2 = _Runner()
    rc, p = _run(monkeypatch, tmp_path, ["a", "e", "c"], r2)
    assert r2.calls == ["c"] and len(p["done"]) == 3 and p["total"] == 3


def test_main_aborts_after_8_consecutive_errors(monkeypatch, tmp_path):
    ids = [f"h{i}" for i in range(12)]
    r = _Runner(fail=ids)
    rc, p = _run(monkeypatch, tmp_path, ids, r)
    assert rc == 3 and len(r.calls) == 8 and "連続" in p["aborted"]


def test_partial_fetch_is_not_saved(monkeypatch, tmp_path):
    # 3 ページ中、各馬の 3 ページ目（fetch の通し番号 % 3 == 0）が失敗 → 2 ページだけの記録は保存せず error
    r = _Runner(page_fail_on=0)
    rc, p = _run(monkeypatch, tmp_path, ["a", "b"], r)
    assert r.saved == {} and p["counts"]["error"] == 2 and p["done"] == {"a": "error", "b": "error"}


def test_stops_when_access_pause_is_active(monkeypatch, tmp_path):
    r = _Runner()
    import src.scraper.scrape_access_pause as ap

    f = tmp_path / "h.txt"
    f.write_text("a b", encoding="utf-8")
    monkeypatch.setattr(B, "PROGRESS", tmp_path / "p.json")
    monkeypatch.setattr(B, "wait_ok", lambda *_: True)
    import src.scraper.run as R

    monkeypatch.setattr(R, "ScraperRunner", lambda: r)
    import src.scraper.backfill_guard as bg

    monkeypatch.setattr(bg, "halted", lambda: None)
    monkeypatch.setattr(ap, "read_access_pause", lambda: {"active": True, "reason": "テスト"})
    assert B.main(["--horses-file", str(f)]) == 3 and r.calls == []


class _Guard:
    tripped = None


class _Storage:
    def __init__(self, ret):
        self.ret, self.n = ret, 0

    def save(self, *a, **k):
        self.n += 1
        return self.ret


class _G:
    """guard_persist の検査用: fetch を n_ok 回成功させたあと、保存（結果の pages_fetched を指定）を呼ぶ。"""

    def __init__(self, save_ret=True):
        self.saved = []
        self.storage = _Storage(save_ret)
        self.client = type("C", (), {"fetch": staticmethod(lambda *a, **k: "html")})()

    def _persist_horse_training(self, hid, result):
        self.storage.save("horse_training", hid, result)
        self.saved.append(hid)


def test_guard_rejects_result_with_fewer_pages_fetched_than_claimed():
    import pytest

    r = _G()
    B.guard_persist(r, _Guard())
    r.client.fetch("u"), r.client.fetch("u")          # 実際に取れたのは 2 ページ
    with pytest.raises(B.PartialFetch):
        r._persist_horse_training("h", {"entries": [{}], "pages_fetched": 3})   # 3 ページ分を名乗る（空ページで打ち切った記録）
    assert r.saved == []
    r.client.fetch("u"), r.client.fetch("u")
    r._persist_horse_training("h", {"entries": [{}], "pages_fetched": 2})
    assert r.saved == ["h"]


def test_guard_raises_when_gcs_save_returns_false():
    import pytest

    r = _G(save_ret=False)
    B.guard_persist(r, _Guard())
    r.client.fetch("u")
    with pytest.raises(RuntimeError):
        r._persist_horse_training("h", {"entries": [{}], "pages_fetched": 1})


def test_main_aborts_after_consecutive_empty_horses(monkeypatch, tmp_path):
    ids = [f"e{i}" for i in range(30)]
    monkeypatch.setattr(B, "MAX_EMPTY_STREAK", 5)
    r = _Runner()
    r.calls = []
    orig = r.scrape_horse_training
    r.scrape_horse_training = lambda hid, force=False: orig("e", force) if hid.startswith("e") else orig(hid, force)
    rc, p = _run(monkeypatch, tmp_path, ids, r)
    assert rc == 3 and len(r.calls) == 5 and "調教なし" in p["aborted"]


def test_corrupt_progress_stops_instead_of_restarting(monkeypatch, tmp_path):
    import pytest

    f = tmp_path / "p.json"
    f.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(B, "PROGRESS", f)
    with pytest.raises(ValueError):
        B.load_progress()


def test_counts_are_recomputed_from_done_on_retry(monkeypatch, tmp_path):
    ids = ["a", "b"]
    rc, p = _run(monkeypatch, tmp_path, ids, _Runner(fail=["a"]))
    assert p["counts"]["error"] == 1
    rc, p = _run(monkeypatch, tmp_path, ids, _Runner(), ["--retry-failed"])
    assert p["counts"] == {"saved": 2, "empty": 0, "error": 0}


def test_cutoff_stops_paging_and_records_truncation():
    pages = [["2016-05-01", "2016-04-01"], ["2015-03-01", "2014-12-20"], ["2013-01-01"], ["2012-01-01"]]

    class _P:
        def __init__(self):
            self.i = 0

        def parse(self, html, **k):
            self.i += 1
            return {"entries": [{"date": d} for d in pages[self.i - 1]]}

    r = _G()
    r.training_parser = _P()
    B.guard_persist(r, _Guard(), cutoff="2015-01-01")
    got = 0
    for _ in range(4):   # scrape_horse_training と同じ: 取得 → 解析、2 ページ目以降の例外で止まる
        try:
            r.client.fetch("u")
        except B.StopPaging:
            break
        r.training_parser.parse("h")
        got += 1
    assert got == 2   # 2 ページ目に 2014-12-20 が入っていたので 3 ページ目は取らない
    res = {"entries": [{}], "pages_fetched": 4}
    r._persist_horse_training("h", res)
    assert r.saved == ["h"] and res["pages_fetched"] == 2 and res["truncated_before"] == "2015-01-01"


def test_cutoff_off_by_default():
    r = _G()
    B.guard_persist(r, _Guard())
    r.client.fetch("u")
    r.client.fetch("u")


def test_redo_truncated_refetches_only_truncated_saved_horses_without_cutoff(monkeypatch, tmp_path):
    r = _Runner()
    rc, p = _run(monkeypatch, tmp_path, ["a", "b", "e"], r)
    assert rc == 0 and p["done"] == {"a": "saved", "b": "saved", "e": "empty"}
    seen = {}
    real_guard = B.guard_persist
    monkeypatch.setattr(B, "guard_persist", lambda runner, guard, cutoff="", reject_empty=False: (seen.update(cutoff=cutoff, reject_empty=reject_empty), real_guard(runner, guard, cutoff, reject_empty)))
    monkeypatch.setattr(B, "is_truncated", lambda h: h in {"b", "e"})   # e は調教なしなので対象外
    r2 = _Runner(fail={"b"})
    rc, p = _run(monkeypatch, tmp_path, ["a", "b", "e"], r2, ["--redo-truncated"])
    assert rc == 0 and r2.calls == ["b"] and seen == {"cutoff": "", "reject_empty": True}
    assert p["redo"] == {"b": "error"} and p["done"]["b"] == "saved"   # 失敗しても打ち切った記録のまま saved
    r3 = _Runner()
    rc, p = _run(monkeypatch, tmp_path, ["a", "b", "e"], r3, ["--redo-truncated"])
    assert r3.calls == []   # 済み（error も）は再開で取らない
    rc, p = _run(monkeypatch, tmp_path, ["a", "b", "e"], r3, ["--redo-truncated", "--retry-failed"])
    assert r3.calls == ["b"] and p["redo"] == {"b": "saved"}
    assert p["finished_at"] and p["redo_meta"]["finished_at"]   # 本取得の完了の記録は残し、取り直しの経過は別に残す


def test_redo_truncated_refuses_before_main_run_finishes(monkeypatch, tmp_path):
    r = _Runner()
    rc, p = _run(monkeypatch, tmp_path, ["a", "b"], r, ["--limit", "1"])
    monkeypatch.setattr(B, "is_truncated", lambda h: True)
    r2 = _Runner()
    rc, p = _run(monkeypatch, tmp_path, ["a", "b"], r2, ["--redo-truncated"])
    assert rc == 5 and r2.calls == [] and "redo" not in p   # 未取得（b）が残っている間は始めない


def test_guard_reject_empty_keeps_existing_record():
    import pytest

    r = _G()
    B.guard_persist(r, _Guard(), "", reject_empty=True)
    r.client.fetch("u")
    with pytest.raises(B.PartialFetch):
        r._persist_horse_training("h", {"entries": [], "pages_fetched": 0})   # ログイン切れ等の 0 件で、打ち切った記録を消さない
    assert r.saved == []


def test_is_truncated_reads_local_record(monkeypatch, tmp_path):
    from src.scraper.horse_training_local import horse_training_json_path

    monkeypatch.setattr(B, "ROOT", tmp_path)
    for hid, rec in (("2012110059", {"entries": [], "truncated_before": "2015-01-01"}), ("2012110060", {"entries": []})):
        f = horse_training_json_path(tmp_path, hid)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec), encoding="utf-8")
    assert B.is_truncated("2012110059") and not B.is_truncated("2012110060") and not B.is_truncated("2012110061")


def test_cutoff_stop_does_not_leave_incomplete_mark():
    r = _G()
    B.guard_persist(r, _Guard(), "2015-01-01")
    r.client.fetch("u")
    r._guard_state["cut"] = True
    import pytest

    with pytest.raises(B.StopPaging):
        r.client.fetch("u")
    res = {"entries": [{}], "pages_fetched": 3, "incomplete": True}
    r._persist_horse_training("h", res)
    assert r.saved == ["h"] and "incomplete" not in res and res["truncated_before"] == "2015-01-01" and res["pages_fetched"] == 1


def test_wait_ok_shows_waiting_then_running(monkeypatch):
    from src.scripts.scraping import backfill_horse_training_recent as R

    states, busy = [], iter([True, True, False])
    monkeypatch.setattr(R, "busy", lambda: next(busy))
    monkeypatch.setattr(R, "load1", lambda: 0.0)
    monkeypatch.setattr(R, "time", __import__("types").SimpleNamespace(time=lambda: 0.0, sleep=lambda *_: None))
    monkeypatch.setattr(R, "busy_names", lambda: ["src.scraper.export_tables"])
    monkeypatch.setattr(R, "_set_fetch_state", lambda s, w=None: states.append(s))
    assert R.wait_ok(3600) is True and states == ["waiting", "running"]
    states.clear()
    monkeypatch.setattr(R, "busy", lambda: False)
    assert R.wait_ok(3600) is True and states == []                       # 待たなければ状態は変えない
    import types

    monkeypatch.setattr(R, "busy", lambda: True)
    now = {"t": 0.0}

    def sleep(sec):
        now["t"] += sec

    monkeypatch.setattr(R, "time", types.SimpleNamespace(time=lambda: now["t"], sleep=sleep))   # このモジュールの時計だけ差し替える
    assert R.wait_ok(60) is False and states == ["waiting"]                # 待ちきれず中断: 実行中には戻さない


def test_wait_ok_names_what_it_waits_for(monkeypatch):
    """待っている相手の名前を出し、相手が変わったときだけ書き直す。待つかどうかの判定は変えない。"""
    import types

    from src.scripts.scraping import backfill_horse_training_recent as R

    calls, busy = [], iter([True, True, True, False, False])
    names = iter([["src.scraper.export_tables"], ["src.scraper.export_tables"], ["src.scraper.run"], []])
    monkeypatch.setattr(R, "busy", lambda: next(busy))
    monkeypatch.setattr(R, "busy_names", lambda: next(names))
    loads = iter([20.0, 20.0, 0.0])   # busy が True の間は負荷を見ない
    monkeypatch.setattr(R, "load1", lambda: next(loads))
    monkeypatch.setattr(R, "time", types.SimpleNamespace(time=lambda: 0.0, sleep=lambda *_: None))
    monkeypatch.setattr(R, "_set_fetch_state", lambda s, w=None: calls.append((s, w)))
    assert R.wait_ok(3600) is True
    assert calls[0] == ("waiting", "他の取得の終了待ち: src.scraper.export_tables")
    assert calls[1] == ("waiting", "他の取得の終了待ち: src.scraper.run")
    assert calls[2] == ("waiting", "負荷の低下待ち（1 分平均 18 未満で再開）")
    assert calls[3] == ("running", None) and len(calls) == 4


def test_busy_names_extracts_module_names(monkeypatch):
    from src.scripts.scraping import backfill_horse_training_recent as R

    ps = ("ARGS\n/opt/conda/bin/python3 -m src.scraper.export_tables --years 2026\n"
          "python3 -m src.scraper.export_tables --x\n"
          "/opt/conda/bin/python3 -m src.scripts.scraping.backfill_horse_training_past --horses-file a\n"
          "python -m src.scraper.run horse-training 1\n"
          "python3 -m src.scraper.raceday_runner --date 2026-11-14\n"
          "python3 -m src.scripts.maintenance.repair_odds\n"
          "python3 -m src.scripts.scraping.backfill_horse_race_info --x\n"
          "grep -P python3 -m src.scraper.auto_scrape_queue\n")
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: __import__("types").SimpleNamespace(stdout=ps))
    # 自分（過去分の取り直し）と grep の行は入れず、重複もまとめ、名前は途中で切らない
    assert R.busy_names() == ["src.scraper.export_tables", "src.scraper.run", "src.scraper.raceday_runner",
                              "src.scripts.maintenance.repair_odds", "src.scripts.scraping.backfill_horse_race_info"]


def test_waiting_for_falls_back_when_nothing_is_seen(monkeypatch):
    from src.scripts.scraping import backfill_horse_training_recent as R

    monkeypatch.setattr(R, "busy_names", lambda: [])
    monkeypatch.setattr(R, "load1", lambda: 1.0)
    assert R._waiting_for() == "他の取得の終了・負荷の低下待ち"   # 判定の直後に相手が終わった: 負荷待ちとは書かない


def test_set_fetch_state_swallows_import_errors(monkeypatch):
    import sys

    from src.scripts.scraping import backfill_horse_training_recent as R

    monkeypatch.setitem(sys.modules, "src.scraper.fetch_registry", None)    # import が失敗する
    R._set_fetch_state("waiting", "x")                                      # 例外を出さない
