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
