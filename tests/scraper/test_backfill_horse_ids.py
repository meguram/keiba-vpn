"""backfill --phase horse の馬 ID の集め方（手元の flat を先に使い、GCS の全件の一覧と読み出しをしない）のテスト。

GCS は偽の storage（呼び出しを記録する）にする。以前の集め方（年なしの list_keys ＋ 1 件ずつ load）を同じ偽データで動かし、
未処理の馬の判定（pending）が変わらないことを確かめる。docs/operations/cost_ledger.md 3 章 #7。
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

from src.scraper import backfill as bf

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 11, 14, 6, 0, tzinfo=JST)
OLD_TS = NOW.timestamp() - 30 * 86400    # GCS の更新が 30 日前（flat に入り済み）


def _race(rid, horses):
    return {"race_id": rid, "entries": [{"horse_id": h} for h in horses]}


class _FakeGcs:
    """GCS の代わり。docs[category][race_id] = JSON。呼び出しを記録する。"""

    def __init__(self, docs, updated=None):
        self.docs = docs
        self.updated = updated or {}
        self.list_keys_calls = []
        self.batch_calls = []
        self.loads = []
        self.cache = {}

    def list_keys(self, category, year=None):
        self.list_keys_calls.append((category, year))
        return sorted(k for k in self.docs.get(category, {}) if year is None or k.startswith(year))

    def batch_list_blobs(self, category, year):
        self.batch_calls.append((category, year))
        return {k: self.updated.get(k, OLD_TS) for k in self.docs.get(category, {}) if k.startswith(year)}

    def load(self, category, key):
        self.loads.append((category, key))
        return self.docs.get(category, {}).get(key)

    def _read_local_cache(self, category, key, allow_stale=False):
        return self.cache.get((category, key))


def _old_collect(storage):
    """直す前の集め方（backfill.py の旧 :347-355）。比べる基準。"""
    ids = set()
    for category in ["race_result", "race_shutuba"]:
        for key in storage.list_keys(category):
            data = storage.load(category, key)
            if data:
                for entry in data.get("entries", []):
                    hid = entry.get("horse_id", "")
                    if hid:
                        ids.add(hid)
    return ids


def _write_flat(tables, year, category, races):
    rows = [{"race_id": rid, "horse_id": h} for rid, horses in races.items() for h in horses]
    d = tables / year
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{category}_flat.parquet"
    pd.DataFrame(rows, columns=["race_id", "horse_id"]).to_parquet(p, index=False)


# GCS の中身: 2024・2025・2026 年。2026 は flat の書き出しより後に増えたレースがある
RESULT = {
    "202405010101": _race("202405010101", ["2021100001", "2021100002"]),
    "202405010102": _race("202405010102", ["2021100003", ""]),          # horse_id が空の行は数えない
    "202505010101": _race("202505010101", ["2022100001", "2021100001"]),
    "202605010101": _race("202605010101", ["2023100001", "000a01234b"]),  # 外国馬の英数字 ID も落とさない
    "202605010102": _race("202605010102", ["2023100002"]),               # flat に無い当年のレース
}
SHUTUBA = {
    "202405010101": _race("202405010101", ["2021100001", "2021100002"]),
    "202505010101": _race("202505010101", ["2022100001", "2022100002"]),
    "202605010101": _race("202605010101", ["2023100001", "000a01234b"]),
    "202605010103": _race("202605010103", ["2023100003"]),               # 出馬表だけある当年の先のレース
}


class CollectHorseIdsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tables = Path(self._tmp.name) / "tables"
        for cat, docs in (("race_result", RESULT), ("race_shutuba", SHUTUBA)):
            for y in ("2024", "2025"):
                _write_flat(self.tables, y, cat, {k: [e["horse_id"] for e in v["entries"]] for k, v in docs.items() if k.startswith(y)})
        # 2026 の flat は書き出し時点のもの（202605010101 だけ）
        _write_flat(self.tables, "2026", "race_result", {"202605010101": ["2023100001", "000a01234b"]})
        _write_flat(self.tables, "2026", "race_shutuba", {"202605010101": ["2023100001", "000a01234b"]})

    def tearDown(self):
        self._tmp.cleanup()

    def _gcs(self, updated=None):
        return _FakeGcs({"race_result": RESULT, "race_shutuba": SHUTUBA}, updated)

    def _collect(self, gcs, allow_gcs):
        return bf._collect_phase_horse_ids(gcs, tables_dir=self.tables, now=NOW, allow_gcs=allow_gcs)

    def test_same_ids_as_before_and_reads_only_new_races_of_current_year(self):
        for allow in (True, False):
            with self.subTest(allow_gcs=allow):
                gcs = self._gcs()
                ids, info = self._collect(gcs, allow)
                self.assertEqual(ids, _old_collect(self._gcs()))
                self.assertIn("000a01234b", ids)
                self.assertNotIn("", ids)
                # 年なしの全件の一覧はしない。一覧は当年の 2 カテゴリだけ
                self.assertEqual(gcs.list_keys_calls, [])
                years = {y for _, y in gcs.batch_calls}
                # ホストは当年だけ一覧。GCP の中は全年を年つきで一覧する（読むのは flat に無いレースだけ）
                self.assertEqual(years, {"2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"} if allow else {"2026"})
                # 読むのは flat に無い当年のレースだけ
                self.assertEqual(sorted(gcs.loads), [("race_result", "202605010102"), ("race_shutuba", "202605010103")])
                self.assertEqual(info["gcs_loads"], 2)

    def test_reloads_current_year_race_updated_recently(self):
        # 202605010101 の出馬表が flat に載った後に GCS で入れ替わった（export は既存の行を書き直さない）。
        # 更新が 8 日以内なら flat にあっても読み直す。flat の mtime には頼らない（毎朝書き直されて常に新しい）
        shutuba = dict(SHUTUBA)
        shutuba["202605010101"] = _race("202605010101", ["2023100001", "2023100009"])
        for age_days, expect in ((5, True), (10, False)):
            with self.subTest(age_days=age_days):
                gcs = _FakeGcs({"race_result": RESULT, "race_shutuba": shutuba},
                               {"202605010101": NOW.timestamp() - age_days * 86400})
                ids, info = bf._collect_phase_horse_ids(gcs, tables_dir=self.tables, now=NOW, allow_gcs=False)
                self.assertEqual("2023100009" in ids, expect)
                self.assertEqual("2023100009" in info["gcs_only_ids"], expect)
                self.assertEqual(("race_shutuba", "202605010101") in gcs.loads, expect)

    def test_list_years_on_gcp(self):
        # GCP の中は list_years の年（2016 より前・年でない値・例外）も扱う
        older = dict(RESULT)
        older["201505010101"] = _race("201505010101", ["2012100001"])
        gcs = _FakeGcs({"race_result": older, "race_shutuba": SHUTUBA})

        def list_years(cat):
            if cat == "race_shutuba":
                raise RuntimeError("一覧の失敗")
            return ["2015", "2024", "unknown"]
        gcs.list_years = list_years
        ids, _ = bf._collect_phase_horse_ids(gcs, tables_dir=self.tables, now=NOW, allow_gcs=True)
        self.assertEqual(ids, _old_collect(_FakeGcs({"race_result": older, "race_shutuba": SHUTUBA})))
        self.assertIn(("race_result", "2015"), gcs.batch_calls)
        self.assertFalse(any(y == "unknown" for _, y in gcs.batch_calls))

    def test_january_also_reads_previous_year(self):
        jan = datetime(2027, 1, 5, 6, 0, tzinfo=JST)
        gcs = self._gcs()
        bf._collect_phase_horse_ids(gcs, tables_dir=self.tables, now=jan, allow_gcs=False)
        self.assertEqual({y for _, y in gcs.batch_calls}, {"2026", "2027"})
        # 前年も読むのは flat に無いレースだけ
        self.assertEqual(sorted(gcs.loads), [("race_result", "202605010102"), ("race_shutuba", "202605010103")])

    def test_past_year_flat_behind_gcs(self):
        # 過去年（2024）に新しく取ったレースが GCS にだけある（flat が追いついていない）
        result = dict(RESULT)
        result["202405010199"] = _race("202405010199", ["2021100099"])
        gcs = _FakeGcs({"race_result": result, "race_shutuba": SHUTUBA})
        ids, _ = bf._collect_phase_horse_ids(gcs, tables_dir=self.tables, now=NOW, allow_gcs=True)
        self.assertIn("2021100099", ids)           # GCP の中は年つきの一覧で拾う
        gcs = _FakeGcs({"race_result": result, "race_shutuba": SHUTUBA})
        ids, _ = bf._collect_phase_horse_ids(gcs, tables_dir=self.tables, now=NOW, allow_gcs=False)
        self.assertNotIn("2021100099", ids)        # ホストは flat が追いつくまで待つ（GCS の過去年は読まない）
        self.assertFalse(any(y == "2024" for _, y in gcs.batch_calls))

    def test_missing_past_year_flat(self):
        for cat in ("race_result", "race_shutuba"):
            (self.tables / "2025" / f"{cat}_flat.parquet").unlink()
        only_2025 = {"2022100001", "2022100002"}
        # GCP の中（転送料なし）: その年だけ年つきの一覧で補い、以前と同じになる
        gcs = self._gcs()
        ids, info = self._collect(gcs, True)
        self.assertEqual(ids, _old_collect(self._gcs()))
        self.assertIn(("race_result", "2025"), gcs.batch_calls)
        self.assertEqual(gcs.list_keys_calls, [])
        # ホスト（転送料あり）: その年は GCS で補わず、警告して飛ばす。その年だけの馬は対象外になる
        gcs = self._gcs()
        ids, info = self._collect(gcs, False)
        self.assertEqual(ids, _old_collect(self._gcs()) - only_2025)
        self.assertEqual(info["skipped_years"], ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2025"])
        self.assertFalse(any(y == "2025" for _, y in gcs.batch_calls))
        self.assertFalse(any(k.startswith("2025") for _, k in gcs.loads))
        self.assertEqual(gcs.list_keys_calls, [])

    def test_missing_current_year_flat_reads_that_year_only(self):
        for cat in ("race_result", "race_shutuba"):
            (self.tables / "2026" / f"{cat}_flat.parquet").unlink()
        gcs = self._gcs()
        ids, _ = self._collect(gcs, False)
        self.assertEqual(ids, _old_collect(self._gcs()))
        self.assertTrue(all(k.startswith("2026") for _, k in gcs.loads))
        self.assertEqual(gcs.list_keys_calls, [])

    def test_no_local_flat(self):
        empty = Path(self._tmp.name) / "none"
        # GCP の中: 以前と同じ全件の読み方に戻す
        gcs = self._gcs()
        ids, info = bf._collect_phase_horse_ids(gcs, tables_dir=empty, now=NOW, allow_gcs=True)
        self.assertEqual(ids, _old_collect(self._gcs()))
        self.assertEqual(info["source"], "gcs_full")
        # ホスト: 全件は読まず、当年だけ
        gcs = self._gcs()
        ids, info = bf._collect_phase_horse_ids(gcs, tables_dir=empty, now=NOW, allow_gcs=False)
        self.assertEqual(gcs.list_keys_calls, [])
        self.assertTrue(all(k.startswith("2026") for _, k in gcs.loads))
        self.assertEqual(ids, {"2023100001", "000a01234b", "2023100002", "2023100003"})

    def test_default_follows_egress_billed(self):
        with mock.patch("src.config.gcp_guard.egress_billed", return_value=True):
            for cat in ("race_result", "race_shutuba"):
                (self.tables / "2025" / f"{cat}_flat.parquet").unlink()
            gcs = self._gcs()
            _, info = bf._collect_phase_horse_ids(gcs, tables_dir=self.tables, now=NOW)
            self.assertIn("2025", info["skipped_years"])
            self.assertEqual({y for _, y in gcs.batch_calls}, {"2026"})
            self.assertEqual(gcs.list_keys_calls, [])


class _Progress:
    def __init__(self, done=()):
        self._data = {"done_horses": list(done)}
        self.marked = []
        self.saves = 0

    def save(self):
        self.saves += 1

    def mark_horses_done(self, ids):
        self.marked.extend(ids)


class _Runner:
    def __init__(self, storage):
        self.storage = storage
        self.scraped = []

    def scrape_horse(self, hid, skip_existing=False, with_history=True):
        self.scraped.append(hid)


class PhaseHorseSireCheckTest(unittest.TestCase):
    def test_local_copy_first_then_gcs(self):
        gcs = _FakeGcs({"horse_result": {
            "2021100002": {"sire": "B"},
            "2021100003": {"sire": ""},
        }})
        gcs.cache[("horse_result", "2021100001")] = {"sire": "A"}   # 手元に写しがある
        runner = _Runner(gcs)
        prog = _Progress()
        ids = {"2021100001", "2021100002", "2021100003", "2021100004"}
        info = {"source": "local_flat", "gcs_lists": [], "gcs_loads": 0, "skipped_years": [], "gcs_only_ids": set(), "flat_ids": ids}
        with mock.patch.object(bf, "_collect_phase_horse_ids", return_value=(ids, info)):
            stats = bf._run_phase_horse(runner, prog, max_horses=10)
        # 手元に sire があれば GCS を読まない。それ以外は GCS で確かめ、sire が無い・記録が無い馬だけ取得する
        self.assertEqual(sorted(k for _, k in gcs.loads), ["2021100002", "2021100003", "2021100004"])
        self.assertEqual(sorted(runner.scraped), ["2021100003", "2021100004"])
        self.assertEqual(stats["horses_skipped"], 2)
        self.assertEqual(sorted(prog.marked), sorted(ids))


    def test_gcs_only_ids_are_kept_until_flat_catches_up(self):
        gcs = _FakeGcs({"horse_result": {}})
        prog = _Progress(done=["2023100001"])
        prog._data[bf._GCS_SEEN_KEY] = ["2023100005", "2023100001"]   # 前の日に GCS で見つけた馬（1 頭はもう flat にある）
        flat = {"2023100001"}
        info = {"source": "local_flat", "gcs_lists": [], "gcs_loads": 0, "skipped_years": [],
                "gcs_only_ids": {"2023100009"}, "flat_ids": flat}
        runner = _Runner(gcs)
        with mock.patch.object(bf, "_collect_phase_horse_ids", return_value=(flat | {"2023100009"}, info)):
            bf._run_phase_horse(runner, prog, max_horses=10)
        # 今日読み直さなかった 2023100005 も対象に残る。flat に入った・処理済みの 2023100001 は記録から外す
        self.assertEqual(sorted(runner.scraped), ["2023100005", "2023100009"])
        self.assertEqual(prog._data[bf._GCS_SEEN_KEY], ["2023100005", "2023100009"])
        self.assertGreaterEqual(prog.saves, 1)


if __name__ == "__main__":
    unittest.main()
