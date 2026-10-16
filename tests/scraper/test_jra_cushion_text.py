"""馬場情報 PDF のテキスト並びの読み取りテスト（実 PDF から取ったトークン列を使う）。"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.scraper import jra_cushion_text as jt

LINK = {"year": 2023, "venue_code": "06", "venue_name": "中山", "kai": 2}

# 2023 年 第2回中山（クッション値あり）
TOK_B = """第３日・第４日（2023年3月3日～5日）
金曜日
土曜日
日曜日
9.8
10.2
10.0
場所
金曜日
土曜日
日曜日
芝コース含水率
ゴール前
9.8
9.5
9.6
（パーセント）
４コーナー
12.0
10.8
10.7
ダートコース含水率
ゴール前
4.0
3.2
1.6
（パーセント）
４コーナー
3.4
3.1
2.2
芝コースクッション値""".split("\n")

# 2019 年 第1回東京（含水率のみ。土曜日が無い「金・日・月」）
TOK_C = """第５日・第６日の含水率
2019年2月8日から11日の含水率
場所
金曜日
日曜日
月曜日
芝コース含水率
ゴール前
16.3
17.3
14.6
（パーセント）
４コーナー
14.0
17.5
14.1
ダートコース含水率
ゴール前
11.1
11.5
7.1
（パーセント）
４コーナー
11.4
11.6
6.9""".split("\n")


class ParseTest(unittest.TestCase):
    def test_format_b_with_cushion(self):
        recs = jt.sections_to_records(jt.parse_sections(TOK_B), LINK)
        self.assertEqual([r["date"] for r in recs], ["2023-03-03", "2023-03-04", "2023-03-05"])
        sat = recs[1]
        self.assertEqual(sat["cushion_value"], 10.2)
        self.assertEqual((sat["turf_moisture_goal"], sat["turf_moisture_4corner"], sat["dirt_moisture_goal"], sat["dirt_moisture_4corner"]), (9.5, 10.8, 3.2, 3.1))
        self.assertFalse(recs[0]["is_race_day"])
        self.assertTrue(sat["is_race_day"])

    def test_format_c_weekday_mapping_skips_saturday(self):
        warns: list[str] = []
        recs = jt.sections_to_records(jt.parse_sections(TOK_C), dict(LINK, year=2019, venue_code="05", venue_name="東京", kai=1), warnings=warns)
        self.assertEqual([r["date"] for r in recs], ["2019-02-08", "2019-02-10", "2019-02-11"])      # 金・日・月（土の 9 日は無い）
        self.assertIsNone(recs[0]["cushion_value"])
        self.assertEqual(recs[1]["turf_moisture_goal"], 17.3)
        self.assertEqual(warns, [])

    def test_count_mismatch_warns(self):
        toks = [t for t in TOK_B if t != "10.7"]            # 芝 4 コーナーが 2 個しか無い
        warns: list[str] = []
        jt.sections_to_records(jt.parse_sections(toks), LINK, warnings=warns)
        self.assertTrue(any("turf_moisture_4corner" in w for w in warns))


TOK_AFTER = """第１日・第２日（2020年10月30日～11月1日）
場所
金曜日
土曜日
日曜日
芝コース含水率
ゴール前
13.0
10.7
13.9
（パーセント）
４コーナー
16.8
15.4
15.1
ダートコース含水率
ゴール前
5.6
5.3
5.2
（パーセント）
４コーナー
5.0
5.7
4.2
金曜日
土曜日
日曜日
8.8
8.8
8.3
第３日・第４日（2020年11月6日～8日）
場所
金曜日
土曜日
日曜日
芝コース含水率
ゴール前
12.5
12.2
13.9""".split("\n")


class LayoutVariantTest(unittest.TestCase):
    def test_cushion_table_after_moisture(self):
        warns: list[str] = []
        recs = jt.sections_to_records(jt.parse_sections(TOK_AFTER), dict(LINK, year=2020, venue_code="03", venue_name="福島", kai=3), warnings=warns)
        first = [r for r in recs if r["date"].startswith(("2020-10-30", "2020-10-31", "2020-11-01"))]
        self.assertEqual([r["cushion_value"] for r in first], [8.8, 8.8, 8.3])
        self.assertEqual(first[1]["turf_moisture_goal"], 10.7)
        self.assertEqual(first[2]["date"], "2020-11-01")             # 月をまたぐ範囲

    def test_weather_heading_variant(self):
        toks = ["第３日・第４日の含水率", "2019年5月3日から5日の気象状況", "場所", "金曜日", "土曜日", "日曜日", "芝コース含水率", "ゴール前", "12.7", "11.2", "11.6"]
        secs = jt.parse_sections(toks)
        self.assertEqual(len(secs), 1)
        self.assertEqual(str(secs[0]["start"]), "2019-05-03")


class MergeTest(unittest.TestCase):
    def test_fills_missing_and_keeps_existing(self):
        existing = [{"date": "2023-03-04", "venue_code": "06", "cushion_value": 10.2, "turf_moisture_goal": None, "dirt_moisture_goal": 3.0}]
        incoming = jt.sections_to_records(jt.parse_sections(TOK_B), LINK)
        rows, st = jt.merge_missing(existing, incoming)
        r = [x for x in rows if x["date"] == "2023-03-04"][0]
        self.assertEqual(r["turf_moisture_goal"], 9.5)               # 空だった項目は埋まる
        self.assertEqual(r["dirt_moisture_goal"], 3.0)               # 既存の値は上書きしない
        self.assertEqual(st["added"], 2)                              # 3/3 と 3/5 は追加
        self.assertTrue(any(m["field"] == "dirt_moisture_goal" for m in st["mismatches"]))      # 3.0 と 3.2 の食い違いは記録

    def test_manual_file_loads(self):
        rows = jt.load_manual(Path("data/meta/cushion_manual"))
        sat = [r for r in rows if r["date"] == "2021-12-04" and r["venue_code"] == "07"][0]
        self.assertEqual((sat["cushion_value"], sat["turf_moisture_goal"]), (10.3, 14.5))
        self.assertEqual(len(rows), 9)


if __name__ == "__main__":
    unittest.main()
