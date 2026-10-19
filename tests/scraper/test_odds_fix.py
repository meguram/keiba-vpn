import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.scraper import odds_fix
from src.scraper.parse_audit import audit_frame, problems
from src.scraper.parsers import RaceResultParser

GOOD = [2.7, 4.7, 5.4, 6.7, 11.0, 13.4, 16.8, 24.6, 33.6, 49.0, 67.0, 100.0]   # Σ(1/odds) ≈ 1.26
BROKEN = [84.0, 90.0, 87.0, 74.0, 91.0, 80.0, 78.0, 88.0, 95.0, 72.0, 83.0, 86.0]   # 追走指数を読んだ値（Σ ≈ 0.15）


def test_plausible_win_odds():
    assert 1.2 < odds_fix.sum_inv_odds(GOOD) < 1.35
    assert odds_fix.is_plausible_win_odds(GOOD)
    assert not odds_fix.is_plausible_win_odds(BROKEN)
    assert odds_fix.is_plausible_win_odds([1.5, 3.0])           # 少頭数は判定しない
    assert odds_fix.is_plausible_win_odds([0.0] + GOOD)         # 取消馬の 0 は数えない
    assert not odds_fix.is_plausible_win_odds([0.0] * 8 + GOOD[:4])   # 大半が 0（odds 欠落）は不自然


def _tables(tmp_path, monkeypatch):
    tdir = tmp_path / "tables" / "2025"
    tdir.mkdir(parents=True)
    rows_rr, rows_ro = [], []
    for rid, odds in (("202501010101", BROKEN), ("202501010102", GOOD)):
        for i, (o, g) in enumerate(zip(odds, GOOD), start=1):
            rows_rr.append({"race_id": rid, "date": "2025-07-26", "horse_number": i, "odds": o, "popularity": 99 if rid.endswith("01") else i})
            rows_ro.append({"race_id": rid, "horse_number": i, "win_odds": g, "popularity": i})
    pd.DataFrame(rows_rr).to_parquet(tdir / "race_result_flat.parquet")
    pd.DataFrame(rows_ro).to_parquet(tdir / "race_odds_flat.parquet")
    monkeypatch.setattr(odds_fix, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(odds_fix, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(odds_fix, "OVERLAY", tmp_path / "out" / "win_odds_fix.parquet")
    monkeypatch.setattr(odds_fix, "RESIDUAL", tmp_path / "out" / "unfixable.txt")
    monkeypatch.setattr(odds_fix, "_CACHE", None)
    monkeypatch.setattr(odds_fix, "_LOOKUP", None)


def test_overlay_fixes_only_broken_races(tmp_path, monkeypatch):
    _tables(tmp_path, monkeypatch)
    res = odds_fix.write_overlay(odds_fix.OVERLAY)
    assert res["races"] == 1 and res["unfixable"] == 0
    ov = pd.read_parquet(odds_fix.OVERLAY)
    assert set(ov["race_id"]) == {"202501010101"}
    assert ov.sort_values("horse_number")["odds_fixed"].tolist() == GOOD
    assert ov.sort_values("horse_number")["popularity_fixed"].tolist() == list(range(1, 13))


def test_apply_pandas_and_fix_result(tmp_path, monkeypatch):
    _tables(tmp_path, monkeypatch)
    odds_fix.write_overlay(odds_fix.OVERLAY)
    monkeypatch.setattr(odds_fix, "_CACHE", None)
    df = pd.read_parquet(tmp_path / "tables" / "2025" / "race_result_flat.parquet")
    out = odds_fix.apply_pandas(df)
    b, g = out[out.race_id == "202501010101"].sort_values("horse_number"), out[out.race_id == "202501010102"].sort_values("horse_number")
    assert b["odds"].tolist() == GOOD and b["odds_raw"].tolist() == BROKEN and b["popularity"].tolist() == list(range(1, 13))
    assert g["odds"].tolist() == GOOD
    res = {"race_id": "202501010101", "entries": [{"horse_number": i, "odds": o, "popularity": 99} for i, o in enumerate(BROKEN, start=1)]}
    fixed = odds_fix.fix_result(res)
    assert [e["odds"] for e in fixed["entries"]] == GOOD and res["entries"][0]["odds"] == BROKEN[0]   # 元は変えない
    untouched = {"race_id": "202509999999", "entries": [{"horse_number": 1, "odds": 3.0}]}
    assert odds_fix.fix_result(untouched) is untouched


def test_env_switch_disables_fix(tmp_path, monkeypatch):
    _tables(tmp_path, monkeypatch)
    odds_fix.write_overlay(odds_fix.OVERLAY)
    monkeypatch.setattr(odds_fix, "_CACHE", None)
    monkeypatch.setenv("KEIBA_ODDS_FIX", "0")
    assert odds_fix.load_overlay().empty


def test_parse_audit_flags_unfixed_broken_odds(tmp_path, monkeypatch):
    _tables(tmp_path, monkeypatch)
    df = pd.read_parquet(tmp_path / "tables" / "2025" / "race_result_flat.parquet").assign(field_size=12, weight=480, finish_position=1)
    rep = audit_frame(df)
    assert rep["missing"].get("odds_sum") == 1 and rep["samples"]["odds_sum"] == ["202501010101"]
    assert rep["missing"].get("popularity_range") == 1      # 人気 99 > 頭数 12
    assert problems(rep)
    odds_fix.write_overlay(odds_fix.OVERLAY)                 # 補正表で直るレースは検出対象から外す
    monkeypatch.setattr(odds_fix, "_CACHE", None)
    assert "odds_sum" not in audit_frame(df)["missing"]


_HEAD = ["着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差", "ﾀｲﾑ指数", "ﾀｲﾑ指数M", "ｽﾀｰﾄ指数", "追走指数", "上がり指数",
         "通過", "上り", "単勝", "人気", "馬体重", "調教ﾀｲﾑ", "厩舎ｺﾒﾝﾄ", "備考", "調教師", "馬主", "賞金(万円)"]


def test_result_parser_reads_win_odds_from_premium_25_column_table():
    """指数列が入った 25 列の会員表示でも、単勝・人気は列名で読む（固定位置だと追走指数・上がり指数を読む）。"""
    row = ["1", "3", "3", '<a href="/horse/2023100001/">ショウナンカリス</a>', "牝2", "55", '<a href="/jockey/01234/">北村友一</a>', "1:08.9", "",
           "87", "87", "88", "84", "103", "6-5", "34.1", "4.8", "3", "408(+4)", "", "", "出遅れ", '<a href="/trainer/00001/">加藤</a>', "馬主", "560.0"]
    html = ('<table class="race_table_01"><tr>' + "".join(f"<th>{h}</th>" for h in _HEAD) + "</tr><tr>"
            + "".join(f"<td>{c}</td>" for c in row) + "</tr></table>")
    e = RaceResultParser()._parse_entries(BeautifulSoup(html, "html.parser"))[0]
    assert e["odds"] == pytest.approx(4.8) and e["popularity"] == 3
    assert e["last_3f"] == pytest.approx(34.1) and e["passing_order"] == "6-5" and e["weight"] == 408
