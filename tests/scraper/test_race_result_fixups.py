"""結果の書き出し・解析の不具合（データ監査 2026-11-16 の C3・C4・C5）の再発防止。合成のデータだけで確かめる。

- C3: 障害レースの面が「芝」「ダート」「空」で保存されている → レース名・出馬表の面で「障」に直す（平地は変えない）。
- C4: race 表の round が 0 のまま → race_id の末尾 2 桁。
- C5: 1 着同着の単勝が {"numbers": "611", "payout": "1,320630"} に連結 → 2 頭分の list に分ける（分けられなければ触らない）。
"""

import json

import pandas as pd
import pytest

from src.scraper import export_tables as et
from src.scraper import odds_fix
from src.scraper.parsers import RaceResultParser
from src.scraper.race_result_fixups import is_steeplechase_name, resolve_surface, split_deadheat_win


@pytest.fixture(autouse=True)
def _no_odds_fix(monkeypatch):
    monkeypatch.setattr(odds_fix, "load_lookup", lambda *a, **k: {})


# ── C3 障害の面 ─────────────────────────────────────────────

@pytest.mark.parametrize("name,dist", [
    ("障害3歳以上未勝利", 2750), ("障害4歳以上オープン", 0), ("第27回新潟ジャンプS(JGIII)", 0),
    ("第27回阪神スプリングJ(JGII)", 3900), ("中山グランドJ", None), ("東京HJ", 3110), ("新潟JS", 2890), ("京都ハイJ", None),
    ("阪神SJ", 3900), ("小倉SJ", 3390),
])
def test_steeplechase_names(name, dist):
    assert is_steeplechase_name(name, dist)


@pytest.mark.parametrize("name,dist", [
    ("ヤングJSFR中京1(1勝)", 2000), ("ヤングJSFR中山2(2勝)", None),   # 平地（ヤングジョッキーズシリーズ）。素の JS で障害にしない
    ("ステイヤーズS(GII)", 3600), ("万葉S(OP)", 3000), ("JRAアニバーサリーS", 1600), ("3歳以上1勝クラス", 1200),
    ("障害3歳以上未勝利", 1600),   # 距離が平地の短さなら名前が障害でも障にしない（誤判定よけ）
    ("", 3000), (None, None),
])
def test_flat_names_are_not_steeplechase(name, dist):
    assert not is_steeplechase_name(name, dist)


def test_resolve_surface_order():
    assert resolve_surface("芝", "障害4歳以上未勝利", 3000) == "障"
    assert resolve_surface("ダート", "中山新春ジャンプS(OP)", 3200) == "障"
    assert resolve_surface("", "ペガサスジャンプS(OP)", 0) == "障"
    assert resolve_surface("芝", "清秋S", 3210, fallback_surface="障") == "障"      # 出馬表が障なら名前に頼らない
    assert resolve_surface("障", "名前の無い障害", 0) == "障"                       # 障を平地に戻さない
    assert resolve_surface("芝", "ヤングJSFR中京1(1勝)", 2000, fallback_surface="芝") == "芝"
    assert resolve_surface("芝", "ステイヤーズS(GII)", 3600) == "芝"


def _result(surface="芝", race_name="障害3歳以上未勝利", rid="202204010304", rnd=0, payoff=None, entries=None):
    return {
        "race_id": rid, "race_name": race_name, "surface": surface, "distance": 2750, "direction": "", "round": rnd,
        "date": "2022-07-30", "venue": "新潟",
        "entries": entries or [{"horse_number": 1, "horse_id": "h1", "finish_position": 1}, {"horse_number": 2, "horse_id": "h2", "finish_position": 2}],
        "payoff": payoff or {"単勝": {"numbers": "1", "payout": "350"}},
    }


def test_export_flat_and_race_rows_fix_steeple_surface():
    d = _result(surface="ダート")
    assert {r["surface"] for r in et._data_to_flat_rows(d, "race_result")} == {"障"}
    assert et._data_to_race_row(d)["surface"] == "障"
    flat = _result(surface="芝", race_name="ヤングJSFR中京1(1勝)")
    flat["distance"] = 2000
    assert {r["surface"] for r in et._data_to_flat_rows(flat, "race_result")} == {"芝"}
    assert et._data_to_race_row(flat)["surface"] == "芝"


def test_export_uses_shutuba_meta_steeple_over_wrong_result_surface():
    d = _result(surface="芝", race_name="清秋S(OP)")          # 名前で決まらない形でも、出馬表のメタが障なら障
    idx = {d["race_id"]: {"surface": "障", "race_class": ""}}
    assert {r["surface"] for r in et._data_to_flat_rows(d, "race_result", idx)} == {"障"}
    assert et._data_to_race_row(d, idx)["surface"] == "障"


def test_local_meta_index_takes_steeple_from_local_shutuba(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "OUTPUT_DIR", tmp_path)
    y = tmp_path / "2022"
    y.mkdir(parents=True)
    pd.DataFrame([
        {"race_id": "A", "horse_number": 1, "surface": "芝", "race_class": "", "distance": 3000},
        {"race_id": "B", "horse_number": 1, "surface": "芝", "race_class": "1勝", "distance": 1600},
    ]).to_parquet(y / "race_result_flat.parquet", index=False)
    pd.DataFrame([
        {"race_id": "A", "horse_number": 1, "surface": "障", "race_class": ""},
        {"race_id": "B", "horse_number": 1, "surface": "ダート", "race_class": "1勝"},
    ]).to_parquet(y / "race_shutuba_flat.parquet", index=False)
    idx = et._build_meta_index_from_race_result_flat("2022")
    assert idx["A"]["surface"] == "障"
    assert idx["B"]["surface"] == "芝"      # 障以外は空のときだけ補う（空でない結果の面は上書きしない）


def test_parser_sets_steeple_from_race_name():
    html = """<html><body><dl class="racedata"><dd><h1>障害4歳以上オープン</h1>
    <p><span>芝 3350m / 天候 : 晴 / 芝 : 良 / 発走 : 11:30</span></p></dd></dl></body></html>"""
    info = RaceResultParser()._parse_race_info(__import__("bs4").BeautifulSoup(html, "html.parser"))
    assert info["race_name"] == "障害4歳以上オープン"
    assert info["surface"] == "障"


def test_parser_keeps_flat_surface():
    html = """<html><body><dl class="racedata"><dd><h1>ステイヤーズS(GII)</h1>
    <p><span>芝内2周3600m / 天候 : 晴 / 芝 : 良 / 発走 : 15:25</span></p></dd></dl></body></html>"""
    info = RaceResultParser()._parse_race_info(__import__("bs4").BeautifulSoup(html, "html.parser"))
    assert info["surface"] == "芝" and info["distance"] == 3600


# ── C4 round ─────────────────────────────────────────────

def test_round_from_race_id_in_flat_and_race_rows():
    d = _result(rid="202204010309", rnd=0)
    assert {r["round"] for r in et._data_to_flat_rows(d, "race_result")} == {9}
    assert et._data_to_race_row(d)["round"] == 9
    d2 = _result(rid="202204010311", rnd=None)
    assert et._data_to_race_row(d2)["round"] == 11
    d3 = _result(rid="202204010311", rnd=11)
    assert et._data_to_race_row(d3)["round"] == 11


def test_race_row_without_round_key_does_not_gain_one():
    d = _result()
    d.pop("round")
    assert "round" not in et._data_to_race_row(d)


# ── C5 同着の単勝 ─────────────────────────────────────────

def _ents(*pairs):
    return [{"horse_number": n, "horse_id": f"h{n}", "finish_position": f} for n, f in pairs]


def test_split_deadheat_win_basic():
    pay = {"単勝": {"numbers": "611", "payout": "1,320630"}, "複勝": [{"numbers": "6", "payout": "430", "popularity": ""}]}
    out = split_deadheat_win(pay, _ents((6, 1), (11, 1), (14, 3)))
    assert out["単勝"] == [{"numbers": "6", "payout": "1,320", "popularity": ""}, {"numbers": "11", "payout": "630", "popularity": ""}]
    assert out["複勝"] == pay["複勝"]
    assert pay["単勝"] == {"numbers": "611", "payout": "1,320630"}       # 元は変えない


@pytest.mark.parametrize("nums,pays,ents,expect", [
    ("1011", "270230", [(10, 1), (11, 1)], [("10", "270"), ("11", "230")]),
    ("312", "1,0801,260", [(3, 1), (12, 1)], [("3", "1,080"), ("12", "1,260")]),
    ("114", "900210", [(4, 1), (11, 1)], [("4", "210"), ("11", "900")]),     # 馬番の降順に連結されていても組を保って昇順に並べる
    ("1"+"2", "1,5001,500", [("1", "1"), ("2", "1")], [("1", "1,500"), ("2", "1,500")]),   # 着順が文字列でもよい
])
def test_split_deadheat_win_variants(nums, pays, ents, expect):
    out = split_deadheat_win({"単勝": {"numbers": nums, "payout": pays}}, _ents(*ents))
    assert [(x["numbers"], x["payout"]) for x in out["単勝"]] == expect


@pytest.mark.parametrize("win,ents", [
    ({"numbers": "6", "payout": "350"}, [(6, 1), (11, 2)]),                         # 同着でない
    ({"numbers": "611", "payout": "1,320630"}, [(6, 1), (11, 2)]),                  # 1 着が 1 頭（連結の形でも分けない）
    ({"numbers": "6114", "payout": "200200200"}, [(6, 1), (11, 1), (4, 1)]),         # 3 頭同着
    ({"numbers": "612", "payout": "1,320630"}, [(6, 1), (11, 1)]),                   # 馬番が 1 着の 2 頭と合わない
    ({"numbers": "611", "payout": "13206300"}, [(6, 1), (11, 1)]),                   # 金額の分け方が決まらない（カンマが無く 4 桁以上）
    ({"numbers": "611", "payout": "1,3206,3001,000"}, [(6, 1), (11, 1)]),            # 3 つ分の金額
    ({"numbers": "111", "payout": "350420"}, [(1, 1), (11, 1)]),                     # 1 と 11 は連結の順が決まらない
])
def test_split_deadheat_win_leaves_ambiguous_untouched(win, ents):
    pay = {"単勝": dict(win)}
    assert split_deadheat_win(pay, _ents(*ents)) is pay


def test_split_deadheat_win_already_list_untouched():
    pay = {"単勝": [{"numbers": "6", "payout": "1,320", "popularity": "2"}, {"numbers": "11", "payout": "630", "popularity": "1"}]}
    assert split_deadheat_win(pay, _ents((6, 1), (11, 1))) is pay


def test_export_flat_and_race_rows_split_deadheat_win():
    d = _result(rid="202504010709", payoff={"単勝": {"numbers": "611", "payout": "1,320630"}},
                entries=_ents((6, 1), (11, 1), (14, 3)))
    for s in {r["payoff"] for r in et._data_to_flat_rows(d, "race_result")} | {et._data_to_race_row(d)["payoff"]}:
        assert json.loads(s)["単勝"] == [{"numbers": "6", "payout": "1,320", "popularity": ""},
                                        {"numbers": "11", "payout": "630", "popularity": ""}]
    assert d["payoff"]["単勝"] == {"numbers": "611", "payout": "1,320630"}   # 渡したデータは変えない
