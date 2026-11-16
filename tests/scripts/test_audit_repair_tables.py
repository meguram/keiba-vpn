"""データ監査の補い（src/scripts/data/audit_repair_tables.py）のテスト。本番の表を書き換えず、作業用の写しだけを直す。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import audit_repair_tables as rt


def _race(rid, n=10, **over):
    hn = np.arange(1, n + 1)
    odds = np.round(np.linspace(2.0, 30.0, n), 1)
    df = pd.DataFrame({"race_id": rid, "date": "2025-07-26", "race_name": "3歳未勝利", "surface": "芝", "race_class": "未勝利",
                       "direction": "右", "distance": 1600, "field_size": n, "round": 0, "horse_number": hn, "horse_id": [f"h{k}" for k in hn],
                       "finish_position": hn, "odds": odds, "popularity": hn, "last_3f": 35.0, "weight": 480, "weight_change": 0,
                       "payoff": json.dumps({"単勝": {"numbers": "1", "payout": str(int(round(odds[0] * 100)))}})})
    for k, v in over.items():
        df[k] = v
    return df


def test_repair_flat_fixes_obstacle_round_shift_and_odds_only_breaks():
    ok = _race("202501010101")
    ob = _race("202501010102", race_name="障害4歳以上未勝利", race_class="", direction="", last_3f=0.0)
    shift = _race("202501010103")
    shift["weight"] = 7
    shift["last_3f"] = 95.0
    odds_only = _race("202501010104")
    odds_only["odds"] = 93.0                         # 1 着のオッズ×100 ≠ 払戻
    one_row = _race("202501010105")
    one_row.loc[3, "weight"] = 2                     # 1 頭だけずれた行
    df, rep = rt.repair_flat(pd.concat([ok, ob, shift, odds_only, one_row], ignore_index=True))
    by = {r: g for r, g in df.groupby("race_id")}
    assert (by["202501010102"]["surface"] == "障").all() and (by["202501010101"]["surface"] == "芝").all()
    assert by["202501010101"]["round"].tolist() == [1] * 10 and by["202501010105"]["round"].iloc[0] == 5
    assert by["202501010103"][["odds", "popularity", "last_3f", "weight"]].isna().all().all()
    assert by["202501010103"]["finish_position"].notna().all()                     # 着順（ラベル）は残す
    assert by["202501010104"]["odds"].isna().all() and by["202501010104"]["popularity"].notna().all()
    assert by["202501010105"]["weight"].isna().sum() == 1
    assert by["202501010102"]["last_3f"].isna().all()                              # 障害の上がり 0 → 欠損
    clean = by["202501010101"]                                                     # 壊れていないレースは round 以外そのまま
    assert clean["odds"].tolist() == ok["odds"].tolist() and clean["weight"].tolist() == ok["weight"].tolist()
    assert rep["shifted_races"] == ["202501010103"] and rep["odds_broken_races"] == ["202501010104"]


def test_main_writes_only_to_out_and_links_the_rest(tmp_path):
    src = tmp_path / "tables"
    (src / "2025").mkdir(parents=True)
    _race("202501010101").to_parquet(src / "2025" / "race_result_flat.parquet")
    pd.DataFrame({"race_id": ["202501010101"], "race_name": ["x"], "surface": ["芝"], "round": [0]}).to_parquet(src / "2025" / "race_result_race.parquet")
    pd.DataFrame({"a": [1]}).to_parquet(src / "2025" / "race_shutuba_flat.parquet")
    before = (src / "2025" / "race_result_flat.parquet").read_bytes()
    out = tmp_path / "fixed"
    import sys
    argv = sys.argv
    try:
        sys.argv = ["x", "--src", str(src), "--out", str(out), "--years", "2025-2025"]
        assert rt.main() == 0
        assert rt.main() == 0                                   # 2 回目でも本番へのリンクの先に書かない
    finally:
        sys.argv = argv
    assert (src / "2025" / "race_result_flat.parquet").read_bytes() == before
    assert not (out / "2025" / "race_result_flat.parquet").is_symlink()
    assert (out / "2025" / "race_shutuba_flat.parquet").is_symlink()
    assert pd.read_parquet(out / "2025" / "race_result_flat.parquet")["round"].iloc[0] == 1


def test_main_refuses_out_inside_src(tmp_path):
    import sys
    (tmp_path / "tables").mkdir()
    argv = sys.argv
    try:
        sys.argv = ["x", "--src", str(tmp_path / "tables"), "--out", str(tmp_path / "tables" / "fixed")]
        with pytest.raises(SystemExit):
            rt.main()
    finally:
        sys.argv = argv


# ── critic の指摘（2026-11-16）への追加 ─────────────────────────────────

def test_young_jockeys_series_is_not_obstacle_but_js_ops_are():
    flat_js = _race("201701010101", race_name="ヤングJSFR中山1(1勝)")
    js_op = _race("201701010102", race_name="新潟JS", race_class="", direction="")
    jg = _race("201701010103", race_name="中山大障害(JG1)")
    df, _ = rt.repair_flat(pd.concat([flat_js, js_op, jg], ignore_index=True))
    s = df.groupby("race_id")["surface"].first()
    assert s["201701010101"] == "芝" and s["201701010102"] == "障" and s["201701010103"] == "障"


def test_odds_are_kept_when_the_win_payout_itself_is_misaligned():
    r = _race("202005050104")
    r["payoff"] = json.dumps({"単勝": {"numbers": "3", "payout": "170", "popularity": "1"}})   # 払戻の側が 3 番（1 着は 1 番）
    df, rep = rt.repair_flat(r)
    assert df["odds"].notna().all() and rep["odds_broken_races"] == []


def test_popularity_is_dropped_when_it_disagrees_with_the_payout():
    r = _race("202306050911")
    r["odds"] = 93.0
    r["payoff"] = json.dumps({"単勝": {"numbers": "1", "payout": "550", "popularity": "2"}})      # 1 着の人気は 1 だが払戻は 2
    df, rep = rt.repair_flat(r)
    assert df["odds"].isna().all() and df["popularity"].isna().all() and rep["popularity_broken_races"] == ["202306050911"]


def test_dead_heat_and_list_payouts_are_not_judged():
    r = _race("202504010709")
    r.loc[1, "finish_position"] = 1
    r["payoff"] = json.dumps({"単勝": [{"numbers": "1", "payout": "300"}, {"numbers": "2", "payout": "400"}]})
    df, rep = rt.repair_flat(r)
    assert rep["odds_broken_races"] == [] and df["odds"].notna().all()


def test_scratched_rows_and_9999_popularity_are_kept_and_shift_clears_weight_change():
    r = _race("202501010101")
    r.loc[0, ["finish_position", "odds", "popularity"]] = [-1, 0.0, 9999]
    s = _race("202501010102")
    s["weight"] = 7
    s["weight_change"] = 3
    df, _ = rt.repair_flat(pd.concat([r, s], ignore_index=True))
    a = df[df["race_id"] == "202501010101"]
    assert a["odds"].iloc[0] == 0.0 and a["popularity"].iloc[0] == 9999                    # 取消の行は触らない
    assert df.loc[df["race_id"] == "202501010102", "weight_change"].isna().all()


def test_main_links_other_years_and_fixes_race_table_and_never_writes_through_links(tmp_path):
    import sys

    import polars as pl

    src = tmp_path / "tables"
    for y in ("2024", "2025"):
        (src / y).mkdir(parents=True)
        _race(f"{y}01010101", race_name="障害4歳以上未勝利").to_parquet(src / y / "race_result_flat.parquet")
        pd.DataFrame({"race_id": [f"{y}01010101"], "race_name": ["障害4歳以上未勝利"], "surface": ["芝"], "round": [0]}).to_parquet(
            src / y / "race_result_race.parquet")
    (src / "_table_quality.json").write_text("{}")
    out = tmp_path / "fixed"
    out.mkdir()
    (out / "2025").symlink_to(src / "2025")                      # 以前の版で年のフォルダごとリンクだった
    before = {p: p.read_bytes() for p in src.rglob("*.parquet")}
    argv = sys.argv
    try:
        sys.argv = ["x", "--src", str(src), "--out", str(out), "--years", "2025-2025"]
        assert rt.main() == 0
        (out / "2025" / "dangling").symlink_to(tmp_path / "gone")   # 壊れたリンクがあっても再実行で落ちない
        assert rt.main() == 0
    finally:
        sys.argv = argv
    assert all(p.read_bytes() == b for p, b in before.items())     # 本番は 1 バイトも変わらない
    assert (out / "2024").is_symlink() and (out / "_table_quality.json").is_symlink()
    race = pd.read_parquet(out / "2025" / "race_result_race.parquet")
    assert race["surface"].iloc[0] == "障" and race["round"].iloc[0] == 1
    assert pl.read_parquet(out / "2025" / "race_result_flat.parquet").height == 10
