"""欠損の原因分類(missing_audit)のテスト。小さな合成データで、分類の優先順位と整合チェックを確認する。"""

from __future__ import annotations

import polars as pl

from src.pipeline.models import missing_audit as M


def _ctx(**cols):
    n = len(next(iter(cols.values())))
    base = {"race_id": [f"r{i}" for i in range(n)], "horse_number": list(range(1, n + 1)), "surface": ["芝"] * n}
    return pl.DataFrame({**base, **cols})


def _by_reason(summary: pl.DataFrame, feature: str) -> dict[str, str]:
    return {r["理由"]: r["原因"] for r in summary.filter(pl.col("feature") == feature).to_dicts()}


def test_index_causes_follow_priority():
    ctx = _ctx(
        speed_max=[None, None, None, None, None, 50.0],
        horse_prior_races=[0, 3, 3, 3, 3, 3],
        ix_race=[True, False, True, True, True, True],
        ix_row=[True, False, False, True, True, True],
        ix_nn_speed_max=[False, False, False, False, True, True],
    )
    summary, suspects = M.classify(ctx, {"speed_max": M.feature_rules({"speed_max"})["speed_max"]})
    reasons = _by_reason(summary, "speed_max")
    assert reasons["行はあるが値が空（初出走）"] == M.RAW_EMPTY
    assert reasons["そのレースの race_index が無い"] == M.RAW_RACE
    assert reasons["race_index にこの馬の行が無い（過去走あり）"] == M.RAW_HORSE
    assert reasons["行はあるが値が空（過去走あり）"] == M.RAW_EMPTY
    assert reasons["生データに値があるのに特徴量が欠損"] == M.PROC
    assert suspects.height == 1 and suspects["原因"][0] == M.PROC


def test_prior_rate_null_is_legit_only_when_no_prior_races():
    ctx = _ctx(horse_prior_win_rate=[None, None, 0.1], horse_prior_races=[0, 5, 5])
    summary, _ = M.classify(ctx, {"horse_prior_win_rate": M.feature_rules({"horse_prior_win_rate"})["horse_prior_win_rate"]})
    r = _by_reason(summary, "horse_prior_win_rate")
    assert r["馬の過去実績なし（prior_races=0）"] == M.LEGIT
    assert r["過去実績があるのに率が欠損（履歴統計の加工）"] == M.PROC


def test_validate_legit_rules_detects_inconsistent_rule():
    # 「初出走は欠損のはず」というルールなのに、初出走の行に値が入っている → 欠損率が下がる
    ctx = _ctx(horse_prior_win_rate=[None, 0.2, 0.3], horse_prior_races=[0, 0, 0])
    v = M.validate_legit_rules(ctx, {"horse_prior_win_rate": M.feature_rules({"horse_prior_win_rate"})["horse_prior_win_rate"]})
    assert v["該当行"][0] == 3 and v["実際に欠損"][0] == 1 and abs(v["欠損率"][0] - 1 / 3) < 1e-9


def test_infer_min_history_finds_threshold_and_none_for_conditional():
    df = pl.DataFrame({"n": [0, 0, 1, 2, 3, 4] * 50, "f": [None, None, 1.0, 1.0, 1.0, 1.0] * 50})
    assert M.infer_min_history(df, "f", "n") == 1
    cond = pl.DataFrame({"n": [i for i in range(13) for _ in range(40)], "f": [None, 1.0] * 260})  # どの過去走数でも半分欠損 = 過去走数では説明できない
    assert M.infer_min_history(cond, "f", "n") is None


def test_value_profile_flags_sentinels_and_text_in_numeric_columns():
    df = pl.DataFrame({"pos": ["1", "2", "3", "中", "-1", "4", "5", "6", "7", "8", "9", "10"]})
    p = M.value_profile(df).to_dicts()[0]
    assert p["数値列に混じる文字列"] == 1 and p["代用値らしい値(-1,0,空,-)"] == 1


def test_finish_status_evidence_separates_timed_rows_and_rank_gaps():
    rr = pl.DataFrame({"race_id": ["a"] * 4 + ["b"] * 3, "horse_number": [1, 2, 3, 4, 1, 2, 3],
                       "finish_position": [1, 2, -1, -1, 1, 3, -1], "time_sec": [60.0, 61.0, 0.0, 70.0, 60.0, 61.0, 0.0],
                       "finish_time": [""] * 7, "passing_order": [""] * 7})
    summary, detail = M.finish_status_evidence(rr)
    d = {r["区分"]: r["行数"] for r in summary.to_dicts()}
    assert d["タイムなし: 出走取消・競走除外・競走中止と整合"] == 2 and d["タイムあり: 失格・降着か解析失敗（要確認）"] == 1
    assert set(detail["種別"]) == {"着順<=0なのにタイムあり", "着順に欠番"}


def test_track_bias_chronology_check_flags_first_race_with_value():
    df = pl.DataFrame({
        "race_id": ["a1", "a2", "a3"], "date": ["2026-01-01"] * 3, "venue_code": ["05"] * 3, "surface": ["芝"] * 3,
        "start_time": ["10:00", "11:00", "12:00"],
        "track_bias_winner_3f_same_day_prior": [35.0, 36.0, 36.5],   # 最初のレース a1 に値がある = 先読み
    })
    wl = pl.DataFrame({"race_id": ["a1", "a2", "a3"], "winner_l3f_nn": [True, True, True]})
    s, leak = M.track_bias_chronology_check(df, wl)
    assert leak["race_id"].to_list() == ["a1"] and s["レース数"].to_list() == [1, 1, 0]
