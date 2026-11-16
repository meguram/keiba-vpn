"""表の中身の監査（src/data_health/table_audit.py）のテスト。合成の表で、欠けの急変・列ずれ・整合の検査を確かめる。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_health import table_audit as ta


def _issue(issues, check, column):
    return next(i for i in issues if i.check == check and i.column == column)


# ── 欠けの集計と急変 ──────────────────────────────────────────────────────

def test_missing_mask_counts_none_nan_and_blank_strings():
    s = pd.Series(["a", "", "  ", None, np.nan, "b"], dtype=object)
    assert ta.missing_mask(s).tolist() == [False, True, True, True, True, False]
    assert ta.missing_mask(pd.Series([1.0, np.nan, 0.0])).tolist() == [False, True, False]


def test_month_key_reads_mixed_date_formats():
    m = ta.month_key(pd.Series(["2025-07-26", "2025/08/01", "20250901", None]))
    assert m.iloc[:3].tolist() == ["2025-07", "2025-08", "2025-09"] and pd.isna(m.iloc[3])


def _profile(values: dict[str, float], rows: int = 500) -> pd.DataFrame:
    return pd.DataFrame([{"month": k, "column": "race_class", "rows": rows, "missing_pct": v, "zero_pct": 0.0} for k, v in values.items()])


def test_detect_jumps_flags_month_above_previous_and_last_year():
    p = _profile({"2025-09": 1.0, "2025-10": 1.0, "2026-09": 1.0, "2026-10": 40.0})
    j = ta.detect_jumps(p)
    assert j["month"].tolist() == ["2026-10"] and j["delta"].iloc[0] == pytest.approx(39.0)


def test_detect_jumps_ignores_seasonal_rise_seen_last_year_and_small_months():
    # 前月より増えたが、前年同月も同じ水準（季節・番組の都合）は拾わない
    p = _profile({"2024-06": 1.0, "2024-07": 29.0, "2025-06": 1.0, "2025-07": 30.0, "2026-06": 1.0, "2026-07": 31.0})
    assert "2026-07" not in set(ta.detect_jumps(p)["month"])
    # 行数の少ない月は判定に使わない
    p2 = pd.concat([_profile({"2026-01": 1.0}), _profile({"2026-02": 90.0}, rows=10)])
    assert ta.detect_jumps(p2).empty


def test_missing_profile_and_period_summary():
    df = pd.DataFrame({"date": ["2023-01-05"] * 4 + ["2025-03-01"] * 4, "race_class": ["A", "", None, "B", "A", "A", "A", "A"],
                       "weight": [480, 0, 500, 0, 470, 480, 490, 500]})
    p = ta.missing_profile(df)
    r = p.set_index(["month", "column"])
    assert r.loc[("2023-01", "race_class"), "missing_pct"] == 50.0 and r.loc[("2023-01", "weight"), "zero_pct"] == 50.0
    s = ta.period_summary(p)
    assert s.loc["race_class", "学習(〜2023)"] == 50.0 and s.loc["race_class", "テスト(2025)"] == 0.0


# ── 枠番と馬番 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expect", [
    (8, [1, 2, 3, 4, 5, 6, 7, 8]),
    (12, [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8]),
    (16, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]),
    (17, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8]),
    (18, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8]),
])
def test_expected_bracket_matches_jra_rule(n, expect):
    assert [ta.expected_bracket(h, n) for h in range(1, n + 1)] == expect
    assert ta.expected_bracket(n + 1, n) is None


# ── 結果の表の行・レースの検査 ────────────────────────────────────────────

def _race(rid="202501010101", n=10, **over):
    hn = np.arange(1, n + 1)
    pop = np.roll(hn, 3)                                   # 人気・着順は馬番と別の並び
    odds = np.round(2.0 + (pop - 1) * 48.0 / max(n - 1, 1), 1)
    df = pd.DataFrame({
        "race_id": rid, "date": "2025-07-26", "race_name": "3歳未勝利", "surface": "芝", "distance": 1600, "track_condition": "良", "weather": "晴",
        "race_class": "未勝利", "grade": "未勝利", "field_size": n, "venue": "札幌", "start_time": "10:00", "round": 1,
        "horse_number": hn, "bracket_number": [ta.expected_bracket(int(h), n) for h in hn], "finish_position": np.roll(hn, 5),
        "horse_id": [f"2022{100000 + int(h):06d}" for h in hn], "sex_age": "牡3", "jockey_weight": 56.0,
        "odds": odds, "popularity": pop, "last_3f": 35.0, "weight": 480, "weight_change": 0, "time_sec": 96.0 + hn * 0.1,
        "payoff": json.dumps({"単勝": {"numbers": str(int(hn[np.roll(hn, 5) == 1][0])), "payout": str(int(round(odds[np.roll(hn, 5) == 1][0] * 100)))}}),
    })
    for k, v in over.items():
        df[k] = v
    return df


def test_clean_race_has_no_issues():
    df = _race()
    issues = ta.check_result_rows(df) + ta.check_result_races(df) + ta.check_horse_birth(df)
    assert [i.as_dict() for i in issues if i.n_bad] == []


def test_column_shift_race_is_flagged():
    df = _race()
    # 2026-02-08 小倉 7R の形: オッズ・人気・上がり・馬体重に別の数字が入る
    df["weight"] = [7, 7, 6, 10, 2, 3, 9, 13, 14, 10]
    df["last_3f"] = 95.0
    df["popularity"] = [102, 96, 96, 101, 91, 91, 92, 90, 93, 85]
    races = ta.check_result_races(df)
    assert _issue(races, "shift", "列ずれのレース").n_bad == 1
    rows = ta.check_result_rows(df)
    assert _issue(rows, "range", "weight").n_bad == 10 and _issue(rows, "range", "last_3f").n_bad == 10


def test_obstacle_race_with_flat_surface_is_flagged():
    df = _race(race_name="障害4歳以上未勝利", surface="芝")
    assert _issue(ta.check_result_races(df), "race_const", "障害の surface").n_bad == 1
    df2 = _race(race_name="障害4歳以上未勝利", surface="障", distance=2890, time_sec=190.0)
    assert _issue(ta.check_result_races(df2), "race_const", "障害の surface").n_bad == 0


def test_bracket_shift_and_popularity_odds_inversion():
    df = _race(n=12)
    df.loc[5, "bracket_number"] = 6       # 12 頭の 6 番は 5 枠
    df.loc[df["popularity"] == 1, "odds"] = 99.0      # 1 番人気なのにオッズが最も高い
    rows = ta.check_result_rows(df)
    assert _issue(rows, "shift", "bracket_number").n_bad == 1
    assert _issue(rows, "shift", "popularity/odds").n_bad >= 1


def test_popularity_gap_from_scratched_horse_is_not_an_inversion():
    df = _race(n=10)
    df["popularity"] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]     # 取消の馬の 9 番人気が抜けた
    df["odds"] = df["popularity"] * 2.0
    assert _issue(ta.check_result_rows(df), "shift", "popularity/odds").n_bad == 0


def test_win_payout_and_dead_heat_concatenation():
    assert ta._win_payout(json.dumps({"単勝": {"numbers": "3", "payout": "1,480"}})) == [("3", 1480.0)]
    # 同着の単勝が連結されて読めない形（2025 年 202504010709 の "611" / "1,320630"）は、1 着の馬番が払戻に無いとして拾う
    df = _race()
    df.loc[1, "finish_position"] = 1          # 馬番 2 と馬番 6 の同着
    df["payoff"] = json.dumps({"単勝": {"numbers": "62", "payout": "1,320630"}})
    assert _issue(ta.check_result_races(df), "payoff", "odds*100==単勝払戻").n_bad == 2


def test_win_odds_times_100_must_match_payout():
    df = _race()
    df["payoff"] = json.dumps({"単勝": {"numbers": "6", "payout": "450"}})   # 1 着（馬番 6）のオッズは 12.7 なのに 450 円
    assert _issue(ta.check_result_races(df), "payoff", "odds*100==単勝払戻").n_bad == 1


def test_horse_birth_year_and_age():
    df = _race()
    df.loc[0, "sex_age"] = "牡5"
    assert _issue(ta.check_horse_birth(df), "horse", "生年×馬齢").n_bad == 1


def test_compare_entries_between_result_and_shutuba():
    r = _race()
    s = _race().iloc[:-1].copy()
    s.loc[0, "horse_number"] = 9
    out = {i.column: i.n_bad for i in ta.compare_entries(r, s)}
    assert out["結果にだけある馬"] == 1 and out["馬番の不一致"] == 1


# ── 学習と推論の欠けの比較 ──────────────────────────────────────────────

def test_compare_missing_lists_large_gaps_both_directions():
    train = pd.DataFrame({"pf_rank": [1.0, 2.0, np.nan, 4.0], "z_hat": [0.1] * 4, "wc_n": [1.0, 2.0, 3.0, 4.0], "only_train": [1, 2, 3, 4]})
    serve = pd.DataFrame({"pf_rank": [np.nan] * 4, "z_hat": [0.2] * 4, "wc_n": [np.nan, np.nan, 1.0, 2.0], "has_comment": [0.0] * 4})
    d = ta.compare_missing(train, serve, threshold=10)
    got = dict(zip(d["column"], d["delta"]))
    assert got["pf_rank"] == 75.0 and got["wc_n"] == 50.0 and "z_hat" not in got
    assert d.set_index("column").loc["only_train", "only"] == "train" and d.set_index("column").loc["has_comment", "only"] == "serve"
    assert set(list(d["column"])[:2]) == {"only_train", "has_comment"}      # 差の大きい順（片方に列ごと無い = 100）


# ── CLI ──────────────────────────────────────────────────────────────────

def test_audit_tables_and_cli_write_outputs(tmp_path):
    root = tmp_path / "tables"
    for y in (2024, 2025):
        (root / str(y)).mkdir(parents=True)
        frames = []
        for m in range(1, 13):
            for k in range(30):
                df = _race(rid=f"{y}0101{m:02d}{k:02d}", date=f"{y}-{m:02d}-10")
                if y == 2025 and m == 10:
                    df["race_class"] = ""      # 2025-10 だけ race_class が空
                frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(root / str(y) / "race_result_flat.parquet")
    res = ta.audit_tables(root, [2024, 2025], ["race_result_flat"])
    j = res["jumps"]
    assert ((j["column"] == "race_class") & (j["month"] == "2025-10")).any()
    out = tmp_path / "out"
    assert ta.main(["--tables-dir", str(root), "--years", "2024-2025", "--tables", "race_result_flat", "--out", str(out)]) == 0
    assert (out / "summary.md").read_text(encoding="utf-8").count("race_class") >= 1 and (out / "issues.json").exists()


def test_parse_years():
    assert ta.parse_years("2016-2018,2020") == [2016, 2017, 2018, 2020]


# ── critic の指摘（2026-11-16）への追加 ─────────────────────────────────

def test_sentinel_values_are_counted_as_missing_not_out_of_range():
    df = _race(n=10)
    df.loc[0, ["finish_position", "odds", "popularity", "weight", "last_3f"]] = [-1, 0.0, 0, 0, 0.0]   # 取消
    o = pd.DataFrame({"race_id": "R", "date": "2025-07-26", "field_size": 10, "horse_number": np.arange(1, 11),
                      "win_odds": [-3.0] + [float(k) for k in range(2, 11)], "popularity": [9999] + list(range(1, 10))})
    rows = ta.check_result_rows(df)
    assert _issue(rows, "range", "odds").n_bad == 0 and _issue(rows, "zero", "odds").n_bad == 1
    assert _issue(rows, "range", "popularity").n_bad == 0 and _issue(rows, "zero", "popularity").n_bad == 1
    orows = ta.check_result_rows(o, "race_odds_flat")
    assert _issue(orows, "range", "win_odds").n_bad == 0 and _issue(orows, "zero", "win_odds").n_bad == 1
    assert _issue(orows, "range", "popularity").n_bad == 0 and _issue(orows, "shift", "popularity/win_odds").n_bad == 0


def test_obstacle_last_3f_and_long_distance_are_not_out_of_range():
    df = _race(race_name="中山グランドジャンプ", surface="障", distance=4260, last_3f=14.2, time_sec=290.0)
    rows = ta.check_result_rows(df)
    assert _issue(rows, "range", "last_3f").n_bad == 0 and _issue(rows, "range", "distance").n_bad == 0
    assert _issue(ta.check_result_races(df), "shift", "列ずれのレース").n_bad == 0


def test_future_and_pre_draw_rows_are_excluded():
    df = _race(n=10, date="2026-12-20")
    df["horse_number"] = 0
    df["bracket_number"] = 0
    df["popularity"] = 0
    df["field_size"] = 53
    rows = ta.check_result_rows(df, "race_shutuba_flat", today=pd.Timestamp("2026-11-16").date())
    assert all(i.n_bad == 0 for i in rows)
    # 枠順の発表後でも、日付が今日より後なら外す
    df2 = _race(n=10, date="2026-12-20")
    df2["field_size"] = 53
    assert _issue(ta.check_result_rows(df2, today=pd.Timestamp("2026-11-16").date()), "range", "field_size").n_total == 0


def test_compare_entries_handles_type_mismatch_and_duplicates():
    r = _race()
    s = _race()
    s["horse_id"] = s["horse_id"].astype(np.int64)        # 片方が int
    s = pd.concat([s, s.iloc[:1]], ignore_index=True)       # 重複行
    s.loc[3, "horse_number"] = 99
    out = {i.column: (i.n_bad, i.n_total) for i in ta.compare_entries(r, s)}
    assert out["馬番の不一致"] == (1, 10) and out["結果にだけある馬"][0] == 0


def test_category_race_id_does_not_inflate_totals():
    df = _race()
    df["race_id"] = pd.Categorical(df["race_id"], categories=["202501010101", "UNUSED"])
    races = ta.check_result_races(df)
    assert _issue(races, "race_const", "distance").n_total == 1


def test_detect_jumps_catches_recurrence_when_last_year_was_also_broken():
    p = _profile({"2025-09": 1.0, "2025-10": 40.0, "2026-09": 1.0, "2026-10": 40.0})
    assert set(ta.detect_jumps(p)["month"]) == {"2025-10", "2026-10"}


def test_detect_jumps_zero_metric_and_current_month_excluded():
    p = pd.DataFrame([{"month": m, "column": "odds", "rows": 500, "missing_pct": 0.0, "zero_pct": z}
                      for m, z in (("2026-09", 0.5), ("2026-10", 30.0), ("2026-11", 60.0))])
    assert ta.detect_jumps(p, metric="zero_pct", before="2026-11")["month"].tolist() == ["2026-10"]


def test_on_time_table_without_payoff_is_not_flagged():
    df = _race()
    df["payoff"] = ""
    assert not [i for i in ta.check_result_races(df, "race_result_on_time_flat") if i.check == "payoff"]


def test_proper_dead_heat_payout_list_is_ok():
    df = _race()
    df.loc[1, "finish_position"] = 1
    w6 = int(round(df.loc[5, "odds"] * 100))
    df["payoff"] = json.dumps({"単勝": [{"numbers": "6", "payout": str(w6 // 2)}, {"numbers": "2", "payout": "700"}]})
    assert _issue(ta.check_result_races(df), "payoff", "odds*100==単勝払戻").n_bad == 0


def test_cli_train_serve_reports_columns_missing_on_one_side(tmp_path, capsys):
    pd.DataFrame({"a": [1.0, 2.0], "only_train": [1, 2]}).to_parquet(tmp_path / "train.parquet")
    pd.DataFrame({"a": [np.nan, np.nan]}).to_parquet(tmp_path / "s1.parquet")
    assert ta.main(["--train", str(tmp_path / "train.parquet"), "--serve", str(tmp_path / "s*.parquet"), "--out", str(tmp_path / "o")]) == 0
    d = pd.read_csv(tmp_path / "o" / "train_vs_serve_missing.csv")
    assert set(d["column"]) == {"a", "only_train"} and d.set_index("column").loc["only_train", "only"] == "train"
    with pytest.raises(SystemExit):
        ta.main(["--train", str(tmp_path / "train.parquet")])
    with pytest.raises(SystemExit):
        ta.main(["--train", str(tmp_path / "train.parquet"), "--serve", str(tmp_path / "none*.parquet")])


def test_parse_years_ignores_trailing_comma():
    assert ta.parse_years("2016,") == [2016]


def test_young_jockeys_series_is_not_treated_as_obstacle():
    df = _race(race_name="ヤングJSFR中山1(1勝)", surface="芝")
    assert _issue(ta.check_result_races(df), "race_const", "障害の surface").n_total == 0


# ── 値の有無による漏れ（生き残りの漏れ） ─────────────────────────────────

def test_presence_leak_flags_columns_whose_presence_predicts_the_target():
    rng = np.random.default_rng(0)
    n = 4000
    year = np.where(np.arange(n) < n // 2, "2018", "2022")
    win = (rng.random(n) < 0.08).astype(float)
    # 2018: 勝った馬ほど記録がある（後の年も走った馬にだけ記録）。2022: 有無は勝ちと無関係
    p_has = np.where(year == "2018", np.where(win > 0, 0.8, 0.3), 0.9)
    leaky = np.where(rng.random(n) < p_has, 1.0, np.nan)
    fair = np.where(rng.random(n) < 0.5, 1.0, np.nan)
    df = pd.DataFrame({"leaky": leaky, "fair": fair})
    out = ta.presence_leak(df, target=pd.Series(win), year=pd.Series(year))
    assert set(out["column"]) == {"leaky"} and out["year"].tolist() == ["2018"] and out["auc"].iloc[0] > 0.6


def test_auc_matches_simple_cases():
    assert ta._auc(np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 1.0, 0.0])) == 1.0
    assert ta._auc(np.array([1, 0, 1, 0]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5


def test_cli_leak_writes_csv(tmp_path):
    rng = np.random.default_rng(1)
    n = 3000
    fp = rng.integers(1, 12, n)
    t = pd.DataFrame({"race_id": np.arange(n).astype(str), "horse_id": "h", "date": "2018-05-01", "finish_position": fp,
                      "sire": np.where((fp == 1) | (rng.random(n) < 0.3), "s1", "")})
    t.to_parquet(tmp_path / "t.parquet")
    assert ta.main(["--leak", str(tmp_path / "t.parquet"), "--out", str(tmp_path / "o")]) == 0
    d = pd.read_csv(tmp_path / "o" / "presence_leak.csv")
    assert d["column"].tolist() == ["sire"] and d["auc"].iloc[0] > 0.6


def test_presence_leak_handles_nan_target_and_year_low_auc_share_and_placeholders():
    rng = np.random.default_rng(2)
    n = 3000
    win = (rng.random(n) < 0.1).astype(float)
    win[:50] = np.nan                                   # 取消など
    year = np.array(["2017"] * n, dtype=object)
    year[50:80] = None
    # 値がある方が勝てない（AUC < 0.5）向きも拾う。__UNKNOWN__ は欠けとみなす
    col = np.where((win == 1) | (rng.random(n) < 0.3), "__UNKNOWN__", "S")
    always = np.ones(n)                                 # 値がある割合が範囲外 → 判定しない
    out = ta.presence_leak(pd.DataFrame({"sire": col, "always": always}), target=pd.Series(win), year=pd.Series(year))
    assert out["column"].tolist() == ["sire"] and out["auc"].iloc[0] < 0.4
    assert np.isnan(ta._auc(np.array([1, 1]), np.array([0.0, 1.0])))


def test_presence_leak_reports_other_years_for_context():
    rng = np.random.default_rng(3)
    n = 6000
    year = np.repeat(["2018", "2021", "2022"], n // 3)
    win = (rng.random(n) < 0.1).astype(float)
    p = np.where(year == "2018", np.where(win > 0, 0.85, 0.3), np.where(win > 0, 0.55, 0.5))
    col = np.where(rng.random(n) < p, 1.0, np.nan)
    out = ta.presence_leak(pd.DataFrame({"c": col}), target=pd.Series(win), year=pd.Series(year))
    r = out.set_index("year").loc["2018"]
    assert r["vs_other_years"] > 0.1 and out["year"].iloc[0] == "2018"


def test_cli_leak_cannot_be_combined_with_train_serve(tmp_path):
    with pytest.raises(SystemExit):
        ta.main(["--leak", str(tmp_path / "t.parquet"), "--train", "a", "--serve", "b"])
