"""系統・種牡馬の成績表（sire_line_stats）: 走ごとの指標・配当の読み取り・概算賞金・休み明け・群の集計と除外。"""

import json

import numpy as np
import pandas as pd
import pytest

from src.research.pedigree import sire_line_stats as T


def _raw():
    """レース r1（芝1600・良・8 頭）と r2（ダ1200・重・8 頭、r1 の 100 日後）。馬 a は r1 で 1 着・r2 で 3 着、b は r1 で 2 着、c は r1 で 8 着。"""
    pay1 = json.dumps({"単勝": {"numbers": "1", "payout": "350"},
                       "複勝": [{"numbers": "1", "payout": "150"}, {"numbers": "2", "payout": "1,200"}, {"numbers": "5", "payout": "300"}]})
    pay2 = json.dumps({"複勝": [{"numbers": "4", "payout": "500"}]})
    rows = []

    def add(rid, date, sf, dist, cond, grade, h, pos, hn, br, po, l3, odds, popu, pay, age="牡3", w=480, wc=0):
        rows.append({"race_id": rid, "date": date, "horse_id": h, "surface": sf, "distance": dist, "direction": "右", "track_condition": cond,
                     "venue": "中山", "grade": grade, "finish_position": pos, "field_size": 8, "passing_order": po, "last_3f": l3,
                     "sex_age": age, "odds": odds, "popularity": popu, "bracket_number": br, "horse_number": hn, "weight": w,
                     "weight_change": wc, "payoff": pay})
    add("r1", "2024-01-06", "芝", 1600, "良", "2勝", "a", 1, 1, 1, "1-1", 34.0, 3.5, 2, pay1)
    add("r1", "2024-01-06", "芝", 1600, "良", "2勝", "b", 2, 2, 2, "5-3", 33.5, 10.0, 5, pay1)
    add("r1", "2024-01-06", "芝", 1600, "良", "2勝", "c", 8, 8, 8, "8-8", 36.0, 50.0, 8, pay1)
    for i, (h, hn) in enumerate((("x1", 3), ("x2", 4), ("x3", 5), ("x4", 6), ("x5", 7))):
        add("r1", "2024-01-06", "芝", 1600, "良", "2勝", h, 3 + i, hn, hn, "4-4", 35.0, 20.0, 3 + i, pay1)
    add("r2", "2024-04-15", "ダート", 1200, "重", "1勝", "a", 3, 4, 3, "2-2", 37.0, 5.0, 1, pay2, age="牡4", wc=12)
    for i in range(7):
        add("r2", "2024-04-15", "ダート", 1200, "重", "1勝", f"y{i}", 1 + i if i < 2 else 4 + i - 2, 10 + i, 1 + i, "5-5", 38.0, 9.0, 2 + i, pay2)
    return pd.DataFrame(rows)


def test_fuku_payouts_reads_list_and_commas():
    fp = T.fuku_payouts(_raw().drop_duplicates("race_id").set_index("race_id").payoff)
    assert fp[("r1", 1)] == 150 and fp[("r1", 2)] == 1200 and fp[("r2", 4)] == 500


def test_run_table_metrics():
    r = T.run_table(_raw()).set_index(["race_id", "horse_id"])
    a1, b1, c1, a2 = r.loc[("r1", "a")], r.loc[("r1", "b")], r.loc[("r1", "c")], r.loc[("r2", "a")]
    assert a1.win == 1 and a1.tan_ret == pytest.approx(3.5) and a1.fuku_ret == pytest.approx(1.5)
    assert b1.win == 0 and b1.tan_ret == 0 and b1.fuku_ret == pytest.approx(12.0)
    assert c1.fuku_ret == 0 and c1.top3 == 0
    assert a2.fuku_ret == pytest.approx(5.0)                                # r2 の 3 着（4 番）は配当に載っている
    assert r.loc[("r2", "y0")].top3 == 1 and r.loc[("r2", "y0")].fuku_ret == 0   # 1 着だが配当に無い（配当の載らない馬）→ 外れの 0、NaN にしない
    assert a1.prize == 1100 and b1.prize == 440 and c1.prize == 0          # 2 勝クラスの概算表
    assert a1.nige == 1 and a1.pos1 == 0 and c1.pos1 == 1 and c1.back3 == 1
    assert b1.push == pytest.approx((4 - 2) / 7)                            # 初角 5 番手 → 最終角 3 番手
    assert b1.l3_best == 1 and b1.l3_rank == 0 and a1.l3_top3 == 1
    assert np.isnan(a1.top3_d) and a1.top3_t == 1 and a1["top3_t_d1"] == 1  # 1600m は 1500〜1800 の帯
    assert a1.top3_w1 == 1 and a1.top3_in == 1 and np.isnan(a1.top3_out)
    assert np.isnan(a1.top3_rest) and a2.top3_rest == 1                    # 初出走は間隔なし、100 日後は休み明け
    assert a2.top3_d_h == 1 and a2.top3_a4 == 1 and a2.top3_wup == 1


def test_aggregate_and_thresholds():
    r = T.run_table(_raw())
    hl = T.horse_level(r)
    assert hl.loc["a", "h_prize"] == 1100 + 175 and hl.loc["a", "h_2win"] == 1 and hl.loc["y0", "h_2win"] == 0
    key = r.horse_id.map(lambda h: "甲" if h in ("a", "b", "c") else "乙")
    agg = T.aggregate(r, key, hl, 0, 0)
    assert agg.loc["甲", "horses"] == 3 and agg.loc["甲", "runs"] == 4
    assert agg.loc["甲", "win"] == pytest.approx(1 / 4)
    assert agg.loc["甲", "top3_t"] == pytest.approx(2 / 3) and agg.loc["甲", "n_top3_t"] == 3   # 芝の走だけの複勝率と走数
    assert agg.loc["甲", "h_prize"] == pytest.approx((1275 + 440 + 0) / 3) and agg.loc["甲", "n_h_prize"] == 3
    assert T.aggregate(r, key, hl, 4, 0).index.tolist() == ["乙"]           # 3 頭の甲は最少頭数で外れる
    row = T.pack(agg)[0]
    assert len(row[3]) == len(T.METRICS) == len(row[4])


def test_metric_keys_unique_and_grouped():
    keys = [m for m, *_ in T.METRICS]
    assert len(keys) == len(set(keys))
    assert {g for _, _, g, _ in T.METRICS} == set(T.GROUPS)


def test_field_counts_only_finishers_and_gap_uses_all_runs():
    raw = _raw()
    # r1 に取消の行を足す（着順なし）。頭数は 8 のまま。障害の走を a に足す（r1 と r2 の間）→ r2 は休み明けではなくなる
    extra = raw.iloc[[0]].copy()
    extra["horse_id"], extra["finish_position"], extra["horse_number"] = "zz", None, 99
    jump = raw[(raw.race_id == "r2") & (raw.horse_id == "a")].copy()
    jump["race_id"], jump["date"], jump["surface"] = "j1", "2024-03-20", "障"
    r = T.run_table(pd.concat([raw, extra, jump], ignore_index=True)).set_index(["race_id", "horse_id"])
    assert r.loc[("r1", "c")].pos1 == 1 and r.loc[("r1", "c")].pct == 0       # 8 頭立ての最下位（取消の行で頭数が増えない）
    assert np.isnan(r.loc[("r2", "a")].top3_rest) and r.loc[("r2", "a")].top3_norest == 1   # 前走は障害（26 日前）


def test_ts_ceiling_once_per_horse_with_min_runs():
    runs = pd.DataFrame({"race_id": [f"r{i}" for i in range(5)], "horse_id": ["h", "h", "h", "g", "g"], "grade": "2勝"})
    ax = pd.DataFrame({"race_id": runs.race_id, "horse_id": runs.horse_id, "surface": "芝", "distance": 1600, "venue": "東京",
                       **{c: [1.0, 2.0, 3.0, 0.0, 1.0] for c in T.SD.AXIS_NAMES}})
    out = T.axis_runs(ax, runs)
    ceil = out.dropna(subset=["ax_TS上限"])
    assert list(ceil.horse_id) == ["h"]                                     # g は 2 走しかないので無し、h は 1 回だけ


def test_prev_gap_skips_scratched_rows_without_status():
    """状態の列が無い年: 着順 -1 で通過順が空の行は取消・除外（走っていない）、通過順があれば中止（走った）"""
    raw = pd.DataFrame({"race_id": ["a", "b", "c", "d"], "horse_id": "h", "date": ["2024-01-01", "2024-03-01", "2024-03-20", "2024-04-01"],
                        "finish_position": [1, -1, -1, 2], "passing_order": ["1-1", "", "3-5", "2-2"]})
    g = T.prev_gap_days(raw)
    assert ("b", "h") not in g.index and ("c", "h") in g.index      # b は取消、c は中止
    assert g[("d", "h")] == 12 and g[("c", "h")] == 79


def test_populations_all_top_horses_and_top_races():
    r = pd.DataFrame({"horse_id": ["a", "a", "b", "c", "c"], "grade": ["未勝利", "2勝", "1勝", "G1", "L"]})
    p = T.populations(r)
    assert p["all"].tolist() == [True] * 5
    assert p["top_h"].tolist() == [True, True, False, True, True]     # a は 2 勝クラスを走ったので未勝利の走も入る。b は入らない
    assert p["top_r"].tolist() == [False, True, False, True, True]    # 2 勝クラス以上のレースの走だけ（L も入る）
