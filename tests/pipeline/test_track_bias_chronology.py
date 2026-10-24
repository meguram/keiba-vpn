"""馬場傾向「当日の直前傾向」が時刻順(先読みなし)で計算されることの回帰テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features.track_bias_pedigree import build_track_bias_pedigree_features


def _day():
    # 同じ日・同じ場・同じ芝。時刻順は r1 → r2 → r3。
    # 馬場(良)と距離帯が違う（r2 は middle、r1/r3 は mile）ので、細分類のブロック順(middle が mile より先)と時刻順が食い違う。
    races = [
        ("r1", "09:55", 1600, 35.0),
        ("r2", "10:30", 2000, 36.0),
        ("r3", "11:05", 1600, 37.0),
    ]
    rr, sb = [], []
    for rid, st, dist, w3f in races:
        for hn, fp, l3f in ((1, 1, w3f), (2, 2, w3f + 1.0)):
            rr.append({"race_id": rid, "horse_number": hn, "date": "2026-01-10", "venue_code": "05", "surface": "芝",
                       "start_time": st, "finish_position": fp, "last_3f": l3f, "track_condition": "良", "distance": dist})
            sb.append({"race_id": rid, "horse_number": hn, "sire": "A", "dam_sire": "B"})
    return pd.DataFrame(rr), pd.DataFrame(sb)


def test_same_day_prior_uses_only_earlier_races():
    rr, sb = _day()
    out = build_track_bias_pedigree_features(rr, sb)
    v = out.drop_duplicates("race_id").set_index("race_id")["track_bias_winner_3f_same_day_prior"]
    assert np.isnan(v["r1"]), "時刻が最初のレースに値がある（後続レースの結果を使っている）"
    assert abs(v["r2"] - 35.0) < 1e-6, "r2 の直前傾向は r1 の勝ち馬だけの平均のはず"
    assert abs(v["r3"] - 35.0) < 1e-6, "r3 は同じ細分類の r1 の勝ち馬（細分類を優先）"


def test_hurdle_winner_labeled_as_turf_does_not_pollute_flat_average():
    """面が『芝』と保存された障害戦（勝ち馬の上がり 約 11 秒）が、同日・同場の平地の直前傾向に混ざらない。"""
    rr, sb = [], []
    races = [("h1", "09:55", 3000, 11.5), ("f1", "10:30", 2000, 35.0), ("f2", "11:05", 1600, 36.0)]
    for rid, st, dist, w3f in races:
        for hn, fp, l3f in ((1, 1, w3f), (2, 2, w3f + (0.5 if rid == "h1" else 1.0))):
            rr.append({"race_id": rid, "horse_number": hn, "date": "2026-01-10", "venue_code": "05", "surface": "芝",
                       "start_time": st, "finish_position": fp, "last_3f": l3f, "track_condition": "良", "distance": dist})
            sb.append({"race_id": rid, "horse_number": hn, "sire": "A", "dam_sire": "B"})
    out = build_track_bias_pedigree_features(pd.DataFrame(rr), pd.DataFrame(sb))
    v = out.drop_duplicates("race_id").set_index("race_id")["track_bias_winner_3f_same_day_prior"]
    assert np.isnan(v["f1"]), "平地の先頭レースに、先行する障害戦の勝ち馬の上がりが混ざっている"
    assert abs(v["f2"] - 35.0) < 1e-6, "f2 の直前傾向は平地 f1 の勝ち馬だけの平均のはず"


def test_zero_last_3f_winner_is_ignored():
    rr, sb = _day()
    rr.loc[(rr.race_id == "r1") & (rr.horse_number == 1), "last_3f"] = 0.0      # 解析不良の勝ち馬
    out = build_track_bias_pedigree_features(rr, sb)
    v = out.drop_duplicates("race_id").set_index("race_id")["track_bias_winner_3f_same_day_prior"]
    assert np.isnan(v["r2"]) or v["r2"] > 30, "上がり 0 の勝ち馬が平均に入っている"
