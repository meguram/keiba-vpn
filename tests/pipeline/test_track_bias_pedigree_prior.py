"""父・母父×面の累計勝率（track_bias_pedigree._prior_win_rate_by_id_surface）が、当該レースを丸ごと除いて数えることのテスト。

以前は行の順（同じレースでは馬番の順）で「前」を数えていたため、同じレースに同じ父の馬がいると、馬番の若い馬の結果が
後ろの馬の値に入っていた（データ監査 2026-11-16、docs/operations/data_audit_20261116.md）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features.track_bias_pedigree import _prior_win_rate_by_id_surface


def _work(rows):
    df = pd.DataFrame(rows, columns=["race_id", "horse_number", "sire", "date", "_st_sec", "finish_position"])
    df["_d"] = pd.to_datetime(df["date"])
    df["surface_group"] = "turf"
    return df


def _rate(out, rid, hn):
    return out.set_index(["race_id", "horse_number"]).loc[(rid, hn), "sire_prior_win_rate_surface"]


def test_same_race_siblings_do_not_count_each_other():
    w = _work([
        ("R1", 1, "S", "2025-01-05", 600, 1),    # 前のレース: 同じ父が 1 勝・1 敗
        ("R1", 2, "S", "2025-01-05", 600, 5),
        ("R2", 1, "S", "2025-01-12", 600, 1),    # 当該レース: 馬番 1 が勝ち
        ("R2", 3, "S", "2025-01-12", 600, 4),
    ])
    out = _prior_win_rate_by_id_surface(w, "sire", "sire")
    assert _rate(out, "R2", 1) == 0.5 and _rate(out, "R2", 3) == 0.5        # 同じレースの馬番 1 の勝ちは数えない
    assert np.isnan(_rate(out, "R1", 1)) and np.isnan(_rate(out, "R1", 2))  # 最初のレースは前が無い


def test_unknown_sire_group_also_excludes_the_current_race():
    w = _work([
        ("R1", 1, "", "2025-01-05", 600, 2),
        ("R2", 1, None, "2025-01-12", 600, 1),
        ("R2", 2, "", "2025-01-12", 600, 3),
        ("R2", 3, "", "2025-01-12", 600, 6),
    ])
    out = _prior_win_rate_by_id_surface(w, "sire", "sire")
    assert [_rate(out, "R2", h) for h in (1, 2, 3)] == [0.0, 0.0, 0.0]


def test_earlier_race_on_the_same_day_counts_and_pending_rows_do_not():
    w = _work([
        ("R1", 1, "S", "2025-01-05", 600, 1),     # 同じ日の前のレース（発走が早い）は数える
        ("R2", 1, "S", "2025-01-05", 900, np.nan),
        ("R2", 2, "S", "2025-01-05", 900, np.nan),
        ("R3", 1, "S", "2025-01-05", 1200, np.nan),
    ])
    w["_pending"] = w["finish_position"].isna()
    out = _prior_win_rate_by_id_surface(w, "sire", "sire")
    assert _rate(out, "R2", 1) == 1.0 and _rate(out, "R3", 1) == 1.0         # 未確定のレースは件数にも勝ちにも入らない


def test_mixed_start_time_within_a_race_is_still_one_race():
    w = _work([
        ("R1", 1, "S", "2025-01-05", 600, 1),
        ("R2", 1, "S", "2025-01-12", 600, 5),
        ("R2", 2, "S", "2025-01-12", np.nan, 1),     # 同じレースで発走時刻が一部欠け
    ])
    out = _prior_win_rate_by_id_surface(w, "sire", "sire")
    assert len(out) == 3
    assert _rate(out, "R2", 1) == 1.0 and _rate(out, "R2", 2) == 1.0          # 同じレースの勝ちは数えない
