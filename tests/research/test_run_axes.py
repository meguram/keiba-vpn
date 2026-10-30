"""実測ラップの走法軸（src/research/race/run_axes.py）。合成データで、欠測・異常走・対象コースの扱いを確かめる。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.race import run_axes as ra

N_RACES, N_HORSES = 30, 12


def _frame(seed: int = 0, venue_of=lambda r: "東京", with_pos: bool = True) -> pd.DataFrame:
    """芝 1600m・8 区間の合成データ。ts・offset・mu_s・sd_s は with_ts の出力を模して付ける。"""
    rng = np.random.default_rng(seed)
    rows = []
    dist = np.arange(1, 9) * 200.0
    for r in range(N_RACES):
        pos_order = rng.permutation(N_HORSES) + 1
        for h in range(N_HORSES):
            lt = 12.0 + rng.normal(0, 0.25, 8)
            lt[-3:] -= rng.uniform(0.2, 0.9)                       # 終盤は速い
            pos = np.where(with_pos, np.clip(pos_order[h] + rng.integers(-2, 3, 8), 1, N_HORSES), 0).astype(float)
            rows.append({
                "race_id": f"R{r:03d}", "horse_id": f"H{r:03d}_{h:02d}", "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=r),
                "venue": venue_of(r), "surface": "芝", "distance": 1600, "track_condition": "良", "finish_order": h + 1, "nent": N_HORSES,
                "lt": lt, "dist": dist, "pc": lt.copy(), "cum": np.cumsum(lt), "pos": pos, "len1": 200.0, "i3": 5, "i4": 6,
                "offset": 0.0, "mu_s": 11.4, "sd_s": 0.3, "ts": 50 - 10 * (lt[-3:].min() - 11.4) / 0.3,
            })
    return pd.DataFrame(rows)


def test_all_axes_are_produced_and_missing_stays_nan():
    t = _frame()
    out = ra.axes_from_frame(t)
    assert len(out) == len(t) and list(out["race_id"]) == list(t["race_id"])
    for c in ["sustain", "split", "debut", "gear", "endure", "front", "cr", "makuri"]:
        assert out[c].notna().all(), c
    assert out["slope"].isna().all()                                 # 東京は急坂適性の対象外（0 ではなく欠測）
    assert not out["abn"].any()
    for c in ["debut", "gear", "endure", "front", "cr", "makuri"]:   # 50 + 10z なので平均はほぼ 50
        assert abs(out[c].mean() - 50) < 3, c


def test_slope_only_for_nakayama_and_hanshin():
    out = ra.axes_from_frame(_frame(venue_of=lambda r: ["中山", "阪神", "東京"][r % 3]))
    venues = _frame(venue_of=lambda r: ["中山", "阪神", "東京"][r % 3])["venue"].to_numpy()
    assert out.loc[venues == "東京", "slope"].isna().all()
    assert out.loc[venues != "東京", "slope"].notna().all()


def test_missing_positions_make_position_axes_nan_not_zero():
    out = ra.axes_from_frame(_frame(with_pos=False))
    for c in ["front", "cr", "makuri"]:
        assert out[c].isna().all(), c
    assert out["endure"].isna().all()                                # 消耗耐性は通過位置を共変量に使うので、位置が無い走は欠測
    for c in ["sustain", "split", "debut", "gear"]:                  # 位置を使わない軸は出る
        assert out[c].notna().all(), c


def test_abnormal_run_is_flagged_but_value_kept():
    t = _frame()
    lt = t.at[0, "lt"].copy()
    lt[-1] = 25.0                                                    # ラスト 1F が 20 秒/200m 超
    t.at[0, "lt"], t.at[0, "pc"] = lt, lt.copy()
    out = ra.axes_from_frame(t)
    assert out.loc[0, "abn"] and not out.loc[1:, "abn"].any()
    assert np.isfinite(out.loc[0, "endure"])                         # 印だけで値は消さない


def test_read_runs_skips_incomplete_entries(tmp_path):
    sec = lambda d, lap, corner="", pos=1: {"distance_m": d, "lap_time": lap, "corner": corner, "position": pos}  # noqa: E731
    good = [sec(200 * (i + 1), 12.0, "3C" if i == 4 else "4C" if i == 5 else "", i + 1) for i in range(8)]
    race = {"race_id": 202601010101, "entries": [
        {"horse_id": "A", "horse_name": "a", "finish_order": 1, "sections": good},
        {"horse_id": "B", "horse_name": "b", "finish_order": 0, "sections": good},                 # 着順なし
        {"horse_id": "C", "horse_name": "c", "finish_order": 2, "sections": good[:5]},             # 区間不足
        {"horse_id": "D", "horse_name": "d", "finish_order": 3, "sections": good[:7] + [sec(1600, 0.0)]},   # 0 秒の区間
    ]}
    f = tmp_path / "lap.jsonl"
    f.write_text(json.dumps(race) + "\n", encoding="utf-8")
    runs = ra.read_runs([f])
    assert list(runs["horse_id"]) == ["A"]
    assert runs.at[0, "i3"] == 4 and runs.at[0, "i4"] == 5 and runs.at[0, "nent"] == 4
