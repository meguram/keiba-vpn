"""過去走の市場指標（odds_prev_market_feat）が、その日より前の値だけで作られること。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.data import odds_prev_market_feat as f


def _tables(tmp_path):
    rows = []
    for d, rid in enumerate(["r1", "r2", "r3"]):
        for h, (odds, pop, fin) in enumerate([(2.0, 1, 2), (4.0, 2, 1), (8.0, 3, 3), (16.0, 4, 4)]):
            rows.append({"race_id": rid, "horse_id": f"h{h}", "date": pd.Timestamp("2025-01-04") + pd.Timedelta(days=7 * d),
                         "jockey_id": "j0" if h < 2 else "j1", "trainer_id": "t0", "finish_position": fin, "win_odds": odds * (1 + d), "popularity": pop})
    df = pd.DataFrame(rows)
    p = tmp_path / "tt.parquet"
    df.drop(columns=["win_odds", "popularity"]).to_parquet(p)
    return p, df[["race_id", "horse_id", "win_odds", "popularity"]]


def test_features_use_only_earlier_runs(tmp_path) -> None:
    p, wp = _tables(tmp_path)
    out = f.build(p, wp).set_index(["race_id", "horse_id"])
    assert out.loc[("r1", "h0")].drop(["pa_mk_jk_lo", "pa_mk_jk_rkr", "pa_mk_tr_lo", "pa_mk_tr_rkr"]).isna().all()
    assert np.isclose(out.loc[("r2", "h1"), "pa_mk_lo1"], np.log(4.0))
    assert np.isclose(out.loc[("r3", "h1"), "pa_mk_lo1"], np.log(8.0))
    assert np.isclose(out.loc[("r3", "h1"), "pa_mk_lo_m3"], (np.log(4.0) + np.log(8.0)) / 2)
    assert np.isclose(out.loc[("r2", "h0"), "pa_mk_bt1"], 0.0 - 1 / 3)      # 1番人気で2着: 人気比0 − 着順比1/3
    assert np.isclose(out.loc[("r3", "h2"), "pa_mk_lo1_gap"], np.log(16.0) - np.log(4.0))   # r2 の 1 番人気は 4.0 倍
    # 騎手・調教師は r1 の同じ日の走を含めない（r1 は全馬が縮め先の全体平均）
    m = float(np.log(wp["win_odds"]).mean())
    assert np.isclose(out.loc[("r1", "h0"), "pa_mk_jk_lo"], m)
    assert not np.isclose(out.loc[("r2", "h0"), "pa_mk_jk_lo"], m)
