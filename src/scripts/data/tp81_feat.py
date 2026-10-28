"""tp-81: 展開利を特徴量として親に足す版の群（v6_eval の外部群 TENKAI / TENKAI_NT）を書き出す。全年（2020〜2025）で作る。

  python -m src.scripts.data.tp81_feat   # ptrain/tenkai_rows.parquet（追走難度 EASE2 つき）と ptrain/tenkai_nt_rows.parquet（つけない）

列: 想定位置(前段ML c1 予測の順位を 0〜1)、想定ペース(u_pace_hat)、その掛け合わせ（前×速い、後ろ×速い、前×遅い）、
要求量 demand_idx(=(1-位置)×(-ペース)×(距離km-0.6))、位置の脚質区分×ペースの積、レース内の相対位置。追走難度版は EASE2_drain / demand / pct を足す。
通過順・ペースは race_result_flat 由来（tp80 の入力）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import tp80_userpos as U
from src.scripts.data import tp80_eval as T80

ROOT = T80.ROOT
OUT = ROOT / "data/local/modeling/ptrain"
KEY = ["race_id", "horse_id"]


def build() -> None:
    D, _career = T80.load()
    D["horse_id"] = D["horse_id"].astype(str)
    u = pd.read_parquet(OUT / "usr_rows.parquet", columns=KEY + ["u_pace_hat"])
    u["race_id"], u["horse_id"] = u["race_id"].astype(str), u["horse_id"].astype(str)
    X = D[KEY + ["horse_number", "field_size", "distance", "ts_c1_pred"]].merge(u, on=KEY, how="left")
    rk = U.order_in_race(X["race_id"].to_numpy(), X["ts_c1_pred"].to_numpy(dtype="float64"), X["horse_number"].to_numpy())
    n = X["field_size"].to_numpy(dtype="float64")
    ph = np.where(n > 1, (rk - 1) / np.maximum(n - 1, 1), 0.5)
    pace = X["u_pace_hat"].to_numpy(dtype="float64")            # 小さい = 速いペース（前半タイムの z）
    dk = X["distance"].to_numpy(dtype="float64") / 1000.0
    f = pd.DataFrame({"race_id": X["race_id"], "horse_id": X["horse_id"]})
    f["tk_pos"] = ph
    f["tk_pace"] = pace
    f["tk_front_fast"] = (1 - ph) * (-pace)
    f["tk_back_fast"] = ph * (-pace)
    f["tk_front_slow"] = (1 - ph) * pace
    f["tk_demand"] = (1 - ph) * (-pace) * (dk - 0.6)
    q = np.digitize(ph, [0.2, 0.45, 0.75]).astype(float)
    f["tk_style_x_pace"] = q * 0.0 + (q - 1.5) * (-pace)
    f["tk_pos_rank"] = rk
    f["tk_pos_rel"] = ph - pd.Series(ph).groupby(X["race_id"].to_numpy()).transform("mean").to_numpy()
    f.drop(columns=[]).to_parquet(OUT / "tenkai_nt_rows.parquet", index=False)
    g = pd.read_parquet(ROOT / "data/local/modeling/tp54/groups/EASE2.parquet", columns=KEY + ["EASE2_drain", "EASE2_demand", "EASE2_pct"])
    g["race_id"], g["horse_id"] = g["race_id"].astype(str), g["horse_id"].astype(str)
    f2 = f.merge(g, on=KEY, how="left")
    f2.to_parquet(OUT / "tenkai_rows.parquet", index=False)
    print("wrote", len(f), f.shape[1], f2.shape[1])


if __name__ == "__main__":
    build()
