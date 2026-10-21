"""tp-58 手設計の数値の掛け合わせ（距離変化×前走着順、斤量×馬体重×距離、枠×脚質×頭数、年齢×間隔 など）を V5 の OOF へのスコア検定で見る。

python -m src.scripts.data.tp58_named
各候補は積/差/対数比。有効スコア（元の 2 列の一次項を除いたもの）も併記。発見期間 2022–23・確認期間 2024。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp58_pairs as pp
from src.pipeline.features.tp58_score import ScoreTest

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("TP58_DIR", ROOT / "data/local/modeling/shared_cache/tp58"))
CACHE, OUT = WORKDIR / "cache", WORKDIR / "out"
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
DISC, CONF = (2022, 2023), (2024,)

NAMED = [
    ("prod", "chg_dist", "d_prev_fin_pct"), ("prod", "chg_dist_up", "d_prev_fin_pct"), ("prod", "chg_dist_down", "d_prev_fin_pct"),
    ("prod", "chg_dist", "d_prev_fin"), ("prod", "distance", "weight_change"), ("prod", "distance", "weight"),
    ("prod", "distance", "jockey_weight"), ("logratio", "jockey_weight", "weight"), ("prod", "age", "rest_log"),
    ("prod", "age", "weight_change"), ("prod", "weight_change", "chg_dist"), ("prod", "a_style", "bracket_number"),
    ("prod", "a_style", "field_size"), ("prod", "mtl_t_pos_early", "bracket_number"), ("prod", "mtl_t_pos_early", "field_size"),
    ("prod", "mtl_t_pos_early", "distance"), ("prod", "mtl_t_pos_late", "distance"), ("prod", "rest_log", "mtl_t_rank_norm_gap"),
    ("prod", "age", "distance"), ("prod", "weight", "jockey_weight"), ("prod", "d_wt_chg_now", "distance"),
    ("prod", "d_wt_chg_now", "age"), ("prod", "rest_log", "d_wt_chg_now"), ("prod", "a_style", "distance"),
    ("prod", "bracket_number", "field_size"), ("prod", "chg_field", "a_style"), ("prod", "chg_dist", "a_style"),
    ("prod", "rest_log", "d_prev_fin_pct"), ("prod", "age", "d_prev_fin_pct"), ("prod", "jockey_weight", "age"),
]


def main() -> int:
    cols = json.loads((CACHE / "columns.json").read_text(encoding="utf-8"))["num"]
    need = sorted({c for _, a, b in NAMED for c in (a, b)} & set(cols))
    F = pd.read_parquet(CACHE / "v5_frame_num.parquet", columns=KEY + ["finish_position", "year", "date"] + need)
    F["race_id"], F["horse_id"] = F["race_id"].astype(str), F["horse_id"].astype(str)
    O = pd.read_parquet(OOF)
    O["race_id"], O["horse_id"] = O["race_id"].astype(str), O["horse_id"].astype(str)
    consts = pp.fit_constants(F, need, (F["year"] <= 2021).to_numpy())
    D = F[F["year"].isin(DISC + CONF)].merge(O[KEY + ["p_win", "p_top2", "p_top3"]], on=KEY, how="inner").sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    year = D["year"].to_numpy()
    periods = {"disc": np.isin(year, DISC), "conf": np.isin(year, CONF)}
    fp = pd.to_numeric(D["finish_position"], errors="coerce").to_numpy()
    races = D["race_id"].to_numpy()
    tests = {
        "win": ScoreTest((fp == 1).astype(float), D["p_win"].to_numpy(), races, periods, "win"),
        "top2": ScoreTest((fp <= 2).astype(float), D["p_top2"].to_numpy(), races, periods, "binary"),
        "top3": ScoreTest((fp <= 3).astype(float), D["p_top3"].to_numpy(), races, periods, "binary"),
    }
    rows, missing = [], []
    for op, a, b in NAMED:
        if a not in need or b not in need:
            missing.append((op, a, b))
            continue
        z = pp.pair_column(D, op, a, b, consts)
        za, zb = pp._std(D, a, consts, races), pp._std(D, b, consts, races)
        r = {"op": op, "a": a, "b": b}
        for t, st in tests.items():
            zz = st.z(z[:, None])
            ez = st.efficient_z(z, np.column_stack([za, zb])) if op == "prod" else {"disc": np.nan, "conf": np.nan}
            for p in ("disc", "conf"):
                r[f"z_{t}_{p}"] = round(float(zz[p][0]), 2)
                r[f"ez_{t}_{p}"] = round(float(ez[p]), 2)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "named_score.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_rows", 200)
    print(T.to_string(index=False))
    print("列なし:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
