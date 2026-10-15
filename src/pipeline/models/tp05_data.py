"""tp-05b〜e・tp-06 共通のデータ読み込み: 学習テーブル + 前段 3 列 + レース内の相対化を作り、レース単位に連続して並べる。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipeline.models import ablation as ab
from src.pipeline.models import win_models as wm
from src.pipeline.models.time_split import yearly_expanding_folds

STAGE_COLS = ["s_time_pred", "s_time_gap", "s_rank_pred"]
REL_BASE = ["f3_r", "sr_perf_ewm2", "sr_perf_ewm6", "sr_exp", "c_res_ewm2", "c_cls_ewm2", "a_l3fz_ewm2", "a_rk_ewm2", "a_style", "b_aff_all",
            "pc_adv", "f2_pace_match", "horse_prior_win_rate", "jockey_prior_win_rate"]


def load_dataset(work: Path) -> dict:
    """戻り値: df（date・race_id・horse_number 順）, BASE（tp-03 採用の 175 列）, REL（相対化の列名）, FEATS（特徴量セット名→列）, folds, meta。"""
    adopted = json.loads((work / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    stage = pd.read_parquet(work / "oof/stage_features_expanding.parquet")
    df = ab.load_table(work / "features/train_table.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    df = df.merge(stage[["race_id", "horse_id"] + STAGE_COLS], on=["race_id", "horse_id"], how="left")
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    rel_cols = [c for c in REL_BASE if c in df.columns]
    rel = wm.add_race_relative(df, rel_cols)
    for c in rel.columns:
        df[c] = rel[c].to_numpy()
    # C2: 相対化をさらに広い列に（展開・ペース・レーティング系など。馬の能力と、同じレースの相手との位置関係を明示する）
    fam = json.loads((work / "features/feature_families.json").read_text(encoding="utf-8"))
    rel2_src = [c for f in ("A", "C", "F2", "F3", "G", "H") for c in fam[f] if c in df.columns and c not in REL_BASE and str(df[c].dtype) != "category" and not c.startswith(("f2_n_", "pc_n_", "sr_n"))]
    g = df.groupby("race_id", sort=False)
    rel2 = {}
    for c in rel2_src:
        x = df[c].astype("float64")
        rel2[c + "_rz"] = ((x - g[c].transform("mean")) / g[c].transform("std").replace(0, float("nan"))).astype("float32")
        rel2[c + "_rrank"] = g[c].rank(ascending=False, method="average").astype("float32")
    rel2 = pd.DataFrame(rel2, index=df.index)
    for c in rel2.columns:
        df[c] = rel2[c].to_numpy()
    feats = {"base": base, "base+S": base + STAGE_COLS, "base+S+C1": base + STAGE_COLS + list(rel.columns),
             "base+S+C1+C2": base + STAGE_COLS + list(rel.columns) + list(rel2.columns)}
    missing = [c for c in feats["base+S+C1+C2"] if c not in df.columns]
    assert not missing, missing
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    return {"df": df, "BASE": base, "REL": list(rel.columns), "REL2": list(rel2.columns), "FEATS": feats, "folds": list(yearly_expanding_folds(df)), "meta": meta}
