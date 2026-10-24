#!/usr/bin/env python3
"""
NB-01 → NB-02 → NB-03 → NB-04 モックデータ計算スクリプト
HTML可視化用のJSONデータを生成する。

実行例:
    cd /home/hirokiakataoka/.../keiba-vpn/repo
    python notebooks/megu_index/gen_mock_nb04_data.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# NB-01/NB-02 モック再利用
from notebooks.megu_index.test_mock_nb01_nb02 import build_mock_race_result_flat, run_nb01, run_nb02
from src.pipeline.megu_index.track_speed import assign_class_rank


def run_nb03(df: pd.DataFrame, coeff_pace: pd.DataFrame, par_time_class: pd.DataFrame) -> pd.DataFrame:
    """
    NB-03 delta_track 計算。
    入力: df (megu_dataset 相当), coeff_pace, par_time_class
    出力: delta_track (date × venue × surface ごとの馬場差)
    """
    df = df.copy()

    # class_rank 付与
    if "class_rank" not in df.columns:
        df["class_rank"] = df.apply(
            lambda r: assign_class_rank(r["grade"], r["race_class"]), axis=1
        )
        df["class_rank"] = df["class_rank"].fillna(2).astype(int)

    # coeff_pace マージ
    cp_slim = coeff_pace[["venue", "surface", "distance", "coeff_pace"]].copy()
    for c in ["venue", "surface"]:
        df[c] = df[c].astype(str)
        cp_slim[c] = cp_slim[c].astype(str)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    cp_slim["distance"] = pd.to_numeric(cp_slim["distance"], errors="coerce")
    df = df.merge(cp_slim, on=["venue", "surface", "distance"], how="left", suffixes=("", "_cp"))
    if "coeff_pace_cp" in df.columns:
        df["coeff_pace"] = df["coeff_pace"].fillna(df["coeff_pace_cp"])
        df.drop(columns=["coeff_pace_cp"], inplace=True)

    # par_time_class マージ
    pt_slim = par_time_class[["venue", "surface", "distance", "class_rank", "par_time_sec"]].copy()
    pt_slim["class_rank"] = pd.to_numeric(pt_slim["class_rank"], errors="coerce").astype("Int64")
    df["class_rank_key"] = df["class_rank"].astype("Int64")
    df = df.merge(
        pt_slim,
        left_on=["venue", "surface", "distance", "class_rank_key"],
        right_on=["venue", "surface", "distance", "class_rank"],
        how="left",
        suffixes=("", "_pt"),
    )
    if "class_rank_pt" in df.columns:
        df.drop(columns=["class_rank_pt"], inplace=True)

    df_valid = df[df["par_time_sec"].notna()].copy()

    # front_split_dev (NB-03 §5 の意味: 実測 − par)
    df_valid["front_split_dev_03"] = (
        df_valid["front_split_sec"] - df_valid["par_front_split_sec"]
    ).fillna(0.0)

    # delta_pace_sec
    df_valid["coeff_pace"] = df_valid["coeff_pace"].fillna(0.0)
    df_valid["delta_pace_sec_03"] = df_valid["coeff_pace"] * df_valid["front_split_dev_03"]
    df_valid["time_after_pace"] = df_valid["adjusted_time_sec"] - df_valid["delta_pace_sec_03"]

    # G1 除外 (class_rank <= 7)
    track_target = df_valid[df_valid["class_rank"] <= 7].copy()

    # 日×会場×コース集計
    track_target["date"] = track_target["date"].astype(str).str[:10]
    delta_track = (
        track_target.groupby(["date", "venue", "surface"])
        .agg(
            mean_time_after_pace=("time_after_pace", "mean"),
            mean_par_time=("par_time_sec", "mean"),
            n_races=("race_id", "nunique"),
        )
        .reset_index()
    )
    delta_track["delta_track_sec"] = (
        delta_track["mean_time_after_pace"] - delta_track["mean_par_time"]
    )
    # sparse fallback
    delta_track["is_fallback"] = delta_track["n_races"] < 3
    delta_track.loc[delta_track["is_fallback"], "delta_track_sec"] = 0.0

    return delta_track[["date", "venue", "surface", "delta_track_sec", "n_races", "is_fallback"]]


def run_nb04(
    df: pd.DataFrame,
    coeff_pace: pd.DataFrame,
    par_time_class: pd.DataFrame,
    delta_track: pd.DataFrame,
) -> pd.DataFrame:
    """
    NB-04 megu_index 計算。
    基準: megu_index = 50 ↔ 1勝クラス 2着馬相当パフォーマンス
    """
    df = df.copy()

    # class_rank 付与
    if "class_rank" not in df.columns:
        df["class_rank"] = df.apply(
            lambda r: assign_class_rank(r["grade"], r["race_class"]), axis=1
        )
        df["class_rank"] = df["class_rank"].fillna(2).astype(int)

    # coeff_pace マージ
    cp_slim = coeff_pace[["venue", "surface", "distance", "coeff_pace"]].copy()
    for c in ["venue", "surface"]:
        df[c] = df[c].astype(str)
        cp_slim[c] = cp_slim[c].astype(str)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    cp_slim["distance"] = pd.to_numeric(cp_slim["distance"], errors="coerce")
    df = df.merge(cp_slim, on=["venue", "surface", "distance"], how="left", suffixes=("", "_cp"))
    if "coeff_pace_cp" in df.columns:
        df["coeff_pace"] = df["coeff_pace"].fillna(df["coeff_pace_cp"])
        df.drop(columns=["coeff_pace_cp"], inplace=True)

    # front_split_dev
    df["front_split_dev"] = (df["front_split_sec"] - df["par_front_split_sec"]).fillna(0.0)

    # delta_pace_sec
    df["coeff_pace"] = df["coeff_pace"].fillna(0.0)
    df["delta_pace_sec"] = df["coeff_pace"] * df["front_split_dev"]

    # delta_track_sec マージ
    df["date"] = df["date"].astype(str).str[:10]
    dt_slim = delta_track[["date", "venue", "surface", "delta_track_sec"]].copy()
    df = df.merge(dt_slim, on=["date", "venue", "surface"], how="left", suffixes=("", "_dt"))
    if "delta_track_sec_dt" in df.columns:
        df["delta_track_sec"] = df["delta_track_sec"].fillna(df["delta_track_sec_dt"])
        df.drop(columns=["delta_track_sec_dt"], inplace=True)
    df["delta_track_sec"] = df["delta_track_sec"].fillna(0.0)

    # par_time_class_sec マージ
    pt_slim = par_time_class[["venue", "surface", "distance", "class_rank", "par_time_sec"]].copy()
    pt_slim["class_rank"] = pd.to_numeric(pt_slim["class_rank"], errors="coerce").astype("Int64")
    df["class_rank_key"] = df["class_rank"].astype("Int64")
    df = df.merge(
        pt_slim,
        left_on=["venue", "surface", "distance", "class_rank_key"],
        right_on=["venue", "surface", "distance", "class_rank"],
        how="left",
        suffixes=("", "_pt"),
    )
    if "class_rank_pt" in df.columns:
        df.drop(columns=["class_rank_pt"], inplace=True)
    df.rename(columns={"par_time_sec": "par_time_class_sec"}, inplace=True)

    # corrected_time
    df["corrected_time"] = df["adjusted_time_sec"] - df["delta_pace_sec"] - df["delta_track_sec"]

    # out_of_range
    time_2nd = (
        df[df["finish_pos"] == 2]
        .groupby("race_id")["adjusted_time_sec"]
        .first()
        .rename("time_2nd")
    )
    df = df.merge(time_2nd, on="race_id", how="left")
    oor = (df["finish_pos"] > 2) & (df["adjusted_time_sec"] > df["time_2nd"] + 2.0)
    df["computation_status"] = "valid"
    df.loc[oor, "computation_status"] = "out_of_range"

    # megu_index
    df["megu_index"] = 50.0 + (df["par_time_class_sec"] - df["corrected_time"]) * 10.0
    df.loc[df["computation_status"] == "out_of_range", "megu_index"] = np.nan
    df.loc[df["par_time_class_sec"].isna(), "megu_index"] = np.nan
    df.loc[df["par_time_class_sec"].isna(), "computation_status"] = "no_par"

    return df


CLASS_RANK_LABEL = {
    1: "未勝利",
    2: "1勝",
    3: "2勝",
    4: "3勝",
    5: "OP/L",
    6: "G3",
    7: "G2",
    8: "G1",
}


def _box_stats(vals: list, rank: int, label: str) -> dict | None:
    """vals のボックスプロット統計を辞書で返す。n < 3 なら None。"""
    if len(vals) < 3:
        return None
    arr = np.array(vals, dtype=float)
    q1  = float(np.percentile(arr, 25))
    med = float(np.percentile(arr, 50))
    q3  = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lo  = float(np.min(arr[arr >= q1 - 1.5 * iqr]))
    hi  = float(np.max(arr[arr <= q3 + 1.5 * iqr]))
    outliers = arr[(arr < lo) | (arr > hi)].tolist()
    return {
        "rank": int(rank),
        "label": label,
        "q1": round(q1, 2), "median": round(med, 2), "q3": round(q3, 2),
        "lo": round(lo, 2), "hi": round(hi, 2),
        "mean": round(float(arr.mean()), 2),
        "n": len(vals),
        "outliers": [round(v, 2) for v in outliers[:30]],
    }


def build_chart_data(df_result: pd.DataFrame) -> dict:
    """Plotly 用チャートデータを構築。"""
    df_valid = df_result[df_result["megu_index"].notna()].copy()
    df_valid["megu_index"] = df_valid["megu_index"].astype(float)

    # --- 1) クラスランク別 box plot (全馬) ---
    class_boxes = []
    for rank in sorted(df_valid["class_rank"].unique()):
        vals = df_valid[df_valid["class_rank"] == rank]["megu_index"].tolist()
        label = CLASS_RANK_LABEL.get(rank, str(rank))
        entry = _box_stats(vals, rank, label)
        if entry:
            class_boxes.append(entry)

    # --- 1b) クラスランク別 box plot (2着馬のみ) ---
    df_2nd = df_valid[df_valid["finish_pos"] == 2]
    class_2nd_boxes = []
    for rank in sorted(df_2nd["class_rank"].unique()):
        vals = df_2nd[df_2nd["class_rank"] == rank]["megu_index"].tolist()
        label = CLASS_RANK_LABEL.get(rank, str(rank))
        entry = _box_stats(vals, rank, label)
        if entry:
            class_2nd_boxes.append(entry)

    # --- 2) 1勝クラス 着順別分布（キャリブレーション確認）---
    df_rank2 = df_valid[df_valid["class_rank"] == 2].copy()
    calib = []
    for pos in [1, 2, 3, 4, 5, 6]:
        sub = df_rank2[df_rank2["finish_pos"] == pos]["megu_index"].tolist()
        if len(sub) < 3:
            continue
        arr = np.array(sub, dtype=float)
        calib.append({
            "pos": pos,
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "n": len(sub),
        })

    # --- 3) サンプルレース（1勝クラス芝1600m から1レース取得）---
    sample_race = []
    mask = (
        (df_valid["class_rank"] == 2)
        & (df_valid["surface"] == "芝")
        & (df_valid["distance"] == 1600)
    )
    race_ids = df_valid.loc[mask, "race_id"].unique()
    if len(race_ids) > 0:
        rid = race_ids[0]
        race_df = df_valid[df_valid["race_id"] == rid].sort_values("finish_pos")
        for _, row in race_df.head(12).iterrows():
            sample_race.append({
                "pos": int(row["finish_pos"]),
                "horse_id": str(row["horse_id"]),
                "time_adj": round(float(row["adjusted_time_sec"]), 2),
                "d_pace": round(float(row["delta_pace_sec"]), 3),
                "d_track": round(float(row["delta_track_sec"]), 3),
                "corr_time": round(float(row["corrected_time"]), 2),
                "par_time": round(float(row["par_time_class_sec"]), 2),
                "megu": None if pd.isna(row["megu_index"]) else round(float(row["megu_index"]), 1),
                "status": str(row["computation_status"]),
            })

    # --- 4) 全体統計 ---
    overall = {
        "n_total": int(len(df_result)),
        "n_valid": int((df_result["computation_status"] == "valid").sum()),
        "n_oor": int((df_result["computation_status"] == "out_of_range").sum()),
        "n_no_par": int((df_result["computation_status"] == "no_par").sum()),
        "mean": round(float(df_valid["megu_index"].mean()), 2),
        "std": round(float(df_valid["megu_index"].std()), 2),
        "rank2_2nd_mean": round(float(
            df_valid[(df_valid["class_rank"] == 2) & (df_valid["finish_pos"] == 2)]["megu_index"].mean()
        ), 2) if (df_valid["class_rank"] == 2).any() else None,
    }

    return {
        "overall": overall,
        "class_boxes": class_boxes,
        "class_2nd_boxes": class_2nd_boxes,
        "calib": calib,
        "sample_race": sample_race,
    }


def main() -> None:
    print("=== モックデータ生成 ===")
    df_raw = build_mock_race_result_flat(seed=42)

    print("=== NB-01 実行 ===")
    df_clean, par_splits = run_nb01(df_raw)

    print("=== NB-02 実行 ===")
    coeff_pace, par_time_class = run_nb02(df_clean, par_splits)

    print("=== NB-03 実行 ===")
    delta_track = run_nb03(df_clean, coeff_pace, par_time_class)
    print(f"  delta_track: {len(delta_track)} 日×会場×コース")

    print("=== NB-04 実行 ===")
    df_result = run_nb04(df_clean, coeff_pace, par_time_class, delta_track)
    print(f"  megu_index 計算: {len(df_result)} 行")
    status = df_result["computation_status"].value_counts()
    print(f"  {status.to_dict()}")

    # 1勝クラス 2着馬のmegu_indexが50に近いか確認
    rank2_2nd = df_result[
        (df_result["class_rank"] == 2) & (df_result["finish_pos"] == 2)
    ]["megu_index"].dropna()
    if len(rank2_2nd) > 0:
        print(f"\n  [キャリブレーション] 1勝クラス2着馬 megu_index 平均: {rank2_2nd.mean():.2f}  (目標≈50)")

    # チャートデータ構築
    chart_data = build_chart_data(df_result)
    out_path = Path(__file__).parent / "mock_nb04_chart_data.json"
    out_path.write_text(json.dumps(chart_data, ensure_ascii=False, indent=2))
    print(f"\n  チャートデータ保存: {out_path}")
    print("  class_boxes: ", [b["label"] for b in chart_data["class_boxes"]])
    print("  overall: ", chart_data["overall"])


if __name__ == "__main__":
    main()
