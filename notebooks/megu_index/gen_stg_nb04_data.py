#!/usr/bin/env python3
"""
NB-04 STGデータ → HTML可視化用 JSON 生成スクリプト

NB-04 実行後に output/nb04/megu_index.parquet が存在する状態で実行する。
出力先: output/nb04/megu_index_chart_data.json

実行例:
    cd /home/hirokiakataoka/.../keiba-vpn/repo
    python notebooks/megu_index/gen_stg_nb04_data.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

NB04_OUTPUT = Path(__file__).resolve().parent / "output" / "nb04"
PARQUET_PATH = NB04_OUTPUT / "megu_index.parquet"
JSON_OUT = NB04_OUTPUT / "megu_index_chart_data.json"

CLASS_RANK_LABEL = {
    1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝",
    5: "OP/L",   6: "G3",  7: "G2",  8: "G1",
}


def _box_stats(vals: list, rank: int, label: str) -> dict | None:
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


def build_chart_data(df: pd.DataFrame) -> dict:
    df_valid = df[df["megu_index"].notna()].copy()
    df_valid["megu_index"] = df_valid["megu_index"].astype(float)

    # 全馬 class_rank 別
    class_boxes = []
    for rank in sorted(df_valid["class_rank"].unique()):
        vals = df_valid[df_valid["class_rank"] == rank]["megu_index"].tolist()
        label = CLASS_RANK_LABEL.get(int(rank), str(rank))
        entry = _box_stats(vals, int(rank), label)
        if entry:
            class_boxes.append(entry)

    # 2着馬のみ class_rank 別
    df_2nd = df_valid[df_valid["finish_pos"] == 2]
    class_2nd_boxes = []
    for rank in sorted(df_2nd["class_rank"].unique()):
        vals = df_2nd[df_2nd["class_rank"] == rank]["megu_index"].tolist()
        label = CLASS_RANK_LABEL.get(int(rank), str(rank))
        entry = _box_stats(vals, int(rank), label)
        if entry:
            class_2nd_boxes.append(entry)

    # 1勝クラス 着順別キャリブレーション
    df_rank2 = df_valid[df_valid["class_rank"] == 2]
    calib = []
    for pos in range(1, 13):
        sub = df_rank2[df_rank2["finish_pos"] == pos]["megu_index"].tolist()
        if len(sub) < 3:
            continue
        arr = np.array(sub, dtype=float)
        calib.append({
            "pos": int(pos),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "n": len(sub),
        })

    # サンプルレース（1勝クラス芝1600m）
    sample_race = []
    mask = (
        (df_valid["class_rank"] == 2)
        & (df_valid["surface"] == "芝")
        & (df_valid["distance"] == 1600)
    )
    race_ids = df_valid.loc[mask, "race_id"].unique()
    if len(race_ids) > 0:
        rid = race_ids[len(race_ids) // 2]   # 中間レースを選択（最初より代表的）
        race_df = df_valid[df_valid["race_id"] == rid].sort_values("finish_pos")
        needed = ["finish_pos", "adjusted_time_sec", "delta_pace_sec",
                  "delta_track_sec", "corrected_time", "par_time_class_sec",
                  "megu_index", "computation_status"]
        for _, row in race_df.head(12).iterrows():
            sample_race.append({
                "pos": int(row["finish_pos"]),
                "time_adj": round(float(row["adjusted_time_sec"]), 2),
                "d_pace": round(float(row.get("delta_pace_sec", 0.0) or 0.0), 3),
                "d_track": round(float(row.get("delta_track_sec", 0.0) or 0.0), 3),
                "corr_time": round(float(row["corrected_time"]), 2),
                "par_time": round(float(row["par_time_class_sec"]), 2),
                "megu": None if pd.isna(row["megu_index"])
                        else round(float(row["megu_index"]), 1),
                "status": str(row["computation_status"]),
            })

    # overall 統計
    r2_2nd = df_valid[(df_valid["class_rank"] == 2) & (df_valid["finish_pos"] == 2)]["megu_index"]
    overall = {
        "n_total": int(len(df)),
        "n_valid": int((df["computation_status"] == "valid").sum()),
        "n_oor":   int((df["computation_status"] == "out_of_range").sum()),
        "n_no_par":int((df["computation_status"] == "no_par").sum()),
        "mean": round(float(df_valid["megu_index"].mean()), 2),
        "std":  round(float(df_valid["megu_index"].std()), 2),
        "rank2_2nd_mean": round(float(r2_2nd.mean()), 2) if len(r2_2nd) > 0 else None,
    }

    return {
        "overall": overall,
        "class_boxes": class_boxes,
        "class_2nd_boxes": class_2nd_boxes,
        "calib": calib,
        "sample_race": sample_race,
    }


def main() -> None:
    if not PARQUET_PATH.exists():
        print(f"[ERROR] parquet が見つかりません: {PARQUET_PATH}", file=sys.stderr)
        print("NB-04 を実行して megu_index.parquet を生成してから再実行してください。", file=sys.stderr)
        sys.exit(1)

    print(f"  読み込み: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  shape: {df.shape}")
    print(f"  computation_status: {df['computation_status'].value_counts().to_dict()}")

    chart_data = build_chart_data(df)

    # キャリブレーション確認
    r2_mean = chart_data["overall"]["rank2_2nd_mean"]
    print(f"\n  [キャリブレーション] 1勝クラス2着馬 megu_index 平均: {r2_mean}  (目標≈50)")

    NB04_OUTPUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(chart_data, ensure_ascii=False, indent=2))
    print(f"\n  保存: {JSON_OUT}")
    print(f"  class_boxes: {[b['label'] for b in chart_data['class_boxes']]}")
    print(f"  class_2nd_boxes: {[b['label'] for b in chart_data['class_2nd_boxes']]}")
    print(f"  overall: {chart_data['overall']}")
    print("\n完了")


if __name__ == "__main__":
    main()
