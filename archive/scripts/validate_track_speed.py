#!/usr/bin/env python3
"""馬場速度レベリング v2 の計算結果妥当性チェック。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.race.track_speed_engine import (  # noqa: E402
    PACE_BASELINES_PATH,
    BASELINES_PATH,
    RACES_DIR,
    TrackSpeedEngine,
    classify_speed,
)


def main() -> None:
    print("=" * 60)
    print("馬場速度レベリング 妥当性検証")
    print("=" * 60)

    # --- アーティファクト ---
    bl = pd.read_parquet(BASELINES_PATH)
    pb = pd.read_parquet(PACE_BASELINES_PATH)
    print(f"\n[アーティファクト]")
    print(f"  タイム基準: {len(bl)} グループ (mean={bl['mean'].mean():.1f}s, std={bl['std'].mean():.2f}s)")
    if "context_key" in pb.columns:
        n_ctx = pb["context_key"].nunique()
        metrics = pb.groupby("context_key")["metric"].first().value_counts()
        print(f"  ペースコホート: {n_ctx} 条件 / 指標 {dict(metrics)}")
    else:
        print(f"  ペース基準: {len(pb)} 行（旧形式の可能性）")

    frames = []
    for p in sorted(RACES_DIR.glob("races_*.parquet")):
        frames.append(pd.read_parquet(p))
    races = pd.concat(frames, ignore_index=True)
    print(f"\n[振り分け済みレース] {len(races)} 件 / {races['date'].astype(str).str[:4].nunique()} 年")

  # 必須列
    need = ["z", "z_raw", "time_2nd", "time_2nd_adj", "pace_adj_sec", "pace_label", "first_half_3f"]
    missing = [c for c in need if c not in races.columns]
    if missing:
        print(f"  WARN: 欠損列 {missing} — 再振り分けが必要")
    else:
        print(f"  ペース補正列: OK")

    # --- z 分布（全体は N(0,1) に近いはずではないが極端値チェック）---
    z = races["z"].dropna()
    z_raw = races["z_raw"].dropna() if "z_raw" in races.columns else z
    print(f"\n[z-score 分布]")
    print(f"  z (補正後): mean={z.mean():.3f} std={z.std():.3f} min={z.min():.2f} max={z.max():.2f}")
    print(f"  z_raw:      mean={z_raw.mean():.3f} std={z_raw.std():.3f} min={z_raw.min():.2f} max={z_raw.max():.2f}")
    pct_extreme = (z.abs() > 3).mean() * 100
    print(f"  |z|>3: {pct_extreme:.1f}% ({(z.abs() > 3).sum()} 件)")

    # レベル分布
    if "label" in races.columns:
        print(f"\n[レベル分布 (補正後 z)]")
        for lbl, cnt in races["label"].value_counts().items():
            print(f"  {lbl}: {cnt} ({100*cnt/len(races):.1f}%)")

    # --- ペース補正効果 ---
    if "pace_adj_sec" in races.columns:
        adj = races["pace_adj_sec"].fillna(0)
        has_fh = races["first_half_3f"].notna()
        print(f"\n[ペース補正]")
        print(f"  前半3Fあり: {has_fh.mean()*100:.1f}%")
        print(f"  補正あり(|adj|>0.05): {(adj.abs() > 0.05).mean()*100:.1f}%")
        print(f"  adj: mean={adj.mean():.3f}s std={adj.std():.3f}s min={adj.min():.2f} max={adj.max():.2f}")
        if "pace_label" in races.columns:
            for lbl in ("前傾", "抑え"):
                sub = races[races["pace_label"] == lbl]
                if len(sub):
                    print(f"  {lbl}: n={len(sub)} adj_mean={sub['pace_adj_sec'].mean():.2f}s z_raw_mean={sub['z_raw'].mean():.2f} z_mean={sub['z'].mean():.2f}")

        # 前傾レース: z_raw が正(遅い見え) → z が下がる方向
        zenkei = races[races["pace_label"] == "前傾"]
        if len(zenkei) > 10:
            delta = (zenkei["z_raw"] - zenkei["z"]).mean()
            print(f"  前傾: 平均(z_raw - z)={delta:.3f} (正ならハイペース補正で高速側へ)")

        osae = races[races["pace_label"] == "抑え"]
        if len(osae) > 10:
            delta = (osae["z_raw"] - osae["z"]).mean()
            print(f"  抑え: 平均(z_raw - z)={delta:.3f} (負ならスローペース補正で低速側へ)")

    # --- 距離0チェック ---
    d0 = (races["distance"].fillna(0) == 0).mean() * 100
    print(f"\n[距離]")
    print(f"  distance=0: {d0:.1f}%")

    # --- 日×場サマリー整合 ---
    eng = TrackSpeedEngine(str(ROOT))
    eng.load_baselines()
    sample_date = races["date"].astype(str).str.replace("-", "", regex=False).iloc[0]
    sample_venue = races["venue"].iloc[0]
    api = eng.query_day(sample_date, sample_venue)
    print(f"\n[API spot check] {sample_date} {sample_venue}")
    print(f"  races={len(api['races'])} summary={list(api['summary'].keys())}")

    # --- 相関: 補正後 z が日次平均と整合 ---
    races["date_norm"] = races["date"].astype(str).str.replace("-", "", regex=False)
    day_surf = races.groupby(["date_norm", "venue", "surface"]).agg(
        mean_z=("z", "mean"),
        mean_z_raw=("z_raw", "mean"),
        n=("z", "count"),
    ).reset_index()
    corr = day_surf["mean_z"].corr(day_surf["mean_z_raw"])
    print(f"\n[日×場×路面 平均z]")
    print(f"  補正前後の相関: {corr:.3f} (1.0に近すぎると補正無効、低すぎると過補正)")

    # 妥当性判定
    issues: list[str] = []
    if pct_extreme > 5:
        issues.append(f"|z|>3 が多い ({pct_extreme:.1f}%)")
    if d0 > 1:
        issues.append(f"distance=0 が残存 ({d0:.1f}%)")
    if missing:
        issues.append("ペース列欠損")
    pace_ctx = pb["context_key"].nunique() if "context_key" in pb.columns else len(pb)
    if pace_ctx < 50:
        issues.append(f"ペースコホート条件が少ない ({pace_ctx})")
    if "pace_adj_sec" in races.columns and (races["pace_adj_sec"].abs() > 0.05).mean() < 0.3:
        issues.append("ペース補正適用率が低い")

    print("\n" + "=" * 60)
    if issues:
        print("要確認:", "; ".join(issues))
    else:
        print("総合: 主要チェックをパス")
    print("=" * 60)


if __name__ == "__main__":
    main()
