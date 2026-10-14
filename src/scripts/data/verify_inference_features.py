#!/usr/bin/env python3
"""推論モードの特徴量が、学習時（結果込み）に作った特徴量と一致するかを、実データの 1 開催日で確かめる。

  python -m src.scripts.data.verify_inference_features --day 2024-07-06

過去の開催日の出馬表（race_shutuba_flat）の行を「結果なしの当日の出走馬」とみなして推論モードで特徴量を作り、
同じ日の学習時の特徴量と比べる。tier 別に、一致しない特徴量を表示する（想定: tier 1 の ``sr_var_today`` 系だけ）。
所要時間（履歴の読み込み・特徴量の計算）も出す。T-60 の時間予算（目標 30 秒・上限 120 秒）の判断に使う。
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import polars as pl

from src.pipeline.features.horse_history.builder import build_horse_history, feature_columns
from src.pipeline.features.horse_history.common import TABLES_DIR, build_repair_map, read_runs_raw
from src.pipeline.features.horse_history.inference import CARD_COLS, build_inference_features, load_tiers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2024-07-06")
    ap.add_argument("--layer-a", default="data/local/modeling/training_pipeline/layer_a.parquet")
    ap.add_argument("--keep-weights", action="store_true", help="馬体重を入れたまま（既定は T-60 と同じく入れる。外すと前日朝の状態）")
    ap.add_argument("--no-weights", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    F, _ = build_horse_history(args.layer_a)
    t_batch = time.perf_counter() - t0
    cols = sorted({c for v in feature_columns(F.columns).values() for c in v})

    t0 = time.perf_counter()
    raw = read_runs_raw()
    rm = build_repair_map(args.layer_a)
    t_load = time.perf_counter() - t0

    year = args.day[:4]
    card = pl.read_parquet(TABLES_DIR / year / "race_shutuba_flat.parquet").filter(pl.col("date") == args.day)
    card = card.select([c for c in CARD_COLS if c in card.columns])
    if args.no_weights:
        card = card.with_columns(pl.lit(None, pl.Int64).alias("weight"))
    out, info = build_inference_features(raw, card, rm)
    ref = F.filter(pl.col("race_id").is_in(card["race_id"].unique().to_list()))
    print(f"日: {args.day} レース {info['races']} 行 {info['rows']}（学習時 {ref.height}）  馬体重あり {info['weights_known_ratio']:.0%}")
    print(f"所要: 履歴の読み込み {t_load:.1f}s / 推論モードの特徴量計算 {info['build_sec']}s（参考: 学習時の全件計算 {t_batch:.0f}s）")

    tiers = load_tiers()
    a = ref.select(["race_id", "horse_id"] + cols).sort(["race_id", "horse_id"])
    b = out.select(["race_id", "horse_id"] + cols).sort(["race_id", "horse_id"])
    if a.height != b.height or not (a.select(["race_id", "horse_id"]).to_numpy() == b.select(["race_id", "horse_id"]).to_numpy()).all():
        print("行の対応が一致しません（出走取消などで出走馬が異なる可能性）")
        return 2
    bad: dict[int, list[str]] = defaultdict(list)
    for c in cols:
        x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
        n = int((np.where(np.isnan(d), np.inf, d) > 1e-6).sum())
        if n:
            bad[tiers.get(c, -1)].append(f"{c}({n}行)")
    for t in (0, 1, 2, -1):
        print(f"tier {t}: 不一致 {len(bad[t])} 個", bad[t] if bad[t] else "")
    return 0 if not bad[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
