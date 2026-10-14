#!/usr/bin/env python3
"""特徴量が「どの生データに依存するか」を、入力を 1 項目ずつ変化させて実測する（摂動法）。

- 今日の出走行の項目（馬体重・斤量・馬場状態・性齢・出走馬の構成・馬番）を変える → 同じ行の特徴量が動けば「今日の項目に依存」。
- 今日の結果（着順・時計・上り・通過順・ラップ）を変える → 同じ行の特徴量が動けば**リーク**（同日の他レース由来の sr_var_today を除く）。
  後の行の特徴量が動けば「過去走のその項目に依存」。

  python -m src.scripts.data.feature_lineage --layer-a data/local/modeling/training_pipeline/layer_a.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.pipeline.features.horse_history.builder import build_from_runs, feature_columns
from src.pipeline.features.horse_history.common import build_repair_map, load_runs

KEY = ["race_id", "horse_id"]


def _in(R: list[str]) -> pl.Expr:
    return pl.col("race_id").is_in(R)


def _shift(col: str, delta: float):
    return lambda r, R: r.with_columns(pl.when(_in(R)).then(pl.col(col) + delta).otherwise(pl.col(col)).alias(col))


TODAY = {  # 今日の出走行の項目（レース前に分かるもの）
    "馬体重(weight)": _shift("weight", 30.0),
    "斤量(jockey_weight)": _shift("jockey_weight", 3.0),
    "馬場状態(going)": lambda r, R: r.with_columns(pl.when(_in(R)).then(pl.when(pl.col("going") >= 2).then(0).otherwise(2)).otherwise(pl.col("going")).alias("going")),
    "性齢(age)": lambda r, R: r.with_columns(pl.when(_in(R)).then(pl.col("age") + 1).otherwise(pl.col("age")).alias("age")),
    "出走馬の構成(1頭除く)": lambda r, R: r.filter(~(_in(R) & (pl.col("horse_number") == pl.col("horse_number").min().over("race_id")))),
}
RESULT = {  # 今日の結果（レース後に分かるもの）
    "着順(finish_position)": lambda r, R: r.with_columns(pl.when(_in(R)).then(pl.col("n_fin") + 1 - pl.col("finish_position")).otherwise(pl.col("finish_position")).alias("finish_position")),
    "走破時計(time_sec)": _shift("time_sec", 1.5),
    "上り3F(last_3f)": _shift("last_3f", 0.8),
    "通過順(passing_order)": lambda r, R: r.with_columns(pl.when(_in(R)).then(pl.lit("1-1")).otherwise(pl.col("passing_order")).alias("passing_order")),
    "ペース(pace_bal)": _shift("pace_bal", 1.0),
    "ラップ(rpci)": _shift("rpci", 2.0),
}


def _changed(base: pl.DataFrame, new: pl.DataFrame, cols: list[str], keys: pl.DataFrame) -> dict[str, int]:
    a = base.join(keys, on=KEY).select(KEY + cols).sort(KEY)
    b = new.join(keys, on=KEY).select(KEY + cols).sort(KEY)
    common = a.join(b.select(KEY), on=KEY).sort(KEY)
    b = b.join(common.select(KEY), on=KEY).sort(KEY)
    out = {}
    for c in cols:
        x, y = common[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
        d = np.where(np.isnan(d), np.inf, d)
        out[c] = int((d > 1e-9).sum())
    return out


def run(layer_a: str, day: str | None = None) -> dict:
    """day（YYYY-MM-DD）の全レースだけを変化させる。複数日・ランダム抽出だと、同じ馬の過去走や対戦相手のレーティングが
    変わって『今日の行』の特徴量が動き、依存関係と区別できなくなる（汚染）ため、1 開催日に絞る。"""
    runs, rmap = load_runs(), build_repair_map(layer_a)
    base, P = build_from_runs(runs, rmap)
    cols = sorted({c for v in feature_columns(base.columns).values() for c in v})
    if day is None:  # 2024 年 7〜10 月（結果ページの列ずれが無い期間）でレース数が最大の日曜
        d = (runs.filter((pl.col("date") >= pl.date(2024, 7, 1)) & (pl.col("date") <= pl.date(2024, 10, 31)) & (pl.col("date").dt.weekday() >= 6))
             .group_by("date").agg(pl.col("race_id").n_unique().alias("n")).sort(["n", "date"], descending=[True, False]))
        day = str(d["date"][0])
    R = runs.filter(pl.col("date") == pl.lit(day).str.to_date())["race_id"].unique().sort().to_list()
    cur = base.filter(_in(R)).select(KEY)
    later = base.filter(~_in(R) & (pl.col("date") > pl.lit(day).str.to_date())).select(KEY)
    res = {"day": day, "races": len(R), "rows_current": cur.height, "rows_later": later.height, "features": {c: {"today": [], "history": [], "leak": []} for c in cols}}
    for name, fn in {**TODAY, **RESULT}.items():
        t0 = time.time()
        new, _ = build_from_runs(fn(runs, R), rmap)
        cur_ch = _changed(base, new, cols, cur)
        late_ch = _changed(base, new, cols, later)
        for c in cols:
            if name in TODAY:
                if cur_ch[c] > 0:
                    res["features"][c]["today"].append(name)
                if late_ch[c] > 0:
                    res["features"][c]["history"].append(name)
            else:
                if cur_ch[c] > 0:
                    res["features"][c]["leak"].append(name)
                if late_ch[c] > 0:
                    res["features"][c]["history"].append(name)
        print(f"{name}: 同じ行で変化 {sum(v > 0 for v in cur_ch.values())} 特徴量 / 後の行で変化 {sum(v > 0 for v in late_ch.values())} 特徴量 ({time.time() - t0:.0f}s)", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-a", default="data/local/modeling/training_pipeline/layer_a.parquet")
    ap.add_argument("--day", default=None, help="変化させる開催日（YYYY-MM-DD）。省略時は 2024 年 7〜10 月でレース数が最大の日曜")
    ap.add_argument("--out", type=Path, default=Path("data/local/modeling/training_pipeline/features/feature_lineage.json"))
    args = ap.parse_args()
    res = run(args.layer_a, args.day)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("保存:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
