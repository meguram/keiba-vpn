"""レース環境 ENV（層 A）の表を作る。再学習の候補評価（s10b）用。

    python -m src.scripts.data.build_race_env build [--years 2016-2025]
        → retrain_out/base_2016/race_env/env_race.parquet（レース単位）, env_horse.parquet（馬単位: 親モデルの土台の行と同じ）

as-of はすべて src/pipeline/features/race_env/asof.py が守る（前日まで・同日の前の R まで）。市場オッズは使わない。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.features.race_env.build import build_from_tables

OUT = Path("/home/jovyan/work/retrain_out/base_2016/race_env")
TRAIN_TABLE = Path("/home/jovyan/work/retrain_out/base_2016/features/train_table.parquet")


def build(y0: int, y1: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = build_from_tables(range(y0, y1 + 1))
    env.to_parquet(OUT / "env_race.parquet")
    print("env_race", env.shape, flush=True)
    t = pd.read_parquet(TRAIN_TABLE, columns=["race_id", "horse_id"])
    t["race_id"] = t["race_id"].astype(str)
    t["horse_id"] = t["horse_id"].astype(str)
    h = t.merge(env, on="race_id", how="left")
    h.to_parquet(OUT / "env_horse.parquet")
    cov = h.drop(columns=["race_id", "horse_id"]).notna().mean().round(3)
    print("env_horse", h.shape, flush=True)
    print(cov.to_string(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--years", default="2016-2025")
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    build(y0, y1)


if __name__ == "__main__":
    main()
