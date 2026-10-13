"""馬の過去走履歴特徴量の共通部（読み込み・クリーニング・last_3f 復元）。"""
from __future__ import annotations
import json, re
from datetime import date
import numpy as np
import polars as pl

from pathlib import Path

from src.config.data_paths import PAGE_REF
from src.pipeline.models.time_split import TRAINING_YEARS

TABLES_DIR = Path(PAGE_REF) / "tables"
YEARS = [int(y) for y in TRAINING_YEARS]   # 2026 年は使わない
# 縮約強度・分位点・クラスタなどの「全体フィット」は、この日以前のデータだけで行う。
# 時系列 CV の最初の fold（valid 2022）の学習終端 2021-12-22（purge+embargo 後）より前に置く。
FIT_END = date(2021, 12, 21)
CLASS_LEVEL = {"新馬": 0.0, "未勝利": 0.0, "1勝": 1.0, "2勝": 2.0, "3勝": 3.0, "OP": 4.0, "L": 4.5, "G3": 5.0, "G2": 6.0, "G1": 7.0}
GOING = {"良": 0, "稍重": 1, "重": 2, "不良": 3}

RUN_COLS = ["race_id", "date", "start_time", "venue_code", "surface", "distance", "direction", "track_condition",
            "grade", "pace", "finish_position", "horse_number", "horse_id", "sex_age", "time_sec",
            "passing_order", "last_3f", "weight", "jockey_id", "trainer_id", "jockey_weight"]


def load_runs(tables_dir: str | Path | None = None) -> pl.DataFrame:
    base = Path(tables_dir) if tables_dir else TABLES_DIR
    df = pl.concat([pl.read_parquet(base / str(y) / "race_result_flat.parquet", columns=RUN_COLS) for y in YEARS])
    df = df.filter((pl.col("finish_position") > 0) & pl.col("venue_code").cast(pl.Int32, strict=False).is_between(1, 10))
    df = df.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d"),
        pl.col("surface").fill_null("障"),
        pl.len().over("race_id").alias("n_fin"),
        pl.col("track_condition").replace_strict(GOING, default=0, return_dtype=pl.Int32).alias("going"),
        pl.col("grade").replace_strict(CLASS_LEVEL, default=1.0, return_dtype=pl.Float64).alias("class_level"),
        pl.col("sex_age").str.slice(1).cast(pl.Int32, strict=False).alias("age"),
        pl.col("sex_age").str.slice(0, 1).alias("sex"),
        pl.col("passing_order").str.extract(r"^(\d+)", 1).cast(pl.Float64).alias("po_first"),
        pl.col("passing_order").str.extract(r"(\d+)$", 1).cast(pl.Float64).alias("po_last"),
        pl.col("pace").str.json_path_match("$.first_half_3f").cast(pl.Float64, strict=False).alias("pace_first"),
        pl.col("pace").str.json_path_match("$.second_half_3f").cast(pl.Float64, strict=False).alias("pace_second"),
        (pl.col("distance").cast(pl.Float64) / 1600.0).log().alias("logd"),
    ).with_columns(
        pl.when(pl.col("distance") <= 1200).then(0).when(pl.col("distance") <= 1500).then(1)
        .when(pl.col("distance") <= 1900).then(2).when(pl.col("distance") <= 2400).then(3).otherwise(4).alias("dist_band"),
        ((pl.col("po_first") - 1) / (pl.col("n_fin") - 1)).clip(0, 1).alias("pos_early"),
        ((pl.col("po_last") - 1) / (pl.col("n_fin") - 1)).clip(0, 1).alias("pos_late"),
        (pl.col("pace_first") - pl.col("pace_second")).alias("pace_bal"),
    ).drop("pace", "passing_order")
    return df.sort(["date", "start_time", "race_id", "horse_number"])


_RAW = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\b.*?\((\d{2}\.\d)\)\s+(\d{3})\(([+-]?\d+)\)")


def build_repair_map(layer_a_path: str | Path) -> pl.DataFrame:
    """出馬表 past_races（直前数走）から (horse, 走の日付) ごとの last_3f・馬体重を復元する。
    各走について『その次走の出走時点』に最初に現れる値だけを採用する（次走以降の対象にしか効かない＝as-of 安全）。"""
    la = pl.read_parquet(layer_a_path, columns=["race_id", "date", "horse_id", "past_races"])
    rows = []
    for d, h, pr in zip(la["date"].to_list(), la["horse_id"].to_list(), la["past_races"].to_list()):
        try:
            lst = json.loads(pr)
        except Exception:
            continue
        for e in lst:
            m = _RAW.match(e.get("raw", ""))
            if m:
                rows.append((h, date(int(m[1]), int(m[2]), int(m[3])), float(m[4]), float(m[5]), d))
    if not rows:
        return pl.DataFrame(schema={"horse_id": pl.Utf8, "run_date": pl.Date, "l3f_fix": pl.Float64,
                                    "weight_fix": pl.Float64, "src_date": pl.Date})
    rm = pl.DataFrame(rows, schema=["horse_id", "run_date", "l3f_fix", "weight_fix", "src_date"], orient="row")
    rm = rm.with_columns(pl.col("src_date").str.to_date("%Y-%m-%d"))
    return rm.sort("src_date").unique(["horse_id", "run_date"], keep="first")


def apply_repair(runs: pl.DataFrame, rm: pl.DataFrame, max_date: date | None = None) -> pl.DataFrame:
    if max_date is not None:
        rm = rm.filter(pl.col("src_date") <= max_date)
    j = runs.join(rm.rename({"run_date": "date"}), on=["horse_id", "date"], how="left")
    ok = lambda c: (pl.col(c) >= 30) & (pl.col(c) <= 50)
    okw = (pl.col("weight") >= 300) & (pl.col("weight") <= 700)
    return j.with_columns(
        pl.when(ok("last_3f")).then(pl.col("last_3f")).when(ok("l3f_fix")).then(pl.col("l3f_fix")).otherwise(None).alias("l3f"),
        pl.when(okw).then(pl.col("weight")).when(pl.col("weight_fix").is_between(300, 700)).then(pl.col("weight_fix")).otherwise(None).alias("wt"),
        ok("last_3f").alias("l3f_raw_ok"),
    ).drop("last_3f", "weight")
