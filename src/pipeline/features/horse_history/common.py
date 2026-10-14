"""馬の過去走履歴特徴量の共通部（読み込み・クリーニング・last_3f 復元）。"""
from __future__ import annotations
import json, re
from datetime import date
import numpy as np
import polars as pl

from pathlib import Path

from src.config.data_paths import PAGE_REF
from src.pipeline.models.training_years import TRAINING_YEARS

TABLES_DIR = Path(PAGE_REF) / "tables"
YEARS = [int(y) for y in TRAINING_YEARS]   # 2026 年は使わない
# 縮約強度・分位点・クラスタなどの「全体フィット」は、この日以前のデータだけで行う。
# 時系列 CV の最初の fold（valid 2022）の学習終端 2021-12-22（purge+embargo 後）より前に置く。
FIT_END = date(2021, 12, 21)
CLASS_LEVEL = {"新馬": 0.0, "未勝利": 0.0, "1勝": 1.0, "2勝": 2.0, "3勝": 3.0, "OP": 4.0, "L": 4.5, "G3": 5.0, "G2": 6.0, "G1": 7.0}
GOING = {"良": 0, "稍重": 1, "重": 2, "不良": 3}

RUN_COLS = ["race_id", "date", "start_time", "venue_code", "surface", "distance", "direction", "track_condition",
            "grade", "pace", "finish_position", "horse_number", "horse_id", "sex_age", "time_sec",
            "passing_order", "last_3f", "weight", "jockey_id", "trainer_id", "jockey_weight", "lap_times", "race_class"]


def _rpci_from_laps(s: str | None) -> float | None:
    """RPCI 相当: 50 × （ラスト 3F を除く 1F 平均 ÷ ラスト 3F の 1F 平均）。小さいほど前傾（ハイペース）、大きいほど後傾（スロー）。
    JRA の RPCI の公式式は未確認のため、同じ考え方の自前定義（pc_ 系の列で使う）。"""
    if not s:
        return None
    try:
        laps = [float(x) for x in json.loads(s)]
    except (ValueError, TypeError):
        return None
    if len(laps) < 5 or any(x < 8 or x > 20 for x in laps):
        return None
    return 50.0 * (sum(laps[:-3]) / (len(laps) - 3)) / (sum(laps[-3:]) / 3)


def read_runs_raw(tables_dir: str | Path | None = None, years: list[int] | None = None) -> pl.DataFrame:
    """race_result_flat を読むだけ（整形前）。years 省略時は学習用の 2020〜2025 年。"""
    base = Path(tables_dir) if tables_dir else TABLES_DIR
    ys = [int(y) for y in years] if years else YEARS
    # 距離は年によって Float64 / Int64 に分かれていることがある。型を揃えてから連結する
    return pl.concat([pl.read_parquet(base / str(y) / "race_result_flat.parquet", columns=RUN_COLS).with_columns(pl.col("distance").cast(pl.Int64))
                      for y in ys])


def load_runs(tables_dir: str | Path | None = None, years: list[int] | None = None) -> pl.DataFrame:
    return clean_runs(read_runs_raw(tables_dir, years))


def clean_runs(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter((pl.col("finish_position") > 0) & pl.col("venue_code").cast(pl.Int32, strict=False).is_between(1, 10))
    df = df.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d"),
        # 障害レースは race_class が空。面が「芝」「ダート」と誤ってラベルされた障害レース（約 590 レース）も「障」にする。
        pl.when(pl.col("race_class").fill_null("") == "").then(pl.lit("障")).otherwise(pl.col("surface").fill_null("障")).alias("surface"),
        pl.len().over("race_id").alias("n_fin"),
        pl.col("track_condition").replace_strict(GOING, default=0, return_dtype=pl.Int32).alias("going"),
        pl.col("grade").replace_strict(CLASS_LEVEL, default=1.0, return_dtype=pl.Float64).alias("class_level"),
        pl.col("sex_age").str.slice(1).cast(pl.Int32, strict=False).alias("age"),
        pl.col("sex_age").str.slice(0, 1).alias("sex"),
        pl.col("pace").str.json_path_match("$.first_half_3f").cast(pl.Float64, strict=False).alias("pace_first"),
        pl.col("pace").str.json_path_match("$.second_half_3f").cast(pl.Float64, strict=False).alias("pace_second"),
        (pl.col("distance").cast(pl.Float64) / 1600.0).log().alias("logd"),
        pl.col("lap_times").map_elements(_rpci_from_laps, return_dtype=pl.Float64, skip_nulls=False).alias("rpci"),
    ).with_columns(
        pl.when(pl.col("distance") <= 1200).then(0).when(pl.col("distance") <= 1500).then(1)
        .when(pl.col("distance") <= 1900).then(2).when(pl.col("distance") <= 2400).then(3).otherwise(4).alias("dist_band"),
        (pl.col("pace_first") - pl.col("pace_second")).alias("pace_bal"),
    ).drop("pace", "lap_times", "race_class")
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


_PO_RE = r"^\d+(-\d+)*$"


def _valid_passing(n_fin_col: str, col: str) -> pl.Expr:
    """通過順が '3-3-2' 形式で、全ての数字が完走頭数以内であること（'105' や '' は不正）。"""
    nums = pl.col(col).str.extract_all(r"\d+").list.eval(pl.element().cast(pl.Int64, strict=False))
    return pl.col(col).str.contains(_PO_RE).fill_null(False) & (nums.list.max() <= pl.col(n_fin_col)).fill_null(False)


def apply_repair(runs: pl.DataFrame, rm: pl.DataFrame, on_time: pl.DataFrame | None = None,
                 max_date: date | None = None) -> pl.DataFrame:
    """last_3f・馬体重・通過順の不正値を補う。優先順: 生値 → race_result_on_time → past_races。
    on_time は (race_id, horse_number, ot_l3f, ot_passing_order, ot_weight)。いずれも当該レース自身の公式結果で、
    以降のレースの履歴としてだけ使われるため as-of 安全。"""
    if max_date is not None:
        rm = rm.filter(pl.col("src_date") <= max_date)
    j = runs.join(rm.rename({"run_date": "date"}), on=["horse_id", "date"], how="left")
    if on_time is not None and on_time.height:
        ot = on_time.with_columns(pl.col("horse_number").cast(pl.Int64))
        j = j.with_columns(pl.col("horse_number").cast(pl.Int64)).join(ot, on=["race_id", "horse_number"], how="left")
    else:
        j = j.with_columns(pl.lit(None, pl.Float64).alias("ot_l3f"), pl.lit(None, pl.Utf8).alias("ot_passing_order"),
                           pl.lit(None, pl.Float64).alias("ot_weight"))
    ok = lambda c: (pl.col(c) >= 30) & (pl.col(c) <= 50)
    okw = lambda c: (pl.col(c) >= 300) & (pl.col(c) <= 700)
    j = j.with_columns(
        pl.when(ok("last_3f")).then(pl.col("last_3f")).when(ok("ot_l3f")).then(pl.col("ot_l3f"))
        .when(ok("l3f_fix")).then(pl.col("l3f_fix")).otherwise(None).alias("l3f"),
        pl.when(okw("weight")).then(pl.col("weight")).when(okw("ot_weight")).then(pl.col("ot_weight"))
        .when(okw("weight_fix")).then(pl.col("weight_fix")).otherwise(None).alias("wt"),
        ok("last_3f").alias("l3f_raw_ok"),
        _valid_passing("n_fin", "passing_order").alias("po_raw_ok"),
    )
    j = j.with_columns(
        pl.when(pl.col("po_raw_ok")).then(pl.col("passing_order"))
        .when(_valid_passing("n_fin", "ot_passing_order")).then(pl.col("ot_passing_order")).otherwise(None).alias("po"),
    ).with_columns(
        pl.col("po").str.extract(r"^(\d+)", 1).cast(pl.Float64).alias("po_first"),
        pl.col("po").str.extract(r"(\d+)$", 1).cast(pl.Float64).alias("po_last"),
    ).with_columns(
        ((pl.col("po_first") - 1) / (pl.col("n_fin") - 1)).clip(0, 1).alias("pos_early"),
        ((pl.col("po_last") - 1) / (pl.col("n_fin") - 1)).clip(0, 1).alias("pos_late"),
    )
    return j.drop("last_3f", "weight", "l3f_fix", "weight_fix", "ot_l3f", "ot_passing_order", "ot_weight", "po", "passing_order")
