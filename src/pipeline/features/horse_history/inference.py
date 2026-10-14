"""推論モード: 当日の出走馬（結果なし）の特徴量を、学習と同じ計算で作る。

出馬表の行を「結果欄がプレースホルダーの走」として履歴の末尾に足し、学習と同じ ``build_from_runs`` を通す。
今日の行の特徴量は今日の結果に依存しない（``permutation_check``・摂動法で確認済み）ため、プレースホルダーの値は結果に影響しない。
``tests/pipeline/test_horse_history_inference.py`` と ``src/scripts/data/verify_inference_features.py`` が、
実際の結果で作った学習時の特徴量と一致することを確認する（``sr_var_today`` は同日の先行レースの時計を使うため例外）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from .builder import build_from_runs
from .common import RUN_COLS, TABLES_DIR, YEARS, clean_runs, read_runs_raw

TIERS_PATH = Path(__file__).resolve().parents[4] / "config" / "modeling" / "feature_tiers.json"

# 出馬表（race_shutuba_flat）から取る列。結果欄（着順・時計・上り・通過順・ペース・ラップ）は持たない。
CARD_COLS = [c for c in RUN_COLS if c not in ("finish_position", "time_sec", "passing_order", "last_3f", "pace", "lap_times")]


def load_tiers(path: Path = TIERS_PATH) -> dict[str, int]:
    return {k: int(v) for k, v in json.loads(path.read_text(encoding="utf-8"))["features"].items()}


_GOING_FULL = {"稍": "稍重", "不": "不良"}
_CLASS_GRADE = [("未勝利", "未勝利"), ("新馬", "新馬"), ("１勝", "1勝"), ("２勝", "2勝"), ("３勝", "3勝")]


def graded_race_names(tables_dir: str | Path | None = None, years: list[int] | None = None) -> dict[str, str]:
    """重賞・リステッドのレース名 → 直近の格（G1/G2/G3/L）。出馬表の格は重賞も ``OP`` としか書かれないため、履歴から引く。"""
    base = Path(tables_dir) if tables_dir else TABLES_DIR
    ys = [int(y) for y in years] if years else YEARS
    want = ["race_id", "race_name", "grade"]
    frames = []
    for y in ys:
        path = base / str(y) / "race_result_flat.parquet"
        if path.exists() and set(want) <= set(pl.read_parquet_schema(path)):
            frames.append(pl.read_parquet(path, columns=want))
    if not frames:
        return {}
    df = pl.concat(frames)
    df = df.filter(pl.col("grade").is_in(["G1", "G2", "G3", "L"]) & pl.col("race_name").is_not_null())
    last = df.group_by("race_id").agg(pl.col("race_name").first(), pl.col("grade").first()).sort("race_id", descending=True)
    # race_id の新しい順に並べているので、逆順に辞書へ入れて最新の格が残るようにする
    return {n: g for n, g in reversed(list(zip(last["race_name"].to_list(), last["grade"].to_list())))}


def normalize_card(card: pl.DataFrame, graded_names: dict[str, str] | None = None) -> pl.DataFrame:
    """出馬表と結果表の表記の違いを、結果表（学習時）に合わせる。

    - 馬場状態: 出馬表は略称（稍・不）→ 稍重・不良（略称のままだと良に落ちる）
    - 格（grade）: 1〜3勝クラスは空欄 → race_class から導く。重賞は ``OP`` → レース名から G1/G2/G3/L を引く（不明なら ``OP`` のまま）
    """
    df = card
    if "track_condition" in df.columns:
        df = df.with_columns(pl.col("track_condition").replace(_GOING_FULL))
    if "grade" in df.columns and "race_class" in df.columns:
        derived = pl.lit(None, pl.Utf8)
        for key, g in reversed(_CLASS_GRADE):
            derived = pl.when(pl.col("race_class").fill_null("").str.contains(key)).then(pl.lit(g)).otherwise(derived)
        df = df.with_columns(pl.when(pl.col("grade").fill_null("") == "").then(derived).otherwise(pl.col("grade")).alias("grade"))
    if graded_names and "race_name" in df.columns and "grade" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("grade") == "OP").then(pl.col("race_name").replace_strict(graded_names, default="OP", return_dtype=pl.Utf8))
            .otherwise(pl.col("grade")).alias("grade"))
    return df


def card_to_raw_runs(card: pl.DataFrame, graded_names: dict[str, str] | None = None) -> pl.DataFrame:
    """出馬表の行を、``read_runs_raw`` と同じ列の走（結果欄はプレースホルダー）にする。"""
    card = normalize_card(card, graded_names)
    df = card.with_columns([pl.lit(None).alias(c) for c in CARD_COLS if c not in card.columns]).select(CARD_COLS)
    df = df.filter(pl.col("horse_id").is_not_null() & (pl.col("horse_number") > 0))
    return df.sort(["race_id", "horse_number"]).with_columns(
        # 着順は未確定。Elo などの更新順にだけ使われ、今日の行の特徴量には影響しない。
        pl.int_range(1, pl.len() + 1).over("race_id").cast(pl.Int64).alias("finish_position"),
        pl.lit(None, pl.Float64).alias("time_sec"),
        pl.lit(None, pl.Float64).alias("last_3f"),
        pl.lit(None, pl.Utf8).alias("passing_order"),
        pl.lit(None, pl.Utf8).alias("pace"),
        pl.lit(None, pl.Utf8).alias("lap_times"),
    )


def build_inference_features(
    raw_hist: pl.DataFrame, card: pl.DataFrame, repair_map: pl.DataFrame, on_time: pl.DataFrame | None = None,
    graded_names: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """``card`` に含まれるレースの特徴量表（学習と同じ列）と、計測値を返す。

    ``raw_hist`` は ``read_runs_raw`` の結果。同じ race_id が履歴にあれば（結果の取り込み後に再実行した場合）履歴側を除く。
    """
    t0 = time.perf_counter()
    today = card_to_raw_runs(card, graded_names)
    race_ids = today["race_id"].unique().to_list()
    hist = raw_hist.filter(~pl.col("race_id").is_in(race_ids))
    # 今日の走より後の日付の履歴は使わない（再演・検証時に、後日のデータが混ざらないようにする）。
    cutoff = today["date"].min()
    hist = hist.filter(pl.col("date") < cutoff)
    runs = clean_runs(pl.concat([hist, today.select(hist.columns)], how="diagonal_relaxed"))
    F, P = build_from_runs(runs, repair_map, on_time)
    out = F.filter(pl.col("race_id").is_in(race_ids))
    info = {
        "races": len(race_ids),
        "rows": out.height,
        "history_rows": hist.height,
        "weights_known_ratio": float((today["weight"].cast(pl.Float64) > 0).fill_null(False).mean()) if today.height else 0.0,
        "going_known": bool(today["track_condition"].is_not_null().all()) if today.height else False,
        "build_sec": round(time.perf_counter() - t0, 1),
    }
    return out, info
