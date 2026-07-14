"""レース結果から馬場速度（基準2着タイムとの乖離）を算出する。"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

CLASS_RANK = {
    "未勝利": 1,
    "新馬": 1,
    "1勝": 2,
    "2勝": 3,
    "3勝": 4,
    "OP": 5,
    "L": 5,
    "G3": 6,
    "G2": 6,
    "G1": 7,
}

CLASS_GROUP_BY_RANK = {
    1: "未勝利",
    2: "1-3勝",
    3: "1-3勝",
    4: "1-3勝",
    5: "OP",
    6: "OP",
    7: "OP",
}

TRACK_SPEED_SURFACES = ("芝", "ダート")
# 障害（surface=障）などは馬場の速さによる補正を行わない


def needs_track_speed_correction(surface) -> bool:
    """芝・ダートのみ馬場速さ補正の対象。障害は False。"""
    return str(surface).strip() in TRACK_SPEED_SURFACES

PAR_POOL_LEVELS: tuple[tuple[str, ...], ...] = (
    ("venue", "surface", "distance", "class_group"),
    ("surface", "distance", "class_group"),
)


def assign_class_rank(grade, race_class) -> int:
    """grade / race_class から class_rank (1-7) を返す。"""
    grade_str = str(grade) if pd.notna(grade) else ""
    if grade_str in CLASS_RANK:
        return CLASS_RANK[grade_str]
    for key in ["G1", "G2", "G3", "L", "OP", "3勝", "2勝", "1勝", "未勝利", "新馬"]:
        if key in grade_str:
            return CLASS_RANK[key]
    rc = str(race_class) if pd.notna(race_class) else ""
    if "新馬" in rc:
        return CLASS_RANK["新馬"]
    if "未勝利" in rc:
        return CLASS_RANK["未勝利"]
    if "1勝" in rc:
        return CLASS_RANK["1勝"]
    if "2勝" in rc:
        return CLASS_RANK["2勝"]
    if "3勝" in rc:
        return CLASS_RANK["3勝"]
    if "G1" in rc:
        return CLASS_RANK["G1"]
    if "G2" in rc:
        return CLASS_RANK["G2"]
    if "G3" in rc:
        return CLASS_RANK["G3"]
    if "OP" in rc or "オープン" in rc:
        return CLASS_RANK["OP"]
    return 4


def assign_class_group(grade, race_class) -> str:
    """class_rank を 未勝利 / 1-3勝 / OP の3区分に集約。"""
    return CLASS_GROUP_BY_RANK[assign_class_rank(grade, race_class)]



TRACK_OUTPUT_COLS = (
    "date_str",
    "class_group",
    "par_2nd_adj_sec",
    "par_pool_level",
    "race_t2nd_sec",
    "race_track_dev_sec",
    "track_dev_sec",
    "tsi_raw",
    "n_races_track",
)


def build_race_table(df: pd.DataFrame) -> pd.DataFrame:
    """レース単位テーブル（2着補正タイム付き）。芝・ダートのみ（障害は基準・乖離の対象外）。"""
    needed = {
        "race_id",
        "date",
        "venue",
        "surface",
        "distance",
        "grade",
        "race_class",
        "finish_pos",
        "adjusted_time_sec",
    }
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"build_race_table: missing columns {sorted(missing)}")

    work = df[df["surface"].isin(TRACK_SPEED_SURFACES)].copy()
    if "finish_pos" not in work.columns and "finish_position" in work.columns:
        work["finish_pos"] = pd.to_numeric(work["finish_position"], errors="coerce")
    work["finish_pos"] = pd.to_numeric(work["finish_pos"], errors="coerce")
    work["distance"] = pd.to_numeric(work["distance"], errors="coerce")
    work["date_str"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work["class_group"] = work.apply(
        lambda r: assign_class_group(r["grade"], r["race_class"]),
        axis=1,
    )

    t2 = (
        work[work["finish_pos"] == 2][["race_id", "adjusted_time_sec"]]
        .drop_duplicates("race_id")
        .rename(columns={"adjusted_time_sec": "t2nd_adj_sec"})
    )
    t1 = (
        work[work["finish_pos"] == 1][["race_id", "adjusted_time_sec"]]
        .drop_duplicates("race_id")
        .rename(columns={"adjusted_time_sec": "t1_adj_sec"})
    )
    races = work.drop_duplicates("race_id")[
        ["race_id", "date_str", "venue", "surface", "distance", "class_group"]
    ].copy()
    races = races.merge(t2, on="race_id", how="left").merge(t1, on="race_id", how="left")
    races["t2nd_adj_sec"] = races["t2nd_adj_sec"].fillna(races["t1_adj_sec"])
    return races.drop(columns=["t1_adj_sec"])


def _par_lookup(train_races: pd.DataFrame, keys: Sequence[str], min_samples: int) -> pd.DataFrame:
    stats = (
        train_races.dropna(subset=["t2nd_adj_sec", "distance", "class_group"])
        .groupby(list(keys), dropna=False)["t2nd_adj_sec"]
        .agg(par_2nd_adj_sec="mean", n_races="count")
        .reset_index()
    )
    stats.loc[stats["n_races"] < min_samples, "par_2nd_adj_sec"] = np.nan
    return stats[list(keys) + ["par_2nd_adj_sec", "n_races"]]


def attach_par_2nd_baseline(
    races: pd.DataFrame,
    train_mask: pd.Series | np.ndarray,
    min_samples: int = 30,
    pool_levels: Iterable[Sequence[str]] = PAR_POOL_LEVELS,
) -> pd.DataFrame:
    """学習期間レースから基準2着タイムを推定し、全レースへ付与。"""
    out = races.copy()
    out["par_2nd_adj_sec"] = np.nan
    out["par_pool_level"] = np.nan
    train = out.loc[train_mask].copy()

    for level_idx, keys in enumerate(pool_levels, start=1):
        lookup = _par_lookup(train, keys, min_samples)
        if lookup.empty:
            continue
        merged = out[list(keys)].merge(
            lookup[list(keys) + ["par_2nd_adj_sec"]],
            on=list(keys),
            how="left",
        )
        fill = out["par_2nd_adj_sec"].isna() & merged["par_2nd_adj_sec"].notna()
        out.loc[fill, "par_2nd_adj_sec"] = merged.loc[fill, "par_2nd_adj_sec"]
        out.loc[fill, "par_pool_level"] = level_idx
    return out


def aggregate_day_course_track_dev(races: pd.DataFrame) -> pd.DataFrame:
    """日×競馬場×馬場（surface）の馬場乖離を集計（direction は使わない）。"""
    work = races.dropna(subset=["race_track_dev_sec"]).copy()
    return (
        work.groupby(["date_str", "venue", "surface"], dropna=False)
        .agg(
            track_dev_sec=("race_track_dev_sec", "mean"),
            n_races_track=("race_id", "nunique"),
        )
        .reset_index()
    )


def attach_track_speed_to_horses(
    df: pd.DataFrame,
    train_years: Sequence[int],
    min_samples: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    馬単位 DataFrame に馬場速度指標を付与する。

    戻り値: (df_with_cols, race_table, day_course_table)
    - race_track_dev_sec: 当該レースの2着補正タイム − 基準（秒）。正=遅い馬場
    - track_dev_sec: 当日コース平均の race_track_dev_sec
    - tsi_raw: -track_dev_sec（正=時計が出やすい／速い馬場）
    """
    races = build_race_table(df)
    if "year" in df.columns:
        year_by_race = df.drop_duplicates("race_id").set_index("race_id")["year"]
        races["year"] = races["race_id"].map(year_by_race)
    else:
        races["year"] = pd.to_datetime(races["date_str"], errors="coerce").dt.year

    train_mask = races["year"].isin(list(train_years))
    races = attach_par_2nd_baseline(races, train_mask, min_samples=min_samples)
    races["race_track_dev_sec"] = races["t2nd_adj_sec"] - races["par_2nd_adj_sec"]
    races["race_t2nd_sec"] = races["t2nd_adj_sec"]

    day_course = aggregate_day_course_track_dev(races)
    day_course["tsi_raw"] = -day_course["track_dev_sec"]

    out = df.copy()
    drop_cols = [c for c in TRACK_OUTPUT_COLS if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out["date_str"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    race_cols = races[
        [
            "race_id",
            "class_group",
            "par_2nd_adj_sec",
            "par_pool_level",
            "race_t2nd_sec",
            "race_track_dev_sec",
        ]
    ]
    out = out.merge(race_cols, on="race_id", how="left")
    out = out.merge(
        day_course[["date_str", "venue", "surface", "track_dev_sec", "tsi_raw", "n_races_track"]],
        on=["date_str", "venue", "surface"],
        how="left",
        suffixes=("", "_day"),
    )

    # 障害など芝・ダート以外: 馬場速さ指標・補正は付与しない（tsi_raw=NaN → 下流で tsi_mean 埋め = 補正0）
    no_track = ~out["surface"].isin(TRACK_SPEED_SURFACES)
    for col in (
        "race_t2nd_sec",
        "race_track_dev_sec",
        "par_2nd_adj_sec",
        "par_pool_level",
        "class_group",
        "track_dev_sec",
        "tsi_raw",
    ):
        if col in out.columns:
            out.loc[no_track, col] = np.nan
    if "n_races_track" in out.columns:
        out.loc[no_track, "n_races_track"] = np.nan

    return out, races, day_course
