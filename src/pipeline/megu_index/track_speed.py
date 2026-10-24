"""レース結果から馬場速度（基準2着タイムとの乖離）を算出する。

3層設計:
  Layer 1 — ペース異常レースの除外 (front_split_dev フィルタ)
  Layer 2 — ロバスト集計 (中央値)
  Layer 3 — 収縮推定 (小サンプル安定化)
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ── クラス区分 ─────────────────────────────────────────────────────────────

CLASS_RANK = {
    "未勝利": 1,
    # 新馬 は class_rank を持たない（assign_class_rank が None を返す）
    "1勝": 2,
    "2勝": 3,
    "3勝": 4,
    "OP": 5,
    "L": 5,
    "G3": 6,
    "G2": 7,
    "G1": 8,
}

CLASS_GROUP_BY_RANK = {
    1: "未勝利",
    2: "1勝",
    3: "2勝",
    4: "3勝",
    5: "G3orOP",
    6: "G3orOP",
    7: "G1orG2",
    8: "G1orG2",
}

# 2歳・3歳（「以上」を伴わない）世代限定戦を検出する正規表現
# 例: 'サラ系２歳 未勝利', '3歳オープン' → 除外  /  '3歳以上' → 対象外
_AGE_LIMIT_PATTERN = re.compile(r"[２2]歳(?!以上)|[３3]歳(?!以上)")

TRACK_SPEED_SURFACES = ("芝", "ダート")

# ── ペースフィルタ / 収縮推定 設定 ─────────────────────────────────────────
MIN_FRONT_SPLIT_SAMPLES: int = 10       # pace params 推定の最小サンプル数
PACE_FILTER_N_SIGMA: float = 2.0        # |front_split_dev| > n_sigma × σ で除外
DEFAULT_K_SHRINKAGE: float = 5.0        # 収縮定数のデフォルト値（σ推定失敗時）
K_SHRINKAGE_CLIP: tuple[float, float] = (2.0, 10.0)  # k のクリップ範囲

# ── 基準2着タイムのプールレベル ────────────────────────────────────────────
PAR_POOL_LEVELS: tuple[tuple[str, ...], ...] = (
    ("venue", "surface", "distance", "class_group"),
    ("surface", "distance", "class_group"),
)

# ── 馬単位出力列（ドロップ対象として管理） ────────────────────────────────
TRACK_OUTPUT_COLS = (
    "date_str",
    "class_group",
    "par_2nd_adj_sec",
    "par_pool_level",
    "race_t2nd_sec",
    "race_track_dev_sec",
    "front_split_dev",
    "track_dev_sec",
    "tsi_raw",
    "n_valid_races",
    "n_races_track",
)


# ── クラス判定ユーティリティ ───────────────────────────────────────────────

def needs_track_speed_correction(surface) -> bool:
    """芝・ダートのみ馬場速さ補正の対象。障害は False。"""
    return str(surface).strip() in TRACK_SPEED_SURFACES


def is_age_restricted(race_class, race_name=None) -> bool:
    """2歳・3歳（「以上」を伴わない）世代限定戦かどうかを判定する。

    対象例: 'サラ系２歳 未勝利', '3歳オープン'
    除外例: '3歳以上1勝クラス'（世代混合なので対象外）
    """
    for text in (race_class, race_name):
        if text is not None and _AGE_LIMIT_PATTERN.search(str(text)):
            return True
    return False


def assign_class_rank(grade, race_class) -> int | None:
    """grade / race_class から class_rank (1-8) を返す。新馬は None。"""
    grade_str = str(grade) if pd.notna(grade) else ""
    if grade_str == "新馬":
        return None
    if grade_str in CLASS_RANK:
        return CLASS_RANK[grade_str]
    for key in ["G1", "G2", "G3", "L", "OP", "3勝", "2勝", "1勝", "未勝利"]:
        if key in grade_str:
            return CLASS_RANK[key]
    rc = str(race_class) if pd.notna(race_class) else ""
    if "新馬" in rc:    return None
    if "未勝利" in rc:  return CLASS_RANK["未勝利"]
    if "1勝" in rc:     return CLASS_RANK["1勝"]
    if "2勝" in rc:     return CLASS_RANK["2勝"]
    if "3勝" in rc:     return CLASS_RANK["3勝"]
    if "G1" in rc:      return CLASS_RANK["G1"]
    if "G2" in rc:      return CLASS_RANK["G2"]
    if "G3" in rc:      return CLASS_RANK["G3"]
    if "OP" in rc or "オープン" in rc:
        return CLASS_RANK["OP"]
    return 4


def assign_class_group(grade, race_class, race_name=None) -> str | None:
    """class_rank を G1orG2 / G3orOP / 3勝 / 2勝 / 1勝 / 未勝利 の6区分に集約。

    新馬・世代限定戦（2歳・3歳限定）は None を返す。
    """
    if is_age_restricted(race_class, race_name):
        return None
    rank = assign_class_rank(grade, race_class)
    if rank is None:
        return None
    return CLASS_GROUP_BY_RANK.get(rank)


# ── レース単位テーブル構築 ─────────────────────────────────────────────────

def build_race_table(df: pd.DataFrame) -> pd.DataFrame:
    """馬単位 DataFrame からレース単位テーブルを作成する。

    race_name 列があれば世代限定戦の検出に使用する。
    front_split_sec 列があれば Layer 1（ペースフィルタ）用に引き継ぐ。
    """
    needed = {
        "race_id", "date", "venue", "surface", "distance",
        "grade", "race_class", "finish_pos", "adjusted_time_sec",
    }
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"build_race_table: missing columns {sorted(missing)}")

    has_race_name   = "race_name"      in df.columns
    has_front_split = "front_split_sec" in df.columns

    work = df[df["surface"].isin(TRACK_SPEED_SURFACES)].copy()
    if "finish_pos" not in work.columns and "finish_position" in work.columns:
        work["finish_pos"] = pd.to_numeric(work["finish_position"], errors="coerce")
    work["finish_pos"] = pd.to_numeric(work["finish_pos"], errors="coerce")
    work["distance"]   = pd.to_numeric(work["distance"],   errors="coerce")
    work["date_str"]   = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if has_race_name:
        work["class_group"] = work.apply(
            lambda r: assign_class_group(r["grade"], r["race_class"], r["race_name"]),
            axis=1,
        )
    else:
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

    race_cols = ["race_id", "date_str", "venue", "surface", "distance", "class_group"]
    if has_front_split:
        race_cols.append("front_split_sec")

    races = work.drop_duplicates("race_id")[race_cols].copy()
    races = races.merge(t2, on="race_id", how="left").merge(t1, on="race_id", how="left")
    races["t2nd_adj_sec"] = races["t2nd_adj_sec"].fillna(races["t1_adj_sec"])
    return races.drop(columns=["t1_adj_sec"])


# ── 基準2着タイム ──────────────────────────────────────────────────────────

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
    """学習期間レースから基準2着タイムを推定し、全レースへ付与する。"""
    out = races.copy()
    out["par_2nd_adj_sec"] = np.nan
    out["par_pool_level"]  = np.nan
    train = out.loc[train_mask].copy()

    for level_idx, keys in enumerate(pool_levels, start=1):
        lookup = _par_lookup(train, keys, min_samples)
        if lookup.empty:
            continue
        merged = out[list(keys)].merge(
            lookup[list(keys) + ["par_2nd_adj_sec"]], on=list(keys), how="left",
        )
        fill = out["par_2nd_adj_sec"].isna() & merged["par_2nd_adj_sec"].notna()
        out.loc[fill, "par_2nd_adj_sec"] = merged.loc[fill, "par_2nd_adj_sec"]
        out.loc[fill, "par_pool_level"]  = level_idx
    return out


# ── Layer 1: ペースフィルタ ────────────────────────────────────────────────

def build_pace_filter_params(df_train: pd.DataFrame) -> pd.DataFrame:
    """学習期の distance × surface 別に front_split_sec の中央値と σ を計算する。

    サンプル数が MIN_FRONT_SPLIT_SAMPLES 未満のセルは NaN にしてフィルタ無効化。

    Returns
    -------
    DataFrame[distance, surface, front_split_median, front_split_sigma]
    """
    if "front_split_sec" not in df_train.columns:
        return pd.DataFrame(
            columns=["distance", "surface", "front_split_median", "front_split_sigma"]
        )

    work = (
        df_train
        .dropna(subset=["front_split_sec", "distance", "surface"])
        .copy()
    )
    work = work[work["surface"].isin(TRACK_SPEED_SURFACES)]
    work["distance"] = pd.to_numeric(work["distance"], errors="coerce")
    work = work.dropna(subset=["distance"])

    grp = work.groupby(["distance", "surface"])["front_split_sec"]
    params = grp.agg(
        front_split_median="median",
        front_split_sigma="std",
        _n="count",
    ).reset_index()

    low_n = params["_n"] < MIN_FRONT_SPLIT_SAMPLES
    params.loc[low_n, ["front_split_median", "front_split_sigma"]] = np.nan
    return params[["distance", "surface", "front_split_median", "front_split_sigma"]]


def attach_pace_filter(
    races: pd.DataFrame,
    pace_params: pd.DataFrame,
    n_sigma: float = PACE_FILTER_N_SIGMA,
) -> pd.DataFrame:
    """races に front_split_dev と is_pace_valid を付与する（Layer 1）。

    front_split_sec が races に存在しない、または pace_params が空の場合は
    全レースを valid として処理を続行する（後方互換）。
    """
    out = races.copy()

    if "front_split_sec" not in races.columns or pace_params.empty:
        out["front_split_dev"] = np.nan
        out["is_pace_valid"]   = True
        return out

    out = out.merge(pace_params, on=["distance", "surface"], how="left")
    out["front_split_dev"] = out["front_split_sec"] - out["front_split_median"]

    # σ が NaN（サンプル不足セル）はフィルタ適用しない → valid のまま
    sigma_ok  = out["front_split_sigma"].notna()
    exceeds   = out["front_split_dev"].abs() > n_sigma * out["front_split_sigma"]
    out["is_pace_valid"] = ~(sigma_ok & exceeds)

    return out.drop(columns=["front_split_median", "front_split_sigma"], errors="ignore")


# ── Layer 2+3: ロバスト集計 + 収縮推定 ─────────────────────────────────────

def _aggregate_day_course_simple(races: pd.DataFrame) -> pd.DataFrame:
    """収縮パラメータ推定専用の単純集計（中央値、収縮なし）。"""
    work = races.dropna(subset=["race_track_dev_sec"])
    if "is_pace_valid" in races.columns:
        work = work[work["is_pace_valid"]]
    return (
        work.groupby(["date_str", "venue", "surface"], dropna=False)
        .agg(
            track_dev_sec=("race_track_dev_sec", "median"),
            n_races_track=("race_id", "nunique"),
        )
        .reset_index()
    )


def build_shrinkage_params(
    train_races: pd.DataFrame,
    train_day_course: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """収縮推定のパラメータを学習期データから推定する。

    k = σ²_within / σ²_between
      σ²_within : 同日・同venue・同surface 内の race_track_dev_sec のばらつき
      σ²_between: 日ごとの track_dev_sec のばらつき（venue×surface ごと）

    Returns
    -------
    venue_prior : DataFrame[venue, surface, tsi_prior]  (tsi_prior = 平均 track_dev_sec)
    k           : 収縮定数 (float)
    """
    # tsi_prior: venue×surface 別の学習期平均 track_dev_sec
    venue_prior = (
        train_day_course.dropna(subset=["track_dev_sec"])
        .groupby(["venue", "surface"])["track_dev_sec"]
        .mean()
        .reset_index()
        .rename(columns={"track_dev_sec": "tsi_prior"})
    )

    # σ²_within: ペース有効レースの日内分散の平均
    work = train_races.dropna(subset=["race_track_dev_sec"])
    if "is_pace_valid" in train_races.columns:
        work = work[work["is_pace_valid"]]
    within_var = (
        work
        .groupby(["date_str", "venue", "surface"])["race_track_dev_sec"]
        .var()
        .dropna()
        .mean()
    )

    # σ²_between: venue×surface ごとの日間 track_dev_sec 分散の平均
    between_var = (
        train_day_course.dropna(subset=["track_dev_sec"])
        .groupby(["venue", "surface"])["track_dev_sec"]
        .var()
        .dropna()
        .mean()
    )

    if pd.isna(within_var) or pd.isna(between_var) or between_var <= 1e-9:
        k = DEFAULT_K_SHRINKAGE
    else:
        k = float(np.clip(within_var / between_var, *K_SHRINKAGE_CLIP))

    return venue_prior, k


def aggregate_day_course_robust(
    races: pd.DataFrame,
    venue_prior: pd.DataFrame | None = None,
    k: float = DEFAULT_K_SHRINKAGE,
) -> pd.DataFrame:
    """日×競馬場×馬場種別の馬場乖離を Layer 2+3 で集計する。

    Layer 2: ペース有効レースの race_track_dev_sec を中央値で集計
    Layer 3: venue×surface の事前値 tsi_prior へ収縮推定
             w = n_valid / (n_valid + k)
             track_dev_sec = w × observed + (1-w) × tsi_prior

    Returns
    -------
    DataFrame[date_str, venue, surface, n_races_track, n_valid_races,
              track_dev_sec, tsi_raw]
    """
    valid_col = "is_pace_valid"

    # n_races_track: race_track_dev_sec が存在する全レース数（フィルタ前）
    all_valid = races.dropna(subset=["race_track_dev_sec"])
    total_counts = (
        all_valid
        .groupby(["date_str", "venue", "surface"], dropna=False)
        .agg(n_races_track=("race_id", "nunique"))
        .reset_index()
    )

    # Layer 2: ペース有効レースの中央値と件数
    if valid_col in races.columns:
        pace_ok = races[races[valid_col]].dropna(subset=["race_track_dev_sec"])
    else:
        pace_ok = all_valid

    layer2 = (
        pace_ok
        .groupby(["date_str", "venue", "surface"], dropna=False)
        .agg(
            race_track_dev_observed=("race_track_dev_sec", "median"),
            n_valid_races=("race_id", "nunique"),
        )
        .reset_index()
    )

    day_course = total_counts.merge(layer2, on=["date_str", "venue", "surface"], how="left")
    day_course["n_valid_races"] = day_course["n_valid_races"].fillna(0).astype(int)

    # prior の付与
    if venue_prior is not None and not venue_prior.empty:
        day_course = day_course.merge(
            venue_prior[["venue", "surface", "tsi_prior"]],
            on=["venue", "surface"],
            how="left",
        )
    else:
        day_course["tsi_prior"] = 0.0
    day_course["tsi_prior"] = day_course["tsi_prior"].fillna(0.0)

    # Layer 3: 収縮推定
    n         = day_course["n_valid_races"].to_numpy(dtype=float)
    w         = n / (n + k)
    observed  = day_course["race_track_dev_observed"].to_numpy(dtype=float)
    prior_arr = day_course["tsi_prior"].to_numpy(dtype=float)

    # n_valid=0（全除外）は prior のみ使用
    day_course["track_dev_sec"] = np.where(
        n > 0,
        w * observed + (1.0 - w) * prior_arr,
        prior_arr,
    )
    day_course["tsi_raw"] = -day_course["track_dev_sec"]

    return day_course[[
        "date_str", "venue", "surface",
        "n_races_track", "n_valid_races",
        "track_dev_sec", "tsi_raw",
    ]]


# ── メインエントリ ─────────────────────────────────────────────────────────

def attach_track_speed_to_horses(
    df: pd.DataFrame,
    train_years: Sequence[int],
    min_samples: int = 30,
    splits_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """馬単位 DataFrame に馬場速度指標を付与する（3層設計）。

    Parameters
    ----------
    df          : 馬単位 DataFrame（adjusted_time_sec 付き）
    train_years : 学習期の年リスト
    min_samples : 基準2着タイム推定の最小サンプル数
    splits_df   : レース単位の front_split_sec テーブル (race_id, front_split_sec)。
                  df に front_split_sec が既に含まれている場合は不要。
                  指定すると Layer 1 のペースフィルタが有効になる。

    Returns
    -------
    (df_with_cols, race_table, day_course_table)
      race_track_dev_sec : 当該レース2着補正タイム − 基準（秒）。正=遅い馬場
      track_dev_sec      : 日×競馬場×馬場種別の馬場乖離（収縮後）
      tsi_raw            : −track_dev_sec（正=時計が出やすい馬場）
    """
    # ── レース単位テーブル構築 ──────────────────────────────────────────
    races = build_race_table(df)

    # front_split_sec が df に含まれていなければ splits_df から補完
    if "front_split_sec" not in races.columns and splits_df is not None:
        races = races.merge(
            splits_df[["race_id", "front_split_sec"]], on="race_id", how="left",
        )

    if "year" in df.columns:
        year_by_race = df.drop_duplicates("race_id").set_index("race_id")["year"]
        races["year"] = races["race_id"].map(year_by_race)
    else:
        races["year"] = pd.to_datetime(races["date_str"], errors="coerce").dt.year

    train_mask = races["year"].isin(list(train_years))

    # ── 基準2着タイム・レース単位乖離 ──────────────────────────────────
    races = attach_par_2nd_baseline(races, train_mask, min_samples=min_samples)
    races["race_track_dev_sec"] = races["t2nd_adj_sec"] - races["par_2nd_adj_sec"]
    races["race_t2nd_sec"]      = races["t2nd_adj_sec"]

    # ── Layer 1: ペースフィルタ ─────────────────────────────────────────
    if "front_split_sec" in races.columns:
        train_df   = df[df["year"].isin(list(train_years))] if "year" in df.columns else df
        pace_params = build_pace_filter_params(train_df)
        races = attach_pace_filter(races, pace_params)
    else:
        races["front_split_dev"] = np.nan
        races["is_pace_valid"]   = True

    # ── 収縮パラメータの推定（学習期の単純集計から σ を計算） ───────────
    train_races     = races[races["year"].isin(list(train_years))].copy()
    train_day_prelim = _aggregate_day_course_simple(train_races)
    venue_prior, k  = build_shrinkage_params(train_races, train_day_prelim)

    # ── Layer 2+3: ロバスト集計 + 収縮推定 ─────────────────────────────
    day_course = aggregate_day_course_robust(races, venue_prior=venue_prior, k=k)

    # ── 馬単位 DataFrame に付与 ─────────────────────────────────────────
    out = df.copy()
    drop_cols = [c for c in TRACK_OUTPUT_COLS if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out["date_str"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    race_cols_merge = races[[
        "race_id", "class_group", "par_2nd_adj_sec", "par_pool_level",
        "race_t2nd_sec", "race_track_dev_sec", "front_split_dev",
    ]]
    out = out.merge(race_cols_merge, on="race_id", how="left")
    out = out.merge(
        day_course[[
            "date_str", "venue", "surface",
            "track_dev_sec", "tsi_raw", "n_valid_races", "n_races_track",
        ]],
        on=["date_str", "venue", "surface"],
        how="left",
        suffixes=("", "_day"),
    )

    # 障害など芝・ダート以外: 馬場速さ指標を付与しない
    no_track = ~out["surface"].isin(TRACK_SPEED_SURFACES)
    for col in (
        "race_t2nd_sec", "race_track_dev_sec", "par_2nd_adj_sec", "par_pool_level",
        "class_group", "front_split_dev", "track_dev_sec", "tsi_raw",
        "n_valid_races", "n_races_track",
    ):
        if col in out.columns:
            out.loc[no_track, col] = np.nan

    return out, races, day_course
