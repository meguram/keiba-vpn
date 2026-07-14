"""
めぐ指数 計算・DB 保存スクリプト

実行方法:
    # STG 環境で単一レースを再計算
    KEIBA_ENV=stg python -m src.pipeline.megu_index.compute --race-id 2026010101010

    # STG 環境で指定日程の全レースを計算
    KEIBA_ENV=stg python -m src.pipeline.megu_index.compute --date 2026-06-01

    # STG 環境で年全体を再計算（バッチ）
    KEIBA_ENV=stg python -m src.pipeline.megu_index.compute --year 2026

    # prod 環境（KEIBA_ENV=prod を設定するだけで DB 接続先が変わる）
    KEIBA_ENV=prod python -m src.pipeline.megu_index.compute --date 2026-06-15

設計上の注意:
    - 本スクリプトは DB の megu_regression_params / megu_par_time に保存済みの係数を使う。
    - 係数の再推定（NB-02）は年次または四半期で別途実施（AREA-11 §8 スケジュール参照）。
    - prod 環境への移行: KEIBA_ENV=prod で実行するだけで prod DB に書き込まれる。
      データソース（page_reference）は環境非依存のため別途 rsync/GCS 経由でコピーしておく。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLES_DIR   = PROJECT_ROOT / "data/page_reference/tables"
CUSHION_DIR  = PROJECT_ROOT / "data/page_reference/cushion"
MODEL_VERSION = "v2"

# パターンB ウェイト [U-1]
WEIGHTS_B = [0.35, 0.25, 0.20, 0.12, 0.08]

# ── career_prize モジュールをインポート ─────────────────────────────────────
from src.pipeline.megu_index.class_bucket import par_class_bucket
from src.pipeline.megu_index.common import adjusted_time_to_megu
from src.pipeline.megu_index.field_quality import attach_fq_and_delta_level
from src.pipeline.megu_index.par_time_resolve import attach_par_time_with_fallback


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

from src.pipeline.megu_index.lap_splits import parse_lap_times as _parse_lap_times
from src.pipeline.megu_index.lap_splits import select_split_point as _select_split_point


def _compute_delta_level(
    df: pd.DataFrame,
    df_hist: Optional[pd.DataFrame],
    beta_level: float,
) -> pd.Series:
    """
    各馬の累積獲得賞金推計から delta_level_sec を計算する。

    beta_level = 0 の場合はゼロを返す（未学習フェーズ）。
    df_hist が None の場合もゼロを返す。

    Returns:
        pd.Series (float), df と同じインデックス
    """
    if beta_level == 0.0 or df_hist is None or df_hist.empty:
        return pd.Series(0.0, index=df.index)

    try:
        # ── 1. df_hist から馬 × 日付別の累積賞金テーブルを構築 ─────────────
        df_h = df_hist.copy()
        df_h["date_parsed"] = pd.to_datetime(df_h["date"], errors="coerce")

        # finish_pos 列の統一
        fp_col = "finish_position" if "finish_position" in df_h.columns else "finish_pos"
        df_h["finish_pos_num"] = pd.to_numeric(df_h.get(fp_col, pd.Series(dtype=float)), errors="coerce")

        # grade 正規化（vectorized: apply を一回）
        df_h["grade_norm"] = df_h.apply(
            lambda r: classify_grade(r.get("grade"), r.get("race_name"), r.get("race_class")),
            axis=1,
        )

        # 賞金推計（vectorized）
        prize_table_grades = list({g for g in df_h["grade_norm"].unique()})
        prize_lookup = {
            (g, pos): estimate_prize(g, pos)
            for g in prize_table_grades
            for pos in range(1, 6)
        }
        df_h["prize_est"] = df_h.apply(
            lambda r: prize_lookup.get(
                (r["grade_norm"], int(r["finish_pos_num"]) if pd.notna(r["finish_pos_num"]) else 0),
                0.0,
            ),
            axis=1,
        )

        # horse_id 別に日付順 cumsum
        df_h_sorted = df_h.sort_values(["horse_id", "date_parsed"])
        df_h_sorted["cum_prize"] = df_h_sorted.groupby("horse_id")["prize_est"].cumsum()

        # ── 2. df の各行 (horse_id, race_date) に対して事前累積賞金を付与 ──
        df_work = df[["horse_id", "date", "grade", "race_name", "race_class"]].copy() \
            if "race_name" in df.columns else df[["horse_id", "date", "grade"]].copy()
        df_work["race_dt"] = pd.to_datetime(df_work["date"], errors="coerce")
        df_work["grade_norm"] = df_work.apply(
            lambda r: classify_grade(r.get("grade"), r.get("race_name"), r.get("race_class")),
            axis=1,
        )

        # merge_asof を活用: horse_id でグループ化し各馬の hist を検索
        # 計算量: O(n_hist log n_hist) + O(n_target)
        prize_by_horse: dict[str, np.ndarray] = {}
        date_by_horse: dict[str, np.ndarray] = {}
        for horse_id, grp in df_h_sorted.groupby("horse_id"):
            prize_by_horse[horse_id] = grp["cum_prize"].values
            date_by_horse[horse_id] = grp["date_parsed"].values

        career_prizes = np.zeros(len(df_work), dtype=float)
        for i, (_, row) in enumerate(df_work.iterrows()):
            hid = str(row["horse_id"])
            if hid not in date_by_horse:
                continue
            target_dt = row["race_dt"]
            if pd.isna(target_dt):
                continue
            dates = date_by_horse[hid]
            prizes = prize_by_horse[hid]
            # target_dt より前の最後の cum_prize
            mask = dates < target_dt.to_datetime64()
            if mask.any():
                career_prizes[i] = float(prizes[mask][-1])

        # ── 3. level_feature 計算 ────────────────────────────────────────────
        from src.pipeline.megu_index.career_prize import CAREER_PRIZE_REFERENCE
        grade_norms = df_work["grade_norm"].values
        level_features = np.array([
            compute_level_feature(float(cp), str(gn))
            for cp, gn in zip(career_prizes, grade_norms)
        ])

        logger.info(
            "delta_level: career_prize avg=%.0f 万円, level_feature avg=%.3f",
            career_prizes.mean(),
            level_features.mean(),
        )
        return pd.Series(beta_level * level_features, index=df.index)

    except Exception as e:
        logger.warning("delta_level 計算失敗、ゼロ代入: %s", e)
        return pd.Series(0.0, index=df.index)


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────────────────────────────────

def _load_result_flat(year: int) -> pd.DataFrame:
    p = TABLES_DIR / str(year) / "race_result_flat.parquet"
    if not p.exists():
        logger.warning("race_result_flat not found: %s", p)
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df


def _load_history_flat_for_level(cutoff_date: str) -> pd.DataFrame:
    """
    delta_level_sec 計算用の過去レース履歴を返す。

    TABLES_DIR 配下の全年の race_result_flat.parquet を結合し、
    cutoff_date（YYYYMMDD）より前の分を返す。

    大量のデータになるため、必要最小限のカラムのみロードして省メモリ化する。

    Args:
        cutoff_date: "YYYYMMDD" 形式。この日付以降のデータは除外。

    Returns:
        DataFrame (horse_id, race_id, date, grade, race_name,
                   race_class, finish_position, finish_pos)
    """
    HIST_COLS = [
        "horse_id", "race_id", "date",
        "grade", "race_name", "race_class",
        "finish_position", "finish_pos",
    ]
    cutoff_dt = pd.to_datetime(cutoff_date, format="%Y%m%d", errors="coerce")
    if pd.isna(cutoff_dt):
        return pd.DataFrame()

    dfs = []
    for year_dir in sorted(TABLES_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        p = year_dir / "race_result_flat.parquet"
        if not p.exists():
            continue
        try:
            # 利用可能なカラムのみ読み込み（pyarrow でスキーマチェック）
            import pyarrow.parquet as pq
            available = set(pq.read_schema(p).names)
            use_cols = [c for c in HIST_COLS if c in available]
            df_y = pd.read_parquet(p, columns=use_cols)
            df_y["date_parsed"] = pd.to_datetime(df_y["date"], errors="coerce")
            df_y = df_y[df_y["date_parsed"] < cutoff_dt]
            if not df_y.empty:
                dfs.append(df_y)
        except Exception as e:
            logger.debug("history flat load skip [%d]: %s", year, e)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _load_result_flat_for_date_from_db(date_str: str) -> pd.DataFrame:
    """
    指定日のレースデータを DB の race_results + races テーブルから直接構築する。
    race_result_flat.parquet に対象日が存在しない場合のフォールバック。
    """
    from src.db.session import get_session
    from sqlalchemy import text

    sql = text("""
        SELECT
            rr.race_id,
            rr.horse_id,
            rr.finish_time_sec       AS time_sec,
            rr.finish_pos,
            rr.jockey_id,
            NULL::text               AS jockey_weight,
            NULL::text               AS sex_age,
            NULL::text               AS lap_times,
            NULL::text               AS pace,
            r.surface,
            r.distance,
            r.direction,
            r.track_condition,
            CAST(SUBSTRING(r.race_id, 5, 2) AS TEXT) AS venue_code,
            TO_CHAR(r.race_date, 'YYYY-MM-DD')        AS date
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        WHERE TO_CHAR(r.race_date, 'YYYY-MM-DD') = :date_str
          AND rr.finish_time_sec IS NOT NULL
          AND rr.finish_time_sec > 0
          AND r.surface IN ('芝', 'ダート')
          AND r.distance > 0
    """)
    try:
        with get_session() as session:
            rows = session.execute(sql, {"date_str": date_str}).fetchall()
            if not rows:
                return pd.DataFrame()
            cols = ["race_id", "horse_id", "time_sec", "finish_pos",
                    "jockey_id", "jockey_weight", "sex_age", "lap_times", "pace",
                    "surface", "distance", "direction", "track_condition",
                    "venue_code", "date"]
            return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        logger.error("DB からの result_flat 読み込みエラー: %s", e)
        return pd.DataFrame()


def _load_cushion(year: int) -> pd.DataFrame:
    p = CUSHION_DIR / f"cushion_{year}.json"
    if not p.exists():
        return pd.DataFrame()
    data = json.loads(p.read_text())
    df = pd.DataFrame(data)
    df = df[df["is_race_day"] == True].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["venue_code"] = df["venue_code"].astype(str).str.strip()
    return df


def _load_regression_params(session, model_version: str | None = None) -> dict[str, float]:
    mv = model_version or MODEL_VERSION
    rows = session.execute(
        text("SELECT param_name, param_value FROM megu_regression_params WHERE model_version=:mv"),
        {"mv": mv},
    ).fetchall()
    params = {r[0]: float(r[1]) for r in rows}
    if params:
        return params
    if mv != "v1":
        rows = session.execute(
            text("SELECT param_name, param_value FROM megu_regression_params WHERE model_version='v1'"),
        ).fetchall()
        return {r[0]: float(r[1]) for r in rows}
    return {}


def _load_par_time(session, model_version: str | None = None) -> pd.DataFrame:
    mv = model_version or MODEL_VERSION
    rows = session.execute(
        text("""
            SELECT distance, course, surface, track_condition, class_bucket,
                   par_time_sec, par_front_split_sec
            FROM megu_par_time WHERE model_version=:mv
        """),
        {"mv": mv},
    ).fetchall()
    df = pd.DataFrame(rows, columns=[
        "distance", "course", "surface", "track_condition", "class_bucket",
        "par_time_sec", "par_front_split_sec",
    ])
    df["par_time_sec"] = pd.to_numeric(df["par_time_sec"], errors="coerce")
    df["par_front_split_sec"] = pd.to_numeric(df["par_front_split_sec"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# めぐ指数計算
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_flat_from_gcs_race_result(df: pd.DataFrame) -> pd.DataFrame:
    """flat の distance=0 / surface 欠損を GCS race_result で補完。"""
    if df.empty or "race_id" not in df.columns:
        return df

    from src.utils.race_card_merge import load_merged_race_card

    dist = pd.to_numeric(df.get("distance"), errors="coerce").fillna(0)
    surf = df.get("surface")
    bad_surf = surf.isna() if surf is not None else pd.Series(True, index=df.index)
    if surf is not None:
        bad_surf = bad_surf | ~surf.astype(str).isin(["芝", "ダート"])
    need_ids = df.loc[(dist <= 0) | bad_surf, "race_id"].astype(str).unique()
    if len(need_ids) == 0:
        return df

    out = df.copy()
    for race_id in need_ids:
        card = load_merged_race_card(race_id)
        if not card:
            continue
        mask = out["race_id"].astype(str) == race_id
        card_dist = int(card.get("distance") or 0)
        card_surf = card.get("surface")
        if card_dist > 0:
            out.loc[mask, "distance"] = card_dist
        if card_surf in ("芝", "ダート"):
            out.loc[mask, "surface"] = card_surf
        for col in ("track_condition", "venue", "race_name", "grade", "race_class", "direction"):
            if col in out.columns and card.get(col):
                empty = out.loc[mask, col].isna() | (out.loc[mask, col].astype(str).str.strip() == "")
                if empty.any():
                    out.loc[mask & empty, col] = card[col]
    return out


def compute_for_dataframe(
    df_flat: pd.DataFrame,
    params: dict[str, float],
    df_par: pd.DataFrame,
    df_hist: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    race_result_flat DataFrame → めぐ指数 DataFrame を返す。

    Parameters
    ----------
    df_flat : race_result_flat（cushion 列マージ済みのもの）
    params  : megu_regression_params の dict
    df_par  : megu_par_time DataFrame
    df_hist : 過去レース結果 (race_result_flat 形式)。
              省略時は delta_level_sec = 0.0 のまま。
              horse_id ごとの累積獲得賞金を推計して delta_level_sec に使用。

    Returns
    -------
    DataFrame with columns:
        race_id, horse_id, finish_time_sec, par_time_sec,
        delta_pace_sec, delta_track_sec, delta_weight_sec, delta_level_sec,
        adjusted_time_sec, megu_index,
        front_split_sec, split_point_m, tsi_raw
    """
    beta_pace   = params.get("beta_pace",   0.0)
    beta_track  = params.get("beta_track",  0.0)
    beta_weight = params.get("beta_weight", 0.0)
    tsi_mean    = params.get("tsi_mean",    0.0)

    df = df_flat.copy()
    df = _enrich_flat_from_gcs_race_result(df)
    df = df[df["surface"].isin(["芝", "ダート"])]
    df["finish_time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df[df["finish_time_sec"].notna() & (df["finish_time_sec"] > 0)]
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df[df["distance"] > 0]

    # ── 前半スプリット ──────────────────────────────────────────────────
    splits = []
    for _, row in df[["race_id", "distance", "lap_times"]].drop_duplicates("race_id").iterrows():
        lap_dict = _parse_lap_times(row["lap_times"], int(row["distance"]))
        if not lap_dict:
            splits.append({"race_id": row["race_id"], "front_split_sec": np.nan, "split_point_m": np.nan})
            continue
        sp = _select_split_point(int(row["distance"]), list(lap_dict.keys()))
        if sp is None:
            splits.append({"race_id": row["race_id"], "front_split_sec": np.nan, "split_point_m": np.nan})
        else:
            splits.append({"race_id": row["race_id"], "front_split_sec": lap_dict[sp], "split_point_m": sp})
    df_splits = pd.DataFrame(splits)
    if df_splits.empty or "race_id" not in df_splits.columns:
        df["front_split_sec"] = np.nan
        df["split_point_m"] = np.nan
    else:
        df = df.merge(df_splits, on="race_id", how="left")

    # ── TSI ────────────────────────────────────────────────────────────
    if "tsi_raw" not in df.columns:
        df["tsi_raw"] = tsi_mean
    df["tsi_normalized"] = df["tsi_raw"].fillna(tsi_mean) - tsi_mean

    # ── 斤量偏差 ───────────────────────────────────────────────────────
    df["sex"] = df["sex_age"].str.extract(r"^(牡|牝|セン)", expand=False).fillna("牡")
    df["base_weight"] = np.where(df["sex"] == "牝", 53.0, 55.0)
    df["jockey_weight_num"] = pd.to_numeric(df["jockey_weight"], errors="coerce")
    df["weight_dev"] = df["jockey_weight_num"] - df["base_weight"]
    df["dist_scale"] = df["distance"] / 2000.0

    # ── track_cat ──────────────────────────────────────────────────────
    track_map = {"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}
    df["track_cat"] = df["track_condition"].map(track_map).fillna("良")
    if "direction" not in df.columns:
        df["direction"] = df.get("course", "")

    df["class_bucket"] = df.apply(
        lambda r: par_class_bucket(r.get("grade"), r.get("race_name"), r.get("race_class")),
        axis=1,
    )

    # ── par_time マージ（v2: クラス別 + 階層フォールバック）────────────────
    df = attach_par_time_with_fallback(df, df_par)

    # ── front_split_dev ────────────────────────────────────────────────
    df["front_split_dev"] = df["front_split_sec"] - df["par_front_split_sec"]

    # ── 各補正値 ───────────────────────────────────────────────────────
    df["delta_pace_sec"]   = beta_pace  * df["front_split_dev"].fillna(0)
    # 芝・ダートのみ馬場補正（障害は surface フィルタで対象外）
    df["delta_track_sec"]  = -beta_track * df["tsi_normalized"].fillna(0)
    df["delta_weight_sec"] = beta_weight * df["weight_dev"].fillna(0) * df["dist_scale"].fillna(1)

    # ── レベル補正 (FQ ベース, AREA-11 §5) ───────────────────────────────
    beta_level = params.get("beta_level", 0.0)
    par_log_fq = params.get("par_log_fq")
    df = attach_fq_and_delta_level(df, df_hist, beta_level, par_log_fq=par_log_fq)

    # ── 補正済みタイム & めぐ指数 ─────────────────────────────────────
    df["adjusted_time_sec"] = (
        df["finish_time_sec"]
        - df["delta_pace_sec"]
        - df["delta_track_sec"]
        - df["delta_weight_sec"]
        - df["delta_level_sec"]
    )
    df["megu_index"] = df.apply(
        lambda r: adjusted_time_to_megu(r["adjusted_time_sec"], r["par_time_final"]),
        axis=1,
    )

    # ── 【最適化】2着基準 out_of_range フラグ ──────────────────────────────
    finish_pos_col = "finish_position" if "finish_position" in df.columns else "finish_pos"
    if finish_pos_col in df.columns:
        df["finish_pos_num"] = pd.to_numeric(df[finish_pos_col], errors="coerce")
        df_2nd = (
            df[df["finish_pos_num"] == 2][["race_id", "finish_time_sec"]]
            .rename(columns={"finish_time_sec": "time_2nd"})
        )
        df_1st = (
            df[df["finish_pos_num"] == 1][["race_id", "finish_time_sec"]]
            .rename(columns={"finish_time_sec": "time_1st"})
        )
        df_2nd = df_2nd.merge(df_1st, on="race_id", how="outer")
        df_2nd["time_2nd"] = df_2nd["time_2nd"].fillna(df_2nd["time_1st"])
        df = df.merge(df_2nd[["race_id", "time_2nd"]], on="race_id", how="left")
        df["out_of_range"] = (
            (df["finish_pos_num"] > 2) &
            (df["finish_time_sec"] > df["time_2nd"].fillna(np.inf) + 2.0)
        )
        df.loc[df["out_of_range"], "megu_index"] = np.nan
        df["computation_status"] = np.where(df["out_of_range"], "out_of_range", "valid")
    else:
        df["computation_status"] = "valid"

    result_cols = [
        "race_id", "horse_id", "finish_time_sec", "par_time_final",
        "delta_pace_sec", "delta_track_sec", "delta_weight_sec", "delta_level_sec",
        "adjusted_time_sec", "megu_index", "field_quality", "computation_status",
        "front_split_sec", "split_point_m", "tsi_raw",
    ]
    return df[[c for c in result_cols if c in df.columns]].copy()


def _upsert_batch(session, df_result: pd.DataFrame, batch_size: int = 500, model_version: str = MODEL_VERSION) -> int:
    """めぐ指数 DataFrame を megu_index テーブルに UPSERT する。"""
    total = len(df_result)
    saved = 0
    for start in range(0, total, batch_size):
        batch = df_result.iloc[start : start + batch_size]
        for _, row in batch.iterrows():
            session.execute(
                text("""
                    INSERT INTO megu_index
                      (race_id, horse_id, finish_time_sec, par_time_sec,
                       delta_pace_sec, delta_track_sec, delta_weight_sec, delta_level_sec,
                       adjusted_time_sec, megu_index, field_quality,
                       front_split_sec, split_point_m, tsi_raw,
                       computation_status, model_version, computed_at)
                    VALUES
                      (:rid, :hid, :ft, :pt, :dp, :dt, :dw, :dl,
                       :at, :mi, :fq, :fsp, :spm, :tsi, :cs, :mv, NOW())
                    ON CONFLICT (race_id, horse_id, model_version) DO UPDATE
                    SET finish_time_sec  = EXCLUDED.finish_time_sec,
                        par_time_sec     = EXCLUDED.par_time_sec,
                        delta_pace_sec   = EXCLUDED.delta_pace_sec,
                        delta_track_sec  = EXCLUDED.delta_track_sec,
                        delta_weight_sec = EXCLUDED.delta_weight_sec,
                        delta_level_sec  = EXCLUDED.delta_level_sec,
                        adjusted_time_sec = EXCLUDED.adjusted_time_sec,
                        megu_index       = EXCLUDED.megu_index,
                        field_quality    = EXCLUDED.field_quality,
                        front_split_sec  = EXCLUDED.front_split_sec,
                        split_point_m    = EXCLUDED.split_point_m,
                        tsi_raw          = EXCLUDED.tsi_raw,
                        computation_status = EXCLUDED.computation_status,
                        computed_at      = NOW()
                """),
                {
                    "rid":  str(row["race_id"]),
                    "hid":  str(row["horse_id"]),
                    "ft":   _to_float(row.get("finish_time_sec")),
                    "pt":   _to_float(row.get("par_time_final")),
                    "dp":   _to_float(row.get("delta_pace_sec", 0)),
                    "dt":   _to_float(row.get("delta_track_sec", 0)),
                    "dw":   _to_float(row.get("delta_weight_sec", 0)),
                    "dl":   _to_float(row.get("delta_level_sec", 0)),
                    "at":   _to_float(row.get("adjusted_time_sec")),
                    "mi":   _to_float(row.get("megu_index")),
                    "fq":   _to_float(row.get("field_quality")),
                    "fsp":  _to_float(row.get("front_split_sec")),
                    "spm":  int(row["split_point_m"]) if pd.notna(row.get("split_point_m")) else None,
                    "tsi":  _to_float(row.get("tsi_raw")),
                    "cs":   str(row.get("computation_status", "valid")),
                    "mv":   model_version,
                },
            )
            saved += 1
        session.commit()
        logger.info("upserted %d / %d", min(start + batch_size, total), total)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# GCS バックアップ保存（ローカル parquet + GCS upload）
# ─────────────────────────────────────────────────────────────────────────────

_MEGU_PARQUET_COLS = [
    "race_id", "horse_id", "megu_index", "computation_status",
    "finish_time_sec", "par_time_final", "adjusted_time_sec",
    "delta_pace_sec", "delta_track_sec", "delta_weight_sec", "delta_level_sec",
]
_GCS_MEGU_BLOB_TPL = "chuou/data/preprocessed/netkeiba/pc/megu_index/{year}/megu_index_flat.parquet"


def _save_megu_parquet_backup(df_save: pd.DataFrame, year: int) -> None:
    """
    計算済み megu_index DataFrame をローカル parquet (TABLES_DIR/{year}/megu_index_flat.parquet)
    に追記・上書き保存し、GCS にバックアップアップロードする。

    - DB が一次ストア。parquet / GCS はバックアップ（stg→prod 再現用）。
    - 同一 (race_id, horse_id) の重複は「新しい行を優先」で解消する。
    - GCS 接続が無い環境（dev / CI）ではローカル保存のみで正常終了する。
    """
    save_cols = [c for c in _MEGU_PARQUET_COLS if c in df_save.columns]
    df_new = df_save[save_cols].copy()

    out_dir = TABLES_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "megu_index_flat.parquet"

    # 既存 parquet とマージ（重複は新データを優先）
    if out_path.exists():
        try:
            df_existing = pd.read_parquet(out_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(
                subset=["race_id", "horse_id"], keep="last"
            )
        except Exception as e:
            logger.warning("既存 parquet 読み込み失敗、上書き保存: %s", e)
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_parquet(out_path, index=False)
    logger.info("megu_index parquet 保存: %s (%d 行)", out_path, len(df_combined))

    # GCS アップロード（失敗してもDBは保存済みなので警告のみ）
    try:
        from src.scraper.storage import HybridStorage
        storage = HybridStorage(str(PROJECT_ROOT))
        if storage.gcs_enabled:
            gcs_blob = _GCS_MEGU_BLOB_TPL.format(year=year)
            bucket = storage._get_bucket()
            blob = bucket.blob(gcs_blob)
            blob.upload_from_filename(str(out_path), content_type="application/octet-stream")
            logger.info(
                "megu_index GCS バックアップ完了: gs://%s/%s",
                storage._bucket_name, gcs_blob,
            )
        else:
            logger.debug("GCS 無効（ローカル parquet のみ保存）")
    except Exception as exc:
        logger.warning("GCS バックアップ失敗（DBは保存済み）: %s", exc)



def aggregate_horse_megu(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    horse_id ごとに直近5走のめぐ指数を集計して A/B/C 3パターンを返す。

    Parameters
    ----------
    df_all : megu_index テーブルの全行（race_id, horse_id, megu_index, 
              surface, distance, date 列を含む）

    Returns
    -------
    DataFrame: horse_id, megu_a, megu_b, megu_c
    """
    df = df_all.copy()
    df["race_date"] = pd.to_datetime(df.get("date", df.get("computed_at")), errors="coerce")
    df = df.sort_values(["horse_id", "race_date"])

    dist_bands = {"sprint": (0, 1499), "mile": (1500, 1799), "middle": (1800, 2399), "long": (2400, 99999)}

    def band(d):
        for name, (lo, hi) in dist_bands.items():
            if lo <= d <= hi:
                return name
        return "middle"

    results = []
    for horse_id, grp in df.groupby("horse_id"):
        grp = grp.sort_values("race_date", ascending=False)
        recent = grp.head(5)
        if len(recent) == 0:
            continue

        # A: 最大
        megu_a = float(recent["megu_index"].max())

        # B: 加重平均 [U-1]
        w = np.array(WEIGHTS_B[: len(recent)])
        w = w / w.sum()
        megu_b = float((recent["megu_index"].values * w).sum())

        # C: 同距離帯 × 同馬場 直近3走最大
        if "surface" in grp.columns and "distance" in grp.columns:
            surf0 = grp.iloc[0]["surface"]
            db0   = band(grp.iloc[0]["distance"])
            cond  = grp[(grp["surface"] == surf0) & (grp["distance"].apply(band) == db0)].head(3)
            megu_c = float(cond["megu_index"].max()) if len(cond) > 0 else np.nan
        else:
            megu_c = np.nan

        results.append({"horse_id": horse_id, "megu_a": megu_a, "megu_b": megu_b, "megu_c": megu_c})

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 公開 API（外部モジュールから直接呼び出し用）
# ─────────────────────────────────────────────────────────────────────────────

def compute_for_date(date_str: str, model_version: str = MODEL_VERSION) -> dict:
    """
    指定日（YYYY-MM-DD）の全レースについてめぐ指数を計算しDB保存する。

    auto_scrape.py の raceday-evening タスク等から直接呼び出せる。

    Parameters
    ----------
    date_str : "YYYY-MM-DD" 形式の日付文字列
    model_version : DBに保存する model_version キー

    Returns
    -------
    dict with keys:
        status       : "ok" | "skipped" | "error"
        megu_valid   : 保存した valid レコード数
        megu_oor     : 保存した out_of_range レコード数
        date         : date_str
        error        : エラーメッセージ（エラー時のみ）
    """
    logger.info("めぐ指数計算 開始: %s", date_str)
    try:
        from src.db.session import get_session, init_engine
        init_engine()

        year = int(date_str[:4])
        with get_session() as session:
            params = _load_regression_params(session, model_version)
            df_par = _load_par_time(session, model_version)

        if not params:
            logger.warning("回帰パラメータ未登録 – めぐ指数をスキップ")
            return {"status": "skipped", "reason": "no_params", "date": date_str,
                    "megu_valid": 0, "megu_oor": 0}

        df_flat = _load_result_flat(year)
        if df_flat.empty:
            logger.warning("race_result_flat が空: year=%d", year)
            return {"status": "skipped", "reason": "no_flat_data", "date": date_str,
                    "megu_valid": 0, "megu_oor": 0}

        # 日付フィルタ
        df_flat["date_str"] = pd.to_datetime(df_flat["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df_flat = df_flat[df_flat["date_str"] == date_str]
        if df_flat.empty:
            logger.info("parquet に対象日なし → DB から直接ロード: %s", date_str)
            df_flat = _load_result_flat_for_date_from_db(date_str)
            if df_flat.empty:
                logger.info("対象日のデータなし: %s", date_str)
                return {"status": "skipped", "reason": "no_data_on_date", "date": date_str,
                        "megu_valid": 0, "megu_oor": 0}
            df_flat["date_str"] = date_str
        else:
            df_flat["date_str"] = date_str

        # cushion マージ
        df_cushion = _load_cushion(year)
        if not df_cushion.empty:
            df_flat["venue_code"] = df_flat["venue_code"].astype(str).str.strip()
            c_merge = df_cushion[["date_str", "venue_code", "cushion_value", "dirt_moisture_goal"]] \
                .drop_duplicates(subset=["date_str", "venue_code"])
            df_flat = df_flat.merge(c_merge, on=["date_str", "venue_code"], how="left")
            tsi_mean = params.get("tsi_mean", 0.0)
            df_flat["tsi_raw"] = np.where(
                df_flat["surface"] == "芝",
                df_flat["cushion_value"].fillna(tsi_mean),
                -df_flat["dirt_moisture_goal"].fillna(-tsi_mean),
            )

        # 過去レース履歴（delta_level 用）: 当該日より前の全年データをロード
        df_hist = _load_history_flat_for_level(date_str)
        df_result = compute_for_dataframe(df_flat, params, df_par, df_hist=df_hist)
        # valid + out_of_range 両方を保存（par_time があるものに限る）
        df_save = df_result[df_result["par_time_final"].notna()].copy()

        if df_save.empty:
            return {"status": "skipped", "reason": "no_results_after_compute", "date": date_str,
                    "megu_valid": 0, "megu_oor": 0}

        with get_session() as session:
            saved = _upsert_batch(session, df_save, model_version=model_version)

        # GCS バックアップ保存（DB 保存後に実行、失敗しても DB は確定済み）
        _save_megu_parquet_backup(df_save, year)

        n_valid = int((df_save["computation_status"] == "valid").sum())
        n_oor   = int((df_save["computation_status"] == "out_of_range").sum())
        logger.info("めぐ指数 保存完了: valid=%d, out_of_range=%d", n_valid, n_oor)
        return {"status": "ok", "date": date_str, "megu_valid": n_valid, "megu_oor": n_oor}

    except Exception as e:
        logger.error("めぐ指数計算エラー: %s", e, exc_info=True)
        return {"status": "error", "date": date_str, "error": str(e),
                "megu_valid": 0, "megu_oor": 0}


# ─────────────────────────────────────────────────────────────────────────────
# CLI エントリポイント
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="めぐ指数計算スクリプト")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--race-id",  help="単一 race_id を計算（例: 2026010101010）")
    group.add_argument("--date",     help="指定日（YYYY-MM-DD）の全レースを計算")
    group.add_argument("--year",     type=int, help="指定年全体を計算（例: 2026）")
    p.add_argument("--dry-run", action="store_true", help="DB に書き込まず計算結果のみ表示")
    p.add_argument("--model-version", default=MODEL_VERSION)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()

    # DB 初期化
    from src.db.session import get_session, init_engine
    init_engine()

    with get_session() as session:
        params = _load_regression_params(session, args.model_version)
        df_par = _load_par_time(session, args.model_version)

    if not params:
        logger.error("回帰パラメータが DB に存在しません。NB-02 を先に実行してください。")
        raise SystemExit(1)

    logger.info("パラメータ: %s", params)

    # 対象年の特定
    if args.race_id:
        year = int(args.race_id[:4])
        years = [year]
    elif args.date:
        year = int(args.date[:4])
        years = [year]
    else:
        years = [args.year]

    # データロード
    dfs = []
    for yr in years:
        df_flat = _load_result_flat(yr)

        # 対象を絞り込む
        if args.race_id:
            if not df_flat.empty:
                df_flat = df_flat[df_flat["race_id"] == args.race_id]
            if df_flat.empty:
                logger.info("parquet に %s なし → DB から直接ロード", args.race_id)
                df_flat = _load_result_flat_for_date_from_db(
                    args.race_id[0:4] + "-" + args.race_id[4:6] + "-" + args.race_id[6:8]
                )
                if not df_flat.empty:
                    df_flat = df_flat[df_flat["race_id"] == args.race_id]
        elif args.date:
            if not df_flat.empty:
                df_flat["date_str"] = pd.to_datetime(df_flat["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                df_flat = df_flat[df_flat["date_str"] == args.date]
            if df_flat.empty:
                logger.info("parquet に %s なし → DB から直接ロード", args.date)
                df_flat = _load_result_flat_for_date_from_db(args.date)

        if df_flat.empty:
            continue

        # cushion マージ
        df_cushion = _load_cushion(yr)
        if not df_cushion.empty:
            df_flat["date_str"] = pd.to_datetime(df_flat["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df_flat["venue_code"] = df_flat["venue_code"].astype(str).str.strip()
            c_merge = df_cushion[["date_str", "venue_code", "cushion_value", "dirt_moisture_goal"]].drop_duplicates(
                subset=["date_str", "venue_code"]
            )
            df_flat = df_flat.merge(c_merge, on=["date_str", "venue_code"], how="left")
            tsi_mean = params.get("tsi_mean", 0.0)
            df_flat["tsi_raw"] = np.where(
                df_flat["surface"] == "芝",
                df_flat["cushion_value"].fillna(tsi_mean),
                -df_flat["dirt_moisture_goal"].fillna(-tsi_mean),
            )

        dfs.append(df_flat)

    if not dfs:
        logger.warning("対象データなし")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info("入力データ: %d 行", len(df_all))

    # 過去レース履歴（delta_level 用）
    target_year = args.year
    hist_cutoff = f"{target_year}0101"
    df_hist = _load_history_flat_for_level(hist_cutoff)

    # めぐ指数計算
    df_result = compute_for_dataframe(df_all, params, df_par, df_hist=df_hist)
    # valid + out_of_range 両方保存（par_time があるもの）
    df_save = df_result[df_result["par_time_final"].notna()].copy()
    n_valid = int((df_save["computation_status"] == "valid").sum())
    n_oor   = int((df_save["computation_status"] == "out_of_range").sum())
    logger.info("計算済み: valid=%d, out_of_range=%d", n_valid, n_oor)

    if args.dry_run:
        print(df_save.head(20).to_string())
        return

    # DB 保存
    with get_session() as session:
        saved = _upsert_batch(session, df_save, model_version=args.model_version)
    logger.info("DB 保存完了: %d 件", saved)

    # GCS バックアップ保存（年単位バッチでも同様に実行）
    for yr in set(years):
        df_yr = df_save[pd.to_datetime(df_all["date"], errors="coerce").dt.year == yr] if len(years) > 1 else df_save
        _save_megu_parquet_backup(df_yr, yr)


if __name__ == "__main__":
    main()
