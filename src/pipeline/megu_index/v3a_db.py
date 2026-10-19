"""めぐ指数 v3a を DB（megu_index テーブル、model_version='v3a'）へ入れるときの変換と upsert。

一括の反映スクリプト（src/scripts/data/reflect_megu_v3a_to_db.py）と、日次の計算（compute.compute_for_date）が
同じ変換を使うように、ここに置く。追加のみ（他の model_version の行は触らない）。
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL_VERSION = "v3a"
USABLE_STATUS = ("ok", "clipped", "restored_out_of_range", "restored_clipped")
NUMERIC_LIMITS = {"megu_index": 99999.9, "par_time_sec": 9999.99, "adjusted_time_sec": 9999.99, "finish_time_sec": 9999.99, "delta_track_sec": 99.999}
ENV_DB_SWITCH = "KEIBA_MEGU_V3A_DB"


def build_index_rows(df: pd.DataFrame) -> pd.DataFrame:
    """v3a の列を持つ DataFrame（年の parquet、または compute_for_dataframe の結果）から DB に入れる行を作る（純粋関数）。絶対指数が付いている走だけ。

    megu_index 列 = 絶対指数(megu_index_v3a_abs_robust)。par_time_sec/adjusted_time_sec/delta_track_sec は相対の計算に使った
    値（v3a_par_sec / v3a_corrected_sec / v3a_track_sec）で、絶対指数とは中心合わせの分だけ恒等式が成り立たない。
    delta_pace/weight/level は v3 の値で v3a の係数と違うため NULL にする。
    """
    ok = df["megu_v3a_status"].isin(USABLE_STATUS) & df["megu_index_v3a_abs_robust"].notna()
    d = df.loc[ok].copy()
    return pd.DataFrame({
        "race_id": d["race_id"].astype(str), "horse_id": d["horse_id"].astype(str),
        "finish_time_sec": d["finish_time_sec"].astype(float).round(2),
        "par_time_sec": d["v3a_par_sec"].astype(float).round(2),
        "delta_pace_sec": np.nan, "delta_track_sec": d["v3a_track_sec"].astype(float).round(3),
        "delta_weight_sec": np.nan, "delta_level_sec": np.nan,
        "adjusted_time_sec": d["v3a_corrected_sec"].astype(float).round(2),
        "megu_index": d["megu_index_v3a_abs_robust"].astype(float).round(1),
        "computation_status": "valid",
    }).reset_index(drop=True)


def check_ranges(rows: pd.DataFrame) -> list[str]:
    bad = []
    for col, lim in NUMERIC_LIMITS.items():
        if col in rows and rows[col].notna().any():
            m = float(rows[col].abs().max())
            if m >= lim:
                bad.append(f"{col}: 最大 {m} が桁あふれ（上限 {lim}）")
    return bad


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def upsert_index(session, rows: pd.DataFrame, batch: int = 1000) -> int:
    sql = text("""
        INSERT INTO megu_index (race_id, horse_id, finish_time_sec, par_time_sec, delta_pace_sec, delta_track_sec, delta_weight_sec, delta_level_sec,
                                adjusted_time_sec, megu_index, computation_status, model_version, computed_at)
        VALUES (:race_id, :horse_id, :finish_time_sec, :par_time_sec, :delta_pace_sec, :delta_track_sec, :delta_weight_sec, :delta_level_sec,
                :adjusted_time_sec, :megu_index, :computation_status, :mv, NOW())
        ON CONFLICT (race_id, horse_id, model_version) DO UPDATE SET
            finish_time_sec=EXCLUDED.finish_time_sec, par_time_sec=EXCLUDED.par_time_sec, delta_track_sec=EXCLUDED.delta_track_sec,
            adjusted_time_sec=EXCLUDED.adjusted_time_sec, megu_index=EXCLUDED.megu_index, computation_status=EXCLUDED.computation_status, computed_at=NOW()
    """)
    n = 0
    recs = [{k: _none(v) for k, v in r.items()} for r in rows.to_dict("records")]
    for i in range(0, len(recs), batch):
        part = recs[i:i + batch]
        session.execute(sql, [{**r, "mv": MODEL_VERSION} for r in part])
        n += len(part)
    return n


def db_switch_enabled(environ: dict | None = None) -> bool:
    return str((environ if environ is not None else os.environ).get(ENV_DB_SWITCH, "")).strip() == "1"


def maybe_upsert_for_result(session, df_result: pd.DataFrame, v3a_enabled: bool, environ: dict | None = None) -> int:
    """日次の計算結果を v3a の行としても DB に入れる（v3a の計算が有効で、環境変数 KEIBA_MEGU_V3A_DB=1 のときだけ）。

    失敗しても日次の計算全体は止めない（parquet と GCS が正本、DB は二次ストア）。入れた行数を返す（無効・対象なしは 0）。
    """
    if not v3a_enabled or not db_switch_enabled(environ) or "megu_v3a_status" not in df_result.columns:
        return 0
    try:
        rows = build_index_rows(df_result)
        if rows.empty:
            return 0
        bad = check_ranges(rows)
        if bad:
            logger.warning("v3a の DB 反映をスキップ（桁あふれ）: %s", bad)
            return 0
        n = upsert_index(session, rows)
        session.commit()
        logger.info("めぐ指数 v3a を DB に保存: %d 行", n)
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning("v3a の DB 反映に失敗（日次の計算は継続）: %s", exc)
        try:
            session.rollback()
        except Exception:  # noqa: BLE001
            pass
        return 0
