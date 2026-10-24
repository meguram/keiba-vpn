"""
NB-03 算出の delta_track を megu_delta_track テーブルへ保存するモジュール。

実行方法:
    # NB-03 の出力 Parquet から保存
    KEIBA_ENV=stg python -m src.pipeline.megu_index.save_delta_track \
        --parquet notebooks/megu_index/output/nb03/delta_track.parquet

    # pandas DataFrame を直接渡す（パイプライン組み込み用）
    from src.pipeline.megu_index.save_delta_track import save_delta_track_to_db
    saved = save_delta_track_to_db(delta_track_df, model_version="stg-v1")

入力 DataFrame の期待カラム:
    date             : str or date  (YYYY-MM-DD)
    venue            : str
    surface          : str          ('芝' or 'ダート')
    delta_track_sec  : float | None (is_fallback=True のとき None または 0.0)
    n_races          : int
    is_fallback      : bool
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL_VERSION = "stg-v1"


def save_delta_track_to_db(
    df: pd.DataFrame,
    model_version: str = MODEL_VERSION,
    batch_size: int = 500,
    session=None,
) -> int:
    """delta_track DataFrame を megu_delta_track テーブルへ UPSERT する。

    Parameters
    ----------
    df:
        NB-03 が出力した delta_track DataFrame。
        必須カラム: date, venue, surface, delta_track_sec, n_races, is_fallback
    model_version:
        保存するモデルバージョン文字列。
    batch_size:
        1 コミットあたりの行数。
    session:
        既存の SQLAlchemy Session を渡す場合に指定。
        None のときは get_session() で新規セッションを取得する。

    Returns
    -------
    int
        保存した行数。
    """
    required = {"date", "venue", "surface", "delta_track_sec", "n_races", "is_fallback"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"delta_track DataFrame に必須カラムが不足: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    def _do_save(sess) -> int:
        total = len(df)
        saved = 0
        for start in range(0, total, batch_size):
            batch = df.iloc[start : start + batch_size]
            for _, row in batch.iterrows():
                dt_val = row["delta_track_sec"]
                dt_float: Optional[float] = (
                    None if (row.get("is_fallback") or pd.isna(dt_val))
                    else float(dt_val)
                )
                sess.execute(
                    text("""
                        INSERT INTO megu_delta_track
                          (date, venue, surface,
                           delta_track_sec, n_races, is_fallback,
                           model_version, computed_at)
                        VALUES
                          (:date, :venue, :surface,
                           :dt, :nr, :fb,
                           :mv, NOW())
                        ON CONFLICT (date, venue, surface, model_version) DO UPDATE
                        SET delta_track_sec = EXCLUDED.delta_track_sec,
                            n_races         = EXCLUDED.n_races,
                            is_fallback     = EXCLUDED.is_fallback,
                            computed_at     = NOW()
                    """),
                    {
                        "date":    row["date"],
                        "venue":   str(row["venue"]),
                        "surface": str(row["surface"]),
                        "dt":      dt_float,
                        "nr":      int(row["n_races"]),
                        "fb":      bool(row["is_fallback"]),
                        "mv":      model_version,
                    },
                )
                saved += 1
            sess.commit()
            logger.info("megu_delta_track upsert %d / %d", min(start + batch_size, total), total)
        return saved

    if session is not None:
        return _do_save(session)

    from src.db.session import get_session  # noqa: PLC0415
    with get_session() as sess:
        return _do_save(sess)


# ── CLI エントリポイント ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.db.session import init_engine

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="NB-03 delta_track を DB に保存")
    parser.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="delta_track.parquet のパス (NB-03 出力)",
    )
    parser.add_argument(
        "--model-version",
        default=MODEL_VERSION,
        help=f"モデルバージョン (デフォルト: {MODEL_VERSION})",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet が見つかりません: {args.parquet}")

    init_engine()
    df_input = pd.read_parquet(args.parquet)
    logger.info("読み込み: %s  shape=%s", args.parquet, df_input.shape)

    n = save_delta_track_to_db(df_input, model_version=args.model_version)
    logger.info("保存完了: %d 行 → megu_delta_track", n)
