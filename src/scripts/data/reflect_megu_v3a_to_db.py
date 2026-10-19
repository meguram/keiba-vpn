"""めぐ指数 v3a を DB（megu_regression_params / megu_par_time / megu_index）へ model_version='v3a' の行として追加する。

追加のみ（v2 など既存の model_version の行は変更・削除しない）。テーブル構造の変更（DDL）はしない。
  - megu_regression_params: v3a の係数（spec.json・centers.json と、v2 から引き継ぐ beta_pace/beta_track/tsi_mean/beta_level）
  - megu_par_time: --with-par を付けたときだけ。par.csv を参考として載せる（計算の正本は config/megu_v3a。DB の列に
    収まらない月別オフセットは載せない。track_condition 列に基準の階層 L1〜L4、class_bucket 列に『世代|クラス』を入れる）
  - megu_index: 絶対指数 megu_index_v3a_abs_robust を megu_index 列に入れる（相対・予測用・信頼度フラグは parquet/GCS が正本）

使い方（ホスト。DB は stg の Cloud SQL）::

  sudo -n env -u DATABASE_URL KEIBA_ENV=stg PYTHONPATH=. nice -n 19 python3 -m src.scripts.data.reflect_megu_v3a_to_db            # dry-run（既定。書き込まない）
  ... -m src.scripts.data.reflect_megu_v3a_to_db --execute [--with-par] [--years 2020-2026]                                    # 反映
  ... -m src.scripts.data.reflect_megu_v3a_to_db --rollback                                                                    # 件数だけ表示（dry-run）
  ... -m src.scripts.data.reflect_megu_v3a_to_db --rollback --execute                                                          # model_version='v3a' の行だけ削除
  ... -m src.scripts.data.reflect_megu_v3a_to_db --backup-dir data/local/backup/megu_db_before_v3a_20261019 --backup-only      # 3 テーブルの現行内容を parquet に退避
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

MODEL_VERSION = "v3a"
SOURCE_VERSION = "v2"
MODEL_DIR = Path("config/megu_v3a")
TABLES_DIR = Path("data/page_reference/tables")
USABLE_STATUS = ("ok", "clipped", "restored_out_of_range", "restored_clipped")
COPIED_PARAMS = ("beta_pace", "beta_track", "tsi_mean", "beta_level")  # v2 から引き継ぐ（馬場のフォールバック・互換のため）
NUMERIC_LIMITS = {"megu_index": 99999.9, "par_time_sec": 9999.99, "adjusted_time_sec": 9999.99, "finish_time_sec": 9999.99, "delta_track_sec": 99.999}


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


# ── 係数 ────────────────────────────────────────────────────────────
def build_param_rows(spec: dict, centers: dict, v2_params: dict[str, tuple[float, float | None]]) -> list[dict]:
    """megu_regression_params に入れる行（param_name, param_value, std_error）を返す（純粋関数）。"""
    rows: list[dict] = [{"param_name": "beta_weight", "param_value": float(spec["beta_weight"]), "std_error": None}]
    for name in COPIED_PARAMS:
        if name in v2_params:
            val, se = v2_params[name]
            rows.append({"param_name": name, "param_value": float(val), "std_error": None if se is None else float(se)})
    scalar = {
        "v3a_base_kg": spec["base_kg"], "v3a_pace_cap": spec["pace_cap"], "v3a_pace_turf_only": 1.0 if spec.get("pace_turf_only") else 0.0,
        "v3a_loro_min_other_races": spec["loro_min_other_races"], "v3a_loro_min_horses": spec["loro_min_horses"],
        "v3a_min_par_n": spec["min_par_n"], "v3a_shrink_k": spec["shrink_k"], "v3a_month_smooth": spec["month_smooth"],
        "v3a_month_shrink_k": spec["month_shrink_k"], "v3a_n_iter": spec["n_iter"],
        "v3a_center_rel_turf": centers["rel"]["芝"], "v3a_center_rel_dirt": centers["rel"]["ダート"],
        "v3a_center_abs_turf": centers["abs"]["芝"], "v3a_center_abs_dirt": centers["abs"]["ダート"],
    }
    rows += [{"param_name": k, "param_value": float(v), "std_error": None} for k, v in scalar.items()]
    return rows


# ── 基準タイム（参考）────────────────────────────────────────────────
def build_par_rows(par: pd.DataFrame) -> list[dict]:
    """par.csv → megu_par_time の行。track_condition に階層(L1〜L4)、class_bucket に『世代|クラス』（純粋関数）。"""
    d = par.copy()
    d["direction"] = d["direction"].fillna("")
    d["gen"] = d["gen"].fillna("")
    d["cls"] = d["cls"].fillna("")
    d["lv"] = d["level"].str.extract(r"^(L\d)", expand=False)
    d["bucket"] = np.where((d["gen"] != "") | (d["cls"] != ""), d["gen"] + "|" + d["cls"], "")
    out = []
    for r in d.itertuples(index=False):
        out.append({
            "distance": int(r.distance), "course": str(r.direction), "surface": str(r.surface), "track_condition": str(r.lv),
            "class_bucket": str(r.bucket), "par_time_sec": round(float(r.par), 2), "par_front_split_sec": None, "sample_count": int(r.n),
        })
    return out


# ── megu_index ──────────────────────────────────────────────────────
def build_index_rows(df: pd.DataFrame) -> pd.DataFrame:
    """年の parquet から DB に入れる行を作る（純粋関数）。絶対指数が付いている走だけ。

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


# ── DB 操作 ─────────────────────────────────────────────────────────
def load_v2_params(session) -> dict[str, tuple[float, float | None]]:
    rows = session.execute(text("SELECT param_name, param_value, std_error FROM megu_regression_params WHERE model_version=:mv"), {"mv": SOURCE_VERSION}).fetchall()
    return {r[0]: (float(r[1]), None if r[2] is None else float(r[2])) for r in rows}


def count_v3a(session) -> dict[str, int]:
    return {t: int(session.execute(text(f"SELECT COUNT(*) FROM {t} WHERE model_version=:mv"), {"mv": MODEL_VERSION}).scalar()) for t in ("megu_regression_params", "megu_par_time", "megu_index")}


def upsert_params(session, rows: list[dict]) -> None:
    session.execute(text("""
        INSERT INTO megu_regression_params (param_name, param_value, std_error, sample_count, model_version, fitted_at)
        VALUES (:param_name, :param_value, :std_error, NULL, :mv, NOW())
        ON CONFLICT (param_name, model_version) DO UPDATE SET param_value=EXCLUDED.param_value, std_error=EXCLUDED.std_error, fitted_at=NOW()
    """), [{**r, "mv": MODEL_VERSION} for r in rows])


def upsert_par(session, rows: list[dict], batch: int = 500) -> None:
    sql = text("""
        INSERT INTO megu_par_time (distance, course, surface, track_condition, class_bucket, par_time_sec, par_front_split_sec, sample_count, model_version, computed_at)
        VALUES (:distance, :course, :surface, :track_condition, :class_bucket, :par_time_sec, :par_front_split_sec, :sample_count, :mv, NOW())
        ON CONFLICT (distance, course, surface, track_condition, class_bucket, model_version)
        DO UPDATE SET par_time_sec=EXCLUDED.par_time_sec, sample_count=EXCLUDED.sample_count, computed_at=NOW()
    """)
    for i in range(0, len(rows), batch):
        session.execute(sql, [{**r, "mv": MODEL_VERSION} for r in rows[i:i + batch]])


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


def backup_tables(session, out_dir: Path) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}
    for t in ("megu_regression_params", "megu_par_time", "megu_index"):
        df = pd.read_sql(text(f"SELECT * FROM {t}"), session.connection())
        df.to_parquet(out_dir / f"{t}.parquet", index=False)
        summary[t] = {str(k): int(v) for k, v in df.groupby("model_version").size().items()}
    (out_dir / "row_counts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def rollback(session, execute: bool) -> dict[str, int]:
    counts = count_v3a(session)
    if execute:
        for t in ("megu_index", "megu_par_time", "megu_regression_params"):
            session.execute(text(f"DELETE FROM {t} WHERE model_version = 'v3a'"))
        session.commit()
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="指定しなければ dry-run（書き込まない）")
    ap.add_argument("--with-par", action="store_true", help="par.csv を megu_par_time に参考として載せる")
    ap.add_argument("--years", default="2020-2026")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--backup-dir", default="")
    ap.add_argument("--backup-only", action="store_true")
    args = ap.parse_args()

    from src.db.session import get_session, init_engine

    init_engine()
    with get_session() as session:
        before = count_v3a(session)
        logger.info("現在の v3a の行数: %s", before)
        if args.backup_dir:
            logger.info("退避: %s", backup_tables(session, Path(args.backup_dir)))
            if args.backup_only:
                return
        if args.rollback:
            logger.info("%s 対象 %s", "削除" if args.execute else "削除予定（dry-run）", rollback(session, args.execute))
            return

        spec = json.loads((MODEL_DIR / "spec.json").read_text())
        centers = json.loads((MODEL_DIR / "centers.json").read_text())
        param_rows = build_param_rows(spec, centers, load_v2_params(session))
        par_rows = build_par_rows(pd.read_csv(MODEL_DIR / "par.csv")) if args.with_par else []
        plan: dict[str, object] = {"megu_regression_params": len(param_rows), "megu_par_time": len(par_rows), "megu_index": {}}
        index_by_year: dict[int, pd.DataFrame] = {}
        for y in parse_years(args.years):
            f = TABLES_DIR / str(y) / "megu_index_flat.parquet"
            rows = build_index_rows(pd.read_parquet(f))
            bad = check_ranges(rows)
            if bad:
                raise SystemExit(f"{y}: 桁あふれ {bad}")
            dup = int(rows.duplicated(["race_id", "horse_id"]).sum())
            if dup:
                raise SystemExit(f"{y}: (race_id, horse_id) の重複 {dup}")
            index_by_year[y] = rows
            plan["megu_index"][y] = len(rows)  # type: ignore[index]
        logger.info("計画: %s（既存 v3a 行は上書き=冪等）", plan)
        if not args.execute:
            logger.info("dry-run: 書き込みはしていません")
            return

        upsert_params(session, param_rows)
        if par_rows:
            upsert_par(session, par_rows)
        session.commit()
        for y, rows in index_by_year.items():
            n = upsert_index(session, rows)
            session.commit()
            got = int(session.execute(text("SELECT COUNT(*) FROM megu_index WHERE model_version=:mv AND race_id LIKE :p"), {"mv": MODEL_VERSION, "p": f"{y}%"}).scalar())
            logger.info("%d: upsert %d 行 / DB の v3a 行 %d", y, n, got)
            if got < n:
                raise SystemExit(f"{y}: 件数が合わない（upsert {n} > DB {got}）")
        logger.info("反映後の v3a の行数: %s", count_v3a(session))


if __name__ == "__main__":
    main()
