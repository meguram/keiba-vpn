"""想定めぐ指数: 条件不一致重みのバックテスト・グリッドサーチ最適化。"""

from __future__ import annotations

import itertools
import logging
import statistics as stats
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.pipeline.megu_index.common import WEIGHTS_B
from src.pipeline.megu_index.condition_weights import (
    DEFAULT_CONDITION_WEIGHTS,
    save_condition_weights,
)
from src.pipeline.megu_index.predict import predict_megu_scores

logger = logging.getLogger(__name__)

MODEL_VERSION = "v2"
HISTORY_LIMIT = 5

# 最適化前の本番相当（条件不一致でも重みを下げない）
NO_DISCOUNT_WEIGHTS: dict[str, float] = {
    "w_match": 1.0,
    "w_surface_only": 1.0,
    "w_distance_only": 1.0,
    "w_both": 1.0,
}


@dataclass
class EvalMetrics:
    mae: float
    rmse: float
    mean_spearman: float
    n_pairs: int
    n_races: int
    score: float  # 複合目的関数（小さいほど良い）


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5:
        return None
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rankx = {rx[i]: i + 1 for i in range(n)}
    ranky = {ry[i]: i + 1 for i in range(n)}
    d = [rankx[i] - ranky[i] for i in range(n)]
    return 1 - 6 * sum(x * x for x in d) / (n * (n * n - 1))


def fetch_eval_frame(
    engine: Engine,
    *,
    year_start: int = 2024,
    year_end: int = 2025,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """
    評価用: 各 (race_id, horse_id) について過去走 JSON 列と実測めぐを返す。

    リーク防止: 過去走は race_date < 対象レース日。
    """
    sql = text("""
        WITH targets AS (
            SELECT
                mi.race_id,
                mi.horse_id,
                mi.megu_index AS actual_megu,
                mi.par_time_sec AS par_time_target,
                r.race_date,
                r.surface AS surface_target,
                r.distance AS distance_target,
                e.jockey_weight,
                COALESCE(NULLIF(e.sex_age, ''), h.sex) AS sex_age
            FROM megu_index mi
            JOIN races r ON r.race_id = mi.race_id
            LEFT JOIN entries e ON e.race_id = mi.race_id AND e.horse_id = mi.horse_id
            LEFT JOIN horses h ON h.horse_id = mi.horse_id
            WHERE mi.model_version = :mv
              AND mi.computation_status = 'valid'
              AND mi.megu_index IS NOT NULL
              AND mi.par_time_sec IS NOT NULL
              AND r.surface IN ('芝', 'ダート')
              AND r.race_date >= :d0 AND r.race_date < :d1
        ),
        hist AS (
            SELECT
                t.race_id,
                t.horse_id,
                t.actual_megu,
                t.race_date AS race_date,
                t.par_time_target,
                t.surface_target,
                t.distance_target,
                t.jockey_weight,
                t.sex_age,
                h.megu_index,
                h.par_time_sec,
                hr.surface,
                hr.distance,
                hr.race_date AS hist_race_date,
                ROW_NUMBER() OVER (
                    PARTITION BY t.race_id, t.horse_id
                    ORDER BY hr.race_date DESC, h.race_id DESC
                ) AS rn
            FROM targets t
            JOIN megu_index h
                ON h.horse_id = t.horse_id
               AND h.model_version = :mv
               AND h.computation_status = 'valid'
               AND h.megu_index IS NOT NULL
               AND h.par_time_sec IS NOT NULL
            JOIN races hr ON hr.race_id = h.race_id
            WHERE hr.race_date < t.race_date
              AND h.race_id != t.race_id
        )
        SELECT
            race_id,
            horse_id,
            actual_megu,
            par_time_target,
            surface_target,
            distance_target,
            jockey_weight,
            sex_age,
            race_date,
            json_agg(
                json_build_object(
                    'megu_index', megu_index,
                    'par_time_sec', par_time_sec,
                    'surface', surface,
                    'distance', distance
                )
                ORDER BY hist_race_date DESC
            ) FILTER (WHERE rn <= :lim) AS history_json
        FROM hist
        WHERE rn <= :lim
        GROUP BY race_id, horse_id, actual_megu, par_time_target, surface_target, distance_target,
                 jockey_weight, sex_age, race_date
        HAVING COUNT(*) >= 1
    """)
    d0 = f"{year_start}-01-01"
    d1 = f"{year_end + 1}-01-01"
    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"mv": model_version, "d0": d0, "d1": d1, "lim": HISTORY_LIMIT},
        )
    logger.info("eval frame: %d pairs (%d races)", len(df), df["race_id"].nunique())
    return df


def predict_from_row(
    row: pd.Series,
    *,
    condition_weights: dict[str, float] | None = None,
    transfer_map: dict | None = None,
    beta_weight: float = 0.0,
) -> float | None:
    hist = row.get("history_json") or []
    if not hist:
        return None
    pt = row.get("par_time_target")
    if pt is None or (isinstance(pt, float) and pd.isna(pt)):
        return None
    jw = row.get("jockey_weight")
    pred = predict_megu_scores(
        list(hist),
        par_time_target=float(pt),
        surface_target=str(row["surface_target"]),
        distance_target=int(row["distance_target"]),
        jockey_weight=float(jw) if jw is not None and not (isinstance(jw, float) and pd.isna(jw)) else None,
        sex_age=str(row["sex_age"]) if row.get("sex_age") and not pd.isna(row.get("sex_age")) else None,
        beta_weight=beta_weight,
        transfer_map=transfer_map or {},
        max_races=HISTORY_LIMIT,
        weights=WEIGHTS_B,
        condition_weights=condition_weights,
    )
    return pred.get("megu_final")


def evaluate_weights(
    df: pd.DataFrame,
    condition_weights: dict[str, float],
    *,
    spearman_weight: float = 5.0,
    beta_weight: float = 0.0,
) -> EvalMetrics:
    """
    複合スコア = MAE - spearman_weight * mean_spearman（Spearman 高いほど良い）。
    """
    preds: list[float] = []
    actuals: list[float] = []
    race_groups: dict[str, list[tuple[float, float]]] = {}

    baseline = dict(DEFAULT_CONDITION_WEIGHTS)
    cw = {**baseline, **condition_weights, "w_match": 1.0}

    for _, row in df.iterrows():
        p = predict_from_row(row, condition_weights=cw, beta_weight=beta_weight)
        if p is None:
            continue
        a = float(row["actual_megu"])
        preds.append(p)
        actuals.append(a)
        rid = str(row["race_id"])
        race_groups.setdefault(rid, []).append((p, a))

    if not preds:
        return EvalMetrics(999.0, 999.0, 0.0, 0, 0, 999.0)

    abs_d = [abs(a - p) for p, a in zip(preds, actuals)]
    delta = [a - p for p, a in zip(preds, actuals)]
    mae = stats.mean(abs_d)
    rmse = (stats.mean([d * d for d in delta])) ** 0.5
    spears = [
        s for g in race_groups.values() if (s := _spearman([x for x, _ in g], [y for _, y in g])) is not None
    ]
    mean_sp = stats.mean(spears) if spears else 0.0
    score = mae - spearman_weight * mean_sp
    return EvalMetrics(
        mae=mae,
        rmse=rmse,
        mean_spearman=mean_sp,
        n_pairs=len(preds),
        n_races=len(race_groups),
        score=score,
    )


def grid_search_condition_weights(
    df: pd.DataFrame,
    *,
    surface_grid: list[float] | None = None,
    distance_grid: list[float] | None = None,
    both_grid: list[float] | None = None,
    spearman_weight: float = 5.0,
    beta_weight: float = 0.0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, EvalMetrics]:
    """3軸グリッドサーチ。w_match は常に 1.0 固定。"""
    surface_grid = surface_grid or [0.15, 0.25, 0.35, 0.45, 0.55]
    distance_grid = distance_grid or [0.30, 0.40, 0.50, 0.60, 0.70]
    both_grid = both_grid or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    combos = list(itertools.product(surface_grid, distance_grid, both_grid))
    rows: list[dict[str, Any]] = []
    best_cw: dict[str, float] = dict(DEFAULT_CONDITION_WEIGHTS)
    best_metrics = EvalMetrics(999.0, 999.0, 0.0, 0, 0, 999.0)

    for i, (ws, wd, wb) in enumerate(combos):
        cw = {
            "w_match": 1.0,
            "w_surface_only": ws,
            "w_distance_only": wd,
            "w_both": wb,
        }
        m = evaluate_weights(df, cw, spearman_weight=spearman_weight, beta_weight=beta_weight)
        rows.append({
            "w_surface_only": ws,
            "w_distance_only": wd,
            "w_both": wb,
            "mae": m.mae,
            "rmse": m.rmse,
            "mean_spearman": m.mean_spearman,
            "score": m.score,
            "n_pairs": m.n_pairs,
        })
        if m.score < best_metrics.score:
            best_metrics = m
            best_cw = cw
        if progress_cb:
            progress_cb(i + 1, len(combos))

    result_df = pd.DataFrame(rows).sort_values("score")
    return best_cw, result_df, best_metrics


def run_optimization(
    engine: Engine,
    *,
    year_start: int = 2024,
    year_end: int = 2025,
    save: bool = True,
) -> dict[str, Any]:
    """フルパイプライン: データ取得 → ベースライン評価 → グリッドサーチ → JSON 保存。"""
    df = fetch_eval_frame(engine, year_start=year_start, year_end=year_end)

    beta_weight = 0.0
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT param_value FROM megu_regression_params "
                "WHERE param_name='beta_weight' AND model_version=:mv LIMIT 1"
            ),
            {"mv": MODEL_VERSION},
        ).fetchone()
        if row and row[0] is not None:
            beta_weight = float(row[0])

    baseline_m = evaluate_weights(df, NO_DISCOUNT_WEIGHTS, beta_weight=beta_weight)
    logger.info(
        "baseline (no discount) MAE=%.3f Spearman=%.3f (n=%d)",
        baseline_m.mae, baseline_m.mean_spearman, baseline_m.n_pairs,
    )

    default_m = evaluate_weights(df, DEFAULT_CONDITION_WEIGHTS, beta_weight=beta_weight)
    logger.info(
        "default weights MAE=%.3f Spearman=%.3f",
        default_m.mae, default_m.mean_spearman,
    )

    best_cw, grid_df, best_m = grid_search_condition_weights(df, beta_weight=beta_weight)
    logger.info(
        "best weights=%s MAE=%.3f Spearman=%.3f score=%.3f",
        best_cw, best_m.mae, best_m.mean_spearman, best_m.score,
    )

    meta = {
        "year_start": year_start,
        "year_end": year_end,
        "baseline": {
            "mae": baseline_m.mae,
            "mean_spearman": baseline_m.mean_spearman,
            "n_pairs": baseline_m.n_pairs,
        },
        "default_weights": {
            "mae": default_m.mae,
            "mean_spearman": default_m.mean_spearman,
        },
        "optimized": {
            "mae": best_m.mae,
            "rmse": best_m.rmse,
            "mean_spearman": best_m.mean_spearman,
            "score": best_m.score,
            "n_pairs": best_m.n_pairs,
            "n_races": best_m.n_races,
        },
        "spearman_weight_in_objective": 5.0,
    }

    out_path = None
    if save:
        out_path = save_condition_weights(best_cw, meta=meta)

    return {
        "best_weights": best_cw,
        "baseline_metrics": baseline_m,
        "best_metrics": best_m,
        "grid_results": grid_df,
        "config_path": str(out_path) if out_path else None,
        "meta": meta,
    }


def main() -> None:
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="想定めぐ指数: 条件不一致重みのグリッドサーチ最適化")
    parser.add_argument("--year-start", type=int, default=2024)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    from sqlalchemy import create_engine

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL が未設定です")
    result = run_optimization(
        create_engine(url),
        year_start=args.year_start,
        year_end=args.year_end,
        save=not args.no_save,
    )
    print("best:", result["best_weights"])
    print("config:", result["config_path"])


if __name__ == "__main__":
    main()
