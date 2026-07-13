"""想定めぐ指数: 実測との MAE 最小化チューニング（1点=0.1秒厳守）。"""

from __future__ import annotations

import itertools
import logging
import random
import statistics as stats
from dataclasses import asdict
from typing import Any, Callable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.pipeline.megu_index.common import WEIGHTS_B
from src.pipeline.megu_index.condition_weights import DEFAULT_CONDITION_WEIGHTS
from src.pipeline.megu_index.optimize_condition_weights import (
    EvalMetrics,
    _spearman,
    fetch_eval_frame,
)
from src.pipeline.megu_index.predict import predict_megu_scores
from src.pipeline.megu_index.predict_params import (
    PredictParams,
    PredictTuning,
    save_predict_params,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "v2"
HISTORY_LIMIT = 5

HISTORY_PRESETS: list[list[float]] = [
    list(WEIGHTS_B),
    [0.50, 0.25, 0.15, 0.07, 0.03],
    [0.40, 0.22, 0.18, 0.12, 0.08],
    [0.60, 0.20, 0.10, 0.06, 0.04],
    [0.20, 0.20, 0.20, 0.20, 0.20],
]


def load_transfer_map(engine: Engine, model_version: str = MODEL_VERSION) -> dict:
    transfer_map: dict = {}
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT surface_from, dist_band_from, surface_to, dist_band_to,
                       delta_mean, delta_std, sample_count
                FROM megu_condition_transfer
                WHERE model_version = :mv
            """),
            {"mv": model_version},
        ).fetchall()
    for tr in rows:
        transfer_map[(tr.surface_from, tr.dist_band_from, tr.surface_to, tr.dist_band_to)] = {
            "delta_sec": float(tr.delta_mean),
            "delta_std": float(tr.delta_std) if tr.delta_std is not None else None,
            "sample_count": int(tr.sample_count),
        }
    return transfer_map


def load_beta_weight(engine: Engine, model_version: str = MODEL_VERSION) -> float:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT param_value FROM megu_regression_params "
                "WHERE param_name='beta_weight' AND model_version=:mv LIMIT 1"
            ),
            {"mv": model_version},
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
        row = conn.execute(
            text(
                "SELECT param_value FROM megu_regression_params "
                "WHERE param_name='beta_weight' AND model_version='v1' LIMIT 1"
            ),
        ).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return 0.612596


def predict_row(
    row: pd.Series,
    params: PredictParams,
    *,
    transfer_map: dict,
    beta_weight: float,
) -> float | None:
    hist = row.get("history_json") or []
    if not hist:
        return None
    pt = row.get("par_time_target")
    if pt is None or (isinstance(pt, float) and pd.isna(pt)):
        return None
    jw = row.get("jockey_weight")
    n = min(len(hist), HISTORY_LIMIT)
    pred = predict_megu_scores(
        list(hist),
        par_time_target=float(pt),
        surface_target=str(row["surface_target"]),
        distance_target=int(row["distance_target"]),
        jockey_weight=float(jw) if jw is not None and not (isinstance(jw, float) and pd.isna(jw)) else None,
        sex_age=str(row["sex_age"]) if row.get("sex_age") and not pd.isna(row.get("sex_age")) else None,
        beta_weight=beta_weight,
        transfer_map=transfer_map,
        max_races=HISTORY_LIMIT,
        weights=params.normalized_history_weights(n),
        condition_weights=params.condition_weights,
        tuning=params.tuning,
    )
    return pred.get("megu_final")


def evaluate_params(
    df: pd.DataFrame,
    params: PredictParams,
    *,
    transfer_map: dict,
    beta_weight: float,
    spearman_weight: float = 2.0,
) -> EvalMetrics:
    preds: list[float] = []
    actuals: list[float] = []
    race_groups: dict[str, list[tuple[float, float]]] = {}

    for _, row in df.iterrows():
        p = predict_row(row, params, transfer_map=transfer_map, beta_weight=beta_weight)
        if p is None:
            continue
        a = float(row["actual_megu"])
        preds.append(p)
        actuals.append(a)
        race_groups.setdefault(str(row["race_id"]), []).append((p, a))

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
    return EvalMetrics(mae=mae, rmse=rmse, mean_spearman=mean_sp, n_pairs=len(preds), n_races=len(race_groups), score=score)


def split_temporal(
    df: pd.DataFrame,
    *,
    train_end: str = "2024-12-31",
    valid_end: str = "2025-06-30",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    レース日付で train / valid / test に分割（リーク防止: レース単位）。

    - train: race_date <= train_end
    - valid: train_end < race_date <= valid_end
    - test:  race_date > valid_end
    """
    out = df.copy()
    out["race_date"] = pd.to_datetime(out["race_date"])
    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(valid_end)
    train = out[out["race_date"] <= t_end].copy()
    valid = out[(out["race_date"] > t_end) & (out["race_date"] <= v_end)].copy()
    test = out[out["race_date"] > v_end].copy()
    return train, valid, test


def metrics_to_dict(m: EvalMetrics, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "mae": m.mae,
        "rmse": m.rmse,
        "mean_spearman": m.mean_spearman,
        "score": m.score,
        "n_pairs": m.n_pairs,
        "n_races": m.n_races,
    }


def _subsample_df(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    rng = random.Random(seed)
    race_ids = df["race_id"].unique().tolist()
    rng.shuffle(race_ids)
    picked: list[str] = []
    count = 0
    for rid in race_ids:
        n = int((df["race_id"] == rid).sum())
        if count + n > max_rows and picked:
            break
        picked.append(rid)
        count += n
    return df[df["race_id"].isin(picked)].copy()


def grid_search_predict(
    df: pd.DataFrame,
    *,
    transfer_map: dict,
    beta_weight: float,
    spearman_weight: float = 2.0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[PredictParams, pd.DataFrame, EvalMetrics]:
    """2段階: 能力ブレンド・bias → 条件重み微調整。"""
    phase1_par = [0.0, 0.25, 0.5, 0.75, 1.0]
    phase1_bias = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    phase1_transfer = [0.0, 0.5, 1.0]
    phase1_hist = HISTORY_PRESETS[:3]

    rows: list[dict[str, Any]] = []
    best_params = PredictParams()
    best_m = EvalMetrics(999.0, 999.0, 0.0, 0, 0, 999.0)

    phase1 = list(itertools.product(phase1_hist, phase1_par, phase1_bias, phase1_transfer))
    for i, (hw, pb, bias, ts) in enumerate(phase1):
        params = PredictParams(
            history_weights=list(hw),
            tuning=PredictTuning(par_blend=pb, ability_bias_sec=bias, transfer_strength=ts),
        )
        m = evaluate_params(
            df, params, transfer_map=transfer_map, beta_weight=beta_weight,
            spearman_weight=spearman_weight,
        )
        rows.append({"phase": 1, "par_blend": pb, "ability_bias_sec": bias, "transfer_strength": ts, "mae": m.mae, "score": m.score})
        if m.score < best_m.score:
            best_m, best_params = m, params
        if progress_cb:
            progress_cb(i + 1, len(phase1))

    surface_grid = [0.25, 0.45, 0.65, 0.85]
    distance_grid = [0.40, 0.60, 0.80]
    both_grid = [0.10, 0.25, 0.40]
    phase2 = list(itertools.product(surface_grid, distance_grid, both_grid))
    for j, (ws, wd, wb) in enumerate(phase2):
        params = PredictParams(
            history_weights=best_params.history_weights,
            condition_weights={"w_match": 1.0, "w_surface_only": ws, "w_distance_only": wd, "w_both": wb},
            tuning=best_params.tuning,
        )
        m = evaluate_params(
            df, params, transfer_map=transfer_map, beta_weight=beta_weight,
            spearman_weight=spearman_weight,
        )
        rows.append({"phase": 2, "w_surface_only": ws, "w_distance_only": wd, "w_both": wb, "mae": m.mae, "score": m.score})
        if m.score < best_m.score:
            best_m, best_params = m, params
        if progress_cb:
            progress_cb(len(phase1) + j + 1, len(phase1) + len(phase2))

    result_df = pd.DataFrame(rows).sort_values("score")
    return best_params, result_df, best_m


def refine_bias(
    df: pd.DataFrame,
    params: PredictParams,
    *,
    transfer_map: dict,
    beta_weight: float,
    spearman_weight: float = 2.0,
) -> tuple[PredictParams, EvalMetrics]:
    """最良付近で ability_bias_sec を 0.1秒刻みで微調整。"""
    center = params.tuning.ability_bias_sec
    best_params = params
    best_m = evaluate_params(
        df, params, transfer_map=transfer_map, beta_weight=beta_weight,
        spearman_weight=spearman_weight,
    )
    for bias in [round(center + d * 0.1, 1) for d in range(-20, 21)]:
        if bias < -3.0 or bias > 3.0:
            continue
        trial = PredictParams(
            history_weights=params.history_weights,
            condition_weights=params.condition_weights,
            tuning=PredictTuning(
                par_blend=params.tuning.par_blend,
                ability_bias_sec=bias,
                transfer_strength=params.tuning.transfer_strength,
            ),
        )
        m = evaluate_params(
            df, trial, transfer_map=transfer_map, beta_weight=beta_weight,
            spearman_weight=spearman_weight,
        )
        if m.score < best_m.score:
            best_m = m
            best_params = trial
    return best_params, best_m


def run_optimization(
    engine: Engine,
    *,
    year_start: int = 2024,
    year_end: int = 2025,
    train_sample: int = 20000,
    save: bool = True,
) -> dict[str, Any]:
    df_full = fetch_eval_frame(engine, year_start=year_start, year_end=year_end)
    df_train = _subsample_df(df_full, train_sample)
    transfer_map = load_transfer_map(engine)
    beta_weight = load_beta_weight(engine)

    default_params = PredictParams()
    baseline_m = evaluate_params(
        df_train, default_params, transfer_map=transfer_map, beta_weight=beta_weight,
    )
    logger.info(
        "baseline MAE=%.3f Spearman=%.3f (train n=%d)",
        baseline_m.mae, baseline_m.mean_spearman, baseline_m.n_pairs,
    )

    def _prog(done: int, total: int) -> None:
        if done % 1000 == 0 or done == total:
            logger.info("grid %d / %d", done, total)

    best_params, grid_df, train_m = grid_search_predict(
        df_train, transfer_map=transfer_map, beta_weight=beta_weight, progress_cb=_prog,
    )
    logger.info(
        "grid best MAE=%.3f tuning=%s cw=%s hw=%s",
        train_m.mae, asdict(best_params.tuning), best_params.condition_weights, best_params.history_weights,
    )

    best_params, refine_m = refine_bias(
        df_train, best_params, transfer_map=transfer_map, beta_weight=beta_weight,
    )
    logger.info("refined bias=%.2f MAE=%.3f", best_params.tuning.ability_bias_sec, refine_m.mae)

    full_m = evaluate_params(
        df_full, best_params, transfer_map=transfer_map, beta_weight=beta_weight,
    )
    logger.info(
        "full eval MAE=%.3f RMSE=%.3f Spearman=%.3f (n=%d)",
        full_m.mae, full_m.rmse, full_m.mean_spearman, full_m.n_pairs,
    )

    best_params.meta = {
        "model_version": MODEL_VERSION,
        "year_start": year_start,
        "year_end": year_end,
        "train_sample": train_sample,
        "baseline": {
            "mae": baseline_m.mae,
            "mean_spearman": baseline_m.mean_spearman,
            "n_pairs": baseline_m.n_pairs,
        },
        "train_optimized": {
            "mae": refine_m.mae,
            "mean_spearman": refine_m.mean_spearman,
            "score": refine_m.score,
        },
        "full_eval": {
            "mae": full_m.mae,
            "rmse": full_m.rmse,
            "mean_spearman": full_m.mean_spearman,
            "n_pairs": full_m.n_pairs,
            "n_races": full_m.n_races,
        },
        "note": "1点=0.1秒は不変。ability_bias_sec は秒単位キャリブレーション。",
    }

    out_path = None
    if save:
        out_path = save_predict_params(best_params)

    return {
        "best_params": best_params,
        "baseline_metrics": baseline_m,
        "train_metrics": refine_m,
        "full_metrics": full_m,
        "grid_results": grid_df,
        "config_path": str(out_path) if out_path else None,
    }


def main() -> None:
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="想定めぐ指数: 実測 MAE 最小化チューニング")
    parser.add_argument("--year-start", type=int, default=2024)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--train-sample", type=int, default=20000)
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
        train_sample=args.train_sample,
        save=not args.no_save,
    )
    p = result["best_params"]
    print("MAE full:", result["full_metrics"].mae)
    print("tuning:", asdict(p.tuning))
    print("condition_weights:", p.condition_weights)
    print("history_weights:", p.history_weights)
    print("config:", result["config_path"])


if __name__ == "__main__":
    main()
