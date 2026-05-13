"""
馬場傾向の当日/前日ブレンド重みを Optuna でチューニング（GCS 不要・ローカル Parquet のみ）。

LayerA に既に含まれる same_day_prior / prev_day / 血統勝率から、
加重列と交互作用を trial ごとに再計算し、valid の NDCG@3 を最大化する。

出力:
  data/meta/modeling/track_bias_weight_best.json

Usage:
  python3 -m research.tune_track_bias_weights --n-trials 60
  python3 -m pipeline.build_layer_a_dataset --track-bias-weights-json data/meta/modeling/track_bias_weight_best.json

LayerA は馬場区分・距離帯列が必要（track_bias_cond_bucket 等）。古い Parquet なら全期間で再ビルドすること。
test メトリクスまで書きたい場合は manifest の test 年（例: 2025）を --years に含める。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = ROOT / "data/modeling/layer_a_train.parquet"
DEFAULT_MANIFEST = ROOT / "data/meta/modeling/dataset_split_manifest.json"
DEFAULT_OUT = ROOT / "data/meta/modeling/track_bias_weight_best.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger(__name__)


def _group_sizes(race_id_series: pd.Series) -> np.ndarray:
    return race_id_series.groupby(race_id_series, sort=False).size().values


def _calc_ndcg(
    preds: np.ndarray,
    labels: pd.Series,
    groups: np.ndarray,
    k: int,
) -> float:
    from sklearn.metrics import ndcg_score

    scores: list[float] = []
    idx = 0
    for g in groups:
        g = int(g)
        if g < 2:
            idx += g
            continue
        y_true = labels.iloc[idx : idx + g].values.reshape(1, -1)
        y_score = preds[idx : idx + g].reshape(1, -1)
        try:
            scores.append(float(ndcg_score(y_true, y_score, k=k)))
        except ValueError:
            pass
        idx += g
    return float(np.mean(scores)) if scores else 0.0


def _train_lgb_rank_ndcg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    groups_valid: np.ndarray,
    seed: int,
    *,
    n_estimators: int,
    early_stopping: int,
) -> tuple[object, float]:
    import lightgbm as lgb

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_child_samples": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "lambda_l1": 0.15,
        "lambda_l2": 0.15,
        "seed": seed,
    }
    dtr = lgb.Dataset(X_train, label=y_train, group=groups_train)
    dva = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        valid_sets=[dva],
        callbacks=[
            lgb.early_stopping(early_stopping),
            lgb.log_evaluation(0),
        ],
    )
    preds = model.predict(X_valid)
    ndcg = _calc_ndcg(preds, y_valid, groups_valid, k=3)
    return model, ndcg


def run_tune(
    *,
    parquet_path: Path,
    manifest_path: Path,
    protocol: str,
    out_path: Path,
    n_trials: int,
    seed: int,
    n_estimators: int,
    early_stopping: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from pipeline.dataset_split import assign_split_column, load_manifest
    from pipeline.track_bias_pedigree import (
        apply_context_multipliers,
        compute_weighted_blend_series,
    )

    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    need = {
        "race_id",
        "finish_position",
        "track_bias_winner_3f_same_day_prior",
        "track_bias_winner_3f_prev_day",
        "track_bias_winner_3f_diff",
        "sire_prior_win_rate_surface",
        "dam_sire_prior_win_rate_surface",
        "track_bias_cond_bucket",
        "track_bias_dist_band",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(
            "LayerA に不足列があります（馬場区分・距離帯付き track_bias を再ビルドしてください）: "
            f"{miss}"
        )

    manifest = load_manifest(manifest_path)
    df = df.copy()
    df["_split"] = assign_split_column(df, manifest, protocol_key=protocol)
    df = df[df["_split"] != "unknown"]
    df = df.sort_values("race_id", kind="mergesort")

    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    y = ((fp >= 1) & (fp <= 3)).astype(int)
    group_col = df["race_id"]
    split_col = df["_split"]
    tr = split_col == "train"
    va = split_col == "valid"
    te = split_col == "test"
    if tr.sum() == 0 or va.sum() == 0:
        raise RuntimeError("train または valid が空です。全期間 LayerA を用意してください。")

    base = df.loc[:, list(need)].copy()
    sdp = base["track_bias_winner_3f_same_day_prior"]
    pdv = base["track_bias_winner_3f_prev_day"]
    diff = base["track_bias_winner_3f_diff"]
    sp = base["sire_prior_win_rate_surface"]
    dp = base["dam_sire_prior_win_rate_surface"]
    cb = base["track_bias_cond_bucket"]
    db = base["track_bias_dist_band"]

    def _norm_mean(arr: list[float]) -> np.ndarray:
        a = np.array(arr, dtype=np.float64)
        m = float(a.mean())
        return a / max(m, 1e-9)

    def build_feature_matrix(
        w_same: float,
        w_prev: float,
        cond_mult: dict[str, float],
        dist_mult: dict[str, float],
    ) -> pd.DataFrame:
        wcol = compute_weighted_blend_series(sdp, pdv, w_same, w_prev)
        wcol = apply_context_multipliers(wcol, cb, db, cond_mult, dist_mult)
        Xm = pd.DataFrame(
            {
                "track_bias_winner_3f_same_day_prior": sdp,
                "track_bias_winner_3f_prev_day": pdv,
                "track_bias_winner_3f_diff": diff,
                "track_bias_winner_3f_weighted": wcol,
                "sire_prior_win_rate_surface": sp,
                "dam_sire_prior_win_rate_surface": dp,
                "sire_x_track_bias_diff": sp * diff,
                "dam_sire_x_track_bias_diff": dp * diff,
                "sire_x_track_bias_weighted": sp * wcol,
                "dam_sire_x_track_bias_weighted": dp * wcol,
            },
            index=base.index,
        )
        return Xm.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    _dc = {"good": 1.0, "yielding_soft": 1.0, "heavy_bad": 1.0, "unknown": 1.0}
    _dd = {"sprint": 1.0, "mile": 1.0, "middle": 1.0, "long": 1.0, "unknown": 1.0}
    feature_cols = list(build_feature_matrix(0.75, 0.25, _dc, _dd).columns)

    y_tr, y_va, y_te = y.loc[tr], y.loc[va], y.loc[te]
    gc_tr = _group_sizes(group_col.loc[tr])
    gc_va = _group_sizes(group_col.loc[va])
    gc_te = _group_sizes(group_col.loc[te])

    import optuna

    def _trial_to_cond_dist(p: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        ck = ("good", "yielding_soft", "heavy_bad")
        dk = ("sprint", "mile", "middle", "long")
        cr = _norm_mean([float(p[f"cond_{k}"]) for k in ck])
        dr = _norm_mean([float(p[f"dist_{k}"]) for k in dk])
        cm = {k: float(cr[i]) for i, k in enumerate(ck)}
        cm["unknown"] = 1.0
        dm = {k: float(dr[i]) for i, k in enumerate(dk)}
        dm["unknown"] = 1.0
        return cm, dm

    def objective(trial: optuna.Trial) -> float:
        t = trial.suggest_float("same_day_ratio", 0.05, 0.95)
        w_same = t
        w_prev = 1.0 - t
        for k in ("good", "yielding_soft", "heavy_bad"):
            trial.suggest_float(f"cond_{k}", 0.5, 1.5)
        for k in ("sprint", "mile", "middle", "long"):
            trial.suggest_float(f"dist_{k}", 0.5, 1.5)
        p = trial.params
        cm, dm = _trial_to_cond_dist(p)
        Xm = build_feature_matrix(w_same, w_prev, cm, dm)
        X_train = Xm.loc[tr]
        X_valid = Xm.loc[va]
        _, ndcg = _train_lgb_rank_ndcg(
            X_train,
            y_tr,
            gc_tr,
            X_valid,
            y_va,
            gc_va,
            seed=seed + trial.number,
            n_estimators=n_estimators,
            early_stopping=early_stopping,
        )
        return ndcg

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = round(time.time() - t0, 1)

    best_p = study.best_params
    best_sw = float(best_p["same_day_ratio"])
    best_pw = float(1.0 - best_p["same_day_ratio"])
    best_cm, best_dm = _trial_to_cond_dist(best_p)

    Xm_best = build_feature_matrix(best_sw, best_pw, best_cm, best_dm)
    X_train = Xm_best.loc[tr]
    X_valid = Xm_best.loc[va]
    X_test = Xm_best.loc[te]

    model, valid_ndcg = _train_lgb_rank_ndcg(
        X_train,
        y_tr,
        gc_tr,
        X_valid,
        y_va,
        gc_va,
        seed=seed,
        n_estimators=max(n_estimators, 400),
        early_stopping=early_stopping,
    )
    preds_valid = model.predict(X_valid)

    from research.evaluate_track_bias_pedigree_only import _evaluate_ranking  # noqa: PLC0415

    m_valid = _evaluate_ranking(preds_valid, y_va, gc_va)
    if len(X_test) > 0:
        preds_test = model.predict(X_test)
        m_test = _evaluate_ranking(preds_test, y_te, gc_te)
    else:
        m_test = {
            "skipped": True,
            "reason": "Parquet に manifest の test 年が含まれていません（例: 2025 を --years に含めて全期間ビルド）",
        }

    importance = dict(
        zip(feature_cols, model.feature_importance(importance_type="gain").tolist())
    )

    result: dict[str, Any] = {
        "version": 2,
        "best_same_day_weight": best_sw,
        "best_prev_day_weight": best_pw,
        "cond_multipliers": best_cm,
        "dist_multipliers": best_dm,
        "metric_objective": "valid_ndcg_3",
        "best_valid_ndcg_3": round(float(study.best_value), 5),
        "refit_valid_ndcg_3": round(valid_ndcg, 5),
        "metrics_valid_at_best": m_valid,
        "metrics_test_at_best": m_test,
        "n_trials": n_trials,
        "n_trials_completed": len(study.trials),
        "elapsed_sec": elapsed,
        "parquet": str(parquet_path),
        "split_protocol": protocol,
        "feature_columns": feature_cols,
        "n_estimators_tune": n_estimators,
        "feature_importance_gain": dict(
            sorted(importance.items(), key=lambda x: -x[1])[:15]
        ),
        "notes": [
            "チューニングは valid の NDCG@3 のみ。test は最終報告用（リーク回避のため study に使わない）。",
            "GCS は不使用。LayerA の列のみ再計算。",
            "metrics_test_at_best は test 年が Parquet に無い場合スキップ（train+valid のみビルド時）。",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "保存: %s (same=%.4f prev=%.4f cond=%s dist=%s valid_ndcg=%.5f)",
        out_path,
        best_sw,
        best_pw,
        {k: round(v, 4) for k, v in best_cm.items() if k != "unknown"},
        {k: round(v, 4) for k, v in best_dm.items() if k != "unknown"},
        study.best_value,
    )
    return result


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="馬場傾向ブレンド重みの Optuna チューニング")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--protocol", default="model_selection_primary")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=250, help="各 trial 内の最大ブースト回数")
    ap.add_argument("--early-stopping", type=int, default=35)
    args = ap.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    r = run_tune(
        parquet_path=args.parquet,
        manifest_path=args.manifest,
        protocol=args.protocol,
        out_path=args.out,
        n_trials=args.n_trials,
        seed=args.seed,
        n_estimators=args.n_estimators,
        early_stopping=args.early_stopping,
    )
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
