"""
血統×馬場傾向特徴のみでの予測寄与検証。

LayerA 母表から track_bias_pedigree 由来の数値列だけを使い、
本番と同様のマニフェスト分割（train/valid/test）で LightGBM LambdaRank を学習し、
AUC / NDCG / top-k 的中率を報告する。

Usage:
  python3 -m research.evaluate_track_bias_pedigree_only
  python3 -m research.evaluate_track_bias_pedigree_only --parquet data/modeling/layer_a_train.parquet
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

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.keiba_logging import script_basic_config  # noqa: E402

# pipeline/track_bias_pedigree.py が付与する数値特徴（他列は一切使わない）
TRACK_BIAS_PEDIGREE_FEATURES = [
    "track_bias_winner_3f_same_day_prior",
    "track_bias_winner_3f_prev_day",
    "track_bias_winner_3f_diff",
    "track_bias_winner_3f_weighted",
    "sire_prior_win_rate_surface",
    "dam_sire_prior_win_rate_surface",
    "sire_x_track_bias_diff",
    "dam_sire_x_track_bias_diff",
    "sire_x_track_bias_weighted",
    "dam_sire_x_track_bias_weighted",
]

DEFAULT_PARQUET = ROOT / "data/modeling/layer_a_train.parquet"
DEFAULT_MANIFEST = ROOT / "data/meta/modeling/dataset_split_manifest.json"
DEFAULT_OUT = ROOT / "data/meta/modeling/track_bias_pedigree_only_metrics.json"


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


def _top_k_hit_rate(
    preds: np.ndarray,
    labels: pd.Series,
    groups: np.ndarray,
    k: int,
) -> float:
    hits = 0
    total = 0
    idx = 0
    for g in groups:
        g = int(g)
        if g < 2:
            idx += g
            continue
        pred_slice = preds[idx : idx + g]
        label_slice = labels.iloc[idx : idx + g].values
        top_k_idx = np.argsort(-pred_slice)[:k]
        if any(label_slice[i] == 1 for i in top_k_idx):
            hits += 1
        total += 1
        idx += g
    return hits / total if total > 0 else 0.0


def _evaluate_ranking(
    preds: np.ndarray,
    y: pd.Series,
    groups: np.ndarray | None,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    m: dict[str, Any] = {}
    yv = y.values if hasattr(y, "values") else y
    if len(np.unique(yv)) > 1:
        m["auc"] = round(float(roc_auc_score(yv, preds)), 4)
        pp = np.clip(preds, 1e-7, 1 - 1e-7)
        m["log_loss"] = round(float(log_loss(yv, pp)), 4)
    else:
        m["auc"] = 0.0
        m["log_loss"] = 0.0
    m["accuracy"] = round(float(accuracy_score(yv, (preds > 0.5).astype(int))), 4)
    if groups is not None and len(groups) > 0:
        m["ndcg_3"] = round(_calc_ndcg(preds, y, groups, k=3), 4)
        m["ndcg_5"] = round(_calc_ndcg(preds, y, groups, k=5), 4)
        m["top1_hit_rate"] = round(_top_k_hit_rate(preds, y, groups, k=1), 4)
        m["top3_hit_rate"] = round(_top_k_hit_rate(preds, y, groups, k=3), 4)
    return m


def _assign_random_race_split(
    df: pd.DataFrame,
    *,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> pd.Series:
    """race_id 単位でシャッフルし、train/valid/test を付与（単年データ検証用）。"""
    rng = np.random.default_rng(seed)
    races = df["race_id"].drop_duplicates().values
    rng.shuffle(races)
    n = len(races)
    n_tr = int(n * train_ratio)
    n_va = int(n * valid_ratio)
    tr_set = set(races[:n_tr])
    va_set = set(races[n_tr : n_tr + n_va])
    te_set = set(races[n_tr + n_va :])
    rid = df["race_id"]
    s = pd.Series("unknown", index=df.index, dtype=object)
    s[rid.isin(tr_set)] = "train"
    s[rid.isin(va_set)] = "valid"
    s[rid.isin(te_set)] = "test"
    return s


def run_eval(
    *,
    parquet_path: Path,
    manifest_path: Path,
    protocol: str,
    out_path: Path,
    seed: int,
    fallback_race_split: bool,
    train_ratio: float,
    valid_ratio: float,
    weights_json: Path | None = None,
) -> dict[str, Any]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet がありません: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    need = {"race_id", "finish_position"} | set(TRACK_BIAS_PEDIGREE_FEATURES)
    missing = [c for c in TRACK_BIAS_PEDIGREE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "LayerA に血統×馬場傾向列がありません。先に "
            "`python3 -m pipeline.build_layer_a_dataset`（track_bias 付き）を実行してください。"
            f" 不足: {missing}"
        )

    if weights_json is not None and weights_json.is_file():
        from pipeline.track_bias_pedigree import (
            load_track_bias_weights_config,
            recompute_weighted_interactions,
        )

        cfg = load_track_bias_weights_config(weights_json)
        df = recompute_weighted_interactions(
            df,
            float(cfg["best_same_day_weight"]),
            float(cfg["best_prev_day_weight"]),
            cond_multipliers=cfg["cond_multipliers"],
            dist_multipliers=cfg["dist_multipliers"],
        )
        logger.info(
            "チューニング済み重みを適用: %s (same=%.4f prev=%.4f, v=%s)",
            weights_json,
            cfg["best_same_day_weight"],
            cfg["best_prev_day_weight"],
            cfg.get("version", 1),
        )

    split_mode = "manifest"
    manifest: dict[str, Any] | None = None
    if fallback_race_split:
        df = df.copy()
        df["_split"] = _assign_random_race_split(
            df, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed
        )
        split_mode = "random_race"
    else:
        from pipeline.dataset_split import assign_split_column, load_manifest

        manifest = load_manifest(manifest_path)
        df = df.copy()
        df["_split"] = assign_split_column(df, manifest, protocol_key=protocol)
        df = df[df["_split"] != "unknown"].copy()
        if df.empty:
            raise RuntimeError("マニフェスト分割後にデータが空です（race_id 年が対象外）")

    feat_cols = [c for c in TRACK_BIAS_PEDIGREE_FEATURES if c in df.columns]
    df = df.sort_values("race_id", kind="mergesort")
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    y = ((fp >= 1) & (fp <= 3)).astype(int)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    group_col = df["race_id"]
    split_col = df["_split"]

    tr = split_col == "train"
    va = split_col == "valid"
    te = split_col == "test"
    if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
        raise RuntimeError(
            f"train/valid/test のいずれかが空です (train={tr.sum()} valid={va.sum()} test={te.sum()})。"
            "Parquet にマニフェスト対象年が含まれるよう `python3 -m pipeline.build_layer_a_dataset` で全期間ビルドするか、"
            "`--fallback-race-split` でレース単位ランダム分割を使ってください。"
        )

    X_train, y_train = X.loc[tr], y.loc[tr]
    X_valid, y_valid = X.loc[va], y.loc[va]
    X_test, y_test = X.loc[te], y.loc[te]
    gc_tr = group_col.loc[tr]
    gc_va = group_col.loc[va]
    gc_te = group_col.loc[te]
    groups_train = _group_sizes(gc_tr)
    groups_valid = _group_sizes(gc_va)
    groups_test = _group_sizes(gc_te)

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError("LightGBM が必要です: pip install lightgbm") from e

    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": seed,
    }
    n_est = 400
    early = 40

    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, group=groups_valid, reference=train_data
    )
    t0 = time.time()
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=n_est,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(early), lgb.log_evaluation(0)],
    )
    train_sec = round(time.time() - t0, 1)

    preds_test = model.predict(X_test)
    preds_valid = model.predict(X_valid)

    metrics_test = _evaluate_ranking(preds_test, y_test, groups_test)
    metrics_valid = _evaluate_ranking(preds_valid, y_valid, groups_valid)
    metrics_test["label_top3_rate"] = round(float(y_test.mean()), 4)
    metrics_valid["label_top3_rate"] = round(float(y_valid.mean()), 4)

    importance = dict(
        zip(feat_cols, model.feature_importance(importance_type="gain").tolist())
    )
    top_feat = dict(sorted(importance.items(), key=lambda x: -x[1])[:20])

    result: dict[str, Any] = {
        "description": "血統×馬場傾向特徴のみ（他列不使用）のランキング検証",
        "parquet": str(parquet_path),
        "split_mode": split_mode,
        "split_protocol": protocol,
        "manifest": str(manifest_path) if manifest else None,
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "n_rows_total": len(df),
        "n_train": int(tr.sum()),
        "n_valid": int(va.sum()),
        "n_test": int(te.sum()),
        "n_races_train": int(len(groups_train)),
        "n_races_valid": int(len(groups_valid)),
        "n_races_test": int(len(groups_test)),
        "training_time_sec": train_sec,
        "seed": seed,
        "metrics_test": metrics_test,
        "metrics_valid": metrics_valid,
        "feature_importance_gain": top_feat,
        "baseline_note": (
            "top3 二値ラベル + LambdaRank。accuracy はクラス不均衡で高く出やすい（参考程度）。"
            "AUC・NDCG・top1/3_hit_rate を主に見る。log_loss はランクスコアをそのまま入れた参照値で厳密でない。"
            "他特徴ゼロのため AUC は 0.5 付近になりやすい。"
        ),
        "label_top3_prevalence_train": round(float(y_train.mean()), 4),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("結果を保存: %s", out_path)
    return result


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="血統×馬場傾向のみの学習検証")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--protocol",
        default="model_selection_primary",
        help="dataset_split_manifest.json の protocols キー",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--weights-json",
        type=Path,
        default=None,
        help="tune_track_bias_weights の出力 JSON（加重列・交互作用を再計算してから評価）",
    )
    ap.add_argument(
        "--fallback-race-split",
        action="store_true",
        help="マニフェストではなく race_id 単位の乱数分割（train/valid/test=0.7/0.15/0.15）。単年 Parquet の試行用",
    )
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--valid-ratio", type=float, default=0.15)
    args = ap.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    result = run_eval(
        parquet_path=args.parquet,
        manifest_path=args.manifest,
        protocol=args.protocol,
        out_path=args.out,
        seed=args.seed,
        fallback_race_split=args.fallback_race_split,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        weights_json=args.weights_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
