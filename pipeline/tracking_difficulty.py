"""
追走難度 (Tracking Difficulty) モジュール

「今回いつもと同じ位置を取るのにどれくらい楽か」を数値化する。

== コンセプト ==
各馬には「好む脚質ポジション」（直近走の通過順平均）がある。
今回のレースで、ゲート位置・隣枠の馬の傾向・頭数・コース特性などから
そのポジションを取る難易度を推定する。

== 特徴量設計 ==
1. 馬の脚質プロファイル
   - 直近走の平均初角通過順位（正規化: pos / field_size）
   - 脚質のバラつき（安定 vs 不安定）
   - 前走からの脚質変化トレンド
2. ゲート要因
   - 馬番（内/外）
   - 枠番の偶奇（偶数枠 = ゲート入り遅い → スタートしやすい）
   - 馬番の正規化位置（horse_number / field_size）
3. 隣枠影響
   - 左隣・右隣馬の平均初角通過順位
   - 隣馬が出遅れ傾向か（初角順位 > 頭数の60%）
   - 隣馬のスタート安定性
4. 場全体の構成
   - 同脚質馬（先行馬/差し馬）の数
   - ペース予測（先行馬比率）
   - 頭数
5. コース・距離要因
   - 芝/ダート
   - 距離帯（短距離/中距離/長距離）
   - 馬場状態

== ターゲット変数 ==
position_deviation = (typical_position_normalized - actual_position_normalized)
正 = 想定より楽にポジション取得、負 = 想定より困難
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline.tracking_difficulty")


# ── 定数 ──

RUNNING_STYLE_THRESHOLDS = {
    "逃げ": 0.15,
    "先行": 0.35,
    "差し": 0.65,
    "追込": 1.0,
}

EXPERIMENT_NAME = "tracking-difficulty"
MODEL_NAME = "tracking-difficulty-lgbm"


# ── ユーティリティ ──

def _parse_passing(passing_order: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", passing_order or "")]


def _norm_pos(position: float, field_size: int) -> float:
    if field_size <= 1:
        return 0.5
    return (position - 1) / (field_size - 1)


def _classify_style(norm_pos: float) -> str:
    for style, threshold in RUNNING_STYLE_THRESHOLDS.items():
        if norm_pos <= threshold:
            return style
    return "追込"


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


# ── 馬の脚質プロファイル ──

def build_horse_profile(race_history: list[dict], max_races: int = 10) -> dict:
    """直近走から馬の脚質プロファイルを構築する。"""
    recent = race_history[:max_races]
    if not recent:
        return {
            "typical_norm_pos": 0.5,
            "style": "差し",
            "pos_std": 0.2,
            "n_races": 0,
            "positions": [],
            "start_tendency": 0.5,
        }

    norm_positions = []
    for r in recent:
        passing = _parse_passing(r.get("passing_order", ""))
        fs = r.get("field_size", 0)
        if passing and fs >= 4:
            norm_positions.append(_norm_pos(passing[0], fs))

    if not norm_positions:
        return {
            "typical_norm_pos": 0.5,
            "style": "差し",
            "pos_std": 0.2,
            "n_races": len(recent),
            "positions": [],
            "start_tendency": 0.5,
        }

    weights = [1.0 / (i + 1) for i in range(len(norm_positions))]
    w_sum = sum(weights)
    typical = sum(p * w for p, w in zip(norm_positions, weights)) / w_sum

    return {
        "typical_norm_pos": round(typical, 4),
        "style": _classify_style(typical),
        "pos_std": round(_safe_std(norm_positions), 4),
        "n_races": len(norm_positions),
        "positions": norm_positions[:5],
        "start_tendency": round(typical, 4),
    }


# ── 学習データ構築 ──

def build_training_row(
    entry: dict,
    race_info: dict,
    horse_profile: dict,
    neighbor_profiles: dict,
    field_profiles: list[dict],
) -> dict | None:
    """1頭1レース分の学習用レコードを構築する。"""
    passing = _parse_passing(entry.get("passing_order", ""))
    fs = race_info.get("field_size", 0)
    if not passing or fs < 4:
        return None

    actual_norm = _norm_pos(passing[0], fs)
    typical = horse_profile["typical_norm_pos"]
    deviation = typical - actual_norm

    hn = entry.get("horse_number", 0)
    bn = entry.get("bracket_number", 0)

    front_runners = sum(
        1 for p in field_profiles if p["style"] in ("逃げ", "先行")
    )
    closers = sum(
        1 for p in field_profiles if p["style"] in ("差し", "追込")
    )
    same_style_count = sum(
        1 for p in field_profiles if p["style"] == horse_profile["style"]
    )

    left_profile = neighbor_profiles.get("left", {})
    right_profile = neighbor_profiles.get("right", {})

    left_tendency = left_profile.get("start_tendency", 0.5)
    right_tendency = right_profile.get("start_tendency", 0.5)
    left_std = left_profile.get("pos_std", 0.2)
    right_std = right_profile.get("pos_std", 0.2)

    # 遅い隣馬（後方スタート傾向）
    neighbor_slow = (
        (1 if left_tendency > 0.5 else 0)
        + (1 if right_tendency > 0.5 else 0)
    )

    # 速い隣馬（前走3番手以内＝0.2以下）のカウント
    # 大内枠（1番）：外側が速ければOK
    # 大外枠（最外）：内側が速ければOK
    # 中枠：両サイドともに速い場合に恩恵
    neighbor_fast = 0
    if hn == 1:
        # 大内枠：右隣（外側）が速ければプラス
        if right_tendency <= 0.2:
            neighbor_fast = 1
    elif hn == fs:
        # 大外枠：左隣（内側）が速ければプラス
        if left_tendency <= 0.2:
            neighbor_fast = 1
    else:
        # 中枠：両サイドともに速い場合のみプラス
        if left_tendency <= 0.2 and right_tendency <= 0.2:
            neighbor_fast = 1

    neighbor_avg_tendency = (left_tendency + right_tendency) / 2

    distance = race_info.get("distance", 1600)
    surface = race_info.get("surface", "芝")
    track_cond = race_info.get("track_condition", "良")

    row = {
        "race_id": race_info.get("race_id", ""),
        "horse_number": hn,
        "horse_id": entry.get("horse_id", ""),

        # ── ターゲット ──
        "position_deviation": round(deviation, 5),
        "actual_norm_pos": round(actual_norm, 4),

        # ── 馬の脚質 ──
        "typical_norm_pos": typical,
        "style_code": list(RUNNING_STYLE_THRESHOLDS.keys()).index(
            horse_profile["style"]
        ),
        "pos_std": horse_profile["pos_std"],
        "n_past_races": horse_profile["n_races"],

        # ── ゲート要因 ──
        "horse_number_norm": round(hn / fs, 4) if fs > 0 else 0.5,
        "bracket_number": bn,
        "is_even_gate": int(hn % 2 == 0),
        "gate_zone": (
            0 if hn <= fs * 0.25 else
            1 if hn <= fs * 0.5 else
            2 if hn <= fs * 0.75 else 3
        ),

        # ── 隣枠影響 ──
        "left_neighbor_tendency": round(left_tendency, 4),
        "right_neighbor_tendency": round(right_tendency, 4),
        "left_neighbor_std": round(left_std, 4),
        "right_neighbor_std": round(right_std, 4),
        "neighbor_slow_count": neighbor_slow,
        "neighbor_fast_count": neighbor_fast,
        "neighbor_avg_tendency": round(neighbor_avg_tendency, 4),
        "neighbor_space_score": round(
            (left_tendency + right_tendency) / 2 - typical + neighbor_fast * 0.1 - neighbor_slow * 0.05, 4
        ),

        # ── 場全体の構成 ──
        "field_size": fs,
        "front_runner_count": front_runners,
        "closer_count": closers,
        "same_style_count": same_style_count,
        "front_runner_ratio": round(front_runners / fs, 4) if fs > 0 else 0,
        "pace_pressure": round(front_runners / max(closers, 1), 4),

        # ── コース要因 ──
        "surface_code": 0 if surface == "芝" else 1,
        "distance": distance,
        "distance_category": (
            0 if distance <= 1400 else
            1 if distance <= 1800 else
            2 if distance <= 2200 else 3
        ),
        "track_condition_code": (
            0 if track_cond == "良" else
            1 if track_cond == "稍" else
            2 if track_cond == "重" else 3
        ),
    }

    return row


def build_training_dataset(storage, years: list[str] | None = None) -> pd.DataFrame:
    """GCSの過去レースデータから学習データセットを構築する。"""
    from scraper.storage import HybridStorage

    if years is None:
        years = ["2023", "2024", "2025", "2026"]

    result_keys = storage.list_keys("race_result")
    logger.info("race_result キー数: %d", len(result_keys))

    all_horse_histories: dict[str, list[dict]] = {}
    rows: list[dict] = []
    processed = 0
    skipped = 0

    for key in sorted(result_keys):
        year = key[:4]
        if year not in years:
            continue

        result_data = storage.load("race_result", key)
        if not result_data or not result_data.get("entries"):
            skipped += 1
            continue

        race_info = {
            "race_id": result_data.get("race_id", key),
            "field_size": result_data.get("field_size", len(result_data["entries"])),
            "distance": result_data.get("distance", 0),
            "surface": result_data.get("surface", ""),
            "track_condition": result_data.get("track_condition", ""),
            "venue": result_data.get("venue", ""),
            "date": result_data.get("date", ""),
        }

        entries = result_data["entries"]
        fs = race_info["field_size"]
        if fs < 4:
            skipped += 1
            continue

        horse_profiles_for_race: dict[int, dict] = {}
        for e in entries:
            hid = e.get("horse_id", "")
            hn = e.get("horse_number", 0)
            history = all_horse_histories.get(hid, [])
            horse_profiles_for_race[hn] = build_horse_profile(history)

        field_profiles = list(horse_profiles_for_race.values())

        for e in entries:
            hn = e.get("horse_number", 0)
            hid = e.get("horse_id", "")
            profile = horse_profiles_for_race.get(hn)
            if not profile or profile["n_races"] < 2:
                continue

            left_hn = hn - 1
            right_hn = hn + 1
            neighbors = {
                "left": horse_profiles_for_race.get(left_hn, {}),
                "right": horse_profiles_for_race.get(right_hn, {}),
            }

            row = build_training_row(e, race_info, profile, neighbors, field_profiles)
            if row is not None:
                rows.append(row)

        for e in entries:
            hid = e.get("horse_id", "")
            if hid:
                if hid not in all_horse_histories:
                    all_horse_histories[hid] = []
                all_horse_histories[hid].insert(0, {
                    "passing_order": e.get("passing_order", ""),
                    "field_size": fs,
                    "finish_position": e.get("finish_position", 0),
                    "bracket_number": e.get("bracket_number", 0),
                    "horse_number": e.get("horse_number", 0),
                    "distance": race_info["distance"],
                    "surface": race_info["surface"],
                    "track_condition": race_info["track_condition"],
                })

        processed += 1
        if processed % 500 == 0:
            logger.info("処理済み: %d レース, 行数: %d", processed, len(rows))

    logger.info(
        "データセット構築完了: %d レース処理, %d スキップ, %d 行",
        processed, skipped, len(rows),
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ── モデル学習 ──

FEATURE_COLUMNS = [
    "typical_norm_pos",
    "style_code",
    "pos_std",
    "n_past_races",
    "horse_number_norm",
    "bracket_number",
    "is_even_gate",
    "gate_zone",
    "left_neighbor_tendency",
    "right_neighbor_tendency",
    "left_neighbor_std",
    "right_neighbor_std",
    "neighbor_slow_count",
    "neighbor_fast_count",
    "neighbor_avg_tendency",
    "neighbor_space_score",
    "field_size",
    "front_runner_count",
    "closer_count",
    "same_style_count",
    "front_runner_ratio",
    "pace_pressure",
    "surface_code",
    "distance",
    "distance_category",
    "track_condition_code",
]


class TrackingDifficultyTrainer:
    """追走難度モデルの学習と MLflow 登録。"""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.mlflow_uri = mlflow_tracking_uri

    def train(
        self,
        df: pd.DataFrame | None = None,
        storage=None,
        years: list[str] | None = None,
        test_ratio: float = 0.2,
    ) -> dict[str, Any]:
        t0 = time.time()

        if df is None:
            if storage is None:
                from scraper.storage import HybridStorage
                storage = HybridStorage()
            df = build_training_dataset(storage, years=years)

        if df.empty:
            return {"error": "学習データなし"}

        logger.info("学習データ: %d行", len(df))

        X = df[FEATURE_COLUMNS].fillna(0).astype(float)
        y = df["position_deviation"].values

        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM 未インストール")
            return {"error": "LightGBM required"}

        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.5,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )

        preds = model.predict(X_test)
        mae = float(np.mean(np.abs(preds - y_test)))
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        corr = float(np.corrcoef(preds, y_test)[0, 1]) if len(y_test) > 2 else 0

        importance = dict(
            zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain").tolist())
        )

        metrics = {
            "mae": round(mae, 5),
            "rmse": round(rmse, 5),
            "correlation": round(corr, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(FEATURE_COLUMNS),
            "best_iteration": model.best_iteration,
            "training_time_sec": round(time.time() - t0, 1),
            "top_features": dict(sorted(importance.items(), key=lambda x: -x[1])[:10]),
            "feature_names": FEATURE_COLUMNS,
            "target_mean": round(float(y.mean()), 5),
            "target_std": round(float(y.std()), 5),
        }

        self._log_mlflow(model, metrics, lgb_params)

        return metrics

    def _log_mlflow(self, model, metrics: dict, lgb_params: dict):
        try:
            import mlflow
            import mlflow.lightgbm

            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)

            run_name = f"td_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_type", "tracking_difficulty")
                mlflow.set_tag("algorithm", "lightgbm_regression")

                mlflow.log_params({k: str(v) for k, v in lgb_params.items()})
                mlflow.log_params({
                    "n_features": metrics["n_features"],
                    "n_train": metrics["n_train"],
                    "n_test": metrics["n_test"],
                })

                log_metrics = {}
                for k in ("mae", "rmse", "correlation", "best_iteration",
                          "training_time_sec"):
                    if k in metrics and isinstance(metrics[k], (int, float)):
                        log_metrics[k] = metrics[k]
                mlflow.log_metrics(log_metrics)

                mlflow.lightgbm.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                )

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as f:
                    json.dump({
                        "feature_names": FEATURE_COLUMNS,
                        "importance": metrics.get("top_features", {}),
                        "target_stats": {
                            "mean": metrics["target_mean"],
                            "std": metrics["target_std"],
                        },
                    }, f, ensure_ascii=False, indent=2)
                    tmp_path = f.name
                mlflow.log_artifact(tmp_path, "feature_info")
                os.unlink(tmp_path)

                logger.info(
                    "MLflow 記録完了: run_id=%s  MAE=%.5f  RMSE=%.5f  corr=%.4f",
                    run.info.run_id,
                    metrics["mae"], metrics["rmse"], metrics["correlation"],
                )

        except Exception as e:
            logger.warning("MLflow 記録失敗 (学習結果はローカルに保持): %s", e)


# ── 推論 ──

_cached_model = None
_cached_model_time = 0


def load_model():
    """MLflowからモデルをロード（キャッシュ付き）。"""
    global _cached_model, _cached_model_time

    if _cached_model is not None and (time.time() - _cached_model_time) < 3600:
        return _cached_model

    try:
        import mlflow
        import mlflow.lightgbm
        from mlflow.tracking import MlflowClient

        from utils.mlflow_client import init_mlflow
        init_mlflow()

        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if versions:
            v = versions[0]
            model_uri = f"models:/{MODEL_NAME}/{v.version}"
            model = mlflow.lightgbm.load_model(model_uri)
            _cached_model = model
            _cached_model_time = time.time()
            logger.info("追走難度モデルロード: version=%s", v.version)
            return model
    except Exception as e:
        logger.warning("MLflow モデルロード失敗: %s", e)

    local_path = Path(__file__).parent.parent / "models" / "tracking_difficulty.txt"
    if local_path.exists():
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(local_path))
        _cached_model = model
        _cached_model_time = time.time()
        return model

    return None


def predict_tracking_difficulty(
    race_data: dict,
    horse_histories: dict[str, list[dict]] | None = None,
    storage=None,
) -> list[dict]:
    """
    レースデータに対して各馬の追走難度を推定する。

    Args:
        race_data: race_shutuba + race_result 等のバンドルデータ
        horse_histories: horse_id -> race_history のマッピング
        storage: HybridStorage (horse_histories が None の場合に使用)

    Returns:
        馬番順の追走難度スコアリスト
    """
    shutuba = race_data.get("race_shutuba") or race_data.get("race_card") or {}
    result = race_data.get("race_result") or {}
    entries = shutuba.get("entries") or result.get("entries") or []
    if not entries:
        return []

    race_info = {
        "race_id": shutuba.get("race_id") or result.get("race_id", ""),
        "field_size": len(entries),
        "distance": shutuba.get("distance") or result.get("distance", 1600),
        "surface": shutuba.get("surface") or result.get("surface", "芝"),
        "track_condition": (
            shutuba.get("track_condition")
            or result.get("track_condition", "良")
        ),
    }

    if horse_histories is None:
        horse_histories = {}
        horses = race_data.get("horses") or {}
        for hid, hdata in horses.items():
            if isinstance(hdata, dict):
                horse_histories[hid] = hdata.get("race_history", [])

    if not horse_histories and storage:
        for e in entries:
            hid = e.get("horse_id", "")
            if hid and hid not in horse_histories:
                hr = storage.load("horse_result", hid)
                if hr and "race_history" in hr:
                    horse_histories[hid] = hr["race_history"]

    profiles: dict[int, dict] = {}
    for e in entries:
        hid = e.get("horse_id", "")
        hn = e.get("horse_number", 0)
        history = horse_histories.get(hid, [])
        profiles[hn] = build_horse_profile(history)

    field_profiles = list(profiles.values())

    model = load_model()
    results = []

    for e in entries:
        hn = e.get("horse_number", 0)
        hid = e.get("horse_id", "")
        bn = e.get("bracket_number", 0)
        profile = profiles.get(hn, build_horse_profile([]))

        left_hn = hn - 1
        right_hn = hn + 1
        neighbors = {
            "left": profiles.get(left_hn, {}),
            "right": profiles.get(right_hn, {}),
        }

        row = _build_inference_row(e, race_info, profile, neighbors, field_profiles)

        if model is not None:
            X = pd.DataFrame([{c: row.get(c, 0) for c in FEATURE_COLUMNS}])
            pred = float(model.predict(X)[0])
        else:
            pred = _heuristic_score(row)

        ease_pct = _deviation_to_ease(pred, profile)

        results.append({
            "horse_number": hn,
            "horse_id": hid,
            "horse_name": e.get("horse_name", ""),
            "bracket_number": bn,
            "tracking_difficulty": {
                "score": round(pred, 4),
                "ease_pct": round(ease_pct, 1),
                "label": _ease_label(ease_pct),
            },
            "profile": {
                "typical_position": profile["typical_norm_pos"],
                "style": profile["style"],
                "stability": round(1.0 - profile["pos_std"], 3),
                "n_races": profile["n_races"],
            },
            "gate_factors": {
                "horse_number": hn,
                "bracket_number": bn,
                "is_even_gate": bool(hn % 2 == 0),
                "gate_zone": row.get("gate_zone", 0),
            },
            "neighbor_factors": {
                "left_tendency": round(
                    neighbors.get("left", {}).get("start_tendency", 0.5), 3
                ),
                "right_tendency": round(
                    neighbors.get("right", {}).get("start_tendency", 0.5), 3
                ),
                "space_score": round(row.get("neighbor_space_score", 0), 3),
                "slow_neighbors": row.get("neighbor_slow_count", 0),
                "fast_neighbors": row.get("neighbor_fast_count", 0),
            },
            "field_factors": {
                "front_runner_count": row.get("front_runner_count", 0),
                "closer_count": row.get("closer_count", 0),
                "same_style_count": row.get("same_style_count", 0),
                "pace_pressure": round(row.get("pace_pressure", 0), 3),
            },
        })

    results.sort(key=lambda x: -x["tracking_difficulty"]["ease_pct"])
    return results


def _build_inference_row(
    entry: dict,
    race_info: dict,
    profile: dict,
    neighbors: dict,
    field_profiles: list[dict],
) -> dict:
    hn = entry.get("horse_number", 0)
    bn = entry.get("bracket_number", 0)
    fs = race_info["field_size"]

    left_profile = neighbors.get("left", {})
    right_profile = neighbors.get("right", {})
    lt = left_profile.get("start_tendency", 0.5)
    rt = right_profile.get("start_tendency", 0.5)

    # 速い隣馬のカウント（枠位置に応じて）
    neighbor_fast = 0
    if hn == 1:
        # 大内枠：右隣が速ければOK
        if rt <= 0.2:
            neighbor_fast = 1
    elif hn == fs:
        # 大外枠：左隣が速ければOK
        if lt <= 0.2:
            neighbor_fast = 1
    else:
        # 中枠：両サイドともに速い場合のみ
        if lt <= 0.2 and rt <= 0.2:
            neighbor_fast = 1

    front_runners = sum(
        1 for p in field_profiles if p.get("style") in ("逃げ", "先行")
    )
    closers = sum(
        1 for p in field_profiles if p.get("style") in ("差し", "追込")
    )
    same_style = sum(
        1 for p in field_profiles if p.get("style") == profile.get("style")
    )

    distance = race_info.get("distance", 1600)
    surface = race_info.get("surface", "芝")
    cond = race_info.get("track_condition", "良")

    return {
        "typical_norm_pos": profile.get("typical_norm_pos", 0.5),
        "style_code": list(RUNNING_STYLE_THRESHOLDS.keys()).index(
            profile.get("style", "差し")
        ),
        "pos_std": profile.get("pos_std", 0.2),
        "n_past_races": profile.get("n_races", 0),
        "horse_number_norm": round(hn / fs, 4) if fs > 0 else 0.5,
        "bracket_number": bn,
        "is_even_gate": int(hn % 2 == 0),
        "gate_zone": (
            0 if hn <= fs * 0.25 else
            1 if hn <= fs * 0.5 else
            2 if hn <= fs * 0.75 else 3
        ),
        "left_neighbor_tendency": round(lt, 4),
        "right_neighbor_tendency": round(rt, 4),
        "left_neighbor_std": round(
            left_profile.get("pos_std", 0.2), 4
        ),
        "right_neighbor_std": round(
            right_profile.get("pos_std", 0.2), 4
        ),
        "neighbor_slow_count": (
            (1 if lt > 0.5 else 0) + (1 if rt > 0.5 else 0)
        ),
        "neighbor_fast_count": neighbor_fast,
        "neighbor_avg_tendency": round((lt + rt) / 2, 4),
        "neighbor_space_score": round(
            (lt + rt) / 2 - profile.get("typical_norm_pos", 0.5) + neighbor_fast * 0.1 - ((1 if lt > 0.5 else 0) + (1 if rt > 0.5 else 0)) * 0.05, 4
        ),
        "field_size": fs,
        "front_runner_count": front_runners,
        "closer_count": closers,
        "same_style_count": same_style,
        "front_runner_ratio": round(front_runners / max(fs, 1), 4),
        "pace_pressure": round(front_runners / max(closers, 1), 4),
        "surface_code": 0 if surface == "芝" else 1,
        "distance": distance,
        "distance_category": (
            0 if distance <= 1400 else
            1 if distance <= 1800 else
            2 if distance <= 2200 else 3
        ),
        "track_condition_code": (
            0 if cond == "良" else
            1 if cond == "稍" else
            2 if cond == "重" else 3
        ),
    }


def _heuristic_score(row: dict) -> float:
    """モデルがない場合のヒューリスティックスコア。"""
    score = 0.0
    score += 0.02 * row.get("is_even_gate", 0)
    score += 0.03 * row.get("neighbor_slow_count", 0)
    score += 0.04 * row.get("neighbor_fast_count", 0)  # 速い隣馬の恩恵
    score += 0.05 * row.get("neighbor_space_score", 0)
    score -= 0.02 * row.get("front_runner_ratio", 0)
    score -= 0.01 * (row.get("same_style_count", 0) - 1)
    score += 0.01 * (row.get("n_past_races", 0) / 10)
    return score


def _deviation_to_ease(deviation: float, profile: dict) -> float:
    """deviation値をイージースコア（0-100）に変換。"""
    base = 50.0
    scaled = deviation * 200
    stability_bonus = (1.0 - profile.get("pos_std", 0.2)) * 10
    experience_bonus = min(profile.get("n_races", 0), 10) * 0.5
    ease = base + scaled + stability_bonus + experience_bonus
    return max(0, min(100, ease))


def _ease_label(ease_pct: float) -> str:
    if ease_pct >= 75:
        return "非常に楽"
    elif ease_pct >= 60:
        return "楽"
    elif ease_pct >= 45:
        return "普通"
    elif ease_pct >= 30:
        return "やや困難"
    else:
        return "困難"


# ── ペース予想 ──

def _estimate_early_lap_times(
    distance: int,
    pace_type: str,
    pace_index: float,
    track_cond: str,
    surface: str,
) -> dict:
    """
    前半ラップタイム（1F, 3F）を推定する。

    基準タイム（良馬場・芝・平均ペース）:
    - 1F (200m): 12.0秒
    - 3F (600m): 35.0秒

    調整要因:
    - ペースタイプ（ハイ/ミドル/スロー）
    - 距離（短距離/中距離/長距離）
    - 馬場状態（良/稍/重）
    - 芝/ダート
    """
    # 基準タイム（秒）
    base_1f = 12.0
    base_3f = 35.0

    # ペースによる調整（ハイペースは速く、スローは遅く）
    pace_adj = 0.0
    if pace_type == "ハイ":
        pace_adj = -0.5 - (pace_index - 55) * 0.02  # -0.5 ~ -1.4秒程度
    elif pace_type == "スロー":
        pace_adj = 0.5 + (40 - pace_index) * 0.02   # +0.5 ~ +1.3秒程度

    # 距離による調整（短距離は速く、長距離は抑える）
    dist_adj = 0.0
    if distance <= 1400:
        dist_adj = -0.3  # 短距離は飛ばす
    elif distance >= 2200:
        dist_adj = 0.4   # 長距離は抑える

    # 馬場状態による調整
    track_adj = 0.0
    if track_cond in ("重", "不良"):
        track_adj = 0.8
    elif track_cond == "稍":
        track_adj = 0.3

    # 芝/ダートによる調整
    surface_adj = 0.0
    if surface == "ダ" or surface == "ダート":
        surface_adj = 0.5  # ダートは基本的に遅い

    # 合計調整
    total_adj_1f = pace_adj + dist_adj + track_adj + surface_adj
    total_adj_3f = total_adj_1f * 3

    first_1f = base_1f + total_adj_1f
    first_3f = base_3f + total_adj_3f

    return {
        "first_1f": round(first_1f, 1),
        "first_3f": round(first_3f, 1),
    }


def predict_race_pace(
    entries: list[dict],
    horse_profiles: dict[int, dict],
    race_info: dict,
) -> dict:
    """
    レース全体のペース予想を行う（前半ラップタイム含む）。

    Returns:
        {
            "pace_type": "スロー" | "ミドル" | "ハイ",
            "pace_index": float (0-100),
            "front_runner_count": int,
            "early_pressure": float,
            "predicted_pattern": str,
            "pace_factors": dict,
            "lap_times": {
                "first_1f": float,  # 最初の1F (200m) タイム
                "first_3f": float,  # 最初の3F (600m) タイム
            }
        }
    """
    field_size = len(entries)
    if field_size == 0:
        return {"pace_type": "不明", "pace_index": 50, "front_runner_count": 0}

    # 脚質分布を集計
    front_runners = []  # 逃げ
    stalkers = []       # 先行
    midpack = []        # 差し
    closers = []        # 追込

    for e in entries:
        hn = e.get("horse_number", 0)
        profile = horse_profiles.get(hn, {})
        style = profile.get("style", "差し")
        typical_pos = profile.get("typical_norm_pos", 0.5)

        if style == "逃げ":
            front_runners.append({"horse_number": hn, "pos": typical_pos})
        elif style == "先行":
            stalkers.append({"horse_number": hn, "pos": typical_pos})
        elif style == "差し":
            midpack.append({"horse_number": hn, "pos": typical_pos})
        else:
            closers.append({"horse_number": hn, "pos": typical_pos})

    # ペース指標の計算
    front_count = len(front_runners)
    stalker_count = len(stalkers)
    early_speed_horses = front_count + stalker_count

    # ペース圧力 = 前に行きたい馬の割合
    early_pressure = early_speed_horses / field_size if field_size > 0 else 0.3

    # 距離補正（短距離はハイペースになりやすい）
    distance = race_info.get("distance", 1600)
    distance_factor = 1.0
    if distance <= 1400:
        distance_factor = 1.2  # 短距離はペースが速くなる
    elif distance >= 2200:
        distance_factor = 0.85  # 長距離はペースが落ち着く

    # 馬場状態補正（重馬場はペースが落ちる）
    track_cond = race_info.get("track_condition", "良")
    track_factor = 1.0
    if track_cond in ("重", "不良"):
        track_factor = 0.8
    elif track_cond == "稍":
        track_factor = 0.9

    # ペース指数の計算（0-100）
    pace_index = (
        early_pressure * 100 * distance_factor * track_factor
    )
    pace_index = max(0, min(100, pace_index))

    # ペースタイプの判定
    if pace_index >= 55:
        pace_type = "ハイ"
        pattern = "前崩れ注意"
    elif pace_index >= 40:
        pace_type = "ミドル"
        pattern = "平均的な流れ"
    else:
        pace_type = "スロー"
        pattern = "後傾ラップ"

    # 特殊パターン検出
    if front_count == 0:
        pattern = "逃げ不在 → スローペース濃厚"
        pace_type = "スロー"
    elif front_count >= 3:
        pattern = "逃げ多数 → 先行争い激化"
        pace_type = "ハイ"
    elif front_count == 1 and stalker_count <= 2:
        pattern = "単騎逃げ → マイペース"
        pace_type = "スロー"

    # 前半ラップタイム推定
    lap_times = _estimate_early_lap_times(
        distance, pace_type, pace_index, track_cond, race_info.get("surface", "芝")
    )

    return {
        "pace_type": pace_type,
        "pace_index": round(pace_index, 1),
        "front_runner_count": front_count,
        "stalker_count": stalker_count,
        "early_pressure": round(early_pressure, 3),
        "predicted_pattern": pattern,
        "pace_factors": {
            "distance_factor": round(distance_factor, 2),
            "track_factor": round(track_factor, 2),
            "early_speed_ratio": round(early_pressure, 3),
        },
        "distribution": {
            "逃げ": front_count,
            "先行": stalker_count,
            "差し": len(midpack),
            "追込": len(closers),
        },
        "lap_times": lap_times,
    }


# ── 位置取り予測 ──

def predict_position_flow(
    entries: list[dict],
    horse_profiles: dict[int, dict],
    tracking_results: list[dict],
    pace_prediction: dict,
) -> list[dict]:
    """
    各馬の位置取りの流れ（序盤→中盤→終盤）を予測する。

    Returns:
        各馬の位置予測リスト [
            {
                "horse_number": int,
                "horse_name": str,
                "positions": {
                    "early": {"position": int, "position_norm": float},
                    "mid": {"position": int, "position_norm": float},
                    "late": {"position": int, "position_norm": float},
                },
                "flow_pattern": str,
                "stamina_concern": bool
            },
            ...
        ]
    """
    field_size = len(entries)
    pace_type = pace_prediction.get("pace_type", "ミドル")
    pace_index = pace_prediction.get("pace_index", 50)

    results = []

    for e in entries:
        hn = e.get("horse_number", 0)
        profile = horse_profiles.get(hn, {})
        style = profile.get("style", "差し")
        typical_pos = profile.get("typical_norm_pos", 0.5)
        pos_std = profile.get("pos_std", 0.2)

        # 追走難度から位置取りの楽さを取得
        td_data = next(
            (r for r in tracking_results if r["horse_number"] == hn),
            {}
        )
        ease_pct = td_data.get("tracking_difficulty", {}).get("ease_pct", 50)

        # 序盤位置の予測
        early_base = typical_pos

        # 追走難度による補正（楽なら前に行きやすい）
        ease_adjustment = (ease_pct - 50) / 200  # -0.25 ~ +0.25
        early_pos_norm = early_base + ease_adjustment

        # ペースによる補正
        if pace_type == "ハイ" and style in ("逃げ", "先行"):
            # ハイペースでは前に行きすぎると苦しい
            early_pos_norm -= 0.05
        elif pace_type == "スロー" and style in ("差し", "追込"):
            # スローペースでは前が遅いので前に付きやすい
            early_pos_norm -= 0.1

        early_pos_norm = max(0, min(1, early_pos_norm))
        early_pos = max(1, min(field_size, int(early_pos_norm * field_size) + 1))

        # 中盤位置の予測
        mid_pos_norm = early_pos_norm

        # ペース影響（ハイペースは前が下がる、スローは差が詰まる）
        if pace_type == "ハイ":
            if style in ("逃げ", "先行"):
                mid_pos_norm += 0.1  # 前が苦しくなる
        elif pace_type == "スロー":
            if style in ("差し", "追込"):
                mid_pos_norm -= 0.05  # 差が詰まる

        mid_pos_norm = max(0, min(1, mid_pos_norm))
        mid_pos = max(1, min(field_size, int(mid_pos_norm * field_size) + 1))

        # 終盤位置の予測
        late_pos_norm = mid_pos_norm

        # 脚質による終盤の動き
        if style == "逃げ":
            if pace_type == "ハイ":
                late_pos_norm += 0.2  # ハイペース逃げは苦しい
            else:
                late_pos_norm += 0.05  # 通常でも少し下がる
        elif style == "先行":
            if pace_type == "ハイ":
                late_pos_norm += 0.1
            else:
                late_pos_norm -= 0.02  # 好位追走なら少し伸びる
        elif style in ("差し", "追込"):
            if pace_type == "ハイ":
                late_pos_norm -= 0.15  # 前が止まるので差せる
            elif pace_type == "スロー":
                late_pos_norm += 0.05  # スローは差せない
            else:
                late_pos_norm -= 0.1  # ミドルペースなら普通に差せる

        late_pos_norm = max(0, min(1, late_pos_norm))
        late_pos = max(1, min(field_size, int(late_pos_norm * field_size) + 1))

        # 流れパターンの判定
        if late_pos < early_pos - 2:
            flow_pattern = "追い込み"
        elif late_pos < early_pos:
            flow_pattern = "差し"
        elif late_pos > early_pos + 2:
            flow_pattern = "後退"
        elif late_pos > early_pos:
            flow_pattern = "微後退"
        else:
            flow_pattern = "粘り込み"

        # スタミナ懸念の判定
        stamina_concern = (
            (pace_type == "ハイ" and style in ("逃げ", "先行")) or
            (pace_type == "スロー" and style in ("差し", "追込"))
        )

        results.append({
            "horse_number": hn,
            "horse_name": e.get("horse_name", ""),
            "style": style,
            "positions": {
                "early": {
                    "position": early_pos,
                    "position_norm": round(early_pos_norm, 3),
                },
                "mid": {
                    "position": mid_pos,
                    "position_norm": round(mid_pos_norm, 3),
                },
                "late": {
                    "position": late_pos,
                    "position_norm": round(late_pos_norm, 3),
                },
            },
            "flow_pattern": flow_pattern,
            "stamina_concern": stamina_concern,
            "confidence": round(max(0, 1.0 - pos_std * 2), 2),
        })

    # 序盤位置でソート
    results.sort(key=lambda x: x["positions"]["early"]["position"])

    return results
