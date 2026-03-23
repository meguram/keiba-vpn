"""
オッズ予測モジュール

レース予測で使用するオッズを、確定前の段階で推定する。

2つのアプローチを組み合わせる:
  1. 過去データベースモデル: 馬の能力指標から単勝・複勝オッズを予測
  2. オッズ推移トラッキング: 定期取得したオッズの推移から確定値を外挿

推移データがある場合はそちらを優先し、なければモデル予測を使用する。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("OddsPredictor")

ODDS_MODEL_PATH = Path("models/odds_predictor.json")
ODDS_HISTORY_DIR = Path("data/local/odds_history")


@dataclass
class PredictedOdds:
    horse_number: int
    predicted_win_odds: float
    predicted_place_odds_min: float
    predicted_place_odds_max: float
    confidence: float
    source: str  # "model", "trajectory", "blend"


class OddsPredictor:
    """
    過去データからオッズを予測するモデル。

    学習対象: 確定済みの単勝オッズ (race_result.entries[].odds)
    特徴量: 馬の能力を示す非市場指標のみ
    """

    FEATURE_COLS = [
        "speed_max", "speed_avg",
        "avg_finish_5", "min_finish_5", "top3_count_5", "win_count_5",
        "avg_last_3f_5", "min_last_3f_5",
        "career_runs", "career_win_rate", "career_top3_rate",
        "same_surface_win_rate", "same_dist_win_rate",
        "training_impression_score",
        "barometer_total",
        "age", "weight",
        "days_since_last",
        "field_size",
    ]

    def __init__(self):
        self._model = None
        self._params: dict[str, Any] = {}

    def train(self, storage) -> dict:
        """
        過去の race_result データからオッズ予測モデルを学習する。

        Returns: {n_samples, metrics, feature_importance}
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm が未インストール")
            return {"error": "lightgbm not installed"}

        from pipeline.feature_builder import build_race_features

        rows = []
        years = storage.list_years("race_result")
        max_per_year = 600
        categories_to_load = [
            ("race_shutuba", "race_card"),
            ("race_index", "speed_index"),
            ("race_barometer", "barometer"),
        ]

        for year in years:
            keys = storage.list_keys("race_result", year)
            for key in keys[:max_per_year]:
                race_id = key.replace(".json", "")
                try:
                    result = storage.load("race_result", race_id)
                    if not result or "entries" not in result:
                        continue

                    odds_map = {}
                    for entry in result["entries"]:
                        hn = entry.get("horse_number", 0)
                        odds_val = entry.get("odds", 0)
                        if hn and odds_val and odds_val > 0:
                            odds_map[hn] = float(odds_val)

                    if not odds_map:
                        continue

                    race_data = {"race_result": result}
                    for storage_cat, data_key in categories_to_load:
                        d = storage.load(storage_cat, race_id)
                        if d:
                            race_data[data_key] = d

                    features_df = build_race_features(race_data)
                    if features_df.empty:
                        continue

                    for _, row in features_df.iterrows():
                        hn = int(row.get("horse_number", 0))
                        if hn in odds_map:
                            r = {c: row.get(c, 0) for c in self.FEATURE_COLS if c in features_df.columns}
                            r["target_log_odds"] = math.log(max(odds_map[hn], 1.0))
                            r["horse_number"] = hn
                            r["race_id"] = race_id
                            rows.append(r)
                except Exception as e:
                    logger.debug("skip %s: %s", key, e)
                    continue

        if len(rows) < 100:
            return {"error": f"学習データ不足: {len(rows)}件"}

        df = pd.DataFrame(rows)
        feature_cols = [c for c in self.FEATURE_COLS if c in df.columns]

        X = df[feature_cols].fillna(0).values
        y = df["target_log_odds"].values

        n_train = int(len(df) * 0.8)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )

        preds_test = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((preds_test - y_test) ** 2)))
        mae = float(np.mean(np.abs(preds_test - y_test)))

        # log(odds) → odds の相対誤差
        pred_odds = np.exp(preds_test)
        actual_odds = np.exp(y_test)
        mape = float(np.mean(np.abs(pred_odds - actual_odds) / actual_odds)) * 100

        importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain").tolist()))

        model_data = {
            "feature_cols": feature_cols,
            "model_txt": model.model_to_string(),
            "metrics": {"rmse": rmse, "mae": mae, "mape": mape},
            "n_train": n_train,
            "n_test": len(y_test),
            "trained_at": time.time(),
        }
        ODDS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ODDS_MODEL_PATH, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        self._model = model
        logger.info("オッズ予測モデル学習完了: RMSE=%.3f, MAE=%.3f, MAPE=%.1f%%", rmse, mae, mape)

        return {
            "n_samples": len(df),
            "n_train": n_train,
            "n_test": len(y_test),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 1),
            "feature_importance": {k: round(v, 1) for k, v in sorted(importance.items(), key=lambda x: -x[1])[:10]},
        }

    def load_model(self) -> bool:
        if self._model is not None:
            return True
        if not ODDS_MODEL_PATH.exists():
            return False
        try:
            import lightgbm as lgb
            with open(ODDS_MODEL_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self._model = lgb.Booster(model_str=data["model_txt"])
            self._params = data
            logger.info("オッズ予測モデルをロード (MAPE=%.1f%%)", data.get("metrics", {}).get("mape", 0))
            return True
        except Exception as e:
            logger.warning("オッズ予測モデルロード失敗: %s", e)
            return False

    def predict(self, features_df: pd.DataFrame) -> list[PredictedOdds]:
        """
        特徴量テーブルから各馬の予測オッズを算出する。
        """
        if not self.load_model():
            return self._heuristic_odds(features_df)

        feature_cols = self._params.get("feature_cols", self.FEATURE_COLS)
        available = [c for c in feature_cols if c in features_df.columns]
        if not available:
            return self._heuristic_odds(features_df)

        X = features_df[available].fillna(0).values
        log_odds_pred = self._model.predict(X)

        results = []
        for i, (_, row) in enumerate(features_df.iterrows()):
            win_odds = max(float(np.exp(log_odds_pred[i])), 1.0)
            place_min = max(win_odds * 0.25, 1.0)
            place_max = max(win_odds * 0.55, place_min + 0.1)

            results.append(PredictedOdds(
                horse_number=int(row.get("horse_number", i + 1)),
                predicted_win_odds=round(win_odds, 1),
                predicted_place_odds_min=round(place_min, 1),
                predicted_place_odds_max=round(place_max, 1),
                confidence=0.7,
                source="model",
            ))

        return results

    @staticmethod
    def _heuristic_odds(features_df: pd.DataFrame) -> list[PredictedOdds]:
        """モデルがない場合のヒューリスティックオッズ推定。"""
        n = len(features_df)
        scores = pd.Series(0.0, index=features_df.index)

        if "speed_max" in features_df.columns:
            sp = features_df["speed_max"].fillna(0)
            if sp.max() > 0:
                scores += sp / sp.max() * 30

        if "career_top3_rate" in features_df.columns:
            scores += features_df["career_top3_rate"].fillna(0) * 25

        if "avg_finish_5" in features_df.columns:
            af = features_df["avg_finish_5"].fillna(10)
            scores += (1 - af / 18).clip(0, 1) * 20

        if "avg_last_3f_5" in features_df.columns:
            l3 = features_df["avg_last_3f_5"].fillna(37)
            scores += (1 - (l3 - 33) / 7).clip(0, 1) * 15

        if "training_impression_score" in features_df.columns:
            scores += features_df["training_impression_score"].fillna(0) * 10

        total = scores.sum()
        if total <= 0:
            total = 1

        results = []
        for i, (_, row) in enumerate(features_df.iterrows()):
            share = scores.iloc[i] / total
            if share <= 0:
                share = 0.01

            win_odds = max(0.8 / share, 1.0)
            place_min = max(win_odds * 0.25, 1.0)
            place_max = max(win_odds * 0.55, place_min + 0.1)

            results.append(PredictedOdds(
                horse_number=int(row.get("horse_number", i + 1)),
                predicted_win_odds=round(win_odds, 1),
                predicted_place_odds_min=round(place_min, 1),
                predicted_place_odds_max=round(place_max, 1),
                confidence=0.3,
                source="heuristic",
            ))

        return results


class OddsTrajectoryTracker:
    """
    オッズの推移を定期取得して保存・外挿するトラッカー。

    各race_idについて、取得時刻ごとのオッズスナップショットを保持し、
    確定時のオッズを線形外挿で予測する。
    """

    def __init__(self):
        ODDS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    def record_snapshot(self, race_id: str, odds_data: dict):
        """現在のオッズスナップショットを記録する。"""
        history_path = ODDS_HISTORY_DIR / f"{race_id}.json"

        history = []
        if history_path.exists():
            try:
                with open(history_path, encoding="utf-8") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, OSError):
                history = []

        snapshot = {
            "timestamp": time.time(),
            "entries": odds_data.get("entries", []),
        }
        history.append(snapshot)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)

        logger.info("オッズスナップショット保存: %s (計%d回)", race_id, len(history))

    def predict_final(self, race_id: str) -> list[PredictedOdds] | None:
        """
        オッズ推移から確定オッズを外挿予測する。
        3回以上のスナップショットが必要。
        """
        history_path = ODDS_HISTORY_DIR / f"{race_id}.json"
        if not history_path.exists():
            return None

        try:
            with open(history_path, encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if len(history) < 2:
            if history:
                return self._latest_as_prediction(history[-1])
            return None

        horse_series: dict[int, list[tuple[float, float, float, float]]] = {}
        for snap in history:
            ts = snap["timestamp"]
            for entry in snap.get("entries", []):
                hn = entry.get("horse_number", 0)
                if not hn:
                    continue
                wo = entry.get("win_odds", 0) or 0
                pm = entry.get("place_odds_min", 0) or 0
                px = entry.get("place_odds_max", 0) or 0
                if wo > 0:
                    horse_series.setdefault(hn, []).append((ts, wo, pm, px))

        if not horse_series:
            return None

        results = []
        for hn, series in sorted(horse_series.items()):
            if len(series) >= 3:
                pred = self._extrapolate(series)
            else:
                pred = series[-1][1:]

            results.append(PredictedOdds(
                horse_number=hn,
                predicted_win_odds=round(max(pred[0], 1.0), 1),
                predicted_place_odds_min=round(max(pred[1], 1.0), 1),
                predicted_place_odds_max=round(max(pred[2], pred[1] + 0.1), 1),
                confidence=min(0.5 + len(series) * 0.1, 0.9),
                source="trajectory",
            ))

        return results if results else None

    @staticmethod
    def _extrapolate(series: list[tuple[float, float, float, float]]) -> tuple[float, float, float]:
        """
        線形回帰で最終オッズを外挿する。
        オッズは発売終了に近づくほど収束するため、
        最新3点に重みを置いた加重線形回帰を使用。
        """
        n = len(series)
        ts = np.array([s[0] for s in series])
        win = np.array([s[1] for s in series])
        pmin = np.array([s[2] for s in series])
        pmax = np.array([s[3] for s in series])

        weights = np.array([1.0 + i * 0.5 for i in range(n)])

        t_norm = ts - ts[0]
        if t_norm[-1] == 0:
            return (win[-1], pmin[-1], pmax[-1])

        # 10%先を外挿
        t_target = t_norm[-1] * 1.1

        def weighted_predict(y):
            wt = weights[-min(n, 5):]
            tn = t_norm[-min(n, 5):]
            yy = y[-min(n, 5):]
            W = np.diag(wt)
            A = np.column_stack([tn, np.ones_like(tn)])
            try:
                beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ yy)
                return float(beta[0] * t_target + beta[1])
            except np.linalg.LinAlgError:
                return float(yy[-1])

        return (
            weighted_predict(win),
            weighted_predict(pmin),
            weighted_predict(pmax),
        )

    @staticmethod
    def _latest_as_prediction(snapshot: dict) -> list[PredictedOdds]:
        results = []
        for entry in snapshot.get("entries", []):
            hn = entry.get("horse_number", 0)
            if not hn:
                continue
            results.append(PredictedOdds(
                horse_number=hn,
                predicted_win_odds=round(entry.get("win_odds", 0) or 1.0, 1),
                predicted_place_odds_min=round(entry.get("place_odds_min", 0) or 1.0, 1),
                predicted_place_odds_max=round(entry.get("place_odds_max", 0) or 1.0, 1),
                confidence=0.5,
                source="latest_snapshot",
            ))
        return results


def get_predicted_odds(
    features_df: pd.DataFrame,
    race_id: str,
    live_odds: dict[int, dict] | None = None,
) -> dict[int, dict]:
    """
    オッズ予測の統合エントリポイント。

    優先順位:
      1. オッズ推移トラッキング（3回以上の履歴がある場合）
      2. モデル予測（学習済みモデルがある場合）
      3. ヒューリスティック推定

    live_odds が渡された場合、推移トラッカーに記録した上で予測に利用する。

    Returns: {horse_number: {predicted_win_odds, predicted_place_odds_min, ...}}
    """
    tracker = OddsTrajectoryTracker()

    if live_odds:
        entries = [
            {"horse_number": hn, **v}
            for hn, v in live_odds.items()
        ]
        tracker.record_snapshot(race_id, {"entries": entries})

    trajectory_preds = tracker.predict_final(race_id)
    if trajectory_preds and len(trajectory_preds) >= len(features_df) * 0.5:
        logger.info("オッズ推移ベースの予測を使用 (confidence=%.1f)", trajectory_preds[0].confidence)
        return _to_dict(trajectory_preds)

    predictor = OddsPredictor()
    model_preds = predictor.predict(features_df)

    if trajectory_preds:
        blended = _blend(trajectory_preds, model_preds)
        return _to_dict(blended)

    return _to_dict(model_preds)


def _to_dict(preds: list[PredictedOdds]) -> dict[int, dict]:
    return {
        p.horse_number: {
            "win_odds": p.predicted_win_odds,
            "place_odds_min": p.predicted_place_odds_min,
            "place_odds_max": p.predicted_place_odds_max,
            "confidence": round(p.confidence, 2),
            "source": p.source,
        }
        for p in preds
    }


def _blend(
    trajectory: list[PredictedOdds],
    model: list[PredictedOdds],
) -> list[PredictedOdds]:
    """推移予測とモデル予測をブレンドする。推移の信頼度に応じて重み付け。"""
    traj_map = {p.horse_number: p for p in trajectory}
    model_map = {p.horse_number: p for p in model}

    blended = []
    all_hns = set(traj_map.keys()) | set(model_map.keys())

    for hn in sorted(all_hns):
        t = traj_map.get(hn)
        m = model_map.get(hn)

        if t and m:
            w = t.confidence
            blended.append(PredictedOdds(
                horse_number=hn,
                predicted_win_odds=round(t.predicted_win_odds * w + m.predicted_win_odds * (1 - w), 1),
                predicted_place_odds_min=round(t.predicted_place_odds_min * w + m.predicted_place_odds_min * (1 - w), 1),
                predicted_place_odds_max=round(t.predicted_place_odds_max * w + m.predicted_place_odds_max * (1 - w), 1),
                confidence=round(max(t.confidence, m.confidence), 2),
                source="blend",
            ))
        elif t:
            blended.append(t)
        elif m:
            blended.append(m)

    return blended
