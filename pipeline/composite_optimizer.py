"""
Composite Score パラメータ最適化シミュレーター

過去の大量レースデータ（確定結果＋確定オッズ＋モデル予測スコア）を用いて、
確率と期待値のバランスパラメータを実戦シミュレーションで最適化する。

最適化対象:
  α (prob_weight)       : 確率 vs 期待値の重み (0.0〜1.0)
  min_prob_honmei_ratio : ◎の最低確率閾値 (均等確率に対する比率)
  min_prob_taikou_ratio : ○の最低確率閾値
  top_n_bet             : 印上位何頭に投票するか

評価指標 (複数を多目的で):
  - ROI          : 投資回収率 (1.0以上が黒字)
  - hit_rate     : 複勝的中率
  - top3_capture : 実際3着以内馬のうち印付けで捕捉できた割合
  - sharpe       : レースごとの収支の Sharpe Ratio (安定性)

最終スコア = ROI_weight * ROI + hit_weight * hit_rate + capture_weight * top3_capture + sharpe_weight * sharpe_ratio
"""

from __future__ import annotations

import json
import math
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("CompositeOptimizer")

OPTIM_RESULT_PATH = Path("models/composite_params.json")


@dataclass
class SimulationConfig:
    alpha_range: tuple[float, float, float] = (0.20, 0.80, 0.05)
    min_prob_honmei_range: tuple[float, float, float] = (0.20, 0.80, 0.10)
    min_prob_taikou_range: tuple[float, float, float] = (0.10, 0.60, 0.10)
    top_n_bet_range: tuple[int, int] = (3, 5)
    bet_type: str = "fukusho"  # "fukusho"=複勝, "tansho"=単勝
    roi_weight: float = 0.40
    hit_weight: float = 0.25
    capture_weight: float = 0.20
    sharpe_weight: float = 0.15


@dataclass
class OptimizedParams:
    prob_weight: float = 0.55
    min_prob_honmei_ratio: float = 0.50
    min_prob_taikou_ratio: float = 0.30
    top_n_bet: int = 5
    best_score: float = 0.0
    roi: float = 0.0
    hit_rate: float = 0.0
    top3_capture: float = 0.0
    sharpe_ratio: float = 0.0
    n_races: int = 0
    n_bets: int = 0
    simulation_date: str = ""


class CompositeOptimizer:
    """
    大規模バックテストによるパラメータ最適化。

    storage から過去レースの (特徴量スコア, 確定オッズ, 確定着順) を取得し、
    パラメータ空間をグリッドサーチして最適な組み合わせを特定する。
    """

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self._cache: list[dict] = []

    def build_backtest_dataset(self, storage, max_races: int = 5000) -> list[dict]:
        """
        GCS から過去レースデータを収集し、シミュレーション用データセットを構築。

        各レースについて以下を収集:
          - horse_number, finish_position, odds (確定単勝), estimated_prob (softmax)
          - place_odds_min/max (あれば)
          - race_id, field_size
        """
        if self._cache:
            return self._cache

        from pipeline.feature_builder import build_race_features

        years = storage.list_years("race_result")
        if not years:
            logger.error("GCSにレース結果データがありません")
            return []

        races = []
        total_processed = 0

        categories_map = [
            ("race_shutuba", "race_card"),
            ("race_index", "speed_index"),
            ("race_barometer", "barometer"),
        ]

        for year in sorted(years, reverse=True):
            keys = storage.list_keys("race_result", year)
            for race_id in keys:
                if total_processed >= max_races:
                    break

                try:
                    result = storage.load("race_result", race_id)
                    if not result or "entries" not in result:
                        continue

                    entries = result["entries"]
                    valid_entries = [
                        e for e in entries
                        if e.get("finish_position", 0) > 0
                        and e.get("odds", 0) > 0
                        and e.get("horse_number", 0) > 0
                    ]
                    if len(valid_entries) < 5:
                        continue

                    race_data = {"race_id": race_id, "race_result": result}
                    for storage_cat, data_key in categories_map:
                        d = storage.load(storage_cat, race_id)
                        if d:
                            race_data[data_key] = d

                    if "race_card" not in race_data:
                        continue

                    features_df = build_race_features(race_data)
                    if features_df.empty or len(features_df) < 5:
                        continue

                    result_map = {
                        e["horse_number"]: e for e in valid_entries
                    }

                    # softmax で推定確率を計算
                    from pipeline.race_day import RaceDayPipeline, PipelineConfig
                    try:
                        pipe = RaceDayPipeline(PipelineConfig(
                            model_name="keiba-lgbm-nopi",
                            model_stage="latest",
                            mlflow_tracking_uri="http://localhost:5000",
                        ))
                        meta = pipe._predict(features_df)
                    except Exception:
                        meta = features_df[["horse_number"]].copy() if "horse_number" in features_df.columns else pd.DataFrame()
                        if meta.empty:
                            continue
                        meta["pred_score"] = self._heuristic_score(features_df)
                        meta = meta.sort_values("pred_score", ascending=False).reset_index(drop=True)

                    raw_scores = meta["pred_score"].values.astype(float)
                    centered = raw_scores - raw_scores.mean()
                    std = max(centered.std(), 1e-6)
                    exp_s = np.exp(centered / std)
                    softmax_probs = exp_s / exp_s.sum()

                    race_entries = []
                    for i, (_, row) in enumerate(meta.iterrows()):
                        hn = int(row.get("horse_number", 0))
                        if hn not in result_map:
                            continue
                        r = result_map[hn]
                        race_entries.append({
                            "horse_number": hn,
                            "prob": float(softmax_probs[i]),
                            "pred_score": float(row["pred_score"]),
                            "finish_position": int(r["finish_position"]),
                            "win_odds": float(r.get("odds", 0)),
                            "popularity": int(r.get("popularity", 0)),
                        })

                    # 複勝オッズがあれば取得
                    odds_data = storage.load("race_odds", race_id)
                    if odds_data and odds_data.get("entries"):
                        fukusho_map = {
                            e["horse_number"]: e
                            for e in odds_data["entries"]
                            if e.get("horse_number")
                        }
                        for re in race_entries:
                            fm = fukusho_map.get(re["horse_number"], {})
                            re["place_odds_min"] = float(fm.get("place_odds_min", 0) or 0)
                            re["place_odds_max"] = float(fm.get("place_odds_max", 0) or 0)
                    else:
                        for re in race_entries:
                            re["place_odds_min"] = max(re["win_odds"] * 0.25, 1.0)
                            re["place_odds_max"] = max(re["win_odds"] * 0.55, 1.1)

                    if len(race_entries) >= 5:
                        races.append({
                            "race_id": race_id,
                            "field_size": len(race_entries),
                            "entries": race_entries,
                        })
                        total_processed += 1

                except Exception as e:
                    logger.debug("skip %s: %s", race_id, e)
                    continue

            if total_processed >= max_races:
                break

        logger.info("バックテストデータ構築完了: %dレース", len(races))
        self._cache = races
        return races

    @staticmethod
    def _heuristic_score(df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)
        for col, w in [("speed_max", 30), ("career_top3_rate", 25), ("avg_last_3f_5", -15)]:
            if col in df.columns:
                v = df[col].fillna(0)
                if v.std() > 0:
                    scores += (v - v.mean()) / v.std() * w
        return scores

    def simulate(
        self,
        races: list[dict],
        alpha: float,
        min_prob_honmei_ratio: float,
        min_prob_taikou_ratio: float,
        top_n_bet: int,
    ) -> dict:
        """
        指定パラメータでの仮想投票シミュレーション。

        投票戦略:
          - composite_score = prob^α × EV^(1-α)
          - composite順に上位 top_n_bet 頭に均等額を投票
          - 複勝 (fukusho) の場合: 3着以内なら place_odds_avg を配当として回収
          - 単勝 (tansho) の場合: 1着なら win_odds を配当として回収
        """
        is_fukusho = self.config.bet_type == "fukusho"

        total_bet = 0
        total_return = 0
        hits = 0
        total_bets = 0
        top3_captured = 0
        top3_total = 0
        race_returns: list[float] = []

        for race in races:
            n = race["field_size"]
            base_prob = 1.0 / max(n, 1)
            mp_honmei = base_prob * min_prob_honmei_ratio
            mp_taikou = base_prob * min_prob_taikou_ratio

            scored = []
            for e in race["entries"]:
                prob = e["prob"]
                if is_fukusho:
                    po_min = e.get("place_odds_min", 0) or 0
                    po_max = e.get("place_odds_max", 0) or 0
                    payout_odds = (po_min + po_max) / 2 if (po_min and po_max) else 0
                else:
                    payout_odds = e["win_odds"]

                if prob > 0 and payout_odds > 0:
                    ev = prob * payout_odds
                    composite = (prob ** alpha) * (ev ** (1 - alpha))
                else:
                    ev = 0
                    composite = prob

                scored.append({
                    **e,
                    "composite": composite,
                    "ev": ev,
                    "payout_odds": payout_odds,
                })

            scored.sort(key=lambda x: x["composite"], reverse=True)

            # 印付けと投票対象の選定（確率閾値つき）
            bet_horses = []
            honmei_set = False
            taikou_set = False
            for s in scored:
                if len(bet_horses) >= top_n_bet:
                    break
                p = s["prob"]
                if not honmei_set and p >= mp_honmei:
                    honmei_set = True
                    bet_horses.append(s)
                elif not taikou_set and p >= mp_taikou:
                    taikou_set = True
                    bet_horses.append(s)
                elif len(bet_horses) < top_n_bet:
                    bet_horses.append(s)

            # 足りなければcomposite順で埋める
            if len(bet_horses) < top_n_bet:
                bet_set = {b["horse_number"] for b in bet_horses}
                for s in scored:
                    if s["horse_number"] not in bet_set and len(bet_horses) < top_n_bet:
                        bet_horses.append(s)

            actual_top3 = {e["horse_number"] for e in race["entries"] if e["finish_position"] <= 3}
            top3_total += len(actual_top3)

            race_bet = 0
            race_ret = 0
            for b in bet_horses:
                bet_amount = 100  # 1口100円
                race_bet += bet_amount
                total_bet += bet_amount
                total_bets += 1

                fp = b["finish_position"]
                is_hit = (fp <= 3 if is_fukusho else fp == 1)
                if is_hit:
                    ret = bet_amount * b["payout_odds"]
                    race_ret += ret
                    total_return += ret
                    hits += 1

                if b["horse_number"] in actual_top3:
                    top3_captured += 1

            race_returns.append(race_ret - race_bet)

        roi = total_return / total_bet if total_bet > 0 else 0
        hit_rate = hits / total_bets if total_bets > 0 else 0
        top3_cap = top3_captured / top3_total if top3_total > 0 else 0

        returns_arr = np.array(race_returns) if race_returns else np.array([0.0])
        sharpe = float(returns_arr.mean() / max(returns_arr.std(), 1e-6))

        return {
            "roi": round(roi, 4),
            "hit_rate": round(hit_rate, 4),
            "top3_capture": round(top3_cap, 4),
            "sharpe_ratio": round(sharpe, 4),
            "total_bet": total_bet,
            "total_return": round(total_return, 1),
            "n_bets": total_bets,
            "hits": hits,
            "n_races": len(races),
        }

    def optimize(self, storage, max_races: int = 3000) -> OptimizedParams:
        """
        グリッドサーチでパラメータ空間を探索し、最適な組み合わせを返す。
        """
        t0 = _time.time()
        races = self.build_backtest_dataset(storage, max_races=max_races)
        if not races:
            logger.error("バックテストデータが空です")
            return OptimizedParams()

        build_time = _time.time() - t0
        logger.info("データ構築: %.1f秒, %dレース", build_time, len(races))

        cfg = self.config
        a_start, a_end, a_step = cfg.alpha_range
        ph_start, ph_end, ph_step = cfg.min_prob_honmei_range
        pt_start, pt_end, pt_step = cfg.min_prob_taikou_range
        nb_lo, nb_hi = cfg.top_n_bet_range

        alphas = np.arange(a_start, a_end + a_step / 2, a_step)
        honmei_ratios = np.arange(ph_start, ph_end + ph_step / 2, ph_step)
        taikou_ratios = np.arange(pt_start, pt_end + pt_step / 2, pt_step)
        top_ns = list(range(nb_lo, nb_hi + 1))

        total_combos = len(alphas) * len(honmei_ratios) * len(taikou_ratios) * len(top_ns)
        logger.info("グリッドサーチ開始: %d通り", total_combos)

        best_score = -float("inf")
        best_params = None
        best_metrics = {}
        all_results: list[dict] = []
        checked = 0

        for alpha in alphas:
            for ph_r in honmei_ratios:
                for pt_r in taikou_ratios:
                    if pt_r >= ph_r:
                        continue
                    for tn in top_ns:
                        metrics = self.simulate(races, float(alpha), float(ph_r), float(pt_r), tn)

                        score = (
                            cfg.roi_weight * metrics["roi"]
                            + cfg.hit_weight * metrics["hit_rate"]
                            + cfg.capture_weight * metrics["top3_capture"]
                            + cfg.sharpe_weight * metrics["sharpe_ratio"]
                        )

                        record = {
                            "alpha": round(float(alpha), 2),
                            "min_prob_honmei_ratio": round(float(ph_r), 2),
                            "min_prob_taikou_ratio": round(float(pt_r), 2),
                            "top_n_bet": tn,
                            "score": round(score, 4),
                            **metrics,
                        }
                        all_results.append(record)

                        if score > best_score:
                            best_score = score
                            best_params = record
                            best_metrics = metrics

                        checked += 1

        elapsed = _time.time() - t0
        logger.info(
            "最適化完了 (%.1f秒, %d通り検証)\n"
            "  最適α=%.2f, honmei_ratio=%.2f, taikou_ratio=%.2f, top_n=%d\n"
            "  ROI=%.4f, hit=%.4f, capture=%.4f, sharpe=%.4f, score=%.4f",
            elapsed, checked,
            best_params["alpha"], best_params["min_prob_honmei_ratio"],
            best_params["min_prob_taikou_ratio"], best_params["top_n_bet"],
            best_params["roi"], best_params["hit_rate"],
            best_params["top3_capture"], best_params["sharpe_ratio"],
            best_params["score"],
        )

        opt = OptimizedParams(
            prob_weight=best_params["alpha"],
            min_prob_honmei_ratio=best_params["min_prob_honmei_ratio"],
            min_prob_taikou_ratio=best_params["min_prob_taikou_ratio"],
            top_n_bet=best_params["top_n_bet"],
            best_score=best_params["score"],
            roi=best_params["roi"],
            hit_rate=best_params["hit_rate"],
            top3_capture=best_params["top3_capture"],
            sharpe_ratio=best_params["sharpe_ratio"],
            n_races=best_params["n_races"],
            n_bets=best_params["n_bets"],
            simulation_date=_time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        self._save_result(opt, all_results[:50])
        return opt

    @staticmethod
    def _save_result(params: OptimizedParams, top_results: list[dict]):
        """最適化結果をファイルに永続化する。"""
        top_sorted = sorted(top_results, key=lambda x: -x["score"])[:20]

        data = {
            "optimized": {
                "prob_weight": params.prob_weight,
                "min_prob_honmei_ratio": params.min_prob_honmei_ratio,
                "min_prob_taikou_ratio": params.min_prob_taikou_ratio,
                "top_n_bet": params.top_n_bet,
            },
            "metrics": {
                "best_score": params.best_score,
                "roi": params.roi,
                "hit_rate": params.hit_rate,
                "top3_capture": params.top3_capture,
                "sharpe_ratio": params.sharpe_ratio,
                "n_races": params.n_races,
                "n_bets": params.n_bets,
            },
            "simulation_date": params.simulation_date,
            "top_combinations": top_sorted,
        }

        OPTIM_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OPTIM_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("最適パラメータを保存: %s", OPTIM_RESULT_PATH)


def load_optimized_params() -> dict | None:
    """保存済みの最適パラメータを読み込む。なければ None。"""
    if not OPTIM_RESULT_PATH.exists():
        return None
    try:
        with open(OPTIM_RESULT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("optimized")
    except Exception:
        return None


def get_composite_params() -> dict:
    """
    現在有効な composite score パラメータを返す。

    最適化結果があればそれを使い、なければデフォルト値を返す。
    """
    opt = load_optimized_params()
    if opt:
        return opt
    return {
        "prob_weight": 0.55,
        "min_prob_honmei_ratio": 0.50,
        "min_prob_taikou_ratio": 0.30,
        "top_n_bet": 5,
    }
