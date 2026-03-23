"""
馬券最適化エンジン — 全馬券種 (単勝・複勝・馬連・ワイド・馬単) の期待値計算とKelly基準

設計思想:
  - モデルの予測スコア (pred_score) から各馬の勝率分布を推定
  - 単独 or 2頭の組み合わせ確率を計算し、オッズとの対比で期待値 (EV) を算出
  - Kelly Criterion で最適賭け金比率を計算
  - fractional Kelly (推奨: 0.25) でリスクを制御
  - 複数候補に対する並列資金配分を最適化

馬券種別:
  tansho  — 単勝 (1着)
  fukusho — 複勝 (3着以内)
  umaren  — 馬連 (2頭の組み合わせ、着順不問、1-2着)
  wide    — ワイド (2頭の組み合わせ、着順不問、1-3着)
  umatan  — 馬単 (2頭の順序あり、1着→2着)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any

import numpy as np

logger = logging.getLogger("pipeline.betting")


@dataclass
class BetCandidate:
    """1つの馬券候補"""
    bet_type: str
    pair: tuple[int, int]
    horse_names: tuple[str, str]
    odds: float
    prob: float
    ev: float
    kelly_fraction: float
    bet_amount: int
    expected_return: float

    @property
    def pair_label(self) -> str:
        if self.bet_type in ("tansho", "fukusho"):
            return f"{self.pair[0]}"
        sep = "→" if self.bet_type == "umatan" else "-"
        return f"{self.pair[0]}{sep}{self.pair[1]}"


SINGLE_BET_TYPES = {"tansho", "fukusho"}
PAIR_BET_TYPES = {"umaren", "wide", "umatan"}


@dataclass
class BettingConfig:
    """馬券最適化の設定"""
    bet_types: list[str] = field(
        default_factory=lambda: ["tansho", "fukusho", "umaren", "wide"]
    )
    min_ev: float = 1.05
    min_prob: float = 0.02
    max_candidates: int = 15
    kelly_fraction: float = 0.25
    min_bet: int = 100
    max_bet_ratio: float = 0.10
    wide_use_min_odds: bool = True
    fukusho_use_min_odds: bool = True
    top_n_for_pairs: int = 6


class BettingOptimizer:
    """
    予測スコアとオッズから最適な馬券ポートフォリオを構築する。

    Usage:
        optimizer = BettingOptimizer(config)
        portfolio = optimizer.optimize(
            predictions_df,    # pred_score, horse_number, horse_name
            pair_odds,         # {"umaren": [...], "wide": [...], "umatan": [...]}
            bankroll=100000,
        )
    """

    def __init__(self, config: BettingConfig | None = None):
        self.config = config or BettingConfig()

    def optimize(
        self,
        predictions: list[dict],
        pair_odds: dict[str, list[dict]],
        bankroll: int = 100000,
        single_odds: dict[str, list[dict]] | None = None,
    ) -> dict[str, Any]:
        """
        予測結果とオッズから最適馬券ポートフォリオを構築する。

        Args:
            predictions: 予測結果リスト (horse_number, pred_score, horse_name,
                         win_odds, place_odds_min, place_odds_max を含むことが望ましい)
            pair_odds: 2連系オッズ辞書 (scraper出力のまま)
            bankroll: 資金
            single_odds: 単勝・複勝オッズ ({"entries": [...]})。
                         None の場合は predictions 内のオッズフィールドを使用。

        Returns:
            {
                "candidates": [BetCandidate, ...],
                "total_bet": int,
                "expected_return": float,
                "expected_roi": float,
                "bankroll": int,
                "remaining": int,
                "prob_distribution": {horse_number: prob},
            }
        """
        probs = self._compute_probabilities(predictions)
        name_map = {p["horse_number"]: p.get("horse_name", f"#{p['horse_number']}")
                    for p in predictions}

        all_odds = self._build_odds_map(pair_odds, predictions, single_odds)

        candidates: list[BetCandidate] = []

        for bet_type in self.config.bet_types:
            type_odds = all_odds.get(bet_type, {})
            for key, odds_val in type_odds.items():
                odds = odds_val if isinstance(odds_val, (int, float)) else odds_val[0]
                if odds <= 0:
                    continue

                if bet_type in SINGLE_BET_TYPES:
                    hn = key[0]
                    prob = self._single_prob(probs, hn, bet_type)
                    names = (name_map.get(hn, "?"), "")
                else:
                    h1, h2 = key
                    prob = self._pair_prob(probs, h1, h2, bet_type)
                    names = (name_map.get(h1, "?"), name_map.get(h2, "?"))

                if prob < self.config.min_prob:
                    continue

                ev = prob * odds
                if ev < self.config.min_ev:
                    continue

                kf = self._kelly(prob, odds) * self.config.kelly_fraction
                kf = max(0.0, min(kf, self.config.max_bet_ratio))

                candidates.append(BetCandidate(
                    bet_type=bet_type,
                    pair=key,
                    horse_names=names,
                    odds=odds,
                    prob=round(prob, 4),
                    ev=round(ev, 3),
                    kelly_fraction=round(kf, 4),
                    bet_amount=0,
                    expected_return=0.0,
                ))

        candidates.sort(key=lambda c: c.ev, reverse=True)
        candidates = candidates[:self.config.max_candidates]

        candidates = self._allocate(candidates, bankroll)

        total_bet = sum(c.bet_amount for c in candidates)
        total_expected = sum(c.expected_return for c in candidates)

        return {
            "candidates": candidates,
            "total_bet": total_bet,
            "expected_return": round(total_expected, 0),
            "expected_roi": round(total_expected / total_bet, 3) if total_bet > 0 else 0.0,
            "bankroll": bankroll,
            "remaining": bankroll - total_bet,
            "prob_distribution": {
                hn: round(p, 4) for hn, p in probs.items()
            },
        }

    # ── 確率推定 ──

    def _compute_probabilities(
        self, predictions: list[dict],
    ) -> dict[int, float]:
        """
        pred_score → softmax で確率分布に変換する。
        温度パラメータ (temperature) でシャープさを調整。
        """
        scores = np.array([p["pred_score"] for p in predictions], dtype=np.float64)
        numbers = [p["horse_number"] for p in predictions]

        temperature = self._estimate_temperature(len(scores))
        scaled = scores / temperature
        scaled -= scaled.max()
        exp_scores = np.exp(scaled)
        softmax = exp_scores / exp_scores.sum()

        return dict(zip(numbers, softmax.tolist()))

    @staticmethod
    def _estimate_temperature(field_size: int) -> float:
        """頭数に応じた温度。多頭数ほど分散を広げる。"""
        if field_size <= 8:
            return 1.0
        if field_size <= 14:
            return 1.2
        return 1.5

    # ── 単独確率 (単勝・複勝) ──

    @staticmethod
    def _single_prob(
        probs: dict[int, float], hn: int, bet_type: str,
    ) -> float:
        """
        単勝: 1着確率 = softmax 確率そのまま
        複勝: 3着以内確率 ≈ 1 - (1-p)^3 の近似 (独立でないが実用十分)
              より正確には: 上位3頭に含まれる確率
        """
        p = probs.get(hn, 0.0)
        if bet_type == "tansho":
            return p
        elif bet_type == "fukusho":
            n = len(probs)
            if n <= 3:
                return 1.0 if p > 0 else 0.0
            top3_prob = 0.0
            sorted_horses = sorted(probs.items(), key=lambda x: -x[1])
            target_rank = next(
                (i for i, (h, _) in enumerate(sorted_horses) if h == hn), n
            )
            if target_rank < 3:
                top3_prob = min(1.0, p * n / 3 * 0.85 + 0.15)
            else:
                top3_prob = min(0.95, p * n / 3 * 0.7)
            return max(0.0, min(1.0, top3_prob))
        return 0.0

    # ── 組み合わせ確率 (馬連・ワイド・馬単) ──

    def _pair_prob(
        self,
        probs: dict[int, float],
        h1: int, h2: int,
        bet_type: str,
    ) -> float:
        """
        2頭の組み合わせ的中確率を計算する。

        馬連: h1,h2 が 1-2着 (順不同)
        ワイド: h1,h2 が共に 1-3着 (順不同)
        馬単: h1 が 1着、h2 が 2着 (順序あり)
        """
        p1 = probs.get(h1, 0.0)
        p2 = probs.get(h2, 0.0)

        if p1 == 0 or p2 == 0:
            return 0.0

        others = {k: v for k, v in probs.items() if k not in (h1, h2)}
        total_others = sum(others.values())

        if bet_type == "umaren":
            prob_h1_first_h2_second = p1 * (p2 / (1 - p1)) if p1 < 1 else 0
            prob_h2_first_h1_second = p2 * (p1 / (1 - p2)) if p2 < 1 else 0
            return prob_h1_first_h2_second + prob_h2_first_h1_second

        elif bet_type == "umatan":
            return p1 * (p2 / (1 - p1)) if p1 < 1 else 0

        elif bet_type == "wide":
            return self._wide_prob(probs, h1, h2)

        return 0.0

    @staticmethod
    def _wide_prob(probs: dict[int, float], h1: int, h2: int) -> float:
        """
        ワイド的中確率: h1, h2 が共に3着以内に入る確率。
        3着以内に入る確率 ≈ Σ(順列) で近似計算。
        """
        p1 = probs.get(h1, 0.0)
        p2 = probs.get(h2, 0.0)
        horses = sorted(probs.keys())
        n = len(horses)

        if n < 3:
            return 0.0

        p_both_top3 = 0.0

        for pos1 in range(3):
            for pos2 in range(3):
                if pos1 == pos2:
                    continue

                remaining_for_third = [h for h in horses if h not in (h1, h2)]

                if pos1 == 0 and pos2 == 1:
                    p_config = p1 * (p2 / max(1 - p1, 1e-10))
                elif pos1 == 1 and pos2 == 0:
                    p_config = p2 * (p1 / max(1 - p2, 1e-10))
                elif pos1 == 0 and pos2 == 2:
                    others_sum = sum(probs.get(h, 0) for h in remaining_for_third)
                    p_second_other = others_sum / max(1 - p1, 1e-10)
                    p_config = p1 * p_second_other * (
                        p2 / max(1 - p1 - others_sum * p1 / max(1 - p1, 1e-10), 1e-10)
                    )
                    p_config = min(p_config, p1 * p2 * 2)
                elif pos1 == 2 and pos2 == 0:
                    others_sum = sum(probs.get(h, 0) for h in remaining_for_third)
                    p_config = p2 * (others_sum / max(1 - p2, 1e-10)) * (
                        p1 / max(1 - p2 - others_sum * p2 / max(1 - p2, 1e-10), 1e-10)
                    )
                    p_config = min(p_config, p1 * p2 * 2)
                elif pos1 == 1 and pos2 == 2:
                    others_sum = sum(probs.get(h, 0) for h in remaining_for_third)
                    p_first_other = others_sum
                    p_config = p_first_other * (p1 / max(1 - p_first_other, 1e-10)) * (
                        p2 / max(1 - p_first_other - p1, 1e-10)
                    )
                    p_config = min(p_config, p1 * p2 * 2)
                elif pos1 == 2 and pos2 == 1:
                    others_sum = sum(probs.get(h, 0) for h in remaining_for_third)
                    p_first_other = others_sum
                    p_config = p_first_other * (p2 / max(1 - p_first_other, 1e-10)) * (
                        p1 / max(1 - p_first_other - p2, 1e-10)
                    )
                    p_config = min(p_config, p1 * p2 * 2)
                else:
                    p_config = 0.0

                p_both_top3 += max(0, p_config)

        return min(p_both_top3, 1.0)

    # ── Kelly Criterion ──

    @staticmethod
    def _kelly(prob: float, odds: float) -> float:
        """
        Kelly Criterion: f* = (p * b - q) / b
        where b = odds - 1, p = win prob, q = 1 - p
        """
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0
        b = odds - 1.0
        q = 1.0 - prob
        f = (prob * b - q) / b
        return max(0.0, f)

    # ── 資金配分 ──

    def _allocate(
        self,
        candidates: list[BetCandidate],
        bankroll: int,
    ) -> list[BetCandidate]:
        """
        Kelly fraction に基づいて100円単位で資金配分する。
        """
        if not candidates:
            return candidates

        total_kelly = sum(c.kelly_fraction for c in candidates)
        if total_kelly <= 0:
            return candidates

        max_total = bankroll * self.config.max_bet_ratio * len(candidates)
        available = min(bankroll, max_total)

        for c in candidates:
            raw_amount = available * (c.kelly_fraction / total_kelly)
            rounded = max(0, int(raw_amount // 100) * 100)
            rounded = max(self.config.min_bet, rounded) if c.kelly_fraction > 0 else 0
            c.bet_amount = rounded
            c.expected_return = round(rounded * c.ev, 0)

        total_allocated = sum(c.bet_amount for c in candidates)
        if total_allocated > bankroll:
            scale = bankroll / total_allocated
            for c in candidates:
                c.bet_amount = max(0, int((c.bet_amount * scale) // 100) * 100)
                c.expected_return = round(c.bet_amount * c.ev, 0)

        candidates = [c for c in candidates if c.bet_amount > 0]
        return candidates

    # ── オッズマップ構築 ──

    def _build_odds_map(
        self,
        pair_odds: dict[str, list[dict]],
        predictions: list[dict] | None = None,
        single_odds: dict[str, list[dict]] | None = None,
    ) -> dict[str, dict[tuple[int, ...], float]]:
        """
        全馬券種のオッズを統一形式 {bet_type: {key: odds}} に変換。
        key は単勝/複勝なら (hn,)、2連系なら (h1, h2)。
        """
        result: dict[str, dict] = {}

        # ── 単勝・複勝 ──
        tansho_map: dict[tuple[int, ...], float] = {}
        fukusho_map: dict[tuple[int, ...], float] = {}

        if single_odds and single_odds.get("entries"):
            for e in single_odds["entries"]:
                hn = e.get("horse_number", 0)
                if hn <= 0:
                    continue
                wo = e.get("win_odds", 0.0)
                if wo > 0:
                    tansho_map[(hn,)] = wo
                po = e.get("place_odds_min", 0.0)
                if not self.config.fukusho_use_min_odds:
                    po = (e.get("place_odds_min", 0.0) + e.get("place_odds_max", 0.0)) / 2
                if po > 0:
                    fukusho_map[(hn,)] = po

        if predictions and (not tansho_map or not fukusho_map):
            for p in predictions:
                hn = p.get("horse_number", 0)
                if hn <= 0:
                    continue
                if not tansho_map.get((hn,)):
                    wo = p.get("win_odds", 0.0)
                    if wo > 0:
                        tansho_map[(hn,)] = wo
                if not fukusho_map.get((hn,)):
                    po = p.get("place_odds_min", 0.0)
                    if not self.config.fukusho_use_min_odds:
                        po_max = p.get("place_odds_max", po)
                        po = (po + po_max) / 2 if po_max > 0 else po
                    if po > 0:
                        fukusho_map[(hn,)] = po

        result["tansho"] = tansho_map
        result["fukusho"] = fukusho_map

        # ── 2連系 ──
        for key in ("umaren", "umatan"):
            entries = pair_odds.get(key, []) if pair_odds else []
            m = {}
            for e in entries:
                pair = tuple(e["pair"])
                odds = e.get("odds", 0.0)
                if odds > 0:
                    m[pair] = odds
            result[key] = m

        wide_entries = pair_odds.get("wide", []) if pair_odds else []
        m = {}
        for e in wide_entries:
            pair = tuple(e["pair"])
            if self.config.wide_use_min_odds:
                odds = e.get("odds_min", 0.0)
            else:
                odds = (e.get("odds_min", 0.0) + e.get("odds_max", 0.0)) / 2
            if odds > 0:
                m[pair] = odds
        result["wide"] = m

        return result


class BetSimulator:
    """
    過去データでの馬券シミュレーター。

    バックテスト用: 確定結果 + 確定オッズ + モデル予測 → 仮想的な投票 → 収支計算
    """

    def __init__(self, config: BettingConfig | None = None):
        self.config = config or BettingConfig()
        self.optimizer = BettingOptimizer(self.config)

    def simulate_race(
        self,
        predictions: list[dict],
        pair_odds: dict[str, list[dict]],
        result_order: list[int],
        bankroll: int = 100000,
    ) -> dict[str, Any]:
        """
        1レースの投票シミュレーション。

        Args:
            predictions: 予測結果リスト
            pair_odds: 2連系オッズ
            result_order: 確定着順 (1着, 2着, 3着, ...) の馬番リスト
            bankroll: 資金

        Returns:
            {
                "bets": [BetCandidate],
                "hits": [BetCandidate],
                "total_bet": int,
                "total_return": int,
                "profit": int,
                "roi": float,
            }
        """
        portfolio = self.optimizer.optimize(predictions, pair_odds, bankroll)
        candidates = portfolio["candidates"]

        if len(result_order) < 2:
            return {
                "bets": candidates, "hits": [], "total_bet": portfolio["total_bet"],
                "total_return": 0, "profit": -portfolio["total_bet"], "roi": 0.0,
            }

        top2 = set(result_order[:2])
        top3 = set(result_order[:3])
        first = result_order[0]
        second = result_order[1]

        hits: list[BetCandidate] = []
        total_return = 0

        for bet in candidates:
            hit = False

            if bet.bet_type == "tansho":
                hit = bet.pair[0] == first
            elif bet.bet_type == "fukusho":
                hit = bet.pair[0] in top3
            elif bet.bet_type == "umaren":
                hit = {bet.pair[0], bet.pair[1]} == top2
            elif bet.bet_type == "umatan":
                hit = bet.pair[0] == first and bet.pair[1] == second
            elif bet.bet_type == "wide":
                hit = {bet.pair[0], bet.pair[1]}.issubset(top3)

            if hit:
                payout = int(bet.bet_amount * bet.odds)
                total_return += payout
                hits.append(bet)

        total_bet = portfolio["total_bet"]
        profit = total_return - total_bet

        return {
            "bets": candidates,
            "hits": hits,
            "total_bet": total_bet,
            "total_return": total_return,
            "profit": profit,
            "roi": round(total_return / total_bet, 3) if total_bet > 0 else 0.0,
        }

    def simulate_batch(
        self,
        races: list[dict],
        initial_bankroll: int = 100000,
        reinvest: bool = False,
    ) -> dict[str, Any]:
        """
        複数レースの連続シミュレーション。

        Args:
            races: [{
                "predictions": [...],
                "pair_odds": {...},
                "result_order": [int, ...],
                "race_id": str,
                "race_name": str,
            }]
            initial_bankroll: 初期資金
            reinvest: True なら累積資金で投票

        Returns:
            バッチ結果サマリー + レース毎の詳細
        """
        bankroll = initial_bankroll
        results = []
        total_bet = 0
        total_return = 0

        for race in races:
            current_bankroll = bankroll if reinvest else initial_bankroll
            if current_bankroll < self.config.min_bet:
                break

            sim = self.simulate_race(
                predictions=race["predictions"],
                pair_odds=race["pair_odds"],
                result_order=race["result_order"],
                bankroll=current_bankroll,
            )
            sim["race_id"] = race.get("race_id", "")
            sim["race_name"] = race.get("race_name", "")
            sim["bankroll_before"] = current_bankroll

            if reinvest:
                bankroll += sim["profit"]
            sim["bankroll_after"] = bankroll if reinvest else initial_bankroll + sim["profit"]

            results.append(sim)
            total_bet += sim["total_bet"]
            total_return += sim["total_return"]

        n_races = len(results)
        n_hit_races = sum(1 for r in results if r["hits"])

        profits = [r["profit"] for r in results]
        avg_profit = np.mean(profits) if profits else 0
        std_profit = np.std(profits) if len(profits) > 1 else 0
        sharpe = float(avg_profit / std_profit) if std_profit > 0 else 0.0

        bankroll_history = [initial_bankroll]
        running = initial_bankroll
        for r in results:
            running += r["profit"]
            bankroll_history.append(running)

        max_drawdown = 0.0
        peak = bankroll_history[0]
        for b in bankroll_history:
            if b > peak:
                peak = b
            dd = (peak - b) / peak if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        return {
            "n_races": n_races,
            "n_hit_races": n_hit_races,
            "hit_rate": round(n_hit_races / n_races, 3) if n_races > 0 else 0,
            "total_bet": total_bet,
            "total_return": total_return,
            "total_profit": total_return - total_bet,
            "roi": round(total_return / total_bet, 3) if total_bet > 0 else 0,
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_drawdown, 3),
            "final_bankroll": bankroll_history[-1],
            "bankroll_history": bankroll_history,
            "race_details": results,
        }
