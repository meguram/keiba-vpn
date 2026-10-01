"""
馬券最適化エンジン (BettingOptimizer / BetSimulator) の実測検証スクリプト。

2026-10-01 TODO対応: docs/git_management/todo/betting.md の
「馬券ポートフォリオ最適化の結果を実際のレース結果でバックテストし、EVが実際にプラスに
なっているか検証する（現状、実績評価の仕組みが見当たらない）」に対応するための調査スクリプト。

## 現状分かったこと
- `src/pipeline/inference/betting.py` には既に `BetSimulator` / `simulate_race` /
  `simulate_batch` というバックテスト機構そのものは実装済みだった（TODOの「仕組みが見当たらない」
  は半分正しくない）。ただし、これを実データに接続して実行するスクリプト・定期実行・テストは
  リポジトリ内のどこにも存在しなかった（`grep -rn "BetSimulator"` でヒットなし）。

## 重要な制約（この環境でEV検証ができなかった理由）
本スクリプトは本来 `load_real_races()` で HybridStorage 経由の実際の
race_shutuba / race_odds / race_pair_odds / race_result
（または `data/features/target/rank_tbl` の確定ラベル）を読み込み、
`BetSimulator.simulate_batch()` に渡して実測ROIを計算することを想定している。

しかし本スクリプトを作成・実行した開発環境では:
  - `.env` に GCS 認証情報が設定されていない
    （`.env.example` の通り「このPCからの GCS 書き込みは無効化されています。
    本番サーバー側の .env に認証情報を設定してください」という運用のため）
  - ローカルキャッシュ (`data/` 以下) にも race_shutuba / race_odds / race_pair_odds /
    race_result / `data/features/*` が一件も存在しない。
    `HybridStorage().list_keys("race_result"|"race_odds"|"race_pair_odds"|"race_shutuba")`
    はいずれも 0 件（実行して確認済み）。
  - `tests/scraper/fixtures/schema_examples/*.json` はスキーマ検証用で、
    どのレースも出走馬が1頭しか入っていないダミーデータであり、
    的中判定や複数頭の確率計算が成立しない（バックテストには使えない）。

つまり「実際のレース結果・実際の確定オッズ」を使った真のEV検証は、
この環境では安全な方法でのスクレイピングなしには実行不可能だった
（本番外部アクセス厳禁のため実行していない）。

代わりに、本番コード (`BettingOptimizer` / `BetSimulator`) を一切変更せずに
以下の2段階の検証を行う:

  1. `validate_probability_approximations()` —
     `BettingOptimizer._single_prob` (複勝) / `_pair_prob` (馬連・ワイド・馬単) が
     計算する確率が、Plackett-Luce モデルによるモンテカルロ真値とどれだけズレているかを
     多数のランダムな出走表で検証する。
     EV = prob * odds である以上、prob の推定が歪んでいれば実データが完璧でもEVは信頼できない
     ため、データの有無に関係なく確認できる・確認すべき項目。

  2. `run_synthetic_backtest()` —
     人工的に生成した「真の強さ」から市場オッズ（パリミュチュエル近似、テイクアウト込み）と
     予測スコアを作り、
       (a) モデルが市場に対して全く優位性を持たない場合 (no_edge)
       (b) モデルが真の確率に近い優位性を持つ場合 (strong_edge)
     の複数シナリオで `BetSimulator.simulate_batch()` を数百レース実行し、
     回収率 (ROI) がどう変わるかを確認する。これは「EV計算の仕組みが機能しているか」
     （モデルに優位性が無ければROIは100%を大きく下回り、優位性があれば上回るはず）
     の整合性チェックであり、「本番モデルの実際のEVが正か」という問いの答えにはならない。

## 実データでの最終検証に必要なこと
本番/VPS環境 (GCSアクセス可能) で `load_real_races()` に実在する race_id のリストを渡し、
`run_synthetic_backtest` の代わりに実データのレース群を `BetSimulator.simulate_batch()` に
通すこと。race_id の収集元としては `data/features/target/rank_tbl/<年>/rank.parquet`
（確定ラベル）や race_lists が使える。

実行方法:
    python -m src.pipeline.inference.betting_backtest
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from src.pipeline.inference.betting import BetSimulator, BettingConfig, BettingOptimizer

# ---------------------------------------------------------------------------
# 1. 本番データ読み込み（GCS/ローカルキャッシュに実データがある環境で使う）
# ---------------------------------------------------------------------------


def load_real_races(race_ids: list[str], storage: Any = None) -> list[dict]:
    """
    実際の race_shutuba / race_odds / race_pair_odds / race_result から
    `BetSimulator.simulate_batch()` 用のレースデータを組み立てる（スクレイピングなし）。

    GCS/ローカルキャッシュに対象レースのデータが無い場合、そのレースはスキップされる。
    本番/VPS環境（GCSアクセス可）で実行することを想定。この開発環境では
    `list_keys()` が常に0件のため、呼び出しても空リストが返る（モジュール docstring 参照）。
    """
    from src.pipeline.inference.race_prediction_service import (
        build_race_prediction_response,
    )

    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()

    races: list[dict] = []
    for race_id in race_ids:
        result = storage.load("race_result", race_id)
        if not result or not result.get("entries"):
            continue
        ordered = sorted(
            result["entries"], key=lambda e: e.get("finish_position") or 999
        )
        result_order = [
            e["horse_number"] for e in ordered if e.get("finish_position")
        ]
        if len(result_order) < 2:
            continue

        pred_resp = build_race_prediction_response(race_id, storage, allow_scrape=False)
        predictions = pred_resp.get("predictions", [])
        if not predictions:
            continue

        odds = storage.load("race_odds", race_id) or {}
        odds_map = {e.get("horse_number"): e for e in odds.get("entries", [])}
        for p in predictions:
            om = odds_map.get(p.get("horse_number"), {})
            p.setdefault("win_odds", om.get("win_odds", 0.0))
            p.setdefault("place_odds_min", om.get("place_odds_min", 0.0))
            p.setdefault("place_odds_max", om.get("place_odds_max", 0.0))

        pair_odds = storage.load("race_pair_odds", race_id) or {}

        races.append(
            {
                "race_id": race_id,
                "race_name": result.get("race_name", ""),
                "predictions": predictions,
                "pair_odds": pair_odds,
                "result_order": result_order,
            }
        )
    return races


# ---------------------------------------------------------------------------
# 2. 確率近似の正解性検証 (Plackett-Luce モンテカルロ真値との比較)
# ---------------------------------------------------------------------------


def _plackett_luce_sample(strengths: dict[int, float], rng: random.Random) -> list[int]:
    """Plackett-Luce モデルで1レース分の着順をサンプリングする。"""
    remaining = dict(strengths)
    order: list[int] = []
    for _ in range(len(strengths)):
        total = sum(remaining.values())
        r = rng.uniform(0, total)
        acc = 0.0
        for hn, w in remaining.items():
            acc += w
            if acc >= r:
                order.append(hn)
                del remaining[hn]
                break
        else:
            # 浮動小数の誤差で誰も選ばれなかった場合のフォールバック
            hn = next(iter(remaining))
            order.append(hn)
            del remaining[hn]
    return order


def validate_probability_approximations(
    n_trials: int = 150,
    mc_samples: int = 3000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    `BettingOptimizer._single_prob` (複勝) / `_pair_prob` (馬連・ワイド・馬単) の
    近似精度を Plackett-Luce モンテカルロ真値と比較する。
    """
    rng = random.Random(seed)
    optimizer = BettingOptimizer()

    errors: dict[str, list[float]] = {"fukusho": [], "umaren": [], "wide": [], "umatan": []}

    for _ in range(n_trials):
        field_size = rng.randint(8, 18)
        raw_strengths = [rng.gammavariate(2.0, 1.0) for _ in range(field_size)]
        strengths = {i + 1: s for i, s in enumerate(raw_strengths)}
        total = sum(strengths.values())
        probs = {hn: s / total for hn, s in strengths.items()}

        top3_count = {hn: 0 for hn in strengths}
        pair_count: dict[str, dict[tuple[int, int], int]] = {
            "umaren": {}, "wide": {}, "umatan": {},
        }
        for _ in range(mc_samples):
            order = _plackett_luce_sample(strengths, rng)
            for hn in order[:3]:
                top3_count[hn] += 1
            first, second = order[0], order[1]
            key_u = tuple(sorted((first, second)))
            pair_count["umaren"][key_u] = pair_count["umaren"].get(key_u, 0) + 1
            pair_count["umatan"][(first, second)] = (
                pair_count["umatan"].get((first, second), 0) + 1
            )
            for a_idx in range(3):
                for b_idx in range(a_idx + 1, 3):
                    key_w = tuple(sorted((order[a_idx], order[b_idx])))
                    pair_count["wide"][key_w] = pair_count["wide"].get(key_w, 0) + 1

        hn_sample = rng.choice(list(strengths.keys()))
        approx_fukusho = optimizer._single_prob(probs, hn_sample, "fukusho")
        true_fukusho = top3_count[hn_sample] / mc_samples
        errors["fukusho"].append(approx_fukusho - true_fukusho)

        others = [h for h in strengths if h != hn_sample]
        h2 = rng.choice(others)
        for bet_type in ("umaren", "wide", "umatan"):
            approx = optimizer._pair_prob(probs, hn_sample, h2, bet_type)
            if bet_type == "umatan":
                true_p = pair_count["umatan"].get((hn_sample, h2), 0) / mc_samples
            else:
                key = tuple(sorted((hn_sample, h2)))
                true_p = pair_count[bet_type].get(key, 0) / mc_samples
            errors[bet_type].append(approx - true_p)

    summary: dict[str, dict[str, float]] = {}
    for bet_type, errs in errors.items():
        arr = np.array(errs)
        summary[bet_type] = {
            "n": len(arr),
            "mean_error": round(float(arr.mean()), 5),
            "mean_abs_error": round(float(np.abs(arr).mean()), 5),
            "max_abs_error": round(float(np.abs(arr).max()), 5),
        }
    return summary


# ---------------------------------------------------------------------------
# 3. 合成データによる統合シミュレーション
# ---------------------------------------------------------------------------


def _generate_synthetic_race(
    race_idx: int,
    rng: random.Random,
    model_edge: float,
    takeout: float = 0.20,
) -> dict:
    """
    1レース分の人工データを生成する（実データではないことに注意）。

    model_edge:
      0.0 -> モデルの予測スコアは市場確率と同じ分布から生成（優位性なし）
      1.0 -> モデルの予測スコアは真の確率をそのまま使う（理想的な優位性）
    takeout: パリミュチュエル市場のテイクアウト率（JRA実勢の近似 ~20%）
    """
    field_size = rng.randint(8, 16)
    raw_strengths = [rng.gammavariate(2.0, 1.0) for _ in range(field_size)]
    total_strength = sum(raw_strengths)
    true_probs = {i + 1: s / total_strength for i, s in enumerate(raw_strengths)}

    # 市場人気 = 真の確率に対数正規ノイズを乗せたもの（公衆予想の誤差を模す）
    market_noise = {
        hn: max(1e-4, p * rng.lognormvariate(0, 0.35)) for hn, p in true_probs.items()
    }
    market_total = sum(market_noise.values())
    market_probs = {hn: v / market_total for hn, v in market_noise.items()}

    # 単勝オッズ ≈ (1 - takeout) / 市場確率 のパリミュチュエル近似
    win_odds = {hn: round((1 - takeout) / p, 1) for hn, p in market_probs.items()}

    # モデル予測スコア: 真の確率と市場確率を edge でブレンド + 推定誤差ノイズ
    model_raw = {}
    for hn in true_probs:
        blended = model_edge * true_probs[hn] + (1 - model_edge) * market_probs[hn]
        model_raw[hn] = max(1e-6, blended * rng.lognormvariate(0, 0.10))
    model_total = sum(model_raw.values())
    pred_scores = {hn: v / model_total for hn, v in model_raw.items()}

    predictions = []
    for hn in true_probs:
        predictions.append({
            "horse_number": hn,
            "horse_name": f"syn_{race_idx}_{hn}",
            "pred_score": pred_scores[hn],
            "win_odds": win_odds[hn],
            "place_odds_min": round(max(1.0, win_odds[hn] * 0.35), 1),
            "place_odds_max": round(max(1.0, win_odds[hn] * 0.55), 1),
        })

    result_order = _plackett_luce_sample(true_probs, rng)

    pair_odds: dict[str, list[dict]] = {"umaren": [], "wide": [], "umatan": []}
    horses = sorted(true_probs.keys())
    for i, h1 in enumerate(horses):
        for h2 in horses[i + 1:]:
            p_pair_approx = market_probs[h1] * market_probs[h2] * 2.0
            fair_odds = (1 - takeout) / max(p_pair_approx, 1e-6)
            pair_odds["umaren"].append({"pair": [h1, h2], "odds": round(fair_odds, 1)})
            pair_odds["wide"].append({
                "pair": [h1, h2],
                "odds_min": round(fair_odds * 0.4, 1),
                "odds_max": round(fair_odds * 0.6, 1),
            })
    for h1 in horses:
        for h2 in horses:
            if h1 == h2:
                continue
            p_pair_approx = market_probs[h1] * market_probs[h2] * 1.3
            fair_odds = (1 - takeout) / max(p_pair_approx, 1e-6)
            pair_odds["umatan"].append({"pair": [h1, h2], "odds": round(fair_odds, 1)})

    return {
        "race_id": f"SYN{race_idx:06d}",
        "race_name": f"synthetic race {race_idx}",
        "predictions": predictions,
        "pair_odds": pair_odds,
        "result_order": result_order,
    }


def run_synthetic_backtest(n_races: int = 500, seed: int = 7) -> dict[str, dict]:
    """
    合成データ（実データではない）で `BetSimulator.simulate_batch()` を実行し、
    モデルの市場優位性の大小によって実現ROIがどう変わるかを確認する。
    """
    rng = random.Random(seed)
    scenarios = {
        "no_edge (model == market)": 0.0,
        "weak_edge": 0.4,
        "strong_edge (model == true prob)": 1.0,
    }
    results: dict[str, dict] = {}
    for label, edge in scenarios.items():
        races = [_generate_synthetic_race(i, rng, model_edge=edge) for i in range(n_races)]
        config = BettingConfig(
            bet_types=["tansho", "fukusho", "umaren", "wide", "umatan"],
        )
        sim = BetSimulator(config)
        batch = sim.simulate_batch(races, initial_bankroll=100_000, reinvest=False)
        results[label] = {
            "n_races": batch["n_races"],
            "hit_rate": batch["hit_rate"],
            "roi": batch["roi"],
            "total_bet": batch["total_bet"],
            "total_profit": batch["total_profit"],
            "max_drawdown": batch["max_drawdown"],
            "sharpe_ratio": batch["sharpe_ratio"],
        }
    return results


def main() -> None:
    print("=== 1. 確率近似の精度検証 (Plackett-Luce モンテカルロ比較) ===")
    approx_summary = validate_probability_approximations()
    for bt, s in approx_summary.items():
        print(
            f"  {bt}: n={s['n']} mean_error={s['mean_error']:+.5f} "
            f"mean_abs_error={s['mean_abs_error']:.5f} max_abs_error={s['max_abs_error']:.5f}"
        )

    print()
    print("=== 2. 合成データによる統合シミュレーション (BetSimulator.simulate_batch) ===")
    print("    ※ 実データではない。モデル優位性(edge)別のROI感度チェック。")
    backtest_summary = run_synthetic_backtest()
    for label, s in backtest_summary.items():
        print(
            f"  [{label}] n={s['n_races']} hit_rate={s['hit_rate']} roi={s['roi']} "
            f"profit={s['total_profit']} bet={s['total_bet']} "
            f"sharpe={s['sharpe_ratio']} max_dd={s['max_drawdown']}"
        )


if __name__ == "__main__":
    main()
