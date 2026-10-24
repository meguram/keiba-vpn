"""馬連・ワイド・3連複の確率モデル（割引 Harville）。設計: docs/html/modeling/chapters/exotic_bet_probabilities.html"""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.inference.betting import BettingConfig, BettingOptimizer
from src.pipeline.inference.exotic_probabilities import (
    RankDistribution,
    fit_rank_distribution,
    normalize_marginal,
)
from src.scripts.data.validate_exotic_probabilities import (
    _weighted_rel_error,
    empirical_truth,
    simulate_thurstone,
)
from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob


def _win(n=12, seed=0):
    return np.random.default_rng(seed).dirichlet(np.ones(n) * 0.8)


def test_alpha_one_is_harville():
    w = _win()
    d = RankDistribution(w)
    assert np.allclose(d.top2(), harville_top2_prob(w)) and np.allclose(d.top3(), harville_top3_prob(w))


@pytest.mark.parametrize("a2,a3", [(1.0, 1.0), (0.7, 0.6), (0.4, 1.2)])
def test_totals_are_consistent_for_any_alpha(a2, a3):
    d = RankDistribution(_win(), a2, a3)
    assert sum(d.all_umaren().values()) == pytest.approx(1.0)
    assert sum(d.all_sanrenpuku().values()) == pytest.approx(1.0)
    assert sum(d.all_wide().values()) == pytest.approx(3.0)  # 3 着以内の 3 頭から取る 2 頭の組 C(3,2)=3
    assert d._p2.sum() == pytest.approx(1.0)
    assert d.top2().sum() == pytest.approx(2.0) and d.top3().sum() == pytest.approx(3.0)
    assert d.win.sum() == pytest.approx(1.0)


def test_pair_and_trio_relations():
    d = RankDistribution(_win(), 0.8, 0.7)
    # 馬連 ≤ ワイド（1-2着は3着以内に含まれる）、ワイド = 3連複の和、馬連 = 馬単 2 通りの和
    for (i, j), p in d.all_umaren().items():
        assert p == pytest.approx(d.umatan(i, j) + d.umatan(j, i))
        assert d.wide(i, j) >= p - 1e-12
    for i, j in [(0, 1), (3, 7)]:
        assert d.wide(i, j) == pytest.approx(sum(d.sanrenpuku(i, j, k) for k in range(d.n) if k not in (i, j)))
    assert d.sanrenpuku(0, 1, 2) == pytest.approx(d.sanrenpuku(2, 0, 1))  # 順不同


def test_lower_alpha_moves_mass_to_longshots():
    w = _win()
    fav, long = int(np.argmax(w)), int(np.argmin(w))
    h, d = RankDistribution(w, 1.0, 1.0), RankDistribution(w, 0.6, 0.6)
    assert d.top3()[long] > h.top3()[long] and d.top3()[fav] < h.top3()[fav]


def test_fit_recovers_known_alpha_and_matches_marginals():
    w = _win(14, 3)
    truth = RankDistribution(w, 0.72, 0.61)
    fit, rep = fit_rank_distribution(w, truth.top2(), truth.top3())
    assert rep.alpha2 == pytest.approx(0.72, abs=0.01) and rep.alpha3 == pytest.approx(0.61, abs=0.01)
    assert rep.top3_rmse_after < 1e-3 < rep.top3_rmse_before
    assert np.allclose(fit.top3(), truth.top3(), atol=2e-3)


def test_fit_without_marginals_stays_harville_and_with_top3_only_uses_one_alpha():
    w = _win()
    _, rep0 = fit_rank_distribution(w)
    assert (rep0.alpha2, rep0.alpha3) == (1.0, 1.0)
    truth = RankDistribution(w, 0.7, 0.7)
    _, rep3 = fit_rank_distribution(w, None, truth.top3())
    assert rep3.alpha2 == rep3.alpha3 == pytest.approx(0.7, abs=0.02) and rep3.used_top3 and not rep3.used_top2


def test_small_fields_do_not_crash():
    for n in (2, 3, 4):
        d, _ = fit_rank_distribution([1.0] * n, [0.5] * n if n > 2 else None, [0.75] * n if n > 3 else None)
        assert sum(d.all_umaren().values()) == pytest.approx(1.0)
        d.all_wide(), d.all_sanrenpuku()


def test_normalize_marginal_keeps_bounds_and_total():
    v = normalize_marginal([0.9, 0.9, 0.9, 0.1, 0.1, 0.1], 3.0)
    assert v.sum() == pytest.approx(3.0) and v.max() <= 1.0 and v.min() > 0


def test_fitted_model_is_much_closer_to_truth_than_harville_on_thurstone_simulation():
    """真の着順が Thurstone 型（Harville と別物）のとき、連対率・複勝率に合わせた方が組合せ確率の誤差が大幅に小さい。"""
    rng = np.random.default_rng(11)
    n = 12
    strength = rng.normal(0, 1, n)
    truth = empirical_truth(simulate_thurstone(strength, 1.0, 150_000, rng), n)
    h = RankDistribution(truth["win"])
    fit, _ = fit_rank_distribution(truth["win"], truth["top2"], truth["top3"])
    for kind, getter in (("umaren", "all_umaren"), ("wide", "all_wide"), ("sanrenpuku", "all_sanrenpuku")):
        e_h = _weighted_rel_error(getattr(h, getter)(), truth[kind])
        e_f = _weighted_rel_error(getattr(fit, getter)(), truth[kind])
        assert e_f < 0.5 * e_h, (kind, e_h, e_f)


# ── BettingOptimizer への組み込み ──

def _preds(with_marginals=True, a2=0.7, a3=0.65):
    w = _win(9, 5)
    d = RankDistribution(w, a2, a3)
    rows = []
    for i, p in enumerate(w):
        r = {"horse_number": i + 1, "horse_name": f"h{i + 1}", "pred_score": float(np.log(p)), "win_prob": float(p)}
        if with_marginals:
            r["top2_prob"], r["top3_prob"] = float(d.top2()[i]), float(d.top3()[i])
        rows.append(r)
    return rows, d


def _odds_for(n=9):
    from itertools import combinations

    return {
        "umaren": [{"pair": [a, b], "odds": 900.0} for a, b in combinations(range(1, n + 1), 2)],
        "wide": [{"pair": [a, b], "odds_min": 400.0, "odds_max": 500.0} for a, b in combinations(range(1, n + 1), 2)],
        "sanrenpuku": [{"combo": list(c), "odds": 9000.0} for c in combinations(range(1, n + 1), 3)],
    }


def test_optimizer_uses_discounted_model_when_predictions_have_win_prob():
    preds, d = _preds()
    cfg = BettingConfig(bet_types=["umaren", "wide", "sanrenpuku", "fukusho"], min_ev=0.0, min_prob=0.0, max_candidates=10_000,
                        top_n_for_pairs=9, max_total_bet=None)
    out = BettingOptimizer(cfg).optimize(preds, _odds_for(), 10**9, single_odds={"entries": [
        {"horse_number": i + 1, "win_odds": 5.0, "place_odds_min": 90.0, "place_odds_max": 99.0} for i in range(9)]})
    pm = out["probability_model"]
    assert pm["kind"] == "discounted_harville" and pm["alpha2"] == pytest.approx(0.7, abs=0.02) and pm["alpha3"] == pytest.approx(0.65, abs=0.02)
    by = {(c.bet_type, c.pair): c for c in out["candidates"]}
    assert by[("umaren", (1, 2))].prob == pytest.approx(d.umaren(0, 1), abs=2e-4)
    assert by[("wide", (1, 2))].prob == pytest.approx(d.wide(0, 1), abs=2e-4)
    assert by[("sanrenpuku", (1, 2, 3))].prob == pytest.approx(d.sanrenpuku(0, 1, 2), abs=2e-4)
    # 複勝の確率はモデルが出した複勝率そのもの（ヒューリスティックではない）
    assert by[("fukusho", (1,))].prob == pytest.approx(float(preds[0]["top3_prob"]), abs=2e-4)


def test_optimizer_falls_back_to_legacy_without_win_prob_or_when_forced():
    preds, _ = _preds()
    legacy_rows = [{k: v for k, v in r.items() if k not in ("win_prob", "top2_prob", "top3_prob")} for r in preds]
    cfg = BettingConfig(bet_types=["umaren"], min_ev=0.0, min_prob=0.0, top_n_for_pairs=9, max_total_bet=None)
    assert BettingOptimizer(cfg).optimize(legacy_rows, _odds_for(), 100000)["probability_model"] == {"kind": "softmax_harville"}
    forced = BettingConfig(bet_types=["umaren"], min_ev=0.0, min_prob=0.0, prob_model="softmax", max_total_bet=None)
    assert BettingOptimizer(forced).optimize(preds, _odds_for(), 100000)["probability_model"]["kind"] == "softmax_harville"


def test_optimizer_without_marginals_is_plain_harville_on_win_prob():
    preds, _ = _preds(with_marginals=False)
    cfg = BettingConfig(bet_types=["umaren"], min_ev=0.0, min_prob=0.0, top_n_for_pairs=9, max_total_bet=None)
    out = BettingOptimizer(cfg).optimize(preds, _odds_for(), 100000)
    pm = out["probability_model"]
    assert pm["kind"] == "discounted_harville" and (pm["alpha2"], pm["alpha3"]) == (1.0, 1.0) and pm["fitted_to"] == []


# ── 学習済みモデルの出力での評価・調整（predictions サブコマンド） ──

def _sample_races(n_races, a2, a3, seed=0, n=12):
    """既知の割引 Harville から 1〜3 着を逐次サンプリングして「実際の結果」を作る。"""
    rng = np.random.default_rng(seed)
    races = []
    for k in range(n_races):
        w = rng.dirichlet(np.ones(n) * 0.8)
        d = RankDistribution(w, a2, a3)
        flat = d._p3.reshape(-1)
        a, b, c = np.unravel_index(rng.choice(flat.size, p=flat / flat.sum()), d._p3.shape)
        nums = list(range(1, n + 1))
        order = [a + 1, b + 1, c + 1]
        pos = {hn: (order.index(hn) + 1 if hn in order else 99) for hn in nums}
        races.append({"rid": f"r{k}", "nums": nums, "win": list(w), "top2": [None] * n, "top3": [None] * n, "finish": order, "pos": pos})
    return races


def test_global_alpha_mle_recovers_known_alpha():
    from src.scripts.data.validate_exotic_probabilities import mle_global_alpha

    a2, a3 = mle_global_alpha(_sample_races(1500, 0.7, 0.6))
    assert a2 == pytest.approx(0.7, abs=0.08) and a3 == pytest.approx(0.6, abs=0.08)
    h2, h3 = mle_global_alpha(_sample_races(1500, 1.0, 1.0, seed=1))
    assert h2 == pytest.approx(1.0, abs=0.1) and h3 == pytest.approx(1.0, abs=0.1)


def test_evaluate_predictions_prefers_global_alpha_over_harville_when_truth_is_discounted():
    from src.scripts.data.validate_exotic_probabilities import evaluate_predictions

    r = evaluate_predictions(_sample_races(600, 0.65, 0.55, seed=2))
    ll = r["realized_logloss"]
    assert ll["global_alpha_sanrenpuku"] < ll["harville_sanrenpuku"] and ll["global_alpha_umaren"] < ll["harville_umaren"]
    assert set(r["calibration"]) == {"win", "top2", "top3"}


def test_default_alpha_comes_from_config_and_ignores_bad_values(tmp_path, monkeypatch):
    from src.pipeline.inference import exotic_probabilities as E

    cfg = tmp_path / "exotic_alpha.json"
    monkeypatch.setattr(E, "ALPHA_CONFIG", cfg)
    E.load_default_alpha.cache_clear()
    assert E.load_default_alpha() == (1.0, 1.0)  # ファイル無し = Harville
    cfg.write_text('{"alpha2": 0.8, "alpha3": 0.7}')
    E.load_default_alpha.cache_clear()
    assert E.load_default_alpha() == (0.8, 0.7)
    _, rep = fit_rank_distribution(_win())  # 連対率・複勝率が無ければ既定値を使う
    assert (rep.alpha2, rep.alpha3) == (0.8, 0.7)
    for bad in ('{"alpha2": 9, "alpha3": 0.7}', "not json", '{"alpha2": "x"}'):
        cfg.write_text(bad)
        E.load_default_alpha.cache_clear()
        assert E.load_default_alpha() == (1.0, 1.0)
    E.load_default_alpha.cache_clear()


def test_write_alpha_config_roundtrip(tmp_path, monkeypatch):
    from src.pipeline.inference import exotic_probabilities as E
    from src.scripts.data.validate_exotic_probabilities import write_alpha_config

    monkeypatch.setattr(E, "ALPHA_CONFIG", tmp_path / "a.json")
    E.load_default_alpha.cache_clear()
    write_alpha_config(0.77, 0.66, 500, "20260101-20261231")
    assert E.load_default_alpha() == (0.77, 0.66)
    E.load_default_alpha.cache_clear()


def test_collect_predicted_races_excludes_mock_predictions():
    import time

    from src.pipeline.inference.race_prediction_service import RacePredictionCache
    from src.scripts.data.validate_exotic_probabilities import collect_predicted_races

    def blob(mock):
        rows = [{"horse_number": i + 1, "win_prob": 0.1, "top2_prob": 0.2, "top3_prob": 0.3} for i in range(8)]
        meta = {"version": RacePredictionCache.cache_version, "model_key": RacePredictionCache.model_key,
                "computed_at_epoch": time.time(), "source": "dev_mock" if mock else "batch"}
        return {"predictions": rows, "_cache_meta": meta, "_meta": {"mock_prediction": True} if mock else {}}

    res = {"entries": [{"horse_number": i + 1, "finish_position": i + 1} for i in range(8)]}

    class S:
        data = {("race_predictions", "real"): blob(False), ("race_predictions", "mock"): blob(True),
                ("race_result", "real"): res, ("race_result", "mock"): res}

        def load(self, cat, key):
            return self.data.get((cat, key))

    races, skipped = collect_predicted_races(S(), ["real", "mock", "none"])
    assert [r["rid"] for r in races] == ["real"] and skipped["mock"] == 1 and skipped["no_prediction"] == 1
    assert [r["rid"] for r in collect_predicted_races(S(), ["real", "mock"], include_mock=True)[0]] == ["real", "mock"]
