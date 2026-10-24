"""凍結した馬券候補の前向き追跡（src/scripts/data/frozen_tracking.py）のテスト。合成データで、判定・精算・冪等・凍結ファイルの変更検知を確かめる。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import frozen_tracking as ft


def _cands(p, odds):
    return pd.DataFrame({"a": range(1, len(p) + 1), "b": range(2, len(p) + 2), "odds": odds, "p_mix": p, "p_model": p})


IDENT = {"x": [0.0, 1.0], "y": [0.0, 1.0]}


def test_calibrate_matches_isotonic():
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(0)
    x = rng.random(500)
    y = (rng.random(500) < x).astype(float)
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
    knots = {"x": list(map(float, ir.X_thresholds_)), "y": list(map(float, ir.y_thresholds_))}
    q = np.concatenate([x[:50], [-0.1, 1.2, 0.0, 1.0]])
    assert np.allclose(ft.calibrate(q, knots), ir.predict(q))


def test_tp62_decide_picks_best_k_under_constraints():
    # k=1 は的中確率の和 12% < 20% で不可、k=2 は最小オッズ 15 ≥ 10 かつ和 22% ≥ 20% で可、3 点目は EV<1 で対象外
    sel = ft.tp62_decide(_cands([0.12, 0.10, 0.08], [20.0, 15.0, 12.0]), IDENT)
    assert sel is not None and sel["k"] == 2 and sel["combos"] == [[1, 2], [2, 3]]
    assert sel["pu"] == pytest.approx(0.22)


def test_tp62_decide_requires_payout_multiple():
    # 最小オッズ 4 < 5×k なので買わない
    assert ft.tp62_decide(_cands([0.3, 0.25], [4.0, 4.5]), IDENT) is None
    # EV<1 のみ
    assert ft.tp62_decide(_cands([0.05, 0.04], [10.0, 12.0]), IDENT) is None
    assert ft.tp62_decide(_cands([], []), IDENT) is None


def test_tp62_decide_max_five_points_and_order():
    p = [0.05] * 8
    odds = [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0]
    sel = ft.tp62_decide(_cands(p, odds), IDENT)
    assert sel is not None and sel["k"] <= 5
    assert sel["odds"] == sorted(sel["odds"], reverse=True)


def test_parse_payoffs():
    raw = json.dumps({"単勝": {"numbers": "3", "payout": "1,170"}, "馬連": [{"numbers": "3-8", "payout": "9,200"}, {"numbers": "8 - 3", "payout": "100"}]})
    p = ft.parse_payoffs(raw)
    assert p["win"] == {3: 1170.0} and p["umaren"] == {(3, 8): 100.0} and not p["broken"]
    broken = ft.parse_payoffs(json.dumps({"単勝": {"numbers": "7 - 97", "payout": "x"}}))
    assert "win" in broken["broken"]
    assert ft.parse_payoffs("not json")["broken"] == {"win", "umaren"}


def test_blend_matches_tp36_edge():
    edge = pytest.importorskip("src.scripts.data.tp36_edge")
    rng = np.random.default_rng(1)
    pw = rng.dirichlet(np.ones(8))
    q = rng.dirichlet(np.ones(8))
    d = pd.DataFrame({"race_id": "r", "p_win": pw, "q_win": q})
    assert np.allclose(ft.blend_win(pw, q, 0.3, 0.7), edge.blend(d, 0.3, 0.7).to_numpy())


def test_tp59_decide():
    g = pd.DataFrame({"horse_number": [1, 2, 3], "p_win": [0.30, 0.30, 0.05], "win_odds": [8.0, 8.0, 3.0],
                      "jockey_id": ["j1", "j2", "j1"], "trainer_id": ["t1", "t1", "t1"]})
    b = ft.tp59_decide(g, {"j1"}, {"t1"})
    assert list(b["horse_number"]) == [1]                      # 2 番は騎手が条件外、3 番は EV<1.5


# ── 合成データでの通し ──
def _setup(tmp_path: Path, monkeypatch):
    n = 8
    nums = list(range(1, n + 1))
    odds = [2.5, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0, 45.0]
    shut = pd.DataFrame({"race_id": "202608010101", "date": "2026-10-24", "start_time": "10:05", "horse_number": nums, "horse_id": [f"h{i}" for i in nums],
                         "jockey_id": [f"j{i}" for i in nums], "trainer_id": [f"t{i}" for i in nums]})
    p_win = np.array([0.10, 0.12, 0.30, 0.15, 0.12, 0.10, 0.07, 0.04])
    p_win = p_win / p_win.sum()
    pred = pd.DataFrame({"race_id": "202608010101", "horse_id": [f"h{i}" for i in nums], "p_win": p_win, "p_top2": np.clip(p_win * 1.9, 0, 0.95), "p_top3": np.clip(p_win * 2.6, 0, 0.97)})
    tdir = tmp_path / "tables" / "2026"
    tdir.mkdir(parents=True)
    shut.to_parquet(tdir / "race_shutuba_flat.parquet")
    pd.DataFrame({"race_id": "202608010101", "horse_number": nums, "win_odds": odds}).to_parquet(tdir / "race_odds_flat.parquet")
    pairs = [(a, b) for a in nums for b in nums if a < b]
    pd.DataFrame({"race_id": "202608010101", "odds_type": "umaren", "pair": [list(p) for p in pairs], "odds": [float(40 + 7 * i) for i, _ in enumerate(pairs)]}).to_parquet(tdir / "race_pair_odds_flat.parquet")
    pd.DataFrame({"race_id": ["202608010101"], "date": ["2026-10-24"], "payoff": [json.dumps({"単勝": {"numbers": "3", "payout": "600"}, "馬連": {"numbers": "3-4", "payout": "1,000"}})]}).to_parquet(tdir / "race_result_flat.parquet")
    pdir = tmp_path / "pred_v4" / "20261024"
    pdir.mkdir(parents=True)
    pred.to_parquet(pdir / "202608010101.parquet")
    c59 = {"name": "t59", "jockey_pos": ["j3"], "trainer_pos": ["t3"], "backtest": {"2026": {"roi": 100.0}}}
    c62 = {"name": "t62", "knots": {"umaren": IDENT}, "backtest_unselected": {"2026": {"roi": 1.0, "race_hit_rate": 0.2, "pred_race_hit": 0.24}}}
    (tmp_path / "c59.json").write_text(json.dumps(c59))
    (tmp_path / "c62.json").write_text(json.dumps(c62))
    (tmp_path / "ft.json").write_text(json.dumps({"blend": {"a_model": 0.3, "b_market": 0.7}}))
    monkeypatch.setattr(ft, "CFG59", tmp_path / "c59.json")
    monkeypatch.setattr(ft, "CFG62", tmp_path / "c62.json")
    monkeypatch.setattr(ft, "CFG_FT", tmp_path / "ft.json")
    monkeypatch.setattr(ft, "TABLES", tmp_path / "tables")
    return {"v4": tmp_path / "pred_v4", "v5": tmp_path / "pred_v5"}


def test_track_is_idempotent_and_settles(tmp_path, monkeypatch):
    pred_dirs = _setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    s1 = ft.run_track(["20261024"], out, ("v4",), ("final",), None, ft.TABLES, pred_dirs, verbose=False)
    led1 = (out / "ledger.jsonl").read_text()
    assert s1["added"] == 2                                            # tp59 と tp62 の 2 行
    rows = [json.loads(x) for x in led1.splitlines()]
    r59 = next(r for r in rows if r["rule"] == "tp59")
    assert r59["n_bets"] == 1 and r59["bets"][0]["horse_number"] == 3 and r59["return_yen"] == 600.0 and r59["hit"]   # 3 番: EV = 0.30×6 = 1.8 ≥ 1.5、単勝 600 円
    s2 = ft.run_track(["20261024"], out, ("v4",), ("final",), None, ft.TABLES, pred_dirs, verbose=False)
    assert s2["added"] == 0 and (out / "ledger.jsonl").read_text() == led1     # 再実行しても履歴は変わらない（追記のみ・冪等）
    card = ft.scorecard(ft.load_ledger(out / "ledger.jsonl"), ft.load_rules())
    c59 = next(c for c in card["cards"] if c["rule"] == "tp59")
    assert c59["bets"] == 1 and c59["roi_pct"] == pytest.approx(600.0)


def test_unsettled_race_not_recorded(tmp_path, monkeypatch):
    pred_dirs = _setup(tmp_path, monkeypatch)
    res = tmp_path / "tables" / "2026" / "race_result_flat.parquet"
    pd.DataFrame({"race_id": ["202608010101"], "date": ["2026-10-24"], "payoff": [None]}).to_parquet(res)
    s = ft.run_track(["20261024"], tmp_path / "out", ("v4",), ("final",), None, ft.TABLES, pred_dirs, verbose=False)
    assert s["added"] == 0 and s["skipped"].get("結果がまだ") == 1


def test_hash_change_is_detected(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    ok, diff = ft.check_hashes(out)
    assert ok and not diff
    assert ft.check_hashes(out)[0]
    c59 = json.loads((tmp_path / "c59.json").read_text())
    c59["jockey_pos"].append("jx")
    (tmp_path / "c59.json").write_text(json.dumps(c59))
    ok2, diff2 = ft.check_hashes(out, write=False)
    assert not ok2 and diff2 == ["c59.json"]
    # 参考値の追記（reference_sd）では変わらない
    (tmp_path / "c59.json").write_text(json.dumps({**c59, "jockey_pos": ["j3"]}))
    cfg = json.loads((tmp_path / "ft.json").read_text())
    cfg["reference_sd"] = {"tp59": 3.0}
    (tmp_path / "ft.json").write_text(json.dumps(cfg))
    assert ft.check_hashes(out, write=False)[0]


def test_needed_races():
    assert ft.needed_races(1.0, 2.0) is None
    assert ft.needed_races(1.3, 3.0) == pytest.approx((1.645 * 3.0 / 0.3) ** 2)
