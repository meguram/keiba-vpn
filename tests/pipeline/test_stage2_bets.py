"""凍結した買い方（第 2 回 70ba29fa7b55）の当日の判定 stage2_bets: 凍結の hash の検査、見送りの理由の順、◎ の選び方、
バックテストの道具（bet_cycles.play）と同じ結果になること。手元に正解のファイル（リポジトリの外）があれば全件で突き合わせる。外部接続なし。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.inference import stage2_bets as sb

TRUTH = Path("/home/jovyan/work/retrain_out/bet_cycles/r2/truth_test2025_70ba29fa7b55.parquet")
CUT = 0.4917373690349473        # 凍結の qcuts.r_max の 2 つ目（q_r_max ≥ 3 で買う）


@pytest.fixture(scope="module")
def frozen():
    return sb.load_frozen()


def _race(r_top=1.0, n=10, pw_top=0.10, pop_top=4):
    """予測の順に並んだ 1 レース。index 2 の馬（pop_top 番人気・勝率 pw_top）の r が r_top、ほかは小さい。"""
    pw = np.linspace(0.25, 0.02, n)
    pw[2] = pw_top
    pw = pw / pw.sum()
    pop = np.arange(1, n + 1, dtype=float)
    pop[[2, pop_top - 1]] = pop[[pop_top - 1, 2]]
    r = np.full(n, -0.2)
    r[2] = r_top
    q = pw / np.exp(r)
    q = q / q.sum()
    odds = 0.8 / q
    return dict(horse_numbers=np.arange(1, n + 1), p_win=pw, q=q, r=r, pop=pop, odds=odds)


def test_frozen_hash_is_checked(frozen, tmp_path):
    assert frozen["hash"] == sb.FROZEN_HASH
    bad = dict(frozen)
    bad["qcuts"] = {**bad["qcuts"], "r_max": [0, 0, 0, 0]}
    p = tmp_path / "f.json"
    p.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(sb.FrozenMismatch):
        sb.load_frozen(p)


def test_buy_picks_r_max_horse_below_favourite(frozen):
    d = sb.decide(race_id="R1", grade="未勝利", frozen=frozen, **_race(r_top=1.0))
    assert d["status"] == "buy" and d["honmei"]["horse_number"] == 3
    assert d["bets"] == [{"bet_type": "tansho", "combo": [3], "yen": 100}]
    assert d["horse_marks"]["3"] == "◎" and d["frozen_hash"] == sb.FROZEN_HASH


@pytest.mark.parametrize("grade,want", [("新馬", "buy"), ("未勝利", "buy"), ("1勝", "class"), ("G1", "class"), ("", "class")])
def test_only_maiden_and_newcomer(frozen, grade, want):
    d = sb.decide(race_id="R1", grade=grade, frozen=frozen, **_race(r_top=1.0))
    assert (d["status"] if want == "buy" else d["skip_reason"]) == want


def test_skip_when_r_max_below_cut(frozen):
    d = sb.decide(race_id="R1", grade="未勝利", frozen=frozen, **_race(r_top=CUT - 1e-6))
    assert d["status"] == "skip" and d["skip_reason"] == "r_max" and d["q_r_max"] == 2
    d = sb.decide(race_id="R1", grade="未勝利", frozen=frozen, **_race(r_top=CUT))          # ちょうど区切りの値は買い（side="right"）
    assert d["status"] == "buy" and d["q_r_max"] == 3


def test_field_checked_first(frozen):
    d = sb.decide(race_id="R1", grade="1勝", frozen=frozen, **_race(n=4, r_top=CUT - 0.1))  # 頭数・r・クラスの全部が外れ → 頭数
    assert d["skip_reason"] == "field"
    d = sb.decide(race_id="R1", grade="1勝", frozen=frozen, **_race(r_top=CUT - 0.1))       # r とクラス → r（_skip の順）
    assert d["skip_reason"] == "r_max"


def test_favourite_and_low_prob_are_not_candidates(frozen):
    d = sb.decide(race_id="R1", grade="未勝利", frozen=frozen, **_race(r_top=2.0, pop_top=1))   # 1 番人気の r が大きいだけ → 例外（r ≥ 9）は無いので候補外
    assert d["status"] == "skip"
    d = sb.decide(race_id="R1", grade="未勝利", frozen=frozen, **_race(r_top=2.0, pw_top=0.04))  # 勝率 5% 未満は候補外
    assert d["status"] == "skip"


def test_horse_without_odds_is_excluded(frozen):
    x = _race(r_top=1.0)
    x["odds"][2] = np.nan
    x["pop"][2] = 99.0
    x["r"][2] = np.nan
    d = sb.decide(race_id="R1", grade="未勝利", frozen=frozen, **x)
    assert d.get("honmei", {}).get("horse_number") != 3


def test_no_prediction_and_skip_result(frozen):
    x = _race()
    x["p_win"][0] = np.nan
    assert sb.decide(race_id="R1", grade="未勝利", frozen=frozen, **x)["skip_reason"] == "no_prediction"
    s = sb.skip_result("no_odds_t60", race_id="R1")
    assert s["status"] == "skip" and "T-60" in s["skip_label"] and s["strategy"] == "stage2_r2"


def test_market_q_and_popularity():
    q = sb.market_q([2.0, 4.0, np.nan, 0])
    assert np.isclose(np.nansum(q), 1.0) and np.isnan(q[2]) and np.isnan(q[3])
    assert list(sb.popularity_from_q([0.1, 0.5, np.nan, 0.4])) == [3.0, 1.0, 99.0, 2.0]


@pytest.mark.skipif(not TRUTH.exists(), reason="正解のファイル（リポジトリの外）が無い")
def test_matches_backtest_truth(frozen):
    """bet-optimizer の正解（2025 年 test、T-60 の再現 5 通り＋確定）と 1 行ずつ同じ（見送りか・◎・r_max・段・頭数）。"""
    t = pd.read_parquet(TRUTH)
    inv = {"OP・L": "OP", "不明": "?"}
    bad = 0
    for row in t.itertuples():
        a = {c: np.array(json.loads(getattr(row, c)) if isinstance(getattr(row, c), str) else getattr(row, c), dtype=float)
             for c in ("hn", "p_win", "q", "r", "pop", "odds_used")}
        d = sb.decide(race_id=row.race_id, horse_numbers=a["hn"].astype(int), p_win=a["p_win"], q=a["q"], r=a["r"], pop=a["pop"],
                      odds=a["odds_used"], grade=inv.get(row.c_class, row.c_class), frozen=frozen)
        buy = d["status"] == "buy"
        rm = (d["r_max"] is None and np.isnan(row.r_max)) or (d["r_max"] is not None and abs(d["r_max"] - row.r_max) < 1e-6)
        ok = buy == bool(row.bet) and (not buy or d["honmei"]["horse_number"] == int(row.H_num)) and d["q_r_max"] == int(row.q_r_max) \
            and rm and d["field"] == row.field
        bad += not ok
    assert bad == 0
