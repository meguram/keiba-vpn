"""◎ を固定した馬券のパターン（src/research/bet_patterns）の買い目と精算の確かめ（作りもののレース 1 つ）。"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.bet_patterns import data as D
from src.research.bet_patterns import search as S
from src.scripts.data import bet_cycles as bc

ROOT = Path(__file__).resolve().parents[2]
RID = "202301010101"
NOF = (0.0, 0.0, None, 1.0, None)       # 閾値なし（全頭が候補）


def _race() -> bc.Race:
    pw = np.array([0.30, 0.20, 0.15, 0.10, 0.09, 0.07, 0.05, 0.04])
    hn = np.arange(1, 9)
    pw, p3 = bc.normalize_probs(pw, None)
    p2 = bc.p2_from(pw, p3, None)
    odds = np.array([2.0, 4.0, 6.0, 9.0, 12.0, 20.0, 30.0, 50.0])
    pop = pd.Series(odds).rank(method="min").to_numpy()
    fin = np.array([2, 3, 1, 4, 5, 6, 7, 8], float)            # ◎ になる 3 番（index 2）が 1 着、1 番が 2 着、2 番が 3 着
    r = np.array([-0.2, 0.0, 0.5, -0.1, 0.1, -0.3, 0.0, -0.5])   # 2 番人気以下・勝率 5% 以上で r 最大 = 3 番
    q = pw / np.exp(r)
    pay = {"tansho": {"3": 500}, "fukusho": {"3": 200, "1": 120, "2": 150}, "umaren": {"1-3": 800},
           "wide": {"1-3": 300, "2-3": 400, "1-2": 200}, "umatan": {"3>1": 1500}, "sanrenpuku": {"1-2-3": 1000},
           "sanrentan": {"3>1>2": 5000}}
    return bc.Race(RID, "2023-01-05", hn, pw, p3, odds, pop, fin, pay, p2, q / q.sum(), r)


def _rows() -> D.Rows:
    body = json.loads((ROOT / "config/bet_cycles/frozen_20261117_r2.json").read_text(encoding="utf-8"))
    r = _race()
    feats = pd.DataFrame([{"date": r.date, "field": 8, "c_class": "未勝利", "q_r_max": 4}])
    p = bc.Period("t", [r], feats, np.ones((1, 1)), {})
    frames = {(RID, h): h for h in range(1, 9)}
    waku = {RID: {"1-3": 700}}
    return D.build_rows(p, body, "a2", frames, waku)


def _pp() -> S.Prep:
    coefs = {k: {"coef": [0.0, 1.0, 0.0]} for k in ("fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan", "wakuren")}
    return S.Prep(_rows(), coefs)


def _play(pp, sid, sh, **kw):
    c, r, p = S.play_rows(pp, {"s": sid, "f": None if sid in ("tan", "fuku") else NOF, "sh": sh, "ev": None, **kw})
    return float(c[0]), float(r[0]), float(p[0])


def test_honmei_is_frozen_rule():
    rows = _rows()
    assert rows.meta["H_num"].tolist() == [3]
    assert rows.counts["in_setting"] == 1


def test_singles_and_pairs():
    pp = _pp()
    assert _play(pp, "tan", (0, 0)) == (100, 500, 1)
    assert _play(pp, "fuku", (0, 0)) == (100, 200, 1)
    # 相手は複勝率（p3）の順: 1 番・2 番・4 番…
    assert _play(pp, "wide_n", (2, 2)) == (200, 700, 2)
    assert _play(pp, "um_n", (1, 1))[1] == 800          # 連対率の順の 1 位は 1 番
    assert _play(pp, "ut_1", (2, 2)) == (200, 1500, 2)
    assert _play(pp, "ut_2", (7, 7))[1] == 0
    assert _play(pp, "ut_m", (2, 2)) == (400, 1500, 4)
    assert _play(pp, "waku", (1, 1)) == (100, 700, 1)
    assert _play(pp, "um_box", (2, 2)) == (300, 800, 3)
    assert _play(pp, "wide_box", (2, 2)) == (300, 900, 3)   # 3 頭のボックスで 3 つとも当たり


def test_trio_and_trifecta():
    pp = _pp()
    assert _play(pp, "trio_1", (7, 7)) == (comb(7, 2) * 100, 1000, comb(7, 2))
    assert _play(pp, "trio_f", (1, 3)) == (200, 1000, 2)     # ◎-1 番-{2 番,4 番}
    assert _play(pp, "trio_f", (2, 5)) == (700, 1000, 7)     # 2×3＋1
    assert _play(pp, "trio_box", (2, 2)) == (100, 1000, 1)
    assert _play(pp, "tri_f", (1, 3)) == (200, 5000, 2)      # 3→1→{2,4}
    assert _play(pp, "tri_f", (3, 10)) == (3 * 6 * 100, 5000, 18)
    assert _play(pp, "tri_2", (7, 7))[1] == 0
    assert _play(pp, "tri_m", (2, 2)) == (600, 5000, 6)


def test_skip_and_stakes():
    pp = _pp()
    assert _play(pp, "trio_1", (7, 7), skip={"cap": 10})[0] == 0          # 21 点は上限 10 で見送り
    c, r, p = _play(pp, "trio_1", (3, 3), stake="payout", skip={})
    assert c >= 300 and p == 3
    # ◎ の勝率 × 単勝 6 倍 < 1 なので Kelly は買わない。期待値の下限・合成オッズの見送りも効く
    assert _play(pp, "tan", (0, 0), stake="kelly", skip={})[0] == 0
    assert _play(pp, "tan", (0, 0), skip={"synth": 7.0})[0] == 0
    assert _play(pp, "tan", (0, 0), skip={"synth": 5.0})[0] == 100


def test_kelly_buys_positive_edge():
    pp = _pp()
    pp.E["tan"] = pp.E["tan"] * 3                  # 単勝 18 倍とみなす → 期待値 > 1
    c, r, p = _play(pp, "tan", (0, 0), stake="kelly", skip={})
    assert c > 0 and c % 100 == 0 and c <= 500 and r == c * 5


def test_dead_heat_and_versions():
    """同着（3連複の当たりが 2 組）と、版（k=2）の平均。"""
    body = json.loads((ROOT / "config/bet_cycles/frozen_20261117_r2.json").read_text(encoding="utf-8"))
    r = _race()
    r.pay["sanrenpuku"] = {"1-2-3": 1000, "1-3-4": 2000}
    feats = pd.DataFrame([{"date": r.date, "field": 8, "c_class": "未勝利", "q_r_max": 4}] * 2)
    p = bc.Period("t", [r, r], feats, np.ones((1, 1)), {}, k=2)
    rows = D.build_rows(p, body, "a2", {(RID, h): h for h in range(1, 9)}, {})
    coefs = {k: {"coef": [0.0, 1.0, 0.0]} for k in ("fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan", "wakuren")}
    pp = S.Prep(rows, coefs)
    assert rows.meta["ver"].tolist() == [0, 1]
    c, ret, _ = S.play_rows(pp, {"s": "trio_1", "f": NOF, "sh": (7, 7), "ev": None})
    assert ret.tolist() == [3000, 3000]
    m = S.period_metrics(pp, [{"s": "trio_1", "f": NOF, "sh": (7, 7), "ev": None}], base=[{"s": "tan", "f": None, "sh": (0, 0), "ev": None}])
    assert m["bet_races"] == 1 and m["races"] == 1 and abs(m["roi"] - 3000 / 2100) < 1e-9
    assert abs(m["diff_roi"] - (3000 / 2100 - 5.0)) < 1e-9 and m["diff_lo"] <= m["diff_hi"]
    assert rows.arr["waku_sold"].tolist() == [0, 0]       # 枠連の払戻が無いレースは買わない


def test_metrics_and_robust():
    pp = _pp()
    m = S.period_metrics(pp, [{"s": "tan", "f": None, "sh": (0, 0), "ev": None}])
    assert m["bet_races"] == 1 and m["roi"] == 5.0 and m["honmei"] == [1.0, 0.0, 0.0, 0.0]
    e = S.eval_spec(pp, {"s": "tan", "f": None, "sh": (0, 0), "ev": None})
    assert S.robust(e["ys"], [2023])[0] == -np.inf                         # 年 100 レース未満は採らない
