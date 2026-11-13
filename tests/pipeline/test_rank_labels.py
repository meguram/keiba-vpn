"""rank_recipes のラベル・損失と、rank_bakeoff_eval の回収率・NDCG を小さな例の手計算と突き合わせる。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.pipeline.models import rank_recipes as rr
from src.scripts.data import rank_bakeoff_eval as ev


def test_topk_relevance_ties_below_k():
    fin = [1, 2, 3, 4, 5, 6, 12, np.nan, 0]
    assert rr.topk_relevance(fin, 5).tolist() == [5, 4, 3, 2, 1, 0, 0, 0, 0]
    assert rr.topk_relevance(fin, 3).tolist() == [3, 2, 1, 0, 0, 0, 0, 0, 0]


def test_topk_binary_and_capped():
    fin = [1, 3, 4, 5, 6, 9, np.nan]
    assert rr.topk_binary(fin, 3).tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert rr.topk_binary(fin, 5).tolist() == [1, 1, 1, 1, 0, 0, 0]
    assert rr.capped_finish(fin, 6).tolist() == [1, 3, 4, 5, 6, 6, 6]


def test_label_gain():
    assert rr.label_gain("exp", 5) == [0, 1, 3, 7, 15, 31]
    assert rr.label_gain("linear", 5) == [0, 1, 2, 3, 4, 5]


def _pl_by_hand(s, order):
    """着順 order（インデックスの列）の Plackett-Luce の負の対数尤度。"""
    rem = list(range(len(s)))
    nll = 0.0
    for i in order:
        nll += math.log(sum(math.exp(s[j]) for j in rem)) - s[i]
        rem.remove(i)
    return nll


def test_plackett_luce_topk_matches_hand():
    torch = pytest.importorskip("torch")
    s = [0.5, -0.2, 1.3, 0.0, 0.7, -1.0]
    fin = [3, 1, 2, 5, 4, 6]                         # 着順 1→index1, 2→2, 3→0, 4→4, 5→3
    score = torch.tensor([s + [0.0]], dtype=torch.float64)   # 7 頭目はパディング
    finish = torch.tensor([fin + [99]], dtype=torch.float64)
    mask = torch.tensor([[True] * 6 + [False]])
    got3 = rr.plackett_luce_topk_loss(score, finish, mask, k=3).item()
    got5 = rr.plackett_luce_topk_loss(score, finish, mask, k=5).item()
    assert got3 == pytest.approx(_pl_by_hand(s, [1, 2, 0]), abs=1e-9)
    assert got5 == pytest.approx(_pl_by_hand(s, [1, 2, 0, 4, 3]), abs=1e-9)
    # k 着より下の馬どうしの並び（スコアの入れ替え）は損失に影響しない
    s7 = s + [0.2]
    fin7 = torch.tensor([fin + [7]], dtype=torch.float64)
    mask7 = torch.ones(1, 7, dtype=torch.bool)
    base = rr.plackett_luce_topk_loss(torch.tensor([s7], dtype=torch.float64), fin7, mask7, k=5).item()
    s7b = list(s7)
    s7b[5], s7b[6] = s7b[6], s7b[5]                  # 6 着と 7 着のスコアを入れ替える
    swapped = rr.plackett_luce_topk_loss(torch.tensor([s7b], dtype=torch.float64), fin7, mask7, k=5).item()
    assert swapped == pytest.approx(base, abs=1e-12)

def test_listnet_target():
    torch = pytest.importorskip("torch")
    s = [0.1, 0.2, 0.3]
    score = torch.tensor([s], dtype=torch.float64)
    finish = torch.tensor([[2.0, 1.0, 9.0]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    got = rr.listnet_topk_loss(score, finish, mask, k=2).item()
    lse = math.log(sum(math.exp(x) for x in s))
    tgt = [1 / 3, 2 / 3, 0.0]                        # 関連度 1・2・0 を比例で正規化
    want = -sum(t * (x - lse) for t, x in zip(tgt, s))
    assert got == pytest.approx(want, abs=1e-9)


def test_ndcg_at():
    fin = np.array([1, 2, 3, 4, 5, 6], float)
    assert ev.ndcg_at(fin, fin) == pytest.approx(1.0)
    pred = np.array([2, 1, 3, 4, 5, 6], float)       # 1・2 着を入れ替え
    disc = 1 / np.log2(np.arange(2, 7))
    ideal = (2.0 ** np.array([5, 4, 3, 2, 1]) - 1) @ disc
    got = (2.0 ** np.array([4, 5, 3, 2, 1]) - 1) @ disc
    assert ev.ndcg_at(pred, fin) == pytest.approx(got / ideal)


def _keys(rid, hn, fin, year=2022):
    return pd.DataFrame({"race_id": rid, "horse_id": [f"h{h}" for h in hn], "horse_number": hn, "finish_position": fin, "year": year})


def test_race_table_roi_and_dead_heat():
    rid = "R1"
    hn = [1, 2, 3, 4, 5]
    fin = [2, 1, 3, 3, 5]                            # 3・4 番が 3 着同着
    keys = _keys(rid, hn, fin)
    # 予測: 3 番 > 2 番 > 1 番 > 4 番 > 5 番
    oof = pd.DataFrame({"race_id": rid, "horse_id": [f"h{h}" for h in hn], "p_win": [0.2, 0.3, 0.35, 0.1, 0.05]})
    pay = {(rid, "tansho"): {"2": 300}, (rid, "fukusho"): {"2": 120, "1": 150, "3": 200, "4": 250},
           (rid, "umaren"): {"1-2": 800}, (rid, "wide"): {"1-2": 300, "2-3": 400, "2-4": 500, "1-3": 600, "1-4": 700}}
    t = ev.race_table(oof, keys, pay)
    r = t.iloc[0]
    assert r["top1_win"] == 0 and r["top1_place"] == 1 and r["top3_in3"] == 3   # 上位 3 頭 = 3・2・1 番（着順 3・1・2）
    assert r["ret_tansho"] == 0
    assert r["ret_fukusho"] == 200                   # 3 番の複勝
    assert r["ret_umaren"] == 0                      # 2-3・1-3 は外れ（馬連は 1-2）
    assert r["ret_wide"] == 400 + 600                # 2-3・1-3 の両方が当たり（3 着同着で 4 番の組も払戻表にある）
    s = ev.summarize(t)
    assert s["roi_fukusho"] == pytest.approx(2.0) and s["roi_wide"] == pytest.approx(5.0) and s["n_pay"] == 1


def test_race_table_skips_races_without_all_payoffs():
    keys = _keys("R2", [1, 2, 3], [1, 2, 3])
    oof = pd.DataFrame({"race_id": "R2", "horse_id": ["h1", "h2", "h3"], "p_win": [0.5, 0.3, 0.2]})
    pay = {("R2", "tansho"): {"1": 150}, ("R2", "fukusho"): {"1": 110}}
    t = ev.race_table(oof, keys, pay)
    assert not bool(t.iloc[0]["has_pay"])
    assert t.iloc[0]["top1_win"] == 1


def test_load_payoff_normalizes_pairs(tmp_path):
    p = tmp_path / "payoff.parquet"
    pd.DataFrame({"race_id": [1, 1, 1, 1], "bet_type": ["umaren", "wide", "tansho", "sanrentan"],
                  "combo": ["10-2", "3-11", "4", "1>2>3"], "payout": [500, 200, 300, 9000]}).to_parquet(p)
    pay = ev.load_payoff(p)
    assert pay[("1", "umaren")] == {"2-10": 500}
    assert pay[("1", "wide")] == {"3-11": 200}
    assert ("1", "sanrentan") not in pay
