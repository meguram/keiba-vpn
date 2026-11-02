"""めぐコラム『血統で狙う』の集計（src/research/columns/sire_roi.py）。"""

from __future__ import annotations

import json

import numpy as np

from src.research.columns import compare_series as cs
from src.research.columns import sire_roi as sr


def test_payouts_parse_single_dead_heat_and_commas():
    p = json.dumps({"単勝": {"numbers": "12", "payout": "1,230", "popularity": "5"},
                    "複勝": [{"numbers": "12", "payout": "130"}, {"numbers": "1", "payout": "180"}]}, ensure_ascii=False)
    assert sr._payouts(p, "単勝") == {12: 12.3}
    assert sr._payouts(p, "複勝") == {12: 1.3, 1: 1.8}
    dh = json.dumps({"単勝": [{"numbers": "4", "payout": "180"}, {"numbers": "7", "payout": "150"}]}, ensure_ascii=False)
    assert sr._payouts(dh, "単勝") == {4: 1.8, 7: 1.5}            # 同着は 2 頭とも払戻
    assert sr._payouts(dh, "複勝") == {}                           # 発売なし


def test_bootstrap_interval_contains_mean_and_narrows_with_n():
    rng = np.random.default_rng(0)
    small = np.array([0.0] * 90 + [10.0] * 10)
    big = np.tile(small, 10)
    lo, hi = sr.boot_ci(small, rng, b=500)
    lo2, hi2 = sr.boot_ci(big, rng, b=500)
    assert lo < 1.0 < hi and lo2 < 1.0 < hi2
    assert hi2 - lo2 < hi - lo
    assert sr.boot_ci(np.array([1.0]), rng) is None


def test_article_keeps_the_chart_compare_series_updates():
    # compare_series.py は sire-roi の「主要種牡馬の産駒 vs それ以外の産駒 勝率」を同じタイトルで更新する。記事の中（折り畳みの中でもよい）に残っていること
    text = (cs.COLUMNS_DIR / "sire-roi.json").read_text(encoding="utf-8")
    assert cs._find_chart(text, "主要種牡馬の産駒 vs それ以外の産駒 勝率") is not None
