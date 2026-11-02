"""めぐコラム『エピファネイアは早熟？』の集計（src/research/columns/early_maturing_bloodline.py）。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.columns import early_maturing_bloodline as em


def test_short_name_drops_the_latin_part():
    assert em.short_name("ハービンジャーHarbinger(英)") == "ハービンジャー"
    assert em.short_name("ヘニーヒューズHenny Hughes(米)") == "ヘニーヒューズ"
    assert em.short_name("Into Mischief") == "Into Mischief"
    assert em.short_name("キズナ") == "キズナ"


def test_benjamini_hochberg():
    q = em.bh([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])           # p×n/順位 を、後ろから累積最小にしたもの
    assert all(qq >= pp for qq, pp in zip(q, [0.01, 0.04, 0.03, 0.5]))


def test_dumps_puts_flat_lists_on_one_line_and_round_trips():
    col = {"body": [{"type": "table", "rows": [["a", "1,000", "x（q<0.001）"]], "series": [{"values": [0.1, None, 0.3]}]}]}
    text = em.dumps(col)
    assert json.loads(text) == col
    assert '"values": [0.1, null, 0.3]' in text and '["a", "1,000", "x（q<0.001）"]' in text


def test_horses_compares_the_same_horse_young_and_old():
    rows = []
    for hid, sire, sex, young, old in [("2018100001", "A", "牡", [1, 1, 0, 0], [0, 0]), ("2018100002", "B", "牝", [0, 0], [1, 0]),
                                       ("2018100003", "A", "牡", [1, 0], [])]:       # 3 頭目は 4 歳以上を走っていないので入らない
        for k, t in enumerate(young):
            rows.append({"horse_id": hid, "sire": sire, "sex2": "牝" if sex == "牝" else "牡・セン", "old": 0, "top3": t, "res3": t - 0.25, "weight": 450 + k, "birth": 2018})
        for k, t in enumerate(old):
            rows.append({"horse_id": hid, "sire": sire, "sex2": "牝" if sex == "牝" else "牡・セン", "old": 1, "top3": t, "res3": t - 0.25, "weight": 460 + k, "birth": 2018})
    h = em.horses(pd.DataFrame(rows))
    assert list(h.index) == ["2018100001", "2018100002"]
    assert h.loc["2018100001", "d_top3"] == -0.5 and h.loc["2018100002", "d_top3"] == 0.5
    assert h.loc["2018100001", "grow"] == 460.5 - 451.5
