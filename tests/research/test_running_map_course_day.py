"""走法マップのコース別の 3 軸の地図（running_map_course）と、その日の馬場傾向（running_map_day）。"""

import numpy as np
import pandas as pd

from src.research.race import running_map_course as rmc
from src.research.race import running_map_day as rmd

KEYS = ["ts", "sustain", "debut", "gear", "endure", "front", "cr", "makuri"]


def test_score_is_one_for_top3_and_zero_from_8th():
    assert rmc.score_of(np.array([1, 2, 3, 4, 5, 6, 7, 8, 12])).round(2).tolist() == [1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0]


def test_expected_score_is_the_mean_over_positions():
    assert rmc.expected_score(np.array([5]))[0] == np.mean([1, 1, 1, 0.8, 0.6])
    assert rmc.expected_score(np.array([16]))[0] == (3 + 0.8 + 0.6 + 0.4 + 0.2) / 16


def _synthetic(n_races=120, field=12, seed=0, good=("endure", "debut", "gear")):
    """good の方面が高い馬ほど上位に来るレース群。"""
    rng = np.random.default_rng(seed)
    rows, Z = [], []
    w = np.array([2.0 if k == good[0] else 1.2 if k == good[1] else 0.7 if k == good[2] else 0.0 for k in KEYS])
    for r in range(n_races):
        z = rng.normal(size=(field, len(KEYS)))
        order = np.argsort(-(z @ w + rng.normal(scale=1.0, size=field)))
        pos = np.empty(field, int)
        pos[order] = np.arange(1, field + 1)
        for i in range(field):
            rows.append({"race_id": f"r{r:04d}", "horse_id": f"h{r}_{i}", "date": pd.Timestamp("2025-01-01"), "venue": "東京",
                         "surface": "ダート", "distance": 1600, "finish_position": pos[i], "n_field": field})
        Z.append(z)
    return pd.DataFrame(rows), np.vstack(Z)


def test_build_picks_the_axes_that_drive_finishing_and_lookup_falls_back_to_the_group():
    tab, Z = _synthetic()
    maps = rmc.build(tab, Z, ~np.isnan(Z), KEYS)
    cm = maps["courses"]["東京|ダート|1600"]
    assert cm["axes"] == ["endure", "debut", "gear"]
    assert cm["importance"][0]["sign"] == 1
    n = cm["extent"]["n"]
    assert len(cm["lift"]) == n * n and len(cm["sig"]) == n * n
    assert np.array(cm["A"]).shape == (3, 2)
    # 最も重要な方面の矢印が真上
    a0 = np.array(cm["A"])[0]
    assert abs(a0[0]) < 1e-6 and a0[1] > 0
    assert rmc.lookup(maps, "東京", "ダート", 1600) is cm
    assert rmc.lookup(maps, "中山", "ダート", 1600) is maps["groups"]["ダート|マイル"]
    assert rmc.lookup(maps, "中山", "芝", 2000) is None


def test_build_ignores_runs_before_since_and_small_fields():
    tab, Z = _synthetic(n_races=30)
    tab["date"] = pd.Timestamp("2023-06-01")
    maps = rmc.build(tab, Z, ~np.isnan(Z), KEYS)
    assert maps["courses"] == {} and maps["groups"] == {}


def test_earlier_race_ids_are_same_day_same_venue_and_before():
    assert rmd.earlier_race_ids("202605041004") == ["202605041001", "202605041002", "202605041003"]
    assert rmd.earlier_race_ids("202605041001") == []


def _day_inputs(strong=True, n_prev=6, field=14, seed=1):
    rng = np.random.default_rng(seed)
    cards, vals = {}, {}
    for k in range(1, n_prev + 1):
        rid = f"2026050410{k:02d}"
        ts = rng.normal(size=field)
        score = (3.0 * ts if strong else 0) + rng.normal(size=field)
        order = np.argsort(-score)
        pos = np.empty(field, int)
        pos[order] = np.arange(1, field + 1)
        ents = []
        for i in range(field):
            hid = f"{rid}_{i}"
            ents.append({"horse_id": hid, "finish_position": int(pos[i])})
            vals[hid] = {**{kk: rng.normal() for kk in KEYS}, "ts": ts[i], "going_dirt": np.nan}
        cards[rid] = {"surface": "ダート", "track_condition": "良", "entries": ents}
    cards["202605041003"]["surface"] = "芝"   # 芝のレースは数えない
    meta = [{"key": k, "label": k, "high": f"{k}高", "low": f"{k}低", "kind": "map"} for k in KEYS]
    meta.append({"key": "going_dirt", "label": "ダートタフ適正", "high": "乾", "low": "湿", "kind": "extra"})
    return cards, (lambda ids: pd.DataFrame(vals).T.reindex(ids)), meta


def test_day_trend_finds_a_strong_axis_and_skips_the_other_surface():
    cards, axes, meta = _day_inputs(strong=True)
    out = rmd.compute("202605041007", {"surface": "ダート"}, cards.get, axes, meta, None)
    assert out["n_races"] == 5 and out["status"] == "trend"
    assert out["trend"][0]["key"] == "ts" and out["trend"][0]["text"] == "ts高"
    assert next(a for a in out["axes"] if a["key"] == "going_dirt")["t"] is None   # 値の無い軸は判定しない


def test_day_trend_without_signal_rarely_reports_a_trend_and_too_few_races_reports_few():
    flagged = 0
    for seed in range(60):
        cards, axes, meta = _day_inputs(strong=False, seed=seed)
        out = rmd.compute("202605041007", {"surface": "ダート"}, cards.get, axes, meta, None)
        flagged += bool(out["trend"])
        assert out["status"] in ("none", "trend")
    assert flagged <= 8   # FDR 5% で、信号の無い日に傾向ありと出るのは 1 割未満
    cards, axes, meta = _day_inputs(strong=False, seed=0)
    few = rmd.compute("202605041003", {"surface": "ダート"}, cards.get, axes, meta, None)
    assert few["status"] == "few"
    assert rmd.compute("202605041001", {"surface": "ダート"}, cards.get, axes, meta, None) is None
