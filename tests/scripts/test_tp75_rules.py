import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp75_rules as T
from src.scripts.data.tp75_inventory import INVENTORY

ROOT = Path(__file__).resolve().parents[2]


def test_market_free_detects_market_columns():
    assert T.market_free("(sex=='牝') & (season=='summer')")
    assert T.market_free("(prev_fin<=3) & (lay_days>=90)")
    for bad in ("popularity<=3", "(prev_pop<=3)&(prev_fin>=10)", "win_odds>=10", "ret_win>0", "m_win>0.2"):
        assert not T.market_free(bad)


def test_library_is_market_free_and_matches_prereg():
    rules = T.library()
    assert all(r["market_free"] for r in rules)
    assert len({r["id"] for r in rules}) == len(rules)
    assert len(T.library.excluded) == 9 and all(not T.market_free(e["expr"]) for e in T.library.excluded)
    pre = json.loads((ROOT / "src/scripts/data/tp75_prereg.json").read_text(encoding="utf-8"))
    assert pre["counts"]["n_rules"] == len(rules)
    assert [r["id"] for r in pre["rules"]] == [r["id"] for r in rules]


def test_geld_n_counts_from_first_gelding_run():
    d = pd.DataFrame({
        "horse_id": ["a"] * 5 + ["b"] * 3 + ["c"] * 2,
        "date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-01-01", "2022-02-01", "2022-03-01", "2022-01-01", "2022-02-01"]),
        "race_id": [f"r{i}" for i in range(10)],
        "sex": ["牡", "牡", "セ", "セ", "セ", "牡", "牡", "牡", "セ", "セ"],
    })
    g = T.geld_n(d)
    assert np.isnan(g.iloc[0]) and np.isnan(g.iloc[1])
    assert g.iloc[2:5].tolist() == [1, 2, 3]
    assert g.iloc[5:8].isna().all()
    assert g.iloc[8:].isna().all()          # 最初からセ（去勢の時点が不明）


def _row(z_by_year: dict, mean=0.01, n=1000, races=500, col="e_m1"):
    def blk(z):
        return {"n": n, "races": races, "mean": mean, "se": mean / z if z else 1.0, "z": z}
    st = {"disc": {"e_m1": blk(z_by_year["disc"]), "e_m3": blk(z_by_year["disc"])}}
    for y in ("y2022", "y2023", "y2024"):
        st[y] = {"e_m1": blk(z_by_year[y]), "e_m3": blk(z_by_year[y])}
        st[y]["e_m1"]["mean"] = z_by_year[y] * 0.01
        st[y]["e_m3"]["mean"] = z_by_year[y] * 0.01
    return {"id": "x", "error": None, "race_level": False, "stats": st}


def test_adopt_tiers_and_replication():
    rows = [
        _row({"disc": 2.5, "y2022": 2.0, "y2023": 1.5, "y2024": 1.0}),       # A, 再現
        _row({"disc": 1.8, "y2022": 1.0, "y2023": 1.0, "y2024": -0.5}),      # B, 過半が同じ向き
        _row({"disc": 2.5, "y2022": 2.0, "y2023": -1.0, "y2024": -1.0}),     # 再現しない
        _row({"disc": 0.5, "y2022": 1.0, "y2023": 1.0, "y2024": 1.0}),       # 区間が 0 を跨ぐ
        _row({"disc": -2.5, "y2022": -2.0, "y2023": -1.0, "y2024": -1.0}),   # 消し A
    ]
    s = T.adopt(rows)
    a = [r["adopt"]["ev_buy"] for r in rows]
    assert a[0]["tier"] == "A" and a[0]["replicated"] and a[0]["adopted"]
    assert a[1]["tier"] == "B" and a[1]["adopted"]
    assert a[2]["tier"] == "A" and not a[2]["replicated"] and not a[2]["adopted"]
    assert a[3]["tier"] is None and not a[3]["adopted"]
    assert rows[4]["adopt"]["ev_kill"]["tier"] == "A" and rows[4]["adopt"]["ev_kill"]["adopted"] and not rows[4]["adopt"]["ev_buy"]["adopted"]
    assert s["ev_buy"]["adopted_A"] == 1 and s["ev_buy"]["adopted_B"] == 1


def test_bh_monotone():
    q = T.bh(np.array([0.001, 0.01, 0.04, 0.5]))
    assert (q <= 1).all() and q[0] <= q[1] <= q[2] <= q[3]


def test_logit_offset_matches_mean():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.4, 5000)
    y = (rng.uniform(size=5000) < np.clip(p * 1.3, 0, 1)).astype(float)
    d = T.logit_offset(p, y)
    q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + d)))
    assert abs(q.sum() - y.sum()) < 1e-3 and d > 0


def test_cluster_merges_overlapping_rules():
    d = pd.DataFrame({"year": [2022] * 100, "a": [1] * 50 + [0] * 50, "b": [1] * 48 + [0] * 52, "c": [0] * 70 + [1] * 30})

    def row(expr, z):
        return {"id": expr, "expr": expr, "stats": {}, "adopt": {"ev_buy": {"adopted": True, "z": z}}}
    cl = T.cluster_set(d, [row("a==1", 3.0), row("b==1", 2.5), row("c==1", 2.0)], "ev_buy")
    assert len(cl) == 2 and sorted(len(c["members"]) for c in cl) == [1, 2]


def test_inventory_covers_chapters_and_has_valid_values():
    chapters = {os.path.basename(f)[:-5] for f in glob.glob(str(ROOT / "docs/html/modeling/chapters/*.html")) + glob.glob(str(ROOT / "docs/html/modeling/*.html"))}
    covered = {os.path.basename(i["source"])[:-5] for i in INVENTORY if i["kind"] == "chapter"}
    missing = chapters - covered - {"rule_personas"}
    assert len(missing) <= 3, sorted(missing)      # 新しい章が増えたときは棚卸しに足す（取りこぼし防止）
    assert covered - chapters == set()
    assert {i["ingest"] for i in INVENTORY} <= {"recomputed", "excluded_market", "baseline", "none"}
    assert all(i["reason"] for i in INVENTORY)


def test_frozen_files_are_market_free_and_consistent():
    for name in ("config/tp75_frozen_rule_ev.json", "config/tp75_frozen_rule_place.json"):
        p = ROOT / name
        if not p.exists():
            continue
        spec = json.loads(p.read_text(encoding="utf-8"))
        assert spec["market_free"] is True and spec["version"] >= 1 and spec["history"]
        assert spec["n_rules"] == len(spec["rules"])
        for r in spec["rules"]:
            assert r["market_free"] and T.market_free(r["expr"]) and r["tier"] in ("A", "B") and r["outcome"] in ("win", "top3") and r["direction"] in ("buy", "kill")
