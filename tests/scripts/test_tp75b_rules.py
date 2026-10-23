import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp75_rules as T1
from src.scripts.data import tp75b_rules as T2

ROOT = Path(__file__).resolve().parents[2]


def test_atoms_are_market_free_and_match_prereg():
    d = pd.DataFrame({"sire_main": ["Turn-To系", "Nasrullah系"], "sire_coarse": ["Halo系", "その他"], "sire_fine": ["Halo系", "Silver Hawk系"],
                      "ds_main": ["Turn-To系", None], "ds_coarse": ["Halo系", None], "ds_fine": ["Halo系", None], "dds_main": [None, None], "dds_coarse": [None, None]})
    specs = T2.atom_specs(d)
    assert all(T1.market_free(a["expr"]) for a in specs)
    pre = json.loads((ROOT / "src/scripts/data/tp75b_prereg.json").read_text(encoding="utf-8"))
    assert pre["counts"]["candidates"] > 0 and pre["design"]["max_order"] == 3
    assert all(T1.market_free(a["expr"]) for a in pre["atoms"])
    assert "不完全な 10 世代表" in pre["design"]["lineage_note"] and pre["design"]["rerun_when"]
    names = [a["name"] for a in pre["atoms"]]
    assert len(names) == len(set(names))


def test_enumerate_itemsets_apriori_and_dims():
    rng = np.random.default_rng(1)
    n = 4000
    rc = np.repeat(np.arange(n // 10), 10)
    A = rng.random(n) < 0.5
    B = rng.random(n) < 0.5
    C = rng.random(n) < 0.5
    A2 = ~A                                   # A と同じ次元（排他）
    M = np.stack([A, A2, B, C])
    dims = ["x", "x", "y", "z"]
    disc = np.ones(n, bool)
    en = T2.enumerate_itemsets(M, dims, rc, disc, 100, 20)
    cs = en["candidates"]
    assert (0, 1) not in cs and not any(0 in c and 1 in c for c in cs)       # 同じ次元は連言にしない
    assert any(len(c) == 3 for c in cs) and all(len(c) <= 3 for c in cs)
    assert en["stats"]["candidates"] == len(cs)
    # 既知のルールと同じ該当集合は除かれる
    import hashlib
    known = {hashlib.md5(np.packbits(B).tobytes()).digest()}
    en2 = T2.enumerate_itemsets(M, dims, rc, disc, 100, 20, known_hashes=known)
    assert (2,) not in en2["candidates"] and en2["stats"]["duplicates_dropped"] >= 1


def test_inc_z_detects_stronger_child():
    rng = np.random.default_rng(2)
    n = 20000
    d = pd.DataFrame({"rc": np.repeat(np.arange(n // 10), 10), "year": 2023})
    parent = rng.random(n) < 0.5
    child = parent & (rng.random(n) < 0.4)
    e = rng.normal(0, 1, n) * 0.3 + np.where(child, 0.1, 0.0)
    d["e_m1"] = e; d["e_m3"] = e; d["e_v1"] = e; d["e_v3"] = e; d["r1"] = 0.0; d["r3"] = 0.0; d["y1"] = 0.0; d["y3"] = 0.0
    d["p_win"] = 0.1; d["p_top3"] = 0.3
    ctx = T1.Ctx(d)
    assert T2.inc_z(ctx, child, parent, "e_m1", +1) > 3
    assert T2.inc_z(ctx, child, parent, "e_m1", -1) < -3
    flat = parent & (rng.random(n) < 0.4)
    assert abs(T2.inc_z(ctx, flat, parent, "e_m3", +1)) < 3.5
