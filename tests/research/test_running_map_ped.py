import pandas as pd

from src.research.race import running_map_ped as rmp
from tests.pipeline.test_tp85_pedprior import _fit


def _setup(monkeypatch, with_models=True):
    pp, _, te, ped, _ = _fit()
    ref = {"surface_mean": {"芝": 1700.0, "ダート": 1500.0}, "pop": {"dirt_share": [0.0, 1.0], "dist_good": [-100.0, 100.0]}}
    monkeypatch.setattr(rmp, "_reference", lambda: ref)
    man = {"dirt_share": {"r2_obs_cal": 0.29, "top10": {"pred_diff": 0.4}, "bottom10": {"pred_diff": -0.3}}}
    monkeypatch.setattr(rmp, "_load", lambda: ({"dirt_share": pp}, man) if with_models else None)
    hs = list(te["horse_id"].head(3))
    monkeypatch.setattr(rmp, "_read_ped", lambda ids: ({h: {1: ped[h][1], 2: ped[h][2]} for h in ids if h in ped}, {h: {"sire": "父A", "damsire": "母父B"} for h in ids if h in ped}))
    return hs


def test_pedigree_for_adds_trait_values_and_clips(monkeypatch):
    hs = _setup(monkeypatch)
    out = rmp.pedigree_for([{"horse_id": h, "sex_age": "牝3"} for h in hs] + [{"horse_id": "unknown", "sex_age": "牡3"}])
    assert [t["key"] for t in out["traits"]] == ["dirt_share"]
    assert out["traits"][0]["r2"] == 0.29 and out["traits"][0]["ref"] == 0.35
    assert set(out["horses"]) == set(hs)
    h = out["horses"][hs[0]]
    mean, lo, hi = h["v"]["dirt_share"]
    assert 0 <= lo <= mean <= hi <= 1 and h["sex"] == "牝" and h["sire"] == "父A" and h["has_damsire"] is True


def test_pedigree_for_returns_none_without_models_or_pedigree(monkeypatch):
    hs = _setup(monkeypatch, with_models=False)
    assert rmp.pedigree_for([{"horse_id": hs[0], "sex_age": "牡3"}]) is None
    _setup(monkeypatch)
    assert rmp.pedigree_for([{"horse_id": "unknown", "sex_age": "牡3"}]) is None


def test_sex_of_defaults_to_male():
    assert rmp._sex_of({"sex_age": "セ5"}) == "セ" and rmp._sex_of({}) == "牡" and rmp._sex_of({"sex_age": "?"}) == "牡"


def test_score_is_percentile_in_half_points_and_can_invert():
    q = [float(x) for x in range(101)]
    assert rmp._score(50.0, q, False) == 5.0 and rmp._score(100.0, q, False) == 10.0 and rmp._score(-5.0, q, False) == 0.0
    assert rmp._score(90.0, q, True) == 1.0 and rmp._score(53.0, q, False) == 5.5
