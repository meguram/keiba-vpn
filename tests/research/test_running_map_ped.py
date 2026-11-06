import pandas as pd

from src.research.race import running_map_ped as rmp
from tests.pipeline.test_tp85_pedprior import _fit


def _setup(monkeypatch, with_models=True):
    pp, _, te, ped, _ = _fit()
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
