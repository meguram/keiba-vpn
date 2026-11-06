import numpy as np
import pandas as pd
import pytest

from src.pipeline.features.tp85_pedprior import PedPrior
from src.scripts.data import tp85_ped_traits as t85


def _nodes(sid):
    return [("S:" + sid, "sire")]


def _data(seed=0, n_sire=40, per=25):
    rng = np.random.default_rng(seed)
    eff = {f"s{i}": rng.normal(0, 1.0) for i in range(n_sire)}
    dam = {f"s{i}": rng.normal(0, 0.5) for i in range(n_sire)}
    rows, ped = [], {}
    for i in range(n_sire):
        for j in range(per):
            h = f"h{i}_{j}"
            d = f"s{(i * 7 + j) % n_sire}"
            sex = "牡" if j % 2 == 0 else "牝"
            y = eff[f"s{i}"] + 0.5 * dam[d] + (0.8 if sex == "牝" else 0.0) + rng.normal(0, 0.7)
            rows.append({"horse_id": h, "y": y, "v": 0.2, "sex": sex, "first_date": pd.Timestamp("2021-01-01")})
            ped[h] = {1: f"s{i}", 2: d}
    return pd.DataFrame(rows), ped, eff


def _fit(sexmode="fix"):
    t, ped, eff = _data()
    t = t.sample(frac=1.0, random_state=1).reset_index(drop=True)
    tr, te = t.iloc[:800].reset_index(drop=True), t.iloc[800:].reset_index(drop=True)
    o, st = t85.fit_variant("A2s", [1, 2], "shared", ped, _nodes, tr, te, sexmode=sexmode)
    meta = {"trait": "toy", "unit": "A2s", "positions": [1, 2], "mode": "shared", "sexmode": sexmode, "slope": 1.0}
    pp = PedPrior(meta, st)
    pp._nodes = _nodes
    return pp, st, te, ped, o


def test_predict_matches_fit():
    pp, st, te, ped, _ = _fit()
    got = pp.predict(ped, te[["horse_id", "sex"]])
    exp = st["m"] + st["s"] * st["te"]["mu"]
    assert np.allclose(got["mean"], exp)
    assert np.allclose(got["sd"], np.sqrt(st["te"]["var_true"]) * st["s"])
    assert (got["ped_n"] == 2).all()


def test_sex_shift_and_interval():
    pp, _, te, ped, _ = _fit()
    got = pp.predict(ped, te[["horse_id", "sex"]]).assign(sex=te["sex"].to_numpy())
    assert got[got["sex"] == "牝"]["mean"].mean() - got[got["sex"] == "牡"]["mean"].mean() > 0.5
    assert np.allclose(got["hi80"] - got["mean"], got["mean"] - got["lo80"])
    assert (got["hi80"] > got["lo80"]).all()


def test_unknown_pedigree_returns_wide_prior():
    pp, st, te, ped, _ = _fit()
    known = pp.predict(ped, te[["horse_id", "sex"]].head(5))
    one = pp.predict({}, pd.DataFrame({"horse_id": ["zz"], "sex": ["牡"]})).iloc[0]
    assert one["ped_n"] == 0
    assert one["sd"] >= known["sd"].min()
    assert one["mean"] == pytest.approx(st["sex_mean"]["牡"] + st["s"] * st["u"][0], abs=1e-6)


def test_slope_scales_deviation_only():
    pp, st, te, ped, _ = _fit()
    base = pp.predict(ped, te[["horse_id", "sex"]].head(20))
    pp.meta["slope"] = 0.5
    half = pp.predict(ped, te[["horse_id", "sex"]].head(20))
    assert np.allclose(half["mean"] - st["m"], 0.5 * (base["mean"] - st["m"]))
    assert np.allclose(half["sd"], base["sd"])


def test_save_load_roundtrip(tmp_path):
    pp, _, te, ped, _ = _fit()
    pp.save(tmp_path)
    q = PedPrior.load("toy", tmp_path)
    q._nodes = _nodes
    a = pp.predict(ped, te[["horse_id", "sex"]].head(30))
    b = q.predict(ped, te[["horse_id", "sex"]].head(30))
    assert np.allclose(a["mean"], b["mean"], atol=1e-4)
    assert np.allclose(a["sd"], b["sd"], atol=1e-4)


def test_calibration_slope_near_one_when_model_true():
    _, _, _, _, o = _fit()
    c = o["calib"]
    assert 0.7 < c["slope"] < 1.3
    assert c["top10"]["actual_diff"] > 0 > c["bottom10"]["actual_diff"]


def test_derived_combines_inputs_and_widens_without_pedigree():
    from src.pipeline.features.tp85_pedprior import Derived

    pp, _, te, ped, _ = _fit()
    d = Derived({"beta": [1.0, 2.0, -1.0], "tau_res": 0.25, "tau_pop": 1.0}, [pp, pp])
    hs = te[["horse_id", "sex"]].head(10)
    got, base = d.predict(ped, hs), pp.predict(ped, hs)
    assert np.allclose(got["mean"], 1.0 + base["mean"])
    assert np.allclose(got["sd"], 0.5)
    assert np.allclose(got["hi80"] - got["mean"], 1.2815515655446004 * 0.5)
    unk = d.predict({}, pd.DataFrame({"horse_id": ["zz"], "sex": ["牡"]})).iloc[0]
    assert unk["ped_n"] == 0 and unk["sd"] == pytest.approx(1.0)
