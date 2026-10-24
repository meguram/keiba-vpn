import math

from src.pipeline.inference import growth_index as G


def _params():
    feats = G.group_features({"stage": "3歳前半"})
    groups = {g: {"names": list(feats[g]), "mean": [0.0] * len(feats[g]), "coef": [0.0] * len(feats[g])} for g in G.GROUPS}
    return {"groups": groups, "intercept": 0.0, "thresholds": [-2.0, -0.5, 0.5, 2.0], "reason_min_points": 0.3}


def test_stage_of_and_levels():
    assert G.stage_of(2, 9) == "2歳前半" and G.stage_of(2, 10) == "2歳後半" and G.stage_of(3, 6) == "3歳前半" and G.stage_of(3, 7) == "3歳後半"
    assert G.stage_of(4, 1) == "4歳" and G.stage_of(6, 1) == "5歳以上" and G.stage_of(1, 5) is None and G.stage_of(None, 5) is None
    assert [G.stage_of_index(v, [-2, -0.5, 0.5, 2]) for v in (-3, -1, 0, 1, 3)] == [0, 1, 2, 3, 4]


def test_group_features_missing_stays_none_and_flags():
    f = G.group_features({"stage": "4歳", "weight_change": 6, "rest_days": 70})
    assert f["weight"]["dw10"] == 0.6 and f["weight"]["dev10"] is None and f["weight"]["miss_dev"] == 1.0 and f["weight"]["miss_dw"] == 0.0
    assert f["weight"]["dw10_young"] == 0.0 and f["weight"]["dw10_long"] == 0.6
    assert f["stage"]["stage=4歳"] == 1.0 and f["stage"]["stage=3歳前半"] == 0.0
    assert f["oikiri"]["miss_tl"] == 1.0 and f["ped"]["miss_ped"] == 1.0


def test_contributions_zero_when_all_missing_and_linear_otherwise():
    p = _params()
    c = G.contributions({"stage": "3歳前半"}, p)
    assert all(abs(v) < 1e-12 for v in c.values())
    # 体重の係数だけ立てる: dw10 の係数 2、平均 0.1 → 前走比 +6kg(=0.6) で 2*(0.6-0.1)
    i = p["groups"]["weight"]["names"].index("dw10")
    p["groups"]["weight"]["coef"][i] = 2.0
    p["groups"]["weight"]["mean"][i] = 0.1
    assert math.isclose(G.contributions({"stage": "3歳前半", "weight_change": 6}, p)["weight"], 1.0)
    # 欠損は平均で埋める＝寄与 0
    assert G.contributions({"stage": "3歳前半"}, p)["weight"] == 0.0


def test_growth_index_label_reasons_and_coverage():
    p = _params()
    p["groups"]["weight"]["coef"][p["groups"]["weight"]["names"].index("dev10")] = 3.0
    r = G.growth_index({"stage": "3歳前半", "weight_dev": 10, "weight_change": 4}, p)
    assert r["index"] == 3.0 and r["level"] == 4 and r["label"] == "大きく成長"
    assert r["reasons"] and r["reasons"][0]["group"] == "weight" and "+10kg" in r["reasons"][0]["text"]
    assert r["coverage"] == {"weight": True, "oikiri": False, "ped": False}


def test_ped_inputs_uses_previous_stage_transition():
    sire = {"banse": 0.5, "wtgain": -1.0, "didx": {"3歳前半→3歳後半": 0.7}, "dwt": {"3歳前半→3歳後半": 1.5}}
    p = G.ped_inputs("3歳後半", sire, None)
    assert p["sire_didx"] == 0.7 and p["sire_dwt"] == 1.5 and p["sire_banse"] == 0.5 and p["dam_sire_banse"] is None
    assert G.ped_inputs("2歳前半", sire, None)["sire_didx"] is None


def test_growth_index_model_contributions_sum_to_prediction():
    import numpy as np
    import pytest

    lgb = pytest.importorskip("lightgbm")
    names = G.feature_names()
    assert len(names) == 47 and names[: len(G.CTRL_NAMES)] == G.CTRL_NAMES
    params = {"level_mean": 60.0, "rest_median": 35.0, "thresholds": [-2.0, -0.5, 0.5, 2.0], "reason_min_points": 0.0}
    base = {"stage": "3歳前半", "weight_change": 4, "weight_dev": 6, "rest_days": 40, "level": 58.0, "n_valid_prior": 4, "stale_prev": 0,
            "distance": 1600, "surface": "芝", "sex": "牡", "f_t1z_vs_self": -0.5, "oikiri_eval": 3.0, "comment_pos": 1.0, "comment_neg": 0.0, "has_training": 1.0}
    rng = np.random.default_rng(0)
    X = np.array([G.feature_vector(base, params)] * 400, dtype=float)
    X = X + rng.normal(0, 0.5, X.shape) * np.isfinite(X)
    y = X[:, names.index("oikiri:f_t1z_vs_self")] * -3 + rng.normal(0, 0.1, 400)
    m = lgb.LGBMRegressor(n_estimators=20, num_leaves=5, min_child_samples=10, verbose=-1).fit(X, y)
    r = G.growth_index_model(base, params, m.booster_)
    pred = float(m.booster_.predict(np.asarray([G.feature_vector(base, params)]))[0])
    assert abs(r["index"] - round(pred, 2)) < 0.011
    bias = float(m.booster_.predict(np.asarray([G.feature_vector(base, params)]), pred_contrib=True)[0][-1])
    assert abs(sum(r["contributions"].values()) + bias - r["index"]) < 0.06          # ctrl + 4 要素 + バイアス = 予測（各項を丸めた誤差の範囲）
    assert set(r["contributions"]) == {"stage", "weight", "oikiri", "ped", "ctrl"} and r["label"] in G.LABELS
    assert r["coverage"]["oikiri"] is True
