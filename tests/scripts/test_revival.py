import json

import numpy as np
import pandas as pd

from src.pipeline.features import feature_sets as fs
from src.scripts.data import revival_ledger as rl
from src.scripts.data import revival_scan as rs


def _frames(n_races=300, per=8, gain=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        s = rng.normal(size=per)
        p = np.exp(s) / np.exp(s).sum()
        w = rng.choice(per, p=p)
        for h in range(per):
            rows.append((f"r{r}", f"h{h}", int(h == w) * 1 + (0 if h == w else 2 + h), p[h]))
    d = pd.DataFrame(rows, columns=["race_id", "horse_id", "rank", "p"])
    lab = d[["race_id", "horse_id", "rank"]]
    base = d[["race_id", "horse_id"]].assign(p_win=0.125, p_top2=0.25, p_top3=0.375, year=2024)
    new = d[["race_id", "horse_id"]].assign(p_win=(0.125 * (1 - gain) + d["p"] * gain), p_top2=0.25, p_top3=0.375, year=2024)
    return base, new, lab


def test_paired_delta_negative_when_new_is_better():
    base, new, lab = _frames(gain=0.8)
    res = rs.paired_delta(base, new, lab, np.random.default_rng(1), 200)
    assert res["d_win"] < 0 and res["hi95_win"] < 0.01 and res["d_top2"] == 0.0
    assert res["n_races"] == 300


def test_paired_delta_zero_when_same():
    base, _, lab = _frames()
    res = rs.paired_delta(base, base.copy(), lab, np.random.default_rng(1), 100)
    assert abs(res["d_win"]) < 1e-12


def _info(periods, d, se=0.00012, t2=0.0, t3=0.0, hi=None):
    return {"periods": periods, "d_win_pool": d, "se_pool": se, "d_top2_pool": t2, "d_top3_pool": t3,
            "hi95_last": d + 1.645 * se if hi is None else hi, "d_win_last": d}


def test_levels_s2_requires_two_same_sign_periods():
    ok = rl.classify_levels(_info({"a": -0.0003, "b": -0.0002}, -0.00026))
    assert ok["S2"] and ok["S3"] and ok["S4"]
    mixed = rl.classify_levels(_info({"a": -0.0006, "b": 0.0001}, -0.00026))
    assert not mixed["S2"] and not mixed["S3"]
    single = rl.classify_levels(_info({"a": -0.0003}, -0.0003))
    assert not single["S2"] and not single["S3"] and single["S4"]


def test_levels_place_must_not_worsen():
    bad = rl.classify_levels(_info({"a": -0.0003, "b": -0.0003}, -0.0003, t3=0.0005))
    assert not any(bad[k] for k in ("S1", "S2", "S3", "S4", "S5"))


def test_levels_s1_matches_current_rule():
    lv = rl.classify_levels(_info({"a": -0.0004, "b": -0.0004}, -0.0004, se=0.00008))
    assert lv["S1"]
    lv2 = rl.classify_levels(_info({"a": -0.00015, "b": -0.00015}, -0.00015, se=0.0001))
    assert not lv2["S1"] and not lv2["S2"] and lv2["S3"]


def test_assign_status_priorities():
    base = {"status_override": "", "adopted": False, "leak": "", "market_dep": False, "kind": "group", "dup_known": False,
            "S2": True, "S3": True, "S4": True, "serving": "要整備"}
    assert rl.assign_status(pd.Series({**base, "adopted": True})) == "採用済み"
    assert rl.assign_status(pd.Series({**base, "leak": "x"})).startswith("不採用のまま(リーク")
    assert rl.assign_status(pd.Series({**base, "market_dep": True})).startswith("不採用のまま(市場")
    assert rl.assign_status(pd.Series({**base, "kind": "composition"})) == "構成(参考)"
    assert rl.assign_status(pd.Series(base)) == "要整備(R1)"
    assert rl.assign_status(pd.Series({**base, "serving": "あり"})) == "R1"
    assert rl.assign_status(pd.Series({**base, "S2": False})) == "要整備(R2)"
    assert rl.assign_status(pd.Series({**base, "S2": False, "S3": False, "S4": False})) == "不採用のまま"
    assert rl.assign_status(pd.Series({**base, "dup_known": True, "S2": False})).startswith("不採用のまま(既存")


def test_pooled_observations_picks_groups_and_skips_stale():
    scan = {"pooled": [
        {"file": "/x/training_pipeline_v4/tp32/stage1/results.json", "ctx": ["groups", "FIRST", "vs_V4"], "Δwin": -0.0003, "win_hi": 0.0, "Δtop2": 0.0, "Δtop3": 0.0},
        {"file": "/x/training_pipeline_v4/tp35/stage1_prelim_stale/results.json", "ctx": ["groups", "LG_cls", "vs_V4"], "Δwin": -0.01, "win_hi": 0.0},
        {"file": "/x/training_pipeline_v4/v6/eval/results.json", "ctx": ["pairs", "z:RB@e", "overall"], "variant": "z:RB@e vs d_CY_BEST", "Δwin": -0.0004, "win_hi": -0.0001},
        {"file": "/x/training_pipeline_v4/v6/eval/results_vs_z:RB+CB.json", "ctx": ["pairs", "z:OWNN", "overall"], "variant": "z:OWNN vs z:RB+CB", "Δwin": 0.0, "win_hi": 0.0001},
    ], "oof_years": []}
    obs = rl.pooled_observations(scan)
    assert set(zip(obs.theme, obs.group)) == {("tp32", "FIRST"), ("v6", "z:RB@e")}


def test_composition_and_bundle_detection():
    assert rl.is_composition("v6", "z:RB+CB") and rl.is_composition("v6", "z:RB@e") and not rl.is_composition("v6", "x_NJ")
    assert rl.is_bundle("ALL_PASSED") and rl.is_bundle("RL_ALL") and rl.is_bundle("BATCH[a+b]") and not rl.is_bundle("RL_A")


def test_feature_sets_config_shape():
    cfg = fs.load()
    assert set(cfg["sets"]) >= {"baseline", "revival_r1", "revival_r1r2", "revival_wide"}
    assert fs.resolve("baseline") == [] and fs.v6_eval_name("baseline") == "z:RB@e"
    r1, r12, wide = (set(fs.resolve(n)) for n in ("revival_r1", "revival_r1r2", "revival_wide"))
    assert r1 <= r12 and r12 <= wide and len(r1) >= 1
    assert fs.v6_eval_name("revival_r1").startswith("z:RB+") and fs.v6_eval_name("revival_r1").endswith("@e")
    assert "未知" in str(_raises(lambda: fs.resolve("nope")))


def _raises(fn):
    try:
        fn()
    except KeyError as exc:
        return exc
    return None


def test_external_groups_only_rev_names_and_paths_resolve():
    ext = fs.external_groups()
    assert ext and all(k.startswith("REV_") for k in ext)
    assert all(p.is_absolute() for p, _ in ext.values())


def test_sets_exclude_leak_market_and_composition():
    df = pd.read_csv(rl.LEDGER_CSV)
    cfg = fs.load()
    names = set(cfg["sets"]["revival_wide"]["add"])
    ids = {rl.short_name(i) for i in df.loc[df["leak"].fillna("").ne("") | df["market_dep"].fillna(False) | df["kind"].ne("group"), "id"]}
    assert not (names & ids)
    assert json.loads(json.dumps(cfg)) == cfg
