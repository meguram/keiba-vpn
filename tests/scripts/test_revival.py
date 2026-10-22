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


def test_external_groups_only_rev_s76_s77_names_and_paths_resolve():
    ext = fs.external_groups()
    assert ext and all(k.startswith(("REV_", "S76_", "S77_")) for k in ext)
    assert all(p.is_absolute() for p, _ in ext.values())


def test_sets_exclude_leak_market_and_composition():
    df = pd.read_csv(rl.LEDGER_CSV)
    cfg = fs.load()
    names = set(cfg["sets"]["revival_wide"]["add"])
    ids = {rl.short_name(i) for i in df.loc[df["leak"].fillna("").ne("") | df["market_dep"].fillna(False) | df["kind"].ne("group"), "id"]}
    assert not (names & ids)
    assert json.loads(json.dumps(cfg)) == cfg


def test_tp76_sets_and_priorities():
    cfg = fs.load()
    p1, p12, allp = (cfg["sets"][n]["add"] for n in ("screen_p1", "screen_p1p2", "screen_all"))
    assert p1 == ["S76_COMPOSITE", "S76_D2", "S76_D5", "S76_D5r"]
    assert p12 == p1 + ["S76_A2", "S76_B3"] and allp[:6] == p12
    assert {"S76_E1", "S76_E2", "S76_D4", "S76_D6", "S76_C", "S76_F", "S76_G"} <= set(allp)
    combo = cfg["sets"]["revival_r1+screen_p1"]["add"]
    assert combo == cfg["sets"]["revival_r1"]["add"] + p1
    # 既存の構成は変えない
    assert cfg["sets"]["baseline"]["add"] == [] and len(cfg["sets"]["revival_r1"]["add"]) == 4


def test_unbuilt_composite_is_left_out_of_eval_name():
    cfg = fs.load()
    assert "S76_COMPOSITE" in fs.resolve("screen_p1", cfg) and "S76_COMPOSITE" not in fs.resolve("screen_p1", cfg, ready_only=True)
    assert "S76_COMPOSITE" not in fs.v6_eval_name("screen_p1", cfg) and "S76_D2" in fs.v6_eval_name("screen_p1", cfg)
    assert "S76_COMPOSITE" in cfg["unbuilt"]


def test_external_groups_include_s76_with_columns():
    ext = fs.external_groups()
    path, cols = ext["S76_D2"]
    assert path.name == "features.parquet" and cols == ["D_venue"]
    assert "S76_COMPOSITE" not in ext


def test_s76_column_groups_do_not_overlap():
    cfg = fs.load()
    seen: dict[str, str] = {}
    for n, g in cfg["groups"].items():
        if n.startswith("S76_") and g.get("cols") and g.get("path"):
            for c in g["cols"]:
                assert c not in seen, (c, seen.get(c), n)
                seen[c] = n


def test_false_positive_rates_recorded():
    m = fs.load()["false_positive_rate"]["measured"]
    assert m["S4"]["fpr"] == 0.025 and m["S1"]["fpr"] == 0.0 and m["S4"]["real_pass"] == 13


def test_tp76_rows_flags():
    df = pd.read_csv(rl.LEDGER_CSV)
    t = df[df["theme"] == "tp76"].set_index("group")
    assert t.loc["D2", "screen_priority"] == "優先1" and t.loc["A2", "screen_priority"] == "優先2" and t.loc["E1", "screen_priority"] == "優先3"
    assert t.loc["C1", "screen_priority"].startswith("束") and t.loc["A1", "screen_priority"] == "未割当"
    assert bool(t.loc["D5", "rerun_after_complete_10gen"]) and not bool(t.loc["D5r", "rerun_after_complete_10gen"])
    assert bool(t.loc["B3", "rerun_after_complete_10gen"]) and t.loc["E1", "serving_effort"] == "S" and t.loc["D2", "serving_effort"] == "L"
    assert "合成スコア" in t.index


def test_s4_3y_flag_for_ledger_groups():
    df = pd.read_csv(rl.LEDGER_CSV)
    assert "s4_3y" in df
    ok = df[df["s4_3y"] == True]  # noqa: E712
    assert (ok["n_periods"] >= 3).all()
