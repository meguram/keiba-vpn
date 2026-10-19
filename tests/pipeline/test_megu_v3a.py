import numpy as np
import pandas as pd

from src.pipeline.megu_index import v3a as V


def _rows(race_id, n=8, base=96.0, day="2025-05-03", venue="05", surface="芝", dist=1600, bias=0.0, weights=None,
          cls=("1勝", "サラ系３歳以上 １勝クラス"), gen_text=None, horse_prefix=None):
    rows = []
    for i in range(n):
        rows.append({
            "race_id": race_id, "horse_id": f"{horse_prefix or race_id}_{i}", "date": day, "venue_code": venue,
            "surface": surface, "distance": dist, "direction": "右", "grade": cls[0], "race_name": "テスト",
            "race_class": gen_text or cls[1], "finish_time_sec": base + bias + 0.1 * i,
            "front_split_dev": 0.0, "jockey_weight_num": 55.0 if weights is None else weights[i],
            "dist_scale": dist / 2000.0, "delta_track_sec": 0.0, "out_of_range": False,
            "finish_pos_num": i + 1, "sex_age": "牡4",
        })
    return rows


def _spec(**kw):
    return V.V3ASpec(**{"loro_min_other_races": 2, "loro_min_horses": 6, **kw})


def test_gen_and_class():
    assert V.gen_of("サラ系２歳 未勝利") == "2歳" and V.gen_of("サラ系３歳以上 １勝クラス") == "3歳以上"
    assert V.gen_of("サラ系３歳 １勝クラス") == "3歳" and V.gen_of("サラ系４歳以上 オープン") == "4歳以上"
    assert V.gen_of("") == "他"
    assert V.class_of("L", "", "") == "OP" and V.class_of("G2", "", "") == "重賞" and V.class_of("新馬", "", "") == "新馬"


def test_loro_uses_only_other_races_of_the_same_day_venue_surface():
    df = pd.DataFrame(_rows("r1", bias=0.5) + _rows("r2", bias=1.0) + _rows("r3", bias=2.0)
                      + _rows("t1", surface="ダート", bias=9.0))   # 別の面は混ざらない
    sp = _spec()
    w = V.prepare(df, sp, beta_pace=0.0)
    resid = w["v3a_c0"] - 96.0 - 0.35   # 各レースの中央値は base+bias+0.35
    loro = V.loro_track(w, resid, sp)
    by = w.groupby("race_id").apply(lambda g: loro[g.index[0]])
    assert abs(by["r1"] - (1.0 + 2.0) / 2) < 1e-6 and abs(by["r2"] - (0.5 + 2.0) / 2) < 1e-6
    assert np.isnan(by["t1"])                       # ダートは 1 レースだけ → 馬場差を作らない（TSI にフォールバック）


def test_loro_needs_enough_other_races():
    df = pd.DataFrame(_rows("r1", bias=0.5) + _rows("r2", bias=1.0))
    sp = _spec()
    w = V.prepare(df, sp, beta_pace=0.0)
    loro = V.loro_track(w, w["v3a_c0"] - 96.0, sp)
    assert loro.isna().all()                        # 他レースが 1 つしか無い


def test_weight_is_standardized_to_55kg_for_all_horses():
    df = pd.DataFrame(_rows("r1", weights=[55, 57, 53, 55, 55, 55, 55, 55]))
    w = V.prepare(df, _spec(beta_weight=0.2), beta_pace=0.0)
    # 57kg は 2kg 重い → 2×0.2×(1600/2000)=0.32 秒の補正（速く見積もる）。性別にかかわらず基準は 55kg
    assert abs(w.loc[1, "v3a_weight"] - 0.32) < 1e-9 and abs(w.loc[2, "v3a_weight"] + 0.32) < 1e-9 and w.loc[0, "v3a_weight"] == 0


def test_pace_applies_to_turf_only_and_is_capped():
    d = _rows("a", surface="芝") + _rows("b", surface="ダート")
    df = pd.DataFrame(d)
    df["front_split_dev"] = 100.0
    w = V.prepare(df, _spec(pace_cap=3.0), beta_pace=0.8)
    assert (w.loc[w.surface == "芝", "v3a_pace"] == 3.0).all() and (w.loc[w.surface == "ダート", "v3a_pace"] == 0.0).all()


def test_par_table_shrinks_thin_cells_toward_the_parent():
    rows = pd.DataFrame({
        "surface": ["芝"] * 40 + ["芝"] * 2, "distance": [1600.0] * 42, "direction": ["右"] * 42,
        "v3a_gen": ["3歳以上"] * 40 + ["3歳"] * 2, "v3a_cls": ["1勝"] * 42,
        "c": [96.0] * 40 + [100.0] * 2,
    })
    tab = V.build_par_table(rows, V.V3ASpec(shrink_k=30.0))
    l1 = tab[(tab.level == "L1_dir_gen_cls") & (tab.gen == "3歳")].iloc[0]
    # 薄いセル(n=2, 平均 100)は親(L2: 3歳, 1勝 は同じセル→L3 の平均 96.19)へ強く縮む
    assert 96.0 < l1["par"] < 97.0 and l1["n"] == 2


def test_apply_model_builds_columns_and_uses_loro_track():
    sp = _spec()
    frames = []
    for k, b in enumerate([0.0, 0.0, 0.0, 0.0]):
        frames += _rows(f"r{k}", day=f"2024-05-0{k + 1}", horse_prefix=f"h{k}")
    ref = pd.DataFrame(frames)
    model = V.build_model(ref, V.V3ASpec(**{**sp.__dict__, "n_iter": 2}), beta_pace=0.0)
    day = pd.DataFrame(_rows("x1", day="2025-05-03", bias=1.0, horse_prefix="a")
                       + _rows("x2", day="2025-05-03", bias=1.0, horse_prefix="b")
                       + _rows("x3", day="2025-05-03", bias=1.0, horse_prefix="c"))
    out = V.apply_model(day, model, beta_pace=0.0)
    # 3 レースとも 1 秒遅い日 → 馬場差で打ち消され、2 着馬の指数はほぼ 50
    s2 = out[out["finish_pos_num"] == 2]["megu_index_v3a"]
    assert s2.between(47, 53).all(), s2.tolist()
    assert (out["v3a_track_src"] == "loro").all() and set(V.V3A_COLUMNS) <= set(out.columns)
    assert out["megu_v3a_version"].iloc[0] == V.V3A_VERSION_LABEL


def test_daily_equals_whole_period_for_the_same_day():
    sp = _spec()
    ref = pd.DataFrame(sum([_rows(f"r{k}", day=f"2024-05-0{k + 1}", horse_prefix=f"h{k}") for k in range(4)], []))
    model = V.build_model(ref, V.V3ASpec(**{**sp.__dict__, "n_iter": 2}), beta_pace=0.0)
    d1 = pd.DataFrame(sum([_rows(f"a{k}", day="2025-05-03", bias=0.7, horse_prefix=f"a{k}") for k in range(3)], []))
    d2 = pd.DataFrame(sum([_rows(f"b{k}", day="2025-05-04", bias=-0.4, horse_prefix=f"b{k}") for k in range(3)], []))
    both = V.apply_model(pd.concat([d1, d2], ignore_index=True), model, beta_pace=0.0)
    one = V.apply_model(d1, model, beta_pace=0.0)
    a = both.iloc[: len(d1)]["megu_index_v3a"].to_numpy()
    assert np.allclose(a, one["megu_index_v3a"].to_numpy(), atol=1e-9)


def test_model_roundtrip(tmp_path):
    ref = pd.DataFrame(sum([_rows(f"r{k}", day=f"2024-05-0{k + 1}", horse_prefix=f"h{k}") for k in range(4)], []))
    model = V.build_model(ref, V.V3ASpec(n_iter=1), beta_pace=0.0, meta={"x": 1})
    model.save(tmp_path / "m")
    back = V.V3AModel.load(tmp_path / "m")
    assert back.spec == model.spec and len(back.par) == len(model.par) and back.centers.keys() == model.centers.keys()


def test_soft_floor_keeps_order_and_compresses_the_slow_tail():
    x = pd.Series([-60.0, -10.0, 20.0, 30.0, 45.0, 70.0])
    y = V.soft_floor(x)
    assert (y.diff().dropna() > 0).all()                 # 順序は保つ
    assert y.iloc[3] == 30.0 and y.iloc[4] == 45.0       # 30 以上は変えない
    assert abs(y.iloc[0] - (30 + 0.1 * (-90))) < 1e-9    # 30 未満は傾き 0.1


def test_rel_and_abs_are_centered_per_surface():
    ref = pd.DataFrame(sum([_rows(f"r{k}", day=f"2024-05-0{k + 1}", horse_prefix=f"h{k}") for k in range(5)], []))
    model = V.build_model(ref, V.V3ASpec(n_iter=2), beta_pace=0.0)
    out = V.apply_model(ref, model, beta_pace=0.0)
    s2 = out[out["finish_pos_num"] == 2]
    assert abs(s2["megu_index_v3a"].mean() - 50.0) < 0.5 and abs(s2["megu_index_v3a_abs"].mean() - 50.0) < 0.5


def test_committed_model_loads_and_env_switch(monkeypatch):
    from pathlib import Path

    d = Path(__file__).resolve().parents[2] / "config" / "megu_v3a"
    m = V.V3AModel.load(d)
    assert set(m.par["level"]) == {"L1_dir_gen_cls", "L2_gen_cls", "L3_cls", "L4_pooled"}
    assert set(m.centers) == {"rel", "abs"} and {"芝", "ダート"} <= set(m.centers["rel"])
    monkeypatch.delenv("KEIBA_MEGU_V3A_DIR", raising=False)
    assert V.load_model_from_env() is None
    monkeypatch.setenv("KEIBA_MEGU_V3A_DIR", str(d))
    assert V.load_model_from_env() is not None
