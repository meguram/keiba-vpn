"""追走難度 v2 最終モデルの推論側（tracking_t6）と、学習側と共有した関数の確認。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.inference import pace_coherence as pcoh
from src.pipeline.inference import tracking_t6 as t6
from src.pipeline.models import tracking_difficulty_v2 as td2
from src.scripts.data import tp53_dataset as tp53
from src.scripts.data.tp54_dataset import run_features


def _results(n_races: int = 12, field: int = 10) -> pd.DataFrame:
    rows = []
    for r in range(n_races):
        for k in range(field):
            first = (k * 3 + r) % field + 1
            rows.append({"race_id": f"2025060101{r:02d}", "date": f"2025-0{1 + r // 6}-{1 + (r % 6) * 4:02d}", "venue": "中山", "surface": "芝", "distance": 1600,
                         "track_condition": "良", "field_size": field, "grade": "1勝クラス", "horse_number": k + 1, "horse_id": f"H{k}", "finish_position": (k + r) % field + 1,
                         "passing_order": f"{first}-{first}-{first}-{max(1, first - 1)}", "last_3f": 34.0 + 0.3 * ((k * 7 + r) % 9), "pace": "M", "odds": 5.0 + k})
    return pd.DataFrame(rows)


def test_actuals_from_results_filters_small_fields_and_builds_positions():
    r = pd.concat([_results(2, 10), _results(1, 6).assign(race_id="X")], ignore_index=True)
    act, race = tp53.actuals_from_results(r, 2025)
    assert set(act["race_id"]) == {"202506010100", "202506010101"}
    a = act[(act["race_id"] == "202506010100") & (act["horse_number"] == 1)].iloc[0]
    assert a["act_early_norm"] == 0.0 and np.isclose(a["act_late_norm"], 0.0)
    assert "act_l3f_dev" in act.columns


def test_history_a_and_b_shapes():
    r = _results(4, 10)
    ha, hb = t6.history_a(r), t6.history_b(r)
    assert list(ha.columns) == t6.HIST_A_COLS and len(ha) == 40
    assert pd.api.types.is_datetime64_any_dtype(ha["date"]) and len(hb) == 40 and hb["early"].notna().all()


def test_pf_columns_matches_apply_learned_pos(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for rid, n in (("R1", 12), ("R2", 9)):
        for k in range(n):
            rows.append({"race_id": rid, "horse_number": k + 1, "year": 2025, "pos_pred": float(rng.uniform(0, 1)) if k != 3 else np.nan})
    pp = pd.DataFrame(rows)
    p = tmp_path / "pp.parquet"
    pp.to_parquet(p)
    monkeypatch.setenv("TP53_POS_PRED", str(p))
    pfh = pp[["race_id", "horse_number"]].assign(pf_early=np.nan)
    pfr = pd.DataFrame({"race_id": ["R1", "R2"]})
    h, r = tp53.apply_learned_pos(2025, pfh, pfr)
    got = t6.pf_columns(pp.assign(pf_early=pp["pos_pred"]))
    assert np.allclose(got["pf_rank"].to_numpy(), h["pf_rank"].to_numpy(), equal_nan=True)
    want = r.set_index("race_id")
    per = got.assign(race_id=pp["race_id"].to_numpy()).groupby("race_id").first()
    for c in ("pf_front", "pf_stalker", "pf_known", "pf_pressure"):
        assert np.allclose(per[c].to_numpy(float), want.loc[per.index, c].to_numpy(float)), c


def test_run_features_asof_is_unchanged_by_the_target_actuals():
    """予測する行の実測を NaN にしても（推論）、実測が入っていても（学習）、その行の過去走の集計は同じ（先読みが無い）。"""
    cfg = pcoh.load_config()
    if not cfg:
        import pytest

        pytest.skip("当て込まれた係数（config/pace_coherence.json）が公開リポジトリに無い")
    keys = [k.split("|") for k in list(cfg["cohort_l3f"])]
    v, s, d, g = next(k for k in keys if k[1] == "芝" and k[3] == "良")
    r = _results(8, 10).assign(venue=v, surface=s, distance=int(d), track_condition=g)
    act, _ = tp53.actuals_from_results(r, 2025)
    act["date"] = pd.to_datetime(act["date"])
    act["race_id"], act["horse_id"] = act["race_id"].astype(str), act["horse_id"].astype(str)
    h = act.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    pe = {"features": ["const", "e", "e_dist", "e_turf"], "coef": [0.0, 0.5, 0.1, 0.2]}
    full, _ = run_features(h.copy(), cfg, pe)
    last = h["date"].max()
    cut = h.copy()
    m = cut["date"] == last
    cut.loc[m, ["act_early_norm", "act_late_norm", "act_l3f", "act_l3f_dev", "finish_position"]] = np.nan
    served, _ = run_features(cut, cfg, pe)
    for c in ("w_adj", "w_dev", "w_dev_front", "w_dev_rear", "w_early", "w_n", "w_nfront", "w_nrear", "prev_act_early_norm", "prev_c_dev", "rest_days", "gate_norm"):
        assert np.allclose(full.loc[m, c].to_numpy(float), served.loc[m, c].to_numpy(float), equal_nan=True), c


def test_past_drop_inputs_uses_only_previous_runs():
    h = pd.DataFrame({"horse_id": ["a"] * 5, "pf_pressure": 0.5, "z_hat": 1.0})
    y = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    out = td2.past_drop_inputs(h, y, pd.Series(0.25, index=h.index))
    assert np.isclose(out["hy1_x_front"].iloc[4], 4.0 * 0.75)
    assert np.isclose(out["hy3_x_press"].iloc[4], np.mean([2.0, 3.0, 4.0]) * 0.5)
    assert np.isnan(out["hy1_x_front"].iloc[0])


def test_ld_cross_formula():
    o = td2.ld_cross(pd.Series([1.0]), pd.Series([2.0]), pd.Series([0.25]))
    assert np.isclose(o["ld_front_x_f3"].iloc[0], 0.75) and np.isclose(o["ld_rear_x_q"].iloc[0], 0.5)


def _ent(race_id: str, n: int, fs: int, nan_pf: int | None = None) -> pd.DataFrame:
    e = pd.DataFrame({"race_id": race_id, "date": "2025-04-05", "horse_id": [f"H{k}" for k in range(n)], "horse_number": np.arange(1, n + 1), "venue": "中山", "surface": "芝",
                      "distance": 1600, "track_condition": "良", "field_size": fs, "pos_pred": 0.5, "pf_early": 0.5, "pf_rank": 1.0, "pf_front": 2, "pf_stalker": 3, "pf_pressure": 0.4,
                      "pf_known": n, "z_hat": 0.1, "w_adj": 0.0, "w_early": 0.5, "w_n": 3.0})
    for c in t6.PP_COLS:
        e[c] = 0.1
    if nan_pf is not None:
        e.loc[nan_pf, "pf_early"] = np.nan
    return e


def test_make_docs_skips_small_fields_and_horses_without_position():
    from types import SimpleNamespace

    model = SimpleNamespace(meta={"version": "t6"})
    ent = pd.concat([_ent("A", 10, 10, nan_pf=2), _ent("B", 6, 6)], ignore_index=True)
    pred = pd.DataFrame({"drain_sec": 0.1, "demand_sec": 0.0, "capacity_sec": 0.1, "ease_pct": 50.0}, index=ent.index)
    docs = t6.make_docs(model, ent, pred)
    assert [d["race_id"] for d in docs] == ["A"] and docs[0]["n_with_v2"] == 9 and "3" not in docs[0]["horses"]


def test_prep_drops_duplicates_and_missing_numbers_and_fills_only_missing_field_size():
    e = pd.concat([_ent("A", 9, 9), _ent("A", 1, 9)], ignore_index=True)
    e.loc[0, "horse_number"] = np.nan
    e.loc[1, "field_size"] = np.nan
    pos = e[["race_id", "horse_id", "pos_pred", *t6.PP_COLS]]
    out = t6._prep(e.drop(columns=["pos_pred", "pf_early", "pf_rank", "pf_front", "pf_stalker", "pf_pressure", "pf_known"]), pos)
    assert len(out) == 8 and out["horse_id"].is_unique
    assert (out["field_size"] == 8).sum() == 1 and (out["field_size"] == 9).sum() == 7


def test_stage_ensemble_t7_inputs_exclude_relative_columns_and_pp_columns_are_mapped(tmp_path):
    import json
    from types import SimpleNamespace

    from src.pipeline.features import serving_stage_rel as ssr
    from src.pipeline.features import serving_v6 as sv6

    members = [{"name": "a", "rel_src": ["x1", "pp_rank"]}, {"name": "b", "rel_src": ["x1", "x2"]}]
    fc = ["x1", "x2", "pp_rank", "rk_x1", "dm_x1", "dn_pp_rank", "rk_x2", "cat"]
    models = {"a": [SimpleNamespace(feature_cols=fc)], "b": [SimpleNamespace(feature_cols=fc)]}
    assert ssr.StageEnsembleT7(members, models).feature_cols == ["x1", "x2", "pp_rank", "cat"]

    d = tmp_path / "tracking_v2" / "2025-04-05"
    d.mkdir(parents=True)
    (d / "R1.json").write_text(json.dumps({"horses": {"1": {"horse_id": "h", "tracking_v2": {"drain_sec": 0.3}, "inputs": {"pp_rank": 2.0, "pp_pos": 0.4, "pf_early": 0.1}}}}))
    base = pd.DataFrame({"race_id": ["R1"], "horse_id": ["h"], "date": ["2025-04-05"]})
    out = sv6.stage_forecast_columns(base, ["EASE2_drain", "pp_rank", "pp_pos", "pp_ahead"], tmp_path)
    assert np.isclose(out["pp_rank"].iloc[0], 2.0) and np.isclose(out["pp_pos"].iloc[0], 0.4) and np.isnan(out["pp_ahead"].iloc[0])


def test_stage_forecast_columns_raises_when_no_doc_has_pp(tmp_path):
    import json

    import pytest

    from src.pipeline.features import serving_v6 as sv6

    d = tmp_path / "tracking_v2" / "2025-04-05"
    d.mkdir(parents=True)
    (d / "R1.json").write_text(json.dumps({"horses": {"1": {"horse_id": "h", "tracking_v2": {"drain_sec": 0.3}, "inputs": {"pf_early": 0.1}}}}))
    base = pd.DataFrame({"race_id": ["R1"], "horse_id": ["h"], "date": ["2025-04-05"]})
    with pytest.raises(KeyError):
        sv6.stage_forecast_columns(base, ["EASE2_drain", "pp_rank"], tmp_path)
