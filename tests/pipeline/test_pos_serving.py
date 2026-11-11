"""位置取り予想の推論（pos_serving）が、学習の履歴特徴量（pos_model）と同じ値を作ること（小さな合成データ）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.inference import pos_serving as ps
from src.scripts.data import pos_model as pm


def _runs(n_days: int = 12, per_day: int = 3, field: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day_horses = rng.choice(30, size=field * per_day, replace=False)      # 同じ日に同じ馬は 1 走だけ
        for r in range(per_day):
            rid = f"2025{d:02d}{r:02d}"
            horses = day_horses[r * field:(r + 1) * field]
            for i, h in enumerate(horses):
                rows.append({"race_id": rid, "date": pd.Timestamp("2025-01-04") + pd.Timedelta(days=7 * d), "venue": "東京", "surface": "芝" if r % 2 == 0 else "ダート",
                             "distance": 1600 + 200 * (r % 2), "field_size": field, "horse_number": i + 1, "horse_id": f"h{h}", "sex_age": "牡4",
                             "jockey_weight": 56.0, "jockey_id": f"j{h % 5}", "trainer_id": f"t{h % 4}", "weight": 480.0 + h, "weight_change": 0.0,
                             "grade": "1勝クラス", "track_condition": "良", "sire": "", "dam_sire": "",
                             "pos_first": float(rng.uniform(0, 1)), "pos_last": float(rng.uniform(0, 1)), "pos_second": float(rng.uniform(0, 1)),
                             "f3_dev": float(rng.normal())})
    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    df["dist_band"] = pd.cut(df["distance"], [0, 1300, 1700, 2100, 9999], labels=False).astype(float)
    return df.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)


def test_card_features_match_training_columns() -> None:
    df = _runs()
    last = df["date"].max()
    for c in ("oi_1f", "oi_3f", "oi_4f", "oi_hill", "oi_eval", "oi_days"):
        df[c] = np.nan
    ref = df.copy()
    h = pm.horse_history(ref)
    ref = pd.concat([ref, h], axis=1)
    m = float(ref["pos_first"].mean())
    ref["j_pf"] = pm.asof_mean(ref, "jockey_id", k=30.0, m=m)
    ref["t_pf"] = pm.asof_mean(ref, "trainer_id", m=m)
    ref["s_pf"] = pm.asof_mean(ref, "sire", k=50.0, m=m)
    ref["ds_pf"] = pm.asof_mean(ref, "dam_sire", k=50.0, m=m)
    ref["_ck"], ref["_ckg"] = pm.course_gate_keys(ref)
    ref["cg_pf"] = pm.asof_mean(ref, "_ckg", k=50.0, m=m) - pm.asof_mean(ref, "_ck", k=50.0, m=m)
    ref = ref[ref["date"] == last]

    consts = {"m_pf": m, "oikiri": {"q": {c: [0.0, 100.0] for c in ("oi_1f", "oi_3f", "oi_4f")}, "ms": {c: {"0.0|2025": [1.0, 1.0]} for c in ("oi_1f", "oi_3f", "oi_4f")}}}
    state = ps.PosState.from_runs(df, consts, before=last)
    card = df[df["date"] == last].drop(columns=["pos_first", "pos_last", "pos_second", "f3_dev"])
    got = ps.card_features(card, state, None, blank_pedigree=False, fill_weight=False).reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    assert list(got["horse_id"]) == list(ref["horse_id"])
    cols = ["h_pf_w5", "h_pf_1", "h_pf_2", "h_pf_3", "h_pf_sd5", "h_pf_min5", "h_pl_3", "h_p2_3", "h_f3d_1", "h_f3d_2", "h_f3d_3", "h_es_3", "h_runs", "h_days",
            "h_dist_chg", "h_surf_chg", "h_pf_same", "j_pf", "t_pf", "s_pf", "ds_pf", "cg_pf"]
    for c in cols:
        a, b = got[c].astype(float).to_numpy(), ref[c].astype(float).to_numpy()
        assert np.allclose(a, b, equal_nan=True, atol=1e-9), c


def test_state_ignores_same_day_and_later_runs() -> None:
    df = _runs()
    consts = {"m_pf": 0.5, "oikiri": {"q": {}, "ms": {}}}
    day = df["date"].unique()[5]
    s = ps.PosState.from_runs(df, consts, before=pd.Timestamp(day))
    assert s.last_date < pd.Timestamp(day)
    assert int(s.sums["jockey_id"]["count"].sum()) == int((df["date"] < pd.Timestamp(day)).sum())


def test_missing_weight_filled_with_last_weight() -> None:
    df = _runs()
    consts = {"m_pf": 0.5, "oikiri": {"q": {c: [0.0, 100.0] for c in ("oi_1f", "oi_3f", "oi_4f")}, "ms": {c: {"0.0|2025": [1.0, 1.0]} for c in ("oi_1f", "oi_3f", "oi_4f")}}}
    last = df["date"].max()
    state = ps.PosState.from_runs(df, consts, before=last)
    card = df[df["date"] == last].drop(columns=["pos_first", "pos_last", "pos_second", "f3_dev"]).assign(weight=np.nan, weight_change=np.nan)
    f = ps.card_features(card, state, None)
    prev = df[df["date"] < last].sort_values("date").groupby("horse_id")["weight"].last()
    exp = f["horse_id"].map(prev)
    assert np.allclose(f["wt"].astype(float), exp.astype(float), equal_nan=True)
    assert (f.loc[exp.notna(), "wt_chg"] == 0).all()
    assert ps.card_features(card, state, None, fill_weight=False)["wt"].isna().all()


def _consts() -> dict:
    return {"m_pf": 0.5, "oikiri": {"q": {c: [0.0, 100.0] for c in ("oi_1f", "oi_3f", "oi_4f")}, "ms": {c: {"0.0|2025": [1.0, 1.0]} for c in ("oi_1f", "oi_3f", "oi_4f")}}}


def test_rejects_state_with_same_day_or_later_runs() -> None:
    import pytest

    df = _runs()
    last = df["date"].max()
    state = ps.PosState.from_runs(df, _consts())                       # before なし = 最終日までの走が入る
    card = df[df["date"] == last].drop(columns=["pos_first", "pos_last", "pos_second", "f3_dev"])
    with pytest.raises(ValueError):
        ps.card_features(card, state, None)


def test_rejects_card_without_required_columns() -> None:
    import pytest

    df = _runs()
    last = df["date"].max()
    state = ps.PosState.from_runs(df, _consts(), before=last)
    card = df[df["date"] == last].drop(columns=["pos_first", "pos_last", "pos_second", "f3_dev", "grade"])
    with pytest.raises(KeyError):
        ps.card_features(card, state, None)


def test_missing_going_filled_and_pedigree_blanked() -> None:
    df = _runs()
    last = df["date"].max()
    state = ps.PosState.from_runs(df, _consts(), before=last)
    card = df[df["date"] == last].drop(columns=["pos_first", "pos_last", "pos_second", "f3_dev"]).assign(track_condition=np.nan)
    f = ps.card_features(card, state, None)
    assert (f["going"].astype(str) == "良").all()
    named = card.assign(track_condition="良", sire="ディープ", dam_sire="キング")
    a = ps.card_features(named, state, None)
    b = ps.card_features(card.assign(track_condition="良"), state, None)
    assert np.allclose(a["s_pf"].astype(float), b["s_pf"].astype(float)) and np.allclose(a["ds_pf"].astype(float), b["ds_pf"].astype(float))


def test_other_columns_match_training_with_missing_labels_and_json_consts() -> None:
    import json

    df = _runs()
    rng = np.random.default_rng(1)
    df.loc[rng.random(len(df)) < 0.15, "pos_first"] = np.nan
    last = df["date"].max()
    for c in ("oi_1f", "oi_3f", "oi_4f", "oi_hill", "oi_eval", "oi_days"):
        df[c] = np.nan
    consts = json.loads(json.dumps(_consts()))
    state = ps.PosState.from_runs(df, consts, before=last)
    card = df[df["date"] == last].drop(columns=["pos_first", "pos_last", "pos_second", "f3_dev"])
    got = ps.card_features(card, state, None, blank_pedigree=False, fill_weight=False).reset_index(drop=True)
    assert list(got.columns) == ["race_id", "horse_id"] + pm.FEATURES
    assert (got["debut"] == (got["h_runs"] == 0).astype(int)).all()

