"""tp-51 のスタッキングのリーク防止と、特徴量グルーピングの網羅性のテスト。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scripts.data import tp51_groups as G
from src.scripts.data import tp51_stack as S


def _frame(n_races=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(8, 15))
        strength = rng.normal(size=n)
        fin = np.argsort(np.argsort(-(strength + rng.normal(scale=1.0, size=n)))) + 1
        for h in range(n):
            row = {"race_id": f"R{r:04d}", "horse_id": f"H{r}_{h}", "year": 2022 + (r % 3), "finish_position": int(fin[h]),
                   "field_size": n, "distance": 1200 + 200 * (r % 6), "surface_code": r % 2, "race_class_code": r % 4}
            for m in S.MEMBERS:
                z = strength[h] + rng.normal(scale=0.7)
                for t, k in (("win", 0.0), ("top2", 0.8), ("top3", 1.5)):
                    row[f"{m}__{t}"] = float(1 / (1 + np.exp(-(z - 2.2 + k))))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["year", "race_id"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def df():
    return _frame()


@pytest.mark.parametrize("name", ["S0_spec", "S3_lr_cross", "S4_lr_full", "S7_group_stack", "S8_segment"])
def test_scheme_does_not_use_validation_labels(df, name):
    """評価年の着順を並べ替えても、予測は変わらない（メタ学習は学習側の着順だけを使う）。"""
    tr = df[df["year"] < 2024].reset_index(drop=True)
    va = df[df["year"] == 2024].reset_index(drop=True)
    fn = S.SCHEMES[name]
    a = fn(tr, va)
    va2 = va.copy()
    va2["finish_position"] = np.random.default_rng(1).permutation(va2["finish_position"].to_numpy())
    b = fn(tr, va2)
    for t in S.TARGETS:
        assert np.allclose(a[t], b[t], atol=1e-12)


def test_forward_uses_only_earlier_years(df):
    """前進評価の学習側には、評価年以降の行が入らない。"""
    seen = {}

    def spy(tr, va):
        seen["train_years"] = sorted(tr["year"].unique())
        seen["valid_years"] = sorted(va["year"].unique())
        return {t: np.full(len(va), 0.2) for t in S.TARGETS}

    S.run_scheme(df, spy, "forward")
    assert seen["train_years"] == [2022, 2023] and seen["valid_years"] == [2024]


def test_loyo_excludes_validation_year(df):
    seen = []

    def spy(tr, va):
        seen.append((sorted(tr["year"].unique()), sorted(va["year"].unique())))
        return {t: np.full(len(va), 0.2) for t in S.TARGETS}

    S.run_scheme(df, spy, "loyo")
    for trn, vl in seen:
        assert set(trn).isdisjoint(vl)


def test_finalize_totals(df):
    va = df[df["year"] == 2024].reset_index(drop=True)
    out = S.finalize(va, {t: np.full(len(va), 0.3) for t in S.TARGETS})
    for t, tot in S.TOT.items():
        assert np.allclose(out.groupby("race_id")[f"p_{t}"].sum(), tot, atol=1e-6)


def test_feature_groups_cover_all_columns():
    """V4 の 1,062 列がすべてグループに入り、OTHER が無い。表（CSV）と assign が一致する。"""
    csv = Path(__file__).resolve().parents[2] / "src/scripts/data/tp51_feature_groups.csv"
    t = pd.read_csv(csv)
    assert len(t) == 1062 and t["column"].is_unique
    assert "OTHER" not in set(t["group"])
    assert dict(zip(t["column"], t["group"])) == G.assign(list(t["column"]))
    assert set(t["group"]) == set(G.group_columns(list(t["column"])))


def test_relative_columns_follow_base_group():
    a = G.assign(["f3_r", "f3_r_rz", "tl_f_t4z", "tl_f_t4z_rrank", "sr_perf_ewm2_rgap"])
    assert a["f3_r"] == a["f3_r_rz"] == "FIELD_PACE"
    assert a["tl_f_t4z"] == a["tl_f_t4z_rrank"] == "TL"
    assert a["sr_perf_ewm2_rgap"] == "HIST_SPEED"


def test_fast_params_scale_learning_rate_and_rounds():
    from src.scripts.data import tp51_group_members as gm

    lp = {"lgb": {"learning_rate": 0.03, "num_leaves": 17}, "n_rounds": 789}
    f = gm.lgb_params(lp, True)
    q = gm.lgb_params(lp, False)
    assert abs(f["lgb"]["learning_rate"] - 0.09) < 1e-12 and f["n_rounds"] == 263
    assert q["lgb"]["learning_rate"] == 0.03 and q["n_rounds"] == 789
    assert lp["lgb"]["learning_rate"] == 0.03          # 元の設定は変えない


def test_group_columns_partition_is_exact():
    cols = ["distance", "tl_f_days", "f3_r", "f3_r_rz", "o_win", "pd_sire_sd_res", "s_time_pred", "mtl_t_pos_early"]
    g = G.group_columns(cols)
    flat = [c for v in g.values() for c in v]
    assert sorted(flat) == sorted(cols) and len(set(flat)) == len(cols)
    assert g["STAGE_AUX"] == ["s_time_pred", "mtl_t_pos_early"]


def test_meta_module_matches_stack_scheme(df):
    """tp51_meta（推論側に載せる形）が、評価に使った S11e（現行 + クロス LR の L3、複勝は現行）と同じ予測を出す。"""
    from src.pipeline.models import tp51_meta as M

    tr = df[df["year"] < 2024].reset_index(drop=True)
    va = df[df["year"] == 2024].reset_index(drop=True)
    expect = S.SCHEMES["S11e_L3lr_win2"](tr, va)
    Ptr, Pva = S.member_arrays(tr), S.member_arrays(va)
    y = S._ys(tr)
    spec = M.fit_cross_lr(Ptr, y, tr["race_id"].to_numpy(), C=0.1)
    base = S.SCHEMES["S0_spec"](tr, va)
    got = M.apply_l3(spec, Pva, base, va["race_id"].to_numpy())
    for t in S.TARGETS:
        assert np.allclose(got[t], expect[t], atol=1e-9)
