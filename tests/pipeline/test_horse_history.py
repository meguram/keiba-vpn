import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.pipeline.features.horse_history.builder import (
    build_horse_history,
    feature_columns,
    permutation_check,
    truncation_check,
)
from src.pipeline.features.horse_history.common import apply_repair, build_repair_map, load_runs


def _synthetic_tables(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    horses = [f"H{i:03d}" for i in range(60)]
    skill = {h: rng.normal() for h in horses}
    per_year = {y: [] for y in range(2020, 2026)}
    for year in range(2020, 2026):
        for week in range(52):
            day = date(year, 1, 1) + timedelta(days=week * 7 + 5)
            day_horses = rng.choice(horses, size=20, replace=False)  # 同じ馬は同日に 2 走しない
            for r in range(2):
                field = day_horses[r * 10:(r + 1) * 10]
                score = np.array([skill[h] + rng.normal(scale=0.7) for h in field])
                order = (-score).argsort().argsort() + 1
                race_id = f"{year}05{week // 8 + 1:02d}{week % 8 + 1:02d}{r + 1:02d}"
                surface = str(rng.choice(["芝", "ダート"]))
                race_class = "" if (week % 10 == 0 and r == 1) else "サラ系３歳以上 １勝クラス"  # 空は障害レース
                dist = int(rng.choice([1200, 1600, 2000]))
                laps = (12.0 + rng.normal(scale=0.3, size=dist // 200)).round(1).tolist()
                laps[-3:] = [x + float(rng.normal(scale=0.4)) for x in laps[-3:]]
                for k, h in enumerate(field):
                    per_year[year].append({
                        "race_id": race_id, "date": day.isoformat(), "start_time": f"{10 + r}:05",
                        "venue_code": "05", "surface": surface, "race_class": race_class, "distance": dist, "direction": "左",
                        "track_condition": "良", "grade": str(rng.choice(["未勝利", "1勝", "2勝"])),
                        "pace": json.dumps({"first_half_3f": 35.0 + rng.normal(), "second_half_3f": 36.0 + rng.normal()}),
                        "finish_position": int(order[k]), "horse_number": k + 1, "horse_id": h, "sex_age": str(rng.choice(["牡3", "牝3", "牡4", "牝4"])),
                        "time_sec": float(95 + order[k] * 0.2), "passing_order": f"{k + 1}-{k + 1}-{order[k]}-{order[k]}",
                        "last_3f": float(34.0 + order[k] * 0.1 + rng.normal(scale=0.3)), "weight": float(rng.choice([450, 470, 490, 510])),
                        "jockey_id": f"J{k % 7}", "trainer_id": f"T{k % 5}", "jockey_weight": float(rng.choice([54, 55, 56, 57, 58])), "lap_times": json.dumps(laps),
                    })
    tables = tmp_path / "tables"
    for y, rows in per_year.items():
        (tables / str(y)).mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(tables / str(y) / "race_result_flat.parquet")
    la = tmp_path / "layer_a.parquet"
    pl.DataFrame({"race_id": ["x"], "date": ["2020-01-01"], "horse_id": ["H000"], "past_races": ["[]"]}).write_parquet(la)
    return tables, la


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hh")
    tables, la = _synthetic_tables(tmp)
    F, P = build_horse_history(la, tables)
    return tables, la, F, P


def test_features_cover_all_runs_and_families(built):
    tables, la, F, P = built
    assert F.height == load_runs(tables).height
    fam = feature_columns(F.columns)
    for k in ("A", "B", "C", "D", "E", "G", "H", "F1", "F2", "F3", "F5"):
        assert fam[k], k
    assert F["date"].dt.year().max() <= 2025


def test_first_run_has_no_history(built):
    _, _, F, _ = built
    first = F.sort(["horse_id", "date", "race_id"]).group_by("horse_id").first()
    assert (first["d_nprior"] == 0).all()
    assert first["a_l3fz_ewm2"].is_null().all()


def test_truncation_invariance(built):
    tables, la, F, _ = built
    runs, rm = load_runs(tables), build_repair_map(la)
    cols = sorted({c for v in feature_columns(F.columns).values() for c in v})
    rep = truncation_check(runs, rm, F, date(2023, 7, 1), cols)
    assert rep["rows"] > 0 and rep["mismatch"] == {}


def test_same_day_results_do_not_leak(built):
    tables, la, F, _ = built
    runs, rm = load_runs(tables), build_repair_map(la)
    cols = sorted({c for v in feature_columns(F.columns).values() for c in v})
    rep = permutation_check(runs, rm, F, date(2023, 7, 1), cols)
    assert rep["rows"] > 0 and rep["mismatch"] == {}


def test_on_time_repairs_corrupted_race(built):
    tables, la, _, _ = built
    runs, rm = load_runs(tables), build_repair_map(la)
    truth = apply_repair(runs, rm)
    rid = runs.filter(pl.col("date") >= date(2023, 10, 20))["race_id"][0]
    bad = runs.with_columns(
        pl.when(pl.col("race_id") == rid).then(pl.lit("105")).otherwise(pl.col("passing_order")).alias("passing_order"),
        pl.when(pl.col("race_id") == rid).then(94.0).otherwise(pl.col("last_3f")).alias("last_3f"),
        pl.when(pl.col("race_id") == rid).then(1.0).otherwise(pl.col("weight")).alias("weight"),
    )
    ot = (
        runs.filter(pl.col("race_id") == rid)
        .select("race_id", "horse_number", pl.col("last_3f").alias("ot_l3f"),
                pl.col("passing_order").alias("ot_passing_order"), pl.col("weight").alias("ot_weight"))
    )
    key = ["race_id", "horse_number"]
    cols = ["l3f", "wt", "pos_early", "pos_late"]
    want = truth.filter(pl.col("race_id") == rid).sort(key).select(cols)
    broken = apply_repair(bad, rm).filter(pl.col("race_id") == rid).sort(key).select(cols)
    fixed = apply_repair(bad, rm, ot).filter(pl.col("race_id") == rid).sort(key).select(cols)
    assert broken["l3f"].is_null().all() and broken["pos_early"].is_null().all()  # 補正なしでは壊れた値を使わない
    assert fixed.equals(want)


def test_rpci_is_race_level_and_pc_features_are_history_only(built):
    tables, la, F, _ = built
    runs = load_runs(tables)
    r = runs.filter(pl.col("rpci").is_not_null())
    assert r.height > 0 and r["rpci"].min() > 30 and r["rpci"].max() < 70
    assert r.group_by("race_id").agg(pl.col("rpci").n_unique()).get_column("rpci").max() == 1  # レース内で同じ値
    first = F.sort(["horse_id", "date", "race_id"]).group_by("horse_id").first()
    assert first["pc_pci_ewm2"].is_null().all() and (first["pc_n_reg0"] == 0).all()


def test_standard_rating_uses_grid_beta_and_is_history_only(built):
    _, _, F, P = built
    assert set(P["sr_beta"].values()) <= set(P["sr_beta_grid"]) | {0.10}
    assert P["elo_delta"] in (0.0, 5.0, 10.0, 20.0, 30.0)
    first = F.sort(["horse_id", "date", "race_id"]).group_by("horse_id").first()
    assert first["sr_perf_ewm2"].is_null().all() and (first["sr_n"] == 0).all()
    assert F["sr_exp"].is_not_null().mean() > 0.5
    # 負担が重いほど補正量(sr_load_pts)は小さい（負の方向）: 期待値は能力より低くなる
    d = F.filter(pl.col("sr_load_dev").is_not_null() & (pl.col("sr_load_dev").abs() > 1))
    heavy = d.filter(pl.col("sr_load_dev") > 0)["sr_load_pts"].mean(); light = d.filter(pl.col("sr_load_dev") < 0)["sr_load_pts"].mean()
    assert heavy < 0 < light


def test_hurdle_races_are_identified_by_blank_race_class(built):
    tables, _, _, _ = built
    runs = load_runs(tables)
    assert runs.filter(pl.col("surface") == "障").height > 0
    raw = pl.concat([pl.read_parquet(f) for f in sorted(tables.glob("*/race_result_flat.parquet"))])
    blank_ids = set(raw.filter(pl.col("race_class") == "")["race_id"].to_list())
    assert set(runs.filter(pl.col("surface") == "障")["race_id"].unique().to_list()) == blank_ids


def test_today_track_variant_uses_only_races_started_90_minutes_earlier(built):
    _, _, F, _ = built
    races = F.group_by("race_id").agg(pl.col("sr_var_n").first(), pl.col("sr_var_today").first(), pl.col("start_time").first(), pl.col("date").first())
    # 合成データは 1 日 2 レース（10:05 と 11:05 の発走で 60 分差）なので、同日の先行レースは「90 分以上前」に当たらない
    assert (races["sr_var_n"] == 0).all()
    assert races.filter(pl.col("date") > pl.col("date").min())["sr_var_today"].is_not_null().mean() > 0.8  # 前の開催日の値で補われる


def test_check_runs_rejects_zero_distance_and_infinite_time(built):
    from src.pipeline.features.horse_history.builder import check_runs

    tables, la, F, P = built
    runs = load_runs(tables)
    check_runs(runs)  # 正常なデータは通る
    with pytest.raises(ValueError, match="距離"):
        check_runs(runs.with_columns(pl.when(pl.col("race_id") == runs["race_id"][0]).then(0).otherwise(pl.col("distance")).alias("distance")))
    with pytest.raises(ValueError, match="無限大"):
        check_runs(runs.with_columns(pl.when(pl.col("race_id") == runs["race_id"][0]).then(float("inf")).otherwise(pl.col("time_sec")).alias("time_sec")))
