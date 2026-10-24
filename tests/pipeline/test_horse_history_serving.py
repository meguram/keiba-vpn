from datetime import date

import numpy as np
import polars as pl
import pytest

from src.pipeline.features.horse_history.builder import build_with_state, feature_columns
from src.pipeline.features.horse_history.common import build_repair_map, load_runs, read_runs_raw
from src.pipeline.features.horse_history.inference import build_inference_features
from src.pipeline.features.horse_history.serving import Bundle, compute_entrants, write_bundle
from tests.pipeline.test_horse_history import _synthetic_tables

RESULT_COLS = ["finish_position", "time_sec", "last_3f", "passing_order", "pace", "lap_times"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sv")
    tables, la = _synthetic_tables(tmp)
    raw = read_runs_raw(tables)
    day = str(raw.filter(pl.col("date").str.starts_with("2025"))["date"].sort()[300])
    rm = build_repair_map(la)
    cut = date.fromisoformat(day)
    runs = load_runs(tables).filter(pl.col("date") < cut)
    F, P, S, elo = build_with_state(runs, rm.filter(pl.col("src_date") < cut))
    b = tmp / "bundle"
    meta = write_bundle(b, F.columns, P, S, elo)
    card = raw.filter(pl.col("date") == day).drop(RESULT_COLS)
    return tables, la, raw, rm, day, b, meta, card


def _mismatch(a: pl.DataFrame, b: pl.DataFrame, cols: list[str], tol: float = 1e-9) -> dict:
    key = ["race_id", "horse_id"]
    a, b = a.sort(key), b.sort(key)
    assert a.height == b.height and (a.select(key).to_numpy() == b.select(key).to_numpy()).all()
    bad = {}
    for c in cols:
        x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
        d = np.where(np.isnan(d), np.inf, d)
        if (d > tol).any():
            bad[c] = int((d > tol).sum())
    return bad


def test_bundle_meta_and_horse_lookup(setup):
    *_, b, meta, card = setup
    bd = Bundle(b)
    assert meta["horses"] > 0 and bd.meta["rows"] == meta["rows"]
    ids = card["horse_id"].unique().to_list()[:5]
    h = bd.horses(ids)
    assert set(h["horse_id"].unique().to_list()) <= set(ids) and "_var" in h.columns and h["date"].max() < date.fromisoformat(setup[4])


def test_bundle_features_equal_full_history_inference(setup):
    tables, la, raw, rm, day, b, meta, card = setup
    cut = date.fromisoformat(day)
    full, _ = build_inference_features(raw.filter(pl.col("date") < day), card, rm.filter(pl.col("src_date") < cut))
    got, info = compute_entrants(Bundle(b), card)
    cols = sorted({c for v in feature_columns(full.columns).values() for c in v})
    assert info["engine"] == "bundle" and got.height == full.height
    assert _mismatch(full, got, cols) == {}


def test_bundle_features_equal_training_batch_for_tier0_and_tier2(setup):
    from src.pipeline.features.horse_history.builder import build_horse_history

    tables, la, raw, rm, day, b, meta, card = setup
    F, _ = build_horse_history(la, tables)
    ref = F.filter(pl.col("race_id").is_in(card["race_id"].unique().to_list()))
    got, _ = compute_entrants(Bundle(b), card)
    cols = sorted({c for v in feature_columns(ref.columns).values() for c in v})
    bad = _mismatch(ref, got, cols)
    assert set(bad) <= {"sr_var_today", "sr_var_n"}, bad  # 同日の先行レースの結果が無いぶんだけ違ってよい


def test_sr_var_today_is_filled_for_races_without_results(setup):
    tables, la, raw, rm, day, b, meta, card = setup
    got, _ = compute_entrants(Bundle(b), card)
    assert got["sr_var_today"].is_null().mean() == 0.0  # 前の開催日の値で補われる（学習時と同じ）


def test_empty_race_info_is_rejected(setup):
    *_, b, meta, card = setup
    with pytest.raises(ValueError):
        compute_entrants(Bundle(b), card.with_columns(pl.lit("").alias("surface")))
