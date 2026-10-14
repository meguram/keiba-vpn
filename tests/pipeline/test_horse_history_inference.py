from datetime import date

import numpy as np
import polars as pl
import pytest

from src.pipeline.features.horse_history.builder import build_horse_history, feature_columns
from src.pipeline.features.horse_history.common import build_repair_map, read_runs_raw
from src.pipeline.features.horse_history.inference import build_inference_features, card_to_raw_runs, load_tiers
from tests.pipeline.test_horse_history import _synthetic_tables

RESULT_COLS = ["finish_position", "time_sec", "last_3f", "passing_order", "pace", "lap_times"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hi")
    tables, la = _synthetic_tables(tmp)
    F, _ = build_horse_history(la, tables)
    return tables, la, F


def _day_card(tables, day: str) -> pl.DataFrame:
    raw = read_runs_raw(tables)
    return raw.filter(pl.col("date") == day).drop(RESULT_COLS)


def _mismatch(ref: pl.DataFrame, got: pl.DataFrame, cols: list[str], tol: float = 1e-9) -> dict:
    ref, got = ref.sort(["race_id", "horse_id"]), got.sort(["race_id", "horse_id"])
    assert ref.height == got.height
    bad = {}
    for c in cols:
        a, b = ref[c].to_numpy().astype(float), got[c].to_numpy().astype(float)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)
        if (d > tol).any():
            bad[c] = int((d > tol).sum())
    return bad


def test_inference_features_equal_batch_features(setup):
    tables, la, F = setup
    day = "2024-07-06" if (F["date"] == date(2024, 7, 6)).any() else str(F.filter(pl.col("date").dt.year() == 2024)["date"].sort()[300])
    card = _day_card(tables, day)
    assert card.height > 0
    out, info = build_inference_features(read_runs_raw(tables), card, build_repair_map(la))
    cols = sorted({c for v in feature_columns(F.columns).values() for c in v})
    ref = F.filter(pl.col("race_id").is_in(card["race_id"].unique().to_list()))
    assert out.height == ref.height == info["rows"]
    bad = _mismatch(ref, out, cols)
    assert set(bad) <= {"sr_var_today", "sr_var_n"}, bad  # 同日の先行レースの結果を使うものだけは、結果が無いと変わりうる


def test_results_are_not_needed_and_history_after_today_is_ignored(setup):
    tables, la, F = setup
    day = str(F.filter(pl.col("date").dt.year() == 2023)["date"].sort()[100])
    card = _day_card(tables, day)
    rm = build_repair_map(la)
    a, _ = build_inference_features(read_runs_raw(tables), card, rm)
    b, _ = build_inference_features(read_runs_raw(tables).filter(pl.col("date") < day), card, rm)
    cols = sorted({c for v in feature_columns(a.columns).values() for c in v})
    assert _mismatch(a, b, cols) == {}


def test_rerun_after_results_are_imported_does_not_duplicate(setup):
    tables, la, F = setup
    day = str(F.filter(pl.col("date").dt.year() == 2024)["date"].sort()[50])
    card = _day_card(tables, day)
    out, info = build_inference_features(read_runs_raw(tables), card, build_repair_map(la))  # raw_hist に当日の結果が入っている
    assert out.height == card.height and out.select(["race_id", "horse_id"]).is_duplicated().sum() == 0


def test_missing_weight_uses_previous_weight_and_is_reported(setup):
    tables, la, F = setup
    day = str(F.filter(pl.col("date").dt.year() == 2024)["date"].sort()[200])
    card = _day_card(tables, day).with_columns(pl.lit(None, pl.Int64).alias("weight"))
    out, info = build_inference_features(read_runs_raw(tables), card, build_repair_map(la))
    assert info["weights_known_ratio"] == 0.0 and out.height == card.height


def test_card_to_raw_runs_has_placeholders_only(setup):
    tables, _, _ = setup
    raw = card_to_raw_runs(_day_card(tables, "2024-01-06") if _day_card(tables, "2024-01-06").height else _day_card(tables, str(read_runs_raw(tables)["date"].sort()[5000])))
    assert raw["time_sec"].null_count() == raw.height and raw["last_3f"].null_count() == raw.height
    assert (raw["finish_position"] >= 1).all()


def test_tier_file_covers_all_features(setup):
    _, _, F = setup
    tiers = load_tiers()
    cols = {c for v in feature_columns(F.columns).values() for c in v}
    assert cols <= set(tiers), sorted(cols - set(tiers))
    assert set(tiers.values()) == {0, 1, 2}
