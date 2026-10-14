import json

import numpy as np
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.pipeline.features.horse_history import accumulate as acc
from src.pipeline.features.horse_history.common import read_runs_raw
from src.pipeline.inference import weight_triggered_predict as wtp
from tests.pipeline.test_horse_history import _synthetic_tables
from tests.pipeline.test_weight_triggered_predict import FakeStorage, _races, _shutuba_from_day

JST = timezone(timedelta(hours=9))


@pytest.fixture(autouse=True)
def _isolate_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_SERVING_BUNDLE_DIR", str(tmp_path / "bundle"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("acc")
    tables, la = _synthetic_tables(tmp)
    raw = read_runs_raw(tables)
    day = str(raw.filter(pl.col("date").str.starts_with("2025"))["date"].sort()[300])
    return tables, la, day


def test_accumulate_writes_year_file_manifest_and_is_stable(env, tmp_path, monkeypatch):
    tables, la, _ = env
    monkeypatch.setenv("KEIBA_FEATURE_STORE_DIR", str(tmp_path / "fs"))
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf"))
    m1 = acc.accumulate(2025, tables, la)
    f = pl.read_parquet(tmp_path / "fs" / "features_2025.parquet")
    assert f.height == m1["rows"] > 0 and f["date"].dt.year().unique().to_list() == [2025]
    assert m1["features"] == 127 and m1["changed_rows_vs_previous"] is None
    m2 = acc.accumulate(2025, tables, la)  # 同じ入力なら、前回との差は 0 行
    assert m2["changed_rows_vs_previous"] == 0
    assert json.loads((tmp_path / "fs" / "manifest_2025.json").read_text(encoding="utf-8"))["rows"] == f.height


def test_skew_report_compares_inference_snapshot_with_batch_features(env, tmp_path, monkeypatch):
    tables, la, day = env
    monkeypatch.setenv("KEIBA_FEATURE_STORE_DIR", str(tmp_path / "fs"))
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf"))
    sh = _shutuba_from_day(tables, day, with_weight=True)
    ctx = wtp.InferenceContext(tables_dir=tables, layer_a=la, years=list(range(2020, 2026)))
    ctx.warm()
    wtp.run_for_races(_races(sh), FakeStorage(sh), ctx, weights_known={k: True for k in sh}, mode="features")
    ymd = day.replace("-", "")
    snaps = list((tmp_path / "inf" / "snapshots" / ymd).glob("*.json"))
    assert len(snaps) == len(sh)
    s = json.loads(snaps[0].read_text(encoding="utf-8"))
    assert s["weights_known"] is True and s["entries"] and "weight" in s["entries"][0]

    m = acc.accumulate(2025, tables, la)
    assert m["skew"][ymd] == "compared"
    rep = json.loads((tmp_path / "fs" / "skew" / f"{ymd}.json").read_text(encoding="utf-8"))
    unexpected = {c for t in ("0", "2") for c in rep["mismatch_by_tier"][t]}
    assert unexpected == set(), unexpected  # 同日の先行レースの結果を使う tier 1 以外は、学習時の値と一致する


def test_skew_report_waits_when_results_are_not_in_history(tmp_path):
    d = tmp_path / "20260101"
    d.mkdir()
    pl.DataFrame({"race_id": ["x"], "horse_id": ["h"], "a_x": [1.0]}).write_parquet(d / "x.parquet")
    F = pl.DataFrame({"race_id": ["y"], "horse_id": ["h"], "date": [datetime(2026, 1, 1).date()], "a_x": [1.0]})
    assert acc.skew_report(F, ["a_x"], d, {"a_x": 0})["status"] == "results_not_yet"


def test_accumulate_writes_serving_bundle_and_bundle_engine_matches_full_engine(env, tmp_path, monkeypatch):
    from src.pipeline.features.horse_history.serving import default_bundle_dir

    tables, la, day = env
    monkeypatch.setenv("KEIBA_FEATURE_STORE_DIR", str(tmp_path / "fs"))
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf_full"))
    monkeypatch.setenv("KEIBA_INFERENCE_ENGINE", "full")
    # 当日より前の履歴だけでバンドルを作るため、当日以降を除いた表を用意する
    cut_tables = tmp_path / "cut"
    for y in range(2020, 2026):
        d = cut_tables / str(y)
        d.mkdir(parents=True)
        pl.read_parquet(tables / str(y) / "race_result_flat.parquet").filter(pl.col("date") < day).write_parquet(d / "race_result_flat.parquet")
    m = acc.accumulate(2025, cut_tables, la)
    assert m["serving_bundle"]["horses"] > 0 and (default_bundle_dir() / "meta.json").exists()

    sh = _shutuba_from_day(tables, day, with_weight=True)
    ymd = day.replace("-", "")
    ctx_full = wtp.InferenceContext(tables_dir=cut_tables, layer_a=la, years=list(range(2020, 2026)))
    ctx_full.warm()
    ev_full = wtp.run_for_races(_races(sh), FakeStorage(sh), ctx_full, weights_known={k: True for k in sh}, mode="features")
    assert {e["engine"] for e in ev_full} == {"full"}

    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf_bundle"))
    monkeypatch.setenv("KEIBA_INFERENCE_ENGINE", "auto")
    ctx_b = wtp.InferenceContext(tables_dir=cut_tables, layer_a=la, years=list(range(2020, 2026)))
    assert ctx_b.warm()["engine"] == "bundle"
    ev_b = wtp.run_for_races(_races(sh), FakeStorage(sh), ctx_b, weights_known={k: True for k in sh}, mode="features")
    assert {e["engine"] for e in ev_b} == {"bundle"}
    for rid in sh:
        a = pl.read_parquet(tmp_path / "inf_full" / "features" / ymd / f"{rid}.parquet").sort("horse_id")
        b = pl.read_parquet(tmp_path / "inf_bundle" / "features" / ymd / f"{rid}.parquet").sort("horse_id")
        assert a.height == b.height
        cols = [c for c in a.columns if c in b.columns and a[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32)]
        for c in cols:
            x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
            d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
            assert not (np.where(np.isnan(d), np.inf, d) > 1e-9).any(), (rid, c)
