import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.pipeline.features.horse_history.common import read_runs_raw
from src.pipeline.inference import weight_triggered_predict as wtp
from tests.pipeline.test_horse_history import _synthetic_tables

JST = timezone(timedelta(hours=9))


class FakeStorage:
    def __init__(self, shutuba: dict[str, dict]):
        self.shutuba = shutuba

    def load(self, category, key, bypass_cache=False):
        assert category == "race_shutuba" and bypass_cache
        return self.shutuba.get(key)


def _shutuba_from_day(tables, day: str, *, with_weight: bool) -> dict[str, dict]:
    raw = read_runs_raw(tables).filter(pl.col("date") == day)
    out = {}
    for rid, g in raw.group_by("race_id", maintain_order=True):
        rid = rid[0]
        r0 = g.row(0, named=True)
        entries = [
            {"horse_number": r["horse_number"], "horse_id": r["horse_id"], "sex_age": r["sex_age"], "jockey_id": r["jockey_id"],
             "trainer_id": r["trainer_id"], "jockey_weight": r["jockey_weight"], "weight": r["weight"] if with_weight else None}
            for r in g.iter_rows(named=True)
        ]
        out[rid] = {"race_id": rid, "date": r0["date"], "venue": "東京", "surface": r0["surface"], "distance": r0["distance"],
                    "direction": r0["direction"], "grade": r0["grade"], "race_class": r0["race_class"], "weather": "晴",
                    "track_condition": r0["track_condition"], "start_time": r0["start_time"], "field_size": g.height, "race_name": "x",
                    "entries": entries}
    return out


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wt")
    tables, la = _synthetic_tables(tmp)
    raw = read_runs_raw(tables)
    day = str(raw.filter(pl.col("date").str.starts_with("2024"))["date"].sort()[400])
    return tables, la, day


def _races(sh):
    return [{"race_id": rid, "post_time": datetime(2024, 7, 6, 10, tzinfo=JST)} for rid in sh]


def test_run_for_races_writes_features_and_events(env, tmp_path, monkeypatch):
    tables, la, day = env
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf"))
    sh = _shutuba_from_day(tables, day, with_weight=True)
    ctx = wtp.InferenceContext(tables_dir=tables, layer_a=la, years=list(range(2020, 2026)))
    ctx.warm()
    evs = wtp.run_for_races(_races(sh), FakeStorage(sh), ctx, weights_known={k: True for k in sh}, mode="features")
    assert [e["status"] for e in evs] == ["ok"] * len(sh)
    ymd = day.replace("-", "")
    for rid in sh:
        f = pl.read_parquet(tmp_path / "inf" / "features" / ymd / f"{rid}.parquet")
        assert f.height == len(sh[rid]["entries"]) and "d_wt_dev" in f.columns
    lines = (tmp_path / "inf" / "events" / f"{ymd}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(sh) and all(json.loads(l)["weights_known"] for l in lines)


def test_missing_card_is_reported_as_error_not_raised(env, tmp_path, monkeypatch):
    tables, la, _ = env
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf"))
    ctx = wtp.InferenceContext(tables_dir=tables, layer_a=la, years=list(range(2020, 2026)))
    evs = wtp.run_for_races([{"race_id": "202405990101", "post_time": datetime(2024, 7, 6, 10, tzinfo=JST)}], FakeStorage({}), ctx)
    assert evs[0]["status"] == "error"


def test_features_without_weight_match_when_weight_is_known_except_tier2(env, tmp_path, monkeypatch):
    import numpy as np
    from src.pipeline.features.horse_history.inference import load_tiers

    tables, la, day = env
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf"))
    ctx = wtp.InferenceContext(tables_dir=tables, layer_a=la, years=list(range(2020, 2026)))
    ctx.warm()
    sh_w, sh_n = _shutuba_from_day(tables, day, with_weight=True), _shutuba_from_day(tables, day, with_weight=False)
    wtp.run_for_races(_races(sh_n), FakeStorage(sh_n), ctx, mode="features")
    ymd = day.replace("-", "")
    no_w = {rid: pl.read_parquet(tmp_path / "inf" / "features" / ymd / f"{rid}.parquet") for rid in sh_n}
    wtp.run_for_races(_races(sh_w), FakeStorage(sh_w), ctx, mode="features")
    with_w = {rid: pl.read_parquet(tmp_path / "inf" / "features" / ymd / f"{rid}.parquet") for rid in sh_w}
    tiers = load_tiers()
    changed = set()
    for rid in sh_w:
        a, b = no_w[rid].sort("horse_id"), with_w[rid].sort("horse_id")
        for c, t in tiers.items():
            if c not in a.columns:
                continue
            x, y = a[c].to_numpy().astype(float), b[c].to_numpy().astype(float)
            d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
            if (np.where(np.isnan(d), np.inf, d) > 1e-9).any():
                changed.add((c, t))
    # 馬体重の有無で変わってよいのは tier 2 だけ（前日に計算できる tier 0 は変わらない）
    assert {c for c, t in changed if t == 0} == set(), changed


def test_entries_without_weight_are_dropped_once_weights_are_announced(env, tmp_path, monkeypatch):
    tables, la, day = env
    monkeypatch.setenv("KEIBA_INFERENCE_DIR", str(tmp_path / "inf"))
    sh = _shutuba_from_day(tables, day, with_weight=True)
    rid = next(iter(sh))
    sh[rid]["entries"][0]["weight"] = None  # 体重が発表されない馬（出走取消・除外）
    n_other = {k: len(v["entries"]) for k, v in sh.items()}
    ctx = wtp.InferenceContext(tables_dir=tables, layer_a=la, years=list(range(2020, 2026)))
    ctx.warm()
    evs = wtp.run_for_races(_races(sh), FakeStorage(sh), ctx, weights_known={k: True for k in sh}, mode="features")
    by = {e["race_id"]: e for e in evs}
    assert by[rid]["rows"] == n_other[rid] - 1 and by[rid]["dropped_no_weight"] == 1
    assert all(by[k]["rows"] == n_other[k] for k in sh if k != rid)
    # 馬体重が未発表（weights_known=False）のレースでは、体重の無い馬を外さない
    evs2 = wtp.run_for_races(_races(sh), FakeStorage(sh), ctx, weights_known={k: False for k in sh}, mode="features")
    assert {e["race_id"]: e["rows"] for e in evs2} == n_other
