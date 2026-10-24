"""追走難度 v2 の前夜の保存・読み出し・実測との突き合わせのテスト。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline.inference import tracking_v2_eval as ev
from src.pipeline.inference import tracking_v2_service as svc


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_TRACKING_V2_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("KEIBA_TRACKING_V2_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("KEIBA_TABLES_DIR", str(tmp_path / "tables"))
    monkeypatch.delenv("KEIBA_TRACKING_V2_RECOMPUTE", raising=False)
    return tmp_path


def _doc(race_id="202610240101", date="2026-10-24", numbers=(1, 2, 3), source="raceday_eve", base=50.0):
    return {"version": svc.VERSION, "race_id": race_id, "date": date, "venue": "東京", "surface": "芝", "distance": 1600, "track_condition": "良",
            "n_horses": len(numbers), "n_with_v2": len(numbers), "horse_numbers": list(numbers), "computed_at": "2026-10-23T18:05:00+09:00", "source": source,
            "tracking_v2_meta": {"version": "tp54-1", "pace_z": 0.1, "definition": "x"},
            "horses": {str(n): {"horse_id": f"h{n}", "tracking_v2": {"ease_pct": base + n, "label": "普通", "drain_sec": 0.1 * n, "demand_sec": 0.2, "capacity_sec": -0.1},
                                "position_early": 0.5, "inputs": {"w_adj": 0.1 * n}} for n in numbers}}


def _resp(numbers=(1, 2, 3)):
    return {"predictions": [{"horse_number": n, "horse_id": f"h{n}", "position_forecast": {"early_pos_norm": 0.5}} for n in numbers]}


def test_cached_is_noop_when_flag_off(store, monkeypatch):
    monkeypatch.delenv("KEIBA_TRACKING_V2", raising=False)
    svc._write(svc.path_for("202610240101", "2026-10-24"), _doc())
    out = svc.attach_tracking_v2_cached(_resp(), "202610240101", None, "2026-10-24", {}, {})
    assert all("tracking_v2" not in p for p in out["predictions"]) and "tracking_v2_meta" not in out


def test_cached_returns_saved_fixed_value_with_same_shape(store, monkeypatch):
    monkeypatch.setenv("KEIBA_TRACKING_V2", "1")
    svc._write(svc.path_for("202610240101", "2026-10-24"), _doc())
    out = svc.attach_tracking_v2_cached(_resp(), "202610240101", None, "2026-10-24", {}, {})
    p = out["predictions"][0]
    assert set(p["tracking_v2"]) == {"ease_pct", "label", "drain_sec", "demand_sec", "capacity_sec"} and p["tracking_v2"]["ease_pct"] == 51.0
    assert out["tracking_v2_meta"]["source"] == "saved" and out["tracking_v2_meta"]["computed_at"] == "2026-10-23T18:05:00+09:00"
    assert out["tracking_v2_meta"]["pace_z"] == 0.1


def test_cancellation_recomputes_and_keeps_eve_file(store, monkeypatch):
    monkeypatch.setenv("KEIBA_TRACKING_V2", "1")
    p_eve = svc.path_for("202610240101", "2026-10-24")
    svc._write(p_eve, _doc())
    before = p_eve.read_text(encoding="utf-8")

    def fake_build(card, past, race_id, *, source, resp=None):
        for p in resp["predictions"]:
            p["tracking_v2"] = {"ease_pct": 33.0, "label": "やや大変", "drain_sec": 0.3}
        resp["tracking_v2_meta"] = {"version": "tp54-1", "pace_z": 0.2, "definition": "x"}
        return _doc(numbers=(1, 3), source=source, base=33.0)

    monkeypatch.setattr(svc, "build_doc", fake_build)
    out = svc.attach_tracking_v2_cached(_resp((1, 3)), "202610240101", None, "2026-10-24", {}, {"date": "2026-10-24"})   # 2 番が取消
    assert out["tracking_v2_meta"]["source"] == "recomputed" and out["predictions"][0]["tracking_v2"]["ease_pct"] == 33.0
    assert p_eve.read_text(encoding="utf-8") == before                      # 前夜の保存は消さない・書き換えない
    rec = svc.load_saved("202610240101", "2026-10-24", recomputed=True)
    assert rec and rec["horse_numbers"] == [1, 3] and rec["source"] == "recomputed"
    # 2 回目は計算し直さず、別保存を返す
    monkeypatch.setattr(svc, "build_doc", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not recompute")))
    out2 = svc.attach_tracking_v2_cached(_resp((1, 3)), "202610240101", None, "2026-10-24", {}, {"date": "2026-10-24"})
    assert out2["tracking_v2_meta"]["source"] == "recomputed_saved"


def test_policy_none_keeps_saved_even_if_horses_change(store, monkeypatch):
    monkeypatch.setenv("KEIBA_TRACKING_V2", "1")
    monkeypatch.setenv("KEIBA_TRACKING_V2_RECOMPUTE", "none")
    svc._write(svc.path_for("202610240101", "2026-10-24"), _doc())
    out = svc.attach_tracking_v2_cached(_resp((1, 3)), "202610240101", None, "2026-10-24", {}, {})
    assert out["tracking_v2_meta"]["source"] == "saved_stale"
    assert [p.get("tracking_v2") is not None for p in out["predictions"]] == [True, True]


def test_no_saved_falls_back_to_live(store, monkeypatch):
    monkeypatch.setenv("KEIBA_TRACKING_V2", "1")
    import src.pipeline.models.tracking_difficulty_v2 as tv2

    def fake_live(resp, past, date, course, card, capture=None):
        resp["tracking_v2_meta"] = {"version": "tp54-1", "pace_z": 0.0, "definition": "x"}
        return resp

    monkeypatch.setattr(tv2, "attach_tracking_v2_to_response", fake_live)
    out = svc.attach_tracking_v2_cached(_resp(), "202610240199", None, "2026-10-24", {}, {})
    assert out["tracking_v2_meta"]["source"] == "live"


def test_precompute_disabled_and_failure_isolation(store, monkeypatch):
    class Storage:
        def load(self, cat, rid):
            return {"date": "2026-10-24", "entries": [{"horse_number": 1}]} if cat == "race_shutuba" else None

    monkeypatch.delenv("KEIBA_TRACKING_V2", raising=False)
    assert svc.precompute_tracking_v2_for_race_ids(["a"], Storage())["status"] == "disabled"
    monkeypatch.setenv("KEIBA_TRACKING_V2", "1")

    def fake(storage, rid, *, source="raceday_eve"):
        if rid == "boom":
            raise RuntimeError("x")
        return (_doc(race_id=rid), "ok") if rid != "skipme" else (None, "pace_unknown")

    monkeypatch.setattr(svc, "build_for_race_reason", fake)
    st = svc.precompute_tracking_v2_for_race_ids(["r1", "boom", "skipme", "r2"], Storage())
    assert (st["ok"], st["fail"], st["skip"]) == (2, 1, 1) and st["skip_reasons"] == {"pace_unknown": 1}
    assert svc.load_saved("r1", "2026-10-24") and svc.load_saved("r2", "2026-10-24")
    st2 = svc.precompute_tracking_v2_for_race_ids(["r1"], Storage(), overwrite=False)
    assert st2["skip"] == 1 and st2["skip_reasons"] == {"exists": 1}


def test_eve_precompute_does_not_fail_the_eve(monkeypatch):
    from src.scraper import auto_scrape_queue as q

    monkeypatch.setenv("KEIBA_EVE_PRECOMPUTE_TRACKING", "0")
    monkeypatch.setenv("KEIBA_EVE_PRECOMPUTE_FINAL_ODDS", "0")
    monkeypatch.setenv("KEIBA_TRACKING_V2", "1")
    monkeypatch.setattr(q, "_storage", lambda: object())
    monkeypatch.setattr(svc, "precompute_tracking_v2_for_race_ids", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    pc1, pc2 = q._eve_precomputes("20261024", ["202610240101"])        # 例外を投げない
    assert "tracking_v2" not in pc1
    monkeypatch.setattr(svc, "precompute_tracking_v2_for_race_ids", lambda *a, **k: {"ok": 1, "skip": 0, "fail": 0, "total": 1})
    pc1, _ = q._eve_precomputes("20261024", ["202610240101"])
    assert pc1["tracking_v2"]["ok"] == 1
    monkeypatch.delenv("KEIBA_TRACKING_V2")
    pc1, _ = q._eve_precomputes("20261024", ["202610240101"])
    assert "tracking_v2" not in pc1


def test_capture_hook_collects_inputs_without_changing_response_keys(monkeypatch):
    from src.pipeline.models import tracking_difficulty_v2 as tv2

    if tv2.load_model() is None:
        pytest.skip("モデルの設定ファイルが無い")
    monkeypatch.setenv("KEIBA_TRACKING_V2", "1")
    resp = {"venue": "東京", "surface": "芝", "distance": 1600, "track_condition": "良",
            "pace_forecast": {"level": "middle", "front_runner_count": 2, "stalker_count": 3, "early_pressure": 0.4, "known_horses": 10, "field_size": 12},
            "predictions": [{"horse_number": n, "horse_id": f"h{n}", "position_forecast": {"early_pos_norm": n / 13.0, "rank": n}} for n in range(1, 13)]}
    cap: dict = {}
    out = tv2.attach_tracking_v2_to_response(resp, {"entries": []}, "2026-10-24", {}, {"entries": []}, capture=cap)
    if "tracking_v2_meta" not in out:
        pytest.skip("pace v2 の設定ファイルが無い")
    assert set(cap) == set(range(1, 13)) and "pf_early" in cap[1]
    assert set(out["predictions"][0]["tracking_v2"]) >= {"ease_pct", "label", "drain_sec"}
    assert "inputs" not in out["predictions"][0]["tracking_v2"]


# ───────────────────────── 突き合わせ

def test_metrics_from_pairs_monotonic():
    rng = np.random.default_rng(0)
    drain = rng.normal(0, 0.2, 2000)
    y = 1.5 * drain + rng.normal(0, 0.3, 2000)
    m = ev.metrics_from_pairs(pd.DataFrame({"drain": drain, "ease": 50 - 100 * drain, "y": y}))
    assert m["n_pairs"] == 2000 and m["corr_drain_y"] > 0.5 and m["decile_monotonic_spearman"] > 0.9 and m["decile_top_minus_bottom_y"] > 0.5
    assert m["corr_ease_y"] < -0.5 and 0 <= m["share_ease_ge60"] <= 1 and len(m["deciles"]) == 10


def test_aggregate_day_week_and_trend(store, monkeypatch):
    date = "2026-10-24"
    rid = "202610240101"
    numbers = list(range(1, 13))
    doc = _doc(race_id=rid, date=date, numbers=numbers)
    svc._write(svc.path_for(rid, date), doc)
    svc._write(svc.path_for("202610240102", date), _doc(race_id="202610240102", date=date, numbers=numbers))
    # 結果表: 上がり 3F は 34.0 + 0.1*番号（レース内の中央値は 34.65）。2 レース目は結果が無い（保存はあるが突き合わせ対象外）
    rows = [{"race_id": rid, "horse_number": n, "last_3f": 34.0 + 0.1 * n, "date": "2026-10-24", "surface": "芝", "distance": 1600, "finish_position": n} for n in numbers]
    rows += [{"race_id": "202610240103", "horse_number": 1, "last_3f": 35.0, "date": "2026-10-24", "surface": "ダート", "distance": 1200, "finish_position": 1}]
    (store / "tables" / "2026").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(store / "tables" / "2026" / "race_result_flat.parquet")
    day = ev.aggregate_day("20261024")
    assert day["status"] == "ok" and day["races_held"] == 2 and day["races_saved"] == 2 and day["coverage"] == 1.0
    # Y = 実測の上がり偏差 − w_adj。番号 n の偏差は 0.1*n − 0.65、w_adj は 0.1*n なので Y は一定（−0.65）
    ys = {p[1]: p[4] for p in day["pairs"] if p[0] == rid}
    assert all(abs(v - (-0.65)) < 1e-6 for v in ys.values()) and len(ys) == 12
    wk = ev.aggregate_week("20261024")
    assert wk["status"] == "ok" and wk["week_start"] == "2026-10-19" and wk["metrics"]["n_pairs"] == 12
    trend = json.loads((store / "eval" / "_trend.json").read_text(encoding="utf-8"))
    assert trend["weeks"][-1]["week_start"] == "2026-10-19"
    assert ev.aggregate_day("20261031")["status"] == "no_results"


def test_served_docs_prefers_recomputed(store):
    svc._write(svc.path_for("202610240101", "2026-10-24"), _doc())
    svc._write(svc.path_for("202610240101", "2026-10-24", recomputed=True), _doc(numbers=(1, 3), source="recomputed"))
    d = ev.served_docs("20261024")
    assert d["202610240101"]["horse_numbers"] == [1, 3] and d["202610240101"]["_recomputed"]


def test_refresh_for_dates_flag_and_isolation(store, monkeypatch):
    monkeypatch.delenv("KEIBA_TRACKING_V2_EVAL", raising=False)
    assert ev.refresh_for_dates(["20261024"]) == {"status": "disabled"}
    monkeypatch.setenv("KEIBA_TRACKING_V2_EVAL", "1")
    monkeypatch.setattr(ev, "aggregate_day", lambda d, **k: (_ for _ in ()).throw(RuntimeError("x")))
    r = ev.refresh_for_dates(["20261024"], reason="t")
    assert r["status"] == "ok" and r["days"]["20261024"] == "error"
