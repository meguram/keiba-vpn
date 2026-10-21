"""ドリフト週次監視（drift_monitor）の合成データテスト。2026 年の実データは使わない。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.scripts.data import drift_monitor as D


def _frame(weeks: int, shift_from: int | None, seed: int = 0, shift_to: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2022-01-03")
    for w in range(weeks):
        for rr in range(24):
            n = 12
            pm = rng.dirichlet(np.ones(n) * 2)
            sharp = 0.6 if (shift_from is not None and shift_from <= w < shift_to) else 1.0       # 尖りが落ちる（モデルの勝率が平ら）
            logit = sharp * np.log(pm) + rng.normal(0, 0.2, n)
            pw = np.exp(logit) / np.exp(logit).sum()
            win = rng.choice(n, p=pm)
            d = start + pd.Timedelta(weeks=w, days=5)
            for i in range(n):
                rows.append({"race_id": f"{w:04d}{rr:04d}", "horse_id": f"h{i}", "date": d, "p_win": pw[i], "p_mkt": pm[i], "y_win": float(i == win)})
    return pd.DataFrame(rows)


def test_rolling_series_and_alerts_fire_after_shift():
    df = _frame(120, shift_from=100)
    s = D.rolling_series(df, 8)
    assert {"gap", "sd_ratio", "ece", "week"} <= set(s.columns)
    base_end = str((pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=95)).date())
    lim = D.baseline_limits(s, "2022-01-01", base_end, 3.0)
    assert lim["sd_ratio"]["lcl"] < lim["sd_ratio"]["mean"]
    mon_start = str((pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=96)).date())
    wk = D.weekly_gap_sigma(df, "2022-01-01", base_end)
    al = D.evaluate_alerts(s, lim, mon_start, 3, {"gap": wk["gap"], "sd_ratio": wk["sd_ratio"]})
    assert al["status"] == "alert"
    assert al["triggers"]["sharpness"]["alert"]
    assert al["triggers"]["sharpness"]["cusum_first_alarm_week"] is not None


def test_no_alert_without_shift_and_ack_resets_cusum():
    df = _frame(120, shift_from=None, seed=1)
    s = D.rolling_series(df, 8)
    base_end = str((pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=95)).date())
    lim = D.baseline_limits(s, "2022-01-01", base_end, 3.0)
    mon_start = str((pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=96)).date())
    wk = D.weekly_gap_sigma(df, "2022-01-01", base_end)
    al = D.evaluate_alerts(s, lim, mon_start, 3, {"gap": wk["gap"], "sd_ratio": wk["sd_ratio"]})
    assert al["status"] in ("ok", "warn")
    df2 = _frame(130, shift_from=100, shift_to=110)                      # 100〜109 週だけ尖りが落ち、その後は戻る
    s2 = D.rolling_series(df2, 8)
    wk2 = {"gap": wk["gap"], "sd_ratio": wk["sd_ratio"]}
    assert D.evaluate_alerts(s2, lim, mon_start, 3, wk2)["triggers"]["sharpness"]["cusum_alarm_now"]       # CUSUM は戻っても鳴り続ける
    after = s2["week"].iloc[-10]
    assert not D.evaluate_alerts(s2, lim, mon_start, 3, wk2, cusum_from=after)["triggers"]["sharpness"]["cusum_alarm_now"]


def test_paired_delta_sign():
    rng = np.random.default_rng(5)
    rows = []
    for r in range(200):
        n = 8
        p = rng.dirichlet(np.ones(n))
        w = rng.choice(n, p=p)
        for i in range(n):
            rows.append({"race_id": f"r{r}", "horse_id": f"h{i}", "p_win": p[i], "y_win": float(i == w), "y_top2": 0.0, "y_top3": 0.0})
    a = pd.DataFrame(rows)
    b = a.assign(p_win=np.clip(a["p_win"] * 0.5 + 0.06, 0, 1))      # 平らにした確率は悪い
    out = D.paired_delta(a, b, "win")
    assert out["delta"] > 0 and out["lo"] < out["delta"] < out["hi"]


def test_feature_section_with_dumps(tmp_path):
    rng = np.random.default_rng(6)
    d = tmp_path / "data/local/inference/drift_features"
    for yr, mu in ((2025, 0.0), (2026, 2.0)):
        (d / str(yr)).mkdir(parents=True)
        pd.DataFrame({"race_id": "r", "horse_id": [f"h{i}" for i in range(400)], "day": f"{yr}0601", "f1": rng.normal(mu, 1, 400),
                      "f2": rng.normal(0, 1, 400)}).to_parquet(d / str(yr) / f"{yr}0601.parquet")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/drift_top_features.json").write_text(json.dumps({"features": ["f1", "f2"]}))
    cfg = {"alert": {"top_features": 50, "psi_alert": 0.25, "psi_warn": 0.1, "psi_alert_min_features": 1, "missing_rate_jump": 0.1}}
    out = D.feature_section(tmp_path, cfg)
    assert out["status"] == "alert" and out["top_psi"][0]["feature"] == "f1"
    assert D.feature_section(tmp_path / "none", cfg)["status"] == "no_dump"


def test_render_html_and_write_outputs(tmp_path):
    df = _frame(60, shift_from=None)
    s = D.rolling_series(df, 8)
    lim = D.baseline_limits(s, "2022-01-01", "2023-12-31", 3.0)
    rep = {"asof": "2026-10-21 10:00", "window_weeks": 8, "baseline_period": "x", "baseline": lim, "alerts": {"status": "ok", "triggers": {}}, "status": "ok",
           "series": s.round(5).to_dict("records"), "features": {"status": "skipped"}, "shadow_v5": {"status": "skipped"},
           "coverage": {"status": "alert", "table": [{"ym": "2026-06", "rows": 10, "block": 0.05, "rec": 0.8}]}}
    page = D.render_html(rep)
    assert "<svg" in page and "ドリフト監視" in page and "調教ページの被覆" in page
    D.write_outputs(rep, tmp_path)
    assert (tmp_path / "latest.json").exists() and (tmp_path / "history.jsonl").read_text().count("\n") == 1


def test_hook_is_default_off(monkeypatch):
    from src.pipeline.inference import drift_monitor_hook as H

    monkeypatch.delenv("KEIBA_DRIFT_MONITOR", raising=False)
    assert H.refresh_for_dates(["20261017"]) == {"status": "disabled"}


def test_coverage_census_and_status():
    from src.scripts.data import drift_coverage as C

    df = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "horse_id": ["h1", "h2", "h1", "h3"],
                       "date": pd.to_datetime(["2026-05-02"] * 2 + ["2026-06-06"] * 2)})
    blocks = {"h1": {"2026-05-02"}, "h2": {"2026-05-02"}, "h3": set()}
    t = C.census(df, blocks=blocks, records_present={"h1": True, "h2": True, "h3": False})
    assert t.set_index("ym").loc["2026-05", "block"] == 1.0
    assert t.set_index("ym").loc["2026-06", "block"] == 0.0 and t.set_index("ym").loc["2026-06", "rec"] == 0.5
    st = C.coverage_status(t, 0.9)
    assert st["status"] == "alert" and st["latest_month"] == "2026-06"
    assert C.coverage_status(t.iloc[:1], 0.9)["status"] == "ok"
    assert C.coverage_status(t.iloc[:0])["status"] == "no_data"
