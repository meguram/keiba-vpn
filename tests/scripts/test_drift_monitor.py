"""ドリフト週次監視（drift_monitor・drift_labelfree・drift_report）の合成データテスト。2026 年の実データは使わない。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.scripts.data import drift_labelfree as LF
from src.scripts.data import drift_monitor as D
from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

CFG = {"baseline": {"from": "2022-01-03", "to": "2023-12-31"}, "test2": {"from": "2024-01-01"},
       "alert": {"window_weeks": 8, "gap_sigma_k": 3.0, "min_races_window": 100, "cusum_k": 0.5, "cusum_h": 5.0, "ewma_lambda": 0.2, "coverage_floor": 0.9, "psi_warn": 0.10,
                     "ece_roi": 0.5, "slope_lo": 0.2, "slope_hi": 5.0}}      # 合成データは窓が小さく ECE・傾きの揺れが大きいので、回収率に効く水準は緩める


def _frame(weeks: int, shift_from: int | None, seed: int = 0, shift_to: int = 10_000, races: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2022-01-03")
    for w in range(weeks):
        for rr in range(races):
            n = 10
            pm = rng.dirichlet(np.ones(n) * 2)
            sharp = 0.5 if (shift_from is not None and shift_from <= w < shift_to) else 1.0       # 尖りが落ちる（モデルの勝率が平ら）
            lg = sharp * np.log(pm) + rng.normal(0, 0.15, n)
            pw = np.exp(lg) / np.exp(lg).sum()
            order = list(rng.choice(n, size=n, replace=False, p=pm / pm.sum())) if False else None
            remaining, order = list(range(n)), []
            while remaining:                                                    # Plackett-Luce で着順を作る
                q = pm[remaining] / pm[remaining].sum()
                k = int(rng.choice(len(remaining), p=q))
                order.append(remaining.pop(k))
            fin = np.empty(n, dtype=int)
            fin[order] = np.arange(1, n + 1)
            p2, p3 = harville_top2_prob(pw), harville_top3_prob(pw)
            d = start + pd.Timedelta(weeks=w, days=5)
            for i in range(n):
                rows.append({"race_id": f"{w:04d}{rr:04d}", "horse_id": f"h{i}", "date": d, "year": d.year, "month": d.month, "src": "forward", "p_win": pw[i],
                             "p_top2": p2[i], "p_top3": p3[i], "p_mkt": pm[i], "y_win": float(fin[i] == 1), "y_top2": float(fin[i] <= 2), "y_top3": float(fin[i] <= 3),
                             "finish_position": fin[i], "surface": "芝" if i % 2 else "ダート", "distance": 1600, "field_size": n, "venue": "東京", "grade": "1勝",
                             "race_class": "x", "age": 3.0, "popularity": i + 1})
    return pd.DataFrame(rows)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    return D.prepare(df)


def _pipeline(df: pd.DataFrame, cusum_from: str | None = None):
    d = _prep(df)
    rt = D.race_table(d)
    edges = D.M.quantile_edges(d[d["week"] <= CFG["baseline"]["to"]]["p_win"].to_numpy(), 10)
    series = D.rolling_series(d, 8, rt, edges)
    lim = D.baseline_limits(series, CFG["baseline"]["from"], CFG["baseline"]["to"], 3.0)
    wkb = D.weekly_baseline(d, CFG["baseline"]["from"], CFG["baseline"]["to"], 10)
    seq = D.evaluate_sequence(series, lim, wkb, CFG["test2"]["from"], CFG["alert"], None, cusum_from)
    return d, series, lim, wkb, seq


def test_rolling_series_columns_and_baseline_limits():
    d, series, lim, wkb, seq = _pipeline(_frame(130, None))
    assert {"gap", "gap_top2", "gap_top3", "ece", "slope", "sd_ratio", "rank_gap", "psi_p", "brier_win", "week_gap"} <= set(series.columns)
    assert lim["sd_ratio"]["lcl"] < lim["sd_ratio"]["mean"] < lim["sd_ratio"]["ucl"]
    assert lim["gap"]["k2_upper"] < lim["gap"]["ucl"]
    assert wkb["gap"]["sd"] > 0


def test_stage_rises_after_flattening_and_stays_ok_without_shift():
    d, series, lim, wkb, seq = _pipeline(_frame(130, 104, seed=1))
    before = seq[seq["week"] < str((pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=104)).date())]
    assert (before["stage"].isin(["ok", "注意", "保留"])).all()
    assert seq["stage"].iloc[-1] in ("警告", "要対応")
    assert seq["flag_dist"].iloc[-1] and seq["sd_ratio"].iloc[-1] < lim["sd_ratio"]["lcl"]
    assert seq["gap_class"].iloc[-1] in ("有意な悪化", "境界")
    calm = _pipeline(_frame(130, None, seed=2))[4]
    assert (calm["stage"] == "要対応").sum() == 0


def test_small_window_is_held_and_cusum_ack_resets():
    d, series, lim, wkb, seq = _pipeline(_frame(130, 104, seed=3, shift_to=112))       # 104〜111 週だけ平ら、その後は戻る
    assert seq["cusum_flag"].any()
    after = seq["week"].iloc[-8]
    seq2 = D.evaluate_sequence(series, lim, wkb, CFG["test2"]["from"], CFG["alert"], None, after)
    assert not seq2["cusum_flag"].iloc[-1]
    held = D.evaluate_sequence(series, lim, wkb, CFG["test2"]["from"], {**CFG["alert"], "min_races_window": 10_000}, None, None)
    assert (held["stage"] == "保留").all() and (held["gap_class"] == "保留").all()


def test_input_coverage_group_flags_and_warning_needs_two_groups():
    d, series, lim, wkb, seq = _pipeline(_frame(130, None, seed=4))
    cov = {w: 0.05 for w in seq["week"]}
    seq2 = D.evaluate_sequence(series, lim, wkb, CFG["test2"]["from"], CFG["alert"], cov)
    assert seq2["flag_inputs"].all() and (seq2["stage"] != "ok").all()


def test_backtest_uses_segment_baselines():
    d = _prep(_frame(130, 104, seed=5))
    bt = D.backtest(d, CFG, segments=[("2023-07-01", "2023-12-31", "2022-12-31"), ("2024-01-01", "2099-12-31", "2023-12-31")])
    assert set(bt["baseline_end"]) == {"2022-12-31", "2023-12-31"}
    assert bt["stage"].iloc[-1] in ("警告", "要対応")


def test_strata_fdr_flags_the_shifted_stratum():
    df = _frame(130, None, seed=6)
    d = _prep(df)
    bad = d["surface"] == "芝"
    d.loc[bad & (d["week"] >= "2023-10-01"), "p_win"] = d.loc[bad & (d["week"] >= "2023-10-01"), "p_win"] * 0.5       # 芝だけ劣化させる
    d["d"] = D.M.bin_ll(d["y_win"], d["p_win"]) - D.M.bin_ll(d["y_win"], d["p_mkt"])
    weeks = sorted(d["week"].unique())[-8:]
    out = D.strata_fdr(d, weeks, "2023-09-30", "2022-01-03", ["surface"], min_rows=50)
    assert out and out[0]["value"] == "芝" and out[0]["flag"]


def test_labelfree_dist_and_new_ids():
    rng = np.random.default_rng(7)

    def frame(sharp, n_races=300):
        rows = []
        for r in range(n_races):
            p = rng.dirichlet(np.ones(10) * sharp)
            rows += [{"race_id": f"r{sharp}{r}", "horse_id": f"h{i}", "p_win": p[i], "p_top3": min(3 * p[i], 1.0), "field_size": 10, "jockey_id": f"j{i}", "trainer_id": f"t{i}"} for i in range(10)]
        return pd.DataFrame(rows)

    base, cur = frame(2.0), frame(8.0)
    sh = LF.dist_shift(base, cur)
    assert sh["p_win"]["psi"] > 0.1 and sh["max_p_win"]["mean_cur"] < sh["max_p_win"]["mean_base"]
    assert LF.dist_metrics(cur)["entropy_rel"] > LF.dist_metrics(base)["entropy_rel"]
    first = pd.Series({"h0": "2020-01-05", "h1": "2026-10-10"})
    n = LF.new_id_rates(cur.iloc[:4].assign(horse_id=["h0", "h1", "h9", "h0"]), first, "2026-10-05", {"j0"}, {"t0", "t1", "t2", "t3"})
    assert n["new_horse"] == 0.5 and n["new_jockey"] == 0.75 and n["new_trainer"] == 0.0


def test_feature_section_with_dumps(tmp_path):
    rng = np.random.default_rng(6)
    d = tmp_path / "data/local/inference/drift_features"
    for yr, mu in ((2025, 0.0), (2026, 2.0)):
        (d / str(yr)).mkdir(parents=True)
        pd.DataFrame({"race_id": "r", "horse_id": [f"h{i}" for i in range(400)], "day": f"{yr}0601", "f1": rng.normal(mu, 1, 400), "f2": rng.normal(0, 1, 400)}
                     ).to_parquet(d / str(yr) / f"{yr}0601.parquet")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/drift_top_features.json").write_text(json.dumps({"features": ["f1", "f2"]}))
    cfg = {"alert": {"top_features": 50, "psi_alert": 0.25, "psi_warn": 0.1, "psi_alert_min_features": 1, "missing_rate_jump": 0.1}}
    out = D.feature_section(tmp_path, cfg)
    assert out["status"] == "alert" and out["top_psi"][0]["feature"] == "f1"
    assert D.feature_section(tmp_path / "none", cfg)["status"] == "no_dump"


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


def test_report_html_scorecard_and_append_only_outputs(tmp_path):
    d, series, lim, wkb, seq = _pipeline(_frame(130, 104, seed=8))
    rep = {"asof": "2026-10-21 10:00", "window_weeks": 8, "stage": seq["stage"].iloc[-1], "latest_result_week": seq["week"].iloc[-1], "periods": {"baseline": CFG["baseline"]},
           "performance": {"limits": lim, "weekly_baseline": wkb, "min_races_window": 100, "sequence": seq.round(5).to_dict("records"),
                           "latest_window": {**D.window_metrics(d.tail(1000)), "gap_ci": {"lo": 0.0, "hi": 0.01}}, "cumulative_since_monitor_start": {},
                           "calibration": {"window": D.M.calib_bins(d["p_win"], d["y_win"]).to_dict("records"), "baseline": D.M.calib_bins(d["p_win"], d["y_win"]).to_dict("records"),
                                           "window_top3": D.M.calib_bins(d["p_top3"], d["y_top3"]).to_dict("records")}},
           "coverage": {"status": "alert", "block": 0.05, "table": [{"ym": "2026-06", "rows": 10, "block": 0.05, "rec": 0.8}]}, "features": {"status": "skipped"},
           "shadow_v5": {"status": "no_v5_shadow"}, "labelfree": {"status": "ok", "shift": {}}, "strata": []}
    rep["scorecard"] = D.scorecard(rep)
    assert any(c["name"].startswith("入力") and c["color"] == "red" for c in rep["scorecard"])
    page = D.write_outputs(rep, tmp_path)
    rep2 = {**rep, "asof": "2026-10-21 11:00"}
    D.write_outputs(rep2, tmp_path)
    assert len(list((tmp_path / "reports").iterdir())) == 2                         # 時刻つきで新規保存（上書きしない）
    assert (tmp_path / "history.jsonl").read_text().count("\n") == 2
    html = (page / "report.html").read_text()
    assert "<svg" in html and "スコアカード" in html and "調教ページの被覆" in html


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


def test_pop_calibration_section_flags_shifted_class():
    import numpy as np
    import pandas as pd

    from src.scripts.data.drift_monitor import pop_calibration_section

    rng = np.random.default_rng(0)

    def make(shift: float, n_races: int = 700) -> pd.DataFrame:
        rows = []
        for r in range(n_races):
            q = rng.dirichlet(np.ones(12))
            for i in range(12):
                pw = float(q[i])
                p3 = min(3 * pw, 0.95)
                true3 = min(max(p3 + (shift if q[i] == q.max() else 0.0), 0.01), 0.99)       # 最人気だけ実績が上にずれる
                rows.append({"race_id": r, "p_mkt": pw, "p_win": pw, "y_win": int(rng.random() < pw), "p_top3": p3, "y_top3": int(rng.random() < true3)})
        return pd.DataFrame(rows)

    base, same, moved = make(0.0), make(0.0), make(0.20)
    ok = pop_calibration_section(same, base)
    bad = pop_calibration_section(moved, base)
    assert ok["status"] == "ok" or len(ok["flags"]) <= 1
    assert bad["status"] == "alert" and any(f["pop"] == "1人気" and f["target"] == "top3" for f in bad["flags"])
