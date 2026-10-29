"""src.utils.cost_watch と gcp_daily_cost_report --watch の検証。GCP・Slack には接続しない。"""
from __future__ import annotations

import os
from unittest import mock

os.environ.setdefault("GCS_BUCKET", "")

from src.scripts.cloud_jobs import gcp_daily_cost_report as job  # noqa: E402
from src.utils import cost_watch as cw  # noqa: E402

PH = cw.load_phases()


def _rows(days: dict[str, dict[str, float]], cur="JPY"):
    return [{"date": d, "service": s, "cost": c, "currency": cur} for d, v in days.items() for s, c in v.items()]


def test_phase_config_matches_scenario_a():
    from src.scripts.ops import cost_estimate as ce

    cfg = ce.load_config()
    fx = float(cfg["fx_jpy_per_usd"])
    hi = ce.estimate(cfg, "A_budget_core", "high")["total_jpy"]
    p5 = next(p for p in PH["phases"] if p["id"] == 5)
    assert abs(p5["expected_max_jpy"] - hi) < 5 and fx > 0
    assert PH["budget_jpy_per_month"] == cfg["budget_jpy_per_month"]


def test_to_jpy_usd_and_unknown():
    assert cw.to_jpy(2, "USD", 150) == 300
    assert cw.to_jpy(5, "JPY", 150) == 5
    try:
        cw.to_jpy(1, "EUR", 150)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_spike_detected_vs_previous_day():
    t = cw.daily_table(_rows({"2026-10-20": {"Cloud Run": 10.0}, "2026-10-21": {"Cloud Run": 100.0, "Cloud Storage": 3.0}}), 150)
    sp = cw.detect_spikes(t, "2026-10-21", PH["spike"])
    assert [s["service"] for s in sp] == ["Cloud Run"] and sp[0]["kind"] == "前日比" and sp[0]["factor"] == 10.0


def test_no_spike_for_small_absolute_increase():
    t = cw.daily_table(_rows({"2026-10-20": {"Cloud Run": 1.0}, "2026-10-21": {"Cloud Run": 20.0}}), 150)
    assert cw.detect_spikes(t, "2026-10-21", PH["spike"]) == []  # 3 倍超だが増分 19 円 < 30 円


def test_spike_vs_seven_day_mean_and_daily_total():
    days = {f"2026-10-{d:02d}": {"Cloud Run": 5.0} for d in range(14, 20)}
    days["2026-10-20"] = {"Cloud Run": 100.0}  # 前日がすでに急増 → 前日比では出ない
    days["2026-10-21"] = {"Cloud Run": 100.0, "Cloud Storage": 120.0}
    sp = cw.detect_spikes(cw.daily_table(_rows(days), 150), "2026-10-21", PH["spike"])
    kinds = {(s["service"], s["kind"]) for s in sp}
    assert ("Cloud Run", "7日平均比") in kinds and ("合計", "日次の上限目安") in kinds


def test_usd_rows_converted():
    t = cw.daily_table(_rows({"2026-10-21": {"Cloud Run": 1.0}}, cur="USD"), 150)
    assert t["2026-10-21"]["Cloud Run"] == 150


def test_month_summary_status():
    ph = {**PH, "current_phase": 2}
    t = {f"2026-10-{d:02d}": {"Cloud Storage": 10.0} for d in range(1, 11)}
    s = cw.month_summary(t, "2026-10-10", ph)
    assert round(s["mtd"]) == 100 and round(s["projected"]) == 310 and s["status"] == "範囲内"
    t2 = {f"2026-10-{d:02d}": {"Cloud Run": 40.0} for d in range(1, 11)}
    assert cw.month_summary(t2, "2026-10-10", ph)["status"] == "上限超過"
    ph4 = {**PH, "current_phase": 4}
    t3 = {f"2026-10-{d:02d}": {"Cloud Storage": 2.0} for d in range(1, 11)}
    assert cw.month_summary(t3, "2026-10-10", ph4)["status"].startswith("下限未満")
    assert "フェーズ2" in cw.format_summary_lines(cw.month_summary(t, "2026-10-10", ph))


def test_over_ceiling_skips_early_month_and_unknown_service():
    ceil = {"Cloud Storage": 100.0}
    t = {f"2026-10-{d:02d}": {"Cloud Storage": 10.0, "Mystery": 99.0} for d in range(1, 11)}
    out = cw.over_ceiling(t, "2026-10-10", ceil)
    assert [o["service"] for o in out] == ["Cloud Storage"]
    assert cw.over_ceiling(t, "2026-10-02", ceil) == []


def test_service_ceilings_use_scenario_a():
    c = cw.service_ceilings(PH)
    assert c["Cloud Storage"] > 0 and c["Cloud Scheduler"] > 0


def test_main_watch_adds_summary_and_separate_alert():
    rows = _rows({"2026-10-20": {"Cloud Run": 10.0}, "2026-10-21": {"Cloud Run": 120.0}})
    with mock.patch.dict(os.environ, {"GCP_BILLING_BQ_TABLE": "p.d.t"}), \
         mock.patch("src.utils.project_env.load_project_dotenv", return_value=None), \
         mock.patch.object(job, "fetch_daily_cost", return_value={"date": "2026-10-21", "rows": [{"service": "Cloud Run", "cost": 0.8}], "total": 0.8}), \
         mock.patch.object(job, "fetch_cost_history", return_value=rows), \
         mock.patch("src.utils.notify.notify_slack", return_value=True) as notify:
        assert job.main(["--date", "2026-10-21", "--watch"]) == 0
    msgs = [c.args[0] for c in notify.call_args_list]
    assert len(msgs) == 2
    assert "今月累計" in msgs[0] and "フェーズ" in msgs[0]
    assert msgs[1].startswith("[GCPコスト警告]") and "Cloud Run" in msgs[1]


def test_main_watch_failure_keeps_base_report():
    with mock.patch.dict(os.environ, {"GCP_BILLING_BQ_TABLE": "p.d.t"}), \
         mock.patch("src.utils.project_env.load_project_dotenv", return_value=None), \
         mock.patch.object(job, "fetch_daily_cost", return_value={"date": "x", "rows": [], "total": 0.0}), \
         mock.patch.object(job, "fetch_cost_history", side_effect=RuntimeError("boom")), \
         mock.patch("src.utils.notify.notify_slack", return_value=True) as notify:
        assert job.main(["--date", "2026-10-21", "--watch"]) == 0
    assert notify.call_count == 1 and "想定との比較は取得失敗" in notify.call_args.args[0]


def test_history_start_covers_month_start_and_8_days():
    assert job._history_start("2026-10-21") == "2026-10-01"
    assert job._history_start("2026-10-03") == "2026-09-25"


def test_watch_headline_is_yen_for_the_target_day():
    from src.scripts.cloud_jobs import gcp_daily_cost_report as job

    rows = [
        {"date": "2026-10-28", "service": "Cloud Run", "cost": 545.0, "currency": "JPY"},
        {"date": "2026-10-28", "service": "Cloud Storage", "cost": 464.0, "currency": "JPY"},
        {"date": "2026-10-27", "service": "Cloud Run", "cost": 100.0, "currency": "JPY"},
    ]
    _extra, _alert, headline = job.build_watch("2026-10-28", rows)
    assert headline == "[GCP日次コストレポート] 2026-10-28: 合計 ¥1,009（Cloud Run: ¥545, Cloud Storage: ¥464）"
    assert "$" not in headline
