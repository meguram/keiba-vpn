"""cost_estimate(移行後の月額コスト見積もり)の計算の単体テスト。GCP には接続しない。"""

from __future__ import annotations

import copy

from src.scripts.ops import cost_estimate as ce


def _cfg():
    return ce.load_config()


def test_budget_core_low_is_within_budget_and_compute_is_free():
    cfg = _cfg()
    r = ce.estimate(cfg, "A_budget_core", "low")
    assert r["within_budget"]
    assert r["items_usd"]["cloud_run_service"] == 0.0
    assert r["items_usd"]["cloud_run_jobs"] == 0.0
    assert abs(r["total_usd"] - sum(r["items_usd"].values())) < 1e-9


def test_fx_scales_jpy_linearly():
    cfg = _cfg()
    a = ce.estimate(cfg, "A_budget_core", "high")
    cfg2 = copy.deepcopy(cfg)
    cfg2["fx_jpy_per_usd"] = cfg["fx_jpy_per_usd"] * 2
    b = ce.estimate(cfg2, "A_budget_core", "high")
    assert abs(b["total_jpy"] - 2 * a["total_jpy"]) < 1e-6
    assert abs(b["total_usd"] - a["total_usd"]) < 1e-9


def test_min_instance_adds_idle_cost():
    cfg = _cfg()
    a = ce.estimate(cfg, "A_budget_core", "low")["items_usd"]["cloud_run_service"]
    c = ce.estimate(cfg, "C_api_min_instance_1", "low")["items_usd"]["cloud_run_service"]
    assert c > a + 10  # 1 vCPU / 2GiB を常時確保すると月 $19 前後


def test_extends_and_components_add_resolve():
    cfg = _cfg()
    sc = ce.resolve_scenario(cfg, "D_cloud_sql")
    assert "cloud_sql" in sc["components"] and "gcs" in sc["components"]
    assert sc["usage"]["low"]["cloud_sql"]["instances"] == 1
    # 基底の想定は変更されない
    base = ce.resolve_scenario(cfg, "A_budget_core")
    assert "cloud_sql" not in base["components"]


def test_usage_add_is_additive_for_jobs():
    cfg = _cfg()
    a = ce.resolve_scenario(cfg, "A_budget_core")["usage"]["low"]["cloud_run_jobs"]["vcpu_seconds"]
    b = ce.resolve_scenario(cfg, "B_scrape_on_cloud_run_jobs")["usage"]["low"]["cloud_run_jobs"]["vcpu_seconds"]
    assert b == a + 90000


def test_override_changes_price():
    cfg = ce.apply_overrides(_cfg(), ["prices.gcs_tokyo_standard_gb_month.value=0.046"])
    base = ce.estimate(_cfg(), "A_budget_core", "high")["items_usd"]["gcs"]
    new = ce.estimate(cfg, "A_budget_core", "high")["items_usd"]["gcs"]
    assert abs((new - base) - 60 * 0.023) < 1e-9  # 上限の保存量（実測 39GiB から成長を見込んだ 60GB）


def test_r2_is_free_within_free_tier():
    cfg = _cfg()
    assert ce.estimate(cfg, "A_budget_core", "high")["items_usd"]["cloudflare_r2"] == 0.0


def test_literal_all_gcp_exceeds_budget_by_a_lot():
    cfg = _cfg()
    r = ce.estimate(cfg, "F_literal_all_gcp", "high")
    assert not r["within_budget"]
    assert r["total_jpy"] > 3 * cfg["budget_jpy_per_month"]


def test_unverified_list_and_render():
    cfg = _cfg()
    assert any(x.startswith("prices.cloud_run_vcpu_second") for x in ce.unverified_prices(cfg))
    assert not any(x.startswith("prices.r2_storage_gb_month") for x in ce.unverified_prices(cfg))
    text = ce.render_table(cfg, ce.estimate_all(cfg, ["A_budget_core"]), "md")
    assert "A_budget_core" in text and "円/月" in text


# ── インスタンス課金（keiba-api は --no-cpu-throttling）と監視ポータルの増分 ──────────

def test_alive_seconds_ping_within_idle_timeout_means_always_on():
    cfg = _cfg()
    month = cfg["seconds_per_month"]
    assert ce._alive_seconds(cfg, {"alive_hours_per_day": 2}) == 2 * 3600 * 30
    assert ce._alive_seconds(cfg, {"alive_hours_per_day": 2, "pinger_interval_min": 5}) == month   # 5 分おきの ping で落ちない
    assert ce._alive_seconds(cfg, {"alive_hours_per_day": 2, "pinger_interval_min": 30}) == 2 * 3600 * 30  # 15 分より空けば落ちる
    assert ce._alive_seconds(cfg, {"alive_hours_per_day": 99}) == month


def test_heartbeat_to_instance_billed_api_blows_the_budget_but_monitor_ingest_does_not():
    cfg = _cfg()
    hb = ce.estimate(cfg, "H_heartbeat_on_instance_api", "low")
    mon = ce.estimate(cfg, "I_ingest_on_monitor", "low")
    assert not hb["within_budget"] and hb["items_usd"]["cloud_run_service"] * cfg["fx_jpy_per_usd"] > 5000
    assert mon["within_budget"] and mon["items_usd"]["cloud_run_service"] == 0.0


def test_monitor_features_add_little_over_the_base_plan():
    cfg = _cfg()
    for lv in ("low", "high"):
        base, plus = ce.estimate(cfg, "A_budget_core", lv), ce.estimate(cfg, "M0_monitor_features", lv)
        assert 0 < plus["total_jpy"] - base["total_jpy"] < 300


def test_instance_free_tier_zero_costs_more():
    cfg = _cfg()
    a = ce.estimate(cfg, "G_api_as_deployed", "low")["total_usd"]
    cfg2 = copy.deepcopy(cfg)
    cfg2["free_tier"]["cloud_run_instance_vcpu_seconds"]["value"] = 0
    cfg2["free_tier"]["cloud_run_instance_gib_seconds"]["value"] = 0
    assert ce.estimate(cfg2, "G_api_as_deployed", "low")["total_usd"] > a


def test_after_decoupling_mid_sits_between_low_and_high_and_cold_start_reads_dominate_gcs():
    cfg = _cfg()
    lo, hi = ce.estimate(cfg, "N_after_decoupling", "low"), ce.estimate(cfg, "N_after_decoupling", "high")
    mid = ce.estimate(cfg, "N_after_decoupling_mid", "low")
    assert lo["total_jpy"] < mid["total_jpy"] < hi["total_jpy"]
    cfg2 = copy.deepcopy(cfg)
    cfg2["scenarios"]["N_after_decoupling_mid"]["usage_override"]["low"]["gcs"]["class_b_ops"] = 950000   # 起動時の読み取りを約1,000件に減らした場合
    assert ce.estimate(cfg2, "N_after_decoupling_mid", "low")["total_jpy"] < mid["total_jpy"] - 400
