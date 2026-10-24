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
    assert abs((new - base) - 100 * 0.023) < 1e-9


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
