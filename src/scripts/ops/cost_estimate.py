"""移行後の月額コスト見積もり。前提は config/cost_assumptions.json（数値を変えて再計算できる）。

使い方:
  python -m src.scripts.ops.cost_estimate --scenario all
  python -m src.scripts.ops.cost_estimate --scenario A_budget_core --fx 160 --format md
  python -m src.scripts.ops.cost_estimate --scenario all --set prices.gcs_tokyo_standard_gb_month.value=0.025

簡略化: Cloud Run の無料枠(vCPU 秒・GiB 秒)は『サービス→ジョブ』の順に共通枠として消費する。
インスタンス課金（billing=instance。デプロイ済みの keiba-api は --no-cpu-throttling）は『生きている秒数』で課金し、ping が 15 分以内の間隔で来ると 24 時間分になる。
GCP には接続しない(純粋な計算)。実績は gcp_daily_cost_report.py の課金エクスポートで置き換える。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "config" / "cost_assumptions.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def _p(cfg: dict, key: str) -> float:
    return float(cfg["prices"][key]["value"])


def _f(cfg: dict, key: str) -> float:
    return float(cfg["free_tier"][key]["value"])


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """--set a.b.c=1.5 を設定に反映する(パスは JSON のキー、最後は数値)。"""
    out = copy.deepcopy(cfg)
    for item in sets:
        path, _, val = item.partition("=")
        keys = path.split(".")
        node = out
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = float(val)
    return out


def _merge_add(base: dict, add: dict) -> dict:
    out = copy.deepcopy(base)
    for comp, vals in add.items():
        cur = out.setdefault(comp, {})
        for k, v in vals.items():
            cur[k] = cur.get(k, 0) + v
    return out


def _merge_override(base: dict, ov: dict) -> dict:
    out = copy.deepcopy(base)
    for comp, vals in ov.items():
        out.setdefault(comp, {}).update(vals)
    return out


def resolve_scenario(cfg: dict, name: str) -> dict:
    """extends / components_add / usage_add / usage_override を解決して {components, usage:{low,high}} を返す。"""
    sc = cfg["scenarios"][name]
    if "extends" in sc:
        base = resolve_scenario(cfg, sc["extends"])
    else:
        base = {"components": list(sc.get("components", [])), "usage": copy.deepcopy(sc.get("usage", {"low": {}, "high": {}}))}
    comps = list(base["components"])
    for c in sc.get("components_add", []):
        if c not in comps:
            comps.append(c)
    usage = {lv: copy.deepcopy(base["usage"].get(lv, {})) for lv in ("low", "high")}
    for lv in ("low", "high"):
        if "usage_add" in sc:
            usage[lv] = _merge_add(usage[lv], sc["usage_add"].get(lv, {}))
        if "usage_override" in sc:
            usage[lv] = _merge_override(usage[lv], sc["usage_override"].get(lv, {}))
    return {"components": comps, "usage": usage, "description": sc.get("description", "")}


# ───────────────────────── 各サービスの計算(USD/月) ─────────────────────────

def _alive_seconds(cfg: dict, svc: dict) -> float:
    """インスタンス課金（--no-cpu-throttling）のサービスが『生きている』秒数。

    課金されるのはリクエストの処理時間ではなく、インスタンスが起動している間（最後のリクエストから idle_timeout_min 分後まで）。
    ping（ハートビート等）の間隔が idle_timeout_min 以下なら、最小インスタンス 0 でも落ちずに 1 か月ずっと生きる。
    """
    month = float(cfg["seconds_per_month"])
    ping = float(svc.get("pinger_interval_min", 0))
    if ping and ping <= float(svc.get("idle_timeout_min", 15)):
        return month
    return min(month, float(svc.get("alive_hours_per_day", 0)) * 3600 * 30)


def _cloud_run(cfg: dict, svc: dict, jobs: dict, monitor: dict | None = None) -> dict[str, float]:
    """サービス・監視用サービス（リクエスト課金 + 最小インスタンスのアイドル）とジョブを計算する。

    svc.billing == "instance" のサービスは、インスタンスが生きている秒数で課金し、専用の無料枠を使う。
    リクエスト課金の無料枠は『監視用サービス（billing=request）→ ジョブ』の順に共通枠として消費する。
    """
    sec = float(cfg["seconds_per_month"])
    instance_mode = svc.get("billing") == "instance"
    mon = monitor or {}
    pool = mon if instance_mode else svc
    req = float(pool.get("requests", 0))
    vcpu, gib = float(pool.get("vcpu", 1)), float(pool.get("gib", 1))
    active = req * float(pool.get("avg_seconds", 0)) + float(pool.get("cold_starts", 0)) * float(pool.get("cold_seconds", 0))
    if not instance_mode and mon:  # リクエスト課金のサービスに監視用サービスを足す
        mreq = float(mon.get("requests", 0))
        active += mreq * float(mon.get("avg_seconds", 0)) + float(mon.get("cold_starts", 0)) * float(mon.get("cold_seconds", 0))
        req += mreq
    svc_cpu, svc_mem = active * vcpu, active * gib
    free_cpu, free_mem = _f(cfg, "cloud_run_vcpu_seconds"), _f(cfg, "cloud_run_gib_seconds")
    svc_cpu_bill = max(0.0, svc_cpu - free_cpu)
    svc_mem_bill = max(0.0, svc_mem - free_mem)
    free_cpu_left, free_mem_left = max(0.0, free_cpu - svc_cpu), max(0.0, free_mem - svc_mem)
    job_cpu, job_mem = float(jobs.get("vcpu_seconds", 0)), float(jobs.get("gib_seconds", 0))
    job_cpu_bill, job_mem_bill = max(0.0, job_cpu - free_cpu_left), max(0.0, job_mem - free_mem_left)
    req_bill = max(0.0, req - _f(cfg, "cloud_run_requests")) / 1e6 * _p(cfg, "cloud_run_request_per_million")
    idle = 0.0
    service_cost = svc_cpu_bill * _p(cfg, "cloud_run_vcpu_second") + svc_mem_bill * _p(cfg, "cloud_run_gib_second") + req_bill
    if instance_mode:
        alive = _alive_seconds(cfg, svc)
        s_v, s_g = float(svc.get("vcpu", 1)), float(svc.get("gib", 1))
        api = (max(0.0, alive * s_v - _f(cfg, "cloud_run_instance_vcpu_seconds")) * _p(cfg, "cloud_run_instance_vcpu_second")
               + max(0.0, alive * s_g - _f(cfg, "cloud_run_instance_gib_seconds")) * _p(cfg, "cloud_run_instance_gib_second"))
        service_cost += api
    else:
        minimum = float(svc.get("min_instances", 0))
        if minimum > 0:
            idle_seconds = max(0.0, sec - active / max(minimum, 1))
            idle = minimum * idle_seconds * (vcpu * _p(cfg, "cloud_run_idle_vcpu_second") + gib * _p(cfg, "cloud_run_idle_gib_second"))
    return {
        "cloud_run_service": service_cost + idle,
        "cloud_run_jobs": job_cpu_bill * _p(cfg, "cloud_run_job_vcpu_second") + job_mem_bill * _p(cfg, "cloud_run_job_gib_second"),
    }


def _gcs(cfg: dict, u: dict) -> float:
    storage = float(u.get("stored_gb", 0)) * _p(cfg, "gcs_tokyo_standard_gb_month")
    ops = float(u.get("class_a_ops", 0)) / 1000 * _p(cfg, "gcs_class_a_per_1000") + float(u.get("class_b_ops", 0)) / 1000 * _p(cfg, "gcs_class_b_per_1000")
    egress = float(u.get("egress_gb", 0)) * _p(cfg, "internet_egress_gb")
    return storage + ops + egress


def _r2(cfg: dict, u: dict) -> float:
    storage = max(0.0, float(u.get("stored_gb", 0)) - _f(cfg, "r2_storage_gb")) * _p(cfg, "r2_storage_gb_month")
    a = max(0.0, float(u.get("class_a_ops", 0)) - _f(cfg, "r2_class_a")) / 1e6 * _p(cfg, "r2_class_a_per_million")
    b = max(0.0, float(u.get("class_b_ops", 0)) - _f(cfg, "r2_class_b")) / 1e6 * _p(cfg, "r2_class_b_per_million")
    return storage + a + b


def _gce(cfg: dict, u: dict) -> float:
    n = float(u.get("instances", 0))
    hours = float(cfg["seconds_per_month"]) / 3600
    return n * (hours * _p(cfg, "gce_e2_micro_tokyo_hour") + hours * _p(cfg, "gce_external_ipv4_hour") + float(u.get("disk_gb", 20)) * _p(cfg, "gce_pd_standard_gb_month"))


def estimate(cfg: dict, name: str, level: str) -> dict[str, Any]:
    sc = resolve_scenario(cfg, name)
    use = sc["usage"][level]
    items: dict[str, float] = {}
    comps = sc["components"]
    if "cloud_run_service" in comps or "cloud_run_jobs" in comps:
        items.update(_cloud_run(cfg, use.get("cloud_run_service", {}), use.get("cloud_run_jobs", {}), use.get("cloud_run_monitor", {})))
    if "gcs" in comps:
        items["gcs"] = _gcs(cfg, use.get("gcs", {}))
    if "scheduler" in comps:
        items["scheduler"] = max(0.0, float(use.get("scheduler", {}).get("jobs", 0)) - _f(cfg, "scheduler_jobs")) * _p(cfg, "scheduler_job_month")
    if "secret_manager" in comps:
        items["secret_manager"] = max(0.0, float(use.get("secret_manager", {}).get("versions", 0)) - _f(cfg, "secret_manager_versions")) * _p(cfg, "secret_manager_version_month")
    if "artifact_registry" in comps:
        items["artifact_registry"] = max(0.0, float(use.get("artifact_registry", {}).get("gb", 0)) - _f(cfg, "artifact_registry_gb")) * _p(cfg, "artifact_registry_gb_month")
    if "logging" in comps:
        items["logging"] = max(0.0, float(use.get("logging", {}).get("gib", 0)) - _f(cfg, "logging_gib")) * _p(cfg, "logging_gib")
    if "cloud_sql" in comps:
        items["cloud_sql"] = float(use.get("cloud_sql", {}).get("instances", 0)) * _p(cfg, "cloud_sql_f1_micro_month")
    if "gce" in comps:
        items["gce"] = _gce(cfg, use.get("gce", {}))
    if "cloud_nat" in comps:
        items["cloud_nat"] = float(use.get("cloud_nat", {}).get("enabled", 0)) * _p(cfg, "cloud_nat_month")
    if "cloudflare_r2" in comps:
        items["cloudflare_r2"] = _r2(cfg, use.get("cloudflare_r2", {}))
    total_usd = sum(items.values())
    fx = float(cfg["fx_jpy_per_usd"])
    return {
        "scenario": name, "level": level, "items_usd": items,
        "total_usd": total_usd, "total_jpy": total_usd * fx,
        "within_budget": total_usd * fx <= float(cfg["budget_jpy_per_month"]),
    }


def estimate_all(cfg: dict, names: list[str] | None = None) -> list[dict[str, Any]]:
    names = names or list(cfg["scenarios"])
    return [estimate(cfg, n, lv) for n in names for lv in ("low", "high")]


def render_table(cfg: dict, results: list[dict], fmt: str = "table") -> str:
    fx = float(cfg["fx_jpy_per_usd"])
    budget = float(cfg["budget_jpy_per_month"])
    lines: list[str] = []
    by: dict[str, dict[str, dict]] = {}
    for r in results:
        by.setdefault(r["scenario"], {})[r["level"]] = r
    if fmt == "md":
        lines.append(f"為替 1 ドル={fx:g} 円、予算 {budget:,.0f} 円/月\n")
        lines.append("| 案 | 下限(円/月) | 上限(円/月) | 予算内 |")
        lines.append("|---|---:|---:|---|")
        for name, lv in by.items():
            lo, hi = lv["low"], lv["high"]
            ok = "○" if hi["within_budget"] else ("△(下限のみ)" if lo["within_budget"] else "×")
            lines.append(f"| {name} | {lo['total_jpy']:,.0f} | {hi['total_jpy']:,.0f} | {ok} |")
        lines.append("")
        for name, lv in by.items():
            keys = sorted(set(lv["low"]["items_usd"]) | set(lv["high"]["items_usd"]))
            lines.append(f"**{name}** の内訳(円/月)")
            lines.append("")
            lines.append("| 項目 | 下限 | 上限 |")
            lines.append("|---|---:|---:|")
            for k in keys:
                lines.append(f"| {k} | {lv['low']['items_usd'].get(k, 0) * fx:,.0f} | {lv['high']['items_usd'].get(k, 0) * fx:,.0f} |")
            lines.append("")
        return "\n".join(lines)
    lines.append(f"為替 1 ドル={fx:g} 円 / 予算 {budget:,.0f} 円/月")
    for name, lv in by.items():
        lines.append(f"\n== {name}: 下限 {lv['low']['total_jpy']:,.0f} 円 / 上限 {lv['high']['total_jpy']:,.0f} 円")
        keys = sorted(set(lv["low"]["items_usd"]) | set(lv["high"]["items_usd"]))
        for k in keys:
            lines.append(f"   {k:<20s} {lv['low']['items_usd'].get(k, 0) * fx:>8,.0f} {lv['high']['items_usd'].get(k, 0) * fx:>8,.0f}")
    return "\n".join(lines)


def unverified_prices(cfg: dict) -> list[str]:
    """公式ページで未確認の単価・無料枠の一覧(要確認として文書に出す)。"""
    out = [f"prices.{k}" for k, v in cfg["prices"].items() if not v.get("verified")]
    out += [f"free_tier.{k}" for k, v in cfg["free_tier"].items() if not v.get("verified")]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--set", action="append", default=[], help="prices.<key>.value=1.5 のように前提を上書き")
    ap.add_argument("--format", choices=["table", "md", "json"], default="table")
    ap.add_argument("--list-unverified", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.fx:
        cfg["fx_jpy_per_usd"] = a.fx
    cfg = apply_overrides(cfg, a.set)
    if a.list_unverified:
        print("\n".join(unverified_prices(cfg)))
        return 0
    names = None if a.scenario == "all" else [a.scenario]
    results = estimate_all(cfg, names)
    print(json.dumps(results, ensure_ascii=False, indent=1) if a.format == "json" else render_table(cfg, results, a.format))
    return 0


if __name__ == "__main__":
    sys.exit(main())
