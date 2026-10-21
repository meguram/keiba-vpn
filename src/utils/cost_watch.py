"""GCP 日次コストの『想定 vs 実績』と急増の検知（純粋な計算。GCP・Slack には触れない）。

入力は課金エクスポートから取った行 ``{"date","service","cost","currency"}``。
想定は ``config/migration_phases.json``（フェーズ別の月額範囲）と ``config/cost_assumptions.json``（案 A のサービス別上限）。
"""

from __future__ import annotations

import calendar
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PHASES_PATH = REPO / "config" / "migration_phases.json"
ASSUMPTIONS_PATH = REPO / "config" / "cost_assumptions.json"


def load_phases(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or PHASES_PATH).read_text(encoding="utf-8"))


def to_jpy(amount: float, currency: str, fx_jpy_per_usd: float) -> float:
    cur = (currency or "").upper()
    if cur == "JPY":
        return float(amount)
    if cur == "USD":
        return float(amount) * fx_jpy_per_usd
    raise ValueError(f"未対応の通貨: {currency!r}（JPY か USD のみ）")


def daily_table(rows: list[dict], fx: float) -> dict[str, dict[str, float]]:
    """{日付: {サービス: 円}}。同じ日・サービスは合算する。"""
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        d = str(r["date"])
        svc = str(r["service"])
        out.setdefault(d, {})
        out[d][svc] = out[d].get(svc, 0.0) + to_jpy(r["cost"], r.get("currency", "USD"), fx)
    return out


def _day(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def detect_spikes(table: dict[str, dict[str, float]], target: str, spike: dict) -> list[dict]:
    """サービス別: 当日が前日の ratio 倍以上、かつ増分が min_abs_jpy 以上なら警告（前日が floor 未満なら floor で割る）。
    7 日平均に対しても同じ判定をする。合計は日次の絶対しきい値（daily_total_alert_jpy）でも警告する。"""
    ratio = float(spike["ratio"])
    min_abs = float(spike["min_abs_jpy"])
    floor = float(spike["floor_jpy"])
    today = table.get(target, {})
    prev_key = (_day(target) - timedelta(days=1)).isoformat()
    prev = table.get(prev_key, {})
    hist_keys = [(_day(target) - timedelta(days=i)).isoformat() for i in range(1, 8)]
    out: list[dict] = []
    for svc, cost in sorted(today.items(), key=lambda kv: -kv[1]):
        base_prev = max(prev.get(svc, 0.0), floor)
        mean7 = sum(table.get(k, {}).get(svc, 0.0) for k in hist_keys) / 7.0
        base_mean = max(mean7, floor)
        if cost - prev.get(svc, 0.0) >= min_abs and cost >= ratio * base_prev:
            out.append({"service": svc, "kind": "前日比", "today": cost, "base": prev.get(svc, 0.0), "factor": cost / base_prev})
        elif cost - mean7 >= min_abs and cost >= ratio * base_mean:
            out.append({"service": svc, "kind": "7日平均比", "today": cost, "base": mean7, "factor": cost / base_mean})
    total = sum(today.values())
    limit = float(spike.get("daily_total_alert_jpy", 0) or 0)
    if limit and total >= limit:
        out.append({"service": "合計", "kind": "日次の上限目安", "today": total, "base": limit, "factor": total / limit})
    return out


def month_summary(table: dict[str, dict[str, float]], target: str, phases_cfg: dict) -> dict[str, Any]:
    """今月累計・月末見込み・フェーズの想定範囲との比較。"""
    t = _day(target)
    prefix = f"{t.year:04d}-{t.month:02d}-"
    mtd = sum(sum(v.values()) for d, v in table.items() if d.startswith(prefix) and d <= target)
    days_in_month = calendar.monthrange(t.year, t.month)[1]
    projected = mtd / t.day * days_in_month
    cur_id = int(phases_cfg["current_phase"])
    phase = next((p for p in phases_cfg["phases"] if int(p["id"]) == cur_id), None)
    lo = float(phase["expected_min_jpy"]) if phase else 0.0
    hi = float(phase["expected_max_jpy"]) if phase else 0.0
    if phase is None:
        status = "フェーズ不明"
    elif projected > hi:
        status = "上限超過"
    elif projected < lo:
        status = "下限未満（取得・ジョブが動いていない可能性）"
    else:
        status = "範囲内"
    budget = float(phases_cfg.get("budget_jpy_per_month", 0) or 0)
    return {
        "mtd": mtd,
        "projected": projected,
        "days_elapsed": t.day,
        "days_in_month": days_in_month,
        "phase_id": cur_id,
        "phase_name": phase["name"] if phase else "",
        "expected_min": lo,
        "expected_max": hi,
        "status": status,
        "budget": budget,
        "budget_pct": (mtd / budget * 100.0) if budget else 0.0,
        "early_month": t.day < 3,
    }


def service_ceilings(phases_cfg: dict, assumptions: dict | None = None) -> dict[str, float]:
    """課金エクスポートのサービス名 → 案 A 上限（円/月）。対応表にあるものだけ。"""
    from src.scripts.ops import cost_estimate as ce

    cfg = assumptions or ce.load_config()
    est = ce.estimate(cfg, phases_cfg.get("_meta", {}).get("scenario", "A_budget_core"), "high")
    fx = float(cfg["fx_jpy_per_usd"])
    out: dict[str, float] = {}
    for svc, comps in (phases_cfg.get("service_to_component") or {}).items():
        out[svc] = sum(est["items_usd"].get(c, 0.0) for c in comps) * fx
    return out


def service_projection(table: dict[str, dict[str, float]], target: str) -> dict[str, float]:
    """サービス別の月末見込み（円）。"""
    t = _day(target)
    prefix = f"{t.year:04d}-{t.month:02d}-"
    dim = calendar.monthrange(t.year, t.month)[1]
    acc: dict[str, float] = {}
    for d, v in table.items():
        if d.startswith(prefix) and d <= target:
            for s, c in v.items():
                acc[s] = acc.get(s, 0.0) + c
    return {s: c / t.day * dim for s, c in acc.items()}


def over_ceiling(table: dict[str, dict[str, float]], target: str, ceilings: dict[str, float]) -> list[dict]:
    """月末見込みが案 A の上限を超えそうなサービス（月初 2 日までは出さない）。"""
    if _day(target).day < 3:
        return []
    out = []
    for svc, proj in sorted(service_projection(table, target).items(), key=lambda kv: -kv[1]):
        cap = ceilings.get(svc)
        if cap is not None and proj > cap and proj - cap >= 30:
            out.append({"service": svc, "projected": proj, "ceiling": cap})
    return out


def format_summary_lines(s: dict[str, Any]) -> str:
    note = "（月初のため目安）" if s["early_month"] else ""
    return (
        f"今月累計 ¥{s['mtd']:,.0f} / 月末見込み ¥{s['projected']:,.0f}{note}"
        f"（予算 ¥{s['budget']:,.0f} の {s['budget_pct']:.0f}%）\n"
        f"フェーズ{s['phase_id']}「{s['phase_name']}」の想定 ¥{s['expected_min']:,.0f}〜¥{s['expected_max']:,.0f}/月 → {s['status']}"
    )


def format_spike_alert(target: str, spikes: list[dict], ceilings_over: list[dict]) -> str:
    lines = [f"[GCPコスト警告] {target}: 費用の急増・想定超えを検知しました"]
    for sp in spikes:
        lines.append(
            f"- {sp['service']}: ¥{sp['today']:,.0f}（{sp['kind']} {sp['factor']:.1f} 倍、基準 ¥{sp['base']:,.0f}）"
        )
    for o in ceilings_over:
        lines.append(f"- {o['service']}: 月末見込み ¥{o['projected']:,.0f} が案Aの上限 ¥{o['ceiling']:,.0f} を超過")
    lines.append("確認: Console の Billing レポート／直前のデプロイ・ジョブの暴走（max-instances・リトライ）を見る")
    return "\n".join(lines)
