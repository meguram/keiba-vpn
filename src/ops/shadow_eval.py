"""影モード（VPS の取得をホストと並走で検証）の合否判定。日次レポート（``ops/fetch_shadow/daily/<日付>.json``）を読んで、
ジョブごと・cron_job_id（止める単位）ごとに『担当を渡してよいか・ホストの cron を止めてよいか』を返す。

基準（``config/fetch_shadow.json`` の ``pass``）: 直近 ``required_days`` 日（昨日まで）がすべて合格で、開催日を ``require_raceday_days`` 日以上含み、
そのジョブのタスクが ``min_task_days`` 日以上実行され、各日の取得成功率・比較一致率・異常通知ゼロを満たすこと。**読めない・足りないときは不合格**（fail-closed）。

使う場所: ``cron_cutover_status``（合否表）/ ``cutover_plan make``（適用スクリプトの生成）/ ``job_owner set``・``can-stop``（担当の移譲・crontab 停止）/
``fetch_shadow can-switch``。``cutover_defaults.require_shadow`` が true の VPS ジョブにだけ課す。ジョブの ``shadow.exempt``（理由つき）で明示的に免除できる。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
REPORT_KEY = "ops/fetch_shadow/daily/{date}.json"


def job_tasks(job: dict[str, Any]) -> list[str]:
    ft = (job.get("fetch_task") or {}).get("task")
    return [t for t in [ft, *(job.get("extra_tasks") or [])] if t]


def window_dates(today: datetime, days: int) -> list[str]:
    """昨日から遡った ``days`` 日（新しい順）。当日分はまだ途中なので含めない。"""
    t = today.astimezone(JST)
    return [(t - timedelta(days=i + 1)).strftime("%Y%m%d") for i in range(days)]


def load_reports(io: Any, today: datetime, days: int) -> dict[str, dict[str, Any]]:
    from src.ops.gcs_io import read_json

    out: dict[str, dict[str, Any]] = {}
    for d in window_dates(today, days):
        r = read_json(io, REPORT_KEY.format(date=d))
        if isinstance(r, dict):
            out[d] = r
    return out


def _check(ok: bool | None, detail: str) -> dict[str, Any]:
    return {"name": "shadow_days", "ok": ok, "detail": detail}


def day_verdict(rep: dict[str, Any], passing: dict[str, Any]) -> tuple[bool, list[str]]:
    """1 日分のレポートが基準を満たすか（レポート作成時に ok / reasons として書く）。"""
    reasons: list[str] = []
    planned = int(rep.get("planned", 0))
    if planned <= 0:
        reasons.append("その日の影のタスクが 0 件（計画が積まれていない）")
        return False, reasons
    executed = int(rep.get("done_ok", 0)) + int(rep.get("failed", 0))
    if executed <= 0:
        reasons.append("実行されたタスクが 0 件")
    else:
        rate = float(rep.get("success_rate", 0.0))
        if rate < float(passing.get("min_success_rate", 0.95)):
            reasons.append(f"取得成功率 {rate:.1%} が基準 {float(passing.get('min_success_rate', 0.95)):.0%} 未満")
    sk = int(rep.get("skipped", 0)) / planned
    if sk > float(passing.get("max_skipped_ratio", 0.1)):
        reasons.append(f"上限超過で取得しなかった割合 {sk:.0%}（上限 {float(passing.get('max_skipped_ratio', 0.1)):.0%}）")
    if int(rep.get("compare_expected", 0)) > 0 and int(rep.get("compared", 0)) == 0:
        reasons.append("比較できる成果物が無い（影の保存か本番データが見つからない）")
    if int(rep.get("compared", 0)) > 0:
        mr = float(rep.get("match_ratio", 0.0))
        if mr < float(passing.get("min_match_ratio", 0.95)):
            reasons.append(f"本番との一致率 {mr:.1%} が基準 {float(passing.get('min_match_ratio', 0.95)):.0%} 未満")
    if int(rep.get("alerts", 0)) > int(passing.get("max_alerts", 0)):
        reasons.append(f"異常通知・アクセス制限の疑い {rep.get('alerts')} 件（上限 {passing.get('max_alerts', 0)}）")
    return not reasons, reasons


def evaluate_job(job: dict[str, Any], doc: dict[str, Any], reports: dict[str, dict[str, Any]], today: datetime, cfg: Any) -> dict[str, Any] | None:
    """VPS ジョブ 1 件の ``shadow_days`` チェック。要求されない（``require_shadow`` が false）ときは None。"""
    from src.ops import job_map

    pol = job_map.cutover_policy(job, doc)
    if job.get("destination") != "vps" or not pol.get("require_shadow", False):
        return None
    sh = job.get("shadow") or {}
    if sh.get("exempt"):
        return _check(True, f"影モードを免除（承認済み: {sh.get('reason') or '理由の記載なし'}）")
    tasks = job_tasks(job)
    names = cfg.task_names()
    out_of_scope = [t for t in tasks if t not in names]
    if out_of_scope:
        return _check(False, f"影モードの対象外のタスク {out_of_scope}（重い・副作用が大きい取得）。config/fetch_shadow.json の tasks に加えるか、"
                              "ジョブの shadow.exempt に理由を書いて承認するまで、止められない")
    p = cfg.passing
    need = int(p.get("required_days", 5))
    dates = window_dates(today, need)
    missing = [d for d in dates if d not in reports]
    if missing:
        return _check(False, f"影モードの日次レポートが足りない（{len(dates) - len(missing)}/{need} 日。欠け: {', '.join(sorted(missing))}）")
    bad = [(d, reports[d].get("reasons") or ["不合格"]) for d in dates if not reports[d].get("ok")]
    if bad:
        d, r = bad[0]
        return _check(False, f"直近 {need} 日のうち不合格 {len(bad)} 日（例 {d}: {'; '.join(r)[:120]}）")
    racedays = sum(1 for d in dates if reports[d].get("is_raceday"))
    need_race = int(p.get("require_raceday_days", 1))
    if racedays < need_race:
        return _check(False, f"開催日が窓の中に {racedays}/{need_race} 日（開催日の取得を見届けるまで待つ）")
    for t in tasks:
        days_with = [d for d in dates if int(((reports[d].get("per_task") or {}).get(t) or {}).get("done_ok", 0)) > 0]
        mind = int((cfg.tasks.get(t) or {}).get("min_task_days", 1))
        if len(days_with) < mind:
            return _check(False, f"{t} が実行された日が {len(days_with)}/{mind} 日")
        for d in days_with:
            pt = reports[d]["per_task"][t]
            if not pt.get("ok", True):
                return _check(False, f"{t} が {d} に不合格（{pt.get('reason') or '成功率・一致率'}）")
    return _check(True, f"直近 {need} 日すべて合格・開催日 {racedays} 日を含む")


def unit_gate(unit_id: str, doc: dict[str, Any], reports: dict[str, dict[str, Any]], today: datetime, cfg: Any) -> tuple[bool, list[str]]:
    """cron_job_id（止める単位）に含まれる VPS ジョブがすべて基準を満たすか。VPS ジョブが無い・要求されない単位は合格。"""
    reasons: list[str] = []
    for j in doc.get("jobs") or []:
        if (j.get("origin") or {}).get("cron_job_id") != unit_id:
            continue
        c = evaluate_job(j, doc, reports, today, cfg)
        if c is not None and not c["ok"]:
            reasons.append(f"{j['id']}: {c['detail']}")
    return not reasons, reasons


def unit_gate_from_env(unit_id: str, *, now: datetime | None = None) -> tuple[bool, list[str]]:
    """GCS の日次レポートを読んで判定する。読めない・設定が無いときは不合格（fail-closed）。"""
    from src.ops import fetch_shadow, job_map

    try:
        doc = job_map.load()
        cfg = fetch_shadow.load_config()
        needs = any((j.get("origin") or {}).get("cron_job_id") == unit_id and j.get("destination") == "vps"
                    and job_map.cutover_policy(j, doc).get("require_shadow", False) for j in doc.get("jobs") or [])
        if not needs:
            return True, []
        override = os.environ.get("KEIBA_FETCH_SHADOW_REPORT_DIR")
        if override:                        # 検証用: ローカルのディレクトリ（<日付>.json）から読む
            reports = {}
            now_ = now or datetime.now(JST)
            for d in window_dates(now_, int(cfg.passing.get("required_days", 5))):
                try:
                    reports[d] = json.loads((Path(override) / f"{d}.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            return unit_gate(unit_id, doc, reports, now_, cfg)
        from src.ops.gcs_io import default_io
        from src.utils.project_env import load_project_dotenv

        load_project_dotenv()
        now_ = now or datetime.now(JST)
        reports = load_reports(default_io(), now_, int(cfg.passing.get("required_days", 5)))
        return unit_gate(unit_id, doc, reports, now_, cfg)
    except Exception as exc:  # noqa: BLE001
        return False, [f"影モードの日次レポートを読めない（安全側で不合格）: {type(exc).__name__}: {exc}"]


def status_gate(status: dict[str, Any] | None, unit_id: str, doc: dict[str, Any]) -> tuple[bool, list[str]]:
    """``cron_cutover_status --json`` の出力から、止める単位の影モード判定を読む。status が無い・shadow_days が無いときは不合格。"""
    from src.ops import job_map

    mine = [j for j in doc.get("jobs") or [] if (j.get("origin") or {}).get("cron_job_id") == unit_id and j.get("destination") == "vps"
            and job_map.cutover_policy(j, doc).get("require_shadow", False)]
    if not mine:
        return True, []
    if not status:
        return False, ["切替判定の結果（--status-file）が無いため、影モードの基準を確認できない"]
    rows = {r["job"]: r for r in status.get("rows") or []}
    reasons: list[str] = []
    for j in mine:
        chk = next((c for c in (rows.get(j["id"]) or {}).get("checks") or [] if c.get("name") == "shadow_days"), None)
        if chk is None:
            reasons.append(f"{j['id']}: 切替判定に shadow_days の結果が無い")
        elif not chk.get("ok"):
            reasons.append(f"{j['id']}: {chk.get('detail')}")
    return not reasons, reasons
