"""『prod で正常に動いているか』をジョブごとに機械判定して、ホストの cron を止めてよいジョブの合否表を出す。

    python -m src.scripts.ops.cron_cutover_status [--json] [--markdown] [--write] [--job ID …] [--now ISO]

判定の基準は ``config/gcp_job_map.json`` の ``cutover_defaults`` とジョブごとの ``cutover``（上書き）:
1. ``consecutive_success``   直近 ``min_consecutive_success`` 回（スキップ・待機は数えない）がすべて成功。件数が足りなければ『データ不足』で不合格。
2. ``start_delay``           その各回の開始が予定時刻から ``max_start_delay_sec`` 秒以内（GCP: 記録の scheduled_at、VPS: タスクの claimed_at − not_before）。
3. ``parity``                成果物があるジョブ（parity.kind ≠ none）は、``parity_min_days``（既定 3）回分の並走検証がすべて一致。
4. ``no_alerts``             直近 ``window_days`` 日にそのジョブの Slack 異常通知（記録 ``ops/alerts/``）が ``max_alerts``（既定 0）件以下。
5. ``raceday_t60_t15``       （race-odds-per-race のみ）直近 ``raceday_min_days`` 回の開催日で、**全レース**の T-60 / T-15 が成功し、失敗が 0。
ホストに残すジョブ（destination=host）は対象外（止めない）。``ready=false`` のジョブは『未着手』（理由つき）。
同じ cron_job_id を持つジョブ（例: 朝・夕の daily-race-lists）は、**全部合格したときだけ**その cron_job_id を止めてよい（cut_unit）。

結果の読み取りは GCS の ``ops/job_runs/``（GCP）、``ops/fetch_tasks/{done,failed}/``（VPS 実行役）、``ops/parity/``、``ops/alerts/``。
``--write`` で ``ops/cutover/status.json`` に残す。実行は読み取りのみ（ホスト・GCP の状態は変えない）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from src.ops import job_map
from src.ops.gcs_io import BlobIO, read_json, write_json

JST = ZoneInfo("Asia/Tokyo")
SUCCESS = {"ok", "partial"}
IGNORED = {"skipped", "waiting"}
STATUS_KEY = "ops/cutover/status.json"


# ───────────────────────── 記録の読み取り ─────────────────────────

class Records:
    """GCS の運用記録を読む（テストでは MemoryIO を渡す）。"""

    def __init__(self, io: BlobIO) -> None:
        self.io = io

    def gcp_runs(self, job_id: str, limit: int = 40) -> list[dict[str, Any]]:
        keys = [k for k in self.io.list(f"ops/job_runs/{job_id}/") if not k.endswith("latest.json")]
        out = []
        for k in sorted(keys, reverse=True)[:limit]:
            r = read_json(self.io, k)
            if isinstance(r, dict):
                out.append(r)
        return out

    def fetch_runs(self, job: dict[str, Any], limit: int = 60) -> list[dict[str, Any]]:
        """VPS 実行役の done / failed を新しい順に返す。定期ジョブは task_id が ``<job_id>-`` で始まるもの、
        レース別に予約されるジョブ（reserved_by）は task_id が ``<task名>-`` で始まるもの。"""
        prefixes = [f"{job['id']}-"] if not job.get("reserved_by") else [f"{t}-" for t in [job["fetch_task"]["task"], *(job.get("extra_tasks") or [])]]
        out = []
        for state in ("done", "failed"):
            for pre in prefixes:
                keys = self.io.list(f"ops/fetch_tasks/{state}/{pre}")
                for k in sorted(keys, reverse=True)[:limit]:
                    t = read_json(self.io, k)
                    if isinstance(t, dict):
                        out.append({"state": state, **t})
        out.sort(key=lambda t: t.get("not_before") or "", reverse=True)
        return out[:limit]

    def race_odds(self, tasks: list[str]) -> list[dict[str, Any]]:
        out = []
        for task in tasks:
            for state in ("done", "failed"):
                for k in self.io.list(f"ops/fetch_tasks/{state}/{task}-"):
                    t = read_json(self.io, k)
                    if isinstance(t, dict):
                        out.append({"state": state, **t})
        return out

    def parity(self, job_id: str, now: datetime, days: int) -> list[dict[str, Any]]:
        out = []
        for i in range(days):
            d = (now - timedelta(days=i)).strftime("%Y%m%d")
            r = read_json(self.io, f"ops/parity/{d}/{job_id}.json")
            if isinstance(r, dict):
                out.append(r)
        return out

    def alerts(self, now: datetime, days: int) -> list[dict[str, Any]]:
        out = []
        for i in range(days):
            d = (now - timedelta(days=i)).strftime("%Y%m%d")
            for k in self.io.list(f"ops/alerts/{d}/"):
                r = read_json(self.io, k)
                if isinstance(r, dict):
                    out.append(r)
        return out

    def expected_races(self, date: str) -> int | None:
        doc = read_json(self.io, f"state/data/page_reference/race_day_schedule/{date}.json")
        if isinstance(doc, dict) and doc.get("slots"):
            return len(doc["slots"])
        return None


# ───────────────────────── 判定 ─────────────────────────

def _check(name: str, ok: bool | None, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def _ts(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


def runs_view(job: dict[str, Any], rec: Records) -> list[dict[str, Any]]:
    """比較用の共通形 ``{ok, delay_sec, at}`` の新しい順（スキップ・待機は除く）。"""
    out: list[dict[str, Any]] = []
    if job["destination"] == "gcp":
        for r in rec.gcp_runs(job["id"]):
            if r.get("status") in IGNORED:
                continue
            s, sch = _ts(r.get("started_at")), _ts(r.get("scheduled_at"))
            out.append({"ok": r.get("status") in SUCCESS, "delay_sec": (s - sch).total_seconds() if s and sch else None,
                        "at": r.get("started_at"), "detail": r.get("reason")})
    elif job["destination"] == "vps":
        for t in rec.fetch_runs(job):
            c, nb = _ts(t.get("claimed_at")), _ts(t.get("not_before"))
            out.append({"ok": t.get("state") == "done", "delay_sec": (c - nb).total_seconds() if c and nb else None,
                        "at": t.get("claimed_at") or t.get("not_before"), "detail": (t.get("outcome") or {}).get("reason")})
    return out


def evaluate_job(job: dict[str, Any], doc: dict[str, Any], rec: Records, now: datetime, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    pol = job_map.cutover_policy(job, doc)
    row: dict[str, Any] = {"job": job["id"], "destination": job["destination"], "cut_unit": (job.get("origin") or {}).get("cron_job_id"),
                           "wave": job.get("wave"), "checks": []}
    if job["destination"] == "host":
        row.update(state="N/A", reason=f"ホストに残す: {job.get('reason', '')}", checks=[])
        return row
    if not job.get("enabled", True):
        row.update(state="無効", reason="enabled=false（ホストでも定期実行していない）")
        return row
    if job["destination"] == "retire":
        row.update(state="対象外", reason=f"廃止予定: {job.get('reason', '')}（依存が無くなってから止める）")
        return row
    if job.get("blockers") or not job.get("ready"):
        row.update(state="未着手", reason="; ".join(job.get("blockers") or ["ready=false"]))
        return row
    checks = row["checks"]
    n = int(pol["min_consecutive_success"])
    runs = runs_view(job, rec)
    if job["destination"] == "vps" and job.get("reserved_by") and job["fetch_task"]["task"].startswith("race-odds"):
        checks.append(_raceday(job, pol, rec, now))
    else:
        recent = runs[:n]
        if len(recent) < n:
            checks.append(_check("consecutive_success", False, f"データ不足（{len(recent)}/{n} 回）"))
        else:
            fails = [r for r in recent if not r["ok"]]
            checks.append(_check("consecutive_success", not fails, f"直近 {n} 回中 失敗 {len(fails)}" + (f"（{fails[0].get('detail')}）" if fails else "")))
        lim = float(pol["max_start_delay_sec"] if job["destination"] == "gcp" else pol.get("max_start_delay_sec_vps", 300))
        delays = [r["delay_sec"] for r in recent if r["delay_sec"] is not None]
        if len(recent) >= n and delays:
            worst = max(delays)
            checks.append(_check("start_delay", worst <= lim, f"最大の遅れ {worst:.0f}s（上限 {lim:.0f}s）"))
        else:
            checks.append(_check("start_delay", False if len(recent) < n else None, "遅れを測れる記録が無い" if len(recent) >= n else "データ不足"))
    par_kind = (job.get("parity") or {}).get("kind", "none")
    if job["destination"] == "gcp" and par_kind != "none" and pol.get("require_parity", True):
        k = int(pol.get("parity_min_days", 3))
        reps = rec.parity(job["id"], now, int(pol["window_days"]))
        good = [r for r in reps if r.get("ok")]
        bad = [r for r in reps if r.get("ok") is False]
        newest = sorted(reps, key=lambda r: r.get("date", ""), reverse=True)[:k]
        ok = len(newest) >= k and all(r.get("ok") for r in newest)
        checks.append(_check("parity", ok, f"直近の並走検証 {len(newest)}/{k} 回、一致 {sum(1 for r in newest if r.get('ok'))}（窓内 一致 {len(good)} / 不一致 {len(bad)}）"))
    ids = {job["id"], f"fetch:{(job.get('fetch_task') or {}).get('task')}"}
    mine = [a for a in alerts if a.get("job") in ids and a.get("severity") in ("warning", "critical")]
    mx = int(pol.get("max_alerts", 0))
    checks.append(_check("no_alerts", len(mine) <= mx, f"窓内の異常通知 {len(mine)} 件（上限 {mx}）" + (f"：{mine[0].get('title')}" if mine else "")))
    ok_all = all(c["ok"] for c in checks)
    row.update(state="合格" if ok_all else "不合格", reason=None if ok_all else "; ".join(f"{c['name']}: {c['detail']}" for c in checks if not c["ok"]))
    return row


def _raceday(job: dict[str, Any], pol: dict[str, Any], rec: Records, now: datetime) -> dict[str, Any]:
    """開催日の T-60 / T-15 が全レースで成功しているか（直近 raceday_min_days 回の開催日）。"""
    need = int(pol.get("raceday_min_days", 3))
    tasks = rec.race_odds(["race-odds-t60", "race-odds-t15"])
    by_date: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        by_date.setdefault(str((t.get("args") or {}).get("date")), []).append(t)
    days = sorted((d for d in by_date if d.isdigit()), reverse=True)[:need]
    if len(days) < need:
        return _check("raceday_t60_t15", False, f"開催日のデータ不足（{len(days)}/{need} 日）")
    problems = []
    for d in days:
        ts = by_date[d]
        failed = [t for t in ts if t["state"] == "failed"]
        races = {(t.get("args") or {}).get("race_id") for t in ts}
        exp = rec.expected_races(d) or len(races)
        done_ids = {(t["task"], (t.get("args") or {}).get("race_id")) for t in ts if t["state"] == "done"}
        missing = [r for r in races if ("race-odds-t60", r) not in done_ids or ("race-odds-t15", r) not in done_ids]
        if failed or missing or len(races) < exp:
            problems.append(f"{d}: レース {len(races)}/{exp}・失敗 {len(failed)}・未完了 {len(missing)}")
    return _check("raceday_t60_t15", not problems, "全レース成功" if not problems else " / ".join(problems))


def evaluate_all(doc: dict[str, Any], rec: Records, now: datetime, *, only: list[str] | None = None) -> dict[str, Any]:
    alerts = rec.alerts(now, int(doc["cutover_defaults"]["window_days"]))
    rows = []
    for j in doc["jobs"]:
        if only and j["id"] not in only:
            continue
        rows.append(evaluate_job(j, doc, rec, now, alerts))
    # cut_unit（同じ cron_job_id）は全部合格したときだけ止めてよい
    units: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("cut_unit") and r["destination"] in ("gcp", "vps"):
            units.setdefault(r["cut_unit"], []).append(r)
    cut = {}
    for u, rs in sorted(units.items()):
        cut[u] = {"ready_to_stop": all(x["state"] == "合格" for x in rs), "jobs": [x["job"] for x in rs],
                  "waiting_on": [x["job"] for x in rs if x["state"] != "合格"]}
    return {"evaluated_at": now.isoformat(timespec="seconds"), "rows": rows, "cut_units": cut,
            "summary": {s: sum(1 for r in rows if r["state"] == s) for s in sorted({r["state"] for r in rows})}}


def render_text(res: dict[str, Any]) -> str:
    lines = [f"切替判定 {res['evaluated_at']}  " + "  ".join(f"{k}:{v}" for k, v in res["summary"].items()), ""]
    w = max(len(r["job"]) for r in res["rows"])
    for r in sorted(res["rows"], key=lambda r: ({"合格": 0, "不合格": 1, "未着手": 2}.get(r["state"], 3), r.get("wave") or 9, r["job"])):
        lines.append(f"{r['state']:<4} {r['job']:<{w}}  {r.get('reason') or ''}")
    lines += ["", "ホストの cron を止めてよい cron_job_id（全ジョブ合格）:"]
    ok = [u for u, v in res["cut_units"].items() if v["ready_to_stop"]]
    lines.append("  " + (", ".join(ok) if ok else "（まだありません）"))
    return "\n".join(lines)


def render_markdown(res: dict[str, Any]) -> str:
    out = ["| 状態 | ジョブ | 移行先 | cron_job_id | 理由 |", "|---|---|---|---|---|"]
    for r in sorted(res["rows"], key=lambda r: ({"合格": 0, "不合格": 1, "未着手": 2}.get(r["state"], 3), r["job"])):
        out.append(f"| {r['state']} | {r['job']} | {r['destination']} | {r.get('cut_unit') or ''} | {(r.get('reason') or '').replace('|', '/')} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--now", default=None)
    ns = ap.parse_args(argv)
    from src.ops.gcs_io import default_io
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    io = default_io()
    now = datetime.fromisoformat(ns.now).astimezone(JST) if ns.now else datetime.now(JST)
    res = evaluate_all(job_map.load(), Records(io), now, only=ns.job or None)
    if ns.write:
        write_json(io, STATUS_KEY, res)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str) if ns.json else render_markdown(res) if ns.markdown else render_text(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
