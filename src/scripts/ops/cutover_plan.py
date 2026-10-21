"""ホストの cron を止める手順を、cron_job_id（= 止める単位 cut_unit）ごとの適用スクリプトとして /tmp に用意する。実行はしない（ユーザーが root で実行）。

    python -m src.scripts.ops.cutover_plan list
    python -m src.scripts.ops.cutover_plan make --unit drift-monitor-weekly [--status-file status.json | --force] [--out-dir /tmp] [--project-dir /home/jovyan/work/keiba-vpn]

できるスクリプト（どれも冪等・バックアップ付き。順に実行する）:
1. ``cutover_<unit>_1_owner.sh``   担当を GCP / VPS に渡す（``config/job_owner.json``。VPS の取得は ``config/fetch_owner.json`` の agent_tasks にも足す）。
                                   これでホストの cron は**何もしなくなる**（crontab の行は残る）＝ GCP と二重に動く瞬間が無い。
2. （1〜2 週間の観察。``cron_cutover_status`` が合格のまま、Slack 異常通知が無いこと）
3. ``cutover_<unit>_2_cron_off.sh``  root で crontab の該当行をコメントアウト（``host_cron_switch.sh disable``。退避付き）。
4. ``cutover_<unit>_rollback.sh``    いつでも戻す: 行を戻す → 担当を外す（この順序。戻す間にホストが先に動かないよう、行を戻してから担当を外す）。

ホストに残すジョブ（destination=host。バックフィル・モデリング・遠隔アクセス・stg のデプロイなど）は対象外で、作れない。
合格していない単位は ``--force`` が無い限り作らない（``--status-file`` は ``cron_cutover_status --json`` の出力）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.ops import job_map

DEFAULT_PROJECT_DIR = "/home/jovyan/work/keiba-vpn"


def units(doc: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """cron_job_id → {jobs, destination, tags, wave}（host / 廃止予定の依存付きは除く）。"""
    doc = doc or job_map.load()
    out: dict[str, dict[str, Any]] = {}
    for j in doc["jobs"]:
        cid = (j.get("origin") or {}).get("cron_job_id")
        if not cid:
            continue
        u = out.setdefault(cid, {"jobs": [], "destinations": set(), "tags": set(), "wave": 0})
        u["jobs"].append(j["id"])
        u["destinations"].add(j["destination"])
        if (j["origin"] or {}).get("tag"):
            u["tags"].add(j["origin"]["tag"])
        u["wave"] = max(u["wave"], j.get("wave") or 0)
    return out


def owner_of(unit: dict[str, Any]) -> str | None:
    d = unit["destinations"]
    if "host" in d:
        return None
    if d == {"gcp"}:
        return "gcp"
    if d == {"vps"}:
        return "vps"
    if d == {"retire"}:
        return "retire"
    return "gcp" if "gcp" in d else "vps" if "vps" in d else None


def vps_tasks(unit_jobs: list[str], doc: dict[str, Any]) -> list[str]:
    tasks: list[str] = []
    for jid in unit_jobs:
        j = job_map.job_by_id(jid, doc)
        if j and j["destination"] == "vps":
            tasks += [j["fetch_task"]["task"], *(j.get("extra_tasks") or [])]
    return sorted(set(tasks))


HEADER = """#!/usr/bin/env bash
# {title}
# 生成: {stamp}（src/scripts/ops/cutover_plan.py）。単位: {unit}（含むジョブ: {jobs}）
set -euo pipefail
PROJECT="${{KEIBA_PROJECT_DIR:-{project}}}"
cd "$PROJECT"
BK="$PROJECT/data/local/meta/cron_backups"; mkdir -p "$BK"; TS="$(date +%Y%m%d_%H%M%S)"
"""

PY_ADD_AGENT = """python3 - "$PROJECT/config/fetch_owner.json" {tasks} <<'PY'
import json, sys
p, tasks = sys.argv[1], sys.argv[2:]
d = json.load(open(p, encoding="utf-8"))
cur = list(d.get("agent_tasks") or [])
for t in tasks:
    if t not in cur:
        cur.append(t)
d["agent_tasks"] = cur
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print("fetch_owner agent_tasks =", cur)
PY
"""
PY_DEL_AGENT = PY_ADD_AGENT.replace("for t in tasks:\n    if t not in cur:\n        cur.append(t)", "cur = [t for t in cur if t not in tasks]")


def render(unit_id: str, doc: dict[str, Any] | None = None, *, project_dir: str = DEFAULT_PROJECT_DIR, now: datetime | None = None) -> dict[str, str]:
    doc = doc or job_map.load()
    u = units(doc).get(unit_id)
    if u is None:
        raise ValueError(f"未知の cron_job_id: {unit_id}")
    owner = owner_of(u)
    if owner is None:
        raise ValueError(f"{unit_id} はホストに残すジョブを含む（止められない）: {sorted(u['jobs'])}")
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    tags = sorted(u["tags"]) or [""]
    tag_args = " ".join(f"--tag {t}" for t in tags[:1] if t)
    v_tasks = vps_tasks(u["jobs"], doc)
    meta = dict(stamp=stamp, unit=unit_id, jobs=",".join(sorted(u["jobs"])), project=project_dir)
    s1 = HEADER.format(title=f"切替 1/2: {unit_id} の担当を {owner} に渡す（ホストの cron は何もしなくなる。crontab の行は残る）", **meta)
    s1 += 'cp config/job_owner.json "$BK/job_owner.before_' + unit_id + '_$TS.json"\n'
    if v_tasks:
        s1 += 'cp config/fetch_owner.json "$BK/fetch_owner.before_' + unit_id + '_$TS.json"\n' + PY_ADD_AGENT.format(tasks=" ".join(v_tasks))
    s1 += f'PYTHONPATH=. python3 -m src.ops.job_owner set {unit_id} {owner}\n'
    s1 += 'echo "次: cron_cutover_status で合格のまま 1〜2 週間観察 → cutover_' + unit_id + '_2_cron_off.sh（root）"\n'
    s2 = HEADER.format(title=f"切替 2/2: {unit_id} の crontab の行をコメントアウト（root で実行。退避付き）", **meta)
    s2 += f'bash scripts/ops/cutover/host_cron_switch.sh disable --job-id {unit_id} {tag_args}\n'
    s2 += 'echo "戻すとき: cutover_' + unit_id + '_rollback.sh"\n'
    rb = HEADER.format(title=f"ロールバック: {unit_id} をホストの cron に戻す（行を戻す → 担当を外す の順）", **meta)
    rb += f'bash scripts/ops/cutover/host_cron_switch.sh enable --job-id {unit_id} {tag_args} || echo "（行は止めていなかった）"\n'
    rb += f'PYTHONPATH=. python3 -m src.ops.job_owner unset {unit_id}\n'
    if v_tasks:
        rb += 'cp config/fetch_owner.json "$BK/fetch_owner.before_rollback_' + unit_id + '_$TS.json"\n' + PY_DEL_AGENT.format(tasks=" ".join(v_tasks))
    rb += 'echo "戻しました。GCP / VPS 側のジョブは Scheduler / reserve-fetch-tasks を止めるか、並走のまま観察してください（二重実行になる間は parity で確認）"\n'
    return {f"cutover_{unit_id}_1_owner.sh": s1, f"cutover_{unit_id}_2_cron_off.sh": s2, f"cutover_{unit_id}_rollback.sh": rb}


def unit_status(status: dict[str, Any] | None, unit_id: str) -> tuple[bool | None, list[str]]:
    if not status:
        return None, []
    cu = (status.get("cut_units") or {}).get(unit_id)
    if cu is None:
        return False, [unit_id]
    return bool(cu["ready_to_stop"]), list(cu.get("waiting_on") or [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    m = sub.add_parser("make")
    m.add_argument("--unit", required=True)
    m.add_argument("--status-file", default=None)
    m.add_argument("--force", action="store_true")
    m.add_argument("--out-dir", default="/tmp")
    m.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    ns = ap.parse_args(argv)
    doc = job_map.load()
    if ns.cmd == "list":
        for cid, u in sorted(units(doc).items(), key=lambda kv: (kv[1]["wave"] == 0, kv[1]["wave"], kv[0])):
            own = owner_of(u)
            print(f"wave{u['wave']}\t{cid}\t{own or 'ホストに残す'}\t{','.join(sorted(u['jobs']))}")
        return 0
    status = json.loads(Path(ns.status_file).read_text(encoding="utf-8")) if ns.status_file else None
    ok, waiting = unit_status(status, ns.unit)
    if not ns.force and ok is not True:
        print(f"作りません: {ns.unit} は切替判定に合格していません（{'未評価（--status-file が必要）' if ok is None else '待ち: ' + ', '.join(waiting)}）。--force で強制できます", file=sys.stderr)
        return 3
    try:
        files = render(ns.unit, doc, project_dir=ns.project_dir)
    except ValueError as exc:
        print(f"作れません: {exc}", file=sys.stderr)
        return 2
    out = Path(ns.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        p = out / name
        p.write_text(text, encoding="utf-8")
        p.chmod(0o755)
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
