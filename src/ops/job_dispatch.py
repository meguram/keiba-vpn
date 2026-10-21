"""Cloud Run Jobs の唯一のエントリ（ディスパッチャ）。``config/gcp_job_map.json`` に載った GCP 向けジョブだけを実行する。

    python -m src.ops.job_dispatch group <group_id>      # Scheduler から（グループ内を直列に実行）
    python -m src.ops.job_dispatch job <job_id> [--arg race_id=… --arg date=…]   # Cloud Tasks（predict-race）や手動
    python -m src.ops.job_dispatch list

守ること:
- **許可リスト**: job_map にある ``destination=gcp`` かつ ``enabled`` のジョブだけ。引数は job_map の steps に書かれた ``['-m', module, …]`` のみ（外から任意のコマンドは渡せない）。
- **メモリ予算**: ジョブの ``memory_mib`` がコンテナの上限を超えるなら実行しない。ジョブは直列。state の合計サイズにも上限（state_sync）。
- **準備未完了**（``ready=false``）のジョブは ``--allow-unready`` が無い限り動かさない。
- 実行ごとに ``ops/job_runs/<job_id>/<時刻>.json`` に記録（開始・終了・遅れ・結果・ピークメモリ）。切替判定（cron_cutover_status）がこれを読む。
- 失敗は Slack（``src.utils.slack_notify.send_payload``）で通知し、終了コード 1 を返す（Cloud Run Jobs が失敗として扱う）。
- 並走期間（``KEIBA_PARALLEL_RUN=1``）は、job_map の ``parallel_env``（例: 予測を影モードにして本番の予測を上書きしない）を適用する。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.ops import job_map
from src.ops.gcs_io import BlobIO, read_json, write_json

JST = ZoneInfo("Asia/Tokyo")
RUN_PREFIX = "ops/job_runs"
FETCH_DONE_PREFIX = "ops/fetch_tasks/done"
_PLACEHOLDER = re.compile(r"\{([a-z_0-9]+(?::[a-z_0-9]+)?)\}")
TAIL_LINES = 60

OK, PARTIAL, FAILED, SKIPPED, WAITING, REFUSED, DEP_FAILED = "ok", "partial", "failed", "skipped", "waiting", "refused", "dep_failed"
BAD = {FAILED, REFUSED, DEP_FAILED}


# ───────────────────────── テンプレート ─────────────────────────

def build_params(now: datetime, extra: dict[str, str] | None = None) -> dict[str, str]:
    d = now.astimezone(JST).date()

    def f(x: Any) -> str:
        return x.strftime("%Y%m%d")

    def iso(x: Any) -> str:
        return x.strftime("%Y-%m-%d")

    weekend = [d - timedelta(days=b) for b in range(1, 9) if (d - timedelta(days=b)).isoweekday() >= 6]
    p = {
        "date": f(d), "today": f(d), "today_iso": iso(d), "yesterday": f(d - timedelta(days=1)), "tomorrow": f(d + timedelta(days=1)),
        "tomorrow_iso": iso(d + timedelta(days=1)), "year": str(d.year), "prev_year": str(d.year - 1),
        "years": f"{d.year - 1},{d.year}" if d.month == 1 else str(d.year),
        "days_ago_10": f(d - timedelta(days=10)), "days_ago_28": f(d - timedelta(days=28)),
        "last_weekend_days": ",".join(f(x) for x in sorted(weekend)),
    }
    p.update(extra or {})
    return p


def render(text: str, params: dict[str, str], models: dict[str, str] | None = None) -> str:
    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key.startswith("model:"):
            name = key.split(":", 1)[1]
            if not models or name not in models:
                raise KeyError(f"モデル {name} が未取得")
            return models[name]
        if key not in params:
            raise KeyError(f"未定義のプレースホルダ {{{key}}}")
        return params[key]

    return _PLACEHOLDER.sub(sub, text)


# ───────────────────────── 環境 ─────────────────────────

def container_memory_limit_mib() -> int | None:
    v = os.environ.get("KEIBA_DISPATCH_MEMORY_MIB")
    if v:
        return int(v)
    for p, parse in (("/sys/fs/cgroup/memory.max", None), ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None)):
        try:
            raw = Path(p).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        n = int(raw)
        return None if n >= 1 << 60 else n // (1024 * 1024)
    return None


def _children_peak_mib() -> int:
    return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024)     # Linux は KB


def default_runner(argv: list[str], env: dict[str, str], timeout: float, cwd: str) -> tuple[int, list[str]]:
    """子プロセスを実行して標準出力へ流しつつ、末尾 TAIL_LINES 行を返す。時間切れは kill して rc=124。"""
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen([sys.executable, *argv], env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(max(timeout, 1.0), kill)
    timer.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip("\n"))
        rc = proc.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        tail.append(f"[dispatch] timeout {timeout:.0f}s で停止")
        return 124, list(tail)
    return rc, list(tail)


@dataclass
class Ctx:
    io: BlobIO
    root: Path = field(default_factory=lambda: Path(".").resolve())
    now_fn: Callable[[], datetime] = lambda: datetime.now(JST)
    runner: Callable[..., tuple[int, list[str]]] = default_runner
    base_env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    memory_limit_mib: int | None = None
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("KEIBA_MODELS_DIR") or tempfile.gettempdir()) / "keiba_models")
    allow_unready: bool = False
    dry_run: bool = False
    alert: Callable[..., Any] | None = None
    pull: Callable[..., Any] | None = None
    push: Callable[..., Any] | None = None
    fetch_model: Callable[..., Path] | None = None
    peak_fn: Callable[[], int] = _children_peak_mib

    def __post_init__(self) -> None:
        if self.alert is None:
            from src.ops import alerting

            self.alert = alerting.notify
        if self.pull is None or self.push is None:
            from src.ops import state_sync

            self.pull = self.pull or state_sync.pull
            self.push = self.push or state_sync.push
        if self.fetch_model is None:
            from src.ops import model_publish

            self.fetch_model = model_publish.fetch


# ───────────────────────── ガード ─────────────────────────

def _race_day_exists(io: BlobIO, date: str) -> bool:
    doc = read_json(io, f"state/data/page_reference/race_lists/{date}.json")
    return bool(doc and (doc.get("races") if isinstance(doc, dict) else doc))


def check_guards(job: dict[str, Any], params: dict[str, str], ctx: Ctx, now: datetime) -> tuple[str, str] | None:
    """実行しない理由 (status, reason)。実行してよければ None。"""
    rd = job.get("requires_race_day")
    if rd and not _race_day_exists(ctx.io, params[rd]):
        return SKIPPED, f"{rd}（{params[rd]}）は開催日ではない（race_lists が無い／空）"
    fd = job.get("requires_fetch_done")
    if fd:
        done = ctx.io.list(f"{FETCH_DONE_PREFIX}/{fd}-{params['today']}")
        if not done:
            dl = job.get("fetch_deadline_jst")
            if dl and now.strftime("%H:%M") >= dl:
                return FAILED, f"取得タスク {fd} が {dl} になっても完了していない"
            return WAITING, f"取得タスク {fd} が未完了（次の tick で再試行）"
    return None


def when_matches(job: dict[str, Any], tick: datetime) -> bool:
    w = job.get("when_jst")
    return True if not w else job_map.cron_matches(w, tick)


# ───────────────────────── 実行 ─────────────────────────

def run_job(job: dict[str, Any], ctx: Ctx, params: dict[str, str], *, tick: datetime | None, results: dict[str, str]) -> dict[str, Any]:
    now = ctx.now_fn()
    rec: dict[str, Any] = {
        "job": job["id"], "group": job.get("group"), "scheduled_at": tick.isoformat(timespec="seconds") if tick else None,
        "started_at": now.isoformat(timespec="seconds"), "status": OK, "reason": None, "steps": [], "state": {},
        "env": {k: ctx.base_env.get(k) for k in ("KEIBA_ENV", "KEIBA_RUN_LOCATION")}, "parallel": ctx.base_env.get("KEIBA_PARALLEL_RUN") == "1",
        "memory_mib_declared": job.get("memory_mib"), "memory_limit_mib": ctx.memory_limit_mib,
    }
    t0 = time.monotonic()

    def finish(status: str, reason: str | None = None) -> dict[str, Any]:
        rec["status"], rec["reason"] = status, reason
        rec["finished_at"] = ctx.now_fn().isoformat(timespec="seconds")
        rec["duration_sec"] = round(time.monotonic() - t0, 1)
        rec["memory_peak_mib"] = ctx.peak_fn()
        if not ctx.dry_run:
            stamp = datetime.fromisoformat(rec["started_at"]).strftime("%Y%m%dT%H%M%S")
            write_json(ctx.io, f"{RUN_PREFIX}/{job['id']}/{stamp}.json", rec)
            write_json(ctx.io, f"{RUN_PREFIX}/{job['id']}/latest.json", rec)
        return rec

    for dep in job.get("depends") or []:
        if results.get(dep) in BAD:
            return finish(DEP_FAILED, f"依存ジョブ {dep} が {results.get(dep)}")
        if results.get(dep) not in (OK, PARTIAL):
            return finish(SKIPPED, f"依存ジョブ {dep} が実行されていない／成功していない（{results.get(dep) or '準備未完了など'}）")
    guard = None if ctx.dry_run else check_guards(job, params, ctx, now)
    if guard:
        status, reason = guard
        out = finish(status, reason)
        if status == FAILED:
            ctx.alert("ジョブの前提を満たさない: " + job["id"], [reason], severity="critical", job=job["id"], io=ctx.io, now=now)
        return out
    lim = ctx.memory_limit_mib
    if lim and job.get("memory_mib", 0) > lim * 1.05:
        reason = f"宣言メモリ {job['memory_mib']}MiB がコンテナの上限 {lim}MiB を超える（大きいサイズの Run Job で実行する）"
        out = finish(REFUSED, reason)
        ctx.alert("ジョブを実行できない: " + job["id"], [reason], severity="critical", job=job["id"], io=ctx.io, now=now)
        return out
    if ctx.dry_run:
        rec["steps"] = [{"argv": [render(a, params, {m: f"<{m}>" for m in job.get("models") or []}) for a in s["argv"]]} for s in job["steps"]]
        return finish(OK, "dry-run")

    deadline = time.monotonic() + float(job.get("timeout_sec", 1800))
    env = dict(ctx.base_env)
    env.update({"KEIBA_ENV": "prod", "KEIBA_RUN_LOCATION": "gcp", "PYTHONUNBUFFERED": "1"})
    if env.get("KEIBA_PARALLEL_RUN") == "1":
        env.update(job.get("parallel_env") or {})
    models: dict[str, str] = {}
    try:
        spec_models = job_map.load()["models"]
        from src.ops.model_publish import env_for, install

        for name in job.get("models") or []:
            path = ctx.fetch_model(ctx.io, name, ctx.models_dir)
            k, v = env_for(spec_models, name, path)
            if k:
                env[k] = v
            models[name] = v
            install(spec_models, name, path, ctx.root)
        rec["state"]["pull"] = ctx.pull(ctx.io, job.get("state_in") or [], ctx.root, params=params)
    except Exception as exc:  # noqa: BLE001
        reason = f"準備に失敗: {type(exc).__name__}: {exc}"
        out = finish(FAILED, reason)
        ctx.alert("ジョブの準備に失敗: " + job["id"], [reason], severity="critical", job=job["id"], io=ctx.io, now=now)
        return out

    started_epoch = time.time() - 1
    status = OK
    for i, st in enumerate(job["steps"]):
        try:
            argv = [render(a, params, models) for a in st["argv"]]
        except KeyError as exc:
            return finish(FAILED, f"step {i}: {exc}")
        senv = {**env, **{k: render(v, params, models) for k, v in (st.get("env") or {}).items()}}
        left = deadline - time.monotonic()
        if left <= 0:
            rec["steps"].append({"argv": argv, "rc": 124, "sec": 0, "tail": ["[dispatch] 全体のタイムアウト"]})
            status = FAILED
            break
        s0 = time.monotonic()
        rc, tail = ctx.runner(argv, senv, left, str(ctx.root))
        rec["steps"].append({"argv": argv, "rc": rc, "sec": round(time.monotonic() - s0, 1), "tail": tail[-20:]})
        if rc not in (st.get("ok_codes") or [0]):
            if st.get("allow_fail"):
                status = PARTIAL if status == OK else status
                continue
            status = FAILED
            break
    if status in (OK, PARTIAL):
        try:
            rec["state"]["push"] = ctx.push(ctx.io, job.get("state_out") or [], ctx.root, since=started_epoch, params=params)
        except Exception as exc:  # noqa: BLE001
            rec["state"]["push_error"] = str(exc)
            status = FAILED
    reason = None
    if status == FAILED:
        bad = next((s for s in rec["steps"] if s["rc"] not in (0,)), None)
        reason = (rec["state"].get("push_error") or (f"step rc={bad['rc']}: " + " ".join(bad["argv"][:3]) if bad else "失敗"))
    out = finish(status, reason)
    if status == FAILED:
        lines = [reason or "失敗"] + ([s for s in (rec["steps"][-1]["tail"][-5:])] if rec["steps"] else [])
        ctx.alert("ジョブが失敗: " + job["id"], lines, severity="critical", job=job["id"], io=ctx.io, now=now)
    elif status == PARTIAL:
        ctx.alert("ジョブが一部失敗: " + job["id"], ["allow_fail の step が失敗（本体は成功）"], severity="warning", job=job["id"], io=ctx.io, now=now)
    return out


def select(doc: dict[str, Any], job_ids: list[str], ctx: Ctx, tick: datetime | None) -> list[dict[str, Any]]:
    out = []
    for jid in job_ids:
        j = job_map.job_by_id(jid, doc)
        if j is None or j["destination"] != "gcp":
            raise SystemExit(f"許可リスト外のジョブ: {jid}")
        if not j.get("enabled", True):
            print(f"[dispatch] {jid}: enabled=false のためスキップ")
            continue
        if not j.get("ready") and not ctx.allow_unready:
            print(f"[dispatch] {jid}: ready=false（{'; '.join(j.get('blockers') or [])}）のためスキップ")
            continue
        if tick is not None and not when_matches(j, tick):
            continue
        out.append(j)
    return out


def run_group(group_id: str, ctx: Ctx, *, extra: dict[str, str] | None = None) -> int:
    doc = job_map.load()
    g = job_map.group_of(group_id, doc)
    if g is None:
        print(f"未定義のグループ: {group_id}", file=sys.stderr)
        return 2
    now = ctx.now_fn()
    tick = job_map.prev_occurrence(g["cron_jst"], now) or now
    jobs = select(doc, g["jobs"], ctx, tick)
    params = build_params(now, extra)
    results: dict[str, str] = {}
    for j in jobs:
        rec = run_job(j, ctx, params, tick=tick, results=results)
        results[j["id"]] = rec["status"]
        print(f"[dispatch] {j['id']}: {rec['status']}" + (f"（{rec['reason']}）" if rec.get("reason") else ""), flush=True)
    return 1 if any(s in BAD for s in results.values()) else 0


def run_single(job_id: str, ctx: Ctx, *, extra: dict[str, str] | None = None) -> int:
    doc = job_map.load()
    jobs = select(doc, [job_id], ctx, None)
    if not jobs:
        return 2
    now = ctx.now_fn()
    params = build_params(now, extra)
    rec = run_job(jobs[0], ctx, params, tick=None, results={})
    print(f"[dispatch] {job_id}: {rec['status']}" + (f"（{rec['reason']}）" if rec.get("reason") else ""), flush=True)
    return 1 if rec["status"] in BAD else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("group")
    g.add_argument("group_id")
    j = sub.add_parser("job")
    j.add_argument("job_id")
    j.add_argument("--arg", action="append", default=[], help="KEY=VALUE（race_id, date など）")
    sub.add_parser("list")
    for p in (g, j):
        p.add_argument("--allow-unready", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--now", default=None, help="ISO 8601（テスト・再現用）")
    ns = ap.parse_args(argv)
    doc = job_map.load()
    if ns.cmd == "list":
        for gid, grp in doc["scheduler_groups"].items():
            print(f"{gid}\t{grp['cron_jst']}\t{grp['size']}\t{','.join(grp['jobs'])}")
        return 0
    errs = job_map.validate(doc)
    if errs:
        print("job_map の検証に失敗:\n" + "\n".join(errs), file=sys.stderr)
        return 3
    from src.ops.gcs_io import MemoryIO, default_io

    io: BlobIO = MemoryIO() if ns.dry_run else default_io()
    now_fn = (lambda: datetime.fromisoformat(ns.now).astimezone(JST)) if ns.now else (lambda: datetime.now(JST))
    ctx = Ctx(io=io, now_fn=now_fn, memory_limit_mib=container_memory_limit_mib(), allow_unready=ns.allow_unready, dry_run=ns.dry_run)
    if ns.cmd == "group":
        return run_group(ns.group_id, ctx)
    extra = dict(a.split("=", 1) for a in ns.arg)
    return run_single(ns.job_id, ctx, extra=extra)


if __name__ == "__main__":
    sys.exit(main())
