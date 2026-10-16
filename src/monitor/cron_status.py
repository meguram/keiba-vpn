"""cron ジョブの状態（待機中/実行中）・最終実行・結果・失敗の原因を集めて、監視ポータルに返す。

材料は 3 つ（上から優先）:
  1. cron_run.sh が残す実行記録  data/local/meta/cron_runs/<job_id>.{running,tsv,tail.txt,fail.txt}
  2. ログの境界行 『cron START/END task=… exit=N』（run_auto_scrape_logged.sh が書く）
  3. ログの更新時刻だけ（結果は『不明』）
ジョブの一覧は実際の crontab から作る（crontab -l。root の cron を見るには監視サーバも root で動かす）。
原因は cron_failure_patterns.classify で出力の末尾と終了コードから導く。

JSON の形は build_status() の docstring を参照。"""

from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

from src.monitor.cron_failure_patterns import classify
from src.scripts.docs import build_cron_schedule_html as CS

REPO_ROOT = Path(os.environ.get("KEIBA_REPO_ROOT") or Path(__file__).resolve().parents[2])
RUNS_DIR = Path(os.environ.get("KEIBA_CRON_RUNS_DIR") or REPO_ROOT / "data" / "local" / "meta" / "cron_runs")
LOG_DIR = REPO_ROOT / "logs"
JST = timezone(timedelta(hours=9))
TAIL_SHOWN = 40
LOG_READ_BYTES = 400_000

# ── job_id ───────────────────────────────────────────────────────────────────

_WRAPPERS = {"cron_run", "yield_to_rescrape", "run_auto_scrape_logged"}
_TASK = re.compile(r"run_auto_scrape_logged\.sh\s+\S+\s+(\S+)")
_EXPLICIT = re.compile(r"cron_run\.sh\s+(\S+)\s+--")


def derive_job_id(cmd: str) -> str:
    """crontab の 1 行（またはコマンド部分）から job_id を決める。cron_run.sh に渡す id と同じ規則:
    run_auto_scrape_logged.sh のタスクは『タスク名』、backfill は 'backfill-<年|all>-<phase>'、それ以外はスクリプト／モジュールの基底名。"""
    if "src.scraper.backfill" in cmd:
        y = re.search(r"--year (\d{4})", cmd)
        ph = re.search(r"--phase (\w+)", cmd)
        return f"backfill-{y.group(1) if y else 'all'}-{ph.group(1) if ph else 'all'}"
    m = _TASK.search(cmd)
    if m:
        return m.group(1)
    m = re.search(r"-m\s+src\.\w+(?:\.\w+)*\.(\w+)", cmd)
    if m:
        return m.group(1)
    for tok in re.findall(r"(?:\S*/)?([\w\-]+)\.(?:sh|py)\b", cmd):
        if tok not in _WRAPPERS:
            return tok
    return "unknown"


def job_id_of_line(cmd: str) -> str:
    """実際に記録される id（cron_run.sh に渡した id）。ラッパー無しの行は derive_job_id と同じ。"""
    m = _EXPLICIT.search(cmd)
    return m.group(1) if m else derive_job_id(cmd)


# ── cron 式 ──────────────────────────────────────────────────────────────────

def _field(spec: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
        else:
            a = int(part)
            b = hi if step > 1 else a
        out.update(range(a, b + 1, step))
    return out


def parse_expr(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int], bool, bool]:
    """'分 時 日 月 曜日' → (分, 時, 日, 月, 曜日(0=日), 日が*か, 曜日が*か)。曜日の 7 は日曜。"""
    mi, ho, dom, mon, dow = expr.split()[:5]
    dows = {d % 7 for d in _field(dow, 0, 7)}
    return _field(mi, 0, 59), _field(ho, 0, 23), _field(dom, 1, 31), _field(mon, 1, 12), dows, dom.startswith("*"), dow.startswith("*")


def next_fire(expr: str, after: datetime, horizon_days: int = 8) -> datetime | None:
    """after より後で最初に cron 式に合う時刻（after と同じタイムゾーン）。cron の仕様どおり、日と曜日が両方指定なら『どちらか』。"""
    mi, ho, dom, mon, dows, dom_star, dow_star = parse_expr(expr)
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    end = t + timedelta(days=horizon_days)
    while t < end:
        if t.month in mon and t.hour in ho and t.minute in mi:
            cdow = (t.weekday() + 1) % 7
            if dom_star and dow_star:
                day_ok = True
            elif dom_star:
                day_ok = cdow in dows
            elif dow_star:
                day_ok = t.day in dom
            else:
                day_ok = t.day in dom or cdow in dows
            if day_ok:
                return t
        t += timedelta(minutes=1)
    return None


# ── crontab の読み込み ───────────────────────────────────────────────────────

def load_crontab_text() -> tuple[str, str]:
    """(crontab のテキスト, 取得元)。KEIBA_CRONTAB_FILE → crontab -l → setup_all_cron.sh show（予定）の順。"""
    f = os.environ.get("KEIBA_CRONTAB_FILE")
    if f and Path(f).is_file():
        return Path(f).read_text(encoding="utf-8"), f
    try:
        out = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5).stdout
        if "cron_run.sh" in out or "run_auto_scrape_logged.sh" in out:
            return out, "crontab -l"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(["bash", str(REPO_ROOT / "scripts/cron/setup_all_cron.sh"), "show"], capture_output=True, text=True, timeout=20).stdout
        return out, "setup_all_cron.sh show（登録される予定の内容）"
    except (OSError, subprocess.SubprocessError):
        return "", "なし"


_CRON_FIELD = re.compile(r"^[\d*/,\-]+$")


def _lines(text: str) -> list[dict]:
    """crontab の各行 → {id, key, expr|None, cmd, when(JST の文章), line}。コメント・環境変数行は除く。"""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or re.match(r"^[A-Z_]+=", line):
            continue
        if line.startswith("@reboot"):
            cmd, expr, when = line[len("@reboot"):].strip(), None, "システム起動時"
        else:
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            mi, ho, dom, mon, dow, cmd = parts
            if not all(_CRON_FIELD.match(x) for x in (mi, ho, dom, mon, dow)):
                continue                       # show の説明文などは cron 行ではない
            expr = f"{mi} {ho} {dom} {mon} {dow}"
            when = CS.to_jst(mi, ho, dow)[0]
        cmd = re.sub(r"\s#\s*KEIBA\S*\s*$", "", cmd)
        key, extra = CS._job_key(cmd)
        out.append({"id": job_id_of_line(cmd), "key": key, "extra": extra, "expr": expr, "cmd": cmd, "when": when, "line": line})
    return out


def _merge_when(whens: list[str]) -> str:
    """['毎日 07:00', '毎日 17:00'] → '毎日 07:00・17:00 JST'。起動時のものは別に並べる。"""
    timed = [w for w in whens if w != "システム起動時"]
    days: dict[str, list[str]] = {}
    for w in timed:
        day, _, rest = w.partition(" ")
        days.setdefault(day, [])
        if rest not in days[day]:
            days[day].append(rest)
    parts = [f"{d} " + "・".join(r) + (" JST" if any(re.search(r"\d\d:\d\d", x) for x in r) else "") for d, r in days.items()]
    if "システム起動時" in whens:
        parts.append("システム起動時")
    return " ／ ".join(parts)


# ── 実行状態 ─────────────────────────────────────────────────────────────────

def _pid_alive(pid: int, job_id: str) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return True                        # /proc が無い環境では生きているとみなす
    return "cron_run" in cmdline or job_id in cmdline


def process_patterns(job_id: str) -> list[str]:
    """ラッパーの記録が無いジョブを ps で探すときの手がかり。"""
    m = re.match(r"backfill-(\w+)-(\w+)$", job_id)
    if m:
        y, ph = m.groups()
        return [rf"src\.scraper\.backfill .*--phase {ph}" + (rf".*--year {y}|--year {y} .*--phase {ph}" if y != "all" else "")]
    return [rf"auto_scrape --task {re.escape(job_id)}(\s|$)", rf"/{re.escape(job_id)}\.(?:sh|py)\b", rf"-m src\.\w+(?:\.\w+)*\.{re.escape(job_id)}\b"]


def _ps_lines() -> list[tuple[int, str]]:
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    res = []
    for ln in out.splitlines()[1:]:
        pid, _, args = ln.strip().partition(" ")
        if pid.isdigit():
            res.append((int(pid), args))
    return res


def _read_running(runs_dir: Path, job_id: str) -> tuple[int, float] | None:
    try:
        pid, start = (runs_dir / f"{job_id}.running").read_text().split()[:2]
        return int(pid), float(start)
    except (OSError, ValueError):
        return None


def _read_tsv_last(runs_dir: Path, job_id: str) -> tuple[float, float, int, int] | None:
    try:
        lines = (runs_dir / f"{job_id}.tsv").read_text().strip().splitlines()
        s, e, ec, pid = lines[-1].split("\t")
        return float(s), float(e), int(ec), int(pid)
    except (OSError, ValueError, IndexError):
        return None


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ── ログの境界行（run_auto_scrape_logged.sh） ───────────────────────────────

_BOUND = re.compile(r"^===== (\S+) cron (START|END)\s+task=(\S+)(?: pid=\d+)?(?:.*?exit=(\d+))?")


def _read_log_tail(path: Path) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - LOG_READ_BYTES))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _log_last_run(text: str, task: str) -> dict | None:
    """ログ末尾から、task の最後の完了した実行（START〜END）を返す。"""
    lines = text.splitlines()
    last_end = None
    for i in range(len(lines) - 1, -1, -1):
        m = _BOUND.match(lines[i])
        if m and m.group(3) == task and m.group(2) == "END":
            last_end = (i, m)
            break
    if last_end is None:
        return None
    i, m = last_end
    start_i = None
    for j in range(i - 1, -1, -1):
        mj = _BOUND.match(lines[j])
        if mj and mj.group(3) == task and mj.group(2) == "START":
            start_i = (j, mj)
            break
    body = lines[(start_i[0] + 1 if start_i else max(0, i - 80)):i]
    return {"ended_at": _iso(m.group(1)), "started_at": _iso(start_i[1].group(1)) if start_i else None,
            "exit_code": int(m.group(4)) if m.group(4) is not None else None, "tail": "\n".join(body[-80:])}


def _iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


# crontab にもラッパー引数にもログ名が出てこないジョブのログ（スクリプトが自分で書く）
LOG_HINTS = {"export_tables_daily": "export_tables_daily.log", "update_jockey_trainer_stats": "jockey_trainer_stats.log",
             "sync_pg_from_gcs": "sync_pg_from_gcs.log", "server_watchdog": "server_watchdog.log"}


def _log_path_of(cmd: str) -> str | None:
    m = re.search(r">>\s*(\S+\.log)", cmd) or re.search(r"run_auto_scrape_logged\.sh\s+\S+\s+\S+\s+(\S+\.log)", cmd)
    return m.group(1) if m else None


def _fmt(dt: datetime | None) -> str | None:
    return dt.astimezone(JST).isoformat(timespec="seconds") if dt else None


def _from_epoch(x: float) -> datetime:
    return datetime.fromtimestamp(x, tz=JST)


def _cause_dict(c: dict | None) -> dict | None:
    return None if c is None else {k: c[k] for k in ("pattern_id", "kind", "label", "hint", "evidence")}


def _result_of(exit_code: int | None, cause: dict | None) -> str:
    if exit_code not in (None, 0):
        return "failed"
    if cause is None:
        return "success" if exit_code == 0 else "unknown"
    return {"failed": "failed", "warning": "warning", "skipped": "skipped"}[cause["kind"]]


# ── 本体 ─────────────────────────────────────────────────────────────────────

def build_status(crontab_text: str | None = None, runs_dir: Path | None = None, log_dir: Path | None = None,
                 now: datetime | None = None, server_tz: tzinfo | None = None, ps_lines: list[tuple[int, str]] | None = None,
                 source: str = "") -> dict:
    """全ジョブの状態を返す。

    {"generated_at", "server_tz", "crontab_source", "summary": {running, waiting, success, failed, warning, skipped, unknown},
     "groups": [{"key", "name", "jobs": [{"id", "name", "desc", "dest", "schedule", "cron_exprs", "yields", "next_run",
        "state": "running"|"waiting", "running_since", "running_pid",
        "last_run": {"started_at", "ended_at", "duration_sec", "exit_code", "result", "source": "record"|"log"|"mtime"} | None,
        "cause": {...} | None, "last_failure": {"at", "cause"} | None, "output_tail": [...], "log_path": str | None}]}]}"""
    runs_dir = runs_dir or RUNS_DIR
    log_dir = log_dir or LOG_DIR
    if crontab_text is None:
        crontab_text, source = load_crontab_text()
    # setup_all_cron.sh show の内容は『システム TZ は UTC』前提で書かれているので、取得元がそれなら UTC で評価する
    tz = server_tz or (timezone.utc if source.startswith("setup_all_cron.sh show") else datetime.now().astimezone().tzinfo)
    now = (now or datetime.now(tz)).astimezone(tz)
    ps = ps_lines if ps_lines is not None else _ps_lines()

    by_id: dict[str, list[dict]] = {}
    for ln in _lines(crontab_text):
        by_id.setdefault(ln["id"], []).append(ln)

    jobs_by_group: dict[str, list[dict]] = {}
    for jid, lines in by_id.items():
        key = next((ln["key"] for ln in lines if ln["key"] in CS.JOBS), "")
        if key:
            g, name, desc, _proc, dest = CS.JOBS[key]
            if key == "backfill":
                name, desc = f"backfill: {lines[0]['extra']}", f"{lines[0]['extra']}を補填"
        else:
            g, name, desc, dest = "ops", jid, "", ""
        exprs = [ln["expr"] for ln in lines if ln["expr"]]
        nexts = [n for n in (next_fire(e, now) for e in exprs) if n]
        job: dict = {"id": jid, "name": name, "desc": desc, "dest": dest, "schedule": _merge_when([ln["when"] for ln in lines]),
                     "cron_exprs": exprs, "yields": key in CS.YIELDS or any("yield_to_rescrape.sh" in ln["cmd"] for ln in lines),
                     "next_run": _fmt(min(nexts)) if nexts else None}
        job.update(_job_runtime(jid, lines, runs_dir, log_dir, ps, now))
        jobs_by_group.setdefault(g, []).append(job)

    summary = {"running": 0, "waiting": 0, "success": 0, "failed": 0, "warning": 0, "skipped": 0, "unknown": 0}
    groups = []
    for gkey, gname, _gdesc in CS.GROUPS:
        jobs = sorted(jobs_by_group.pop(gkey, []), key=lambda j: j["name"])
        groups.append({"key": gkey, "name": gname, "jobs": jobs})
    for gkey, jobs in jobs_by_group.items():
        groups.append({"key": gkey, "name": gkey, "jobs": jobs})
    for grp in groups:
        for j in grp["jobs"]:
            summary[j["state"]] += 1
            summary[(j["last_run"] or {}).get("result", "unknown")] += 1
    return {"generated_at": _fmt(now), "server_tz": str(now.tzname()), "crontab_source": source, "summary": summary, "groups": groups}


def _job_runtime(jid: str, lines: list[dict], runs_dir: Path, log_dir: Path, ps: list[tuple[int, str]], now: datetime) -> dict:
    # 実行中か
    state, since, pid = "waiting", None, None
    rec = _read_running(runs_dir, jid)
    if rec and _pid_alive(rec[0], jid):
        state, since, pid = "running", _from_epoch(rec[1]), rec[0]
    elif not (runs_dir / f"{jid}.tsv").exists() and not rec:         # 記録の無いジョブは ps で探す
        pats = [re.compile(p) for p in process_patterns(jid)]
        for p, args in ps:
            if p != os.getpid() and "ps -eo" not in args and any(rx.search(args) for rx in pats) and "cron_status" not in args:
                state, pid = "running", p
                break

    # 最後の実行
    last_run, cause, tail_text, log_rel, last_fail = None, None, "", None, None
    tsv = _read_tsv_last(runs_dir, jid)
    if tsv:
        s, e, ec, _ = tsv
        tail_text = _read_text(runs_dir / f"{jid}.tail.txt")
        cause = classify(ec, tail_text)
        last_run = {"started_at": _fmt(_from_epoch(s)), "ended_at": _fmt(_from_epoch(e)), "duration_sec": int(e - s), "exit_code": ec,
                    "result": _result_of(ec, cause), "source": "record"}
        ftxt = _read_text(runs_dir / f"{jid}.fail.txt")
        if ftxt:
            hm = re.match(r"# (\S+) exit=(\d+)", ftxt)
            fc = classify(int(hm.group(2)) if hm else 1, ftxt)
            last_fail = {"at": _fmt(_iso(hm.group(1))) if hm else None, "cause": _cause_dict(fc)}
    else:
        names = [p for p in (_log_path_of(ln["cmd"]) for ln in lines) if p] + ([LOG_HINTS[jid]] if jid in LOG_HINTS else [])
        logs = [log_dir / Path(p).name for p in names]
        logs = sorted({l for l in logs if l.exists()}, key=lambda l: l.stat().st_mtime, reverse=True)
        if logs:
            log_rel = f"logs/{logs[0].name}"
            text = _read_log_tail(logs[0])
            lr = _log_last_run(text, jid)
            if lr:
                ec = lr["exit_code"]
                tail_text = lr["tail"]
                cause = classify(ec, tail_text)
                dur = int((lr["ended_at"] - lr["started_at"]).total_seconds()) if lr["started_at"] and lr["ended_at"] else None
                last_run = {"started_at": _fmt(lr["started_at"]), "ended_at": _fmt(lr["ended_at"]), "duration_sec": dur, "exit_code": ec,
                            "result": _result_of(ec, cause), "source": "log"}
            else:
                mt = _from_epoch(logs[0].stat().st_mtime)
                tail_text = "\n".join(text.splitlines()[-80:])
                last_run = {"started_at": _fmt(mt), "ended_at": _fmt(mt), "duration_sec": None, "exit_code": None, "result": "unknown", "source": "mtime"}
        if last_run and last_run["result"] == "failed" and cause:
            last_fail = {"at": last_run["ended_at"], "cause": _cause_dict(cause)}
    if log_rel is None:
        for ln in lines:
            p = _log_path_of(ln["cmd"]) or LOG_HINTS.get(jid)
            if p:
                log_rel = f"logs/{Path(p).name}"
                break
    if last_fail is None and last_run and last_run["result"] == "failed" and cause:
        last_fail = {"at": last_run["ended_at"], "cause": _cause_dict(cause)}
    return {"state": state, "running_since": _fmt(since), "running_pid": pid, "last_run": last_run, "cause": _cause_dict(cause),
            "last_failure": last_fail, "output_tail": [l for l in tail_text.splitlines() if l.strip()][-TAIL_SHOWN:], "log_path": log_rel}
