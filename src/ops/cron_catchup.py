"""ホストの毎日の定時ジョブを、PC（cron）が止まっていて時刻を逃した日に、起動後に 1 回だけ追いつき実行する（anacron 的）。

    python -m src.ops.cron_catchup run        # 逃した分を実行する（cron から 15 分おきに呼ぶ）
    python -m src.ops.cron_catchup plan       # 実行せずに、何をするかだけ出す

追いつきの対象は ``config/cron_catchup.json`` の許可リストのジョブだけ（毎日 1 回の、時刻に縛られないバッチ）。
次をすべて満たすときだけ、crontab のその行のコマンド（cron_run.sh 経由。実行記録も通常と同じく残る）を 1 回実行する。

1. 直前の予定時刻 pf（crontab の式。cron の時刻＝ホストは UTC で評価）から ``grace_min`` 分以上たち、``max_late_hours`` 以内で、
   いまと pf が JST で同じ日付（日をまたいだら翌日の予定に任せる。日付を見て動くジョブが翌日の分を二重に処理しないように）。
2. pf 以降に、そのジョブが始まった記録（cron_status.RUNS_DIR の <job>.tsv）が無く、いま動いてもいない
   （.running の印は、PC の停止より前のものなら残骸とみなす。ps で cron_run.sh <job> の起動中も確かめる）。
3. pf の時刻に cron が動いていなかった。判定は次のどちらか。
   - cron のデーモンの起動が pf より後（PC・コンテナの再起動）。起動から ``settle_min`` 分は待つ。
   - 15 分おきの通知の記録の閉じた空白（scrape_status_notify.host_down_windows。スリープなど）。
     空白が閉じる＝通知が再開した後なので、復帰時に cron 自身が飛んだ予定を実行する分と重ならない。
4. この pf の分をまだ追いついていない（MARK_DIR の <job>.json の印）。

各ジョブの実行の直前に 2〜4 を見直す（前のジョブを実行している間に本来の cron や手動で始まった分は走らせない）。
印は実行の前に書くので、失敗しても同じ pf の分を二度は走らせない。追いつきが入るのは逃した予定の代わりだけなので、
1 日に走る回数は予定どおり（1 日 1 回）を超えない。読むのはローカルのファイルだけ（GCS・API は使わない）。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "cron_catchup.json"
from src.monitor.cron_status import RUNS_DIR as _RUNS_DIR  # noqa: E402  ホストのローカルの記録の置き場は cron_status の定義に合わせる

META_DIR = _RUNS_DIR.parent
MARK_DIR = Path(os.environ.get("KEIBA_CRON_CATCHUP_DIR") or META_DIR / "cron_catchup")
JST = timezone(timedelta(hours=9))
_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
# cron がジョブに渡す既定の環境（crontab の環境変数の行で上書きされる）
CRON_BASE_ENV = {"SHELL": "/bin/sh", "PATH": "/usr/bin:/bin"}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"jobs": list(d.get("jobs") or []), "max_late_hours": float(d.get("max_late_hours", 18)),
            "grace_min": float(d.get("grace_min", 20)), "settle_min": float(d.get("settle_min", 5))}


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def crontab_entries(text: str) -> list[dict[str, Any]]:
    """crontab の各ジョブ行 → {id, expr, cmd, env}。env はその行より前の環境変数の行（cron と同じく、後の行は効かない）。"""
    from src.monitor.cron_status import _lines

    env: dict[str, str] = {}
    env_at: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if m and not re.match(r"^[\d*/,\-@]", line):
            env[m.group(1)] = _unquote(m.group(2))
        else:
            env_at.setdefault(line, dict(env))
    return [{"id": ln["id"], "expr": ln["expr"], "cmd": ln["cmd"], "env": env_at.get(ln["line"], {})}
            for ln in _lines(text) if ln.get("expr")]


def _boot_time() -> float | None:
    try:
        return next(float(ln.split()[1]) for ln in Path("/proc/stat").read_text().splitlines() if ln.startswith("btime "))
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def cron_daemon_started_at() -> float | None:
    """cron のデーモンの起動時刻（epoch）。見つからなければ None。"""
    btime = _boot_time()
    if btime is None:
        return None
    hz = os.sysconf("SC_CLK_TCK")
    best = None
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / "comm").read_text().strip() not in ("cron", "crond"):
                continue
            ticks = int((p / "stat").read_text().rsplit(")", 1)[1].split()[19])
        except (OSError, ValueError, IndexError):
            continue
        t = btime + ticks / hz
        best = t if best is None else min(best, t)
    return best


def down_windows(now: datetime) -> list[tuple[float, float]]:
    """ホストの停止期間（scrape_status_notify と同じ判定。保存された分も合わせる）。閉じた空白だけ（include_tail は使わない:
    通知のジョブ自身として動いているときだけ使う決まり。通知が固まっただけの空白を停止と誤認しないため）。失敗したら空。"""
    try:
        from src.monitor.cron_status import RUNS_DIR
        from src.scripts.data import scrape_status_notify as ssn

        found = ssn.host_down_windows(RUNS_DIR, now)
        state_p = META_DIR / "slack_sent" / "scrape_status_state.json"
        try:
            saved = (json.loads(state_p.read_text(encoding="utf-8")).get("_meta") or {}).get("down") or []
        except (OSError, ValueError, AttributeError):
            saved = []
        return ssn.merge_down_windows(saved, found, now)
    except Exception:  # noqa: BLE001
        return []


def last_start(job_id: str, runs_dir: Path) -> float | None:
    from src.scripts.data.scrape_status_notify import read_runs

    runs = read_runs(job_id, runs_dir, limit=50)
    return max((s for s, _e, _c in runs), default=None)


def _ps_running(job_id: str) -> bool:
    """cron_run.sh <job> の起動中（.running を書く前の約 1 秒も含めて拾う）。"""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    pat = re.compile(rf"cron_run\.sh\s+{re.escape(job_id)}\s+--")
    return any(pat.search(ln) for ln in out.splitlines() if "cron_catchup" not in ln)


def is_running(job_id: str, runs_dir: Path, *, boot: float | None = None, ps: Callable[[str], bool] = _ps_running) -> bool:
    """.running の印が生きているか、ps で起動中か。印の開始が PC の起動より前なら残骸（PID の再利用で誤認しない）。"""
    try:
        pid_s, start_s = (runs_dir / f"{job_id}.running").read_text().split()[:2]
        pid, start = int(pid_s), float(start_s)
        boot = _boot_time() if boot is None else boot
        if Path(f"/proc/{pid}").exists() and (boot is None or start >= boot):
            return True
    except (OSError, ValueError):
        pass
    return ps(job_id)


def read_mark(job_id: str, mark_dir: Path = MARK_DIR) -> dict:
    try:
        d = json.loads((mark_dir / f"{job_id}.json").read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def write_mark(job_id: str, data: dict, mark_dir: Path = MARK_DIR) -> None:
    mark_dir.mkdir(parents=True, exist_ok=True)
    tmp = mark_dir / f".{job_id}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, mark_dir / f"{job_id}.json")


def _still_missed(job: str, pf_ts: float, pf_iso: str, *, runs_dir: Path, mark_dir: Path, boot: float | None,
                  ps: Callable[[str], bool]) -> bool:
    ls = last_start(job, runs_dir)
    if ls is not None and ls >= pf_ts - 60:          # 予定どおり（または後で）始まっている
        return False
    if read_mark(job, mark_dir).get("pf") == pf_iso:
        return False                                  # この予定の分は追いつき済み（失敗していても二度は走らせない）
    return not is_running(job, runs_dir, boot=boot, ps=ps)


def plan(now: datetime, entries: list[dict], cfg: dict, *, runs_dir: Path, mark_dir: Path = MARK_DIR,
         cron_started: float | None, down: list[tuple[float, float]], boot: float | None = None,
         ps: Callable[[str], bool] = _ps_running) -> list[dict]:
    """追いつき実行するジョブの一覧（予定時刻の古い順）。各要素: {id, pf, pf_ts, cmd, env, why}。"""
    from src.monitor.cron_status import cron_timezone
    from src.scripts.data.scrape_status_notify import prev_fire

    cnow = now.astimezone(cron_timezone())
    t_now = now.timestamp()
    out = []
    for job in cfg["jobs"]:
        fires = [(f, e) for e in entries if e["id"] == job for f in [prev_fire(e["expr"], cnow)] if f]
        if not fires:
            continue
        pf, ent = max(fires, key=lambda x: x[0])     # 1 日に複数の予定がある行は、直前の予定とその行のコマンド・環境
        pf_ts = pf.timestamp()
        late_min = (t_now - pf_ts) / 60
        if late_min < cfg["grace_min"] or late_min > cfg["max_late_hours"] * 60:
            continue
        if pf.astimezone(JST).date() != now.astimezone(JST).date():
            continue                                  # JST の日付が変わったら、翌日の予定に任せる
        if cron_started is not None and cron_started > pf_ts:
            if t_now - cron_started < cfg.get("settle_min", 5) * 60:
                continue                              # 起動直後は待つ（cron 自身・他の起動処理と重ねない）
            why = "cron の起動が予定時刻より後（PC・コンテナの停止）"
        elif any(a <= pf_ts <= b for a, b in down):
            why = "予定時刻にホストが止まっていた（通知の記録の空白）"
        else:
            continue                                  # cron は動いていたのに始まっていない: 追いつきではなく、通知（未開始）で人が見る
        if not _still_missed(job, pf_ts, pf.isoformat(), runs_dir=runs_dir, mark_dir=mark_dir, boot=boot, ps=ps):
            continue
        out.append({"id": job, "pf": pf.isoformat(), "pf_ts": pf_ts, "cmd": ent["cmd"], "env": ent["env"], "why": why})
    return sorted(out, key=lambda x: x["pf_ts"])


def job_env(item: dict) -> dict[str, str]:
    """cron がそのジョブに渡す環境に近いもの: 既定（SHELL・PATH）＋HOME・LOGNAME＋crontab のその行より前の環境変数の行。
    追いつきの行の環境（PYTHONPATH など）は混ぜない。"""
    env = dict(CRON_BASE_ENV)
    for k in ("HOME", "LOGNAME", "USER"):
        if k in os.environ:
            env[k] = os.environ[k]
    env.update(item.get("env") or {})
    return env


def _run_cmd(item: dict) -> int:
    env = job_env(item)
    return subprocess.run([env.get("SHELL") or "/bin/sh", "-c", item["cmd"]], env=env, cwd=env.get("HOME") or "/").returncode


def _jst(dt: datetime | None = None) -> str:
    return (dt or datetime.now(timezone.utc)).astimezone(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def run(now: datetime | None = None, *, dry: bool = False, runner: Callable[[dict], int] = _run_cmd,
        crontab_text: str | None = None, cfg: dict | None = None, runs_dir: Path | None = None, mark_dir: Path = MARK_DIR,
        cron_started: float | None | bool = False, down: list | None = None, boot: float | None | bool = False,
        ps: Callable[[str], bool] = _ps_running, log=print) -> list[dict]:
    from src.monitor.cron_status import RUNS_DIR, load_crontab_text

    now = now or datetime.now(timezone.utc)
    cfg = cfg or load_config()
    text = crontab_text if crontab_text is not None else load_crontab_text()[0]
    runs_dir = runs_dir or RUNS_DIR
    boot_t = _boot_time() if boot is False else boot
    items = plan(now, crontab_entries(text), cfg, runs_dir=runs_dir, mark_dir=mark_dir,
                 cron_started=cron_daemon_started_at() if cron_started is False else cron_started,
                 down=down_windows(now) if down is None else down, boot=boot_t, ps=ps)
    done = []
    for it in items:
        if dry:
            log(f"[{_jst(now)}] 追いつき（予定のみ）: {it['id']}（予定 {_jst(datetime.fromisoformat(it['pf']))}・{it['why']}）")
            done.append(it)
            continue
        # 実行の直前に見直す（前のジョブを実行している間に、本来の cron・手動で始まっていたら走らせない）
        if not _still_missed(it["id"], it["pf_ts"], it["pf"], runs_dir=runs_dir, mark_dir=mark_dir, boot=boot_t, ps=ps):
            log(f"[{_jst()}] 追いつきを取りやめ: {it['id']}（その間に始まった・済んだ）")
            continue
        try:
            write_mark(it["id"], {"pf": it["pf"], "started_at": _jst(), "why": it["why"]}, mark_dir)
        except OSError as e:
            log(f"[{_jst()}] 印を書けないので追いつかない: {it['id']}（{e}）")   # 印が無いまま走らせると二重になりうる
            continue
        log(f"[{_jst()}] 追いつき開始: {it['id']}（予定 {_jst(datetime.fromisoformat(it['pf']))}・{it['why']}）")
        try:
            rc = runner(it)
        except Exception as e:  # noqa: BLE001 — 1 件の失敗で残りを止めない
            log(f"[{_jst()}] 追いつきの起動に失敗: {it['id']}（{e}）")
            rc = -1
        it["rc"] = rc
        try:
            mark = read_mark(it["id"], mark_dir)
            mark.update({"rc": rc, "ended_at": _jst()})
            write_mark(it["id"], mark, mark_dir)
        except OSError:
            pass
        log(f"[{_jst()}] 追いつき終了: {it['id']} rc={rc}")
        done.append(it)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ホストの毎日の定時ジョブの追いつき実行（anacron 的）")
    ap.add_argument("cmd", choices=("run", "plan"))
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        if not run(dry=True):
            print("追いつき実行の対象なし")
        return 0
    try:
        MARK_DIR.mkdir(parents=True, exist_ok=True)
        lk = open(MARK_DIR / ".lock", "a")            # 作業ツリーの中（/tmp の他のユーザーのファイルに阻まれない）
    except OSError as e:
        print(f"ロックを作れない（{e}）。今回は何もしない")
        return 1
    with lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("前の追いつき実行がまだ動いている。今回は何もしない")
            return 0
        items = run()
    return 1 if any(it.get("rc") for it in items) else 0


if __name__ == "__main__":
    sys.exit(main())
