"""VPS の取得の薄い実行役。GCS のタスク表（fetch_task）を 60 秒おきに pull し、期限が来たものを 1 本ずつ実行する。

- 有効化: ``KEIBA_FETCH_AGENT_ENABLED=1``（既定オフ。オフなら何もせず終了コード 0）。
- 担当: ``config/fetch_owner.json`` の ``agent_tasks``（``KEIBA_FETCH_AGENT_TASKS=a,b`` で上書き）。担当外は pending のまま残す。
- 実行: 子プロセス（``python -m src.ops.fetch_child``）。systemd の ``MemoryMax=250M`` に収まる前提で、親は軽く保つ（pandas 等を import しない）。
- 失敗: ``max_attempts`` まで再試行。期限（deadline）超過は実行せず failed。アクセス制限（403/429 の疑い）は再試行せず、一定時間 claim を止める。
- 秘密: 子プロセスの環境（VPS の .env）だけが持つ。タスクの引数・結果・ログの末尾に出さない（既知の秘密値は伏せ字）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from src.ops import access_hold
from src.ops import fetch_task as ft
from src.ops.fetch_child import RESULT_PREFIX
from src.ops.fetch_owner import agent_tasks as owner_agent_tasks

logger = logging.getLogger("ops.fetch_agent")

ROOT = Path(__file__).resolve().parents[2]
_SECRET_NAME = re.compile(r"(PASS|SECRET|TOKEN|KEY|COOKIE|LOGIN|WEBHOOK|CREDENTIAL)", re.I)
_JWT = re.compile(r"eyJ[A-Za-z0-9_.-]{10,}")
TAIL_BYTES = 3000

# タスクごとの実行時間の上限（秒）。超えたら子プロセスのグループごと終了して失敗にする。
TIMEOUTS: dict[str, int] = {
    "race-odds-t60": 240, "race-odds-t15": 600, "race-odds-t10": 90, "race-odds-t05": 90, "race-odds-t02": 90,
    "race-weight-check": 120, "entries-sync": 900,
}
DEFAULT_TIMEOUT = 1800


@dataclass
class RunResult:
    rc: int
    tail: str = ""
    result: dict[str, Any] | None = None
    timed_out: bool = False
    elapsed: float = 0.0


Runner = Callable[[ft.FetchTask, int], RunResult]


@dataclass
class AgentConfig:
    agent_id: str
    enabled: bool = False
    tasks: frozenset[str] = frozenset()
    poll_seconds: float = 60.0
    stale_seconds: int = 900
    access_pause_seconds: int = 1800
    hold_on_restriction: bool = True      # 制限の疑いを受けたら、人が解除するまで新しいタスクを取らない（別経路へ切り替えない）
    python: str = sys.executable
    prune_days: int = 7

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AgentConfig":
        env = dict(os.environ) if env is None else env
        raw = env.get("KEIBA_FETCH_AGENT_TASKS", "").strip()
        tasks = {t.strip() for t in raw.split(",") if t.strip()} if raw else owner_agent_tasks()
        return cls(
            agent_id=env.get("KEIBA_FETCH_AGENT_ID") or f"vps-{socket.gethostname()}",
            enabled=env.get("KEIBA_FETCH_AGENT_ENABLED", "").strip().lower() in ("1", "true", "yes"),
            tasks=frozenset(tasks & ft.ALLOWED_TASKS),
            poll_seconds=float(env.get("KEIBA_FETCH_AGENT_POLL_SEC", "60")),
        )


def redact(text: str, env: dict[str, str] | None = None) -> str:
    """ログ・結果に残す前に、既知の秘密値と JWT 風の文字列を伏せる。"""
    env = os.environ if env is None else env
    for k, v in env.items():
        if v and len(v) >= 6 and _SECRET_NAME.search(k):
            text = text.replace(v, "<masked>")
    return _JWT.sub("<masked>", text)


def build_argv(t: ft.FetchTask, python: str) -> list[str]:
    base = [python, "-m", "src.ops.fetch_child"]
    if t.task in ft.RACE_ARG_TASKS:
        return base + ["race", t.task, str(t.args["race_id"]), str(t.args["date"])]
    return base + ["auto", t.task]


def subprocess_runner(python: str = sys.executable, cwd: Path = ROOT, extra_env: dict[str, str] | None = None) -> Runner:
    def run(t: ft.FetchTask, timeout: int) -> RunResult:
        t0 = time.time()
        with tempfile.TemporaryFile() as out:       # 出力はファイルへ（親のメモリを増やさない）
            proc = subprocess.Popen(build_argv(t, python), cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True, env={**os.environ, **(extra_env or {})})
            timed_out = False
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc = -9
            size = out.seek(0, os.SEEK_END)
            out.seek(max(0, size - 200_000))        # 結果行は最後の方にある
            text = out.read().decode("utf-8", "replace")
        result = None
        for line in reversed(text.splitlines()):
            if line.startswith(RESULT_PREFIX):
                try:
                    result = json.loads(line[len(RESULT_PREFIX):])
                except ValueError:
                    pass
                break
        return RunResult(rc=rc, tail=redact(text[-TAIL_BYTES:]), result=result, timed_out=timed_out,
                         elapsed=round(time.time() - t0, 1))
    return run


class FetchAgent:
    publish_events = True       # 馬体重の検知イベントを置く（影モードの実行役は False＝本番のイベントに触れない）
    skip_announced = True       # 発表済みの馬体重は取り直さない（影モードは比較のため取り直す）
    shadow: "FetchAgent | None" = None   # 影モードの実行役（本番のタスクが無いときに 1 本ずつ回す。同じ VPS の同じ IP なので停止を共有する）

    def __init__(self, queue: ft.TaskQueue, cfg: AgentConfig, runner: Runner,
                 clock: Callable[[], datetime] = ft.now_jst, sleep: Callable[[float], None] = time.sleep) -> None:
        self.q, self.cfg, self.runner, self.clock, self.sleep = queue, cfg, runner, clock, sleep
        self._cache: dict[str, ft.FetchTask] = {}
        self._paused_until: datetime | None = None
        self._last_prune: datetime | None = None
        self.stats = {"done": 0, "failed": 0, "requeued": 0, "expired": 0}

    # --- 1 回分 ---------------------------------------------------------------------------
    def housekeeping(self, now: datetime) -> None:
        self.stats["expired"] += len(self.q.expire_overdue(now, self._cache))
        self.q.reap_stale(self.cfg.stale_seconds, now)
        if self._last_prune is None or now - self._last_prune > timedelta(hours=6):
            self.q.prune(self.cfg.prune_days, now)
            self._last_prune = now
            live = set(self.q.list_ids("pending"))
            for k in [k for k in self._cache if k not in live]:
                del self._cache[k]

    def next_sleep(self, now: datetime) -> float:
        """次に起動できそうな pending までの秒数（poll_seconds を上限）。直前オッズ（lead 60 秒）を取り逃がさないため。"""
        nxt = None
        for t in self._cache.values():
            if t.task in self.cfg.tasks:
                s = t.start_at()
                if s > now and (nxt is None or s < nxt):
                    nxt = s
        if nxt is None:
            return self.cfg.poll_seconds
        return max(1.0, min(self.cfg.poll_seconds, (nxt - now).total_seconds()))

    def tick(self) -> str:
        """1 本だけ実行して終わる。戻り値: 'idle' | 'paused' | 'held'（制限の疑いで保留中。人が解除するまで動かない） | 'ran:<task_id>:<done|failed|requeued|expired|lost>'"""
        now = self.clock()
        if self.cfg.hold_on_restriction and access_hold.read_hold() is not None:
            return "held"
        if self._paused_until and now < self._paused_until:
            return "paused"
        self.housekeeping(now)
        due = self.q.due_pending(now, self._cache, only=self.cfg.tasks)
        if not due:
            return "idle"
        t = self.q.claim(due[0].task_id, self.cfg.agent_id, now)
        if t is None:
            return f"ran:{due[0].task_id}:lost"
        self._cache.pop(t.task_id, None)
        if self.skip_announced and t.task == ft.RACE_WEIGHT_TASK and self.q.store.get(self._weight_event_key(t)) is not None:
            self.q.complete(t, {"agent": self.cfg.agent_id, "skipped": "already_announced"}, self.clock())
            self.stats["done"] += 1
            return f"ran:{t.task_id}:done"
        # 時刻厳守のタスクは not_before まで待つ（lead_seconds 前に起動してよいが、取得は not_before 以降）
        wait = (ft.parse_ts(t.not_before) - self.clock()).total_seconds()
        if 0 < wait <= t.lead_seconds + 5:
            self.sleep(wait)
        now = self.clock()
        if now > ft.parse_ts(t.deadline):
            self.q._finish(t, "failed", {"reason": "deadline_exceeded", "agent": self.cfg.agent_id}, now)
            self.stats["expired"] += 1
            return f"ran:{t.task_id}:expired"
        timeout = TIMEOUTS.get(t.task, DEFAULT_TIMEOUT)
        res = self.runner(t, timeout)
        errors = list((res.result or {}).get("errors") or [])
        outcome = {"agent": self.cfg.agent_id, "rc": res.rc, "elapsed_sec": res.elapsed, "timed_out": res.timed_out,
                   "result": res.result, "tail": res.tail if (res.rc != 0 or errors) else ""}
        ok = res.rc == 0 and not errors and not res.timed_out
        if ok:
            self.q.complete(t, outcome, self.clock())
            self.stats["done"] += 1
            if self.publish_events:
                self._publish_events(t, res)
            return f"ran:{t.task_id}:done"
        restricted = bool((res.result or {}).get("access_restriction"))
        if restricted:
            outcome["access_restriction"] = True
            self._paused_until = self.clock() + timedelta(seconds=self.cfg.access_pause_seconds)
            logger.error("アクセス制限の疑い。人が解除するまで新しいタスクを取りません（別の経路へは切り替えません）: %s", t.task_id)
            if self.cfg.hold_on_restriction:
                access_hold.set_hold(f"制限の疑い: task={t.task} id={t.task_id}", task_id=t.task_id)
                self._notify_restricted(t)
            self._on_restricted(t, outcome)
        state = self.q.fail(t, outcome, self.clock(), retry=not restricted)
        self.stats["requeued" if state == "requeued" else "failed"] += 1
        return f"ran:{t.task_id}:{state}"

    def _notify_restricted(self, t: ft.FetchTask) -> None:
        """Slack に注意を出す（秘密は含めない）。通知の失敗で取得処理を止めない。"""
        try:
            from src.utils import slack_notify as SN

            SN.notify_event("取得元のアクセス制限の疑い（取得を保留中）",
                            [f"タスク: {t.task} ({t.task_id})", "別の User-Agent・IP・経路への切り替えは行いません。",
                             "原因を確認し、再開してよいと判断したら VPS で `python -m src.ops.access_hold clear` を実行してください。"],
                            severity="critical", channel="alert")
        except Exception:  # noqa: BLE001
            logger.exception("制限の通知に失敗")

    def _on_restricted(self, t: ft.FetchTask, outcome: dict[str, Any]) -> None:
        """アクセス制限の疑いを検知したとき（影モードの実行役が通知・停止の記録に使う）。"""

    def _share_pause(self) -> None:
        """本番と影は同じ VPS の同じ IP。どちらかが制限を疑ったら両方止める。"""
        if self.shadow is None:
            return
        ps = [p for p in (self._paused_until, self.shadow._paused_until) if p is not None]
        if ps:
            self._paused_until = self.shadow._paused_until = max(ps)

    @staticmethod
    def _weight_event_key(t: ft.FetchTask) -> str:
        return f"ops/events/weights_announced/{t.args['date']}/{t.args['race_id']}.json"

    def _publish_events(self, t: ft.FetchTask, res: RunResult) -> None:
        """馬体重が出たら GCS にイベントを置く（GCP 側がこれを見て予測を起動する）。create-only で冪等。"""
        if t.task == ft.RACE_WEIGHT_TASK and (res.result or {}).get("announced"):
            key = self._weight_event_key(t)
            body = json.dumps({"race_id": t.args["race_id"], "date": t.args["date"], "detected_at": ft.iso(self.clock()),
                               "agent": self.cfg.agent_id, "ratio": res.result.get("ratio")}, ensure_ascii=False)
            self.q.store.create(key, body)

    # --- 常駐 -----------------------------------------------------------------------------
    def run_forever(self) -> None:
        logger.info("取得の実行役を開始: id=%s tasks=%s", self.cfg.agent_id, sorted(self.cfg.tasks))
        stop = {"v": False}
        signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
        while not stop["v"]:
            try:
                state = self.tick() if self.cfg.tasks else "idle"
                if self.shadow is not None and not state.startswith("ran:"):   # 本番のタスクが無いときだけ影を 1 本回す
                    self._share_pause()
                    state = self.shadow.tick() if self.shadow._paused_until is None or self.clock() >= self.shadow._paused_until else "paused"
                    self._share_pause()
            except Exception:  # noqa: BLE001
                logger.exception("tick で例外（次の周期で再試行）")
                state = "error"
            if state.startswith("ran:"):
                continue                      # まだ期限の来たタスクがあるかもしれないので待たずに次へ
            if state == "idle":
                waits = [self.next_sleep(self.clock())] + ([self.shadow.next_sleep(self.clock())] if self.shadow is not None else [])
                self.sleep(min(waits))
            else:
                self.sleep(self.cfg.poll_seconds)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a = list(sys.argv[1:] if argv is None else argv)
    cfg = AgentConfig.from_env()
    if not cfg.enabled:
        logger.info("KEIBA_FETCH_AGENT_ENABLED が 1 でないため何もしません（既定オフ）")
        return 0
    from src.ops import fetch_shadow as fs

    shadow_on = fs.enabled()
    if not cfg.tasks and not shadow_on:
        logger.warning("担当タスクが空です（config/fetch_owner.json の agent_tasks / KEIBA_FETCH_AGENT_TASKS）。何もしません")
        return 0
    bucket = os.environ.get("GCS_BUCKET", "")
    store = ft.GcsTaskStore(bucket)
    queue = ft.TaskQueue(store)
    agent = FetchAgent(queue, cfg, subprocess_runner(cfg.python))
    if shadow_on:
        agent.shadow = fs.build_shadow_agent(store, cfg, cfg.python, owned=set(cfg.tasks))
        logger.info("影モード: 対象=%s", sorted(agent.shadow.cfg.tasks))
    if "--dry-run" in a:
        due = queue.due_pending(only=cfg.tasks)
        print(json.dumps([{"task_id": t.task_id, "not_before": t.not_before} for t in due], ensure_ascii=False))
        if agent.shadow is not None:
            sdue = agent.shadow.q.due_pending(only=agent.shadow.cfg.tasks)
            print(json.dumps({"shadow": [{"task_id": t.task_id, "not_before": t.not_before} for t in sdue]}, ensure_ascii=False))
        return 0
    if "--once" in a:
        state = agent.tick() if cfg.tasks else "idle"
        if agent.shadow is not None and not state.startswith("ran:"):
            state = "shadow:" + agent.shadow.tick()
        print(state)
        return 0
    agent.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
