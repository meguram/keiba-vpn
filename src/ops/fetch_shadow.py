"""VPS の取得の『影モード』。ホストが取得を続けたまま、VPS の実行役が同じ取得を実行して、結果を ``shadow/<日付>/`` 以下に保存する。

目的: prod（VPS）の取得が数日間うまくいくのを見届けるまで、stg（ホスト PC）の取得は止めない。影の成果物と本番データの日次比較が基準を
満たすまで、ホスト cron を止める手順（cutover_plan / job_owner / host_cron_switch.sh）は動かない（``shadow_eval`` の判定）。

- 有効化: ``KEIBA_FETCH_SHADOW_ENABLED=1``（既定オフ。実行役 ``KEIBA_FETCH_AGENT_ENABLED=1`` も必要）。対象・上限は ``config/fetch_shadow.json``。
- 影のタスク表は本番とは別の名前空間 ``ops/fetch_tasks_shadow/``。本番のタスク表・イベント（馬体重検知→予測の起動）には一切触れない。
- 書き込みは ``shadow_storage``（``shadow/`` 以外は例外）。netkeiba へのリクエストが増えるので、1 日の上限・ジッター・アクセス制限時の即停止を持つ。
- 二重のアクセスになる（ホスト + VPS）。DEC-023（同時に叩かない）は、同じ時刻に重ならないようジッターで外し、日量を上限で抑えて守る。

  python -m src.ops.fetch_shadow can-switch <cron_job_id>   # 担当を渡してよいか（影モードの基準を満たしたか）。満たさなければ終了コード 6
  python -m src.ops.fetch_shadow status                      # 設定・今日の予算・停止状態
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from src.ops import fetch_task as ft

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "fetch_shadow.json"
SHADOW_QUEUE_PREFIX = "ops/fetch_tasks_shadow"
SHADOW_ENV = {"KEIBA_FETCH_SHADOW": "1", "KEIBA_SYNC_PED_TBL_ON_SHUTUBA": "0", "KEIBA_SLACK_NOTIFY": "0"}
BLOCKED_EXIT = 6


def enabled(env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("KEIBA_FETCH_SHADOW_ENABLED", "").strip().lower() in ("1", "true", "yes")


# ───────────────────────── 設定 ─────────────────────────

@dataclass
class ShadowConfig:
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_requests_per_day: int = 2500
    access_pause_seconds: int = 3600
    max_restrictions_per_day: int = 2
    passing: dict[str, Any] = field(default_factory=dict)

    def task_names(self) -> frozenset[str]:
        return frozenset(t for t in self.tasks if t in ft.ALLOWED_TASKS)

    def est_requests(self, task: str) -> int:
        return int((self.tasks.get(task) or {}).get("est_requests", 5))

    def jitter_range(self, task: str) -> tuple[int, int]:
        lo, hi = (self.tasks.get(task) or {}).get("jitter", [30, 120])
        return int(lo), int(max(lo, hi))


def load_config(path: Path | None = None, env: dict[str, str] | None = None) -> ShadowConfig:
    env = os.environ if env is None else env
    p = Path(env.get("KEIBA_FETCH_SHADOW_CONFIG") or path or CONFIG_PATH)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ShadowConfig()
    lim = d.get("limits") or {}
    return ShadowConfig(tasks=dict(d.get("tasks") or {}), max_requests_per_day=int(lim.get("max_requests_per_day", 2500)),
                        access_pause_seconds=max(1800, int(lim.get("access_pause_seconds", 3600))),
                        max_restrictions_per_day=int(lim.get("max_restrictions_per_day", 2)), passing=dict(d.get("pass") or {}))


# ───────────────────────── 影のタスク表（名前空間を分ける）─────────────────────────

class PrefixedStore:
    """本番のタスク表のキー ``ops/fetch_tasks/…`` を ``ops/fetch_tasks_shadow/…`` に付け替える。
    それ以外のキー（イベント・本番データなど）には触れない: 触ろうとしたら例外（影が本番へ漏れない保証）。"""

    def __init__(self, inner: Any, shadow_prefix: str = SHADOW_QUEUE_PREFIX) -> None:
        self.inner, self.shadow_prefix = inner, shadow_prefix

    def _to(self, key: str) -> str:
        if not key.startswith(ft.PREFIX + "/"):
            raise ValueError(f"影のタスク表は {ft.PREFIX}/ 以外のキーに触れません: {key!r}")
        return self.shadow_prefix + key[len(ft.PREFIX):]

    def get(self, key: str) -> Any:
        return self.inner.get(self._to(key))

    def create(self, key: str, text: str) -> bool:
        return self.inner.create(self._to(key), text)

    def delete(self, key: str) -> bool:
        return self.inner.delete(self._to(key))

    def list(self, prefix: str) -> list[str]:
        return [ft.PREFIX + k[len(self.shadow_prefix):] for k in self.inner.list(self._to(prefix))]


# ───────────────────────── 計画（本番の計画から影のタスクを作る）─────────────────────────

def jitter_seconds(task_id: str, lo: int, hi: int) -> int:
    """task_id から決まる擬似乱数（再計画しても同じ値＝冪等）。"""
    h = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
    return lo + (h % (hi - lo + 1))


def shadow_tasks_from(real: list[ft.FetchTask], cfg: ShadowConfig, *, exclude: set[str] | frozenset[str] = frozenset(),
                      env: dict[str, str] | None = None) -> list[ft.FetchTask]:
    """本番の計画（全タスク）から、影モードの対象だけを、ジッターで少し後ろにずらして返す。実行役が担当済みのタスクは除く（本番で実行されるため）。"""
    if not enabled(env):
        return []
    names = cfg.task_names()
    out: list[ft.FetchTask] = []
    for t in real:
        if t.task not in names or t.task in exclude:
            continue
        thin = int((cfg.tasks.get(t.task) or {}).get("thin", 1))          # 繰り返しの多いタスク（馬体重の検知）は間引く
        if thin > 1 and int(hashlib.sha256(t.task_id.encode()).hexdigest()[8:12], 16) % thin != 0:
            continue
        lo, hi = cfg.jitter_range(t.task)
        shift = timedelta(seconds=jitter_seconds(t.task_id, lo, hi))
        nb = ft.parse_ts(t.not_before) + shift
        grace = max(60, int((ft.parse_ts(t.deadline) - ft.parse_ts(t.not_before)).total_seconds()))
        out.append(ft.make_task(t.task, nb, args=t.args or None, task_id=t.task_id, lead_seconds=t.lead_seconds,
                                grace_seconds=grace, max_attempts=1))
    return out


# ───────────────────────── 1 日の上限・停止状態（VPS ローカルの小さな状態ファイル）─────────────────────────

def _state_dir(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("KEIBA_FETCH_SHADOW_STATE_DIR") or (ROOT / "data" / "local" / "meta" / "fetch_shadow"))


class ShadowState:
    """当日のリクエスト見積りの累計と、アクセス制限の回数・停止期限。JST の日が変わるとリセット。"""

    def __init__(self, cfg: ShadowConfig, path: Path | None = None, clock: Callable[[], datetime] = ft.now_jst) -> None:
        self.cfg, self.clock = cfg, clock
        self.path = (path or _state_dir() / "state.json")

    def _load(self, now: datetime) -> dict[str, Any]:
        today = now.strftime("%Y%m%d")
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        if d.get("date") != today:
            d = {"date": today, "spent": 0, "restrictions": 0, "paused_until": d.get("paused_until")}
        return d

    def _save(self, d: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def try_spend(self, n: int) -> bool:
        now = self.clock()
        d = self._load(now)
        if d["spent"] + n > self.cfg.max_requests_per_day:
            return False
        d["spent"] += n
        self._save(d)
        return True

    def paused_until(self) -> datetime | None:
        d = self._load(self.clock())
        return ft.parse_ts(d["paused_until"]) if d.get("paused_until") else None

    def record_restriction(self) -> datetime:
        """アクセス制限の疑い。access_pause_seconds 止める。1 日に max_restrictions_per_day 回に達したら翌日まで止める。戻り値: 再開してよい時刻。"""
        now = self.clock()
        d = self._load(now)
        d["restrictions"] += 1
        until = now + timedelta(seconds=self.cfg.access_pause_seconds)
        if d["restrictions"] >= self.cfg.max_restrictions_per_day:
            until = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
        d["paused_until"] = ft.iso(until)
        self._save(d)
        return until

    def snapshot(self) -> dict[str, Any]:
        return self._load(self.clock())


# ───────────────────────── 実行（予算つき）─────────────────────────

class BudgetedRunner:
    """実行の前に当日の上限を確認する。超えるときは実行せず ``skipped=budget_exhausted``（取得しない・失敗にもしない）。"""

    def __init__(self, inner: Callable[..., Any], state: ShadowState, cfg: ShadowConfig) -> None:
        self.inner, self.state, self.cfg = inner, state, cfg

    def __call__(self, t: ft.FetchTask, timeout: int) -> Any:
        from src.ops.fetch_agent import RunResult

        if not self.state.try_spend(self.cfg.est_requests(t.task)):
            return RunResult(rc=0, result={"skipped": "budget_exhausted", "errors": []})
        return self.inner(t, timeout)


def notify_restriction(task: str, until: datetime) -> None:
    """Slack の注意（SN.send_payload 経由＝環境名つき）。失敗しても取得側には影響させない。"""
    try:
        from src.utils import slack_notify as SN

        SN.notify_event("影モード: アクセス制限の疑い（自動停止）",
                        [f"タスク: {task}", f"再開予定: {ft.iso(until)}", "ホストの取得は続いています。VPS の IP が制限されていないか確認してください"],
                        severity="warning", channel="alert")
    except Exception:  # noqa: BLE001
        pass


def build_shadow_agent(store: Any, agent_cfg: Any, python: str, *, owned: set[str] | frozenset[str] = frozenset(),
                       cfg: ShadowConfig | None = None, state: ShadowState | None = None,
                       runner: Callable[..., Any] | None = None, clock: Callable[[], datetime] = ft.now_jst,
                       sleep: Callable[[float], None] | None = None, notify: Callable[[str, datetime], None] = notify_restriction) -> Any:
    from src.ops import fetch_agent as fa

    cfg = cfg or load_config()
    state = state or ShadowState(cfg, clock=clock)
    names = cfg.task_names() - set(owned)
    acfg = fa.AgentConfig(agent_id=f"{agent_cfg.agent_id}-shadow", enabled=True, tasks=frozenset(names),
                          poll_seconds=agent_cfg.poll_seconds, stale_seconds=agent_cfg.stale_seconds,
                          access_pause_seconds=cfg.access_pause_seconds, python=python)
    inner = runner or fa.subprocess_runner(python, extra_env=SHADOW_ENV)
    kw: dict[str, Any] = {"clock": clock}
    if sleep is not None:
        kw["sleep"] = sleep
    return _agent_class()(ft.TaskQueue(PrefixedStore(store)), acfg, BudgetedRunner(inner, state, cfg), state=state, notify=notify, **kw)


_AGENT_CLS: type | None = None


def _agent_class() -> type:
    global _AGENT_CLS
    if _AGENT_CLS is not None:
        return _AGENT_CLS
    from src.ops.fetch_agent import FetchAgent

    class ShadowAgent(FetchAgent):
        """影のタスク表を処理する実行役。イベントを出さない・『馬体重は発表済み』で飛ばさない・アクセス制限で止める。"""

        publish_events = False
        skip_announced = False

        def __init__(self, queue: Any, cfg: Any, runner: Any, *, state: ShadowState, notify: Callable[[str, datetime], None], **kw: Any) -> None:
            super().__init__(queue, cfg, runner, **kw)
            self.state, self.notify = state, notify

        def tick(self) -> str:
            until = self.state.paused_until()
            if until and self.clock() < until:
                self._paused_until = until
                return "paused"
            return super().tick()

        def _on_restricted(self, t: Any, outcome: dict[str, Any]) -> None:
            until = self.state.record_restriction()
            self._paused_until = until
            self.notify(t.task, until)

    _AGENT_CLS = ShadowAgent
    return ShadowAgent


def __getattr__(name: str) -> Any:        # 遅延 import（fetch_agent が重い依存を持たないよう、循環を避ける）
    if name == "ShadowAgent":
        return _agent_class()
    raise AttributeError(name)


# ───────────────────────── CLI ─────────────────────────

def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 2 and a[0] == "can-switch":
        from src.ops import shadow_eval

        ok, reasons = shadow_eval.unit_gate_from_env(a[1])
        for r in reasons:
            print(r, file=sys.stderr)
        if ok:
            print(f"{a[1]}: 影モードの基準を満たしています（または影モードの対象外のジョブだけです）")
            return 0
        print(f"{a[1]}: 影モードの基準を満たしていないため、担当を渡せません・ホストの cron を止められません", file=sys.stderr)
        return BLOCKED_EXIT
    if a == ["status"]:
        cfg = load_config()
        st = ShadowState(cfg)
        print(json.dumps({"enabled": enabled(), "tasks": sorted(cfg.task_names()), "state": st.snapshot(),
                          "limit": cfg.max_requests_per_day, "pass": cfg.passing}, ensure_ascii=False, indent=1, default=str))
        return 0
    print("usage: python -m src.ops.fetch_shadow can-switch <cron_job_id> | status", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
