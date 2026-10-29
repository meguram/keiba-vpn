"""取得タスクの表（GCS 上のキュー）。GCP の司令塔がタスクを積み、VPS の実行役が pull して 1 本ずつ実行する。

レイアウト: ``gs://<BUCKET>/ops/fetch_tasks/{pending,running,done,failed}/<task_id>.json``

排他: 状態遷移はすべて「作成のみ可（``ifGenerationMatch=0``）」の書き込みで行う。
claim = ``running/<id>`` の作成に成功した者が勝ち（同時に 2 台が取っても片方は PreconditionFailed）。
その後 ``pending/<id>`` を消す。claim 後にクラッシュしたタスクは ``reap_stale`` が pending へ戻す。

netkeiba のログイン情報など秘密は、タスクにもタスクの結果にも入れない（引数は allowlist で検証する）。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

PREFIX = "ops/fetch_tasks"
STATES = ("pending", "running", "done", "failed")

# --- allowlist ---------------------------------------------------------------------------------
# 1. auto_scrape のモード（``python -m src.scraper.auto_scrape --task <mode>``）。取得中心で『1 回で終わる』ものだけ。
#    除外: raceday-eve（追走難度・想定オッズの前計算で ML を読む=VPS の 250M に収まらない。取得だけの raceday-eve-fetch を使う。計算は GCP Jobs）、
#    feature-accumulate・horse-name-index（計算）、raceday-runner / raceday-weight-runner / raceday-result-runner
#    （1 日居座る常駐ループ。1 本ずつ実行する実行役を塞ぐので、race-odds-* / race-weight-check / race-result-t15 の個別タスクに置き換える）。
AUTO_SCRAPE_MODES: frozenset[str] = frozenset({
    "entries-sync",
    "recent-retry",
    "barometer-retry",
    "daily-race-lists",
    "weekly-update",
    "catchup-missing",
    "jra-baba-morning",
    "raceday-evening",
    "raceday-eve-fetch",     # raceday-eve の取得だけ（計算は GCP の eve-precompute）。KEIBA_EVE_FETCH_ONLY=1
    "raceday-comment",
    "new-horse-pedigree",
    "shutuba-audit",         # 過去7日の出馬表の取りこぼしを、その場で取り直す（fetch_child が KEIBA_SHUTUBA_AUDIT_DIRECT=1 にする）
    "sire-line",             # 系統台帳の GCS 同期（通信は GCS のみ）
    "page-quality-check",    # 代表ページの取得とスキーマ照合（約 10 リクエスト。要約を ops/page_quality/ へ）
})

# 2. 1 レースの取得（開催日）。引数: {"race_id": 12 桁, "date": "YYYYMMDD"}
#    オッズ T-60 / T-15 / 直前 T-10 / T-5 / T-2 と、馬体重の検知（出馬表を取り直し、馬体重が入ったか判定）、
#    発走 15 分後の確定結果（race-result-t15。常駐の raceday-result-runner の 1 レース分）。
RACE_ODDS_TASKS: dict[str, str] = {
    "race-odds-t60": "t60",
    "race-odds-t15": "t15",
    "race-odds-t10": "t10",
    "race-odds-t05": "t05",
    "race-odds-t02": "t02",
}
RACE_WEIGHT_TASK = "race-weight-check"
RACE_RESULT_TASK = "race-result-t15"
RACE_ARG_TASKS: frozenset[str] = frozenset(RACE_ODDS_TASKS) | {RACE_WEIGHT_TASK, RACE_RESULT_TASK}

ALLOWED_TASKS: frozenset[str] = AUTO_SCRAPE_MODES | RACE_ARG_TASKS

# 実行役が受け持てるタスク（job_map の vps ジョブを ready=true にできる範囲。job_map.validate が見る）。
# 実際の担当は config/fetch_owner.json の agent_tasks（KEIBA_FETCH_AGENT_TASKS で上書き）で、そこに無いタスクは pending のまま残し、ホスト等に任せる。
# 2026-10-29: 全部を影モードで並走させる方針（棚卸し §14）で、重いもの（weekly-update など）も加えた。250M に収まるかは影の並走で実測する。
DEFAULT_AGENT_TASKS: frozenset[str] = RACE_ARG_TASKS | {
    "entries-sync", "recent-retry", "barometer-retry", "daily-race-lists", "jra-baba-morning",
    "raceday-evening", "raceday-eve-fetch", "raceday-comment", "new-horse-pedigree", "weekly-update", "shutuba-audit", "page-quality-check",
}

# 直前オッズは時刻が命。実行役は期限のこの秒数前から待機する（既定の lead_seconds）。
DEFAULT_LEAD_SECONDS: dict[str, int] = {"race-odds-t10": 60, "race-odds-t05": 60, "race-odds-t02": 60}
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_GRACE_SECONDS_ANY = 1800
# 直前系は遅れて取ると履歴を汚す（auto_scrape_queue.LATE_TOLERANCE_SECONDS と同じ 120 秒）。
DEFAULT_GRACE_SECONDS: dict[str, int] = {"race-odds-t10": 120, "race-odds-t05": 120, "race-odds-t02": 120}

_RACE_ID = re.compile(r"^\d{12}$")
_DATE = re.compile(r"^\d{8}$")
_TASK_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_FORBIDDEN_ARG_KEYS = re.compile(r"(pass|secret|token|key|cookie|login)", re.I)


class TaskError(ValueError):
    """タスクの内容が不正（allowlist 外・引数不正・秘密らしいキー）。"""


def now_jst() -> datetime:
    return datetime.now(JST)


def parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=JST)


def iso(dt: datetime) -> str:
    return dt.astimezone(JST).isoformat(timespec="seconds")


@dataclass
class FetchTask:
    task_id: str
    task: str
    args: dict[str, Any] = field(default_factory=dict)
    not_before: str = ""            # この時刻以降に実行する（ISO 8601）
    lead_seconds: int = 0           # not_before の何秒前から待機・起動してよいか
    deadline: str = ""              # この時刻を過ぎたら実行しない（failed に deadline_exceeded）
    attempts: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    claimed_by: str | None = None
    outcome: dict[str, Any] | None = None
    # 任意（監査・復旧用）
    created_at: str | None = None
    claimed_at: str | None = None
    finished_at: str | None = None

    def validate(self) -> None:
        if not _TASK_ID.match(self.task_id or ""):
            raise TaskError(f"task_id が不正: {self.task_id!r}")
        if self.task not in ALLOWED_TASKS:
            raise TaskError(f"allowlist にないタスク: {self.task!r}")
        for k in self.args:
            if _FORBIDDEN_ARG_KEYS.search(str(k)):
                raise TaskError(f"引数に秘密らしいキーは使えない: {k!r}")
        if self.task in RACE_ARG_TASKS:
            if set(self.args) != {"race_id", "date"}:
                raise TaskError(f"{self.task} の引数は race_id と date のみ")
            if not _RACE_ID.match(str(self.args["race_id"])):
                raise TaskError(f"race_id が不正: {self.args['race_id']!r}")
            if not _DATE.match(str(self.args["date"])):
                raise TaskError(f"date は YYYYMMDD: {self.args['date']!r}")
        elif self.args:
            raise TaskError(f"{self.task} は引数を取らない: {sorted(self.args)}")
        try:
            nb, dl = parse_ts(self.not_before), parse_ts(self.deadline)
        except (TypeError, ValueError) as e:
            raise TaskError(f"not_before / deadline が ISO 8601 でない: {e}") from e
        if dl < nb:
            raise TaskError("deadline が not_before より前")
        if not (0 <= int(self.lead_seconds) <= 3600):
            raise TaskError("lead_seconds は 0〜3600")
        if not (1 <= int(self.max_attempts) <= 5):
            raise TaskError("max_attempts は 1〜5")

    def start_at(self) -> datetime:
        return parse_ts(self.not_before) - timedelta(seconds=int(self.lead_seconds))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FetchTask":
        d = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def make_task(task: str, not_before: datetime, *, args: dict[str, Any] | None = None, task_id: str | None = None,
              deadline: datetime | None = None, lead_seconds: int | None = None,
              max_attempts: int = DEFAULT_MAX_ATTEMPTS, grace_seconds: int | None = None) -> FetchTask:
    """司令塔向けの生成ヘルパ。task_id は決定的（同じ内容を二重に積まない）。"""
    args = dict(args or {})
    grace = DEFAULT_GRACE_SECONDS.get(task, DEFAULT_GRACE_SECONDS_ANY) if grace_seconds is None else grace_seconds
    tid = task_id or "-".join(
        [task, not_before.astimezone(JST).strftime("%Y%m%dT%H%M")] + [str(args[k]) for k in sorted(args)]
    )
    t = FetchTask(
        task_id=tid, task=task, args=args, not_before=iso(not_before),
        lead_seconds=DEFAULT_LEAD_SECONDS.get(task, 0) if lead_seconds is None else lead_seconds,
        deadline=iso(deadline or (not_before + timedelta(seconds=grace))),
        max_attempts=max_attempts, created_at=iso(now_jst()),
    )
    t.validate()
    return t


# --- ストレージ（GCS / テスト用のメモリ実装）----------------------------------------------------
class TaskStore(Protocol):
    def get(self, key: str) -> tuple[str, int] | None: ...
    def create(self, key: str, text: str) -> bool: ...          # ifGenerationMatch=0。既にあれば False
    def delete(self, key: str) -> bool: ...                      # 無ければ False
    def list(self, prefix: str) -> list[str]: ...


class MemoryTaskStore:
    """テスト・ローカル用。GCS の generation 条件つき書き込みと同じ意味を持つ。"""

    def __init__(self) -> None:
        self._d: dict[str, tuple[str, int]] = {}
        self._gen = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[str, int] | None:
        with self._lock:
            return self._d.get(key)

    def create(self, key: str, text: str) -> bool:
        with self._lock:
            if key in self._d:
                return False
            self._gen += 1
            self._d[key] = (text, self._gen)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._d.pop(key, None) is not None

    def list(self, prefix: str) -> list[str]:
        with self._lock:
            return sorted(k for k in self._d if k.startswith(prefix))


class GcsTaskStore:
    """GCS 実装。dev 環境では生成できない（gcp_guard）。認証は VPS のサービスアカウント（読み書きは ops/fetch_tasks/ のみ）。"""

    def __init__(self, bucket: str, client: Any | None = None) -> None:
        from src.config.gcp_guard import assert_gcp_allowed

        assert_gcp_allowed("fetch_task GcsTaskStore")
        if not bucket:
            raise TaskError("バケット名が空（GCS_BUCKET）")
        if client is None:
            from src.config.gcp_credentials import storage_client  # 遅延 import

            client = storage_client()
        self._bucket = client.bucket(bucket)

    def get(self, key: str) -> tuple[str, int] | None:
        blob = self._bucket.get_blob(key)
        if blob is None:
            return None
        return blob.download_as_text(), int(blob.generation)

    def create(self, key: str, text: str) -> bool:
        from google.api_core.exceptions import PreconditionFailed

        try:
            self._bucket.blob(key).upload_from_string(text, content_type="application/json", if_generation_match=0)
            return True
        except PreconditionFailed:
            return False

    def delete(self, key: str) -> bool:
        from google.api_core.exceptions import NotFound

        try:
            self._bucket.blob(key).delete()
            return True
        except NotFound:
            return False

    def list(self, prefix: str) -> list[str]:
        return sorted(b.name for b in self._bucket.list_blobs(prefix=prefix))


# --- キュー操作 --------------------------------------------------------------------------------
def key_of(state: str, task_id: str) -> str:
    if state not in STATES:
        raise TaskError(f"不明な状態: {state}")
    return f"{PREFIX}/{state}/{task_id}.json"


def _id_of(key: str) -> str:
    return key.rsplit("/", 1)[-1][: -len(".json")]


class TaskQueue:
    def __init__(self, store: TaskStore) -> None:
        self.store = store

    def exists_anywhere(self, task_id: str) -> str | None:
        for st in STATES:
            if self.store.get(key_of(st, task_id)) is not None:
                return st
        return None

    def enqueue(self, t: FetchTask) -> bool:
        """積む。同じ task_id が既にどこかの状態にあれば積まない（冪等）。"""
        t.validate()
        if self.exists_anywhere(t.task_id):
            return False
        return self.store.create(key_of("pending", t.task_id), t.to_json())

    def load(self, state: str, task_id: str) -> FetchTask | None:
        got = self.store.get(key_of(state, task_id))
        return FetchTask.from_json(got[0]) if got else None

    def list_ids(self, state: str) -> list[str]:
        return [_id_of(k) for k in self.store.list(f"{PREFIX}/{state}/")]

    def due_pending(self, now: datetime | None = None, cache: dict[str, FetchTask] | None = None,
                    only: frozenset[str] | set[str] | None = None) -> list[FetchTask]:
        """起動してよい（not_before - lead_seconds <= now）pending を、実行予定の早い順に返す。
        running / done / failed に同じ id があるもの（claim 途中のクラッシュ残り）は除く。
        ``cache``: task_id → FetchTask の使い回し（pending の中身は変わらないので GET を省く）。
        ``only``: このタスク名だけを対象にする（段階移行用）。"""
        now = now or now_jst()
        gone = set()
        for st in ("running", "done", "failed"):
            gone.update(self.list_ids(st))
        cache = cache if cache is not None else {}
        out: list[FetchTask] = []
        for tid in self.list_ids("pending"):
            if tid in gone:
                continue
            t = cache.get(tid) or self.load("pending", tid)
            if t is None:
                continue
            cache[tid] = t
            if only is not None and t.task not in only:
                continue
            try:
                t.validate()
            except TaskError:
                continue   # 不正なタスクは実行しない（expire_overdue が failed へ）
            if t.start_at() <= now:
                out.append(t)
        out.sort(key=lambda t: (parse_ts(t.not_before), t.task_id))
        return out

    def claim(self, task_id: str, agent: str, now: datetime | None = None) -> FetchTask | None:
        """running/<id> を作成できた者だけが所有者になる。負けたら None。"""
        now = now or now_jst()
        t = self.load("pending", task_id)
        if t is None:
            return None
        t.attempts += 1
        t.claimed_by = agent
        t.claimed_at = iso(now)
        if not self.store.create(key_of("running", task_id), t.to_json()):
            return None
        self.store.delete(key_of("pending", task_id))
        return t

    def _finish(self, t: FetchTask, state: str, outcome: dict[str, Any], now: datetime | None) -> None:
        t.outcome = outcome
        t.finished_at = iso(now or now_jst())
        self.store.create(key_of(state, t.task_id), t.to_json())      # 既にあれば（二重完了）上書きしない
        self.store.delete(key_of("running", t.task_id))

    def complete(self, t: FetchTask, outcome: dict[str, Any], now: datetime | None = None) -> None:
        self._finish(t, "done", outcome, now)

    def fail(self, t: FetchTask, outcome: dict[str, Any], now: datetime | None = None, *, retry: bool = True) -> str:
        """失敗を記録。再試行できて期限内なら pending へ戻す（"requeued"）。そうでなければ failed（"failed"）。"""
        now = now or now_jst()
        if retry and t.attempts < t.max_attempts and now < parse_ts(t.deadline):
            t.outcome = {**outcome, "requeued": True}
            t.claimed_by = None
            running_key = key_of("running", t.task_id)
            if self.store.delete(running_key):
                self.store.create(key_of("pending", t.task_id), t.to_json())
            return "requeued"
        self._finish(t, "failed", outcome, now)
        return "failed"

    def reap_stale(self, stale_seconds: int = 900, now: datetime | None = None) -> list[str]:
        """claim したまま止まった running を、pending（再試行可・期限内）か failed へ戻す。"""
        now = now or now_jst()
        moved: list[str] = []
        for tid in self.list_ids("running"):
            t = self.load("running", tid)
            if t is None or not t.claimed_at:
                continue
            if (now - parse_ts(t.claimed_at)).total_seconds() < stale_seconds:
                continue
            res = self.fail(t, {"reason": "stale_running", "claimed_by": t.claimed_by}, now)
            moved.append(f"{tid}:{res}")
        return moved

    def expire_overdue(self, now: datetime | None = None, cache: dict[str, FetchTask] | None = None) -> list[str]:
        """期限（deadline）を過ぎた pending を実行せず failed に記録する。不正なタスクも failed へ。"""
        now = now or now_jst()
        moved: list[str] = []
        cache = cache if cache is not None else {}
        for tid in self.list_ids("pending"):
            t = cache.get(tid) or self.load("pending", tid)
            if t is None:
                continue
            cache[tid] = t
            reason = None
            try:
                t.validate()
                if now > parse_ts(t.deadline):
                    reason = "deadline_exceeded"
            except TaskError as e:
                reason = f"invalid:{e}"
            if reason:
                cache.pop(tid, None)
                self.store.delete(key_of("pending", tid))
                self._finish(t, "failed", {"reason": reason}, now)
                moved.append(f"{tid}:{reason}")
        return moved

    def prune(self, keep_days: int = 7, now: datetime | None = None) -> int:
        """done / failed のうち完了から keep_days 日を過ぎたものを消す（一覧が肥大して LIST が遅くなるのを防ぐ）。"""
        now = now or now_jst()
        n = 0
        for st in ("done", "failed"):
            for tid in self.list_ids(st):
                t = self.load(st, tid)
                ts = t.finished_at or t.not_before if t else None
                if t and ts and (now - parse_ts(ts)).days >= keep_days:
                    n += int(self.store.delete(key_of(st, tid)))
        return n
