"""過去分補填（2016〜2019 年）の安全装置。取得先への負荷を抑えるため、夜ごとの件数の上限・段階的な引き上げ・悪化の検知による中断・時間窓を一か所で扱う。

方針（docs/operations/backfill_2016.md）:
  * 上限は『制限を受けずに 3 日以上続いた 1 日あたりの総件数（約 3,200 件、通常運用を含む）』から通常運用の分（約 1,150 件）を引いた約 2,000 件/夜。超えない。
  * 最初の 1 夜はその半分（1,000 件）。問題のない夜が 1 夜できてから 2,000 件に上げる。悪化した夜があれば半分に戻す。
  * 取得間隔は変えない（既存のスロットル 2.2〜4.0 秒＋バースト休止。実測の中央値は約 4 秒）。並走しない（同時 1 本）。
  * 403/429/制限ページは既存の AccessGuard（scrape_access_pause に保留を書く）で全停止し、人が解除するまで再開しない。
    503・5xx・応答の遅延・通信エラーの増加は、その夜だけ自動で中断する。UA・接続元は切り替えない。
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))
STATE_PATH = Path(os.environ.get("KEIBA_BACKFILL_PAST_STATE") or Path(__file__).resolve().parents[2] / "data" / "local" / "meta" / "backfill_past_state.json")

# 全取得（通常運用＋補填）の 1 日の総件数の上限。2026-10-12〜10-22 の 11 日連続で HTTP 400(bot_blocked)/403/429 なしだった最大が 2,681 件（10/19）。
# 10/8〜10/9 は 3,241・4,213 件でも制限なしだったが、その直後の 10/10〜10/11 に bot_blocked の HTTP 400 が計 2,407 件出た（3,690 件の日）。
# 『制限なしで続いた範囲』は後者を採らず、制限後に安定して続いた 2,700 件までとする。
TOTAL_DAILY_LIMIT = int(os.environ.get("BACKFILL_PAST_TOTAL_DAILY_LIMIT", "2700"))
EXPECTED_WEEKDAY_OPS = int(os.environ.get("BACKFILL_PAST_EXPECTED_OPS", "1150"))   # 平日の通常運用の件数（朝の窓の後に走る分）
CAP_MAX = int(os.environ.get("BACKFILL_PAST_CAP_MAX", str((TOTAL_DAILY_LIMIT - EXPECTED_WEEKDAY_OPS) // 100 * 100)))   # 既定 1,500（100 件単位に切り下げ）
CAP_FIRST = CAP_MAX // 2
WINDOW = os.environ.get("BACKFILL_PAST_WINDOW", "02:30-05:45")          # JST。この外では始めない（途中でも終端で止める）
OTHER_FETCH_PATTERN = r"src\.scraper\.(auto_scrape|backfill)|backfill_horse_training|run_priority_rescrape|src\.scripts\.scraping\.backfill_"

# AIMD 型の速度調整: 応答が悪化したら待ちを倍に増やし（乗法的に遅く）、悪化なしが続けば少しずつ戻す（加法的に速く）。
# 戻しても、既存のスロットルより速くはならない（実績の最短間隔を超えない）。
AIMD_CLEAN_STEP = 50              # 悪化なしが続いた件数ごとに
AIMD_RECOVER_SEC = 0.5            # 追加の待ちをこの秒数だけ減らす
AIMD_FIRST_DELAY = 4.0            # 悪化の最初の反応で足す待ち
AIMD_MAX_DELAY = 60.0

# 悪化の検知（その夜を中断）
RECENT_REQUESTS = 100
MAX_BAD_IN_RECENT = 3            # 直近 100 件の中の 403/429/5xx がこの件数に達したら中断
MAX_TRANSPORT_ERRORS = 2         # 通信エラー（接続・タイムアウト）がこの件数に達したら中断
LATENCY_WINDOW = 30
LATENCY_FACTOR = 2.5             # 直近の応答時間の中央値が、その夜の最初の基準の何倍を超えたら中断
LATENCY_ABS_SEC = 8.0            # 基準に関わらず、中央値がこの秒数を超えたら中断


def _now() -> datetime:
    return datetime.now(JST)


def parse_window(spec: str = WINDOW) -> tuple[int, int]:
    a, b = spec.split("-")
    h1, m1 = (int(x) for x in a.split(":"))
    h2, m2 = (int(x) for x in b.split(":"))
    return h1 * 60 + m1, h2 * 60 + m2


def in_window(now: datetime | None = None, spec: str = WINDOW) -> bool:
    now = now or _now()
    start, end = parse_window(spec)
    m = now.hour * 60 + now.minute
    return start <= m < end


def is_raceday(now: datetime | None = None) -> bool:
    """土日（JRA の開催日）は補填しない。通常運用の取得を優先する。"""
    return (now or _now()).weekday() >= 5


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {"nights": []}
    except (OSError, ValueError):
        return {"nights": []}


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def daily_budget(counted_today: int, now: datetime | None = None) -> int:
    """この夜に使える上限: 日の総上限 − その日すでに数えた件数 − これから走る通常運用の見込み。"""
    return max(0, TOTAL_DAILY_LIMIT - int(counted_today) - EXPECTED_WEEKDAY_OPS)


def stage_cap(state: dict[str, Any], cap_max: int = CAP_MAX) -> int:
    """最初の 1 夜は半分。直前の夜が問題なく終わったときだけ上限まで上げる。悪化した夜の翌夜は半分に戻す。"""
    nights = state.get("nights") or []
    if not nights:
        return cap_max // 2
    last = nights[-1]
    return cap_max if last.get("healthy") else cap_max // 2


def other_fetch_running() -> bool:
    """他の取得系プロセス（開催日ランナー・他の補填・取り直し）が動いているか。自分は除く。"""
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return False
    import re

    me = os.getpid()
    for line in out.splitlines()[1:]:
        pid_s, _, args = line.strip().partition(" ")
        if not pid_s.isdigit() or int(pid_s) == me:
            continue
        if "backfill_guard" in args or "grep" in args:
            continue
        if re.search(OTHER_FETCH_PATTERN, args):
            return True
    return False


def wait_for_quiet(max_wait_sec: int = 1800, poll: float = 30.0, busy=other_fetch_running, sleep=time.sleep) -> bool:
    """他の取得が終わるまで待つ。待ちきれなければ False（その夜は始めない）。"""
    t0 = time.time()
    while busy():
        if time.time() - t0 > max_wait_sec:
            return False
        sleep(poll)
    return True


class NightGuard:
    """1 回の実行（1 夜）の取得件数・応答の健全性を数え、止めるべきかを返す。NetkeibaClient の応答オブザーバとして使う。"""

    HOSTS = ("netkeiba.com",)

    def __init__(self, cap: int, window: str = WINDOW, clock=_now, state_path: Path = STATE_PATH, sleep=time.sleep) -> None:
        self.cap = int(cap)
        self.window = window
        self.clock = clock
        self.state_path = state_path
        self.requests = 0
        self.bad = 0                       # 403/429/5xx の累計
        self.transport_errors = 0
        self.status_counts: dict[int, int] = {}
        self._recent_bad: deque[int] = deque(maxlen=RECENT_REQUESTS)
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._baseline: float | None = None
        self._first: list[float] = []
        self.stop_reason: str | None = None
        self.started_at = clock()
        self.extra_delay = 0.0              # AIMD: 次のリクエストの前に足す待ち（秒）
        self._clean_run = 0
        self.sleep = sleep

    # NetkeibaClient の応答オブザーバ
    def __call__(self, url: str, resp: Any) -> None:
        if not any(h in url for h in self.HOSTS):
            return
        code = int(getattr(resp, "status_code", 0) or 0)
        self.requests += 1
        self.status_counts[code] = self.status_counts.get(code, 0) + 1
        is_bad = code in (403, 429) or 500 <= code < 600
        self._recent_bad.append(1 if is_bad else 0)
        if is_bad:
            self.bad += 1
        try:
            sec = float(resp.elapsed.total_seconds())
        except Exception:  # noqa: BLE001
            sec = None
        if sec is not None:
            self._latencies.append(sec)
            if self._baseline is None:
                self._first.append(sec)
                if len(self._first) >= LATENCY_WINDOW:
                    self._baseline = statistics.median(self._first)
        self._evaluate()
        self._aimd(is_bad, sec)

    def _aimd(self, is_bad: bool, sec: float | None) -> None:
        slow = sec is not None and self._baseline is not None and sec > max(2.0, self._baseline * 2.0)
        if is_bad or slow:
            self._clean_run = 0
            self.extra_delay = min(AIMD_MAX_DELAY, max(AIMD_FIRST_DELAY, self.extra_delay * 2))
        else:
            self._clean_run += 1
            if self._clean_run >= AIMD_CLEAN_STEP and self.extra_delay > 0:
                self._clean_run = 0
                self.extra_delay = max(0.0, self.extra_delay - AIMD_RECOVER_SEC)
        if self.extra_delay > 0 and not self.stop_reason:
            self.sleep(self.extra_delay)       # 応答の直後（次のリクエストの前）に待つ

    def add_external(self, n: int = 1) -> None:
        """オブザーバを通らない取得（オッズ・2連系オッズの API セッション）を件数に足す（日次カウンタにも）。"""
        self.requests += n
        try:
            from src.scraper.request_counter import count_request

            count_request(n)
        except Exception:  # noqa: BLE001
            pass

    def fail(self, reason: str) -> None:
        """呼び出し側が検知した異常（連続して取得できない等）で、その夜を止める。"""
        if not self.stop_reason:
            self.stop_reason = reason

    def on_error(self, url: str, exc: BaseException) -> None:
        if not any(h in url for h in self.HOSTS):
            return
        self.transport_errors += 1
        self._evaluate()

    def _evaluate(self) -> None:
        if self.stop_reason:
            return
        if sum(self._recent_bad) >= MAX_BAD_IN_RECENT:
            self.stop_reason = f"403/429/5xx が直近 {len(self._recent_bad)} 件中 {sum(self._recent_bad)} 件"
        elif self.transport_errors >= MAX_TRANSPORT_ERRORS:
            self.stop_reason = f"通信エラー {self.transport_errors} 件"
        elif len(self._latencies) >= LATENCY_WINDOW:
            med = statistics.median(self._latencies)
            if med > LATENCY_ABS_SEC or (self._baseline and med > self._baseline * LATENCY_FACTOR and med > 2.0):
                self.stop_reason = f"応答の遅延（直近中央値 {med:.1f} 秒、基準 {self._baseline or 0:.1f} 秒）"

    def should_stop(self) -> str | None:
        """止める理由（文字列）または None。ループの区切りごとに呼ぶ。"""
        if self.stop_reason:
            return self.stop_reason
        if self.requests >= self.cap:
            return f"この夜の上限 {self.cap} 件に到達"
        if not in_window(self.clock(), self.window):
            return f"時間窓 {self.window}（JST）の外"
        return None

    def healthy(self) -> bool:
        return self.stop_reason is None and self.bad == 0 and self.transport_errors == 0

    def record(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """夜の結果を状態ファイルに追記する（翌夜の上限の判断に使う）。"""
        state = load_state(self.state_path)
        rec = {
            "date": self.started_at.strftime("%Y-%m-%d"),
            "started": self.started_at.isoformat(timespec="seconds"),
            "ended": self.clock().isoformat(timespec="seconds"),
            "cap": self.cap,
            "requests": self.requests,
            "bad_5xx_403_429": self.bad,
            "transport_errors": self.transport_errors,
            "status_counts": {str(k): v for k, v in sorted(self.status_counts.items())},
            "median_latency_sec": round(statistics.median(self._latencies), 2) if self._latencies else None,
            "extra_delay_end_sec": self.extra_delay,
            "effective_requests_per_hour": round(self.requests / max(1e-9, (self.clock() - self.started_at).total_seconds() / 3600), 0),
            "stop_reason": self.stop_reason,
            "healthy": self.healthy(),
        }
        if extra:
            rec.update(extra)
        state.setdefault("nights", []).append(rec)
        state["nights"] = state["nights"][-60:]
        save_state(state, self.state_path)
        return rec
