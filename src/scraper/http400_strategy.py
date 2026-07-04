"""
HTTP 400 インテリジェント回復エンジン
======================================

netkeiba.com から受け取る 400 レスポンスを「原因」ごとに分類し、
原因に応じた最適なリカバリ戦略を適用する。

== 分類 ==
  CONTENT_NOT_FOUND  コンテンツが存在しない（正規 404 相当）→ リトライしない
  COOKIE_EXPIRED     セッション切れ          → 再ログインのみ（高速）
  BOT_BLOCKED        アンチボット検知         → 完全セッション再構築 + 長め待機
  TRANSIENT          一時的サーバエラー       → UA 変更のみ + 短い待機

== 段階的エスカレーション（原因が BOT_BLOCKED / COOKIE_EXPIRED の場合）==
  Tier 1 (attempt=0): UA ローテーションのみ        + 待機  3〜 8 秒
  Tier 2 (attempt=1): 再ログインのみ                + 待機  8〜18 秒
  Tier 3 (attempt≥2): 完全セッション再構築 + 再ログイン + 待機 18〜50 秒

  CONTENT_NOT_FOUND の場合は Tier 1 のみ試行し、それでも同じなら raise。

== グローバル 400 レートモニター ==
  スライディングウィンドウで 400 発生率を追跡し、急増を検出。
  「短時間に多発」ならば Tier 3 から始めて積極的に回復する。
"""

from __future__ import annotations

import enum
import logging
import random
import threading
import time
from collections import deque

logger = logging.getLogger("scraper.http400_strategy")

# ---------------------------------------------------------------------------
# 400 原因分類
# ---------------------------------------------------------------------------

class Cause400(enum.Enum):
    CONTENT_NOT_FOUND = "content_not_found"
    COOKIE_EXPIRED    = "cookie_expired"
    BOT_BLOCKED       = "bot_blocked"
    TRANSIENT         = "transient"


# ページ内テキストによる原因マッピング
_NOT_FOUND_PATTERNS: list[str] = [
    "お探しのページが見つかりません",
    "お探しのページは見つかりません",
    "ページが見つかりません",
    "該当データはありません",
    "該当するデータが見当たりません",
    "No Data",
    "データがありません",
    "このページは存在しません",
]

_COOKIE_EXPIRED_PATTERNS: list[str] = [
    "ログインしてください",
    "ログインが必要です",
    "会員ログイン",
    "セッションが切れました",
    "再度ログイン",
]

_BOT_BLOCKED_PATTERNS: list[str] = [
    "アクセスが制限されています",
    "アクセスが集中",
    "しばらく時間をおいて",
    "自動アクセス",
    "ロボット",
    "bot",
    "Bot",
    "access denied",
    "Access Denied",
    "Forbidden",
    "ブロック",
]


def classify_400_cause(resp: "requests.Response") -> Cause400:  # noqa: F821
    """
    400 レスポンスの原因を本文テキストから推定する。

    HTTPレスポンスの Content-Type が HTML でない場合や
    本文が空の場合は TRANSIENT として扱う。
    """
    try:
        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type and "text" not in content_type:
            return Cause400.TRANSIENT

        # EUC-JP / UTF-8 両対応でデコード
        raw = resp.content[:4096]
        for enc in ("utf-8", "euc-jp", "shift_jis", "latin-1"):
            try:
                text = raw.decode(enc, errors="replace")
                break
            except Exception:
                continue
        else:
            return Cause400.TRANSIENT

        for pat in _NOT_FOUND_PATTERNS:
            if pat in text:
                logger.debug("400 原因推定: CONTENT_NOT_FOUND (pattern=%r)", pat)
                return Cause400.CONTENT_NOT_FOUND

        for pat in _COOKIE_EXPIRED_PATTERNS:
            if pat in text:
                logger.debug("400 原因推定: COOKIE_EXPIRED (pattern=%r)", pat)
                return Cause400.COOKIE_EXPIRED

        for pat in _BOT_BLOCKED_PATTERNS:
            if pat.lower() in text.lower():
                logger.debug("400 原因推定: BOT_BLOCKED (pattern=%r)", pat)
                return Cause400.BOT_BLOCKED

        # パターン不一致 → デフォルトはブロック疑いとして扱う
        return Cause400.BOT_BLOCKED

    except Exception as e:
        logger.debug("400 原因分類でエラー: %s", e)
        return Cause400.TRANSIENT


# ---------------------------------------------------------------------------
# グローバル 400 レートモニター
# ---------------------------------------------------------------------------

class _RateMonitor:
    """
    スライディングウィンドウで 400 発生率を追跡する。
    スレッドセーフ。複数の NetkeibaClient インスタンスで共有される。
    """

    def __init__(self, window_seconds: float = 300.0, threshold: int = 4) -> None:
        self._window = window_seconds
        self._threshold = threshold
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def record(self) -> None:
        """400 発生を記録する。"""
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            self._evict(now)

    def rate_count(self) -> int:
        """ウィンドウ内の 400 発生件数を返す。"""
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return len(self._timestamps)

    def is_high_rate(self) -> bool:
        """発生率が閾値を超えているか。"""
        return self.rate_count() >= self._threshold

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


# プロセス内グローバルインスタンス（全 Client で共有）
_GLOBAL_RATE_MONITOR = _RateMonitor(window_seconds=300.0, threshold=4)


def global_400_rate_monitor() -> _RateMonitor:
    return _GLOBAL_RATE_MONITOR


# ---------------------------------------------------------------------------
# 段階的リカバリー
# ---------------------------------------------------------------------------

def get_recovery_tier(attempt: int, cause: Cause400, monitor: _RateMonitor) -> int:
    """
    試行回数・原因・グローバル発生率から適用する Tier を決定する。

    Tier 0: 何もしない（これは呼び出し側でスキップ扱い）
    Tier 1: UA ローテーションのみ
    Tier 2: 再ログインのみ（セッション保持）
    Tier 3: 完全セッション再構築 + 再ログイン
    """
    # 高発生率の場合は最初から Tier 3
    if monitor.is_high_rate():
        return 3

    # CONTENT_NOT_FOUND は Tier 1 のみ試行
    if cause == Cause400.CONTENT_NOT_FOUND:
        return 1 if attempt == 0 else -1  # -1 = give up

    # COOKIE_EXPIRED は Tier 2 → 3
    if cause == Cause400.COOKIE_EXPIRED:
        if attempt == 0:
            return 2
        return 3

    # BOT_BLOCKED / TRANSIENT は Tier 1 → 2 → 3
    return min(attempt + 1, 3)


def sleep_for_tier(tier: int) -> None:
    """Tier に応じたスリープ時間を適用する。"""
    ranges = {
        1: (3.0, 8.0),
        2: (8.0, 18.0),
        3: (18.0, 50.0),
    }
    lo, hi = ranges.get(tier, (5.0, 15.0))
    wait = random.uniform(lo, hi)
    logger.info("400 回復待機 Tier%d: %.1f 秒", tier, wait)
    time.sleep(wait)


def apply_recovery(
    tier: int,
    cause: Cause400,
    client: "NetkeibaClient",  # noqa: F821 – forward ref OK at runtime
    url: str,
    attempt: int,
) -> None:
    """
    指定 Tier のリカバリー処理を実行する。

    client は NetkeibaClient のインスタンス。
    このモジュールは client に直接依存しないため、
    必要なメソッドのみをダックタイピングで呼ぶ。
    """
    from src.scraper.client import _UA_POOL, _build_browser_headers  # 循環回避のため局所 import

    if tier == 1:
        # UA ローテーションのみ
        client._current_ua = random.choice(_UA_POOL)
        client._session.headers.update(_build_browser_headers(client._current_ua))
        logger.warning(
            "HTTP 400 Tier1 [%s] — UA ローテーション: %s",
            cause.value, url,
        )

    elif tier == 2:
        # 再ログインのみ（セッション再構築はしない）
        client._logged_in = False
        client.login()
        logger.warning(
            "HTTP 400 Tier2 [%s] — 再ログイン: %s",
            cause.value, url,
        )

    elif tier == 3:
        # 完全セッション再構築 + 再ログイン
        was_logged = client._logged_in
        client._refresh_session()
        if was_logged:
            client.login()
        logger.warning(
            "HTTP 400 Tier3 [%s] — セッション完全再構築 + 再ログイン: %s",
            cause.value, url,
        )

    sleep_for_tier(tier)
