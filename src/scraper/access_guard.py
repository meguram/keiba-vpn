"""アクセス制限の即時検知。スクレイピング中のエラーには敏感に対応する。

方針: **「ページが存在しない」以外のエラーは、基本的にアクセス制限によるものとみなす。**
通常のスクレイパーは 1 カテゴリの失敗として握りつぶして次へ進む（``ScraperRunner._fetch_parse_save`` は
``except Exception``）ため、そのまま続けると制限を悪化させる。``NetkeibaClient`` の応答・エラーのオブザーバとして動く。

制限の疑いとして数えるもの（netkeiba.com のみ。既定は 1 回で即時停止）:
  * HTTP ステータス 400 以上（404 を含む）。ただし本文が「ページが見つかりません」等の **存在しないページ** は除く
  * 接続エラー・タイムアウト・リダイレクト過多など、応答を得られないエラー
  * 2xx でも、本文に「アクセスが制限されています」「アクセスが集中」「Access Denied」等が出ているブロックページ
数えないもの: パース結果が空・スキーマ不適合（データ側の問題。``schema_violations`` に記録される）。

trip すると ``AccessRestrictionDetected``（``BaseException``）を送出する。``except Exception`` に捕まらず、
ワーカー・ジョブの処理を突き抜けて呼び出し元（スクリプト）に届く。一度 trip したら、以降の全リクエスト
（別スレッドを含む）も同じ例外で即座に失敗する。

利用: ``with AccessGuard() as guard: queue.process_queue()``
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

# 本文の判定に使う語（http400_strategy の分類と同じ「存在しない」判定を流用）
BLOCK_PAGE_PATTERNS = ("アクセスが制限されています", "アクセスが集中", "Access Denied", "access denied")


class AccessRestrictionDetected(BaseException):
    """アクセス制限の疑いを検知した。``except Exception`` では捕まらない（意図的）。"""

    def __init__(self, url: str, status: int | None, count: int = 1, kind: str = "http_status", detail: str = "") -> None:
        self.url, self.status, self.count, self.kind, self.detail = url, status, count, kind, detail
        what = {"http_status": f"HTTP {status}", "transport_error": f"通信エラー {detail}",
                "block_page": "アクセス制限のページ"}.get(kind, kind)
        super().__init__(f"{what} を検知（アクセス制限の疑い）: {url}")


def _body(resp: Any, limit: int = 4096) -> str:
    """応答本文の先頭（EUC-JP / UTF-8 どちらでも読めるように）。取れなければ空文字。"""
    try:
        raw = getattr(resp, "content", None)
        if isinstance(raw, (bytes, bytearray)):
            for enc in ("utf-8", "euc-jp", "shift_jis"):
                try:
                    return bytes(raw[:limit]).decode(enc)
                except UnicodeDecodeError:
                    continue
            return bytes(raw[:limit]).decode("utf-8", errors="replace")
        txt = getattr(resp, "text", "")
        return txt[:limit] if isinstance(txt, str) else ""
    except Exception:  # noqa: BLE001
        return ""


def is_page_not_found(resp: Any) -> bool:
    """本文が「存在しないページ」を示しているか（アクセス制限ではない）。"""
    from src.scraper.http400_strategy import _NOT_FOUND_PATTERNS

    body = _body(resp)
    return any(p in body for p in _NOT_FOUND_PATTERNS)


def is_block_page(resp: Any) -> bool:
    body = _body(resp)
    return any(p in body for p in BLOCK_PAGE_PATTERNS)


class AccessGuard:
    HOSTS = ("netkeiba.com",)

    def __init__(self, statuses: set[int] | frozenset[int] | None = None, threshold: int = 1, *,
                 exempt_not_found: bool = True, watch_transport_errors: bool = True, watch_block_pages: bool = True) -> None:
        """statuses=None は「400 以上のすべて」。集合を渡すとそのステータスだけ。"""
        self.statuses = None if statuses is None else frozenset(statuses)
        self.threshold = max(1, int(threshold))
        self.exempt_not_found = exempt_not_found
        self.watch_transport_errors = watch_transport_errors
        self.watch_block_pages = watch_block_pages
        self.tripped: AccessRestrictionDetected | None = None
        self.tripped_at: str | None = None
        self._consecutive = 0
        self._lock = threading.Lock()
        self.seen: dict[int, int] = {}
        self.not_found = 0                     # 「ページが存在しない」として見逃した件数（参考）

    def _counts_status(self, code: int) -> bool:
        return (code >= 400) if self.statuses is None else (code in self.statuses)

    def _signal(self, url: str, status: int | None, kind: str, detail: str = "") -> None:
        self._consecutive += 1
        if self._consecutive >= self.threshold and self.tripped is None:
            self.tripped = AccessRestrictionDetected(url, status, self._consecutive, kind, detail)
            self.tripped_at = datetime.now().astimezone().isoformat(timespec="seconds")
            raise self.tripped

    # NetkeibaClient の応答オブザーバとして呼ばれる（ステータス判定・リトライの前）
    def __call__(self, url: str, resp: Any) -> None:
        if self.tripped is not None:
            raise self.tripped
        if not any(h in url for h in self.HOSTS):
            return
        code = int(getattr(resp, "status_code", 0) or 0)
        with self._lock:
            self.seen[code] = self.seen.get(code, 0) + 1
            if self._counts_status(code):
                if self.exempt_not_found and is_page_not_found(resp):
                    self.not_found += 1               # ページが存在しないだけ。制限とはみなさない（連続カウントもそのまま）
                    return
                self._signal(url, code, "http_status")
            elif 200 <= code < 300:
                if self.watch_block_pages and is_block_page(resp):
                    self._signal(url, code, "block_page")
                else:
                    self._consecutive = 0

    # 接続エラー・タイムアウトなど、応答を得られなかったとき
    def on_error(self, url: str, exc: BaseException) -> None:
        if self.tripped is not None:
            raise self.tripped
        if not self.watch_transport_errors or not any(h in url for h in self.HOSTS):
            return
        with self._lock:
            self._signal(url, None, "transport_error", f"{type(exc).__name__}: {str(exc)[:120]}")

    def __enter__(self) -> "AccessGuard":
        from src.scraper import client

        client.add_response_observer(self)
        return self

    def __exit__(self, *exc: object) -> None:
        from src.scraper import client

        client.remove_response_observer(self)
