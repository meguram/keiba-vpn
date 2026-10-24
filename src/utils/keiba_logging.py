"""
リポジトリ全体で共有するログ書式（日付・時刻）と uvicorn 用 log_config ビルド。

* 1 行目のフォーマット: ``YYYY-MM-DD HH:MM:SS [LEVEL] logger: message``
* **asctime は常に Asia/Tokyo**（Docker / VPS が UTC でもログは日本時間で揃える）
* サーバ再起動スクリプトは ``logs/server_YYYYMMDD_HHMMSS.log`` へ追記（ファイル名の日時も JST）。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any

from zoneinfo import ZoneInfo

from uvicorn.logging import AccessFormatter, DefaultFormatter

# ログ・運用スケジュールの表示基準（netkeiba / JRA 前提）
LOG_TZ = ZoneInfo("Asia/Tokyo")


class JstUvicornDefaultFormatter(DefaultFormatter):
    """uvicorn 既定ログ行の asctime を Asia/Tokyo に（マルチワーカー時に pickle 可能なトップレベルクラス）。"""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return format_time_as_jst(record, datefmt or self.datefmt)


class JstUvicornAccessFormatter(AccessFormatter):
    """uvicorn アクセスログの asctime を Asia/Tokyo に。"""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return format_time_as_jst(record, datefmt or self.datefmt)

# ログ1行内の日付・時刻（秒まで）
STANDARD_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

# 通常の application / ライブラリ用（uvicorn 以外の StreamHandler 等）
STANDARD_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# サーバ起動用ログファイル名の接尾子（--prod / dev 共通。リポジトリ logs/ 配下に置く想定）
# 例: server_20260428_153045.log
def server_log_filename_stem(now=None) -> str:
    t = now or datetime.now(LOG_TZ)
    if getattr(t, "tzinfo", None) is None:
        t = t.replace(tzinfo=LOG_TZ)
    else:
        t = t.astimezone(LOG_TZ)
    return f"server_{t:%Y%m%d_%H%M%S}"


def format_time_as_jst(record: logging.LogRecord, datefmt: str | None) -> str:
    """logging.Formatter.formatTime 互換。record.created は UTC 基準のエポック秒。"""
    dt = datetime.fromtimestamp(record.created, tz=LOG_TZ)
    return dt.strftime(datefmt or STANDARD_DATE_FMT)


class JstFormatter(logging.Formatter):
    """ログ行の %(asctime)s を Asia/Tokyo で整形する。"""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return format_time_as_jst(record, datefmt or self.datefmt)


def standard_log_formatter() -> logging.Formatter:
    return JstFormatter(STANDARD_LOG_FORMAT, datefmt=STANDARD_DATE_FMT)


def apply_root_logging_basic(*, level: int = logging.INFO) -> None:
    """
    CLI/スクリプト用: root に1本だけ StreamHandler＋上記フォーマット（未設定時のみ）。

    既存ハンドラありのときは上書きしない（ライブラリが先に触った場合）。
    """
    root = logging.getLogger()
    if root.handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(standard_log_formatter())
    root.addHandler(h)
    root.setLevel(level)
    for noisy in ("urllib3", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def script_basic_config(
    level: int = logging.INFO,
    handlers: list[logging.Handler] | None = None,
    *,
    stream: Any = None,
) -> None:
    """
    CLI/スクリプト用: root へ 1 回、標準日時フォーマットを付与（Python 3.8+ は force で再設定）。

    ``handlers`` 未指定時は ``StreamHandler(stream)`` 1 本。既定の ``stream`` は ``stderr``。
    """
    fmt = standard_log_formatter()
    if handlers is None:
        s = stream if stream is not None else sys.stderr
        h = logging.StreamHandler(s)
        h.setFormatter(fmt)
        hs: list[logging.Handler] = [h]
    else:
        hs = list(handlers)
        for h in hs:
            h.setFormatter(fmt)
    k: dict[str, Any] = {"level": level, "handlers": hs}
    if sys.version_info >= (3, 8):
        k["force"] = True
    logging.basicConfig(**k)


def build_uvicorn_log_config(*, use_colors: bool | None = None) -> dict[str, Any]:
    """
    Uvicorn に渡す dict。DefaultFormatter/AccessFormatter で **asctime＋日付** を付与。

    * ``use_colors`` … None のとき、TTY なら従来どおり色（uvicorn 内部判定）。
    * asctime は **Asia/Tokyo**（stg / Docker UTC でも日本時間で一致）。
    * フォーマッタは **モジュール直下のクラス**（ワーカー spawn 時の log_config pickle 対応）。
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": JstUvicornDefaultFormatter,
                "fmt": "%(asctime)s [%(name)s] %(levelprefix)s %(message)s",
                "datefmt": STANDARD_DATE_FMT,
                "use_colors": use_colors,
            },
            "access": {
                "()": JstUvicornAccessFormatter,
                "fmt": (
                    '%(asctime)s [%(name)s] %(levelprefix)s '
                    '%(client_addr)s - "%(request_line)s" %(status_code)s'
                ),
                "datefmt": STANDARD_DATE_FMT,
                "use_colors": use_colors,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
