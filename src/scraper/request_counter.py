"""netkeiba への GET の日次カウンタ（全取得者を合算）。1 リクエストごとに 1 バイトを日付別ファイルへ追記する（O_APPEND なのでプロセス間で数えられる）。

過去分補填は、この日の合計が『制限なしで続いた総件数』を超えないように、夜の予算を決めるために読む（backfill_guard.daily_budget）。
数えるのは NetkeibaClient の GET と、補填が API セッションで取るオッズ。失敗しても取得は止めない。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

_JST = timezone(timedelta(hours=9))
DIR = Path(os.environ.get("KEIBA_REQUEST_COUNT_DIR") or Path(__file__).resolve().parents[2] / "data" / "local" / "meta" / "request_counts")


def _path(day: str | None = None) -> Path:
    return DIR / f"{day or datetime.now(_JST).strftime('%Y%m%d')}.count"


def count_request(n: int = 1) -> None:
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        with open(_path(), "ab") as f:
            f.write(b"." * n)
    except OSError:
        pass


def count_today(day: str | None = None) -> int:
    try:
        return _path(day).stat().st_size
    except OSError:
        return 0
