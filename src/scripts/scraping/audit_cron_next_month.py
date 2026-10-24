#!/usr/bin/env python3
"""
登録済み crontab のうち、標準5フィールド形式の行について
「今から約30日間（JST）」に何回発火するか集計する。

@reboot 等は対象外（別途注記）。
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    print("Python 3.9+ と zoneinfo が必要です", file=sys.stderr)
    raise SystemExit(2)

JST = ZoneInfo("Asia/Tokyo")


@dataclass
class CronLine:
    raw: str
    minute: str
    hour: str
    dom: str
    month: str
    dow: str
    command: str


def _parse_cron_line(line: str) -> CronLine | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("@"):
        return None
    parts = line.split()
    if len(parts) < 6:
        return None
    return CronLine(
        raw=line,
        minute=parts[0],
        hour=parts[1],
        dom=parts[2],
        month=parts[3],
        dow=parts[4],
        command=" ".join(parts[5:]),
    )


def _match_field(field: str, value: int, min_v: int, max_v: int) -> bool:
    if field == "*":
        return True
    if "/" in field:
        base, step = field.split("/", 1)
        step = int(step)
        if base == "*":
            return (value - min_v) % step == 0
        # e.g. */3 from min_v
        return value in range(min_v, max_v + 1, step)  # loose; ok for */3 watchdog
    if "," in field:
        return any(_match_field(p.strip(), value, min_v, max_v) for p in field.split(","))
    if "-" in field and field[0].isdigit():
        a, b = field.split("-", 1)
        return int(a) <= value <= int(b)
    return int(field) == value


def _matches_cron(cl: CronLine, dt: datetime) -> bool:
    """dt は aware (JST)。"""
    if not _match_field(cl.minute, dt.minute, 0, 59):
        return False
    if not _match_field(cl.hour, dt.hour, 0, 23):
        return False
    if not _match_field(cl.dom, dt.day, 1, 31):
        return False
    if not _match_field(cl.month, dt.month, 1, 12):
        return False
    # crontab dow: 0=Sunday ... 6=Saturday (Vixie)
    dow = (dt.weekday() + 1) % 7
    if not _match_field(cl.dow, dow, 0, 6):
        return False
    return True


def count_fires(cl: CronLine, start: datetime, end: datetime) -> int:
    n = 0
    cur = start.replace(second=0, microsecond=0)
    if cur.second:
        cur = cur.replace(second=0)
    # 1分刻みで走査（ジョブ数が少ない前提）
    while cur <= end:
        if _matches_cron(cl, cur):
            n += 1
        cur += timedelta(minutes=1)
    return n


def main() -> int:
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if r.returncode != 0:
        print("crontab なしまたは取得失敗:", r.stderr.strip())
        return 1

    now = datetime.now(JST)
    end = now + timedelta(days=30)
    print(f"集計窓: {now.isoformat(timespec='seconds')} ～ {end.isoformat(timespec='seconds')} (JST)\n")

    optional_tags = (
        "KEIBA-VPN-RACEDAY-EVE",
        "KEIBA_BACKFILL",
        "KEIBA_JT_STATS",
        "KEIBA-VPN-WATCHDOG",
    )
    found_tags = {t: False for t in optional_tags}

    total_lines = 0
    for line in r.stdout.splitlines():
        cl = _parse_cron_line(line)
        if not cl:
            for t in optional_tags:
                if t in line:
                    found_tags[t] = True
            continue
        if "#" in cl.command:
            # cron 行末コメントは command に含まれる — 簡易にタグ検出
            pass
        for t in optional_tags:
            if t in line:
                found_tags[t] = True
        total_lines += 1
        n = count_fires(cl, now, end)
        short_cmd = cl.command[:100] + ("…" if len(cl.command) > 100 else "")
        print(f"回数(30日): {n:4d}  |  {cl.minute} {cl.hour} {cl.dom} {cl.month} {cl.dow}")
        print(f"  {short_cmd}\n")

    print("--- タグ検出（行コメント・コマンド内）---")
    for t, ok in found_tags.items():
        print(f"  {t}: {'あり' if ok else 'なし'}")

    print(f"\n解釈できた cron 行数: {total_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
