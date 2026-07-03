#!/usr/bin/env python3
"""
crontab 相当の「今から約30日間」に実データ取得が走る予定のものだけを列挙し、順に実行する。

- raceday-eve: 各開催日 R について、実際の cron は (R-1) 18:00 JST。窓 [now, now+30d] に
  その時刻が入る R だけ ``run_raceday_eve_for_date`` を実行。
- horse-name-index: CLI は金曜以外スキップするため、窓内の金曜 18:00 の回数にかかわらず
  週次処理（馬名インデックス + 成長曲線）を1回だけ実行（内容は週次バッチの正本）。

Usage:
  cd /path/to/keiba-vpn && python3 scripts/run_scrapes_in_cron_window.py
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path

from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from src.scraper.auto_scrape import (
        _load_race_calendar,
        _save_status,
        run_raceday_eve_for_date,
    )
    from src.utils.horse_name_index import run_weekly_horse_name_index_update
    from src.pipeline.inference.growth_curve_service import run_weekly_growth_curve_update
    from src.utils.keiba_logging import script_basic_config

    script_basic_config()
    import logging

    log = logging.getLogger("run_scrapes_in_cron_window")

    now = datetime.now(JST)
    end = now + timedelta(days=30)
    cal = _load_race_calendar()

    eve_targets: list[tuple[str, str]] = []
    for d in cal.get("race_days", []):
        rd = d["date"]
        rdate = date.fromisoformat(rd)
        eve = datetime.combine(rdate - timedelta(days=1), dtime(18, 0), tzinfo=JST)
        if now < eve <= end:
            eve_targets.append((rd, eve.isoformat(timespec="seconds")))

    log.info("窓: %s ～ %s (JST)", now.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"))
    log.info("raceday-eve 相当の開催日数: %d", len(eve_targets))
    for rd, ev in eve_targets:
        log.info("  R=%s (eve at %s)", rd, ev)

    for i, (rd, _) in enumerate(eve_targets, 1):
        ymd = rd.replace("-", "")
        log.info("=== [%d/%d] run_raceday_eve_for_date(%s) ===", i, len(eve_targets), rd)
        t0 = time.time()
        try:
            result = run_raceday_eve_for_date(ymd)
            _save_status("raceday-eve", result)
            log.info(
                "完了 exit相当 status=%s err=%s (%.0fs)",
                result.get("status"),
                result.get("error_count", 0),
                time.time() - t0,
            )
        except Exception as exc:
            log.exception("raceday-eve 失敗: %s", rd)
            _save_status("raceday-eve", {"status": "error", "error": str(exc), "date": rd})

    log.info("=== horse-name-index + growth-curve（週次・強制1回）===")
    t0 = time.time()
    try:
        result_index = run_weekly_horse_name_index_update(_REPO)
        _save_status("horse-name-index", result_index)
        log.info("馬名インデックス: %s (%.0fs)", result_index, time.time() - t0)
        t1 = time.time()
        result_gc = run_weekly_growth_curve_update(_REPO)
        _save_status("growth-curve-weekly", result_gc)
        log.info("成長曲線: %s (%.0fs)", result_gc, time.time() - t1)
    except Exception:
        log.exception("週次バッチ失敗")
        return 1

    log.info("全ジョブ終了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
