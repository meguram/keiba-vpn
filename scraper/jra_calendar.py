"""
JRA年間開催スケジュール取得モジュール

JRA公式カレンダーJSON API を利用して、年間の全開催日・開催場・
前日発表日を構造化データとして取得・保存する。

API: https://www.jra.go.jp/keiba/common/calendar/json/YYYYMM.json

Usage:
  python -m scraper.jra_calendar                 # 今年のスケジュール取得
  python -m scraper.jra_calendar --year 2025     # 指定年
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("scraper.jra_calendar")

CALENDAR_API = "https://www.jra.go.jp/keiba/common/calendar/json/{ym}.json"

VENUE_CODE_MAP = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}

# 全場の計測完了後にスクレイプする (一番遅い場の時刻に合わせる)
# 実績: 前日発表 → 中京09:30 / 阪神10:00 / 中山10:30 (最遅)
# 実績: 当日朝  → 中京06:30 / 中山07:00 / 阪神07:30 (最遅)
PRE_DAY_POLL_WINDOW = ("10:30", "11:30")
RACE_DAY_POLL_WINDOW = ("07:30", "08:30")


class JRACalendarScraper:
    """JRA公式カレンダーAPIから年間開催スケジュールを取得。"""

    def __init__(self, output_dir: str = "data/jra_baba"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })

    def fetch_year(self, year: int) -> dict[str, Any]:
        """
        指定年の全開催スケジュールを取得。

        Returns:
            {
                "year": 2026,
                "fetched_at": "2026-03-18T21:00:00",
                "race_days": [
                    {"date": "2026-01-04", "weekday": "日曜", "venues": [...], "is_irregular": false},
                    ...
                ],
                "poll_schedule": [
                    {"date": "2026-01-03", "type": "pre_day", "poll_window": ["09:00", "11:30"], "race_date": "2026-01-04"},
                    {"date": "2026-01-04", "type": "race_day", "poll_window": ["05:30", "08:30"], "venues": [...]},
                    ...
                ]
            }
        """
        race_days = self._fetch_race_days(year)
        if not race_days:
            logger.warning("%d年のスケジュールデータなし", year)
            return {"year": year, "race_days": [], "poll_schedule": []}

        poll_schedule = self._build_poll_schedule(race_days)

        result = {
            "year": year,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "total_race_days": len(race_days),
                "irregular_days": len([d for d in race_days if d["is_irregular"]]),
                "total_poll_days": len(poll_schedule),
            },
            "race_days": race_days,
            "poll_schedule": poll_schedule,
        }

        self._save(year, result)
        return result

    def _fetch_race_days(self, year: int) -> list[dict]:
        """全月のJSONを取得して実開催日のみ抽出。"""
        race_days: list[dict] = []
        for month in range(1, 13):
            ym = f"{year}{month:02d}"
            url = CALENDAR_API.format(ym=ym)
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code != 200:
                    logger.debug("%d月: %d", month, resp.status_code)
                    continue
            except Exception as e:
                logger.error("%d月取得失敗: %s", month, e)
                continue

            data = resp.json()
            for entry in data:
                m = int(entry["month"])
                for dd in entry["data"]:
                    venues = []
                    grade_races = []
                    for info in dd.get("info", []):
                        for race in info.get("race", []):
                            name = race.get("name", "")
                            if name:
                                venues.append(self._parse_venue(name))
                        for gr in info.get("gradeRace", []):
                            grade_races.append(gr.get("name", ""))

                    if not venues:
                        continue

                    day_num = int(dd["date"])
                    weekday_str = dd.get("day", "")
                    date_str = f"{year}-{m:02d}-{day_num:02d}"

                    is_irregular = weekday_str not in ("土曜", "日曜")

                    race_days.append({
                        "date": date_str,
                        "weekday": weekday_str,
                        "venues": venues,
                        "grade_races": [g for g in grade_races if g],
                        "is_irregular": is_irregular,
                    })

            count = len([d for d in race_days if d["date"].startswith(f"{year}-{month:02d}")])
            if count > 0:
                logger.info("  %d月: %d開催日", month, count)

        race_days.sort(key=lambda x: x["date"])
        return race_days

    @staticmethod
    def _parse_venue(name: str) -> dict:
        """'1回中山' → {kai: 1, venue: '中山', venue_code: '06'}"""
        import re
        m = re.match(r"(\d+)回(.+)", name.strip())
        if not m:
            return {"kai": 0, "venue": name, "venue_code": ""}
        venue = m.group(2)
        return {
            "kai": int(m.group(1)),
            "venue": venue,
            "venue_code": VENUE_CODE_MAP.get(venue, ""),
        }

    def _build_poll_schedule(self, race_days: list[dict]) -> list[dict]:
        """
        開催日リストからポーリングスケジュールを構築。
        各開催日 → 当日朝ポーリング + 前日ポーリング(前日発表)。
        """
        race_date_set = {d["date"] for d in race_days}
        race_day_map = {d["date"]: d for d in race_days}

        poll_entries: dict[str, dict] = {}

        for rd in race_days:
            rd_date = rd["date"]

            # 当日: 朝計測 (05:30〜08:30)
            if rd_date not in poll_entries:
                poll_entries[rd_date] = {
                    "date": rd_date,
                    "type": "race_day",
                    "poll_window": list(RACE_DAY_POLL_WINDOW),
                    "venues": [v["venue"] for v in rd["venues"]],
                }
            else:
                existing = poll_entries[rd_date]
                if existing["type"] == "pre_day":
                    existing["type"] = "both"
                    existing["poll_window"] = list(RACE_DAY_POLL_WINDOW)
                    existing["pre_day_window"] = list(PRE_DAY_POLL_WINDOW)
                    existing["venues"] = [v["venue"] for v in rd["venues"]]

            # 前日: 前日発表 (09:00〜11:30)
            y, m, d = map(int, rd_date.split("-"))
            prev = date(y, m, d) - timedelta(days=1)
            prev_str = prev.isoformat()

            if prev_str in poll_entries:
                existing = poll_entries[prev_str]
                if existing["type"] == "race_day":
                    existing["type"] = "both"
                    existing["pre_day_window"] = list(PRE_DAY_POLL_WINDOW)
                    existing["pre_for"] = rd_date
            else:
                poll_entries[prev_str] = {
                    "date": prev_str,
                    "type": "pre_day",
                    "poll_window": list(PRE_DAY_POLL_WINDOW),
                    "race_date": rd_date,
                    "venues": [v["venue"] for v in rd["venues"]],
                }

        schedule = sorted(poll_entries.values(), key=lambda x: x["date"])
        return schedule

    def _save(self, year: int, data: dict):
        out = self.output_dir / f"race_calendar_{year}.json"
        out.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("保存: %s", out)

    def get_schedule(self, year: int | None = None) -> dict[str, Any]:
        """保存済みスケジュールをロード。なければ取得。"""
        if year is None:
            year = date.today().year
        path = self.output_dir / f"race_calendar_{year}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            age_days = (datetime.now() - datetime.fromisoformat(data["fetched_at"])).days
            if age_days < 30:
                return data
            logger.info("スケジュール更新 (%d日経過)", age_days)
        return self.fetch_year(year)


def get_today_poll_windows(output_dir: str = "data/jra_baba") -> list[tuple[str, str]]:
    """
    今日のポーリング窓を返す。
    外部からの利用を想定したヘルパー関数。

    Returns:
        [("05:30", "08:30")] or [("09:00", "11:30")] or
        [("05:30", "08:30"), ("09:00", "11:30")] or []
    """
    scraper = JRACalendarScraper(output_dir=output_dir)
    try:
        schedule = scraper.get_schedule()
    except Exception:
        return []

    today = date.today().isoformat()
    for entry in schedule.get("poll_schedule", []):
        if entry["date"] == today:
            windows = []
            t = entry["type"]
            if t == "race_day":
                windows.append(tuple(entry["poll_window"]))
            elif t == "pre_day":
                windows.append(tuple(entry["poll_window"]))
            elif t == "both":
                windows.append(tuple(entry["poll_window"]))
                if "pre_day_window" in entry:
                    windows.append(tuple(entry["pre_day_window"]))
            return windows
    return []


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="JRA年間開催スケジュール取得")
    parser.add_argument("--year", type=int, default=date.today().year,
                        help="取得年 (default: 今年)")
    args = parser.parse_args()

    scraper = JRACalendarScraper()
    result = scraper.fetch_year(args.year)

    print(f"\n=== {args.year}年 JRA開催スケジュール ===")
    print(f"開催日: {result['summary']['total_race_days']}日")
    print(f"変則開催: {result['summary']['irregular_days']}日")
    print(f"馬場監視日: {result['summary']['total_poll_days']}日")

    irregular = [d for d in result["race_days"] if d["is_irregular"]]
    if irregular:
        print(f"\n--- 変則開催一覧 ---")
        for d in irregular:
            v = ", ".join(v["venue"] for v in d["venues"])
            print(f"  {d['date']} ({d['weekday']}) {v}")


if __name__ == "__main__":
    main()
