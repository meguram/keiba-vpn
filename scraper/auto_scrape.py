"""
JRA開催スケジュール連動 自動スクレイピング

JRA年間カレンダーと db.netkeiba の更新タイミングに合わせて
各種データを自動取得する。cron から --task で呼び出す。

┌─────────────────────────────────────────────────────────────────┐
│  タスク            │ 時刻         │ 対象日     │ 内容          │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  raceday-runner    │ 07:30 起動   │ 開催日のみ │ 朝: レース一覧│
│  (常駐プロセス)    │ 各R 15分前   │            │ 各R 15分前:   │
│                    │              │            │  出馬表+オッズ│
│                    │              │            │  +馬柱+追切   │
│                    │              │            │  +SmartRC     │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  raceday-evening   │ 17:30        │ 開催日のみ │ 結果          │
│                    │              │            │ + 確定オッズ  │
│                    │              │            │ + SmartRC     │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  weekly-update     │ 金 17:30     │ 毎週金曜   │ 先週分        │
│                    │              │            │ 結果+馬情報   │
│                    │              │            │ +指数+調教    │
└─────────────────────────────────────────────────────────────────┘

raceday-runner の動作:
  07:30 起動 → レース一覧取得 → 発走時刻を全取得
  → 各レースの発走15分前まで sleep
  → 15分前: 出馬表 + オッズ + 馬柱 + 追切 + SmartRC スクレイプ
  → 全レース完了後に終了

db.netkeiba の更新:
  - 基本 金曜17:00 に先週分のデータベースが更新される
  - 出馬表は当日レース前15分に全情報が揃う

Usage:
  python -m scraper.auto_scrape --task raceday-runner   # 開催日常駐 (cron 07:30)
  python -m scraper.auto_scrape --task raceday-evening   # 結果取得 (cron 17:30)
  python -m scraper.auto_scrape --task weekly-update      # 週次更新 (cron 金 17:30)
  python -m scraper.auto_scrape --status
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("scraper.auto_scrape")

LEAD_MINUTES = 15  # 発走何分前にスクレイプするか


def _load_race_calendar(output_dir: str = "data/jra_baba") -> dict:
    from scraper.jra_calendar import JRACalendarScraper
    scraper = JRACalendarScraper(output_dir=output_dir)
    return scraper.get_schedule(date.today().year)


def _is_race_day(calendar: dict, target: str | None = None) -> bool:
    target = target or date.today().isoformat()
    return any(d["date"] == target for d in calendar.get("race_days", []))


def _get_race_day_venues(calendar: dict, target: str | None = None) -> list[str]:
    target = target or date.today().isoformat()
    for d in calendar.get("race_days", []):
        if d["date"] == target:
            return [v["venue"] for v in d.get("venues", [])]
    return []


def _last_week_race_dates(calendar: dict) -> list[str]:
    today = date.today()
    last_sunday = today - timedelta(days=today.weekday() + 1)
    last_monday = last_sunday - timedelta(days=6)
    start = last_monday.isoformat()
    end = last_sunday.isoformat()
    return sorted(
        d["date"].replace("-", "")
        for d in calendar.get("race_days", [])
        if start <= d["date"] <= end
    )


# ═══════════════════════════════════════════════════════════════════
#  ステータス管理
# ═══════════════════════════════════════════════════════════════════

_STATUS_FILE = Path("data/meta/auto_scrape_status.json")


def _save_status(task: str, result: dict):
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status = {}
    if _STATUS_FILE.exists():
        try:
            status = json.loads(_STATUS_FILE.read_text())
        except Exception:
            pass
    status[task] = {"last_run": datetime.now().isoformat(timespec="seconds"), **result}
    _STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2))


def _load_status() -> dict:
    if _STATUS_FILE.exists():
        try:
            return json.loads(_STATUS_FILE.read_text())
        except Exception:
            pass
    return {}


# ═══════════════════════════════════════════════════════════════════
#  発走時刻の取得
# ═══════════════════════════════════════════════════════════════════

def _fetch_race_schedule(runner, date_str: str) -> list[dict]:
    """
    レース一覧を取得し、出馬表から発走時刻を収集。
    各レースの start_time を datetime に変換して返す。

    Returns: [{"race_id": "...", "venue": "...", "round": 1,
               "race_name": "...", "post_time": datetime, "start_time_str": "10:05"}, ...]
    """
    races = runner.scrape_race_list(date_str)
    if not races:
        return []

    today = date.today()
    schedule = []

    for race in races:
        rid = race["race_id"]
        card = runner.scrape_race_card(rid, skip_existing=True)

        start_str = ""
        if card:
            start_str = card.get("start_time", "")

        if not start_str:
            # 出馬表にstart_timeがない場合、R番号から推定
            rnd = race.get("round", 0)
            if isinstance(rnd, str):
                rnd = int(rnd) if rnd.isdigit() else 0
            if rnd <= 6:
                h = 9 + (rnd * 30) // 60
                m = 45 + (rnd * 30) % 60
            else:
                h = 12 + ((rnd - 5) * 30) // 60
                m = ((rnd - 5) * 30) % 60
            start_str = f"{h:02d}:{m:02d}"

        try:
            h, m = map(int, start_str.split(":"))
            post_dt = datetime.combine(today, datetime.min.time().replace(hour=h, minute=m))
        except (ValueError, TypeError):
            continue

        schedule.append({
            "race_id": rid,
            "venue": race.get("venue", card.get("venue", "") if card else ""),
            "round": race.get("round", ""),
            "race_name": race.get("race_name", card.get("race_name", "") if card else ""),
            "post_time": post_dt,
            "start_time_str": start_str,
        })

    schedule.sort(key=lambda x: x["post_time"])
    return schedule


# ═══════════════════════════════════════════════════════════════════
#  タスク: 開催日ランナー (各R発走15分前スクレイプ)
# ═══════════════════════════════════════════════════════════════════

def task_raceday_runner():
    """
    開催日に常駐するプロセス。

    1. レース一覧を取得し全レースの発走時刻を確定
    2. 各レースの発走15分前まで待機
    3. 15分前になったら出馬表+オッズ+馬柱+追切+SmartRCをスクレイプ
    4. 全レース完了で終了
    """
    calendar = _load_race_calendar()
    today_str = date.today().isoformat()

    if not _is_race_day(calendar, today_str):
        logger.info("今日 (%s) は開催日ではありません", today_str)
        _save_status("raceday-runner", {"status": "skipped", "reason": "non-race-day"})
        return

    venues = _get_race_day_venues(calendar, today_str)
    logger.info("=" * 60)
    logger.info("  開催日ランナー起動: %s", today_str)
    logger.info("  開催場: %s", ", ".join(venues))
    logger.info("=" * 60)

    from scraper.run import ScraperRunner
    runner = ScraperRunner()
    date_fmt = today_str.replace("-", "")

    # Step 1: レーススケジュール取得
    logger.info("[Phase 1] レーススケジュール取得中...")
    schedule = _fetch_race_schedule(runner, date_fmt)

    if not schedule:
        logger.warning("レーススケジュール取得失敗")
        _save_status("raceday-runner", {"status": "error", "reason": "no-schedule"})
        return

    first_post = schedule[0]["post_time"]
    last_post = schedule[-1]["post_time"]
    logger.info("  全 %d レース", len(schedule))
    logger.info("  最初の発走: %s (%s %sR %s)",
                first_post.strftime("%H:%M"),
                schedule[0]["venue"], schedule[0]["round"], schedule[0]["race_name"])
    logger.info("  最後の発走: %s (%s %sR %s)",
                last_post.strftime("%H:%M"),
                schedule[-1]["venue"], schedule[-1]["round"], schedule[-1]["race_name"])
    logger.info("  スクレイプ開始: %s (最初の発走%d分前)",
                (first_post - timedelta(minutes=LEAD_MINUTES)).strftime("%H:%M"), LEAD_MINUTES)
    logger.info("")

    # Step 2: 各レースを発走15分前にスクレイプ
    stats = {"races": len(schedule), "scraped": 0, "errors": []}
    scraped_ids: set[str] = set()

    for i, race in enumerate(schedule):
        rid = race["race_id"]
        target_time = race["post_time"] - timedelta(minutes=LEAD_MINUTES)
        now = datetime.now()

        if now < target_time:
            wait_sec = (target_time - now).total_seconds()
            logger.info("[%d/%d] %s %sR %s 発走 %s → %s まで待機 (あと %.0f分)",
                        i + 1, len(schedule),
                        race["venue"], race["round"], race["race_name"][:12],
                        race["start_time_str"],
                        target_time.strftime("%H:%M"),
                        wait_sec / 60)
            time.sleep(wait_sec)

        # 同時刻に複数レースがある場合もあるので、到達したら前後の未取得レースもまとめて処理
        batch = []
        for r in schedule:
            if r["race_id"] in scraped_ids:
                continue
            r_target = r["post_time"] - timedelta(minutes=LEAD_MINUTES)
            if datetime.now() >= r_target:
                batch.append(r)

        for r in batch:
            _scrape_single_race(runner, r, stats)
            scraped_ids.add(r["race_id"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("  全レースのプレレーススクレイプ完了")
    logger.info("  スクレイプ: %d/%d レース", stats["scraped"], stats["races"])
    if stats["errors"]:
        logger.warning("  エラー: %d 件", len(stats["errors"]))
    logger.info("=" * 60)

    _save_status("raceday-runner", {
        "status": "ok", "date": today_str, "venues": venues,
        **{k: v for k, v in stats.items() if k != "errors"},
        "error_count": len(stats["errors"]),
    })


def _scrape_single_race(runner, race: dict, stats: dict):
    """1レース分のプレレースデータをスクレイプ。"""
    rid = race["race_id"]
    label = f"{race['venue']} {race['round']}R {race['race_name'][:12]}"
    now_str = datetime.now().strftime("%H:%M:%S")

    logger.info("  ★ [%s] %s (%s) スクレイプ開始", now_str, label, rid)

    tasks = [
        ("出馬表",     lambda: runner.scrape_race_card(rid, skip_existing=False)),
        ("単複オッズ", lambda: runner.scrape_odds(rid, skip_existing=False)),
        ("2連オッズ",  lambda: runner.scrape_pair_odds(rid, skip_existing=False)),
        ("馬柱・調教", lambda: runner.scrape_shutuba_past(rid, skip_existing=False)),
        ("追い切り",   lambda: runner.scrape_oikiri(rid, skip_existing=False)),
        ("SmartRC",    lambda: runner.scrape_smartrc(rid)),
    ]

    for task_name, task_fn in tasks:
        try:
            result = task_fn()
            if result:
                logger.info("    ✓ %s", task_name)
            else:
                logger.warning("    - %s (データなし)", task_name)
        except Exception as e:
            logger.error("    ✗ %s: %s", task_name, e)
            stats["errors"].append(f"{task_name}:{rid}")

    stats["scraped"] += 1
    time.sleep(random.uniform(1.0, 3.0))


# ═══════════════════════════════════════════════════════════════════
#  タスク: 開催日夕方 (結果・確定オッズ)
# ═══════════════════════════════════════════════════════════════════

def task_raceday_evening():
    """
    開催日の全レース終了後に実行。

    取得対象:
      - race_result (レース結果)
      - race_odds (確定オッズ)
      - race_pair_odds (確定2連系オッズ)
      - smartrc_race (SmartRC fullresults)
    """
    calendar = _load_race_calendar()
    today_str = date.today().isoformat()

    if not _is_race_day(calendar, today_str):
        logger.info("今日 (%s) は開催日ではありません", today_str)
        _save_status("raceday-evening", {"status": "skipped", "reason": "non-race-day"})
        return

    venues = _get_race_day_venues(calendar, today_str)
    logger.info("=" * 60)
    logger.info("  開催日夕方スクレイプ: %s", today_str)
    logger.info("  開催場: %s", ", ".join(venues))
    logger.info("=" * 60)

    from scraper.run import ScraperRunner
    runner = ScraperRunner()
    date_fmt = today_str.replace("-", "")

    races = runner.scrape_race_list(date_fmt)
    if not races:
        logger.warning("レース一覧が空: %s", date_fmt)
        _save_status("raceday-evening", {"status": "error", "reason": "no-races"})
        return

    stats = {"races": len(races), "results": 0, "odds": 0, "pair_odds": 0,
             "smartrc": 0, "errors": []}

    for i, race in enumerate(races):
        rid = race["race_id"]
        logger.info("[%d/%d] %s %s", i + 1, len(races), rid, race.get("race_name", ""))

        for task_name, task_key, task_fn in [
            ("結果",       "results",   lambda: runner.scrape_race_result(rid, skip_existing=False)),
            ("確定オッズ", "odds",      lambda: runner.scrape_odds(rid, skip_existing=False)),
            ("確定2連",    "pair_odds", lambda: runner.scrape_pair_odds(rid, skip_existing=False)),
            ("SmartRC",    "smartrc",   lambda: runner.scrape_smartrc(rid)),
        ]:
            try:
                result = task_fn()
                if result:
                    stats[task_key] += 1
            except Exception as e:
                logger.error("  %sエラー [%s]: %s", task_name, rid, e)
                stats["errors"].append(f"{task_key}:{rid}")

        if i < len(races) - 1:
            time.sleep(random.uniform(2.0, 5.0))

    logger.info("=" * 60)
    logger.info("  完了: 結果=%d, オッズ=%d, 2連=%d, SmartRC=%d",
                stats["results"], stats["odds"], stats["pair_odds"], stats["smartrc"])
    logger.info("=" * 60)

    _save_status("raceday-evening", {
        "status": "ok", "date": today_str, "venues": venues,
        **{k: v for k, v in stats.items() if k != "errors"},
        "error_count": len(stats["errors"]),
    })


# ═══════════════════════════════════════════════════════════════════
#  タスク: 金曜週次更新
# ═══════════════════════════════════════════════════════════════════

def task_weekly_update():
    """
    毎週金曜17:30に実行。db.netkeiba が先週分を更新した後に取得。

    取得対象:
      - 先週の全開催日 の race_result / race_index / race_barometer
      - 出走馬の horse_result (馬情報の更新)
    """
    today = date.today()
    if today.weekday() != 4:
        logger.info("今日 (%s) は金曜ではありません", today.isoformat())
        _save_status("weekly-update", {"status": "skipped", "reason": "not-friday"})
        return

    calendar = _load_race_calendar()
    target_dates = _last_week_race_dates(calendar)

    if not target_dates:
        logger.info("先週の開催日なし")
        _save_status("weekly-update", {"status": "skipped", "reason": "no-races-last-week"})
        return

    logger.info("=" * 60)
    logger.info("  週次更新: 先週 %d 開催日のデータ取得", len(target_dates))
    logger.info("  対象日: %s", ", ".join(target_dates))
    logger.info("=" * 60)

    from scraper.run import ScraperRunner
    runner = ScraperRunner()

    total_stats = {"dates": len(target_dates), "races": 0, "results": 0,
                   "index": 0, "barometer": 0, "horses_updated": 0, "errors": []}

    all_horse_ids: set[str] = set()

    for date_str in target_dates:
        logger.info("--- %s ---", date_str)
        races = runner.scrape_race_list(date_str)
        if not races:
            continue

        total_stats["races"] += len(races)

        for i, race in enumerate(races):
            rid = race["race_id"]

            for task_name, task_key, task_fn in [
                ("結果",   "results",   lambda: runner.scrape_race_result(rid, skip_existing=False)),
                ("指数",   "index",     lambda: runner.scrape_speed_index(rid, skip_existing=False)),
                ("偏差値", "barometer", lambda: runner.scrape_barometer(rid, skip_existing=False)),
            ]:
                try:
                    result = task_fn()
                    if result:
                        total_stats[task_key] += 1
                        if task_key == "results":
                            for entry in result.get("entries", []):
                                hid = entry.get("horse_id", "")
                                if hid:
                                    all_horse_ids.add(hid)
                except Exception as e:
                    logger.error("  %sエラー [%s]: %s", task_name, rid, e)
                    total_stats["errors"].append(f"{task_key}:{rid}")

            if i < len(races) - 1:
                time.sleep(random.uniform(2.0, 5.0))

        time.sleep(random.uniform(10.0, 20.0))

    if all_horse_ids:
        logger.info("馬情報更新: %d 頭", len(all_horse_ids))
        updated = 0
        skipped = 0
        for hid in sorted(all_horse_ids):
            if runner.is_horse_fresh(hid):
                skipped += 1
                continue
            try:
                runner.scrape_horse(hid, skip_existing=False, with_history=True)
                updated += 1
            except Exception as e:
                logger.error("馬情報エラー [%s]: %s", hid, e)
                total_stats["errors"].append(f"horse:{hid}")
            if updated % 10 == 0:
                time.sleep(random.uniform(3.0, 8.0))

        total_stats["horses_updated"] = updated
        logger.info("馬情報: %d頭更新, %d頭スキップ", updated, skipped)

    logger.info("=" * 60)
    logger.info("  週次更新完了: %d日, 結果=%d, 指数=%d, 偏差値=%d, 馬=%d頭",
                total_stats["dates"], total_stats["results"],
                total_stats["index"], total_stats["barometer"],
                total_stats["horses_updated"])
    logger.info("=" * 60)

    _save_status("weekly-update", {
        "status": "ok", "target_dates": target_dates,
        **{k: v for k, v in total_stats.items() if k != "errors"},
        "error_count": len(total_stats["errors"]),
    })


# ═══════════════════════════════════════════════════════════════════
#  ステータス表示
# ═══════════════════════════════════════════════════════════════════

def show_status():
    status = _load_status()
    calendar = _load_race_calendar()
    today = date.today()

    print("\n=== 自動スクレイプ ステータス ===")

    task_labels = {
        "raceday-runner": "開催日ランナー (各R 15分前スクレイプ)",
        "raceday-evening": "開催日夕 (結果・確定オッズ)",
        "weekly-update": "金曜週次 (結果・指数・馬情報)",
    }

    for task_id, label in task_labels.items():
        info = status.get(task_id)
        if info:
            last = info.get("last_run", "")
            s = info.get("status", "")
            detail = ""
            if s == "ok":
                dt = info.get("date", "")
                venues = info.get("venues", [])
                scraped = info.get("scraped", info.get("results", ""))
                total = info.get("races", "")
                if venues:
                    detail = f"{dt} {', '.join(venues)} ({scraped}/{total}R)"
                elif "target_dates" in info:
                    detail = f"{len(info['target_dates'])}日分"
            elif s == "skipped":
                detail = info.get("reason", "")
            print(f"  [{task_id:18s}] {label}")
            print(f"    最終: {last}  状態: {s}  {detail}")
        else:
            print(f"  [{task_id:18s}] {label}")
            print(f"    (未実行)")

    print("\n--- 次回の予定 ---")
    upcoming = None
    for d in calendar.get("race_days", []):
        if d["date"] >= today.isoformat():
            upcoming = d
            break

    if upcoming:
        rd = upcoming["date"]
        v = ", ".join(vv["venue"] for vv in upcoming["venues"])
        wd = ["月", "火", "水", "木", "金", "土", "日"][date.fromisoformat(rd).weekday()]
        days_until = (date.fromisoformat(rd) - today).days
        when = {0: "今日", 1: "明日"}.get(days_until, f"{days_until}日後")
        print(f"  次の開催: {rd} ({wd}) {v} ({when})")
        print(f"    07:30 ランナー起動 → 各R発走15分前にスクレイプ")
        print(f"    17:30 結果+確定オッズ取得")

    nf = today + timedelta(days=(4 - today.weekday()) % 7)
    if nf == today and datetime.now().hour >= 18:
        nf += timedelta(days=7)
    print(f"  次の金曜: {nf.isoformat()} 17:30 → 週次更新 (db.netkeiba反映後)")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

TASKS = {
    "raceday-runner": task_raceday_runner,
    "raceday-evening": task_raceday_evening,
    "weekly-update": task_weekly_update,
}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="JRA開催スケジュール連動 自動スクレイピング")
    parser.add_argument("--task", choices=list(TASKS.keys()),
                        help="実行するタスク")
    parser.add_argument("--status", action="store_true",
                        help="ステータス表示")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.task:
        logger.info("=== auto_scrape: %s 開始 ===", args.task)
        start = time.time()
        try:
            TASKS[args.task]()
        except Exception as e:
            logger.error("タスク失敗: %s", e, exc_info=True)
            _save_status(args.task, {"status": "error", "error": str(e)})
        elapsed = time.time() - start
        logger.info("=== auto_scrape: %s 完了 (%.0f秒) ===", args.task, elapsed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
