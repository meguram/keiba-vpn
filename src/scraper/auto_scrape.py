"""
JRA開催スケジュール連動 自動スクレイピング

JRA年間カレンダーと db.netkeiba の更新タイミングに合わせて
各種データを自動取得する。cron から --task で呼び出す。

┌─────────────────────────────────────────────────────────────────┐
│  タスク            │ 時刻         │ 対象日     │ 内容          │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  raceday-eve       │ 前日 18:00   │ 翌開催日   │ 出馬表+馬柱   │
│                    │              │ の前日のみ │ +追い切り     │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  raceday-runner    │ 07:30 起動   │ 開催日のみ │ 朝: レース一覧│
│  (常駐プロセス)    │ 各R 15分前   │            │ 各R 15分前:   │
│                    │              │            │  出馬表+オッズ│
│                    │              │            │  +SmartRC     │
│                    │              │            │  +JRA馬場     │
│                    │              │            │  →完了後AI予測│
│                    │              │            │ ※馬柱・追切は │
│                    │              │            │  前日取得済なら│
│                    │              │            │  スキップ     │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  raceday-result-   │ 07:30 起動   │ 開催日のみ │ 各R発走15分後:│
│  runner            │ 各R+15分後   │            │  速報結果取得 │
│  (常駐プロセス)    │              │            │               │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  raceday-evening   │ 17:30        │ 開催日のみ │ 結果          │
│                    │              │            │ + 確定オッズ  │
│                    │              │            │ + SmartRC     │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  weekly-update     │ 金 17:30     │ 毎週金曜   │ 先週分        │
│                    │              │            │ 結果+馬情報   │
│                    │              │            │ +指数+調教    │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  daily-race-lists  │ 毎日 07:00   │ 毎日       │ 今日+14日先   │
│                    │              │            │ のrace_lists  │
│                    │              │            │ 取得・更新    │
├────────────────────┼──────────────┼────────────┼───────────────┤
│  jra-baba-morning  │ 毎10分       │ 開催日+前日│ JRA馬場情報   │
│  (cron 5:00-9:00)  │ 05:00-09:00  │            │ クッション値  │
│                    │              │            │ 含水率・馬場  │
└─────────────────────────────────────────────────────────────────┘

raceday-eve の動作:
  前日 18:00 起動 → 翌開催日の全レース一覧を取得
  → race_shutuba (出馬表) を全レース分取得（前日時点の最新枠順・騎手）
  → race_shutuba_past (馬柱・調教) と race_oikiri (追い切り) を全レース分取得
  → 追走難度キャッシュを事前計算（出馬表のみ・result 非使用）
  → T-15 では出馬表/馬柱・追切は skip_existing=True でスキップ可
  → 終了 (非常駐)

raceday-runner の動作:
  07:30 起動 → レース一覧取得 → 発走時刻を全取得
  → 各レースの発走15分前まで sleep
  → 15分前: 出馬表 + オッズ + SmartRC + JRA馬場ライブ スクレイプ
     (馬柱・追い切りは前日取得済みならスキップ、未取得の場合はフォールバック取得)
  → 各 R 取得完了をトリガに AI 予測（モック / KEIBA_PRE_RACE_PREDICT_ENABLED=1）
  → 全レース完了後に終了

db.netkeiba の更新:
  - 基本 金曜17:00 に先週分のデータベースが更新される
  - 出馬表は当日レース前15分に全情報が揃う

Usage:
  python -m src.scraper.auto_scrape --task raceday-runner          # 開催日常駐 T-15 (cron 07:30)
  python -m src.scraper.auto_scrape --task raceday-result-runner   # 開催日常駐 T+15速報 (cron 07:30)
  python -m src.scraper.auto_scrape --task raceday-evening         # 結果取得 (cron 17:30)
  python -m src.scraper.auto_scrape --task weekly-update       # 週次更新 (cron 金 17:30)
  python -m src.scraper.auto_scrape --task daily-race-lists    # 日次race_lists更新 (cron 07:00)
  python -m src.scraper.auto_scrape --status
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

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger("scraper.auto_scrape")

LEAD_MINUTES = 15    # 発走何分前にスクレイプするか
RESULT_OFFSET_MINUTES = 15  # 発走何分後に速報結果を取得するか
RACE_LIST_FETCH_DAYS_AHEAD = 14  # daily-race-lists で先読みする日数


def _load_race_calendar(output_dir: str = "data/page_reference/cushion") -> dict:
    from src.scraper.jra_calendar import JRACalendarScraper
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


def _future_race_dates_iso(calendar: dict) -> list[str]:
    """今日〜カレンダー末尾までの全開催日を ISO 形式 (YYYY-MM-DD) リストで返す。"""
    start = date.today().isoformat()
    return sorted(
        d["date"]
        for d in calendar.get("race_days", [])
        if d["date"] >= start
    )


def _missing_past_race_dates(calendar: dict, since: str | None = None) -> list[str]:
    """
    カレンダーに存在するが race_lists ファイルがない過去の開催日を返す。

    since: 検索開始日 (YYYY-MM-DD)。省略時は今年の1月1日から。
    昨日以前を対象にする（当日は取得中の可能性があるため除外）。
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    start = since or f"{date.today().year}-01-01"
    from src.config.data_paths import RACE_LISTS_DIR

    race_list_dir = RACE_LISTS_DIR

    calendar_past = sorted(
        d["date"]
        for d in calendar.get("race_days", [])
        if start <= d["date"] <= yesterday
    )

    missing: list[str] = []
    for d in calendar_past:
        stem = d.replace("-", "")
        if not (race_list_dir / f"{stem}.json").exists():
            missing.append(d)
    return missing


# ═══════════════════════════════════════════════════════════════════
#  ステータス管理
# ═══════════════════════════════════════════════════════════════════

_STATUS_FILE = Path("data/local/meta/auto_scrape_status.json")


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

    from src.scraper.run import ScraperRunner
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
    """1レース分のプレレースデータをスクレイプ（T-15 バンドル + 予測トリガ）。"""
    from src.scraper.raceday_pre_race_pipeline import (
        run_t15_pre_race_bundle,
        trigger_pre_race_predict,
    )

    rid = race["race_id"]
    label = f"{race['venue']} {race['round']}R {race['race_name'][:12]}"
    now_str = datetime.now().strftime("%H:%M:%S")

    logger.info("  ★ [%s] %s (%s) T-15 バンドル開始", now_str, label, rid)

    bundle = run_t15_pre_race_bundle(runner, race)
    for task_name, ok in bundle.netkeiba_tasks.items():
        if ok:
            logger.info("    ✓ %s", task_name)
        else:
            logger.warning("    - %s (データなし or 失敗)", task_name)
    if bundle.jra_baba_refreshed:
        logger.info("    ✓ JRA 馬場ライブ")
    stats["errors"].extend(bundle.errors)
    stats["scraped"] += 1

    trigger_pre_race_predict(race, bundle_result=bundle)
    time.sleep(random.uniform(1.0, 3.0))


# ═══════════════════════════════════════════════════════════════════
#  タスク: 開催日速報結果ランナー (各R発走15分後に速報結果取得)
# ═══════════════════════════════════════════════════════════════════

def task_raceday_result_runner():
    """
    開催日に常駐するプロセス。各レース発走後 RESULT_OFFSET_MINUTES 分後に速報結果を取得する。

    raceday-runner (T-15 事前取得) とは独立して動作する。
    raceday-evening (夕方一括) は確定オッズ・SmartRC込みで引き続き別途実行される。
    """
    calendar = _load_race_calendar()
    today_str = date.today().isoformat()

    if not _is_race_day(calendar, today_str):
        logger.info("今日 (%s) は開催日ではありません", today_str)
        _save_status("raceday-result-runner", {"status": "skipped", "reason": "non-race-day"})
        return

    venues = _get_race_day_venues(calendar, today_str)
    logger.info("=" * 60)
    logger.info("  速報結果ランナー起動: %s", today_str)
    logger.info("  開催場: %s", ", ".join(venues))
    logger.info("=" * 60)

    from src.scraper.run import ScraperRunner
    runner = ScraperRunner()
    date_fmt = today_str.replace("-", "")

    logger.info("[Phase 1] レーススケジュール取得中...")
    schedule = _fetch_race_schedule(runner, date_fmt)

    if not schedule:
        logger.warning("レーススケジュール取得失敗")
        _save_status("raceday-result-runner", {"status": "error", "reason": "no-schedule"})
        return

    first_post = schedule[0]["post_time"]
    last_post = schedule[-1]["post_time"]
    logger.info("  全 %d レース", len(schedule))
    logger.info("  最初の速報取得予定: %s (%s %sR %s 発走%d分後)",
                (first_post + timedelta(minutes=RESULT_OFFSET_MINUTES)).strftime("%H:%M"),
                schedule[0]["venue"], schedule[0]["round"], schedule[0]["race_name"],
                RESULT_OFFSET_MINUTES)
    logger.info("  最後の速報取得予定: %s (%s %sR %s 発走%d分後)",
                (last_post + timedelta(minutes=RESULT_OFFSET_MINUTES)).strftime("%H:%M"),
                schedule[-1]["venue"], schedule[-1]["round"], schedule[-1]["race_name"],
                RESULT_OFFSET_MINUTES)

    stats = {"races": len(schedule), "scraped": 0, "errors": []}
    scraped_ids: set[str] = set()

    for i, race in enumerate(schedule):
        rid = race["race_id"]
        target_time = race["post_time"] + timedelta(minutes=RESULT_OFFSET_MINUTES)
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

        # 同時刻に複数レースがある場合はまとめて処理
        batch = []
        for r in schedule:
            if r["race_id"] in scraped_ids:
                continue
            if datetime.now() >= r["post_time"] + timedelta(minutes=RESULT_OFFSET_MINUTES):
                batch.append(r)

        for r in batch:
            r_rid = r["race_id"]
            label = f"{r['venue']} {r['round']}R {r['race_name'][:12]}"
            logger.info("  ★ [%s] 速報結果取得: %s (%s)",
                        datetime.now().strftime("%H:%M:%S"), label, r_rid)
            try:
                result = runner.scrape_race_result_on_time(r_rid, skip_existing=False)
                if result:
                    stats["scraped"] += 1
                    logger.info("    ✓ 速報結果保存: %s", r_rid)
                else:
                    logger.warning("    - 速報結果なし: %s", r_rid)
            except Exception as e:
                logger.error("    ✗ 速報結果エラー [%s]: %s", r_rid, e)
                stats["errors"].append(r_rid)
            scraped_ids.add(r_rid)
            time.sleep(random.uniform(1.0, 2.0))

    logger.info("")
    logger.info("=" * 60)
    logger.info("  速報結果ランナー完了: %d/%d レース", stats["scraped"], stats["races"])
    if stats["errors"]:
        logger.warning("  エラー: %d 件", len(stats["errors"]))
    logger.info("=" * 60)

    if stats["scraped"] > 0:
        _trigger_track_speed_for_date(today_str)

    _save_status("raceday-result-runner", {
        "status": "ok", "date": today_str, "venues": venues,
        **{k: v for k, v in stats.items() if k != "errors"},
        "error_count": len(stats["errors"]),
    })


# ═══════════════════════════════════════════════════════════════════
#  タスク: 開催前日夕方 (出馬表・馬柱・追い切り + 追走難度 precompute)
# ═══════════════════════════════════════════════════════════════════

def run_raceday_eve_for_date(race_date_str: str) -> dict:
    """
    翌開催日 (race_date_str, YYYYMMDD) の全レースについて前日夕方データを取得する。

    - race_shutuba: 出馬表（枠順・馬番・騎手など。追走難度の入力）
    - race_shutuba_past: 馬柱・調教（前日 17 時頃 netkeiba 確定）
    - race_oikiri: 追い切り
    - 完了後: 追走難度を pre_race_only で storage キャッシュ

    Returns: stats dict
    """
    date_iso = f"{race_date_str[:4]}-{race_date_str[4:6]}-{race_date_str[6:]}"
    logger.info("=" * 60)
    logger.info("  前日夕方スクレイプ (raceday-eve): 翌開催日 %s", date_iso)
    logger.info("=" * 60)

    from src.scraper.run import ScraperRunner
    runner = ScraperRunner()

    races = runner.scrape_race_list(race_date_str)
    if not races:
        logger.warning("レース一覧が空: %s", race_date_str)
        return {"status": "error", "reason": "no-races", "date": date_iso}

    stats: dict = {
        "races": len(races),
        "shutuba": 0,
        "shutuba_past": 0,
        "oikiri": 0,
        "smartrc": 0,
        "errors": [],
    }

    for i, race in enumerate(races):
        rid = race["race_id"]
        logger.info("[%d/%d] %s %s", i + 1, len(races), rid, race.get("race_name", ""))

        for task_name, task_key, task_fn in [
            ("出馬表", "shutuba", lambda r=rid: runner.scrape_race_card(r, skip_existing=False)),
            ("馬柱・調教", "shutuba_past", lambda r=rid: runner.scrape_shutuba_past(r, skip_existing=False)),
            ("追い切り",  "oikiri",       lambda r=rid: runner.scrape_oikiri(r, skip_existing=False)),
            ("SmartRC指数", "smartrc",    lambda r=rid: runner.scrape_smartrc(r, race_date_str)),
        ]:
            try:
                result = task_fn()
                if result:
                    stats[task_key] += 1
                else:
                    logger.warning("  - %s (%s): データなし", task_name, rid)
            except Exception as e:
                logger.error("  %sエラー [%s]: %s", task_name, rid, e)
                stats["errors"].append(f"{task_key}:{rid}")

        if i < len(races) - 1:
            time.sleep(random.uniform(2.0, 4.0))

    logger.info("=" * 60)
    logger.info(
        "  前日夕方スクレイプ完了: 出馬表=%d, 馬柱=%d, 追切=%d, SmartRC=%d / %d レース",
        stats["shutuba"],
        stats["shutuba_past"],
        stats["oikiri"],
        stats["smartrc"],
        stats["races"],
    )
    logger.info("=" * 60)

    precompute_stats: dict[str, Any] = {"ok": 0, "skip": 0, "fail": 0, "total": 0}
    import os as _os

    if _os.environ.get("KEIBA_EVE_PRECOMPUTE_TRACKING", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        try:
            from src.pipeline.inference.tracking_difficulty_service import (
                precompute_tracking_for_race_ids,
            )

            race_ids = [str(r["race_id"]) for r in races if r.get("race_id")]
            precompute_stats = precompute_tracking_for_race_ids(
                race_ids,
                runner.storage,
                pre_race_only=True,
            )
            logger.info(
                "  追走難度 precompute: ok=%d skip=%d fail=%d / %d",
                precompute_stats["ok"],
                precompute_stats["skip"],
                precompute_stats["fail"],
                precompute_stats["total"],
            )
        except Exception as exc:
            logger.warning("追走難度 precompute バッチ失敗: %s", exc)
            precompute_stats = {"error": str(exc)}

    return {
        "status": "ok",
        "date": date_iso,
        **{k: v for k, v in stats.items() if k != "errors"},
        "error_count": len(stats["errors"]),
        "tracking_precompute": precompute_stats,
    }


def task_raceday_eve():
    """
    翌日が開催日なら、前日 18:00 に出馬表・馬柱・追い切りを取得する (cron 18:00)。

    出馬表は追走難度・位置取り予測の入力。馬柱は前日 17:00 確定分をこの時点で取得する。
    T-15 バンドルでは取得済みタスクは skip_existing=True でスキップされる。
    """
    calendar = _load_race_calendar()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    if not _is_race_day(calendar, tomorrow):
        logger.info("明日 (%s) は開催日ではありません — スキップ", tomorrow)
        _save_status("raceday-eve", {"status": "skipped", "reason": "non-race-day-tomorrow",
                                     "tomorrow": tomorrow})
        return

    tomorrow_fmt = tomorrow.replace("-", "")
    result = run_raceday_eve_for_date(tomorrow_fmt)
    _save_status("raceday-eve", result)


# ═══════════════════════════════════════════════════════════════════
#  馬場速度計算トリガー (raceday-evening 完了後)
# ═══════════════════════════════════════════════════════════════════

def _trigger_track_speed_for_date(date_str: str) -> None:
    """
    race_result_flat.parquet を増分更新し、当日の馬場速度指数を計算する。
    raceday-evening でレース結果がすべて保存された後に呼び出す。

    Steps:
      1. export_category_chunked("race_result", year) で flat parquet を差分追記
      2. TrackSpeedEngine.assign_races() で当日分の perf_index を計算
         → data/analysis/track_speed/races_{year}.parquet にアップサート
    """
    year = date_str[:4]
    date_fmt = date_str.replace("-", "")

    logger.info("[馬場速度] flat parquet 増分更新開始 (%s / race_result)", year)
    try:
        from src.scraper.export_tables import export_category_chunked
        from src.scraper.storage import HybridStorage
        storage = HybridStorage()
        result = export_category_chunked(year, "race_result", storage)
        flat_rows = result.get("flat_rows", -1)
        skipped = result.get("skipped", False)
        if skipped:
            logger.info("[馬場速度] flat parquet: %s", result.get("reason", "スキップ"))
        else:
            logger.info("[馬場速度] flat parquet 更新完了: +%d 行", flat_rows)
    except Exception as e:
        logger.error("[馬場速度] flat parquet 更新失敗: %s", e)
        return

    logger.info("[馬場速度] assign_races 開始 (%s)", date_fmt)
    try:
        from src.research.race.track_speed_engine import TrackSpeedEngine
        base_dir = Path(__file__).resolve().parents[2]
        eng = TrackSpeedEngine(str(base_dir))
        eng.assign_races(
            years=[year],
            date_min=date_fmt,
            date_max=date_fmt,
            progress_cb=lambda msg: logger.info("[馬場速度] %s", msg),
        )
        logger.info("[馬場速度] assign_races 完了: %s", date_fmt)
    except Exception as e:
        logger.error("[馬場速度] assign_races 失敗: %s", e, exc_info=True)


# ═══════════════════════════════════════════════════════════════════
#  タスク: 開催日夕方 (結果・確定オッズ)
# ═══════════════════════════════════════════════════════════════════

def run_raceday_evening_for_date(date_str: str) -> dict:
    """
    指定日の全レース結果・確定オッズ・SmartRCを取得し、馬場速度計算をトリガーする。
    cron タスクとマニュアル実行の両方から呼び出せる共通ロジック。

    Returns: stats dict
    """
    calendar = _load_race_calendar()
    venues = _get_race_day_venues(calendar, date_str)

    logger.info("=" * 60)
    logger.info("  開催日夕方スクレイプ: %s", date_str)
    logger.info("  開催場: %s", ", ".join(venues) if venues else "(カレンダー外 or 未登録)")
    logger.info("=" * 60)

    from src.scraper.run import ScraperRunner
    runner = ScraperRunner()
    date_fmt = date_str.replace("-", "")

    races = runner.scrape_race_list(date_fmt)
    if not races:
        logger.warning("レース一覧が空: %s", date_fmt)
        return {"status": "error", "reason": "no-races", "date": date_str}

    stats: dict = {"races": len(races), "result_on_time": 0, "odds": 0,
                   "pair_odds": 0, "smartrc": 0, "errors": []}

    for i, race in enumerate(races):
        rid = race["race_id"]
        logger.info("[%d/%d] %s %s", i + 1, len(races), rid, race.get("race_name", ""))

        for task_name, task_key, task_fn in [
            # race_result_on_time のみ書き込む（race_result は weekly-update が担当）
            ("速報結果",   "result_on_time", lambda: runner.scrape_race_result_on_time(rid, skip_existing=False)),
            ("確定オッズ", "odds",            lambda: runner.scrape_odds(rid, skip_existing=False)),
            ("確定2連",    "pair_odds",       lambda: runner.scrape_pair_odds(rid, skip_existing=False)),
            ("SmartRC",    "smartrc",         lambda: runner.scrape_smartrc(rid)),
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
    logger.info("  完了: 速報=%d, オッズ=%d, 2連=%d, SmartRC=%d",
                stats["result_on_time"], stats["odds"], stats["pair_odds"], stats["smartrc"])
    logger.info("=" * 60)

    if stats["result_on_time"] > 0:
        _trigger_track_speed_for_date(date_str)

    return {"status": "ok", "date": date_str, "venues": venues,
            **{k: v for k, v in stats.items() if k != "errors"},
            "error_count": len(stats["errors"])}


def run_weekly_update_for_dates(target_dates: list[str]) -> dict:
    """
    指定した開催日リストの結果・指数・馬情報を更新する。
    cron タスクとマニュアル実行の両方から呼び出せる共通ロジック。

    Returns: stats dict
    """
    if not target_dates:
        return {"status": "skipped", "reason": "no-dates"}

    logger.info("=" * 60)
    logger.info("  週次更新: %d 開催日のデータ取得", len(target_dates))
    logger.info("  対象日: %s", ", ".join(target_dates))
    logger.info("=" * 60)

    from src.scraper.run import ScraperRunner
    runner = ScraperRunner()

    total_stats: dict = {"dates": len(target_dates), "races": 0, "results": 0,
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

    return {"status": "ok", "target_dates": target_dates,
            **{k: v for k, v in total_stats.items() if k != "errors"},
            "error_count": len(total_stats["errors"])}


def task_raceday_evening():
    """開催日の全レース終了後に実行 (cron 17:30 JST)。"""
    calendar = _load_race_calendar()
    today_str = date.today().isoformat()

    if not _is_race_day(calendar, today_str):
        logger.info("今日 (%s) は開催日ではありません", today_str)
        _save_status("raceday-evening", {"status": "skipped", "reason": "non-race-day"})
        return

    result = run_raceday_evening_for_date(today_str)
    _save_status("raceday-evening", result)


# ═══════════════════════════════════════════════════════════════════
#  タスク: 金曜週次更新
# ═══════════════════════════════════════════════════════════════════

def task_weekly_update():
    """毎週金曜17:30に実行 (cron)。db.netkeiba が先週分を更新した後に取得。"""
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

    result = run_weekly_update_for_dates(target_dates)
    _save_status("weekly-update", result)


def run_catchup_for_dates(target_dates: list[str]) -> dict:
    """
    指定した過去開催日のレースデータ（一覧・結果・オッズ・SmartRC）を補完取得する。
    cron タスクとマニュアル実行の両方から呼び出せる共通ロジック。

    各日について run_raceday_evening_for_date() を実行する。
    Returns: 集計 stats dict
    """
    if not target_dates:
        return {"status": "skipped", "reason": "no-dates"}

    logger.info("=" * 60)
    logger.info("  過去データ補完: %d 開催日", len(target_dates))
    logger.info("  対象日: %s", ", ".join(target_dates))
    logger.info("=" * 60)

    total: dict = {"dates": len(target_dates), "races": 0, "results": 0,
                   "odds": 0, "pair_odds": 0, "smartrc": 0,
                   "ok_dates": [], "fail_dates": [], "skip_dates": [], "error_count": 0}

    for date_str in target_dates:
        logger.info("--- 補完: %s ---", date_str)
        try:
            result = run_raceday_evening_for_date(date_str)
            if result.get("status") == "ok":
                total["ok_dates"].append(date_str)
                for k in ("races", "results", "odds", "pair_odds", "smartrc"):
                    total[k] += result.get(k, 0)
                total["error_count"] += result.get("error_count", 0)
            elif result.get("reason") == "no-races":
                logger.info("  レース一覧なし: %s (netkeiba未掲載の可能性)", date_str)
                total["skip_dates"].append(date_str)
            else:
                total["fail_dates"].append(date_str)
        except Exception as e:
            logger.error("  補完失敗 [%s]: %s", date_str, e, exc_info=True)
            total["fail_dates"].append(date_str)

        if date_str != target_dates[-1]:
            time.sleep(random.uniform(3.0, 6.0))

    status = "ok" if not total["fail_dates"] else (
        "partial" if total["ok_dates"] else "error"
    )
    return {**total, "status": status,
            "last_run": datetime.now().isoformat(timespec="seconds")}


def task_catchup_missing_dates():
    """
    カレンダー上の過去開催日のうち race_lists が存在しない日のデータを補完する。
    毎日 09:00 cron または手動実行で使う。
    """
    calendar = _load_race_calendar()
    missing = _missing_past_race_dates(calendar)

    if not missing:
        logger.info("過去の欠損開催日なし")
        _save_status("catchup-missing", {"status": "skipped", "reason": "no-missing-dates",
                                          "last_run": datetime.now().isoformat(timespec="seconds")})
        return

    logger.info("過去の欠損開催日: %d 日 (%s〜%s)", len(missing), missing[0], missing[-1])
    result = run_catchup_for_dates(missing)
    _save_status("catchup-missing", result)


def task_daily_race_lists():
    """毎日 07:00 に実行 (cron)。今日〜カレンダー末尾までの全開催日の race_lists を取得・更新する。"""
    calendar = _load_race_calendar()
    target_dates = _future_race_dates_iso(calendar)

    if not target_dates:
        logger.info("今後の開催日なし（カレンダー末尾到達または未取得）")
        _save_status("daily-race-lists", {"status": "skipped", "reason": "no-upcoming-races"})
        return

    logger.info("daily-race-lists: カレンダー末尾まで %d 日分取得開始 %s〜%s",
                len(target_dates), target_dates[0], target_dates[-1])

    from src.scraper.run import ScraperRunner
    runner = ScraperRunner()

    ok_dates: list[str] = []
    fail_dates: list[str] = []
    total_races = 0

    for d in target_dates:
        try:
            date_fmt = d.replace("-", "")  # YYYYMMDD 形式が必要
            races = runner.scrape_race_list(date_fmt)
            ok_dates.append(d)
            total_races += len(races)
            logger.info("race_lists [%s]: %d レース", d, len(races))
        except Exception as e:
            logger.error("race_lists [%s] 失敗: %s", d, e, exc_info=True)
            fail_dates.append(d)

    result: dict = {
        "status": "ok" if not fail_dates else ("partial" if ok_dates else "error"),
        "last_run": datetime.now().isoformat(timespec="seconds"),
        "target_dates": target_dates,
        "ok_dates": ok_dates,
        "fail_dates": fail_dates,
        "total_races": total_races,
    }
    _save_status("daily-race-lists", result)
    logger.info("daily-race-lists 完了: ok=%d fail=%d 合計%dレース",
                len(ok_dates), len(fail_dates), total_races)


# ═══════════════════════════════════════════════════════════════════
#  ステータス表示
# ═══════════════════════════════════════════════════════════════════

def show_status():
    status = _load_status()
    calendar = _load_race_calendar()
    today = date.today()

    print("\n=== 自動スクレイプ ステータス ===")

    task_labels = {
        "raceday-eve":           "前日夕 (出馬表+馬柱・追切 18:00 +追走難度)",
        "raceday-runner":        "開催日ランナー (出馬表・オッズ各R 15分前)",
        "raceday-result-runner": "速報結果ランナー (各R発走15分後に速報取得)",
        "raceday-evening":       "開催日夕 (結果・確定オッズ)",
        "weekly-update":         "金曜週次 (結果・指数・馬情報)",
        "catchup-missing":       "過去欠損補完 (race_listsなし開催日)",
        "daily-race-lists":      "日次レース一覧 (今日〜カレンダー末尾)",
        "jra-baba-morning":      "JRA馬場情報 (朝ポーリング 05:00-09:00)",
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
        eve = (date.fromisoformat(rd) - timedelta(days=1)).isoformat()
        print(f"    {eve} 18:00 前日夕 → 出馬表+馬柱・追切 + 追走難度キャッシュ")
        print(f"    {rd} 07:30 ランナー起動 → オッズ・馬場各R発走15分前（出馬表は前日取得済）")
        print(f"    {rd} 17:30 結果+確定オッズ取得")

    nf = today + timedelta(days=(4 - today.weekday()) % 7)
    if nf == today and datetime.now().hour >= 18:
        nf += timedelta(days=7)
    print(f"  次の金曜: {nf.isoformat()} 17:30 → 週次更新 (db.netkeiba反映後)")


# ═══════════════════════════════════════════════════════════════════
#  タスク: JRA馬場情報 (朝ポーリング)
# ═══════════════════════════════════════════════════════════════════

def task_jra_baba_morning():
    """
    JRA公式馬場ページから馬場情報を取得する (cron 毎10分 05:00-09:00)。

    jra_baba_live.run_cron_job() に委譲する。
    - 当日が開催日でなければ即終了
    - 時刻が前日発表窓 or 当日朝計測窓の外なら即終了
    - 窓内かつ新データあればスクレイプして data/jra_baba/ に保存
    """
    from src.scraper.jra_baba_live import run_cron_job
    today = date.today().isoformat()
    logger.info("JRA馬場情報チェック: %s %s", today, datetime.now().strftime("%H:%M"))
    count = run_cron_job()
    if count > 0:
        logger.info("JRA馬場情報: %d レコード保存", count)
        _save_status("jra-baba-morning", {
            "status": "ok",
            "date": today,
            "records": count,
        })
    else:
        logger.debug("JRA馬場情報: 変更なし or 対象外 (ステータス更新なし)")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

TASKS = {
    "raceday-eve":            task_raceday_eve,
    "raceday-runner":         task_raceday_runner,
    "raceday-result-runner":  task_raceday_result_runner,
    "raceday-evening":        task_raceday_evening,
    "weekly-update":          task_weekly_update,
    "catchup-missing":        task_catchup_missing_dates,
    "daily-race-lists":       task_daily_race_lists,
    "jra-baba-morning":       task_jra_baba_morning,
}


def main():
    script_basic_config()

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
