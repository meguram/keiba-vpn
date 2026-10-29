"""直近の開催日の「公開が遅れて取れなかった／まだ取れていない」データを、未取得のものだけ取り直す（cron: recent-retry）。

背景（2026-10-14 の監査）: backfill は 1 回走ると日付を「完了」扱いにするため、レース当日の深夜に走って db.netkeiba の
確定結果が 0 件だった日（例: 10/11）が、以降ずっと再試行されなかった。weekly-update（金曜）まで確定結果が入らず、
翌日以降の推論に使う履歴が欠ける。厩舎コメント・結果ラップ・タイム指数にも定期経路が無かった。

対象は直近 ``LOOKBACK_DAYS`` 日の開催日（昨日以前）。未取得のレース × カテゴリだけを、1 キーあたり ``MIN_INTERVAL_H`` 時間以上空け、
最大 ``MAX_ATTEMPTS`` 回まで試す（公開前の問い合わせを繰り返さないため）。1 回の起動での取得数は ``MAX_REQUESTS`` まで。
状態: ``data/local/meta/recent_retry_state.json``。
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("scraper.recent_retry")

JST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 21
MIN_INTERVAL_H = 3.0
MAX_ATTEMPTS = 30
MAX_REQUESTS = 400
STATE_PATH = Path("data/local/meta/recent_retry_state.json")

# (カテゴリ, ScraperRunner のメソッド名)。確定結果を最初に取る。
CATEGORIES: list[tuple[str, str]] = [
    ("race_result", "scrape_race_result"),
    ("race_result_lap", "scrape_race_result_lap"),
    ("race_index", "scrape_speed_index"),
    ("race_trainer_comment", "scrape_trainer_comment"),
]


def due_dates(calendar: dict, now: datetime, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    """昨日以前の直近の開催日（YYYYMMDD、古い順）。当日は結果が出ていないため含めない。"""
    today = now.date().isoformat()
    oldest = (now - timedelta(days=lookback_days)).date().isoformat()
    return sorted(d["date"].replace("-", "") for d in calendar.get("race_days", []) if oldest <= d["date"] < today)


def should_try(state: dict[str, Any], key: str, now: datetime, *, min_interval_h: float = MIN_INTERVAL_H, max_attempts: int = MAX_ATTEMPTS) -> bool:
    """このキーを今回取りに行くか（純粋関数）。試行回数の上限と、前回からの間隔で判断する。"""
    s = state.get(key)
    if not s:
        return True
    if int(s.get("attempts", 0)) >= max_attempts:
        return False
    last = s.get("last")
    if not last:
        return True
    return now - datetime.fromisoformat(last) >= timedelta(hours=min_interval_h)


def select_targets(
    race_ids_by_date: dict[str, list[str]],
    exists: Callable[[str, str], bool],
    state: dict[str, Any],
    now: datetime,
    categories: list[tuple[str, str]] = CATEGORIES,
    max_requests: int = MAX_REQUESTS,
) -> list[tuple[str, str, str]]:
    """取りに行く (category, method, race_id) を、カテゴリ優先・日付の古い順に返す（純粋関数）。"""
    out: list[tuple[str, str, str]] = []
    for cat, method in categories:
        for date in sorted(race_ids_by_date):
            for rid in race_ids_by_date[date]:
                if exists(cat, rid) or not should_try(state, f"{cat}/{rid}", now):
                    continue
                out.append((cat, method, rid))
                if len(out) >= max_requests:
                    return out
    return out


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run(calendar: dict, now: datetime | None = None) -> dict[str, Any]:
    from src.scraper.run import ScraperRunner

    now = now or datetime.now(JST)
    dates = due_dates(calendar, now)
    runner = ScraperRunner()
    by_date: dict[str, list[str]] = {}
    for d in dates:
        rl = runner.storage.load("race_lists", d) or {}
        by_date[d] = [r["race_id"] for r in rl.get("races", [])]
    state = load_state()
    targets = select_targets(by_date, runner.storage.exists, state, now)
    saved = {c: 0 for c, _ in CATEGORIES}
    pending = {c: 0 for c, _ in CATEGORIES}
    errors = 0
    date_of = {rid: d for d, rids in by_date.items() for rid in rids}
    result_dates: set[str] = set()
    for i, (cat, method, rid) in enumerate(targets):
        key = f"{cat}/{rid}"
        s = state.setdefault(key, {"attempts": 0})
        s["attempts"] = int(s.get("attempts", 0)) + 1
        s["last"] = datetime.now(JST).isoformat(timespec="seconds")
        try:
            ok = bool(getattr(runner, method)(rid, skip_existing=False))
        except Exception as exc:
            logger.error("recent-retry 失敗 [%s]: %s", key, exc)
            errors += 1
            ok = False
        saved[cat] += int(ok)
        pending[cat] += int(not ok)
        if ok and cat == "race_result" and rid in date_of:
            result_dates.add(date_of[rid])
        if ok:
            state.pop(key, None)
        if i % 20 == 19:
            save_state(state)
        time.sleep(random.uniform(2.0, 5.0))
    save_state(state)
    if (os.environ.get("KEIBA_RUN_LOCATION") or "").strip().lower() == "vps":
        # VPS は取得だけ（250MB 枠・flat と馬場速度の正本は GCS）。馬場速度は GCP の export-tables-daily、予測成績は prediction-results-refresh が行う
        return {"dates": dates, "targets": len(targets), "saved": saved, "pending": pending, "error_count": errors,
                "track_speed_dates": [], "post_processing": "gcp"}
    # 馬場速度は確定結果（race_result）から計算する。当日夜は未取得なので、取れた日付の分をここで計算する
    for d in sorted(result_dates):
        try:
            from src.scraper.auto_scrape import _trigger_track_speed_for_date

            _trigger_track_speed_for_date(d)
        except Exception as exc:
            logger.error("recent-retry 馬場速度の計算に失敗 [%s]: %s", d, exc)
            errors += 1
    # 確定結果が取れた日の予測結果を確定として集計し直す（KEIBA_PREDICTION_RESULTS=1 のときだけ。失敗しても本体は成功）
    if result_dates:
        from src.pipeline.inference.prediction_results import refresh_for_dates

        refresh_for_dates(sorted(result_dates), reason="recent-retry")
    return {"dates": dates, "targets": len(targets), "saved": saved, "pending": pending, "error_count": errors,
            "track_speed_dates": sorted(result_dates)}
