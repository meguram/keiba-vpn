"""
auto_scrape の **netkeiba 系スクレイプ** をすべて ``ScrapeJobQueue`` 経由に寄せる。

* 環境変数 ``KEIBA_AUTO_SCRAPE_USE_QUEUE`` … ``0`` / ``false`` / ``no`` 以外はキュー経由（既定: 有効）
* JRA 公式 ``jra_baba_live``・馬名インデックス・成長曲線パイプラインは従来どおり直実行（キュー対象外）

``kick_process_queue_background`` + ``wait_until_queue_idle`` でバッチ完了を待つ。
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

logger = logging.getLogger("scraper.auto_scrape_queue")

_JST = ZoneInfo("Asia/Tokyo")

LEAD_MINUTES = 15
T60_LEAD_MINUTES = 60
RESULT_OFFSET_MINUTES = 15

# T-15 バンドル相当（smart_skip=False で直 pending。直前オッズ再取得を優先）
# オッズは phase=t15 として保存され、T-60 のスナップショットは snapshots に履歴化される。
T15_RACE_TASKS: list[str] = [
    "race_shutuba",
    "race_odds_t15",
    "race_pair_odds_t15",
    "race_trifecta_odds_t15",
    "race_shutuba_past",
    "race_oikiri",
    "smartrc",
]

# T-60 はオッズ（単複・馬連・ワイド・馬単・3連複）のみ。馬場ライブ・予測トリガは呼ばない。
ODDS_T60_TASKS: list[str] = ["race_odds_t60", "race_pair_odds_t60", "race_trifecta_odds_t60"]

# 発走直前の単複オッズ（race_odds のみ・1 レース 1 リクエスト）。KEIBA_ODDS_LATE_SNAPSHOTS=1 のときだけ取る。
# 直前のオッズ変動（最後の数分の資金流入）の検証用履歴を貯める。phase は t10 / t05 / t02（snapshots に積まれる）。
LATE_LEADS: tuple[int, ...] = (10, 5, 2)
LATE_KIND_TASK: dict[str, str] = {"t10": "race_odds_t10", "t05": "race_odds_t05", "t02": "race_odds_t02"}
ODDS_LATE_TASKS: list[str] = list(LATE_KIND_TASK.values())
# 予定時刻からこの秒数を超えて遅れた直前取得は捨てる（t05 の名で 4 分前のオッズを積まない）。
LATE_TOLERANCE_SECONDS = 120
# 重い取得の前に、この秒数以内に直前オッズが来るなら、その時刻まで待ってから取る（バンドルの最中に過ぎるのを防ぐ）。
LATE_LOOKAHEAD_SECONDS = 90

# 馬体重の検知（raceday-weight-runner）: 各R T-65〜T-30 に出馬表を取り直し、馬体重が入ったら特徴量・予測を起動する。
from src.scraper.raceday_weight_watch import WEIGHT_WATCH_TASKS  # noqa: E402,F401

# 前日夕方（raceday-eve）: 出馬表・馬柱・追い切り・SmartRC
EVE_TASKS: list[str] = ["race_shutuba", "race_shutuba_past", "race_oikiri", "smartrc"]

# 開催日夕方（raceday-evening）: 速報結果・確定オッズ（履歴を壊さないよう phase=final 固定）・SmartRC
EVENING_TASKS: list[str] = ["race_result_on_time", "race_odds_final", "race_pair_odds_final", "race_trifecta_odds_final", "smartrc"]


def enabled_tasks(tasks: list[str]) -> list[str]:
    """3連単オッズ（1 レース約 100KB・1 リクエスト）のタスクは KEIBA_ODDS_TRIFECTA=1 のときだけ実行する。"""
    if os.environ.get("KEIBA_ODDS_TRIFECTA", "").strip() in ("1", "true", "yes"):
        return list(tasks)
    return [t for t in tasks if not t.startswith("race_trifecta_odds")]


def late_snapshots_enabled() -> bool:
    """発走直前（T-10 / T-5 / T-2）の単複オッズを取るか。既定は off（``KEIBA_ODDS_LATE_SNAPSHOTS=1`` で on）。"""
    return os.environ.get("KEIBA_ODDS_LATE_SNAPSHOTS", "").strip().lower() in ("1", "true", "yes")


def _kind_order(kind: str) -> int:
    """同時刻のイベントの順序: t60 → 直前（時刻が重要）→ t15（重いバンドル）。"""
    return 0 if kind == "t60" else (2 if kind == "t15" else 1)


def build_odds_events(
    schedule: list[dict],
    lead_t60: int = T60_LEAD_MINUTES,
    lead_t15: int = LEAD_MINUTES,
    late_leads: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    """各レースの T-60 / T-15（と、指定があれば直前の T-10 / T-5 / T-2）イベントを時刻順に並べて返す（純粋関数）。

    要素: ``{"kind": "t60"|"t15"|"t10"|"t05"|"t02", "at": 実行時刻, "t15_at": そのレースの T-15 時刻, "race": race}``。
    同時刻なら t60 → 直前 → t15 の順、それも同じなら schedule の並び順。
    """
    events: list[dict[str, Any]] = []
    for idx, race in enumerate(schedule):
        post = race["post_time"]
        t15_at = post - timedelta(minutes=lead_t15)
        events.append({"kind": "t60", "at": post - timedelta(minutes=lead_t60),
                       "t15_at": t15_at, "race": race, "_i": idx})
        events.append({"kind": "t15", "at": t15_at, "t15_at": t15_at, "race": race, "_i": idx})
        for lead in late_leads:
            events.append({"kind": f"t{lead:02d}", "at": post - timedelta(minutes=lead),
                           "t15_at": t15_at, "race": race, "_i": idx})
    events.sort(key=lambda e: (e["at"], _kind_order(e["kind"]), e["_i"]))
    for e in events:
        del e["_i"]
    return events


def select_due_events(
    events: list[dict[str, Any]],
    done: set[tuple[str, str]],
    now: datetime,
) -> tuple[list[dict], list[dict], list[dict]]:
    """``now`` までに到達した未処理イベントを (t60, t15, skipped_t60) に振り分ける（純粋関数）。

    既に T-15 時刻を過ぎたレースの T-60 は取得する意味が薄い（T-15 が同じデータを取る）ため skipped。
    返した全イベントは呼び出し側が ``done`` に追加する想定。
    """
    t60: list[dict] = []
    t15: list[dict] = []
    skipped: list[dict] = []
    for e in events:
        if e["kind"] not in ("t60", "t15"):
            continue                       # 直前イベントは select_due_late_events が扱う
        key = (e["race"]["race_id"], e["kind"])
        if key in done or e["at"] > now:
            continue
        if e["kind"] == "t15":
            t15.append(e)
        elif e["t15_at"] <= now:
            skipped.append(e)
        else:
            t60.append(e)
    return t60, t15, skipped


def select_due_late_events(
    events: list[dict[str, Any]],
    done: set[tuple[str, str]],
    now: datetime,
    tolerance_sec: float = LATE_TOLERANCE_SECONDS,
) -> tuple[list[dict], list[dict]]:
    """``now`` までに到達した未処理の直前イベントを (実行する, 捨てる) に振り分ける（純粋関数）。

    予定時刻から ``tolerance_sec`` を超えて遅れたもの、発走時刻を過ぎたものは捨てる（遅れて取ったオッズを
    t05 / t02 の名で積むと、直前変動の検証がずれる）。返した全イベントは呼び出し側が ``done`` に追加する想定。
    """
    due: list[dict] = []
    stale: list[dict] = []
    for e in events:
        if e["kind"] not in LATE_KIND_TASK:
            continue
        key = (e["race"]["race_id"], e["kind"])
        if key in done or e["at"] > now:
            continue
        late_sec = (now - e["at"]).total_seconds()
        if late_sec > tolerance_sec or now >= e["race"]["post_time"]:
            stale.append(e)
        else:
            due.append(e)
    return due, stale


def use_queue() -> bool:
    v = os.environ.get("KEIBA_AUTO_SCRAPE_USE_QUEUE", "1").strip().lower()
    return v not in ("0", "false", "no")


def _storage() -> Any:
    from src.scraper.run import REPO_ROOT
    from src.scraper.storage import HybridStorage

    return HybridStorage(str(REPO_ROOT))


def _queue() -> Any:
    from src.scraper.job_queue import ScrapeJobQueue

    return ScrapeJobQueue()


def _kick_and_wait(
    queue: Any, *, timeout_sec: float = 72000.0, poll_sec: float = 3.0
) -> tuple[bool, str, dict[str, Any]]:
    from src.scraper.job_queue import kick_process_queue_background
    from src.scraper.verify_horse_scrape_completeness import wait_until_queue_idle

    kick_process_queue_background()
    time.sleep(0.65)
    ok, msg, last = wait_until_queue_idle(
        queue, timeout_sec=timeout_sec, poll_sec=poll_sec
    )
    if not ok:
        logger.warning("キュー待機: %s last=%s", msg, last)
    return ok, msg, last


def _fetch_race_schedule_storage(storage: Any, date_fmt: str) -> list[dict]:
    """発走時刻表。``race_day_schedule`` があれば優先、無ければ race_lists + race_shutuba で合成。"""
    try:
        snap = storage.load("race_day_schedule", date_fmt)
    except Exception:
        snap = None
    if isinstance(snap, dict) and snap.get("slots"):
        from src.scraper.race_day_schedule import schedule_payload_to_runtime_list

        return schedule_payload_to_runtime_list(snap)

    rl = storage.load("race_lists", date_fmt)
    if not rl:
        return []
    races = rl.get("races") or []
    try:
        ys, ms, ds = int(date_fmt[:4]), int(date_fmt[4:6]), int(date_fmt[6:8])
        race_day = date(ys, ms, ds)
    except (ValueError, IndexError):
        race_day = datetime.now(_JST).date()

    schedule: list[dict[str, Any]] = []
    for race in races:
        rid = race.get("race_id")
        if not rid or not isinstance(rid, str):
            continue
        rid = rid.strip()
        if not rid:
            continue
        card = storage.load("race_shutuba", rid) or {}
        start_str = str(card.get("start_time") or "").strip()

        if not start_str:
            rnd = race.get("round", 0)
            if isinstance(rnd, str):
                rnd = int(rnd) if rnd.isdigit() else 0
            elif not isinstance(rnd, int):
                rnd = 0
            if rnd <= 6:
                h = 9 + (rnd * 30) // 60
                m = 45 + (rnd * 30) % 60
            else:
                h = 12 + ((rnd - 5) * 30) // 60
                m = ((rnd - 5) * 30) % 60
            start_str = f"{h:02d}:{m:02d}"

        try:
            h, m = map(int, start_str.split(":"))
            post_dt = datetime.combine(
                race_day,
                datetime.min.time().replace(hour=h, minute=m),
                tzinfo=_JST,
            )
        except (ValueError, TypeError):
            continue

        schedule.append(
            {
                "race_id": rid,
                "venue": race.get("venue", card.get("venue", "") if card else ""),
                "round": race.get("round", ""),
                "race_name": race.get("race_name", card.get("race_name", "") if card else ""),
                "post_time": post_dt,
                "start_time_str": start_str,
            }
        )

    schedule.sort(key=lambda x: x["post_time"])
    return schedule


def _ensure_race_list_date(storage: Any, queue: Any, ymd: str) -> bool:
    # まずローカルファイル（data/page_reference/race_lists/）を確認（高速・GCS不要）
    from pathlib import Path

    local_path = Path("data/page_reference/race_lists") / f"{ymd}.json"
    if not local_path.exists():
        # プロジェクトルート基準で再試行（CWD が異なる環境向け）
        import os
        alt = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "data/page_reference/race_lists" / f"{ymd}.json"
        local_path = alt
    if local_path.exists():
        import json as _json
        try:
            _data = _json.loads(local_path.read_text(encoding="utf-8"))
            if _data.get("races"):
                return True
        except Exception:
            pass

    # ローカルになければ GCS を確認
    rl = storage.load("race_lists", ymd)
    if rl and rl.get("races"):
        return True

    # GCS にもなければ取得キューに投入してキューが空くのを待つ（タイムアウト短め）
    from src.scraper.period_runners import enqueue_date_tasks_for_race_period

    enqueue_date_tasks_for_race_period(
        storage,
        queue,
        start_date=ymd,
        end_date=ymd,
        tasks=["race_list"],
        limit=10,
        dry_run=False,
        smart_skip=True,
    )
    # キューが全部ドレインするのを待つのではなく、race_list 取得を個別にポーリング
    import json as _json2
    import time as _time
    t0 = _time.time()
    while _time.time() - t0 < 300.0:  # 最大 5 分待機
        _time.sleep(5.0)
        # ローカルファイルが生成されたら OK
        if local_path.exists():
            try:
                _d = _json2.loads(local_path.read_text(encoding="utf-8"))
                if _d.get("races"):
                    return True
            except Exception:
                pass
        rl2 = storage.load("race_lists", ymd)
        if rl2 and rl2.get("races"):
            return True
    return False


def _eve_precomputes(race_date_str: str, race_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    precompute_stats: dict[str, Any] = {"ok": 0, "skip": 0, "fail": 0, "total": 0}
    final_odds_precompute_stats: dict[str, Any] = {"ok": 0, "skip": 0, "fail": 0, "total": 0}
    if not race_ids:
        return precompute_stats, final_odds_precompute_stats

    if os.environ.get("KEIBA_EVE_PRECOMPUTE_TRACKING", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        try:
            from src.pipeline.inference.tracking_difficulty_service import (
                precompute_tracking_for_race_ids,
            )

            st = _storage()
            precompute_stats = precompute_tracking_for_race_ids(
                race_ids, st, pre_race_only=True
            )
            logger.info(
                "  追走難度 precompute: ok=%d skip=%d fail=%d / %d",
                precompute_stats.get("ok", 0),
                precompute_stats.get("skip", 0),
                precompute_stats.get("fail", 0),
                precompute_stats.get("total", 0),
            )
        except Exception as exc:
            logger.warning("追走難度 precompute バッチ失敗: %s", exc)
            precompute_stats = {"error": str(exc)}

    if os.environ.get("KEIBA_EVE_PRECOMPUTE_FINAL_ODDS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        try:
            from src.pipeline.inference.final_odds_service import (
                precompute_final_odds_for_race_ids,
            )

            st = _storage()
            final_odds_precompute_stats = precompute_final_odds_for_race_ids(race_ids, st)
            logger.info(
                "  想定オッズ precompute: ok=%d skip=%d fail=%d / %d",
                final_odds_precompute_stats.get("ok", 0),
                final_odds_precompute_stats.get("skip", 0),
                final_odds_precompute_stats.get("fail", 0),
                final_odds_precompute_stats.get("total", 0),
            )
        except Exception as exc:
            logger.warning("想定オッズ precompute バッチ失敗: %s", exc)
            final_odds_precompute_stats = {"error": str(exc)}

    return precompute_stats, final_odds_precompute_stats


def _eve_horse_training(storage: Any, race_ids: list[str]) -> dict[str, Any]:
    """前日夕方に、出走馬の調教ページ（最新 1 ページ）を取り直す。KEIBA_EVE_HORSE_TRAINING=1 のときだけ（既定は無効）。"""
    if os.environ.get("KEIBA_EVE_HORSE_TRAINING", "0").strip().lower() not in ("1", "true", "yes"):
        return {"enabled": False}
    try:
        from src.scraper.horse_training_recent import horse_ids_for_races, refresh_for_horses
        from src.scraper.run import ScraperRunner

        hids = horse_ids_for_races(storage, race_ids)
        logger.info("  調教ページの取り直し: 出走馬 %d 頭", len(hids))
        st = refresh_for_horses(ScraperRunner(), hids)
        logger.info("  調教ページの取り直し結果: %s", st)
        return {"enabled": True, **st}
    except Exception as exc:  # noqa: BLE001
        logger.warning("調教ページの取り直しに失敗: %s", exc)
        return {"enabled": True, "error": str(exc)}


def _eve_horse_profile(storage: Any, race_ids: list[str]) -> dict[str, Any]:
    """前日夕方に、出走馬のうち horse_result が無い馬（新馬・初出走など）の基本情報を取得して馬主を揃える。
    KEIBA_EVE_HORSE_PROFILE=1 のときだけ（既定は無効）。既存の horse_result は上書きしない。"""
    from src.scraper import horse_profile_eve as hp

    if not hp.enabled():
        return {"enabled": False}
    try:
        st = hp.fill_for_races(storage, race_ids)
        logger.info("  馬の基本情報（馬主）の取得結果: %s", st)
        return {"enabled": True, **st}
    except Exception as exc:  # noqa: BLE001
        logger.warning("馬の基本情報の取得に失敗: %s", exc)
        return {"enabled": True, "error": str(exc)}


def run_raceday_eve_for_date(race_date_str: str) -> dict[str, Any]:
    date_iso = f"{race_date_str[:4]}-{race_date_str[4:6]}-{race_date_str[6:]}"
    logger.info("=" * 60)
    logger.info("  前日夕方 (キュー経由 raceday-eve): 翌開催日 %s", date_iso)
    logger.info("=" * 60)

    storage = _storage()
    queue = _queue()
    if not _ensure_race_list_date(storage, queue, race_date_str):
        return {"status": "error", "reason": "no-races", "date": date_iso}

    from src.scraper.period_runners import collect_jra_race_job_specs_for_period

    specs, meta = collect_jra_race_job_specs_for_period(
        storage,
        start_date=race_date_str,
        end_date=race_date_str,
        jra_only=True,
        limit=None,
    )
    if not specs:
        return {"status": "error", "reason": "no-races", "date": date_iso}

    tasks = list(EVE_TASKS)
    full = [
        {
            **sp,
            "tasks": tasks,
            "smart_skip": False,
            "date": race_date_str,
        }
        for sp in specs
    ]
    st_add = queue.bulk_add_jobs(full)
    logger.info("raceday-eve キュー投入: %s races=%d meta=%s", st_add, len(specs), meta)
    ok, msg, _ = _kick_and_wait(queue, timeout_sec=72000.0)
    if not ok:
        return {
            "status": "error",
            "reason": f"queue-wait:{msg}",
            "date": date_iso,
            "bulk_add": st_add,
        }

    race_ids = [str(s["target_id"]) for s in specs if s.get("target_id")]
    training_stats = _eve_horse_training(storage, race_ids)
    profile_stats = _eve_horse_profile(storage, race_ids)
    pc1, pc2 = _eve_precomputes(race_date_str, race_ids)

    return {
        "status": "ok",
        "date": date_iso,
        "races": len(specs),
        "shutuba": len(specs),
        "shutuba_past": len(specs),
        "oikiri": len(specs),
        "smartrc": len(specs),
        "error_count": 0,
        "tracking_precompute": pc1,
        "final_odds_precompute": pc2,
        "horse_training": training_stats,
        "horse_profile": profile_stats,
        "queue_bulk_add": st_add,
    }


def run_raceday_evening_for_date(date_str: str) -> dict[str, Any]:
    from src.scraper import auto_scrape as _as

    calendar = _as._load_race_calendar()
    date_fmt = date_str.replace("-", "")[:8]
    iso_day = f"{date_fmt[:4]}-{date_fmt[4:6]}-{date_fmt[6:8]}"
    venues = _as._get_race_day_venues(calendar, iso_day)

    logger.info("=" * 60)
    logger.info("  開催日夕方 (キュー経由): %s", iso_day)
    logger.info("  開催場: %s", ", ".join(venues) if venues else "(カレンダー外 or 未登録)")
    logger.info("=" * 60)

    storage = _storage()
    queue = _queue()
    if not _ensure_race_list_date(storage, queue, date_fmt):
        return {"status": "error", "reason": "no-races", "date": iso_day}

    from src.scraper.period_runners import collect_jra_race_job_specs_for_period

    specs, _ = collect_jra_race_job_specs_for_period(
        storage,
        start_date=date_fmt,
        end_date=date_fmt,
        jra_only=True,
        limit=None,
    )
    if not specs:
        return {"status": "error", "reason": "no-races", "date": iso_day}

    # 確定オッズは phase=final で履歴化（T-60/T-15 のスナップショットを上書きで失わない）
    tasks = enabled_tasks(EVENING_TASKS)
    full = [
        {**sp, "tasks": tasks, "smart_skip": False, "date": date_fmt} for sp in specs
    ]
    st_add = queue.bulk_add_jobs(full)
    ok, msg, _ = _kick_and_wait(queue, timeout_sec=72000.0)
    n_ok = len(specs) if ok else 0
    stats: dict[str, Any] = {
        "races": len(specs),
        "result_on_time": n_ok,
        "results": n_ok,
        "odds": n_ok,
        "pair_odds": n_ok,
        "smartrc": n_ok,
        "errors": [],
        "queue_bulk_add": st_add,
    }
    if not ok:
        stats["status"] = "error"
        stats["reason"] = f"queue-wait:{msg}"
        stats["error_count"] = 1
        return stats

    stats["status"] = "ok"
    stats["date"] = iso_day
    stats["venues"] = venues
    stats["error_count"] = 0
    if stats["result_on_time"] > 0:
        _as._trigger_track_speed_for_date(iso_day)
    return stats


def _weekly_collect_horse_ids_from_specs(
    storage: Any, all_specs: list[dict[str, Any]],
) -> set[str]:
    """
    レース週次ジョブ完了直後に出走馬 ID を集める。

    ワーカが別スレッドで ``race_result`` を保存したあと、同一プロセス内の
    ``HybridStorage.load`` メモリキャッシュが古いままだと entries が空になり、
    馬ジョブが一頭も投入されず「ワーカーが止まった」ように見えることがある。
    そのため ``invalidate_load_cache`` 後に再読込する。

    ``race_result`` にまだ entries が無い場合は ``race_shutuba`` にフォールバックする
    （period_runners.collect_horse_ids_for_race_period と同趣旨）。
    """
    try:
        storage.invalidate_load_cache("race_result", "")
        storage.invalidate_load_cache("race_shutuba", "")
    except Exception as e:
        logger.debug("週次: race_result/shutuba キャッシュ無効化をスキップ: %s", e)

    out: set[str] = set()
    for sp in all_specs:
        rid = str(sp.get("target_id") or "").strip()
        if not rid:
            continue
        rr = storage.load("race_result", rid) or {}
        entries = list(rr.get("entries") or [])
        if not entries:
            card = storage.load("race_shutuba", rid) or {}
            entries = list(card.get("entries") or [])
        for e in entries:
            if not isinstance(e, dict):
                continue
            hid = (e.get("horse_id") or "").strip()
            if hid:
                out.add(hid)
    return out


def run_weekly_update_for_dates(target_dates: list[str]) -> dict[str, Any]:
    if not target_dates:
        return {"status": "skipped", "reason": "no-dates"}

    from src.scraper import auto_scrape as _as

    logger.info("=" * 60)
    logger.info("  週次更新 (キュー経由): %d 開催日", len(target_dates))
    logger.info("  対象日: %s", ", ".join(target_dates))
    logger.info("=" * 60)

    storage = _storage()
    queue = _queue()
    all_specs: list[dict[str, Any]] = []
    for ds in target_dates:
        ymd = ds.replace("-", "")
        if not _ensure_race_list_date(storage, queue, ymd):
            logger.warning("race_lists なしスキップ: %s", ymd)
            continue
        from src.scraper.period_runners import collect_jra_race_job_specs_for_period

        specs, _ = collect_jra_race_job_specs_for_period(
            storage,
            start_date=ymd,
            end_date=ymd,
            jra_only=True,
            limit=None,
        )
        for sp in specs:
            all_specs.append({**sp, "date": ymd})

    if not all_specs:
        return {"status": "skipped", "reason": "no-races-in-period"}

    # レース系: 毎週 db 反映後の確定値を取り直す（smart_skip=False → skip_existing=False）
    tasks = ["race_result", "race_index", "race_barometer"]
    full = [{**sp, "tasks": tasks, "smart_skip": False} for sp in all_specs]
    st_add = queue.bulk_add_jobs(full)
    ok, msg, _ = _kick_and_wait(queue, timeout_sec=172800.0)
    if not ok:
        return {
            "status": "error",
            "reason": f"queue-wait:{msg}",
            "dates": len(target_dates),
            "queue_bulk_add": st_add,
        }

    all_horse_ids = _weekly_collect_horse_ids_from_specs(storage, all_specs)
    logger.info(
        "週次: レースジョブ完了 → 馬ID収集 %d 頭（レース %d）",
        len(all_horse_ids),
        len(all_specs),
    )
    if not all_horse_ids:
        logger.warning(
            "週次: 馬IDが0件のため馬・血統ジョブをスキップします（race_result / race_shutuba の entries を確認）",
        )

    # 馬プロフィール・成績 (horse_result): 週次は上書き再取得。
    # 血統 HTML（horse_ped）と 5gen JSON は ``horse_pedigree_5gen`` ジョブで取得（不変ならスキップ）。
    horse_specs = [
        {
            "job_kind": "horse",
            "target_id": hid,
            "tasks": ["horse_profile"],
            "smart_skip": False,
            "skip_pedigree": True,
        }
        for hid in sorted(all_horse_ids)
    ]
    st_h = queue.bulk_add_jobs(horse_specs) if horse_specs else {}
    if horse_specs:
        ok2, msg2, _ = _kick_and_wait(queue, timeout_sec=172800.0)
        if not ok2:
            logger.warning("週次 馬プロフィール待機: %s", msg2)

    # 血統: horse_ped（HTML）+ horse_pedigree_5gen（JSON）。ワーカで smart_skip。
    # 索引上 5 世代揃いならキューに載せない。duplicate で既存 pending に合流した場合も完走待ちする。
    st_ped: dict[str, int] = {}
    if all_horse_ids:
        ped_index: dict[str, dict] = {}
        try:
            from pathlib import Path

            from src.research.pedigree.pedigree_local_store import load_full_pedigree_index

            ped_dir = Path(storage._base_dir) / "data" / "local" / "horse_pedigree_5gen"
            if ped_dir.is_dir():
                ped_index = load_full_pedigree_index(storage, path=ped_dir)
        except Exception as e:
            logger.warning("週次: 血統索引の読込をスキップ（全頭キュー投入）: %s", e)
        st_ped = queue.add_horse_jobs_bulk(
            sorted(all_horse_ids),
            ["horse_pedigree_5gen"],
            smart_skip=True,
            skip_pedigree_5gen_if_complete=True,
            pedigree_index=ped_index or None,
        )
        ped_work = (
            int(st_ped.get("created", 0) or 0)
            + int(st_ped.get("requeued", 0) or 0)
            + int(st_ped.get("duplicate", 0) or 0)
        )
        if ped_work > 0:
            ok3, msg3, _ = _kick_and_wait(queue, timeout_sec=172800.0)
            if not ok3:
                logger.warning("週次 血統(horse_ped+5gen) 待機: %s", msg3)
        else:
            logger.info(
                "週次: 血統ジョブは索引上すべて完備のためキュー追加なし（horse_ped / 5gen はワーカでスキップ）",
            )

    total_stats: dict[str, Any] = {
        "status": "ok",
        "target_dates": target_dates,
        "dates": len(target_dates),
        "races": len(all_specs),
        "results": len(all_specs),
        "index": len(all_specs),
        "barometer": len(all_specs),
        "horses_updated": len(all_horse_ids),
        "horse_ids_collected": len(all_horse_ids),
        "errors": [],
        "error_count": 0,
        "queue_race_bulk_add": st_add,
        "queue_horse_bulk_add": st_h,
        "queue_horse_pedigree_5gen_bulk": st_ped,
    }

    if all_horse_ids and _as._today_jst().weekday() == 4:
        try:
            from src.utils.horse_name_index import run_weekly_horse_name_index_update

            idx = run_weekly_horse_name_index_update(Path(__file__).resolve().parents[2])
            total_stats["horse_name_index"] = idx
            _as._save_status("horse-name-index", idx)
        except Exception as e:
            logger.warning("週次更新後の馬名インデックス再構築をスキップ: %s", e)

    return _as.finalize_weekly_update(target_dates, total_stats)


def run_catchup_for_dates(target_dates: list[str]) -> dict[str, Any]:
    if not target_dates:
        return {"status": "skipped", "reason": "no-dates"}

    total: dict[str, Any] = {
        "dates": len(target_dates),
        "races": 0,
        "results": 0,
        "odds": 0,
        "pair_odds": 0,
        "smartrc": 0,
        "ok_dates": [],
        "fail_dates": [],
        "skip_dates": [],
        "error_count": 0,
    }

    for date_str in target_dates:
        logger.info("--- 補完 (キュー経由): %s ---", date_str)
        try:
            result = run_raceday_evening_for_date(date_str)
            if result.get("status") == "ok":
                total["ok_dates"].append(date_str)
                for k in ("races", "results", "odds", "pair_odds", "smartrc"):
                    total[k] = int(total.get(k, 0)) + int(result.get(k, 0) or 0)
                total["error_count"] += int(result.get("error_count", 0) or 0)
            elif result.get("reason") == "no-races":
                total["skip_dates"].append(date_str)
            else:
                total["fail_dates"].append(date_str)
        except Exception as e:
            logger.error("  補完失敗 [%s]: %s", date_str, e, exc_info=True)
            total["fail_dates"].append(date_str)

        if date_str != target_dates[-1]:
            time.sleep(random.uniform(3.0, 6.0))

    status = (
        "ok"
        if not total["fail_dates"]
        else ("partial" if total["ok_dates"] else "error")
    )
    from src.scraper import auto_scrape as _as

    return {**total, "status": status, "last_run": _as._now_jst_iso()}


def task_daily_race_lists() -> None:
    from src.scraper import auto_scrape as _as

    calendar = _as._load_race_calendar()
    target_dates = _as.race_list_poll_dates(_as._future_race_dates_iso(calendar), _as._today_jst())
    if not target_dates:
        logger.info("今回取りに行くレース一覧なし（先の日付は週 1 回の確認日のみ）")
        _as._save_status(
            "daily-race-lists",
            {"status": "skipped", "reason": "no-dates-to-poll"},
        )
        return

    storage = _storage()
    queue = _queue()
    extras: dict[str, Any] = {
        "job_kind": "date",
        "tasks": ["race_list"],
        "smart_skip": True,
    }
    full = [
        dict(extras, target_id=d.replace("-", "")[:8]) for d in target_dates
    ]
    st_add = queue.bulk_add_jobs(full)
    logger.info(
        "daily-race-lists キュー投入: %d 日 bulk=%s", len(full), st_add
    )
    ok, msg, _ = _kick_and_wait(queue, timeout_sec=172800.0)
    ok_dates: list[str] = []
    fail_dates: list[str] = []
    total_races = 0
    for d in target_dates:
        ymd = d.replace("-", "")[:8]
        rl = storage.load("race_lists", ymd)
        from src.scraper.race_list_completeness import is_race_list_complete

        if is_race_list_complete(rl):
            ok_dates.append(d)
            total_races += len((rl or {}).get("races") or [])
        else:
            fail_dates.append(d)

    result: dict[str, Any] = {
        "status": "ok" if not fail_dates else ("partial" if ok_dates else "error"),
        "last_run": _as._now_jst_iso(),
        "target_dates": target_dates,
        "ok_dates": ok_dates,
        "fail_dates": fail_dates,
        "total_races": total_races,
        "queue_wait_ok": ok,
        "queue_wait_msg": msg,
        "queue_bulk_add": st_add,
    }
    _as._save_status("daily-race-lists", result)


def _raceday_direct_enabled() -> bool:
    """開催日ランナーの T-60/T-15 を、グローバルキューを待たずに直接実行するか。既定は従来どおりキュー経由（``KEIBA_RACEDAY_DIRECT=1`` で有効）。"""
    return os.environ.get("KEIBA_RACEDAY_DIRECT", "0").strip().lower() in ("1", "true", "yes")


def _run_race_tasks_direct(runner: Any, race_id: str, date_fmt: str, tasks: list[str]) -> list[str]:
    """1 レースのタスクを順に直接実行する。失敗は握りつぶさず文字列で返す（アクセス制限の疑いも含む）。
    キュー経由だと、他ジョブの滞留・固まったジョブ・アクセス一時停止で T-60/T-15 の時刻に間に合わなくなるため。"""
    from src.scraper.access_guard import AccessRestrictionDetected
    from src.scraper.queue_tasks import _race_task

    errors: list[str] = []
    for t in tasks:
        try:
            _race_task(runner, race_id, t, date_fmt, smart_skip=False)
        except AccessRestrictionDetected as e:
            errors.append(f"{race_id}:{t}:access_restriction:{e}")
            logger.error("アクセス制限の疑い（%s %s）: %s", race_id, t, e)
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{race_id}:{t}:{type(e).__name__}:{e}")
            logger.error("直接取得に失敗 %s %s: %s", race_id, t, e)
        time.sleep(random.uniform(0.5, 1.5))
    return errors


def task_raceday_runner() -> None:
    """開催日常駐ランナー（キュー版）。各レース T-60 でオッズのみ、T-15 でバンドル（オッズ含む）を取得する。

    T-60 のオッズ取得はこのキュー版のみ（KEIBA_AUTO_SCRAPE_USE_QUEUE=0 の直接版は従来どおり T-15 のみ）。
    ``KEIBA_ODDS_LATE_SNAPSHOTS=1`` のときは、T-10 / T-5 / T-2 に単複オッズだけを追加で取る（1 レース +3 リクエスト）。
    直前イベントは同じ起床の中で T-60 / T-15 より先に処理する（時刻が命。重いバンドルの後ろに回さない）。
    """
    from src.scraper import auto_scrape as _as
    from src.scraper.raceday_pre_race_pipeline import (
        PreRaceBundleResult,
        refresh_jra_baba_live,
        trigger_pre_race_predict,
    )

    calendar = _as._load_race_calendar()
    today_str = _as._today_jst().isoformat()
    if not _as._is_race_day(calendar, today_str):
        logger.info("今日 (%s) は開催日ではありません", today_str)
        _as._save_status("raceday-runner", {"status": "skipped", "reason": "non-race-day"})
        return

    venues = _as._get_race_day_venues(calendar, today_str)
    date_fmt = today_str.replace("-", "")
    storage = _storage()
    queue = _queue()

    if not _ensure_race_list_date(storage, queue, date_fmt):
        _as._save_status(
            "raceday-runner", {"status": "error", "reason": "no-race-list"}
        )
        return

    schedule = _fetch_race_schedule_storage(storage, date_fmt)
    if not schedule:
        _as._save_status("raceday-runner", {"status": "error", "reason": "no-schedule"})
        return

    logger.info("  開催日ランナー (キュー経由): %s 全%dレース", today_str, len(schedule))
    stats = {"races": len(schedule), "scraped": 0, "odds_t60": 0, "odds_late": 0, "late_skipped": 0, "errors": []}
    late_on = late_snapshots_enabled()
    events = build_odds_events(schedule, late_leads=LATE_LEADS if late_on else ())
    if late_on:
        logger.info("  直前オッズ (T-%s) を取得する（KEIBA_ODDS_LATE_SNAPSHOTS=1）", "/".join(str(m) for m in LATE_LEADS))
    done: set[tuple[str, str]] = set()
    direct = _raceday_direct_enabled()
    _runner_box: list[Any] = []

    def _get_runner() -> Any:
        if not _runner_box:
            from src.scraper.run import ScraperRunner

            _runner_box.append(ScraperRunner())
        return _runner_box[0]

    def _run_late_due() -> None:
        """到達済みの直前オッズ（T-10 / T-5 / T-2）を取る。重い T-15 バンドルの途中でもレースの合間に呼び、時刻のずれを 1 レース分に抑える。"""
        if not late_on:
            return
        late_due, late_stale = select_due_late_events(events, done, datetime.now(_JST))
        for x in late_stale:
            done.add((x["race"]["race_id"], x["kind"]))
            stats["late_skipped"] += 1
            logger.info("直前オッズ %s スキップ（予定から遅れた／発走後）: %s", x["kind"].upper(), x["race"]["race_id"])
        if not late_due:
            return
        if direct:
            for x in late_due:
                stats["errors"] += _run_race_tasks_direct(
                    _get_runner(), x["race"]["race_id"], date_fmt, [LATE_KIND_TASK[x["kind"]]])
        else:
            queue.bulk_add_jobs([
                {"job_kind": "race", "target_id": x["race"]["race_id"], "date": date_fmt,
                 "tasks": [LATE_KIND_TASK[x["kind"]]], "smart_skip": False}
                for x in late_due
            ])
            ok, msg, _ = _kick_and_wait(queue, timeout_sec=600.0)
            if not ok:
                logger.error("直前オッズ キュー待機失敗: %s", msg)
                stats["errors"].append(msg)
        for x in late_due:
            done.add((x["race"]["race_id"], x["kind"]))
            stats["odds_late"] += 1

    def _before_heavy_step() -> None:
        """重い取得（T-15 バンドルなど）に入る前に呼ぶ。到達済みの直前オッズを取り、直前オッズが LATE_LOOKAHEAD_SECONDS 以内に
        来るならその時刻まで待ってから取る（バンドルが 1〜3 分かかって、その間に T-5 / T-2 が過ぎるのを防ぐ）。"""
        if late_on:
            now = datetime.now(_JST)
            soon = [
                e["at"] for e in events
                if e["kind"] in LATE_KIND_TASK and (e["race"]["race_id"], e["kind"]) not in done
                and now < e["at"] <= now + timedelta(seconds=LATE_LOOKAHEAD_SECONDS) and e["at"] < e["race"]["post_time"]
            ]
            if soon:
                time.sleep(max(0.0, (min(soon) - now).total_seconds()))
        _run_late_due()

    logger.info("  取得方式: %s", "直接実行（キューを待たない）" if direct else "キュー経由")

    for ev in events:
        if (ev["race"]["race_id"], ev["kind"]) in done:
            continue
        now = datetime.now(_JST)
        if now < ev["at"]:
            wait_sec = (ev["at"] - now).total_seconds()
            logger.info(
                "[%s] %s まで待機 (あと %.0f分)",
                ev["kind"].upper(),
                ev["at"].strftime("%H:%M"),
                wait_sec / 60,
            )
            time.sleep(wait_sec)

        _run_late_due()

        t60_batch, t15_batch, skipped = select_due_events(events, done, datetime.now(_JST))
        for x in skipped:
            done.add((x["race"]["race_id"], "t60"))
            logger.info("T-60 スキップ（既に T-15 時刻を過ぎた）: %s", x["race"]["race_id"])

        if t60_batch:
            specs = [
                {
                    "job_kind": "race",
                    "target_id": x["race"]["race_id"],
                    "date": date_fmt,
                    "tasks": enabled_tasks(ODDS_T60_TASKS),
                    "smart_skip": False,
                }
                for x in t60_batch
            ]
            if direct:
                for x in t60_batch:
                    _before_heavy_step()
                    stats["errors"] += _run_race_tasks_direct(_get_runner(), x["race"]["race_id"], date_fmt, enabled_tasks(ODDS_T60_TASKS))
            else:
                queue.bulk_add_jobs(specs)
                ok, msg, _ = _kick_and_wait(queue, timeout_sec=72000.0)
                if not ok:
                    logger.error("T-60 キュー待機失敗: %s", msg)
                    stats["errors"].append(msg)
            for x in t60_batch:
                done.add((x["race"]["race_id"], "t60"))
                stats["odds_t60"] += 1

        batch = [x["race"] for x in t15_batch]
        specs = [
            {
                "job_kind": "race",
                "target_id": r["race_id"],
                "date": date_fmt,
                "tasks": enabled_tasks(T15_RACE_TASKS),
                "smart_skip": False,
            }
            for r in batch
        ]
        if specs:
            if direct:
                for r in batch:
                    _before_heavy_step()
                    stats["errors"] += _run_race_tasks_direct(_get_runner(), r["race_id"], date_fmt, enabled_tasks(T15_RACE_TASKS))
            else:
                queue.bulk_add_jobs(specs)
                ok, msg, _ = _kick_and_wait(queue, timeout_sec=72000.0)
                if not ok:
                    logger.error("T-15 キュー待機失敗: %s", msg)
                    stats["errors"].append(msg)

            _before_heavy_step()
            baba_ok = refresh_jra_baba_live()
            for r in batch:
                _before_heavy_step()
                rid = r["race_id"]
                bundle = PreRaceBundleResult(race_id=rid, jra_baba_refreshed=baba_ok, ok=True)
                trigger_pre_race_predict(r, bundle_result=bundle)
                done.add((rid, "t15"))
                stats["scraped"] += 1
                time.sleep(random.uniform(1.0, 3.0))

    _as._save_status(
        "raceday-runner",
        {
            "status": "ok",
            "date": today_str,
            "venues": venues,
            "races": stats["races"],
            "scraped": stats["scraped"],
            "odds_t60": stats["odds_t60"],
            "odds_late": stats["odds_late"],
            "late_skipped": stats["late_skipped"],
            "error_count": len(stats["errors"]),
        },
    )


def task_raceday_result_runner() -> None:
    from src.scraper import auto_scrape as _as

    calendar = _as._load_race_calendar()
    today_str = _as._today_jst().isoformat()
    if not _as._is_race_day(calendar, today_str):
        logger.info("今日 (%s) は開催日ではありません", today_str)
        _as._save_status(
            "raceday-result-runner", {"status": "skipped", "reason": "non-race-day"}
        )
        return

    venues = _as._get_race_day_venues(calendar, today_str)
    date_fmt = today_str.replace("-", "")
    storage = _storage()
    queue = _queue()

    if not _ensure_race_list_date(storage, queue, date_fmt):
        _as._save_status(
            "raceday-result-runner",
            {"status": "error", "reason": "no-race-list"},
        )
        return

    schedule = _fetch_race_schedule_storage(storage, date_fmt)
    if not schedule:
        _as._save_status(
            "raceday-result-runner", {"status": "error", "reason": "no-schedule"}
        )
        return

    stats = {"races": len(schedule), "scraped": 0, "errors": []}
    scraped_ids: set[str] = set()

    for i, race in enumerate(schedule):
        target_time = race["post_time"] + timedelta(minutes=RESULT_OFFSET_MINUTES)
        now = datetime.now(_JST)
        if now < target_time:
            wait_sec = (target_time - now).total_seconds()
            logger.info(
                "[%d/%d] 速報取得 %s まで待機 (あと %.0f分)",
                i + 1,
                len(schedule),
                target_time.strftime("%H:%M"),
                wait_sec / 60,
            )
            time.sleep(wait_sec)

        batch = []
        for r in schedule:
            if r["race_id"] in scraped_ids:
                continue
            if datetime.now(_JST) >= r["post_time"] + timedelta(
                minutes=RESULT_OFFSET_MINUTES
            ):
                batch.append(r)

        specs = [
            {
                "job_kind": "race",
                "target_id": r["race_id"],
                "date": date_fmt,
                "tasks": ["race_result_on_time"],
                "smart_skip": False,
            }
            for r in batch
        ]
        if specs:
            queue.bulk_add_jobs(specs)
            ok, msg, _ = _kick_and_wait(queue, timeout_sec=72000.0)
            if not ok:
                stats["errors"].append(msg)
        for r in batch:
            scraped_ids.add(r["race_id"])
            stats["scraped"] += 1
            time.sleep(random.uniform(1.0, 2.0))

    if stats["scraped"] > 0:
        _as._trigger_track_speed_for_date(today_str)
        # 予測結果の暫定集計（KEIBA_PREDICTION_RESULTS=1 のときだけ。失敗しても本体は成功）
        from src.pipeline.inference.prediction_results import refresh_for_dates

        refresh_for_dates([today_str], reason="raceday-result-runner")

    _as._save_status(
        "raceday-result-runner",
        {
            "status": "ok",
            "date": today_str,
            "venues": venues,
            "races": stats["races"],
            "scraped": stats["scraped"],
            "error_count": len(stats["errors"]),
        },
    )


def task_raceday_weight_runner() -> None:
    """馬体重の検知ランナー（stg ホストの cron で毎朝 08:00 JST に起動。開催日以外は見送る）。

    各レースの T-65〜T-30 に出馬表（race_shutuba）を ``KEIBA_WEIGHT_WATCH_POLL_SEC``（既定 90 秒）間隔で取り直し、
    9 割以上の馬に馬体重が入ったら推論モードで特徴量を作る（``KEIBA_WEIGHT_TRIGGER_MODE=predict`` なら既存の予測も起動）。
    期限までに入らなければ、馬体重なしで起動する。既存の ``raceday-runner``（T-60 オッズ・T-15 バンドル）とは独立。

    リハーサル: ``KEIBA_WEIGHT_WATCH_DATE=YYYYMMDD KEIBA_WEIGHT_WATCH_REPLAY=1`` で、取得も待機もせず保存済みの出馬表だけで一巡する。
    """
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.scraper import auto_scrape as _as
    from src.scraper import raceday_weight_watch as ww

    replay = os.environ.get("KEIBA_WEIGHT_WATCH_REPLAY", "").strip().lower() in ("1", "true", "yes")
    ymd = (os.environ.get("KEIBA_WEIGHT_WATCH_DATE") or "").strip()
    today_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}" if len(ymd) == 8 else _as._today_jst().isoformat()
    date_fmt = today_str.replace("-", "")

    if not ymd:
        calendar = _as._load_race_calendar()
        if not _as._is_race_day(calendar, today_str):
            logger.info("今日 (%s) は開催日ではありません", today_str)
            _as._save_status("raceday-weight-runner", {"status": "skipped", "reason": "non-race-day"})
            return

    storage = _storage()
    queue = _queue()
    if not replay and not _ensure_race_list_date(storage, queue, date_fmt):
        _as._save_status("raceday-weight-runner", {"status": "error", "reason": "no-race-list"})
        return
    schedule = _fetch_race_schedule_storage(storage, date_fmt)
    if not schedule:
        _as._save_status("raceday-weight-runner", {"status": "error", "reason": "no-schedule"})
        return

    ctx = wtp.InferenceContext()
    warm = ctx.warm()
    windows = ww.plan_windows(schedule)
    far_future = max(w.deadline for w in windows) + timedelta(days=1)
    now_fn = (lambda: far_future) if replay else (lambda: datetime.now(_JST))
    poll = ww.poll_sec()
    logger.info("  馬体重検知ランナー: %s 全%dレース mode=%s replay=%s poll=%ss 窓 T-%s〜T-%s",
                today_str, len(schedule), wtp.trigger_mode(), replay, poll, ww.start_min(), ww.deadline_min())

    done: set[str] = set()
    stats: dict[str, Any] = {"announced": 0, "deadline": 0, "errors": 0, "polls": 0}
    while True:
        now = now_fn()
        wake = ww.next_wake(windows, done, now, poll)
        if wake is None:
            break
        if wake > now and not replay:
            wait = (wake - now).total_seconds()
            logger.info("[weight] 次の取得まで待機 (あと %.0f秒)", wait)
            time.sleep(wait)
        now = now_fn()
        due = ww.open_windows(windows, done, now)
        if not due:
            continue

        if not replay:
            queue.bulk_add_jobs([
                {"job_kind": "race", "target_id": w.race_id, "date": date_fmt,
                 "tasks": list(ww.WEIGHT_WATCH_TASKS), "smart_skip": False}
                for w in due
            ])
            ok, msg, _ = _kick_and_wait(queue, timeout_sec=1800.0)
            stats["polls"] += 1
            if not ok:
                logger.error("[weight] キュー待機失敗: %s", msg)

        ready: list[dict] = []
        known: dict[str, bool] = {}
        for w in due:
            card = storage.load("race_shutuba", w.race_id, bypass_cache=True)
            verdict = ww.decide(card, w, now_fn())
            if verdict == "wait":
                continue
            done.add(w.race_id)
            stats[verdict] += 1
            known[w.race_id] = verdict == "announced"
            ready.append(w.race)
            logger.info("[weight] %s %s (馬体重 %.0f%%)", w.race_id, verdict, 100 * ww.weights_ratio(card))
        if ready:
            for ev in wtp.run_for_races(ready, storage, ctx, weights_known=known):
                if ev.get("status") != "ok":
                    stats["errors"] += 1

    _as._save_status("raceday-weight-runner", {
        "status": "ok", "date": today_str, "races": len(schedule), "replay": replay,
        "mode": wtp.trigger_mode(), "history_last_date": warm.get("history_last_date"), **stats,
    })
