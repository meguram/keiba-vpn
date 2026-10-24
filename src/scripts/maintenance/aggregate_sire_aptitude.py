"""
父・母父の舞台適性統計を集計して DB に保存する（毎週月曜実行）。

使い方:
  python -m src.scripts.maintenance.aggregate_sire_aptitude
  python -m src.scripts.maintenance.aggregate_sire_aptitude --week 2026-W28
  python -m src.scripts.maintenance.aggregate_sire_aptitude --dry-run
"""

from __future__ import annotations

import argparse
import collections
import datetime
import sys
from typing import Any

from src.scraper.storage import HybridStorage
from src.utils.distance_band import distance_group_key
from src.utils.logger import get_logger

logger = get_logger("AggSireAptitude")

_TRACK_CONDITION_ALIASES = {
    "firm": "良", "good": "良", "good_to_firm": "稍重",
    "yielding": "稍重", "soft": "重", "heavy": "不良",
}


def _norm_condition(raw: str) -> str:
    r = (raw or "").strip()
    return _TRACK_CONDITION_ALIASES.get(r, r) or "良"


def _norm_surface(raw: str) -> str:
    r = (raw or "").strip()
    if r in ("芝", "turf", "T"):
        return "芝"
    if r in ("ダート", "dirt", "D"):
        return "ダート"
    return r or "芝"


def _current_week_label() -> str:
    today = datetime.date.today()
    y, w, _ = today.isocalendar()
    return f"{y}-W{w:02d}"


def run_aggregation(storage: HybridStorage, week_label: str, dry_run: bool = False) -> dict:
    """全 race_result を走査して SireAptitudeCache を upsert する。"""
    from src.db.session import get_session, init_engine
    from src.db.models import SireAptitudeCache
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    logger.info("集計開始: week=%s dry_run=%s", week_label, dry_run)

    # race_result 全キー取得
    try:
        all_race_keys = storage.list_keys("race_result")
    except Exception as e:
        logger.error("race_result キー一覧取得失敗: %s", e)
        return {"status": "error", "error": str(e)}

    logger.info("対象レース数: %d", len(all_race_keys))

    # horse_result キャッシュ（同一馬を何度も読まない）
    horse_cache: dict[str, dict] = {}

    # 集計バッファ: key=(sire_name, sire_type, surface, distance_band, track_condition)
    # value: {n_runs, n_wins, n_place, win_odds_sum_for_wins, place_odds_sum_for_place}
    stats: dict[tuple, dict] = collections.defaultdict(lambda: {
        "n_runs": 0, "n_wins": 0, "n_place": 0,
        "win_odds_acc": 0.0, "place_odds_acc": 0.0,
    })

    processed = 0
    skipped = 0

    for race_key in sorted(all_race_keys):
        race_id = race_key.replace(".json", "")
        try:
            result = storage.load("race_result", race_id)
        except Exception:
            skipped += 1
            continue

        if not result or not isinstance(result, dict):
            skipped += 1
            continue

        surface = _norm_surface(result.get("surface", ""))
        distance = result.get("distance") or result.get("distance_m") or 0
        distance_band = distance_group_key(distance)
        track_condition = _norm_condition(result.get("track_condition", ""))

        entries = result.get("entries") or result.get("results") or []
        if not entries:
            skipped += 1
            continue

        for entry in entries:
            horse_id = entry.get("horse_id", "")
            if not horse_id:
                continue

            finish_pos_raw = entry.get("finish_position") or entry.get("finish_pos") or 99
            try:
                finish_pos = int(finish_pos_raw)
            except (TypeError, ValueError):
                finish_pos = 99

            win_odds_raw = entry.get("odds") or entry.get("win_odds") or 0
            try:
                win_odds = float(win_odds_raw)
            except (TypeError, ValueError):
                win_odds = 0.0

            place_odds_raw = entry.get("place_odds") or entry.get("place_odds_avg") or 0
            try:
                place_odds = float(place_odds_raw)
            except (TypeError, ValueError):
                place_odds = 0.0

            # 馬の父・母父を取得
            if horse_id not in horse_cache:
                try:
                    hr = storage.load("horse_result", horse_id) or {}
                    horse_cache[horse_id] = hr
                except Exception:
                    horse_cache[horse_id] = {}
            hr = horse_cache[horse_id]

            sire_name = (hr.get("sire") or "").strip()
            dam_sire_name = (hr.get("dam_sire") or "").strip()

            for sire_n, sire_t in [(sire_name, "sire"), (dam_sire_name, "dam_sire")]:
                if not sire_n:
                    continue
                key = (sire_n, sire_t, surface, distance_band, track_condition)
                s = stats[key]
                s["n_runs"] += 1
                if finish_pos == 1:
                    s["n_wins"] += 1
                    if win_odds > 0:
                        s["win_odds_acc"] += win_odds
                if 1 <= finish_pos <= 3:
                    s["n_place"] += 1
                    if place_odds > 0:
                        s["place_odds_acc"] += place_odds

        processed += 1
        if processed % 500 == 0:
            logger.info("処理済み: %d/%d", processed, len(all_race_keys))

    logger.info("集計完了: レース=%d スキップ=%d 種牡馬キー=%d", processed, skipped, len(stats))

    if dry_run:
        sample = list(stats.items())[:5]
        return {"status": "dry_run", "n_keys": len(stats), "sample": [
            {"key": k, "stats": v} for k, v in sample
        ]}

    # DB upsert
    init_engine()
    n_upserted = 0
    batch: list[dict] = []

    for (sire_name, sire_type, surface, distance_band, track_condition), s in stats.items():
        n_runs = s["n_runs"]
        n_wins = s["n_wins"]
        n_place = s["n_place"]
        win_rate = round(n_wins / n_runs, 4) if n_runs > 0 else None
        place_rate = round(n_place / n_runs, 4) if n_runs > 0 else None
        # roi_win = (Σ win_odds when won) / n_runs  (each bet 100 yen, so × 100 / 100 = just odds/runs)
        roi_win = round(s["win_odds_acc"] / n_runs, 4) if n_runs > 0 else None
        roi_place = round(s["place_odds_acc"] / n_runs, 4) if n_runs > 0 else None

        batch.append({
            "sire_name": sire_name,
            "sire_type": sire_type,
            "surface": surface,
            "distance_band": distance_band,
            "track_condition": track_condition,
            "n_runs": n_runs,
            "n_wins": n_wins,
            "n_place": n_place,
            "win_rate": win_rate,
            "place_rate": place_rate,
            "roi_win": roi_win,
            "roi_place": roi_place,
            "week_label": week_label,
            "computed_at": datetime.datetime.now(tz=datetime.timezone.utc),
        })

        if len(batch) >= 200:
            _upsert_batch(batch)
            n_upserted += len(batch)
            batch = []

    if batch:
        _upsert_batch(batch)
        n_upserted += len(batch)

    logger.info("DB upsert 完了: %d 件", n_upserted)
    return {
        "status": "ok",
        "week_label": week_label,
        "n_races_processed": processed,
        "n_keys_aggregated": len(stats),
        "n_upserted": n_upserted,
    }


def _upsert_batch(rows: list[dict]) -> None:
    from src.db.session import get_session
    from src.db.models import SireAptitudeCache
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    with get_session() as session:
        stmt = pg_insert(SireAptitudeCache).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_sire_aptitude_cache",
            set_={
                "n_runs": stmt.excluded.n_runs,
                "n_wins": stmt.excluded.n_wins,
                "n_place": stmt.excluded.n_place,
                "win_rate": stmt.excluded.win_rate,
                "place_rate": stmt.excluded.place_rate,
                "roi_win": stmt.excluded.roi_win,
                "roi_place": stmt.excluded.roi_place,
                "computed_at": stmt.excluded.computed_at,
            },
        )
        session.execute(stmt)
        session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="父・母父 舞台適性統計を集計して DB に保存")
    parser.add_argument("--week", default=None, help="週ラベル (例: 2026-W28)。省略時は今週")
    parser.add_argument("--dry-run", action="store_true", help="DB 書き込みを行わず集計結果のみ表示")
    args = parser.parse_args()

    week_label = args.week or _current_week_label()
    storage = HybridStorage()
    result = run_aggregation(storage, week_label, dry_run=args.dry_run)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
