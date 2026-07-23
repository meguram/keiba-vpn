"""
stg DB ETL スクリプト
=====================
GCS の race_shutuba / race_result データを読み込み、
PostgreSQL stg DB の races / jockeys / trainers / horses / entries / race_results
テーブルにデータを投入する。

また、DB に prediction_results (モック予測) を生成して保存する。

Usage:
  python3 -m src.scripts.data.etl_stg_db [--year 2026] [--recent-days 90]
  python3 -m src.scripts.data.etl_stg_db --recent-days 30 --dry-run
  python3 -m src.scripts.data.etl_stg_db --dates 20260718,20260719
  python3 -m src.scripts.data.etl_stg_db --force  # PG 充足済みも再同期
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.project_env import load_project_dotenv

load_project_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# KEIBA_ENV=stg で DB URL を上書き
_DB_URL = (
    os.environ.get("DATABASE_URL")
    or "postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db_stg"
)

MOCK_MODEL_VERSION = "stg-mock-v1"
RACE_LISTS_DIR = PROJECT_ROOT / "data" / "page_reference" / "race_lists"
ETL_SYNC_DIR = PROJECT_ROOT / "data" / "local" / "meta" / "etl_sync"


def _should_gen_mock_predictions() -> bool:
    env = (os.environ.get("KEIBA_ENV") or "dev").strip().lower()
    if env == "prod":
        return False
    flag = (os.environ.get("KEIBA_ETL_MOCK_PREDICTIONS") or "").strip().lower()
    if flag in ("0", "false", "no"):
        return False
    return env in ("stg", "dev") or flag in ("1", "true", "yes")


def _normalize_fk_id(value: str | None) -> str | None:
    """FK 参照用 ID。空文字は NULL 扱い（entries_jockey_id_fkey 回避）。"""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _get_engine():
    from sqlalchemy import create_engine

    return create_engine(_DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)


def _upsert_jockey(session, jockey_id: str, jockey_name: str):
    from sqlalchemy.dialects.postgresql import insert
    from src.db.models import Jockey

    stmt = insert(Jockey).values(jockey_id=jockey_id, jockey_name=jockey_name)
    stmt = stmt.on_conflict_do_nothing()
    session.execute(stmt)


def _upsert_trainer(session, trainer_id: str, trainer_name: str):
    from sqlalchemy.dialects.postgresql import insert
    from src.db.models import Trainer

    stmt = insert(Trainer).values(trainer_id=trainer_id, trainer_name=trainer_name)
    stmt = stmt.on_conflict_do_nothing()
    session.execute(stmt)


def _upsert_horse(session, horse_id: str, horse_name: str, sex: str = None, birth_year: int = None):
    from sqlalchemy.dialects.postgresql import insert
    from src.db.models import Horse

    vals: dict = {"horse_id": horse_id, "horse_name": horse_name or horse_id}
    if sex:
        vals["sex"] = sex
    if birth_year:
        vals["birth_year"] = birth_year
    stmt = insert(Horse).values(**vals)
    stmt = stmt.on_conflict_do_nothing()
    session.execute(stmt)


def _upsert_race(session, race_id: str, card: dict):
    from sqlalchemy.dialects.postgresql import insert
    from src.db.models import Race

    def _parse_time(t: str):
        if not t:
            return None
        try:
            return datetime.strptime(t, "%H:%M").time()
        except Exception:
            try:
                return datetime.strptime(t, "%H:%M:%S").time()
            except Exception:
                return None

    def _parse_date(d: str):
        if not d:
            return None
        try:
            return datetime.strptime(d, "%Y%m%d").date()
        except Exception:
            try:
                return date.fromisoformat(d)
            except Exception:
                return None

    vals = {
        "race_id": race_id,
        "race_name": (card.get("race_name") or "")[:200],
        "venue": (card.get("venue") or "")[:20],
        "surface": (card.get("surface") or "")[:10],
        "distance": card.get("distance"),
        "direction": (card.get("direction") or "")[:10],
        "weather": (card.get("weather") or "")[:20],
        "track_condition": (card.get("track_condition") or "")[:10],
        "start_time": _parse_time(card.get("start_time") or ""),
        "race_date": _parse_date(card.get("date") or ""),
        "grade": (card.get("grade") or "")[:20],
        "race_class": (card.get("race_class") or "")[:100],
        "field_size": card.get("entries_count") or len(card.get("entries") or []),
    }
    stmt = insert(Race).values(**vals)
    stmt = stmt.on_conflict_do_update(
        index_elements=["race_id"],
        set_={k: v for k, v in vals.items() if k != "race_id"},
    )
    session.execute(stmt)


def _upsert_entry(session, race_id: str, entry: dict):
    from sqlalchemy.dialects.postgresql import insert
    from src.db.models import Entry

    horse_id = entry.get("horse_id") or ""
    if not horse_id:
        return

    vals = {
        "race_id": race_id,
        "horse_id": horse_id,
        "post_no": entry.get("post_no") or entry.get("horse_number"),
        "bracket_number": entry.get("bracket_number") or entry.get("bracket_no"),
        "jockey_id": _normalize_fk_id(entry.get("jockey_id")),
        "trainer_id": _normalize_fk_id(entry.get("trainer_id")),
        "sex_age": (entry.get("sex_age") or "")[:10],
        "weight": entry.get("weight"),
        "weight_change": entry.get("weight_change"),
        "jockey_weight": entry.get("jockey_weight"),
    }
    stmt = insert(Entry).values(**vals)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_entries_race_horse",
        set_={k: v for k, v in vals.items() if k not in ("race_id", "horse_id")},
    )
    session.execute(stmt)


def _upsert_race_result(session, race_id: str, horse_id: str, result: dict):
    from sqlalchemy.dialects.postgresql import insert
    from src.db.models import RaceResult

    vals = {
        "race_id": race_id,
        "horse_id": horse_id,
        "finish_pos": result.get("finish_pos") or result.get("rank") or result.get("finish_position"),
        "finish_time_sec": result.get("finish_time_sec") or result.get("time_sec"),
        "margin": (result.get("margin") or "")[:20],
        "last_3f_sec": result.get("last_3f_sec") or result.get("last_3f"),
        "weight": result.get("weight"),
        "jockey_id": _normalize_fk_id(result.get("jockey_id")),
    }
    stmt = insert(RaceResult).values(**vals)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_race_results_race_horse",
        set_={k: v for k, v in vals.items() if k not in ("race_id", "horse_id")},
    )
    session.execute(stmt)


def _gen_mock_predictions(race_id: str, horse_ids: list[str]) -> list[dict]:
    """horse_ids を入力に、正規化した mock 予測確率を返す。"""
    if not horse_ids:
        return []
    rng = random.Random(race_id)  # race_id シードで再現性あり
    raw = [rng.uniform(0.5, 3.0) for _ in horse_ids]
    total = sum(raw)
    win_probs = [v / total for v in raw]
    results = []
    for i, horse_id in enumerate(horse_ids):
        wp = win_probs[i]
        pp = min(wp * 2.2, 0.95)
        results.append({
            "horse_id": horse_id,
            "win_prob": round(wp, 4),
            "place_prob": round(pp, 4),
            "show_prob": round(min(pp * 1.3, 0.98), 4),
            "predicted_position": i + 1,
            "predicted_running_style": rng.choice(["逃", "先", "差", "追"]),
            "expected_win_roi": round(rng.uniform(-0.3, 0.8), 2),
            "expected_show_roi": round(rng.uniform(-0.1, 0.5), 2),
        })
    return results


def _upsert_mock_predictions(session, race_id: str, horse_ids: list[str]):
    from sqlalchemy.dialects.postgresql import insert
    from src.db.models import PredictionResult

    preds = _gen_mock_predictions(race_id, horse_ids)
    for p in preds:
        vals = {
            "race_id": race_id,
            "horse_id": p["horse_id"],
            "model_version": MOCK_MODEL_VERSION,
            "win_prob": p["win_prob"],
            "place_prob": p["place_prob"],
            "show_prob": p["show_prob"],
            "predicted_position": p["predicted_position"],
            "predicted_running_style": p["predicted_running_style"],
            "expected_win_roi": p["expected_win_roi"],
            "expected_show_roi": p["expected_show_roi"],
        }
        stmt = insert(PredictionResult).values(**vals)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_prediction_race_horse_model",
            set_={k: v for k, v in vals.items() if k not in ("race_id", "horse_id", "model_version")},
        )
        session.execute(stmt)


def get_race_ids_for_dates(target_dates: list[str]) -> list[str]:
    """race_lists から race_id 一覧を取得。"""
    race_ids = []
    for d in target_dates:
        p = RACE_LISTS_DIR / f"{d}.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for r in data.get("races") or []:
                rid = r.get("race_id")
                if rid:
                    race_ids.append(rid)
        except Exception as e:
            logger.warning("race_lists 読み込みエラー %s: %s", d, e)
    return race_ids


def load_race_card_from_gcs(race_id: str) -> dict | None:
    """GCS から race_shutuba / race_result をマージして取得。"""
    try:
        from src.utils.race_card_merge import load_merged_race_card

        return load_merged_race_card(race_id)
    except Exception as e:
        logger.debug("GCS load エラー %s: %s", race_id, e)
        return None


def load_race_result_from_gcs(race_id: str) -> dict | None:
    """GCS から race_result を取得。"""
    try:
        from src.scraper.storage import HybridStorage
        storage = HybridStorage()
        return storage.load("race_result", race_id)
    except Exception as e:
        logger.debug("GCS race_result load エラー %s: %s", race_id, e)
        return None


def process_race(session, race_id: str, dry_run: bool = False) -> dict:
    """1 レース分のデータを DB に投入。"""
    card = load_race_card_from_gcs(race_id)
    if not card:
        return {"race_id": race_id, "status": "no_data"}

    if dry_run:
        entries = card.get("entries") or []
        return {"race_id": race_id, "status": "dry_run", "entries": len(entries)}

    try:
        # races テーブル
        _upsert_race(session, race_id, card)

        entries_data = card.get("entries") or []
        horse_ids = []

        for entry in entries_data:
            horse_id = entry.get("horse_id") or ""
            if not horse_id:
                continue
            horse_ids.append(horse_id)

            horse_name = entry.get("horse_name") or horse_id
            sex = entry.get("sex") or entry.get("sex_age", "")
            _upsert_horse(session, horse_id, horse_name, sex=sex[:5] if sex else None)

            jockey_id = _normalize_fk_id(entry.get("jockey_id"))
            if jockey_id:
                _upsert_jockey(session, jockey_id, entry.get("jockey_name") or jockey_id)

            trainer_id = _normalize_fk_id(entry.get("trainer_id"))
            if trainer_id:
                _upsert_trainer(session, trainer_id, entry.get("trainer_name") or trainer_id)

            _upsert_entry(session, race_id, entry)

        # race_results テーブル
        result_data = load_race_result_from_gcs(race_id)
        if result_data:
            result_entries = result_data.get("entries") or result_data.get("results") or []
            for re in result_entries:
                horse_id = re.get("horse_id") or ""
                if not horse_id:
                    continue
                if horse_id not in horse_ids:
                    _upsert_horse(session, horse_id, re.get("horse_name") or horse_id)
                    horse_ids.append(horse_id)
                jockey_id = _normalize_fk_id(re.get("jockey_id"))
                if jockey_id:
                    _upsert_jockey(session, jockey_id, re.get("jockey_name") or jockey_id)
                _upsert_race_result(session, race_id, horse_id, re)

        if horse_ids and _should_gen_mock_predictions():
            _upsert_mock_predictions(session, race_id, horse_ids)

        session.flush()
        return {"race_id": race_id, "status": "ok", "horses": len(horse_ids)}

    except Exception as e:
        logger.error("race_id=%s 投入エラー: %s", race_id, e)
        session.rollback()
        return {"race_id": race_id, "status": "error", "error": str(e)}


def get_target_dates(
    year: int | None,
    recent_days: int | None,
    *,
    dates: list[str] | None = None,
) -> list[str]:
    """対象日付リストを生成（YYYYMMDD 文字列のリスト）。"""
    if dates:
        return sorted({d.strip() for d in dates if d and len(d.strip()) == 8 and d.strip().isdigit()})

    available = sorted(
        p.stem for p in RACE_LISTS_DIR.glob("*.json")
        if p.stem.isdigit() and len(p.stem) == 8
    )
    today = date.today()

    if recent_days is not None:
        cutoff = today - timedelta(days=recent_days)
        cutoff_str = cutoff.strftime("%Y%m%d")
        return [d for d in available if d >= cutoff_str]

    if year is not None:
        prefix = str(year)
        return [d for d in available if d.startswith(prefix)]

    # デフォルト: 2026年全日
    return [d for d in available if d.startswith("2026")]


def query_pg_status_batch(engine, race_ids: list[str]) -> dict[str, dict[str, int | bool]]:
    """race_id ごとの PG 充足状況（GCS download なし）。"""
    if not race_ids:
        return {}

    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT r.race_id,
                       1 AS has_race,
                       (SELECT COUNT(*) FROM entries e WHERE e.race_id = r.race_id) AS entry_cnt,
                       (SELECT COUNT(*) FROM race_results rr
                        WHERE rr.race_id = r.race_id
                          AND rr.finish_pos > 0 AND rr.finish_time_sec > 0) AS finisher_cnt
                FROM races r
                WHERE r.race_id = ANY(:ids)
            """),
            {"ids": race_ids},
        ).fetchall()

    out: dict[str, dict[str, int | bool]] = {}
    for row in rows:
        out[row.race_id] = {
            "has_race": bool(row.has_race),
            "entry_cnt": int(row.entry_cnt or 0),
            "finisher_cnt": int(row.finisher_cnt or 0),
        }
    return out


def load_gcs_race_keys(years: set[str]) -> tuple[set[str], set[str]]:
    """GCS list のみで race_shutuba / race_result の race_id 集合を取得。"""
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    shutuba: set[str] = set()
    results: set[str] = set()
    for year in sorted(years):
        try:
            shutuba |= set(storage.batch_list_blobs("race_shutuba", year).keys())
        except Exception as e:
            logger.warning("batch_list race_shutuba %s: %s", year, e)
        try:
            results |= set(storage.batch_list_blobs("race_result", year).keys())
        except Exception as e:
            logger.warning("batch_list race_result %s: %s", year, e)
    return shutuba, results


def filter_races_needing_sync(
    race_ids: list[str],
    pg_status: dict[str, dict[str, int | bool]],
    gcs_shutuba: set[str],
    gcs_results: set[str],
    *,
    skip_if_complete: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """同期対象 race_id とスキップ理由を返す。"""
    to_sync: list[str] = []
    skipped: dict[str, str] = {}

    for rid in race_ids:
        if rid not in gcs_shutuba:
            skipped[rid] = "no_gcs_shutuba"
            continue

        if not skip_if_complete:
            to_sync.append(rid)
            continue

        pg = pg_status.get(rid, {})
        has_race = bool(pg.get("has_race"))
        entry_cnt = int(pg.get("entry_cnt") or 0)
        finisher_cnt = int(pg.get("finisher_cnt") or 0)
        has_gcs_result = rid in gcs_results

        if not has_race or entry_cnt == 0:
            to_sync.append(rid)
            continue

        if has_gcs_result and finisher_cnt == 0:
            to_sync.append(rid)
            continue

        if not has_gcs_result:
            skipped[rid] = "pg_entries_ok_no_gcs_result"
            continue

        skipped[rid] = "pg_complete"

    return to_sync, skipped


def save_etl_sync_run(
    target_dates: list[str],
    *,
    stats: dict[str, int],
    skipped: int,
    reason: str = "",
) -> None:
    """日次 ETL 実行サマリをローカルに記録（再 download 判定用）。"""
    payload = {
        "target_dates": target_dates,
        "stats": stats,
        "skipped_races": skipped,
        "reason": reason,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    stamp = datetime.now().strftime("%Y%m%d")
    path = ETL_SYNC_DIR / stamp[:4] / f"{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_etl(
    *,
    year: int | None = None,
    recent_days: int | None = None,
    dates: list[str] | None = None,
    dry_run: bool = False,
    batch_size: int = 50,
    skip_if_pg_complete: bool = True,
    reason: str = "",
) -> dict[str, Any]:
    """GCS → PostgreSQL 増分 ETL の共通エントリ。"""
    target_dates = get_target_dates(year, recent_days, dates=dates)
    logger.info(
        "ETL 開始 reason=%s dates=%d recent_days=%s skip_if_complete=%s",
        reason or "-",
        len(target_dates),
        recent_days,
        skip_if_pg_complete,
    )

    race_ids = get_race_ids_for_dates(target_dates)
    logger.info("race_lists レース数: %d", len(race_ids))
    if not race_ids:
        return {"status": "skipped", "reason": "no_race_ids", "target_dates": target_dates}

    if dry_run:
        pg_status: dict[str, dict[str, int | bool]] = {}
        try:
            engine = _get_engine()
            pg_status = query_pg_status_batch(engine, race_ids)
        except Exception as e:
            logger.warning("PG 未接続のため skip 判定なし（全件候補）: %s", e)
        years = {rid[:4] for rid in race_ids}
        gcs_shutuba, gcs_results = load_gcs_race_keys(years)
        to_sync, skipped_map = filter_races_needing_sync(
            race_ids,
            pg_status,
            gcs_shutuba,
            gcs_results,
            skip_if_complete=skip_if_pg_complete and bool(pg_status),
        )
        logger.info(
            "[DRY RUN] sync=%d skip=%d (sample sync: %s)",
            len(to_sync),
            len(skipped_map),
            to_sync[:5],
        )
        return {
            "status": "dry_run",
            "target_dates": target_dates,
            "total_races": len(race_ids),
            "to_sync": len(to_sync),
            "skipped": len(skipped_map),
        }

    engine = _get_engine()
    pg_status = query_pg_status_batch(engine, race_ids)
    years = {rid[:4] for rid in race_ids}
    gcs_shutuba, gcs_results = load_gcs_race_keys(years)
    to_sync, skipped_map = filter_races_needing_sync(
        race_ids,
        pg_status,
        gcs_shutuba,
        gcs_results,
        skip_if_complete=skip_if_pg_complete,
    )
    logger.info(
        "同期対象: %d / %d (skip=%d)",
        len(to_sync),
        len(race_ids),
        len(skipped_map),
    )

    if not to_sync:
        save_etl_sync_run(target_dates, stats={"ok": 0, "error": 0, "no_data": 0}, skipped=len(skipped_map), reason=reason)
        return {
            "status": "ok",
            "target_dates": target_dates,
            "total_races": len(race_ids),
            "synced": 0,
            "skipped": len(skipped_map),
            "stats": {"ok": 0, "error": 0, "no_data": 0},
        }

    from sqlalchemy.orm import Session as SASession

    stats = {"ok": 0, "error": 0, "no_data": 0}
    with engine.connect() as conn:
        batch: list[str] = []

        def flush_batch(batch_ids: list[str]) -> None:
            if not batch_ids:
                return
            with SASession(conn) as session:
                for rid in batch_ids:
                    r = process_race(session, rid, dry_run=False)
                    key = r["status"] if r["status"] in stats else "error"
                    stats[key] += 1
                try:
                    session.commit()
                    logger.info(
                        "コミット %d件 (ok=%d err=%d no_data=%d)",
                        len(batch_ids),
                        stats["ok"],
                        stats["error"],
                        stats["no_data"],
                    )
                except Exception as e:
                    logger.error("コミットエラー: %s", e)
                    session.rollback()

        for race_id in to_sync:
            batch.append(race_id)
            if len(batch) >= batch_size:
                flush_batch(batch)
                batch = []
        if batch:
            flush_batch(batch)

    save_etl_sync_run(target_dates, stats=stats, skipped=len(skipped_map), reason=reason)
    logger.info(
        "ETL 完了: ok=%d error=%d no_data=%d synced=%d skipped=%d",
        stats["ok"],
        stats["error"],
        stats["no_data"],
        len(to_sync),
        len(skipped_map),
    )
    return {
        "status": "ok",
        "target_dates": target_dates,
        "total_races": len(race_ids),
        "synced": len(to_sync),
        "skipped": len(skipped_map),
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="stg DB ETL: GCS → PostgreSQL")
    parser.add_argument("--year", type=int, default=None, help="対象年 (例: 2026)")
    parser.add_argument("--recent-days", type=int, default=None, help="直近N日分")
    parser.add_argument(
        "--dates",
        type=str,
        default=None,
        help="対象日 YYYYMMDD カンマ区切り",
    )
    parser.add_argument("--dry-run", action="store_true", help="DB書き込みをスキップ")
    parser.add_argument("--batch-size", type=int, default=50, help="DBコミット単位")
    parser.add_argument(
        "--skip-if-pg-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PG 充足済みレースは GCS download をスキップ（デフォルト: 有効）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="--no-skip-if-pg-complete と同義（全件再同期）",
    )
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",")] if args.dates else None
    skip_if_complete = args.skip_if_pg_complete and not args.force

    run_etl(
        year=args.year,
        recent_days=args.recent_days,
        dates=dates,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        skip_if_pg_complete=skip_if_complete,
        reason="cli",
    )


if __name__ == "__main__":
    main()
