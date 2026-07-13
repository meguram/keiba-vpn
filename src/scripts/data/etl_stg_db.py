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
        "jockey_id": entry.get("jockey_id"),
        "trainer_id": entry.get("trainer_id"),
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
        "jockey_id": result.get("jockey_id"),
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

            jockey_id = entry.get("jockey_id")
            if jockey_id:
                _upsert_jockey(session, jockey_id, entry.get("jockey_name") or jockey_id)

            trainer_id = entry.get("trainer_id")
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
                _upsert_race_result(session, race_id, horse_id, re)

        # mock predictions
        if horse_ids:
            _upsert_mock_predictions(session, race_id, horse_ids)

        session.flush()
        return {"race_id": race_id, "status": "ok", "horses": len(horse_ids)}

    except Exception as e:
        logger.error("race_id=%s 投入エラー: %s", race_id, e)
        session.rollback()
        return {"race_id": race_id, "status": "error", "error": str(e)}


def get_target_dates(year: int | None, recent_days: int | None) -> list[str]:
    """対象日付リストを生成（YYYYMMDD 文字列のリスト）。"""
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


def main():
    parser = argparse.ArgumentParser(description="stg DB ETL: GCS → PostgreSQL")
    parser.add_argument("--year", type=int, default=None, help="対象年 (例: 2026)")
    parser.add_argument("--recent-days", type=int, default=None, help="直近N日分")
    parser.add_argument("--dry-run", action="store_true", help="DB書き込みをスキップ")
    parser.add_argument("--workers", type=int, default=4, help="並列GCS取得スレッド数")
    parser.add_argument("--batch-size", type=int, default=50, help="DBコミット単位")
    args = parser.parse_args()

    target_dates = get_target_dates(args.year, args.recent_days)
    logger.info("対象日付数: %d", len(target_dates))

    race_ids = get_race_ids_for_dates(target_dates)
    logger.info("対象レース数: %d", len(race_ids))

    if not race_ids:
        logger.warning("対象レースが見つかりません")
        return

    if args.dry_run:
        logger.info("[DRY RUN] 最初の5件確認:")
        for rid in race_ids[:5]:
            card = load_race_card_from_gcs(rid)
            if card:
                entries = card.get("entries") or []
                logger.info("  %s: %s (%d頭)", rid, card.get("race_name", "?"), len(entries))
            else:
                logger.info("  %s: GCS データなし", rid)
        return

    from sqlalchemy.orm import Session as SASession

    engine = _get_engine()
    stats = {"ok": 0, "error": 0, "no_data": 0}

    logger.info("DB投入開始 (workers=%d, batch=%d)...", args.workers, args.batch_size)

    # バッチ処理
    with engine.connect() as conn:
        batch: list[str] = []

        def flush_batch(batch: list[str]):
            if not batch:
                return
            with SASession(conn) as session:
                for rid in batch:
                    r = process_race(session, rid, dry_run=False)
                    stats[r["status"] if r["status"] in stats else "error"] += 1
                try:
                    session.commit()
                    logger.info(
                        "コミット %d件 (ok=%d err=%d no_data=%d)",
                        len(batch), stats["ok"], stats["error"], stats["no_data"]
                    )
                except Exception as e:
                    logger.error("コミットエラー: %s", e)
                    session.rollback()

        for i, race_id in enumerate(race_ids):
            batch.append(race_id)
            if len(batch) >= args.batch_size:
                flush_batch(batch)
                batch = []

        if batch:
            flush_batch(batch)

    logger.info(
        "完了: ok=%d error=%d no_data=%d (total=%d)",
        stats["ok"], stats["error"], stats["no_data"], len(race_ids),
    )


if __name__ == "__main__":
    main()
