"""今後の開催日に出走予定の馬のうち、血統テーブル（ped_tbl）が無い馬（主に新馬・初出走）の 5 世代血統を取得する。

出走予定馬は保存済みの出馬表（race_shutuba）から集める。取得は ``ScraperRunner.scrape_horse_pedigree_5gen``
（GCS 保存 + ローカルの 5 世代・10 世代・ped_tbl の更新まで行う）。ped_tbl が既にある馬には何もしない。
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_RACE_ID = re.compile(r"^\d{12}$")
MAX_HORSES_PER_RUN = 150
MIN_PARQUET_BYTES = 32


def race_ids_from_list_json(path: Path) -> list[str]:
    """レース一覧 JSON（data/page_reference/race_lists/{YYYYMMDD}.json）から race_id を拾う。形式の違いに備えて再帰で探す。"""
    if not path.is_file():
        return []
    found: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            rid = x.get("race_id")
            if isinstance(rid, str) and _RACE_ID.match(rid):
                found.append(rid)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(json.loads(path.read_text(encoding="utf-8")))
    return list(dict.fromkeys(found))


def ped_tbl_path(horse_id: str, base_dir: Path) -> Path:
    from src.pipeline.features.horse_entity_layout import ped_parquet_path

    return ped_parquet_path(horse_id, base_dir)


def has_ped_tbl(horse_id: str, base_dir: Path) -> bool:
    p = ped_tbl_path(horse_id, base_dir)
    return p.is_file() and p.stat().st_size >= MIN_PARQUET_BYTES


def upcoming_dates(race_days: Iterable[str], today: date, horizon_days: int = 7) -> list[str]:
    """開催日（YYYY-MM-DD）のうち today の翌日から horizon_days 日先までを YYYYMMDD で返す。"""
    lo, hi = today + timedelta(days=1), today + timedelta(days=horizon_days)
    out = []
    for d in sorted(set(race_days)):
        try:
            dd = date.fromisoformat(d)
        except ValueError:
            continue
        if lo <= dd <= hi:
            out.append(d.replace("-", ""))
    return out


def missing_pedigree_horses(horse_ids: Iterable[str], base_dir: Path) -> list[str]:
    ids = sorted({str(h).strip() for h in horse_ids if h and str(h).strip().isdigit()})
    return [h for h in ids if not has_ped_tbl(h, base_dir)]


def run(date_strs: list[str], *, base_dir: Path, storage: Any, runner: Any, max_horses: int = MAX_HORSES_PER_RUN, sleep: bool = True) -> dict:
    """date_strs（YYYYMMDD）の出走予定馬のうち ped_tbl が無い馬の血統を取得する。"""
    races: list[str] = []
    for d in date_strs:
        races += race_ids_from_list_json(base_dir / "data" / "page_reference" / "race_lists" / f"{d}.json")
    horses: set[str] = set()
    no_card = 0
    for rid in races:
        card = storage.load("race_shutuba", rid)
        if not card:
            no_card += 1
            continue
        horses.update(str(e.get("horse_id")) for e in (card.get("entries") or []) if isinstance(e, dict) and e.get("horse_id"))
    missing = missing_pedigree_horses(horses, base_dir)
    stats: dict = {"dates": date_strs, "races": len(races), "races_without_card": no_card, "horses": len(horses),
                   "missing": len(missing), "fetched": 0, "failed": [], "deferred": 0}
    for i, h in enumerate(missing[:max_horses]):
        try:
            rec = runner.scrape_horse_pedigree_5gen(h, skip_existing=True)
            if rec:
                stats["fetched"] += 1
            else:
                stats["failed"].append(h)
        except Exception as e:  # 1 頭の失敗で止めない（ブロック疑いの例外は呼び出し側のログに残る）
            logger.error("血統取得エラー [%s]: %s", h, e)
            stats["failed"].append(h)
        if sleep and i < len(missing) - 1:
            time.sleep(random.uniform(2.0, 4.0))
    stats["deferred"] = max(0, len(missing) - max_horses)
    stats["status"] = "ok" if not stats["failed"] else "partial"
    return stats
