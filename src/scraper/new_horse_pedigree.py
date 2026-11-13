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
# 1 回（1 日 1 回）に既存を確かめる上限（GCS の 5 世代表と血統 HTML の有無を、馬 ID の名前指定で読む。一覧はしない）。
# 飛ばす経路でも、祖先の系統（台帳に無い祖先）や種付けのページを取りに行くことがある。そのときは通信の回数（client の数）で
# 見分けて max_horses に数える。確かめる頭数そのものもこの上限で抑える。
# 費用（最悪・月。VPS は GCP の外）: 600 頭 × 読み 2〜3 回 × 30 ≒ 5.4 万回 → 約 3 円、600 × 約 5KB × 30 ≒ 90MB → 転送 約 2 円。
MAX_CHECKS_PER_RUN = 600
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


def run(date_strs: list[str], *, base_dir: Path, storage: Any, runner: Any, max_horses: int = MAX_HORSES_PER_RUN, max_checks: int = MAX_CHECKS_PER_RUN, sleep: bool = True) -> dict:
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
                   "missing": len(missing), "fetched": 0, "skipped_existing": 0, "failed": [], "deferred": 0}
    # 上限（max_horses）は、実際に血統ページを取りに行った頭数で数える。GCS に 5 世代表が既にあって飛ばした馬は数えない
    # （VPS には手元の ped_tbl が無く全馬が missing に見えるため、飛ばした馬で上限を使い切ると本当の新馬が後回しになる）。
    # 既存の確認（GCS の読み）は max_checks 頭まで。残りは翌日の回に回る（deferred）。
    requested = 0
    # 新しい生まれ年（ID が大きい＝新馬）から確かめる。上限で打ち切っても、毎日同じ古い馬だけを見て新馬に届かない、を避ける
    missing = sorted(missing, reverse=True)
    client = getattr(runner, "client", None)

    def _reqs() -> int:
        return int(getattr(client, "_request_count", 0) or 0)

    for i, h in enumerate(missing):
        if requested >= max_horses or i >= max_checks:
            stats["deferred"] = len(missing) - i
            break
        try:
            before = _reqs()
            rec = runner.scrape_horse_pedigree_5gen(h, skip_existing=True)
            if getattr(runner, "_last_pedigree_5gen_skip", False):
                stats["skipped_existing"] += 1
                if _reqs() == before:
                    continue                            # 通信なし（GCS の確認だけ）: 上限に数えず、待たない
                requested += 1                          # 飛ばす経路でも祖先の系統・種付けのページを取った: 上限に数え、間を空ける
                if sleep and i < len(missing) - 1:
                    time.sleep(random.uniform(2.0, 4.0))
                continue
            requested += 1
            if rec:
                stats["fetched"] += 1
            else:
                stats["failed"].append(h)
        except Exception as e:  # 1 頭の失敗で止めない（ブロック疑いの例外は呼び出し側のログに残る）
            requested += 1
            logger.error("血統取得エラー [%s]: %s", h, e)
            stats["failed"].append(h)
        if sleep and i < len(missing) - 1:
            time.sleep(random.uniform(2.0, 4.0))
    stats["status"] = "ok" if not stats["failed"] else "partial"
    return stats
