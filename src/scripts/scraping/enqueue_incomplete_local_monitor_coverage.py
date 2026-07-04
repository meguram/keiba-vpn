#!/usr/bin/env python3
"""
モニター scrape-summary と同じカテゴリ集合について、ローカル実ファイル（ミラー / L2 / レガシー）
の存在だけで欠損を数え、100% 未満の開催日を date_all ジョブとしてキューに投入する。

対象年: 2020–2026。開催日は include_date_in_monitor_summary でフィルタ。

--dry-run で投入せず件数のみ表示。

事前回避: race_lists 無し・モニター対象外・JRA0 の日はスキップ。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

_SUMMARY_SOURCES = [
    "race_shutuba",
    "race_result",
    "race_index",
    "race_odds",
    "race_pair_odds",
    "race_paddock",
    "race_barometer",
    "race_trainer_comment",
]


def _stems_in_dir(d: Path, *, min_bytes: int = 4) -> set[str]:
    if not d.is_dir():
        return set()
    out: set[str] = set()
    for p in d.glob("*.json"):
        try:
            if p.is_file() and p.stat().st_size >= min_bytes:
                out.add(p.stem)
        except OSError:
            continue
    return out


def _local_stems_for_cat_year(base: Path, cat: str, year: str) -> set[str]:
    """その年のレースID stem 集合（ミラー・キャッシュ・レガシー flat）。"""
    s: set[str] = set()
    s |= _stems_in_dir(base / "data" / "local" / "mirror" / cat / year)
    s |= _stems_in_dir(base / "data" / "cache" / cat / year)
    leg = base / "data" / "local" / cat
    if leg.is_dir():
        for p in leg.glob(f"{year}*.json"):
            try:
                if p.is_file() and p.stat().st_size >= 4:
                    s.add(p.stem)
            except OSError:
                continue
    return s


def _iter_target_dates_local(storage, ylo: int, yhi: int) -> list[str]:
    stems = sorted(storage.list_keys("race_lists"))
    out: list[str] = []
    for s in stems:
        if len(s) < 4 or not s[:4].isdigit():
            continue
        y = int(s[:4])
        if y < ylo or y > yhi:
            continue
        out.append(s)
    return out


def _jra_race_ids_for_date(storage, date_key: str) -> tuple[list[str] | None, str]:
    from src.scraper.monitor_future_eligible import include_date_in_monitor_summary
    from src.scraper.missing_races import is_jra_race_id

    rl = storage.load("race_lists", date_key)
    if not rl:
        return None, "no_race_list"
    raw = rl.get("races") or []
    meta = rl.get("_meta") if isinstance(rl, dict) else None
    if not include_date_in_monitor_summary(date_key, raw, meta):
        return [], "excluded"
    jra = [str(r["race_id"]) for r in raw if r.get("race_id") and is_jra_race_id(str(r["race_id"]))]
    if not jra:
        return [], "no_jra"
    return jra, "ok"


def _filled_from_index(jra: list[str], cat_index: dict[str, set[str]]) -> int:
    filled = 0
    for rid in jra:
        for cat in _SUMMARY_SOURCES:
            if rid in cat_index.get(cat, ()):
                filled += 1
    return filled


def main() -> int:
    from src.utils.project_env import find_project_root, load_project_dotenv

    load_project_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year-from", type=int, default=2020)
    ap.add_argument("--year-to", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-smart-skip", action="store_true", help="smart_skip=False で投入")
    args = ap.parse_args()
    ylo = min(args.year_from, args.year_to)
    yhi = max(args.year_from, args.year_to)

    from src.scraper.job_queue import ScrapeJobQueue, kick_process_queue_background
    from src.scraper.storage import HybridStorage

    base = find_project_root()
    storage = HybridStorage(str(base))
    dates = _iter_target_dates_local(storage, ylo, yhi)

    # 年ごとにローカル stem 集合を構築（モニター8カテゴリ）
    year_indices: dict[str, dict[str, set[str]]] = {}
    for y in range(ylo, yhi + 1):
        ys = str(y)
        year_indices[ys] = {cat: _local_stems_for_cat_year(Path(base), cat, ys) for cat in _SUMMARY_SOURCES}

    incomplete: list[tuple[str, int, int, float]] = []
    skipped_no_rl = 0
    skipped_filter = 0
    skipped_no_jra = 0

    for d in dates:
        jra, st = _jra_race_ids_for_date(storage, d)
        if st == "no_race_list":
            skipped_no_rl += 1
            continue
        if st == "excluded":
            skipped_filter += 1
            continue
        if st == "no_jra":
            skipped_no_jra += 1
            continue
        assert jra is not None
        ys = d[:4]
        cat_index = year_indices.get(ys) or {}
        n_cats = len(_SUMMARY_SOURCES)
        total = len(jra) * n_cats
        filled = _filled_from_index(jra, cat_index)
        if total <= 0:
            continue
        pct = 100.0 * filled / total
        if pct < 99.999:
            incomplete.append((d, filled, total, pct))

    print(
        f"対象開催日キー数(年{ylo}-{yhi}): {len(dates)} / "
        f"ローカル未完了日: {len(incomplete)} "
        f"(race_lists 無し: {skipped_no_rl}, モニター対象外: {skipped_filter}, JRA0: {skipped_no_jra})",
        flush=True,
    )
    for row in incomplete[:40]:
        print(f"  {row[0]}  {row[1]}/{row[2]}  ({row[3]:.1f}%)", flush=True)
    if len(incomplete) > 40:
        print(f"  ... 他 {len(incomplete) - 40} 日", flush=True)

    if args.dry_run:
        print("[dry-run] キュー投入をスキップしました", flush=True)
        return 0

    smart_skip = not args.no_smart_skip
    q = ScrapeJobQueue()
    created = requeued = duplicate = 0
    for d, filled, total, pct in sorted(incomplete, key=lambda x: x[0], reverse=True):
        job_spec = {
            "job_kind": "date",
            "target_id": d,
            "tasks": ["date_all"],
            "smart_skip": smart_skip,
        }
        r = q.add_job(job_spec)
        act = r.get("action", "created")
        if act == "created":
            created += 1
        elif act == "requeued":
            requeued += 1
        else:
            duplicate += 1

    kick_process_queue_background()
    print(
        f"投入完了: created={created} requeued={requeued} duplicate={duplicate} "
        f"(smart_skip={smart_skip})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
