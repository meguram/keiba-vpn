"""馬ID → 父・母父の対応表（horse_sire_map.parquet）を作る。めぐ分析の『種牡馬×コース』が読む。

python -m src.scripts.data.build_horse_sire_map            # ホストで実行。差分だけ読む（既存の行は読み直さない）
python -m src.scripts.data.build_horse_sire_map --full     # 全頭を読み直す

対象の馬: race_result_flat（2020 年以降）に出た馬 + 今日から lookahead 日先までの出馬表（race_shutuba）の馬。
元データ: data/local/horse_pedigree_5gen/<先頭4文字>/<horse_id>.json の ancestors（世代1・位置0 = 父、世代2・位置2 = 母父）。
キーは ID（名前では結合しない）。血統 JSON がまだ無い馬は行を作らず、次回の実行で拾う。
出力: data/page_reference/pedigree_race_index/horse_sire_map.parquet（pedigree_race_index は ops/mirror で Cloud Run にも届く）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("build_horse_sire_map")

ROOT = Path(__file__).resolve().parents[3]
PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
FLAT_DIR = ROOT / "data/page_reference/tables"
OUT = ROOT / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
COLUMNS = ["horse_id", "sire_id", "sire_name", "dam_sire_id", "dam_sire_name"]


def parse_pedigree(doc: dict[str, Any]) -> dict[str, Any] | None:
    """血統 JSON から父・母父の ID と名前を取る。父の ID が取れなければ None。"""
    pick: dict[tuple[int, int], dict] = {}
    for a in doc.get("ancestors") or []:
        try:
            pick[(int(a["generation"]), int(a["position"]))] = a
        except (KeyError, TypeError, ValueError):
            continue
    sire, dam_sire = pick.get((1, 0)), pick.get((2, 2))
    if not sire or not sire.get("horse_id"):
        return None
    return {
        "horse_id": str(doc.get("horse_id") or ""),
        "sire_id": str(sire["horse_id"]),
        "sire_name": sire.get("name") or doc.get("sire"),
        "dam_sire_id": str(dam_sire["horse_id"]) if dam_sire and dam_sire.get("horse_id") else None,
        "dam_sire_name": (dam_sire.get("name") if dam_sire else None) or doc.get("dam_sire"),
    }


def _read(horse_id: str, ped_dir: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads((ped_dir / horse_id[:4] / f"{horse_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    row = parse_pedigree(doc)
    if row:
        row["horse_id"] = horse_id
    return row


def result_horse_ids(flat_dir: Path = FLAT_DIR) -> set[str]:
    ids: set[str] = set()
    for f in sorted(flat_dir.glob("20*/race_result_flat.parquet")):
        ids |= set(pd.read_parquet(f, columns=["horse_id"])["horse_id"].astype(str))
    return ids


def entry_horse_ids(storage: Any, today: date, lookahead_days: int) -> set[str]:
    """今日から lookahead 日先までの出馬表の馬。"""
    ids: set[str] = set()
    for i in range(lookahead_days + 1):
        ymd = (today + timedelta(days=i)).strftime("%Y%m%d")
        try:
            races = (storage.load("race_lists", ymd) or {}).get("races") or []
        except Exception:
            logger.exception("race_lists の読み込みに失敗: %s", ymd)
            continue
        for r in races:
            rid = str(r.get("race_id") or "")
            if not rid:
                continue
            try:
                card = storage.load("race_shutuba", rid) or {}
            except Exception:
                continue
            ids |= {str(e["horse_id"]) for e in card.get("entries") or [] if isinstance(e, dict) and e.get("horse_id")}
    return ids


def build(out: Path = OUT, ped_dir: Path = PED_DIR, horse_ids: set[str] | None = None, *, full: bool = False, workers: int = 8) -> dict[str, int]:
    ids = horse_ids if horse_ids is not None else result_horse_ids()
    old = pd.DataFrame(columns=COLUMNS)
    if out.exists() and not full:
        old = pd.read_parquet(out)
    todo = sorted(ids - set(old["horse_id"].astype(str)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(lambda h: _read(h, ped_dir), todo) if r]
    new = pd.DataFrame(rows, columns=COLUMNS)
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates("horse_id", keep="last").sort_values("horse_id").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return {"targets": len(ids), "read": len(todo), "added": len(new), "no_pedigree": len(todo) - len(new), "total": len(merged)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="既存の行も読み直す")
    ap.add_argument("--lookahead-days", type=int, default=14)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from src.scraper.storage import HybridStorage

    ids = result_horse_ids() | entry_horse_ids(HybridStorage(), date.today(), args.lookahead_days)
    res = build(args.out, horse_ids=ids, full=args.full)
    logger.info("horse_sire_map: %s", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
