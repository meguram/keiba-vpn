"""
2020–2025（任意年）の race_result_flat に出る全 horse_id が、
``data/local/horse_pedigree_5gen`` に JSON として存在し ancestors>=5 であることを検証する。

不足分は ``--backfill-gcs`` で HybridStorage.load → ローカルミラーへ書き込み（GCS にある場合）。

例::

    python3 -m src.research.pedigree.verify_race_horses_pedigree_local
    python3 -m src.research.pedigree.verify_race_horses_pedigree_local --backfill-gcs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TABLES = Path("data/local/tables")
DEFAULT_PED = Path("data/local/horse_pedigree_5gen")


def ped_json_path(ped_root: Path, horse_id: str) -> Path:
    h = (horse_id or "").strip()
    sub = h[:4] if len(h) >= 4 else "_"
    return ped_root / sub / f"{h}.json"


def collect_race_horse_ids(tables_dir: Path, years: list[int]) -> set[str]:
    out: set[str] = set()
    for y in years:
        p = tables_dir / str(y) / "race_result_flat.parquet"
        if not p.exists():
            logger.warning("無い: %s", p)
            continue
        tbl = pq.read_table(p, columns=["horse_id"])
        for x in tbl.column(0).to_pylist():
            s = str(x or "").strip()
            if s:
                out.add(s)
    return out


def _parse_years(s: str | None) -> list[int] | None:
    if s is None or not str(s).strip():
        return None
    out: list[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out or None


def verify(
    *,
    tables_dir: Path,
    ped_root: Path,
    min_ancestors: int,
    years: list[int] | None,
) -> tuple[set[str], list[str], list[str]]:
    """戻り値: (全ID, ファイル無し, ancestors 不足)"""
    if years is None:
        years_use = sorted(
            int(d.name) for d in tables_dir.iterdir() if d.is_dir() and d.name.isdigit()
        )
    else:
        years_use = list(years)
    ids = collect_race_horse_ids(tables_dir, years_use)
    missing: list[str] = []
    weak: list[str] = []
    for hid in sorted(ids):
        path = ped_json_path(ped_root, hid)
        if not path.is_file():
            missing.append(hid)
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(hid)
            continue
        if len(rec.get("ancestors") or []) < min_ancestors:
            weak.append(hid)
    return ids, missing, weak


def backfill_from_gcs(ped_root: Path, horse_ids: list[str], *, min_ancestors: int) -> tuple[int, list[str]]:
    from src.scraper.storage import HybridStorage

    storage = HybridStorage(".")
    n = 0
    still: list[str] = []
    for hid in horse_ids:
        rec = storage.load("horse_pedigree_5gen", hid)
        if rec and len(rec.get("ancestors") or []) >= min_ancestors:
            path = ped_json_path(ped_root, hid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            n += 1
        else:
            still.append(hid)
    return n, still


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="race_result 出走馬のローカル血統 JSON 検証")
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    ap.add_argument("--pedigree-dir", type=Path, default=DEFAULT_PED)
    ap.add_argument("--min-ancestors", type=int, default=5)
    ap.add_argument(
        "--years",
        type=str,
        default="2020,2021,2022,2023,2024,2025",
        help="カンマ区切り。空にすると tables 内の全年",
    )
    ap.add_argument(
        "--backfill-gcs",
        action="store_true",
        help="不足分を HybridStorage 経由で取得し pedigree-dir に書く",
    )
    args = ap.parse_args()
    years_f = _parse_years(args.years)

    all_ids, missing, weak = verify(
        tables_dir=args.tables_dir,
        ped_root=args.pedigree_dir,
        min_ancestors=args.min_ancestors,
        years=years_f,
    )
    logger.info("race_result ユニーク馬: %d", len(all_ids))
    logger.info("ローカル JSON 欠: %d / ancestors<%d: %d", len(missing), args.min_ancestors, len(weak))

    if args.backfill_gcs and missing:
        filled, still = backfill_from_gcs(
            args.pedigree_dir, missing, min_ancestors=args.min_ancestors
        )
        logger.info("GCS から補完: %d 件 / 未取得のまま: %d", filled, len(still))
        if still:
            logger.error("未取得のままの horse_id 例: %s", still[:30])
            sys.exit(2)
        all_ids, missing, weak = verify(
            tables_dir=args.tables_dir,
            ped_root=args.pedigree_dir,
            min_ancestors=args.min_ancestors,
            years=years_f,
        )

    if missing or weak:
        logger.error("検証失敗: 欠=%d 弱=%d 例=%s", len(missing), len(weak), (missing + weak)[:20])
        sys.exit(1)
    logger.info("検証 OK: 全 %d 頭が %s に血統 JSON（ancestors>=%d）",
                len(all_ids), args.pedigree_dir, args.min_ancestors)


if __name__ == "__main__":
    main()
