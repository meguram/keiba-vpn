"""
出馬表などで得た ``horse_id`` リストに対し、ローカル ``horse_pedigree_5gen`` JSON から
``data/features/horse/ped_tbl`` の Parquet を **欠けている分だけ** 生成する。

- ``build_horse_entity_store`` と同じ flatten / merge / クロス列ロジックを再利用。
- 既存 Parquet が十分なサイズであればスキップ（再生成しない）。

CLI::

  python -m pipeline.sync_ped_tbl_for_horses --horse-ids 2019100001,2020100002
  python -m pipeline.sync_ped_tbl_for_horses --horse-ids-file ids.txt

環境変数（出馬表取得時の自動同期）::

  KEIBA_SYNC_PED_TBL_ON_SHUTUBA=1   # これを設定したときのみ ``race_shutuba`` 保存後に ped_tbl を同期。
  KEIBA_PED_TBL_MERGE_GEN5=1       # 未設定時は 1（5 世代目種牡馬接木）。0 で主表のみ。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pipeline.build_horse_entity_store import (
    _flatten_pedigree_json,
    _flatten_pedigree_json_with_gen5_merge,
    _load_pedigree_json,
)
from pipeline.horse_entity_layout import ped_parquet_path
from pipeline.id_value_policy import sanitize_netkeiba_string_id
from utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

_SYNC_ENV = "KEIBA_SYNC_PED_TBL_ON_SHUTUBA"
_MERGE_ENV = "KEIBA_PED_TBL_MERGE_GEN5"


def _env_flag(name: str, *, default: bool = True) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


def normalize_horse_id_list(horse_ids: Iterable[str]) -> list[str]:
    """空・ゼロ類似を除き、重複を除いた文字列 ID リスト。"""
    s = pd.Series([str(x).strip() for x in horse_ids if x is not None], dtype="string")
    if s.empty:
        return []
    out = sanitize_netkeiba_string_id(s)
    return sorted(out.dropna().astype(str).unique().tolist())


def sync_ped_tbl_for_horses(
    horse_ids: Iterable[str],
    *,
    base_dir: Path | str = ".",
    pedigree_json_dir: Path | None = None,
    merge_gen5_sires: bool | None = None,
    skip_if_parquet_exists: bool = True,
    min_parquet_bytes: int = 32,
) -> dict[str, Any]:
    """
    各 ``horse_id`` について ``ped_tbl`` Parquet が無ければローカル JSON から生成する。

    Returns:
        requested, written, skipped_existing_parquet, missing_json, errors
    """
    base = Path(base_dir)
    ped_dir = Path(pedigree_json_dir) if pedigree_json_dir else base / "data" / "local" / "horse_pedigree_5gen"
    if merge_gen5_sires is None:
        merge_gen5_sires = _env_flag(_MERGE_ENV, default=True)

    ids = normalize_horse_id_list(horse_ids)
    stats: dict[str, Any] = {
        "requested": len(ids),
        "written": 0,
        "skipped_existing_parquet": 0,
        "missing_json": 0,
        "errors": 0,
        "merge_gen5_sires": bool(merge_gen5_sires),
        "pedigree_json_dir": str(ped_dir),
    }

    if not ped_dir.is_dir():
        logger.warning("pedigree ディレクトリがありません: %s", ped_dir)
        return stats

    for hid in ids:
        outp = ped_parquet_path(hid, base)
        try:
            if skip_if_parquet_exists and outp.is_file() and outp.stat().st_size >= min_parquet_bytes:
                stats["skipped_existing_parquet"] += 1
                continue
            rec = _load_pedigree_json(ped_dir, hid)
            if not rec:
                stats["missing_json"] += 1
                continue
            if merge_gen5_sires:
                pdf = _flatten_pedigree_json_with_gen5_merge(rec, hid, ped_dir)
            else:
                pdf = _flatten_pedigree_json(rec, hid)
            outp.parent.mkdir(parents=True, exist_ok=True)
            pdf.to_parquet(outp, index=False)
            stats["written"] += 1
        except Exception as e:
            stats["errors"] += 1
            logger.warning("ped_tbl 生成失敗 %s: %s", hid, e)
    return stats


def sync_ped_tbl_after_shutuba(
    shutuba_data: dict[str, Any],
    *,
    base_dir: Path | str,
    pedigree_json_dir: Path | None = None,
    merge_gen5_sires: bool | None = None,
) -> dict[str, Any] | None:
    """
    ``race_shutuba`` 保存直後用。entries から ``horse_id`` を集め ``sync_ped_tbl_for_horses`` を呼ぶ。

    環境変数 ``KEIBA_SYNC_PED_TBL_ON_SHUTUBA=1`` のときのみ実行（未設定はオフ）。
    """
    if not _env_flag(_SYNC_ENV, default=False):
        return None
    entries = shutuba_data.get("entries") or []
    raw_ids = [e.get("horse_id") for e in entries if isinstance(e, dict)]
    if not raw_ids:
        return {"requested": 0, "written": 0, "skipped_existing_parquet": 0, "missing_json": 0, "errors": 0}
    return sync_ped_tbl_for_horses(
        raw_ids,
        base_dir=base_dir,
        pedigree_json_dir=pedigree_json_dir,
        merge_gen5_sires=merge_gen5_sires,
        skip_if_parquet_exists=True,
    )


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="欠損している ped_tbl Parquet をローカル JSON から生成")
    ap.add_argument("--base-dir", type=Path, default=Path("."))
    ap.add_argument("--pedigree-json-dir", type=Path, default=None)
    ap.add_argument("--horse-ids", type=str, default="", help="カンマ区切りの horse_id")
    ap.add_argument("--horse-ids-file", type=Path, default=None, help="1行1頭の horse_id ファイル")
    ap.add_argument("--no-merge-gen5", action="store_true", help="5世代目種牡馬接木をしない")
    ap.add_argument("--force", action="store_true", help="既存 Parquet があっても上書き")
    args = ap.parse_args()

    ids: list[str] = []
    if args.horse_ids.strip():
        ids.extend(re.split(r"[\s,]+", args.horse_ids.strip()))
    if args.horse_ids_file and args.horse_ids_file.is_file():
        for line in args.horse_ids_file.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                ids.append(t)
    if not ids:
        logger.error("--horse-ids または --horse-ids-file を指定してください")
        return 1

    st = sync_ped_tbl_for_horses(
        ids,
        base_dir=args.base_dir,
        pedigree_json_dir=args.pedigree_json_dir,
        merge_gen5_sires=not args.no_merge_gen5,
        skip_if_parquet_exists=not args.force,
    )
    logger.info(
        "ped_tbl 同期: requested=%d written=%d skipped_existing=%d missing_json=%d errors=%d",
        st["requested"],
        st["written"],
        st["skipped_existing_parquet"],
        st["missing_json"],
        st["errors"],
    )
    return 1 if st["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
