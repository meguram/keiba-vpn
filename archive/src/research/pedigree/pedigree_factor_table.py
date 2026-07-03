"""
horse_id と血統因子ブレンド（6スロット）を 1 行にまとめたテーブルを生成する。

目的
----
- 本来の理想に近づける: **1 回の join（horse_id）**で道悪・ダート等の軸（``wet_turf``, ``dirt`` 等）を参照し、
  馬場傾向特徴と同じバッチで合成できる。
- 実行時は ``storage.load`` を馬頭数回しない。事前に Parquet を焼いて LayerA や推論で ``merge`` する。

データの限界
------------
- 血統表は現状 **5 世代スクレイプ**（``horse_pedigree_5gen``）が前提。スロットは gen≤5 の母系深部まで。
- 真の「10 世代 HTML」が揃うまでは、**拡張血統が入ったら本テーブルを再ビルド**すればよい。

出力
----
- 既定: ``data/modeling/horse_pedigree_factors.parquet``
- メタ: ``data/modeling/horse_pedigree_factors.meta.json``

Usage::

  python3 -m src.research.pedigree.pedigree_factor_table --out data/modeling/horse_pedigree_factors.parquet

  # 種牡馬因子 JSON を明示
  python3 -m src.research.pedigree.pedigree_factor_table --stats data/research/sire_factor_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "data/local/modeling/horse_pedigree_factors.parquet"
DEFAULT_META = ROOT / "data/local/modeling/horse_pedigree_factors.meta.json"
DEFAULT_STATS = ROOT / "data/local/research/sire_factor_stats.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.keiba_logging import script_basic_config  # noqa: E402


def build_pedigree_factor_dataframe(
    ped_index: dict[str, dict],
    stats_data: dict[str, Any],
) -> pd.DataFrame:
    from src.research.pedigree.sire_factor_aptitude import (
        AXIS_IDS,
        compute_blended_factors_from_ancestors,
        count_resolved_slots,
    )

    rows: list[dict[str, Any]] = []
    for hid, rec in ped_index.items():
        hid = (hid or "").strip()
        if not hid:
            continue
        ancestors = rec.get("ancestors") or []
        if not ancestors:
            continue
        fac = compute_blended_factors_from_ancestors(ancestors, stats_data)
        nslot = count_resolved_slots(ancestors, stats_data)
        row: dict[str, Any] = {"horse_id": hid, "ped_n_slots_resolved": nslot}
        for k in AXIS_IDS:
            row[f"ped_{k}"] = fac.get(k, 0.0)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["horse_id", "ped_n_slots_resolved"])

    return pd.DataFrame(rows)


def run_build(
    *,
    stats_path: Path,
    out_parquet: Path,
    meta_path: Path,
    storage: Any = None,
) -> dict[str, Any]:
    from src.research.pedigree.pedigree_local_store import load_full_pedigree_index
    from src.research.pedigree.sire_factor_stats import load_sire_factor_stats

    t0 = time.time()
    stats_data = load_sire_factor_stats(stats_path, storage=storage)
    ped_index = load_full_pedigree_index(storage)
    if not ped_index:
        raise RuntimeError(
            "血統インデックスが空です。data/research/_ped_snapshot_cache.jsonl.gz を用意するか、"
            "GCS からスナップショットを取得できる環境で実行してください。"
        )

    df = build_pedigree_factor_dataframe(ped_index, stats_data)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    meta: dict[str, Any] = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_parquet": str(out_parquet),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "stats_source": str(stats_path),
        "pedigree_source": "pedigree_local_store / horse_pedigree_5gen snapshot",
        "note": (
            "6スロット加権ブレンド（sire_factor_aptitude）。"
            "馬場関連は ped_wet_turf, ped_dirt 等。LayerA では horse_id で left merge。"
        ),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = round(time.time() - t0, 1)
    logger.info(
        "書き出し: %s (%d 行, %.1fs)",
        out_parquet,
        len(df),
        elapsed,
    )
    meta["elapsed_sec"] = elapsed
    return meta


def load_pedigree_factors_table(path: Path | str | None = None) -> pd.DataFrame:
    """推論・母表 merge 用に Parquet を読み込む。"""
    p = Path(path) if path else DEFAULT_OUT
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_parquet(p)


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="horse_id × 血統因子ブレンド Parquet を生成")
    ap.add_argument("--stats", type=Path, default=DEFAULT_STATS, help="sire_factor_stats.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    args = ap.parse_args()

    storage = None
    try:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage(str(ROOT))
    except Exception as e:
        logger.info("HybridStorage なし（ローカル gz のみ）: %s", e)

    try:
        meta = run_build(
            stats_path=args.stats,
            out_parquet=args.out,
            meta_path=args.meta,
            storage=storage,
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.exception("ビルド失敗")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
