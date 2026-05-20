"""5gen / 10gen 血統 JSON 全件から ancestor_id → 馬名 のマップを抽出。

`data/local/horse_pedigree_5gen/<prefix>/<horse_id>.json` および
`data/local/horse_pedigree_10gen/<year>/<horse_id>.json` を全件走査し、
`ancestors[]` 内に格納されている `{horse_id, name}` ペアを集約する。

10gen ページや探索モードで母系領域 (10gen) を選んだ際に、
種牡馬名が `horse_id` のまま表示されてしまう問題への対策として、
`art["horse_name_map"]` への補完用マップとして利用する。

出力:
    data/research/pedigree_10gen/ancestor_id_to_name.parquet
        columns: horse_id (string), name (string)

使用:
    python -m src.research.pedigree.build_ancestor_name_map
"""
from __future__ import annotations

import gc
import json
import logging
import time
from glob import glob
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PED_5GEN_DIR = ROOT / "data/local/horse_pedigree_5gen"
PED_10GEN_DIR = ROOT / "data/local/horse_pedigree_10gen"
OUT_PATH = ROOT / "data/local/research/pedigree_10gen/ancestor_id_to_name.parquet"


def _scan_dir(root_dir: Path, name_map: dict[str, str], stats: dict) -> None:
    if not root_dir.exists():
        return
    files = sorted(glob(str(root_dir / "**" / "*.json"), recursive=True))
    stats[f"files_{root_dir.name}"] = len(files)
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            stats["read_errors"] = stats.get("read_errors", 0) + 1
            continue
        # 自身の name (5gen / 10gen フォーマットによってはトップレベルにあるとは限らない)
        own_id = str(data.get("horse_id") or "").strip()
        own_name = str(data.get("name") or "").strip()
        if own_id and own_name and own_id not in name_map:
            name_map[own_id] = own_name
        # ancestors[] からの抽出
        for anc in (data.get("ancestors") or []):
            ah = str(anc.get("horse_id") or "").strip()
            an = str(anc.get("name") or "").strip()
            if ah and an and ah not in name_map:
                name_map[ah] = an
        # sire / dam / dam_sire のヘッダ名 (5gen JSON は name のみで id を持たないことがあるので
        # 補完目的のみ。重複チェックで上書きしない)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  ... %s: %d/%d files (%.1fs) — %d names so far",
                root_dir.name, i + 1, len(files), elapsed, len(name_map),
            )
            gc.collect()
    stats[f"scanned_{root_dir.name}"] = len(files)


def build(out_path: Path = OUT_PATH, *, quiet: bool = False) -> dict:
    if not quiet:
        print(f"[anc_name] scan {PED_5GEN_DIR} + {PED_10GEN_DIR}")
    name_map: dict[str, str] = {}
    stats: dict = {}
    _scan_dir(PED_5GEN_DIR, name_map, stats)
    _scan_dir(PED_10GEN_DIR, name_map, stats)
    if not quiet:
        print(f"[anc_name] 集約完了: {len(name_map):,} 件 (errors={stats.get('read_errors', 0)})")
    df = pd.DataFrame({
        "horse_id": list(name_map.keys()),
        "name": list(name_map.values()),
    })
    df["horse_id"] = df["horse_id"].astype("string")
    df["name"] = df["name"].astype("string")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")
    if not quiet:
        print(f"[anc_name] 書き出し: {out_path} ({len(df):,} rows)")
    stats["n_names"] = len(df)
    stats["out_path"] = str(out_path)
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
