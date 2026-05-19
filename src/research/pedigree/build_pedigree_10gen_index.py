"""
直近 10 世代までの「父系領域 / 母系領域」祖先インデックスを構築する。

既存 ``data/local/horse_pedigree_5gen/{prefix}/{horse_id}.json`` には各馬の 5 世代分
(最大 62 頭) の祖先が格納されている。さらに「種牡馬クロージャ」展開によって
血統表に登場する祖先たち自身の 5 世代血統 JSON も同じディレクトリに揃っている。
このため **追加スクレイピング無しで 5gen × 5gen = 10gen** を再構築できる。

本スクリプトは race_records.parquet に出走履歴のある全馬を対象に、

    horse_id, ancestor_id, side ('father'|'mother')

の long format を ``data/research/pedigree_10gen/horse_ancestor_long.parquet`` に書き出す。

側面 (side) の判定:
    各 ancestor の generation g, position p (0-indexed) について最上位ビット
    ``(p >> (g-1)) & 1`` が 0 なら父系領域、1 なら母系領域。
    例: gen=1 pos=0 → 父系 / gen=2 pos=2 → 母系 (=母父)。
    5gen 目祖先 X の祖先 (= 元馬から見た 6-10 gen) は、X が父系か母系かを
    そのまま継承する (元馬の position の最上位ビット = X の最上位ビット)。

Usage:
    python -m src.research.pedigree.build_pedigree_10gen_index
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.pedigree.pedigree_index_long_sink import LongParquetSink  # noqa: E402
from src.research.pedigree.pedigree_local_store import (  # noqa: E402
    load_full_pedigree_index,
)

logger = logging.getLogger(__name__)

PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
RACE_REC = ROOT / "data/research/bloodline_meta_cluster/race_records.parquet"
OUT_DIR = ROOT / "data/research/pedigree_10gen"
OUT_LONG = OUT_DIR / "horse_ancestor_long.parquet"
OUT_INV = OUT_DIR / "ancestor_to_horses.parquet"
OUT_NAME = OUT_DIR / "ancestor_id_to_name.parquet"
OUT_META = OUT_DIR / "meta.json"


def _expand_10gen(horse_id: str, idx: dict[str, dict]) -> tuple[set[str], set[str]]:
    """horse_id の祖先を直近 10 世代まで再帰展開し、father / mother 集合を返す。

    Returns:
        (father_ancestor_ids, mother_ancestor_ids)
        - 自身 (horse_id) は含めない。
        - インブリードや上下世代重複は set で 1 度のみカウント。
    """
    rec = idx.get(horse_id)
    father: set[str] = set()
    mother: set[str] = set()
    if not rec:
        return father, mother

    for anc in rec.get("ancestors") or []:
        hid = str(anc.get("horse_id") or "").strip()
        if not hid:
            continue
        try:
            g = int(anc.get("generation", 0))
            p = int(anc.get("position", -1))
        except (TypeError, ValueError):
            continue
        if g < 1 or p < 0:
            continue
        is_father_side = ((p >> (g - 1)) & 1) == 0
        target = father if is_father_side else mother
        target.add(hid)
        if g == 5:
            sub = idx.get(hid)
            if not sub:
                continue
            for sub_anc in sub.get("ancestors") or []:
                sub_hid = str(sub_anc.get("horse_id") or "").strip()
                if sub_hid:
                    target.add(sub_hid)
    return father, mother


def build(
    *,
    out_long: Path = OUT_LONG,
    out_inv: Path = OUT_INV,
    out_name: Path = OUT_NAME,
    out_meta: Path = OUT_META,
    race_rec_path: Path = RACE_REC,
    ped_dir: Path = PED_DIR,
    limit: int | None = None,
    quiet: bool = False,
    long_buffer_rows: int = 800_000,
) -> dict:
    """全競走馬の 10 世代祖先インデックスを構築して parquet に書き出す。"""
    out_long.parent.mkdir(parents=True, exist_ok=True)

    if not race_rec_path.exists():
        raise FileNotFoundError(
            f"race_records.parquet が無い: {race_rec_path}\n"
            "  -> python -m src.research.pedigree.build_meta_cluster_artifacts を実行してください"
        )

    if not quiet:
        print(f"[10gen] race_records から horse_id 一覧を抽出: {race_rec_path}")
    rec = pd.read_parquet(race_rec_path, columns=["horse_id"])
    rec["horse_id"] = rec["horse_id"].astype(str)
    horse_ids = sorted(set(rec["horse_id"]))
    if limit:
        horse_ids = horse_ids[:limit]
    if not quiet:
        print(f"[10gen] 対象 horse_id 数 = {len(horse_ids):,}")

    if not quiet:
        print(f"[10gen] pedigree index を構築 (5gen JSON ロード): {ped_dir}")
    t0 = time.time()
    idx = load_full_pedigree_index(None, path=ped_dir)
    if not quiet:
        print(f"[10gen] index 構築完了: {len(idx):,} 頭 ({time.time()-t0:.1f}s)")

    sink = LongParquetSink(out_long, max_buffer_rows=long_buffer_rows)
    n_with_data = 0
    n_father_total = 0
    n_mother_total = 0
    t0 = time.time()
    for i, hid in enumerate(horse_ids):
        father, mother = _expand_10gen(hid, idx)
        if father or mother:
            n_with_data += 1
        n_father_total += len(father)
        n_mother_total += len(mother)
        for a in father:
            sink.append_triplet(hid, a, "father")
        for a in mother:
            sink.append_triplet(hid, a, "mother")
        if not quiet and (i + 1) % 5000 == 0:
            print(f"[10gen]  {i+1:,}/{len(horse_ids):,} ({(i+1)/len(horse_ids)*100:.1f}%) "
                  f"rows={sink.total_rows:,} elapsed={time.time()-t0:.1f}s")

    if not quiet:
        print(f"[10gen] 展開完了: {n_with_data:,} 馬にデータあり / "
              f"father 平均 {n_father_total/max(1,n_with_data):.1f} / "
              f"mother 平均 {n_mother_total/max(1,n_with_data):.1f}")

    sink.close()
    n_rows_long = sink.total_rows
    del sink
    gc.collect()

    if not quiet:
        print(f"[10gen] 書き出し: {out_long} ({n_rows_long:,} rows)")

    if not quiet:
        print(f"[10gen] inverted index を構築 ancestor_id × side -> horses")
    inv_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    pf = pq.ParquetFile(out_long)
    for rgi in range(pf.num_row_groups):
        chunk = pf.read_row_group(rgi, columns=["horse_id", "ancestor_id", "side"])
        d = chunk.to_pydict()
        for h, a, s in zip(d["horse_id"], d["ancestor_id"], d["side"]):
            inv_map[(a, s)].append(h)
        del chunk, d
        gc.collect()
    inv_rows = [
        {
            "ancestor_id": k[0],
            "side": k[1],
            "n_horses": len(v),
            "horse_ids": v,
        }
        for k, v in inv_map.items()
    ]
    del inv_map
    gc.collect()
    df_inv = pd.DataFrame(inv_rows)
    del inv_rows
    gc.collect()
    if not quiet:
        print(f"[10gen] 書き出し: {out_inv} (ancestor_id 数 = {df_inv['ancestor_id'].nunique():,})")
    df_inv.to_parquet(out_inv, index=False, compression="zstd")

    # ── ancestor_id → name マップ (各 5gen JSON の ancestors[].name から抽出) ──
    if not quiet:
        print(f"[10gen] ancestor_id -> name マップを構築 (idx 全件から)")
    name_map: dict[str, str] = {}
    for hid, rec in idx.items():
        # 自身 (horse_id) の name は idx には記録されないので親キーで。idx の構造は
        # {horse_id: {ancestors: [...], sire, dam, dam_sire, ...}}
        # ancestors の各 entry: {generation, position, name, horse_id, side}
        for anc in (rec.get("ancestors") or []):
            ah = str(anc.get("horse_id") or "").strip()
            an = (anc.get("name") or "").strip()
            if ah and an and ah not in name_map:
                name_map[ah] = an
    # 自身の name (sire/dam ヘッダから取れる範囲)
    # idx[hid] のトップレベルに "name" がある場合に対応
    for hid, rec in idx.items():
        nm = (rec.get("name") or "").strip() if isinstance(rec, dict) else ""
        if nm and hid not in name_map:
            name_map[hid] = nm

    df_name = pd.DataFrame({
        "horse_id": list(name_map.keys()),
        "name": list(name_map.values()),
    })
    df_name["horse_id"] = df_name["horse_id"].astype("string")
    df_name["name"] = df_name["name"].astype("string")
    if not quiet:
        print(f"[10gen] 書き出し: {out_name} ({len(df_name):,} 件)")
    df_name.to_parquet(out_name, index=False, compression="zstd")

    import json as _json
    meta = {
        "n_horses_target": len(horse_ids),
        "n_horses_with_data": n_with_data,
        "n_rows": int(n_rows_long),
        "n_unique_ancestors": int(df_inv["ancestor_id"].nunique()),
        "n_ancestor_names": int(len(df_name)),
        "father_avg_per_horse": n_father_total / max(1, n_with_data),
        "mother_avg_per_horse": n_mother_total / max(1, n_with_data),
        "out_long": str(out_long),
        "out_inv": str(out_inv),
        "out_name": str(out_name),
        "ped_dir": str(ped_dir),
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    out_meta.write_text(_json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(f"[10gen] meta: {out_meta}")
        print(f"[10gen] DONE")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭 N 頭で動作確認 (デフォルト: 全件)")
    parser.add_argument(
        "--long-buffer-rows",
        type=int,
        default=800_000,
        help="long 表を Parquet に追記するまでメモリに溜める最大行数 (既定 800000)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(limit=args.limit, quiet=args.quiet, long_buffer_rows=args.long_buffer_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
