"""直近 10 世代までの 3 視点 (father / dam_sire_line / dam_dam_line) 祖先インデックスを構築。

既存の 2 値 (father/mother) インデックス (``build_pedigree_10gen_index.py``) を細粒度化し、
ユーザー要望:
    ① 父にもつ条件      = side='father'          = 1代目が父である祖先全部 (= 父系領域)
    ② 母父母以降に持つ条件 = side='dam_sire_line'   = 1代目=母 AND 2代目=父 (= 母父系)
    ③ 母母以降に持つ条件   = side='dam_dam_line'    = 1代目=母 AND 2代目=母 (= 母母系)
の 3 視点に対応する。

side 判定 (generation g, position p):
    - g=1, p=0  : father       (= 父そのもの。父系領域に含める)
    - g=1, p=1  : (=母そのもの)  → 'dam' (母父系・母母系どちらにも属さない / 単独枠)
    - g>=2:
        * (p >> (g-1)) & 1 == 0  → 1代目が父 → 'father'
        * (p >> (g-1)) & 1 == 1  → 1代目が母
              ┣ (p >> (g-2)) & 1 == 0 → 2代目が父 → 'dam_sire_line'
              ┗ (p >> (g-2)) & 1 == 1 → 2代目が母 → 'dam_dam_line'

5gen 目 (g=5) の祖先 X の祖先 (= 元馬から見た 6-10gen) は、X の side をそのまま継承する
(X が 'dam_sire_line' なら、X の祖先も全て 'dam_sire_line' 領域)。

Usage:
    python -m src.research.pedigree.build_pedigree_10gen_3view_index [--limit N]

出力:
    data/research/pedigree_10gen_3view/horse_ancestor_long.parquet
        cols: horse_id, ancestor_id, side
    data/research/pedigree_10gen_3view/ancestor_to_horses.parquet
        cols: ancestor_id, side, n_horses, horse_ids
    data/research/pedigree_10gen_3view/meta.json
"""
from __future__ import annotations

import argparse
import gc
import json as _json
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
    expand_10gen_ancestors_from_record,
    load_full_pedigree_index,
)

logger = logging.getLogger(__name__)

PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
PED_10GEN_DIR = ROOT / "data/local/horse_pedigree_10gen"
RACE_REC = ROOT / "data/page_reference/note_aptitude_race/race_records.parquet"
OUT_DIR = ROOT / "data/local/research/pedigree_10gen_3view"
OUT_LONG = OUT_DIR / "horse_ancestor_long.parquet"
OUT_INV = OUT_DIR / "ancestor_to_horses.parquet"
OUT_META = OUT_DIR / "meta.json"


def _load_10gen_json(ped_dir: Path, horse_id: str) -> dict | None:
    """1 頭分の 10gen 統合 JSON を読む (全件辞書に載せないためのヘルパ)。"""
    prefix = horse_id[:4] if len(horse_id) >= 4 else "0000"
    p = ped_dir / prefix / f"{horse_id}.json"
    if not p.exists():
        return None
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, _json.JSONDecodeError):
        return None


def _side_of(g: int, p: int) -> str | None:
    """generation g, position p から 3 視点 side を判定。"""
    if g < 1 or p < 0:
        return None
    is_dam_side = ((p >> (g - 1)) & 1) == 1  # 1代目が母か
    if not is_dam_side:
        return "father"
    if g == 1:
        # 母そのもの (=g=1, p=1)。母父系でも母母系でもないので別扱い
        return "dam"
    is_dam_dam = ((p >> (g - 2)) & 1) == 1  # 2代目が母か
    return "dam_dam_line" if is_dam_dam else "dam_sire_line"


def _expand_10gen_3view(horse_id: str, idx: dict[str, dict]) -> dict[str, set[str]]:
    """horse_id の祖先を直近 10 世代まで再帰展開し、3 視点別の集合を返す。

    Returns:
        {'father': set, 'dam_sire_line': set, 'dam_dam_line': set}
        (g=1 の母そのものは 'dam' 視点。出力には含めない)
    """
    out = {"father": set(), "dam_sire_line": set(), "dam_dam_line": set()}
    rec = idx.get(horse_id)
    if not rec:
        return out

    for anc in rec.get("ancestors") or []:
        hid = str(anc.get("horse_id") or "").strip()
        if not hid:
            continue
        try:
            g = int(anc.get("generation", 0))
            p = int(anc.get("position", -1))
        except (TypeError, ValueError):
            continue
        side = _side_of(g, p)
        if side is None or side == "dam":
            continue
        out[side].add(hid)
        if g == 5:
            # 5gen 祖先 X の祖先 (= 6-10gen) を X の side でカウント
            sub = idx.get(hid)
            if not sub:
                continue
            for sub_anc in sub.get("ancestors") or []:
                sub_hid = str(sub_anc.get("horse_id") or "").strip()
                if sub_hid:
                    out[side].add(sub_hid)
    return out


def build(
    *,
    out_long: Path = OUT_LONG,
    out_inv: Path = OUT_INV,
    out_meta: Path = OUT_META,
    race_rec_path: Path = RACE_REC,
    ped_dir: Path = PED_DIR,
    ped_10gen_dir: Path | None = PED_10GEN_DIR,
    use_10gen_merged: bool = True,
    limit: int | None = None,
    quiet: bool = False,
    long_buffer_rows: int = 800_000,
) -> dict:
    """3 視点祖先インデックスを構築する。

    use_10gen_merged=True かつ ``ped_10gen_dir`` にデータがある場合:
        10gen 統合ファイル ``data/local/horse_pedigree_10gen/{prefix}/{horse_id}.json``
        を **対象馬ごとに** 読み、各レコード内の ``ancestors[].side`` を集計する。
        全頭を ``horse_id -> dict`` に載せないためメモリに優しい。
    そうでない場合:
        従来通り 5gen JSON を再帰展開する (5gen 全件インデックスが必要)。
    """
    out_long.parent.mkdir(parents=True, exist_ok=True)

    if not race_rec_path.exists():
        raise FileNotFoundError(f"race_records.parquet が無い: {race_rec_path}")

    if not quiet:
        print(f"[10gen3v] race_records から horse_id 一覧抽出: {race_rec_path}")
    rec = pd.read_parquet(race_rec_path, columns=["horse_id"])
    rec["horse_id"] = rec["horse_id"].astype(str)
    horse_ids = sorted(set(rec["horse_id"]))
    if limit:
        horse_ids = horse_ids[:limit]
    if not quiet:
        print(f"[10gen3v] 対象 horse_id 数 = {len(horse_ids):,}")

    # データソース選択: 10gen 統合ファイル優先
    using_merged = (
        use_10gen_merged
        and ped_10gen_dir is not None
        and ped_10gen_dir.is_dir()
        and any(ped_10gen_dir.iterdir())
    )
    if not quiet:
        print(f"[10gen3v] データソース: {'10gen 統合 (高速)' if using_merged else '5gen 動的展開'}")

    idx: dict[str, dict] | None = None
    if using_merged:
        if not quiet:
            print("[10gen3v] 10gen 統合: 対象馬の JSON のみ都度ロード (全頭インデックスは作らない)")
    else:
        if not quiet:
            print(f"[10gen3v] pedigree index 構築 (5gen JSON ロード): {ped_dir}")
        t0 = time.time()
        idx = load_full_pedigree_index(None, path=ped_dir)
        if not quiet:
            print(f"[10gen3v] index 構築完了: {len(idx):,} 頭 ({time.time()-t0:.1f}s)")

    sink = LongParquetSink(out_long, max_buffer_rows=long_buffer_rows)
    n_with_data = 0
    counts = {"father": 0, "dam_sire_line": 0, "dam_dam_line": 0}
    t0 = time.time()
    p10 = ped_10gen_dir if ped_10gen_dir is not None else PED_10GEN_DIR
    for i, hid in enumerate(horse_ids):
        if using_merged:
            rec = _load_10gen_json(p10, hid)
            views_all = expand_10gen_ancestors_from_record(rec)
            views = {
                "father": views_all.get("father", set()),
                "dam_sire_line": views_all.get("dam_sire_line", set()),
                "dam_dam_line": views_all.get("dam_dam_line", set()),
            }
        else:
            views = _expand_10gen_3view(hid, idx or {})
        any_data = any(views[s] for s in views)
        if any_data:
            n_with_data += 1
        for s, aset in views.items():
            counts[s] += len(aset)
            for a in aset:
                sink.append_triplet(hid, a, s)
        if not quiet and (i + 1) % 5000 == 0:
            print(f"[10gen3v]  {i+1:,}/{len(horse_ids):,} ({(i+1)/len(horse_ids)*100:.1f}%) "
                  f"rows={sink.total_rows:,} elapsed={time.time()-t0:.1f}s")

    if not quiet:
        print(f"[10gen3v] 展開完了: {n_with_data:,} 馬にデータあり / "
              f"father avg={counts['father']/max(1,n_with_data):.1f} / "
              f"dam_sire_line avg={counts['dam_sire_line']/max(1,n_with_data):.1f} / "
              f"dam_dam_line avg={counts['dam_dam_line']/max(1,n_with_data):.1f}")

    sink.close()
    n_rows_long = sink.total_rows
    del sink
    gc.collect()

    if not quiet:
        print(f"[10gen3v] 書き出し: {out_long} ({n_rows_long:,} rows)")

    if not quiet:
        print(f"[10gen3v] inverted index 構築")
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
        print(f"[10gen3v] inv 書き出し: {out_inv} (ancestor 数={df_inv['ancestor_id'].nunique():,})")
    df_inv.to_parquet(out_inv, index=False, compression="zstd")

    meta = {
        "n_horses_target": len(horse_ids),
        "n_horses_with_data": n_with_data,
        "n_rows": int(n_rows_long),
        "n_unique_ancestors": int(df_inv["ancestor_id"].nunique()),
        "father_avg_per_horse": counts["father"] / max(1, n_with_data),
        "dam_sire_line_avg_per_horse": counts["dam_sire_line"] / max(1, n_with_data),
        "dam_dam_line_avg_per_horse": counts["dam_dam_line"] / max(1, n_with_data),
        "out_long": str(out_long),
        "out_inv": str(out_inv),
        "ped_dir": str(ped_dir),
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    out_meta.write_text(_json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(f"[10gen3v] meta: {out_meta}")
        print(f"[10gen3v] DONE")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="先頭 N 頭で動作確認")
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
