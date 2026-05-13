"""
母系（二分木の母側 subtree）に現れる種牡馬 × 舞台別の集計（ファーストステップ）

入力
----
- ``data/local/tables/{year}/race_result_flat.parquet`` … 出走・着順・場・芝ダ・距離・馬場
- 5世代血統索引: **既定**は ``data/local/horse_pedigree_5gen/**/*.json``（シャードミラー）が
  あればそちらを優先。無ければ ``KEIBA_PEDIGREE_JSONL_GZ`` / ``data/research/_ped_snapshot_cache.jsonl.gz``
- （任意）``data/research/sire_factor_stats.json`` の ``sires`` … 既定では **母系・当馬基準
  世代 3〜10 の窓**に入った ``horse_id`` を辞書で種牡馬に限定。``--no-stallions-only`` は
  辞書絞りのみ外す（窓の範囲は同じで、父系全体へは広がらない）

母系の定義
----------
``research/pedigree_similarity.is_maternal_side(generation, position)`` と同一（当馬の母の subtree）。

世代 3〜10 の取り方
-------------------
- 当馬の 5 世代表上で母系かつ ``min_gen <= generation <= 5`` のノードをシードとする。
- 各シード祖先 ``X``（当馬の木での世代 = g）について、``X`` の 5 世代表の各祖先（世代 r）を
  当馬基準の世代 ``g + r`` として数える（最大 g=5, r=5 → 10 世代）。
- シードは母系のみ。拡張先の ``X`` の 5 世代表は **父・母両方** を含める（母系 = 母側 subtree 内の
  すべての祖先）。

出力
----
- Parquet: ``stallion_id``, ``stage_key``, ``track_condition_bucket``, ``full_stage_key``,
  ``starts``, ``wins``, ``top3``, ``sum_finish``, ``avg_finish`` など

CLI
---
    # --years を省略すると ``data/local/tables`` にある全エクスポート年の race_result を対象
    python3 -m research.build_maternal_stallion_stage_stats \\
        --out data/meta/modeling/maternal_stallion_stage_stats.parquet

    # ローカル JSON ミラー優先（無ければ gz）。GCS で gz を取りたい場合は --fetch-pedigree
    python3 -m research.build_maternal_stallion_stage_stats --fetch-pedigree

    # gz のみ使う（シャードディレクトリを無視）
    python3 -m research.build_maternal_stallion_stage_stats --pedigree-gz-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

# リポジトリルートをパスに追加
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.track_bias_pedigree import stage_key, track_condition_bucket  # noqa: E402
from research.pedigree_similarity import is_maternal_side  # noqa: E402
from research.pedigree_local_store import (  # noqa: E402
    load_full_pedigree_index,
    local_pedigree_json_dir_has_records,
)
from research.sire_factor_stats import load_sire_factor_stats  # noqa: E402
from utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger(__name__)

TABLES_DIR = Path("data/local/tables")
DEFAULT_LOCAL_PEDIGREE_DIR = Path("data/local/horse_pedigree_5gen")
DEFAULT_OUT = Path("data/meta/modeling/maternal_stallion_stage_stats.parquet")
META_OUT = Path("data/meta/modeling/maternal_stallion_stage_stats_meta.json")


def _norm_venue(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def full_stage_key(
    venue_code: Any,
    surface: Any,
    distance: Any,
    track_condition: Any,
) -> str:
    """場 × 芝ダ障 × 距離帯 × 馬場3区分。"""
    v = _norm_venue(venue_code)
    sk = stage_key(venue_code, surface, distance)
    cb = track_condition_bucket(track_condition)
    return f"{sk}|{cb}"


def _parse_finish_position(fp: Any) -> int | None:
    if fp is None or (isinstance(fp, float) and pd.isna(fp)):
        return None
    s = str(fp).strip()
    if not s or s in ("取消", "除外", "中止"):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def extract_maternal_stallion_ids(
    subject_id: str,
    index: dict[str, dict],
    *,
    stallion_ids: set[str] | None,
    min_gen: int = 3,
    max_gen: int = 10,
) -> set[str]:
    """
    当馬の母系 subtree から、世代 [min_gen, max_gen] に現れる horse_id を列挙。
    stallion_ids が与えられた場合はその集合に含まれる ID のみ（種牡馬辞書ベース）。
    """
    out: set[str] = set()
    rec0 = index.get(subject_id)
    if not rec0:
        return out
    ancestors0 = rec0.get("ancestors") or []

    for anc in ancestors0:
        try:
            g = int(anc.get("generation", 0))
            pos = int(anc.get("position", 0))
        except (TypeError, ValueError):
            continue
        if not is_maternal_side(g, pos):
            continue
        if g < 1 or g > 5:
            continue
        hid = str(anc.get("horse_id") or "").strip()
        if not hid:
            continue

        # 当馬の 5 世代表だけで完結する分（g + 0 は無いので g がそのまま祖先世代）
        if min_gen <= g <= max_gen:
            if stallion_ids is None or hid in stallion_ids:
                out.add(hid)

        # シード X より下の 5 世代（父・母両方）
        recx = index.get(hid)
        if not recx:
            continue
        for anc2 in recx.get("ancestors") or []:
            try:
                r = int(anc2.get("generation", 0))
            except (TypeError, ValueError):
                continue
            if r < 1 or r > 5:
                continue
            global_gen = g + r
            if global_gen < min_gen or global_gen > max_gen:
                continue
            hid2 = str(anc2.get("horse_id") or "").strip()
            if not hid2:
                continue
            if stallion_ids is None or hid2 in stallion_ids:
                out.add(hid2)

    return out


def load_stallion_id_set(path: Path | None) -> set[str]:
    data = load_sire_factor_stats(path)
    sires = data.get("sires") or {}
    return {str(k).strip() for k in sires.keys() if str(k).strip()}


def iter_race_result_paths(
    tables_dir: Path,
    years: list[int] | None,
) -> list[Path]:
    if not tables_dir.is_dir():
        return []
    ys = []
    for d in sorted(tables_dir.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        yi = int(d.name)
        if years is not None and yi not in years:
            continue
        p = d / "race_result_flat.parquet"
        if p.exists():
            ys.append(p)
    return ys


def run(
    *,
    tables_dir: Path,
    out_path: Path,
    meta_path: Path,
    pedigree_path: Path | None,
    sire_stats_path: Path | None,
    years: list[int] | None,
    stallions_only: bool,
    min_gen: int,
    max_gen: int,
    fetch_pedigree: bool = False,
) -> None:
    stallion_ids: set[str] | None
    if stallions_only:
        stallion_ids = load_stallion_id_set(sire_stats_path)
        logger.info("種牡馬 ID 集合: %d 頭（sire_factor_stats）", len(stallion_ids))
    else:
        stallion_ids = None
        logger.info("種牡馬フィルタなし（祖先の全 horse_id を対象）")

    storage = None
    if fetch_pedigree:
        try:
            from scraper.storage import HybridStorage

            storage = HybridStorage(".")
            logger.info("fetch-pedigree: HybridStorage で血統スナップショット取得を試行")
        except Exception as e:
            logger.warning("fetch-pedigree: HybridStorage 利用不可 (%s)", e)

    index = load_full_pedigree_index(
        storage,
        path=pedigree_path,
        force_refresh=False,
    )
    if not index:
        logger.error(
            "血統インデックスが空です。%s を用意するか GCS から取得してください。",
            pedigree_path or "KEIBA_PEDIGREE_JSONL_GZ",
        )
        sys.exit(1)
    logger.info("血統インデックス: %d 頭", len(index))

    # (stallion_id, full_stage_key) -> stats
    acc: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "starts": 0.0,
            "wins": 0.0,
            "top3": 0.0,
            "sum_finish": 0.0,
            "sum_fp_count": 0.0,
        }
    )

    paths = iter_race_result_paths(tables_dir, years)
    if not paths:
        logger.error("race_result_flat.parquet が見つかりません: %s", tables_dir)
        sys.exit(1)

    years_resolved = sorted(int(p.parent.name) for p in paths)
    logger.info(
        "対象年（resolved）: %s — 全 %d ファイル",
        years_resolved,
        len(paths),
    )

    total_rows = 0
    missing_ped = 0
    missing_fp = 0
    stallion_cache: dict[str, frozenset[str]] = {}

    read_cols = [
        "horse_id",
        "venue_code",
        "surface",
        "distance",
        "track_condition",
        "finish_position",
    ]

    def get_stallions(hid: str) -> frozenset[str]:
        if hid in stallion_cache:
            return stallion_cache[hid]
        s = extract_maternal_stallion_ids(
            hid,
            index,
            stallion_ids=stallion_ids,
            min_gen=min_gen,
            max_gen=max_gen,
        )
        fs = frozenset(s)
        stallion_cache[hid] = fs
        return fs

    for path in paths:
        logger.info("読み込み %s", path)
        try:
            tbl = pq.read_table(path, columns=read_cols)
        except Exception as e:
            logger.warning("スキップ %s: %s", path, e)
            continue
        df = tbl.to_pandas()
        total_rows += len(df)

        for row in df.itertuples(index=False):
            hid = str(row.horse_id or "").strip()
            if not hid:
                continue
            if hid not in index:
                missing_ped += 1

            fp = _parse_finish_position(row.finish_position)
            if fp is None:
                missing_fp += 1
                continue

            stallions = get_stallions(hid)
            if not stallions:
                continue

            fsk = full_stage_key(
                row.venue_code,
                row.surface,
                row.distance,
                row.track_condition,
            )

            win = 1.0 if fp == 1 else 0.0
            t3 = 1.0 if fp <= 3 else 0.0

            for sid in stallions:
                a = acc[(sid, fsk)]
                a["starts"] += 1.0
                a["wins"] += win
                a["top3"] += t3
                a["sum_finish"] += float(fp)
                a["sum_fp_count"] += 1.0

    rows_out = []
    for (sid, fsk), a in acc.items():
        n = a["starts"]
        if n <= 0:
            continue
        parts = fsk.split("|")
        # stage_key = venue|surface_group|dist_band ; cond = last part
        cond_bucket = parts[-1] if parts else "unknown"
        stage_simple = "|".join(parts[:-1]) if len(parts) > 1 else fsk

        rows_out.append(
            {
                "stallion_id": sid,
                "full_stage_key": fsk,
                "stage_key": stage_simple,
                "track_condition_bucket": cond_bucket,
                "venue_code": parts[0] if parts else "",
                "surface_group": parts[1] if len(parts) > 1 else "",
                "distance_band": parts[2] if len(parts) > 2 else "",
                "starts": int(n),
                "wins": int(a["wins"]),
                "top3": int(a["top3"]),
                "win_rate": float(a["wins"] / n),
                "top3_rate": float(a["top3"] / n),
                "avg_finish": float(a["sum_finish"] / max(1.0, a["sum_fp_count"])),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows_out)
    if result.empty:
        logger.warning("集計結果が空です（血統・種牡馬条件・レース年を確認してください）")
    result.to_parquet(out_path, index=False)
    logger.info("保存: %s (%d 行)", out_path, len(result))

    meta = {
        "tables_dir": str(tables_dir),
        "years_filter": years,
        "years_resolved": years_resolved,
        "min_gen": min_gen,
        "max_gen": max_gen,
        "stallions_only": stallions_only,
        "fetch_pedigree_attempted": fetch_pedigree,
        "pedigree_source": (
            "json_dir"
            if isinstance(pedigree_path, Path) and pedigree_path.is_dir()
            else "jsonl_gz"
        ),
        "pedigree_path": str(pedigree_path) if pedigree_path else None,
        "pedigree_keys": len(index),
        "unique_horses_stallion_cache": len(stallion_cache),
        "race_result_rows_scanned": total_rows,
        "rows_missing_pedigree_index": missing_ped,
        "rows_missing_finish_position": missing_fp,
        "output_rows": len(result),
        "unique_stallions": int(result["stallion_id"].nunique()) if len(result) else 0,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("メタ: %s", meta_path)


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="母系種牡馬 × 舞台別集計")
    ap.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--meta", type=Path, default=META_OUT)
    ap.add_argument(
        "--pedigree-gz",
        type=Path,
        default=None,
        help="5世代血統 JSONL.gz（指定時はディレクトリより優先。既定は gz のみ）",
    )
    ap.add_argument(
        "--pedigree-dir",
        type=Path,
        default=None,
        help=(
            "5gen JSON シャードのルート（例: data/local/horse_pedigree_5gen）。"
            "省略時はそのパスに *.json があれば自動採用。--pedigree-gz-only で無効化"
        ),
    )
    ap.add_argument(
        "--pedigree-gz-only",
        action="store_true",
        help="ローカル JSON ディレクトリを使わず jsonl.gz（KEIBA_PEDIGREE_JSONL_GZ）のみ",
    )
    ap.add_argument(
        "--sire-stats",
        type=Path,
        default=None,
        help="sire_factor_stats.json（--stallions-only 時の種牡馬 ID 源）",
    )
    ap.add_argument(
        "--years",
        type=str,
        default=None,
        help="カンマ区切り年（例: 2022,2023,2024）。省略時は tables 内の全エクスポート年",
    )
    ap.add_argument(
        "--fetch-pedigree",
        action="store_true",
        help="血統 gz が無いとき HybridStorage 経由で GCS スナップショット取得を試す",
    )
    ap.add_argument(
        "--no-stallions-only",
        action="store_true",
        help="sire_factor_stats の sires 辞書で絞らない（母系・世代 min_gen〜max_gen の窓は同じ）",
    )
    ap.add_argument("--min-gen", type=int, default=3)
    ap.add_argument("--max-gen", type=int, default=10)
    args = ap.parse_args()

    years_list: list[int] | None = None
    if args.years:
        years_list = [int(x.strip()) for x in args.years.split(",") if x.strip()]

    ped_index_path: Path | None = args.pedigree_gz
    if ped_index_path is None and not args.pedigree_gz_only:
        cand = args.pedigree_dir if args.pedigree_dir is not None else DEFAULT_LOCAL_PEDIGREE_DIR
        if local_pedigree_json_dir_has_records(cand):
            ped_index_path = cand

    run(
        tables_dir=args.tables_dir,
        out_path=args.out,
        meta_path=args.meta,
        pedigree_path=ped_index_path,
        sire_stats_path=args.sire_stats,
        years=years_list,
        stallions_only=not args.no_stallions_only,
        min_gen=args.min_gen,
        max_gen=args.max_gen,
        fetch_pedigree=args.fetch_pedigree,
    )


if __name__ == "__main__":
    main()
