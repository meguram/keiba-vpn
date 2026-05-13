"""
2020–2025 出走馬 + 各馬の 5 世代表の「牡スロット」に現れる horse_id を再帰的に辿った集合
（種牡馬クロージャ）の 5 世代血統をキュー投入し、GCS 上の horse_pedigree_5gen をローカルに
ミラーする（horse_id 先頭4桁でサブディレクトリ分割）。

牡スロットは ``research.pedigree.pedigree_similarity.is_male_pedigree_slot``（同一世代内で position が偶数）
と同一の定義。完全な 5 世代表では牡は **31 スロット**（``MALE_PEDIGREE_SLOTS_5GEN``）＝
一頭あたり最大 31 頭分の種牡馬（牡）ノード。種牡馬因子 JSON の **6 スロット加権**
（``sire_factor_aptitude.ANCESTOR_SLOTS``）とは別で、こちらは **表に載る牡をすべて**
キュー対象にする。

インデックスに無い馬は集合に含めるが展開はできない（未取得分はワーカ取得後に
再 enqueue すれば辿れる）。

1) キュー投入（既存はインデックス／GCS で判定し **キューに載せない** + ワーカ側 smart_skip）::

    python3 -m src.scraper.horse_pedigree_5gen_bulk enqueue \\
        --tables-dir data/local/tables --years 2020,2021,2022,2023,2024,2025 --fetch-pedigree

2) ワーカでキューを消化したのち、GCS → ローカル JSON（4桁シャード）::

    python3 -m src.scraper.horse_pedigree_5gen_bulk mirror-local \\
        --out-dir data/local/horse_pedigree_5gen

**GCS にだけあり out_dir に無いキー**は ``storage.load`` で GCS から取得してローカルに書き込む（
``--skip-existing`` 既定時は、**既に out_dir に十分なサイズのファイルがある**ときだけスキップ）。
既定では HybridStorage のメモリ/L2 を迂回し **GCS を直接参照**する（``--use-storage-cache`` でキャッシュ利用可）。

サマリの ``written_this_run`` は「今回新規に書いた件数」であり、
``skipped_existing_local`` と足して GCS キー数と一致すれば全件ローカルにある（リポジトリルートで実行すること）。

種牡馬クロージャの列挙にはローカル血統スナップショットが必要なため、無い場合は ``--fetch-pedigree`` を付与。
旧挙動（母系＋sire_factor_stats フィルタ）が必要な場合は ``--legacy-maternal``。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pyarrow.parquet as pq  # noqa: E402

from src.research.pedigree.build_maternal_stallion_stage_stats import (  # noqa: E402
    extract_maternal_stallion_ids,
    load_stallion_id_set,
)
from src.research.pedigree.pedigree_local_store import (  # noqa: E402
    load_full_pedigree_index,
    stallion_pedigree_closure_horse_ids,
)
from src.scraper.job_queue import PRIORITY_URGENT_PEDIGREE_5GEN, ScrapeJobQueue  # noqa: E402
from src.scraper.storage import HybridStorage  # noqa: E402
from src.utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/local/horse_pedigree_5gen")
MANIFEST = Path("data/meta/pedigree_5gen_bulk_manifest.json")


def _parse_years_arg(years_str: str) -> list[int]:
    """``2020,2021`` または ``2020-2025``（ハイフン範囲1つ）に対応。"""
    raw = (years_str or "").strip()
    if not raw:
        return []
    if "," in raw:
        chunks = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        chunks = [raw]
    out: list[int] = []
    for chunk in chunks:
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            try:
                y0, y1 = int(a.strip()), int(b.strip())
            except ValueError:
                continue
            lo, hi = min(y0, y1), max(y0, y1)
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(chunk))
            except ValueError:
                continue
    return sorted(set(out))


def collect_race_horse_ids(tables_dir: Path, years: list[int]) -> set[str]:
    out: set[str] = set()
    for y in years:
        p = tables_dir / str(y) / "race_result_flat.parquet"
        if not p.exists():
            logger.warning("スキップ（無い）: %s", p)
            continue
        tbl = pq.read_table(p, columns=["horse_id"])
        s = tbl.column(0).to_pylist()
        for h in s:
            t = str(h or "").strip()
            if t:
                out.add(t)
    return out


def collect_maternal_stallion_ids(
    race_horses: set[str],
    index: dict[str, dict],
    *,
    stallion_ids: set[str] | None,
) -> set[str]:
    """母系 3–10 世代に現れる horse_id（stallion_ids でフィルタ可）。"""
    out: set[str] = set()
    cache: dict[str, frozenset[str]] = {}

    def maternal(hid: str) -> frozenset[str]:
        if hid in cache:
            return cache[hid]
        s = frozenset(
            extract_maternal_stallion_ids(
                hid, index, stallion_ids=stallion_ids, min_gen=3, max_gen=10
            )
        )
        cache[hid] = s
        return s

    for hid in race_horses:
        if hid not in index:
            continue
        out |= maternal(hid)
    return out


def cmd_enqueue(args: argparse.Namespace) -> None:
    years = _parse_years_arg(args.years)
    if not years:
        logger.error("有効な --years がありません: %r", args.years)
        sys.exit(1)
    rh = collect_race_horse_ids(Path(args.tables_dir), years)
    logger.info("出走馬（ユニーク）: %d 頭", len(rh))

    storage = HybridStorage(args.base_dir)
    if args.fetch_pedigree:
        logger.info("--fetch-pedigree: スナップショット取得を試みます")

    index = load_full_pedigree_index(storage, path=args.pedigree_gz)
    logger.info("血統索引: %d 頭", len(index))

    closure, hit_cap = stallion_pedigree_closure_horse_ids(
        rh, index, max_nodes=args.max_closure_nodes
    )
    if hit_cap:
        logger.warning(
            "種牡馬クロージャが max_closure_nodes=%d に達しました（打ち切り）",
            args.max_closure_nodes,
        )
    logger.info("5gen 牡スロットから再帰的に収集した馬 ID: %d 頭", len(closure))

    extra_maternal: set[str] = set()
    if args.legacy_maternal:
        stallion_ids: set[str] | None
        if args.maternal_all_ancestor_ids:
            stallion_ids = None
            logger.info("legacy 母系: 種牡馬辞書で絞らず祖先 horse_id すべて")
        else:
            stallion_ids = load_stallion_id_set(
                Path(args.sire_stats) if args.sire_stats else None
            )
            logger.info(
                "legacy 母系: sire_factor_stats の sires でフィルタ (%d 頭)",
                len(stallion_ids or ()),
            )
        extra_maternal = collect_maternal_stallion_ids(rh, index, stallion_ids=stallion_ids)
        logger.info("legacy 母系から追加の馬 ID: %d 頭", len(extra_maternal))

    all_ids = sorted(closure | extra_maternal)
    logger.info("キュー投入候補（重複除く）: %d 頭", len(all_ids))

    q = ScrapeJobQueue()
    stats = q.add_horse_jobs_bulk(
        all_ids,
        ["horse_pedigree_5gen"],
        priority=PRIORITY_URGENT_PEDIGREE_5GEN,
        smart_skip=True,
        pedigree_index=index,
        skip_pedigree_5gen_if_complete=not args.no_enqueue_skip_complete,
    )
    logger.info("キュー結果: %s", stats)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "years": years,
        "unique_race_horses": len(rh),
        "stallion_closure_horses": len(closure),
        "stallion_closure_hit_cap": hit_cap,
        "legacy_maternal_extra": len(extra_maternal),
        "union_candidates": len(all_ids),
        "queue_bulk_stats": stats,
        "pedigree_index_size": len(index),
        "legacy_maternal": bool(args.legacy_maternal),
        "maternal_filter": (
            "all_ancestors"
            if args.maternal_all_ancestor_ids
            else "sire_factor_stats"
        )
        if args.legacy_maternal
        else None,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("マニフェスト: %s", MANIFEST)


def cmd_mirror_local(args: argparse.Namespace) -> None:
    storage = HybridStorage(args.base_dir)
    if not storage.gcs_enabled:
        logger.error("GCS が無効です。.env の GCS 設定を確認してください。")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = storage.list_keys("horse_pedigree_5gen")
    nkeys = len(keys)
    logger.info("GCS horse_pedigree_5gen キー数: %d", nkeys)
    logger.info("ローカル出力先（絶対パス）: %s", out_dir)

    # ローカル（out_dir）に無い分は GCS から取る。既定でキャッシュを迂回して GCS を読む。
    bypass_cache = not getattr(args, "use_storage_cache", False)
    if bypass_cache:
        logger.info("GCS 直接読み（--use-storage-cache 無し）: out_dir に無いキーは GCS からコピーします")

    ok = miss = skip = 0
    for i, key in enumerate(keys):
        if (i + 1) % 500 == 0:
            logger.info("進捗 %d / %d", i + 1, nkeys)
        key = str(key).strip()
        if len(key) < 4:
            sub = "_"
        else:
            sub = key[:4]
        dest = out_dir / sub / f"{key}.json"
        if args.skip_existing and dest.exists() and dest.stat().st_size > 10:
            skip += 1
            continue
        data = storage.load(
            "horse_pedigree_5gen", key, bypass_cache=bypass_cache
        )
        if data is None:
            miss += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        ok += 1

    local_count = sum(1 for _ in out_dir.rglob("*.json"))
    tally = ok + miss + skip
    summary = {
        "gcs_key_count": nkeys,
        "written_this_run": ok,
        "missing_load": miss,
        "skipped_existing_local": skip,
        "local_json_count_after": local_count,
        "out_dir": str(out_dir),
        "success": miss == 0 and tally == nkeys,
        "note": (
            "written_this_run は今回ディスクに新規書き込んだ件数。"
            "skipped_existing_local は既にローカルにありスキップした件数。"
            "合計で GCS キー数と一致すればミラーは完了。"
        ),
        "written": ok,
        "skipped_local": skip,
    }
    logger.info(
        "ミラー完了: GCS %d 件のうち 今回新規書込 %d 件 / ローカル既存でスキップ %d 件 / 読込失敗 %d 件",
        nkeys,
        ok,
        skip,
        miss,
    )
    logger.info("ローカル *.json 件数（再帰）: %d", local_count)
    if miss:
        logger.warning("読込失敗が %d 件あります。GCS にオブジェクトが無い、または一時的な取得エラーの可能性があります。", miss)
    elif tally != nkeys:
        logger.error("内部不整合: ok+miss+skip=%d != gcs_key_count=%d", tally, nkeys)
    elif local_count != nkeys:
        logger.warning(
            "ローカル JSON 件数 %d が GCS キー数 %d と一致しません（別パスへの古いファイル等）。",
            local_count,
            nkeys,
        )
    else:
        logger.info("全件ローカルに揃いました（GCS キー数と一致）。")

    p = Path(args.summary_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="horse_pedigree_5gen 一括キュー / ローカルミラー")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser(
        "enqueue",
        help="出走馬 + 5gen 牡スロットの種牡馬クロージャをキューに追加",
    )
    pe.add_argument("--base-dir", type=Path, default=Path("."))
    pe.add_argument("--tables-dir", type=Path, default=Path("data/local/tables"))
    pe.add_argument("--years", type=str, default="2020,2021,2022,2023,2024,2025")
    pe.add_argument("--fetch-pedigree", action="store_true")
    pe.add_argument(
        "--pedigree-gz",
        type=Path,
        default=None,
        help="KEIBA_PEDIGREE_JSONL_GZ の上書きパス",
    )
    pe.add_argument(
        "--max-closure-nodes",
        type=int,
        default=500_000,
        help="種牡馬クロージャ BFS の上限ノード数（既定 500000）",
    )
    pe.add_argument(
        "--legacy-maternal",
        action="store_true",
        help="旧方式: 母系＋sire_factor_stats（または --maternal-all-ancestor-ids）の追加集合を併せる",
    )
    pe.add_argument(
        "--no-enqueue-skip-complete",
        action="store_true",
        help="既に 5 世代揃っている馬もキューに載せる（重複ジョブ・ワーカ負荷のデバッグ用）",
    )
    pe.add_argument(
        "--sire-stats",
        type=Path,
        default=None,
        help="sire_factor_stats.json（--legacy-maternal 時の母系フィルタ用）",
    )
    pe.add_argument(
        "--maternal-all-ancestor-ids",
        action="store_true",
        help="--legacy-maternal 時: 母系の種牡馬辞書を使わず母系の全祖先 horse_id を含める",
    )
    pe.set_defaults(func=cmd_enqueue)

    pm = sub.add_parser("mirror-local", help="GCS の horse_pedigree_5gen をローカルに書き出す")
    pm.add_argument("--base-dir", type=Path, default=Path("."))
    pm.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    pm.add_argument("--skip-existing", action="store_true", default=True)
    pm.add_argument("--no-skip-existing", action="store_true", help="既存ローカルファイルも上書き")
    pm.add_argument(
        "--use-storage-cache",
        action="store_true",
        help=(
            "HybridStorage のメモリ/L2 キャッシュを使う（既定はオフ＝GCS を直接読み、"
            "out_dir に無いオブジェクトを確実に GCS から取得）"
        ),
    )
    pm.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/meta/pedigree_5gen_mirror_summary.json"),
    )
    pm.set_defaults(func=cmd_mirror_local)

    args = ap.parse_args()
    if getattr(args, "no_skip_existing", False):
        args.skip_existing = False
    args.func(args)


if __name__ == "__main__":
    main()
