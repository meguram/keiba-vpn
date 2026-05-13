"""
セカンドステップ: 種牡馬同士の祖先グラフ（父・母父の2本で接続）

対象となった種牡馬（既定: ファーストステップの Parquet の stallion_id 一意、
または sire_factor_stats の sires）について、各馬の 5 世代表から

- **父** … (generation=1, position=0)
- **母父** … (generation=2, position=2)  … ``scripts/scrape_pedigree_5gen.build_pedigree_record`` と API ``_tail_sire_line`` と同じスロット

の ``horse_id`` を取り、有向辺を張る。

**既定の閉包**: 対象の主馬から BFS で辿り、索引に主馬がある限り **父 (1,0)・母父 (2,2) 方向に再帰的に** 辺を伸ばす（主馬 JSON が無い ID で途切れるまで）。``--shallow-one-hop`` で従来の 1 ホップのみに戻せる。

辺の向き（時間の流れ）
----------------------
祖先 → 子孫（その父が子を生んだ／母父は母の父として子の祖先）

- ``relation=sire``       … ``source`` が ``target`` の父
- ``relation=dam_sire``   … ``source`` が ``target`` の母父

可視化
------
``--write-html`` で ``index.html`` + ``graph.json`` を同じディレクトリに出す。
ビューア HTML は ``research/templates/stallion_ancestor_graph_viewer.html`` をコピーしたもの（拡大縮小・等倍・全体表示、物理 ON/OFF、ばね長・文字・辺の太さスライダー。直接編集して再ビルドすると反映される）。
ブラウザでは同一ディレクトリを ``python3 -m http.server`` で配信して開く（file:// では fetch がブロックされる場合あり）。

CLI
---
    python3 -m research.build_stallion_ancestor_graph \\
        --from-parquet data/meta/modeling/maternal_stallion_stage_stats.parquet \\
        --pedigree-gz data/local/horse_pedigree_5gen \\
        --out-dir data/meta/modeling/stallion_tree
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from research.pedigree_local_store import load_full_pedigree_index  # noqa: E402
from research.sire_factor_stats import load_sire_factor_stats  # noqa: E402
from utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger(__name__)

# 5 世代表スロット（netkeiba / 本リポジトリの血統パースと一致）
SIRE_SLOT = (1, 0)
DAM_SIRE_SLOT = (2, 2)


def ancestor_id_at(rec: dict | None, generation: int, position: int) -> str:
    if not rec:
        return ""
    for a in rec.get("ancestors") or []:
        if not isinstance(a, dict):
            continue
        try:
            g = int(a.get("generation", -1))
            p = int(a.get("position", -1))
        except (TypeError, ValueError):
            continue
        if g == generation and p == position:
            return str(a.get("horse_id") or "").strip()
    return ""


def ancestor_name_at(rec: dict | None, generation: int, position: int) -> str:
    if not rec:
        return ""
    for a in rec.get("ancestors") or []:
        if not isinstance(a, dict):
            continue
        try:
            g = int(a.get("generation", -1))
            p = int(a.get("position", -1))
        except (TypeError, ValueError):
            continue
        if g == generation and p == position:
            return str(a.get("name") or "").strip()
    return ""


def load_target_ids(
    *,
    from_parquet: Path | None,
    parquet_column: str,
    from_sire_stats: bool,
    sire_stats_path: Path | None,
) -> set[str]:
    if from_parquet is not None:
        if not from_parquet.exists():
            raise FileNotFoundError(from_parquet)
        df = pd.read_parquet(from_parquet, columns=[parquet_column])
        s = df[parquet_column].astype(str).str.strip()
        return set(s[s != ""].unique())
    if from_sire_stats:
        data = load_sire_factor_stats(sire_stats_path)
        sires = data.get("sires") or {}
        return {str(k).strip() for k in sires.keys() if str(k).strip()}
    raise ValueError("対象種牡馬を指定してください: --from-parquet または --from-sire-stats")


def resolve_name(
    hid: str,
    *,
    index: dict[str, dict],
    sire_stats: dict[str, Any],
) -> str:
    if not hid:
        return ""
    sires = sire_stats.get("sires") or {}
    ent = sires.get(hid)
    if isinstance(ent, dict) and (ent.get("name") or "").strip():
        return str(ent["name"]).strip()
    rec = index.get(hid)
    if rec:
        return str(rec.get("horse_name") or rec.get("name") or "").strip()
    return ""


def build_graph(
    target_ids: set[str],
    index: dict[str, dict],
    sire_stats: dict[str, Any],
    *,
    expand_parents: bool,
    recursive_closure: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """nodes, edges, meta

    ``recursive_closure`` … 真なら、対象の主馬から BFS し索引に主馬がある限り
    各ノードについて (1,0) 父・(2,2) 母父へ矢印を再帰的に張る。
    偽なら従来どおり「対象の主馬」からの 1 ホップのみ。
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()

    meta: dict[str, Any] = {
        "target_count": len(target_ids),
        "targets_with_subject_pedigree": 0,
        "targets_without_subject_pedigree": 0,
        "missing_sire_id": 0,
        "missing_dam_sire_id": 0,
        "edges_sire": 0,
        "edges_dam_sire": 0,
        "skipped_sire_not_in_scope": 0,
        "skipped_dam_sire_not_in_scope": 0,
        "nodes_expanded": 0,
        "recursive_parent_closure": bool(recursive_closure),
        "closure_nodes_processed": 0,
    }
    with_subject = target_ids & set(index.keys())
    meta["targets_with_subject_pedigree"] = len(with_subject)
    meta["targets_without_subject_pedigree"] = len(target_ids) - len(with_subject)

    def ensure_node(nid: str, *, in_target: bool, name_hint: str = "") -> None:
        if not nid:
            return
        if nid not in nodes:
            nm = name_hint or resolve_name(nid, index=index, sire_stats=sire_stats)
            if not nm:
                nm = nid
            nodes[nid] = {
                "id": nid,
                "label": nm,
                "in_target": bool(in_target),
            }
            if not in_target:
                meta["nodes_expanded"] += 1
        else:
            if in_target:
                nodes[nid]["in_target"] = True

    for hid in sorted(target_ids):
        ensure_node(hid, in_target=True)

    def emit_edges_for_hid(hid: str, *, enqueue_parents: bool) -> None:
        rec = index.get(hid)
        if not rec:
            return

        sire_id = ancestor_id_at(rec, *SIRE_SLOT)
        ds_id = ancestor_id_at(rec, *DAM_SIRE_SLOT)
        if not sire_id:
            meta["missing_sire_id"] += 1
        if not ds_id:
            meta["missing_dam_sire_id"] += 1

        sn = ancestor_name_at(rec, *SIRE_SLOT)
        dsn = ancestor_name_at(rec, *DAM_SIRE_SLOT)

        for rel, pid, pname, sk in (
            ("sire", sire_id, sn, "edges_sire"),
            ("dam_sire", ds_id, dsn, "edges_dam_sire"),
        ):
            if not pid:
                continue
            if not expand_parents and pid not in target_ids:
                meta["skipped_sire_not_in_scope" if rel == "sire" else "skipped_dam_sire_not_in_scope"] += 1
                continue

            ensure_node(hid, in_target=hid in target_ids)
            ensure_node(pid, in_target=pid in target_ids, name_hint=pname)
            ek = (pid, hid, rel)
            if ek in edge_keys:
                continue
            edge_keys.add(ek)
            edges.append(
                {
                    "source": pid,
                    "target": hid,
                    "relation": rel,
                }
            )
            meta[sk] += 1
            if enqueue_parents and pid in index and pid not in closure_seen:
                q.append(pid)

    if recursive_closure:
        closure_seen: set[str] = set()
        q: deque[str] = deque(sorted(with_subject))

        while q:
            hid = q.popleft()
            if hid in closure_seen:
                continue
            closure_seen.add(hid)
            meta["closure_nodes_processed"] = len(closure_seen)
            ensure_node(hid, in_target=hid in target_ids)
            emit_edges_for_hid(hid, enqueue_parents=True)
    else:
        for hid in sorted(with_subject):
            emit_edges_for_hid(hid, enqueue_parents=False)

    node_list = list(nodes.values())
    return node_list, edges, meta


_VIEWER_TEMPLATE = (
    Path(__file__).resolve().parent / "templates" / "stallion_ancestor_graph_viewer.html"
)


def write_html_viewer(out_dir: Path) -> Path:
    """``graph.json`` と同ディレクトリに ``index.html`` を書く（テンプレは research/templates）。"""
    html_path = out_dir / "index.html"
    tpl = _VIEWER_TEMPLATE
    if not tpl.is_file():
        raise FileNotFoundError(f"ビューアテンプレートが見つかりません: {tpl}")
    html_path.write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8")
    return html_path


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="種牡馬祖先グラフ（父・母父）")
    ap.add_argument(
        "--from-parquet",
        type=Path,
        default=None,
        help="stallion_id 列を含む Parquet（例: maternal_stallion_stage_stats.parquet）",
    )
    ap.add_argument(
        "--parquet-column",
        type=str,
        default="stallion_id",
        help="種牡馬 ID 列名",
    )
    ap.add_argument(
        "--from-sire-stats",
        action="store_true",
        help="--from-parquet が無いとき、sire_factor_stats の全 sires を対象にする",
    )
    ap.add_argument(
        "--sire-stats",
        type=Path,
        default=None,
        help="sire_factor_stats.json のパス",
    )
    ap.add_argument(
        "--pedigree-gz",
        type=Path,
        default=None,
        help="5世代血統 JSONL.gz",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/meta/modeling/stallion_tree"),
        help="graph.json / meta.json / 任意で index.html",
    )
    ap.add_argument(
        "--expand-parents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="親が対象集合外でも辺を張る（既定: 有効）。--no-expand-parents は親を対象集合に限定",
    )
    ap.add_argument(
        "--shallow-one-hop",
        action="store_true",
        help="対象主馬から父・母父へ1ホップのみ（親方向の再帰なし・従来）",
    )
    ap.add_argument(
        "--write-html",
        action="store_true",
        help="out-dir に vis-network 用 index.html を出力",
    )
    ap.add_argument(
        "--demo-all-subjects",
        action="store_true",
        help="母系 Parquet ではなく、血統索引の主馬 ID 全件を対象にする（辺が張れるかの検証用）",
    )
    args = ap.parse_args()

    if not args.from_parquet and not args.from_sire_stats and not args.demo_all_subjects:
        args.from_sire_stats = True
        logger.info("対象未指定のため --from-sire-stats を有効にしました")

    if args.demo_all_subjects:
        target_ids: set[str] = set()
        logger.info("--demo-all-subjects: 対象は後段で索引の主馬キーに置き換えます")
    else:
        target_ids = load_target_ids(
            from_parquet=args.from_parquet,
            parquet_column=args.parquet_column,
            from_sire_stats=args.from_sire_stats,
            sire_stats_path=args.sire_stats,
        )
        logger.info("対象種牡馬: %d 頭", len(target_ids))

    index = load_full_pedigree_index(None, path=args.pedigree_gz, force_refresh=False)
    if not index:
        logger.error("血統インデックスが空です。KEIBA_PEDIGREE_JSONL_GZ を用意してください。")
        sys.exit(1)
    logger.info("血統インデックス: %d 頭", len(index))

    if args.demo_all_subjects:
        target_ids = set(index.keys())
        logger.info("--demo-all-subjects: 主馬 %d 頭を対象にグラフを構築", len(target_ids))

    sire_stats = load_sire_factor_stats(args.sire_stats)
    nodes, edges, meta = build_graph(
        target_ids,
        index,
        sire_stats,
        expand_parents=args.expand_parents,
        recursive_closure=not args.shallow_one_hop,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_meta: dict[str, Any] = {
        "demo_all_subjects": bool(args.demo_all_subjects),
    }
    if meta.get("targets_without_subject_pedigree") == meta.get("target_count"):
        extra_meta["warning"] = (
            "母系 Parquet の stallion_id は血統スナップショットの「主馬」ID と一致しないため辺が張れません。"
            "種牡馬を horse_pedigree_5gen で主馬取得するか、--demo-all-subjects で索引主馬の検証グラフを出力してください。"
        )
    payload = {
        "meta": {
            **meta,
            **extra_meta,
            "sire_slot": list(SIRE_SLOT),
            "dam_sire_slot": list(DAM_SIRE_SLOT),
            "edge_direction": "ancestor_to_descendant",
            "expand_parents": args.expand_parents,
            "shallow_one_hop": args.shallow_one_hop,
        },
        "nodes": nodes,
        "edges": edges,
    }
    graph_path = out_dir / "graph.json"
    graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("保存: %s (nodes=%d edges=%d)", graph_path, len(nodes), len(edges))

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(payload["meta"], ensure_ascii=False, indent=2), encoding="utf-8")

    if args.write_html:
        hp = write_html_viewer(out_dir)
        logger.info("HTML: %s （このディレクトリで http.server して開く）", hp)

    # 母系対象で辺が 0 のとき、索引主馬の補助グラフを同ディレクトリに出す
    if (
        not args.demo_all_subjects
        and len(edges) == 0
        and meta.get("targets_without_subject_pedigree", 0) == meta.get("target_count", 0)
    ):
        demo_targets = set(index.keys())
        dn, de, dm = build_graph(
            demo_targets,
            index,
            sire_stats,
            expand_parents=args.expand_parents,
            recursive_closure=not args.shallow_one_hop,
        )
        aux = out_dir / "graph_index_subjects.json"
        aux.write_text(
            json.dumps(
                {
                    "meta": {
                        **dm,
                        "graph_kind": "index_subjects_fallback",
                        "sire_slot": list(SIRE_SLOT),
                        "dam_sire_slot": list(DAM_SIRE_SLOT),
                    },
                    "nodes": dn,
                    "edges": de,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "補助: %s (nodes=%d edges=%d) … vis は graph.json のまま。index_subjects を見る場合は JSON を差し替え",
            aux,
            len(dn),
            len(de),
        )


if __name__ == "__main__":
    main()
