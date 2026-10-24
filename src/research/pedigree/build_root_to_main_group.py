"""父系ツリーの各 root (父不明) について、配下子孫の系統分類 (sid_to_main) を集計し、
**多数決で代表 L1 系統** (Turn-To系 / Native Dancer系 / Northern Dancer系 / Nasrullah系 /
非主流) を決定する。

入力:
    data/research/pedigree_race_index/full_sire_tree.parquet
    data/research/pedigree_race_index/full_sire_tree_nodes.parquet
    data/research/bloodline_meta_cluster/sid_to_main.json

出力:
    data/research/bloodline_meta_cluster/root_to_main_group.json
        {root_horse_id: {"main_group": str,
                          "vote_counts": {l1: int, ...},
                          "n_descendants_classified": int,
                          "name": str}}

Usage:
    python -m src.research.pedigree.build_root_to_main_group
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EDGES = ROOT / "data/page_reference/pedigree_race_index/full_sire_tree.parquet"
NODES = ROOT / "data/page_reference/pedigree_race_index/full_sire_tree_nodes.parquet"
SID2MAIN = ROOT / "data/page_reference/note_aptitude_race/sid_to_main.json"
OUT = ROOT / "data/page_reference/note_aptitude_race/root_to_main_group.json"


def main() -> int:
    print(f"[load] {EDGES}", flush=True)
    edges = pd.read_parquet(EDGES)
    nodes = pd.read_parquet(NODES)
    sid_to_main: dict[str, str] = json.loads(SID2MAIN.read_text(encoding="utf-8"))

    # parent → children マップ
    children: dict[str, list[str]] = defaultdict(list)
    for _, r in edges.iterrows():
        children[str(r["sire_id"])].append(str(r["child_id"]))
    print(f"[graph] parent->children entries: {len(children):,}", flush=True)

    # name lookup
    name_of: dict[str, str] = dict(zip(nodes["horse_id"].astype(str), nodes["name"].astype(str)))

    # root 一覧
    roots = nodes[nodes["is_root"] == True]
    print(f"[roots] total roots: {len(roots)}", flush=True)

    # 各 root について BFS で子孫全 ID を集めて sid_to_main 多数決
    result: dict[str, dict] = {}
    for _, r in roots.iterrows():
        root_id = str(r["horse_id"])
        # BFS
        descendants: list[str] = []
        stack = [root_id]
        seen = {root_id}
        while stack:
            cur = stack.pop()
            for ch in children.get(cur, []):
                if ch not in seen:
                    seen.add(ch)
                    descendants.append(ch)
                    stack.append(ch)
        # 多数決
        votes = Counter()
        for d in descendants:
            mg = sid_to_main.get(d)
            if mg:
                votes[mg] += 1
        if votes:
            main_g, _ = votes.most_common(1)[0]
        else:
            # 自分自身も判定対象に
            main_g = sid_to_main.get(root_id, "非主流")
        result[root_id] = {
            "name": name_of.get(root_id, root_id),
            "main_group": main_g,
            "vote_counts": dict(votes),
            "n_descendants_classified": int(sum(votes.values())),
            "n_descendants_total": int(len(descendants)),
        }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {OUT}", flush=True)
    # 集計サマリ
    by_main = Counter(v["main_group"] for v in result.values())
    print()
    print("=== 各 L1 系統に振り分けられた root 数 ===")
    for k, v in by_main.most_common():
        print(f"  {k:18}: {v} 個")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
