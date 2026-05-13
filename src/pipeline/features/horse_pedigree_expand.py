"""
5 世代目の種牡馬ノードに **各馬の 5 世代血統** を接木し、主馬から最大深さ 10 の祖先表を組み立てるためのヘルパ。

設計の正: ``docs/modeling/horse_pedigree_10gen_merge_design.md``

座標写像・アンカー抽出・枝の連結と path 重複（主表優先）を提供する。枝 JSON の取得は呼び出し側
（``data/local/horse_pedigree_5gen`` 等）で行う。

**出力方針**: 牝スロットの行は載せず、**牡（種牡馬）スロットのみ**を ``path_fm`` / ``(generation, position)``
でツリー上に位置づけ、当該スロットの ``horse_id`` のみを繋ぐ（牝馬 ID は不要）。
"""

from __future__ import annotations

import re
from typing import Any

# 主表の最深世代（netkeiba 5 世代表）
PRIMARY_MAX_GENERATION = 5
# 5 世代目牡の 5 世代枝を足したときの主馬からの最大深さ
MERGED_MAX_GLOBAL_DEPTH = PRIMARY_MAX_GENERATION + PRIMARY_MAX_GENERATION

_ZERO_LIKE_ID = re.compile(r"^0+\.?0*$", re.IGNORECASE)


def _scalar_netkeiba_id_valid(value: Any) -> str | None:
    """``sanitize_netkeiba_string_id``（Series 向け）と整合するスカラー判定。"""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "<na>", "null"):
        return None
    if _ZERO_LIKE_ID.fullmatch(s):
        return None
    return s


def fm_path_from_gp(generation: int, position: int) -> str:
    """
    主表（または枝表）の (generation, position) を、父=F / 母=M のパスに変換する。

    ``research.pedigree.pedigree_similarity.is_paternal_side`` と同じ分割:
    世代 ``g`` では先頭 ``2**(g-1)`` 位置が父系側（当馬の父の subtree）、残りが母系側。
    """
    if generation < 1 or position < 0:
        return ""
    if generation == 1:
        return "F" if position == 0 else "M"
    half = 2 ** (generation - 1)
    if position < half:
        return "F" + fm_path_from_gp(generation - 1, position)
    return "M" + fm_path_from_gp(generation - 1, position - half)


def global_depth_after_merge(local_generation: int, *, anchor_primary_generation: int = PRIMARY_MAX_GENERATION) -> int:
    """枝表の local_generation（1=アンカーの父母）を主馬基準の深さに写す。"""
    return anchor_primary_generation + int(local_generation)


def iter_gen5_male_anchor_horse_ids(primary_rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """
    主表からアンカー (H_id, path_fm_S_to_H) を列挙。

    Returns:
        (anchor_horse_id, path_fm) のリスト。path_fm は主馬→H（長さ5）。
    """
    from src.research.pedigree.pedigree_similarity import is_male_pedigree_slot

    out: list[tuple[str, str]] = []
    for row in primary_rows:
        try:
            g = int(row.get("generation", 0))
            p = int(row.get("position", 0))
        except (TypeError, ValueError):
            continue
        if g != PRIMARY_MAX_GENERATION:
            continue
        if not is_male_pedigree_slot(g, p):
            continue
        hid = _scalar_netkeiba_id_valid(row.get("horse_id"))
        if not hid:
            continue
        path = fm_path_from_gp(g, p)
        if len(path) != PRIMARY_MAX_GENERATION:
            continue
        out.append((hid, path))
    return out


def merge_primary_and_branches(
    subject_horse_id: str,
    primary_rows: list[dict[str, Any]],
    branch_by_anchor: dict[str, list[dict[str, Any]]],
    *,
    subject_horse_name: str = "",
) -> list[dict[str, Any]]:
    """
    主表 + 各アンカーの枝表をロング行に統合する（FM パス付き）。

    branch_by_anchor[H]: H の ``horse_pedigree_5gen`` の ``ancestors`` リスト（H 自身は含まない）。

    付与列:
        path_fm, merged_global_depth, source (primary | merged_gen5_sire), anchor_horse_id（primary は None）

    重複 ``path_fm`` は **主表優先**（同一 path の merged 行は捨てる）。

    主表・枝とも **牡スロットの行だけ**を出力する（牝スロットはスキップ）。
    """
    from src.pipeline.features.horse_entity_layout import is_male_pedigree_slot_upto

    merged: list[dict[str, Any]] = []

    for row in primary_rows:
        try:
            g = int(row.get("generation", 0))
            p = int(row.get("position", 0))
        except (TypeError, ValueError):
            continue
        if not is_male_pedigree_slot_upto(g, p):
            continue
        path = fm_path_from_gp(g, p)
        base = dict(row)
        base.update(
            {
                "subject_horse_id": str(subject_horse_id),
                "subject_horse_name": subject_horse_name or str(base.get("subject_horse_name") or ""),
                "path_fm": path,
                "merged_global_depth": len(path),
                "source": "primary",
                "anchor_horse_id": None,
            }
        )
        merged.append(base)

    for hid, path_sh in iter_gen5_male_anchor_horse_ids(primary_rows):
        for row in branch_by_anchor.get(hid) or []:
            try:
                lg = int(row.get("generation", 0))
                lp = int(row.get("position", 0))
            except (TypeError, ValueError):
                continue
            if not is_male_pedigree_slot_upto(lg, lp):
                continue
            depth = global_depth_after_merge(lg)
            if depth > MERGED_MAX_GLOBAL_DEPTH:
                continue
            path_h = fm_path_from_gp(lg, lp)
            full_path = path_sh + path_h
            base = dict(row)
            base.update(
                {
                    "subject_horse_id": str(subject_horse_id),
                    "subject_horse_name": subject_horse_name or str(base.get("subject_horse_name") or ""),
                    "path_fm": full_path,
                    "merged_global_depth": len(full_path),
                    "source": "merged_gen5_sire",
                    "anchor_horse_id": hid,
                    "generation": PRIMARY_MAX_GENERATION + lg,
                    "position": lp,
                }
            )
            merged.append(base)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in merged:
        k = str(r.get("path_fm", ""))
        if r.get("source") == "merged_gen5_sire" and k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    return deduped
