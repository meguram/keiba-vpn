"""
ped_tbl 向け: 種牡馬スロット行に「クロス」（重複・父系/母系/全体での出現と濃度％）を付与する。

- ``path_fm`` 先頭が ``F`` → 主馬の **父由来** 側（``ped_root_side=paternal``）。
- 先頭が ``M`` → **母由来**（``ped_root_side=maternal``）。
- 同一 ``ancestor_horse_id`` の出現回数を、全体・父系 subtree（``path_fm`` が ``F`` で始まる行）・母系（``M`` で始まる行）で数える。
- **両系統クロス**: 同一 ID が父系にも母系にも出現 → ``ancestor_cross_both_roots=1``。
- **濃度％**: 当該 ID の出現回数 / その範囲内の有効 ancestor 行数 × 100。
- **subject_***: 主馬単位の集計を全行に複製。
"""

from __future__ import annotations

import pandas as pd


def add_pedigree_cross_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ``path_fm``・``ancestor_horse_id`` が揃っている ``ped_tbl`` DataFrame に列を追加する。
    ``ancestor_horse_id`` は事前に欠損正規化済みであることを想定。
    """
    out = df.copy()
    if out.empty:
        return out

    path = out["path_fm"].astype(str)
    first = path.str[0]
    out["ped_root_side"] = first.map({"F": "paternal", "M": "maternal"}).astype("string")

    aid = out["ancestor_horse_id"]
    valid = aid.notna() & (aid.astype(str).str.strip() != "") & (aid.astype(str) != "<NA>")
    aid_key = aid.where(valid, pd.NA)

    vc_global = aid_key.dropna().value_counts()
    pat_mask = valid & path.str.startswith("F", na=False)
    mat_mask = valid & path.str.startswith("M", na=False)
    vc_pat = aid_key[pat_mask].dropna().value_counts()
    vc_mat = aid_key[mat_mask].dropna().value_counts()

    out["ancestor_occurrence_global"] = aid_key.map(vc_global).fillna(0).astype("int32")
    out["ancestor_occurrence_paternal"] = aid_key.map(vc_pat).fillna(0).astype("int32")
    out["ancestor_occurrence_maternal"] = aid_key.map(vc_mat).fillna(0).astype("int32")

    both_ids = set(vc_pat[vc_pat >= 1].index) & set(vc_mat[vc_mat >= 1].index)
    out["ancestor_cross_both_roots"] = (
        valid & aid.astype(str).isin(both_ids)
    ).astype("int8")

    n_g = int(valid.sum())
    n_pat = int(pat_mask.sum())
    n_mat = int(mat_mask.sum())

    def _pct(num: pd.Series, denom: int) -> pd.Series:
        if denom <= 0:
            return pd.Series(pd.NA, index=out.index, dtype="Float64")
        return (100.0 * num.astype("float64") / float(denom)).astype("Float64")

    out["ancestor_pct_tree_global"] = _pct(out["ancestor_occurrence_global"], n_g)
    out["ancestor_pct_paternal_subtree"] = _pct(out["ancestor_occurrence_paternal"], n_pat)
    out["ancestor_pct_maternal_subtree"] = _pct(out["ancestor_occurrence_maternal"], n_mat)
    out.loc[~valid, "ancestor_pct_tree_global"] = pd.NA
    out.loc[~valid, "ancestor_pct_paternal_subtree"] = pd.NA
    out.loc[~valid, "ancestor_pct_maternal_subtree"] = pd.NA

    distinct = int(aid_key.dropna().nunique())
    dup_ids = int((vc_global >= 2).sum())
    both_cnt = len(both_ids)
    excess = int((vc_global - 1).clip(lower=0).sum())

    out["subject_distinct_ancestor_id_count"] = pd.Series(distinct, index=out.index, dtype="int32")
    out["subject_duplicate_ancestor_id_count"] = pd.Series(dup_ids, index=out.index, dtype="int32")
    out["subject_cross_both_roots_ancestor_id_count"] = pd.Series(both_cnt, index=out.index, dtype="int32")
    out["subject_cross_duplicate_excess_slots"] = pd.Series(excess, index=out.index, dtype="int32")

    return out


# ped_tbl に付与する列名（build_horse_entity_store の PED_TBL_LONG_COLUMNS と整合）
PED_CROSS_COLUMN_NAMES: list[str] = [
    "ped_root_side",
    "ancestor_occurrence_global",
    "ancestor_occurrence_paternal",
    "ancestor_occurrence_maternal",
    "ancestor_cross_both_roots",
    "ancestor_pct_tree_global",
    "ancestor_pct_paternal_subtree",
    "ancestor_pct_maternal_subtree",
    "subject_distinct_ancestor_id_count",
    "subject_duplicate_ancestor_id_count",
    "subject_cross_both_roots_ancestor_id_count",
    "subject_cross_duplicate_excess_slots",
]
