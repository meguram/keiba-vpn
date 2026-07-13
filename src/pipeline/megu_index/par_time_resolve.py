"""megu_par_time の階層フォールバックマージ。"""

from __future__ import annotations

import pandas as pd

# 馬場状態のサンプル不足時: 重・不良 → 稍重 → 良（AREA-11 v2 補足）
_TRACK_CAT_FALLBACK: dict[str, list[str]] = {
    "重・不良": ["稍重", "良"],
    "稍重": ["良"],
}


def _prep_par(df_par: pd.DataFrame) -> pd.DataFrame:
    out = df_par.copy()
    out = out.rename(columns={"course": "direction", "track_condition": "track_cat"})
    if "class_bucket" not in out.columns:
        out["class_bucket"] = ""
    out["class_bucket"] = out["class_bucket"].fillna("").astype(str)
    out["par_time_sec"] = pd.to_numeric(out["par_time_sec"], errors="coerce")
    return out


def attach_par_time_with_fallback(
    df: pd.DataFrame,
    df_par: pd.DataFrame,
    *,
    class_col: str = "class_bucket",
) -> pd.DataFrame:
    """
    レース行に par_time_final / par_front_split_sec を付与する。

    フォールバック順（AREA-11 v2）:
      L1: dist × surface × direction × track_cat × class
      L2: dist × surface × track_cat × class
      L3: dist × surface × direction × track_cat（class=''）
      L4: dist × surface × track_cat（class=''）
      L5: L1–L4 を馬場良化（重・不良→稍重→良）で再試行
    """
    if df.empty:
        return df

    par = _prep_par(df_par)
    out = df.copy()
    out["track_cat"] = out.get("track_cat", out.get("track_condition"))
    track_map = {"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}
    if "track_cat" in out.columns:
        out["track_cat"] = out["track_cat"].map(track_map).fillna(out["track_cat"])
    out["direction"] = out.get("direction", out.get("course", ""))
    if class_col not in out.columns:
        out[class_col] = ""
    out[class_col] = out[class_col].fillna("").astype(str)

    out["par_time_sec"] = pd.NA
    out["par_front_split_sec"] = pd.NA
    out["par_match_level"] = ""

    def _merge_level(
        frame: pd.DataFrame,
        keys: list[str],
        par_sub: pd.DataFrame,
        level: str,
        only_null: bool = True,
    ) -> pd.DataFrame:
        if par_sub.empty:
            return frame
        cols = keys + ["par_time_sec", "par_front_split_sec"]
        use_cols = [c for c in cols if c in par_sub.columns]
        merged = frame.merge(
            par_sub[use_cols].drop_duplicates(keys),
            on=keys,
            how="left",
            suffixes=("", "_new"),
        )
        mask = merged["par_time_sec_new"].notna() if only_null else pd.Series(True, index=merged.index)
        if only_null:
            mask &= merged["par_time_sec"].isna()
        merged.loc[mask, "par_time_sec"] = merged.loc[mask, "par_time_sec_new"]
        if "par_front_split_sec_new" in merged.columns:
            merged.loc[mask, "par_front_split_sec"] = merged.loc[mask, "par_front_split_sec_new"]
        merged.loc[mask, "par_match_level"] = level
        drop_cols = [c for c in merged.columns if c.endswith("_new")]
        return merged.drop(columns=drop_cols)

    # L1: full + class
    par_l1 = par[par["class_bucket"] != ""]
    out = _merge_level(
        out,
        ["distance", "surface", "direction", "track_cat", class_col],
        par_l1.rename(columns={"class_bucket": class_col}),
        "L1_class",
    )

    # L2: no direction + class
    par_l2 = (
        par_l1.groupby(["distance", "surface", "track_cat", "class_bucket"], as_index=False)
        .agg(par_time_sec=("par_time_sec", "mean"), par_front_split_sec=("par_front_split_sec", "mean"))
    )
    out = _merge_level(
        out,
        ["distance", "surface", "track_cat", class_col],
        par_l2.rename(columns={"class_bucket": class_col}),
        "L2_no_direction",
    )

    # L3: full, class pooled
    par_l3 = par[par["class_bucket"] == ""]
    out = _merge_level(
        out,
        ["distance", "surface", "direction", "track_cat"],
        par_l3,
        "L3_pooled",
    )

    # L4: no direction, class pooled
    par_l4 = (
        par_l3.groupby(["distance", "surface", "track_cat"], as_index=False)
        .agg(par_time_sec=("par_time_sec", "mean"), par_front_split_sec=("par_front_split_sec", "mean"))
    )
    out = _merge_level(out, ["distance", "surface", "track_cat"], par_l4, "L4_pooled")

    # L5: 馬場状態フォールバック（サンプル不足セル向け）
    if out["par_time_sec"].isna().any():
        for src_cat, alt_cats in _TRACK_CAT_FALLBACK.items():
            mask_src = out["par_time_sec"].isna() & (out["track_cat"] == src_cat)
            if not mask_src.any():
                continue
            for alt_cat in alt_cats:
                mask = out["par_time_sec"].isna() & (out["track_cat"] == src_cat)
                if not mask.any():
                    break
                sub = out.loc[mask].copy()
                sub["track_cat"] = alt_cat
                sub = _merge_level(
                    sub,
                    ["distance", "surface", "direction", "track_cat", class_col],
                    par_l1.rename(columns={"class_bucket": class_col}),
                    "L5_track_class",
                )
                sub = _merge_level(
                    sub,
                    ["distance", "surface", "track_cat", class_col],
                    par_l2.rename(columns={"class_bucket": class_col}),
                    "L5_track_class",
                )
                sub = _merge_level(
                    sub,
                    ["distance", "surface", "direction", "track_cat"],
                    par_l3,
                    "L5_track_pooled",
                )
                sub = _merge_level(sub, ["distance", "surface", "track_cat"], par_l4, "L5_track_pooled")
                resolved = sub["par_time_sec"].notna()
                if resolved.any():
                    idx = sub.index[resolved]
                    out.loc[idx, "par_time_sec"] = sub.loc[idx, "par_time_sec"]
                    out.loc[idx, "par_front_split_sec"] = sub.loc[idx, "par_front_split_sec"]
                    out.loc[idx, "par_match_level"] = sub.loc[idx, "par_match_level"]

    out["par_time_final"] = out["par_time_sec"]
    return out
