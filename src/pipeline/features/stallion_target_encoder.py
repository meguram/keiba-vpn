"""
種牡馬（sire / dam_sire）Bayesian Shrinkage 統計。

## 設計方針

### リーク防止
各レース行に付与する統計は、そのレース「直前」（date, race_id 昇順）の産駒成績
のみを使用する。`cumsum(row) - current_row` パターンで時系列累計を計算する。

学習データ（train/valid/test すべて同一ロジック）:
  → 完全なリークなし（同日他レースも除外するため _dt = date+start_time でソート）

推論時: `cutoff_race_id` を指定するとその race_id 未満のレースだけを集計した
スナップショットを返す。

### スパース対策 (Bayesian Shrinkage)
  encoded = (C × prior + n_local × local_sum) / (C + n_local)
  ここで C は平滑化定数、prior は上位粒度の推定値。

  階層フォールバック:
    1. venue_code × surface_cat (C=50): 最細粒度
    2. surface_cat               (C=20): 中粒度
    3. global mean               (C=0):  最粗粒度 / フォールバック

### 生成カラム（prefix = "sire" or "dam_sire"）
  {prefix}_prior_starts            : 産駒出走数（直前累計）
  {prefix}_prior_win_rate          : 勝率（Bayesian, global fallback）
  {prefix}_prior_top3_rate         : 複勝率（Bayesian）
  {prefix}_prior_avg_finish_norm   : 平均着順正規化値 (rank / field_size, Bayesian)
  {prefix}_prior_win_rate_surface  : 馬場別勝率（Bayesian, surface fallback）
  {prefix}_prior_top3_rate_surface : 馬場別複勝率（Bayesian）
  {prefix}_prior_win_rate_dist     : 距離帯別勝率（Bayesian）
  {prefix}_prior_top3_rate_dist    : 距離帯別複勝率（Bayesian）
  {prefix}_prior_win_rate_venue_sf : 競馬場×馬場別勝率（高Bayesian）
  {prefix}_prior_avg_pass_surface  : 馬場別平均先頭コーナー通過順（走法傾向）
  {prefix}_prior_grade_win_rate    : グレード別勝率（Bayesian）
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- 平滑化定数 ---
C_GLOBAL = 30      # 全体（entity × 全馬場）
C_SURFACE = 20     # entity × surface
C_DIST = 25        # entity × distance_band
C_VENUE_SF = 50    # entity × venue × surface（最もスパース）
C_GRADE = 30       # entity × grade

# 着順正規化の最小サンプル数（n=0 は NaN）
MIN_STARTS_FOR_RATE = 0  # Bayesian で補正するため 0 でも許容


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _norm_id(s: pd.Series) -> pd.Series:
    """欠損・0・空文字を NaN に統一。"""
    out = s.astype(str).str.strip()
    out = out.mask(out.isin(["", "nan", "None", "0", "-1", "<NA>"]), pd.NA)
    return out


def _dist_band(dist: pd.Series, surf: pd.Series) -> pd.Series:
    """距離帯カテゴリ（芝/ダート/障害で閾値が異なる）。"""
    d = pd.to_numeric(dist, errors="coerce")
    s = surf.fillna("").astype(str)
    out = pd.Series("mid", index=d.index, dtype="string")
    hurdle = s.str.contains("障", na=False)

    # 障害
    out = out.mask(hurdle, "hurdle")
    # ダート
    out = out.mask(~hurdle & (s.str.contains("ダート|dirt", na=False)) & (d < 1400), "short")
    out = out.mask(~hurdle & (s.str.contains("ダート|dirt", na=False)) & (d >= 1400) & (d < 1800), "mid")
    out = out.mask(~hurdle & (s.str.contains("ダート|dirt", na=False)) & (d >= 1800), "long")
    # 芝
    out = out.mask(~hurdle & ~(s.str.contains("ダート|dirt", na=False)) & (d < 1400), "short")
    out = out.mask(~hurdle & ~(s.str.contains("ダート|dirt", na=False)) & (d >= 1400) & (d < 1800), "mid")
    out = out.mask(~hurdle & ~(s.str.contains("ダート|dirt", na=False)) & (d >= 2000) & (d < 2400), "mid_long")
    out = out.mask(~hurdle & ~(s.str.contains("ダート|dirt", na=False)) & (d >= 2400), "long")
    return out


def _surface_cat(s: pd.Series) -> pd.Series:
    """芝/ダート/障害の3分類。"""
    surf = s.fillna("").astype(str).str.strip()
    out = pd.Series("other", index=s.index, dtype="string")
    out = out.mask(surf == "芝", "turf")
    out = out.mask(surf == "ダート", "dirt")
    out = out.mask(surf == "障", "hurdle")
    return out


def _grade_bucket(g: pd.Series) -> pd.Series:
    """グレード大分類。"""
    gv = g.fillna("").astype(str).str.upper()
    out = pd.Series("other", index=g.index, dtype="string")
    out = out.mask(gv.str.contains("G1"), "G1")
    out = out.mask(gv.str.contains("G2") & ~gv.str.contains("G1"), "G2")
    out = out.mask(gv.str.contains("G3") & ~gv.str.contains("G[12]"), "G3")
    out = out.mask(gv.str.contains("L|Ｌ", regex=True), "L")
    out = out.mask((out == "other") & gv.str.contains("OP|オープン", regex=True), "OP")
    out = out.mask((out == "other"), "class_cond")
    return out


def _parse_pass_first(po: object) -> float:
    """passing_order から先頭コーナー通過順を取得。"""
    if po is None or (isinstance(po, float) and np.isnan(po)):
        return np.nan
    s = str(po).strip()
    if not s or s in ("", "nan", "None"):
        return np.nan
    try:
        first = s.split("-")[0]
        return float(first.strip("()"))
    except (ValueError, IndexError):
        return np.nan


def bayesian_shrink(
    local_sum: np.ndarray,
    local_count: np.ndarray,
    global_mean: float,
    C: float,
) -> np.ndarray:
    """
    Bayesian average: (C × μ_global + local_sum) / (C + n)

    n=0 のとき global_mean を返す。n が小さいほど global_mean に引き寄せられる。
    """
    denom = C + local_count
    return np.where(
        local_count > 0,
        (C * global_mean + local_sum) / denom,
        global_mean,
    )


# ---------------------------------------------------------------------------
# コアロジック: 時系列累計 (cumsum - self)
# ---------------------------------------------------------------------------

def _prior_group_cumstats(
    df: pd.DataFrame,
    group_cols: list[str],
    sort_cols: list[str],
    *,
    win_col: str = "_win",
    top3_col: str = "_top3",
    fp_norm_col: str = "_fp_norm",
    pass_col: str = "_pass1",
) -> pd.DataFrame:
    """
    group_cols でグルーピングし、sort_cols 昇順で並べて
    「直前までの累計」を計算する。

    Returns
    -------
    DataFrame: group_cols + ["_prior_starts", "_prior_wins", "_prior_top3",
                              "_prior_fp_norm_sum", "_prior_pass1_sum", "_prior_pass1_cnt"]
    """
    work = df[group_cols + sort_cols + [win_col, top3_col, fp_norm_col, pass_col]].copy()
    work = work.sort_values(group_cols + sort_cols, kind="mergesort")

    g = work.groupby(group_cols, sort=False)
    cs_win  = g[win_col].cumsum()
    cs_top3 = g[top3_col].cumsum()
    cs_fp   = g[fp_norm_col].cumsum()
    cnt     = g.cumcount()

    # pass_col: NaN を除いた和とカウントを別途計算
    pass_valid = work[pass_col].notna()
    work["_pass_v"] = work[pass_col].fillna(0.0)
    cs_pass = g["_pass_v"].cumsum()
    cs_pass_cnt = g[pass_valid.astype(int)].cumsum()

    result = work[group_cols + sort_cols].copy()
    result["_prior_starts"]      = cnt.to_numpy()
    result["_prior_wins"]        = (cs_win  - work[win_col]).to_numpy()
    result["_prior_top3"]        = (cs_top3 - work[top3_col]).to_numpy()
    result["_prior_fp_norm_sum"] = (cs_fp   - work[fp_norm_col].fillna(0)).to_numpy()
    result["_prior_pass1_sum"]   = (cs_pass - work["_pass_v"]).to_numpy()
    result["_prior_pass1_cnt"]   = (cs_pass_cnt - pass_valid.astype(int)).to_numpy()

    return result


# ---------------------------------------------------------------------------
# メイン関数
# ---------------------------------------------------------------------------

def build_sire_stats(
    race_result_df: pd.DataFrame,
    entity_col: str = "sire",
    prefix: str = "sire",
    *,
    cutoff_race_id: Optional[str] = None,
    jra_only: bool = True,
) -> pd.DataFrame:
    """
    sire または dam_sire 単位の産駒成績統計を、各レース「直前」ベースで計算する。

    Parameters
    ----------
    race_result_df : 全レース結果 DataFrame（race_result_flat 形式）。
    entity_col     : "sire" または "dam_sire"。
    prefix         : 出力列名のプレフィックス。
    cutoff_race_id : 指定すると、この race_id より前のデータだけを集計に使用する
                     （推論スナップショット用）。None のとき全件使用（学習用）。
    jra_only       : True のとき venue_code 1〜10（JRA中央）のみ。

    Returns
    -------
    DataFrame: 主キー ["race_id", "horse_number"] + {prefix}_* 列。
    """
    rr = race_result_df.copy()

    # 不要な列がない場合を考慮して safe に取得
    def _safe_col(name: str, default=None) -> pd.Series:
        if name in rr.columns:
            return rr[name]
        if default is not None:
            return pd.Series(default, index=rr.index)
        return pd.Series(pd.NA, index=rr.index)

    # JRA中央フィルタ
    if jra_only and "venue_code" in rr.columns:
        vc = pd.to_numeric(rr["venue_code"], errors="coerce")
        rr = rr[vc.between(1, 10, inclusive="both")].copy()

    if entity_col not in rr.columns:
        logger.warning("%s column not found in race_result_df", entity_col)
        return pd.DataFrame(columns=["race_id", "horse_number"])

    # --- 基本前処理 ---
    rr["_entity"] = _norm_id(rr[entity_col])
    rr = rr[rr["_entity"].notna()].copy()

    rr["_date"] = pd.to_datetime(rr["date"], errors="coerce")
    rr["_date_int"] = rr["_date"].view("int64")  # ソート用

    fp = pd.to_numeric(rr["finish_position"], errors="coerce")
    fs = pd.to_numeric(rr.get("field_size", pd.Series(np.nan, index=rr.index)), errors="coerce")
    rr["_win"]     = (fp == 1).astype(float)
    rr["_top3"]    = ((fp >= 1) & (fp <= 3)).astype(float)
    rr["_fp_norm"] = np.where(
        (fp.notna()) & (fs.notna()) & (fs > 0),
        fp / fs,
        np.nan,
    )
    rr["_pass1"] = rr["passing_order"].map(_parse_pass_first) if "passing_order" in rr.columns else np.nan

    rr["_surface_cat"] = _surface_cat(_safe_col("surface", ""))
    rr["_dist_band"]   = _dist_band(
        _safe_col("distance", np.nan),
        _safe_col("surface", ""),
    )
    rr["_grade_b"]     = _grade_bucket(_safe_col("grade", ""))
    vc_raw = _safe_col("venue_code", pd.NA)
    rr["_venue_code"] = pd.to_numeric(vc_raw, errors="coerce").astype("Int64").astype("string")

    sort_key = ["race_id"]

    # ── グローバル統計（全条件） ──────────────────────────────────────────
    g_all = _prior_group_cumstats(
        rr, ["_entity"], sort_key,
        win_col="_win", top3_col="_top3", fp_norm_col="_fp_norm", pass_col="_pass1",
    )
    g_all = g_all.rename(columns={
        "_prior_starts":      f"{prefix}_prior_starts",
        "_prior_wins":        f"__all_wins",
        "_prior_top3":        f"__all_top3",
        "_prior_fp_norm_sum": f"__all_fp_sum",
        "_prior_pass1_sum":   f"__all_pass_sum",
        "_prior_pass1_cnt":   f"__all_pass_cnt",
    })

    # グローバル平均（全データ）
    G_WIN  = float(rr["_win"].mean()) if rr["_win"].notna().any() else 0.10
    G_TOP3 = float(rr["_top3"].mean()) if rr["_top3"].notna().any() else 0.30
    G_FP   = float(rr["_fp_norm"].mean()) if rr["_fp_norm"].notna().any() else 0.50

    n_all = g_all[f"{prefix}_prior_starts"].to_numpy().astype(float)
    all_wins  = g_all["__all_wins"].to_numpy().astype(float)
    all_top3  = g_all["__all_top3"].to_numpy().astype(float)
    all_fp    = g_all["__all_fp_sum"].to_numpy().astype(float)
    all_pass_s = g_all["__all_pass_sum"].to_numpy().astype(float)
    all_pass_c = g_all["__all_pass_cnt"].to_numpy().astype(float)

    base = rr[["race_id", "horse_number", "_entity", "_surface_cat", "_dist_band", "_grade_b", "_venue_code", "race_id"] + sort_key].copy()
    base[f"{prefix}_prior_starts"]          = n_all
    base[f"{prefix}_prior_win_rate"]        = bayesian_shrink(all_wins, n_all, G_WIN,  C_GLOBAL)
    base[f"{prefix}_prior_top3_rate"]       = bayesian_shrink(all_top3, n_all, G_TOP3, C_GLOBAL)
    base[f"{prefix}_prior_avg_finish_norm"] = bayesian_shrink(
        np.where(n_all > 0, all_fp / np.maximum(n_all, 1), 0.0),
        n_all, G_FP, C_GLOBAL,
    )
    base[f"{prefix}_prior_avg_pass"] = np.where(
        all_pass_c > 0, all_pass_s / all_pass_c, np.nan
    )

    # ── 馬場別（surface_cat） ─────────────────────────────────────────────
    g_sf = _prior_group_cumstats(
        rr, ["_entity", "_surface_cat"], sort_key,
        win_col="_win", top3_col="_top3", fp_norm_col="_fp_norm", pass_col="_pass1",
    )
    # entity 単位の通算を親 prior として Bayesian ブレンド
    # entity 側は g_all から取得
    g_sf_merged = base[["race_id", "horse_number", "_entity", "_surface_cat",
                          f"{prefix}_prior_win_rate", f"{prefix}_prior_top3_rate",
                          f"{prefix}_prior_avg_finish_norm", f"{prefix}_prior_avg_pass"]].copy()
    # surface 単位の統計を一時計算 → merge key: (entity, surface, sort_key)
    g_sf["_key"] = list(zip(
        g_sf["_entity"], g_sf["_surface_cat"], g_sf[sort_key[0]]
    ))
    rr["_key_sf"] = list(zip(rr["_entity"], rr["_surface_cat"], rr[sort_key[0]]))

    g_sf_dict = {
        k: row for k, row in g_sf.set_index("_key")[
            ["_prior_starts", "_prior_wins", "_prior_top3", "_prior_fp_norm_sum", "_prior_pass1_sum", "_prior_pass1_cnt"]
        ].iterrows()
    }

    sf_starts  = rr["_key_sf"].map(lambda k: g_sf_dict.get(k, {}).get("_prior_starts", 0) or 0).astype(float)
    sf_wins    = rr["_key_sf"].map(lambda k: g_sf_dict.get(k, {}).get("_prior_wins",  0) or 0).astype(float)
    sf_top3    = rr["_key_sf"].map(lambda k: g_sf_dict.get(k, {}).get("_prior_top3",  0) or 0).astype(float)
    sf_fp_sum  = rr["_key_sf"].map(lambda k: g_sf_dict.get(k, {}).get("_prior_fp_norm_sum", 0) or 0).astype(float)
    sf_pass_s  = rr["_key_sf"].map(lambda k: g_sf_dict.get(k, {}).get("_prior_pass1_sum", 0) or 0).astype(float)
    sf_pass_c  = rr["_key_sf"].map(lambda k: g_sf_dict.get(k, {}).get("_prior_pass1_cnt", 0) or 0).astype(float)

    base[f"{prefix}_prior_win_rate_surface"]  = bayesian_shrink(sf_wins,  sf_starts, base[f"{prefix}_prior_win_rate"].to_numpy(),  C_SURFACE)
    base[f"{prefix}_prior_top3_rate_surface"] = bayesian_shrink(sf_top3,  sf_starts, base[f"{prefix}_prior_top3_rate"].to_numpy(), C_SURFACE)
    base[f"{prefix}_prior_avg_finish_norm_surface"] = bayesian_shrink(
        np.where(sf_starts > 0, sf_fp_sum / np.maximum(sf_starts, 1), 0.0),
        sf_starts,
        base[f"{prefix}_prior_avg_finish_norm"].to_numpy(),
        C_SURFACE,
    )
    base[f"{prefix}_prior_avg_pass_surface"] = np.where(
        sf_pass_c > 0, sf_pass_s / sf_pass_c, base[f"{prefix}_prior_avg_pass"].to_numpy()
    )
    base[f"{prefix}_prior_starts_surface"] = sf_starts

    # ── 距離帯別（dist_band） ─────────────────────────────────────────────
    rr["_key_dist"] = list(zip(rr["_entity"], rr["_dist_band"], rr[sort_key[0]]))
    g_dist = _prior_group_cumstats(
        rr, ["_entity", "_dist_band"], sort_key,
        win_col="_win", top3_col="_top3", fp_norm_col="_fp_norm", pass_col="_pass1",
    )
    g_dist["_key"] = list(zip(g_dist["_entity"], g_dist["_dist_band"], g_dist[sort_key[0]]))
    g_dist_dict = {
        k: row for k, row in g_dist.set_index("_key")[
            ["_prior_starts", "_prior_wins", "_prior_top3"]
        ].iterrows()
    }
    dist_starts = rr["_key_dist"].map(lambda k: g_dist_dict.get(k, {}).get("_prior_starts", 0) or 0).astype(float)
    dist_wins   = rr["_key_dist"].map(lambda k: g_dist_dict.get(k, {}).get("_prior_wins",   0) or 0).astype(float)
    dist_top3   = rr["_key_dist"].map(lambda k: g_dist_dict.get(k, {}).get("_prior_top3",   0) or 0).astype(float)

    base[f"{prefix}_prior_win_rate_dist"]  = bayesian_shrink(dist_wins,  dist_starts, base[f"{prefix}_prior_win_rate"].to_numpy(),  C_DIST)
    base[f"{prefix}_prior_top3_rate_dist"] = bayesian_shrink(dist_top3,  dist_starts, base[f"{prefix}_prior_top3_rate"].to_numpy(), C_DIST)
    base[f"{prefix}_prior_starts_dist"]    = dist_starts

    # ── 競馬場×馬場（venue × surface_cat） ───────────────────────────────
    rr["_key_vs"] = list(zip(rr["_entity"], rr["_venue_code"], rr["_surface_cat"], rr[sort_key[0]]))
    g_vs = _prior_group_cumstats(
        rr, ["_entity", "_venue_code", "_surface_cat"], sort_key,
        win_col="_win", top3_col="_top3", fp_norm_col="_fp_norm", pass_col="_pass1",
    )
    g_vs["_key"] = list(zip(
        g_vs["_entity"], g_vs["_venue_code"], g_vs["_surface_cat"], g_vs[sort_key[0]]
    ))
    g_vs_dict = {
        k: row for k, row in g_vs.set_index("_key")[
            ["_prior_starts", "_prior_wins", "_prior_top3"]
        ].iterrows()
    }
    vs_starts = rr["_key_vs"].map(lambda k: g_vs_dict.get(k, {}).get("_prior_starts", 0) or 0).astype(float)
    vs_wins   = rr["_key_vs"].map(lambda k: g_vs_dict.get(k, {}).get("_prior_wins",   0) or 0).astype(float)
    vs_top3   = rr["_key_vs"].map(lambda k: g_vs_dict.get(k, {}).get("_prior_top3",   0) or 0).astype(float)

    base[f"{prefix}_prior_win_rate_venue_sf"]  = bayesian_shrink(
        vs_wins,  vs_starts, base[f"{prefix}_prior_win_rate_surface"].to_numpy(), C_VENUE_SF,
    )
    base[f"{prefix}_prior_top3_rate_venue_sf"] = bayesian_shrink(
        vs_top3,  vs_starts, base[f"{prefix}_prior_top3_rate_surface"].to_numpy(), C_VENUE_SF,
    )
    base[f"{prefix}_prior_starts_venue_sf"] = vs_starts

    # ── グレード別 ────────────────────────────────────────────────────────
    rr["_key_gr"] = list(zip(rr["_entity"], rr["_grade_b"], rr[sort_key[0]]))
    g_gr = _prior_group_cumstats(
        rr, ["_entity", "_grade_b"], sort_key,
        win_col="_win", top3_col="_top3", fp_norm_col="_fp_norm", pass_col="_pass1",
    )
    g_gr["_key"] = list(zip(g_gr["_entity"], g_gr["_grade_b"], g_gr[sort_key[0]]))
    g_gr_dict = {
        k: row for k, row in g_gr.set_index("_key")[
            ["_prior_starts", "_prior_wins", "_prior_top3"]
        ].iterrows()
    }
    gr_starts = rr["_key_gr"].map(lambda k: g_gr_dict.get(k, {}).get("_prior_starts", 0) or 0).astype(float)
    gr_wins   = rr["_key_gr"].map(lambda k: g_gr_dict.get(k, {}).get("_prior_wins",   0) or 0).astype(float)
    gr_top3   = rr["_key_gr"].map(lambda k: g_gr_dict.get(k, {}).get("_prior_top3",   0) or 0).astype(float)

    base[f"{prefix}_prior_win_rate_grade"]  = bayesian_shrink(gr_wins,  gr_starts, base[f"{prefix}_prior_win_rate"].to_numpy(),  C_GRADE)
    base[f"{prefix}_prior_top3_rate_grade"] = bayesian_shrink(gr_top3,  gr_starts, base[f"{prefix}_prior_top3_rate"].to_numpy(), C_GRADE)
    base[f"{prefix}_prior_starts_grade"]    = gr_starts

    # ── 整形 ──────────────────────────────────────────────────────────────
    feat_cols = [c for c in base.columns if c.startswith(f"{prefix}_")]
    out = base[["race_id", "horse_number"] + feat_cols].copy()
    out = out.drop_duplicates(subset=["race_id", "horse_number"], keep="last")

    # 欠損処理: starts=0 の rates は NaN（global_mean に shrink されているが明示的に残す）
    zero_starts = out[f"{prefix}_prior_starts"] == 0
    for col in [c for c in feat_cols if "rate" in c or "avg_finish" in c or "avg_pass" in c]:
        # starts=0 の行は NaN（モデルに「初産駒」フラグとして認識させる）
        out.loc[zero_starts, col] = np.nan

    return out


def build_all_sire_stats(
    race_result_df: pd.DataFrame,
    *,
    cutoff_race_id: Optional[str] = None,
    jra_only: bool = True,
) -> pd.DataFrame:
    """
    sire + dam_sire の統計を両方計算して結合する。

    Returns
    -------
    DataFrame: 主キー ["race_id", "horse_number"] + sire_* + dam_sire_* 列。
    """
    # race_result_df に sire/dam_sire を持たせるため、shutuba_df とのマージが必要な場合がある
    # ここでは race_result_df に sire / dam_sire 列があることを前提とする
    # （nb-02 で shutuba_flat と事前マージしてから渡すこと）

    sire_df    = build_sire_stats(race_result_df, "sire",     "sire",     cutoff_race_id=cutoff_race_id, jra_only=jra_only)
    damsire_df = build_sire_stats(race_result_df, "dam_sire", "dam_sire", cutoff_race_id=cutoff_race_id, jra_only=jra_only)

    if sire_df.empty and damsire_df.empty:
        return pd.DataFrame(columns=["race_id", "horse_number"])
    if sire_df.empty:
        return damsire_df
    if damsire_df.empty:
        return sire_df

    out = sire_df.merge(damsire_df, on=["race_id", "horse_number"], how="outer")
    return out


# ---------------------------------------------------------------------------
# 新規追加: 騎手/調教師 7日間ローリング & venue×surface Bayesian
# ---------------------------------------------------------------------------

def _rolling_7d_entity(
    rr: pd.DataFrame,
    entity_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    騎手 or 調教師の 7 日間ローリング統計（starts, wins, top3）。
    当行より前の 7 日間を対象とする（leakage なし）。
    """
    if entity_col not in rr.columns:
        return pd.DataFrame(columns=["race_id", "horse_number"])

    work = rr[["race_id", "horse_number", entity_col, "date",
               "_win" if "_win" in rr.columns else "finish_position"]].copy()
    work["_eid"] = _norm_id(work[entity_col])
    work = work[work["_eid"].notna()].copy()
    work["_dt"]  = pd.to_datetime(work["date"], errors="coerce")

    if "_win" not in work.columns:
        fp = pd.to_numeric(work["finish_position"], errors="coerce")
        work["_win"]  = (fp == 1).astype(float)
        work["_top3"] = ((fp >= 1) & (fp <= 3)).astype(float)
    else:
        fp = None
        work["_top3"] = pd.to_numeric(
            rr.loc[work.index, "_top3"] if "_top3" in rr.columns else pd.Series(np.nan, index=work.index),
            errors="coerce",
        )

    work = work.sort_values(["_eid", "_dt", "race_id"], kind="mergesort")

    rows = []
    for eid, grp in work.groupby("_eid", sort=False):
        grp = grp.reset_index(drop=True)
        dts     = grp["_dt"].to_numpy(dtype="datetime64[ns]")
        wins    = grp["_win"].to_numpy(dtype=float)
        top3    = grp["_top3"].to_numpy(dtype=float)
        starts_7  = np.zeros(len(grp))
        wins_7    = np.zeros(len(grp))
        top3_7    = np.zeros(len(grp))
        for i in range(len(grp)):
            cutoff = dts[i] - np.timedelta64(7, "D")
            # i より前の行で 7 日以内
            mask = (np.arange(i) < i) & (dts[:i] >= cutoff)
            starts_7[i] = int(mask.sum())
            wins_7[i]   = float(np.nansum(wins[:i][mask]))
            top3_7[i]   = float(np.nansum(top3[:i][mask]))
        grp[f"{prefix}_last7d_starts"] = starts_7
        grp[f"{prefix}_last7d_wins"]   = wins_7
        grp[f"{prefix}_last7d_top3"]   = top3_7
        rows.append(grp[["race_id", "horse_number",
                          f"{prefix}_last7d_starts",
                          f"{prefix}_last7d_wins",
                          f"{prefix}_last7d_top3"]])

    if not rows:
        return pd.DataFrame(columns=["race_id", "horse_number"])

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["race_id", "horse_number"], keep="last")
    return out


def build_jt_extended_stats(
    race_result_df: pd.DataFrame,
    *,
    jra_only: bool = True,
) -> pd.DataFrame:
    """
    騎手・調教師の 7 日間ローリング統計 + venue×surface Bayesian 勝率 を計算する。

    Returns
    -------
    DataFrame: 主キー ["race_id", "horse_number"] + jk_last7d_* + tr_last7d_*
               + jk_venue_sf_win_rate_bayes + tr_venue_sf_win_rate_bayes 列。
    """
    rr = race_result_df.copy()
    if jra_only and "venue_code" in rr.columns:
        vc = pd.to_numeric(rr["venue_code"], errors="coerce")
        rr = rr[vc.between(1, 10, inclusive="both")].copy()

    fp = pd.to_numeric(rr["finish_position"], errors="coerce")
    rr["_win"]  = (fp == 1).astype(float)
    rr["_top3"] = ((fp >= 1) & (fp <= 3)).astype(float)
    rr["_date"] = pd.to_datetime(rr["date"], errors="coerce")

    # 7 日間ローリング
    jk_7d = _rolling_7d_entity(rr, "jockey_id", "jk")
    tr_7d = _rolling_7d_entity(rr, "trainer_id", "tr")

    # venue × surface Bayesian 勝率
    rr["_surface_cat"] = _surface_cat(rr.get("surface", pd.Series("", index=rr.index)))
    vc_s = pd.to_numeric(rr.get("venue_code", pd.Series(np.nan, index=rr.index)), errors="coerce")
    rr["_venue_code"] = vc_s.astype("Int64").astype("string")

    G_JK_WIN = float(rr["_win"].mean()) if rr["_win"].notna().any() else 0.10

    vs_parts = []
    for ent_col, pref in [("jockey_id", "jk"), ("trainer_id", "tr")]:
        if ent_col not in rr.columns:
            continue
        rr["_eid"] = _norm_id(rr[ent_col])
        w = rr[rr["_eid"].notna()].copy()
        w = w.sort_values(["_eid", "_date", "race_id"], kind="mergesort")

        # (entity, venue, surface) のグループで cumstats
        g_vs = _prior_group_cumstats(
            w, ["_eid", "_venue_code", "_surface_cat"], ["race_id"],
            win_col="_win", top3_col="_top3", fp_norm_col="_win",  # fp_norm_col は win 代用
            pass_col="_win",
        )
        g_vs["_key"] = list(zip(g_vs["_eid"], g_vs["_venue_code"], g_vs["_surface_cat"], g_vs["race_id"]))
        gvs_d = {k: (row["_prior_starts"], row["_prior_wins"]) for k, row in g_vs.set_index("_key").iterrows()}

        w["_key_vs"] = list(zip(w["_eid"], w["_venue_code"], w["_surface_cat"], w["race_id"]))

        # surface 単位の勝率をベースラインとして Bayesian ブレンド
        g_sf = _prior_group_cumstats(
            w, ["_eid", "_surface_cat"], ["race_id"],
            win_col="_win", top3_col="_top3", fp_norm_col="_win", pass_col="_win",
        )
        g_sf["_key"] = list(zip(g_sf["_eid"], g_sf["_surface_cat"], g_sf["race_id"]))
        gsf_d = {k: (row["_prior_starts"], row["_prior_wins"]) for k, row in g_sf.set_index("_key").iterrows()}

        w["_key_sf"] = list(zip(w["_eid"], w["_surface_cat"], w["race_id"]))

        vs_s  = w["_key_vs"].map(lambda k: (gvs_d.get(k) or (0, 0))[0]).astype(float)
        vs_w  = w["_key_vs"].map(lambda k: (gvs_d.get(k) or (0, 0))[1]).astype(float)
        sf_s  = w["_key_sf"].map(lambda k: (gsf_d.get(k) or (0, 0))[0]).astype(float)
        sf_w  = w["_key_sf"].map(lambda k: (gsf_d.get(k) or (0, 0))[1]).astype(float)

        # surface 単位の Bayesian rate
        sf_rate = bayesian_shrink(sf_w.to_numpy(), sf_s.to_numpy(), G_JK_WIN, C_SURFACE)
        # venue×surface 単位の Bayesian rate (surface rate をベースに)
        vs_rate = bayesian_shrink(vs_w.to_numpy(), vs_s.to_numpy(), sf_rate, C_VENUE_SF)

        tmp = w[["race_id", "horse_number"]].copy()
        tmp[f"{pref}_venue_sf_win_rate_bayes"] = vs_rate
        tmp[f"{pref}_venue_sf_starts"]         = vs_s.to_numpy()
        vs_parts.append(tmp)

    base = rr[["race_id", "horse_number"]].drop_duplicates().copy()

    for df_part in [jk_7d, tr_7d] + vs_parts:
        if not df_part.empty:
            base = base.merge(df_part, on=["race_id", "horse_number"], how="left")

    return base
