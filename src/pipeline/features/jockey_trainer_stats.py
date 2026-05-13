"""
騎手・調教師の成績統計（特徴量ストア外の専用ディレクトリに保存）。

出力先: ``jt_race_features`` は ``data/features/race_horse_tbl/<YYYY>/jt_race_features.parquet``（年別）、
lookup は ``jockey_tbl`` / ``trainer_tbl``（単一ファイル）。
マニフェスト・マージ仕様 JSON は ``data/features/jockey_trainer_stats/``。

定期更新: ``scripts/cron/update_jockey_trainer_stats.sh``（``flock`` で多重起動防止）と
``scripts/cron/setup_jockey_trainer_stats_cron.sh``（cron 登録ヘルパ）。

1. ``jt_race_features.parquet``（各年ディレクトリ内）
   **主マージキー** ``race_id``, ``horse_id``（``JT_RACE_FEATURES_PRIMARY_KEYS``）— 出馬表ベース行への一次 join 用。
   検証・二次マージ用に ``jt_result_date``, ``jt_race_datetime``, ``jt_row_jockey_id``,
   ``jt_row_trainer_id`` を同梱（出馬表の ``jockey_id`` / ``trainer_id`` と照合し、別テーブルの騎手・調教師特徴へ繋ぐ前提）。
   各レース行について **当レースより前** のみを使った統計（学習・検証でリークしない）。
   列接頭辞 ``jk_``（騎手） / ``tr_``（調教師）。
   マージ仕様は ``_merge_spec.json``。

2. ``lookup_jockey.parquet`` / ``lookup_trainer.parquet``
   キー ``jockey_id`` / ``trainer_id``。学習データ全体の **最終更新時点** の累計・直近指標。
   当日予測で ID だけ引く用途向け（当日以前の全履歴で再計算したスナップショット）。

3. ``lookup_jockey_slices.parquet`` / ``lookup_trainer_slices.parquet``
   長形式: ``jockey_id`` / ``trainer_id``, ``slice_kind``, ``slice_key``, ``starts``, ``wins``, ``top3``, ``avg_finish``。
   舞台・斤量帯・グレード帯・季節など **条件別の通算**（推論時マージ用。学習時は ``jt_race_features`` を推奨）。

通過順（``passing_order``）から先頭コーナー位置・セグメント平均・頭数正規化（先頭/field_size）の
騎手・調教師 **直前平均** を付与（``*_avg_pass_*``）。

**欠損方針**: 各ブロックの ``*_starts`` が 0 または欠損の行は、対応する勝数・率・平均着・通過平均を **欠損** にする
（過去データが無いのに 0 が立つことを避ける）。

既存の ``historical_entity_stats``（馬・騎手・調教師の単純累計）と重複する列は避け、
ここではローリング・当日・条件別を中心に付与する。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.features.feature_layout import (
    BLOCK_JOCKEY_TBL,
    BLOCK_RACE_HORSE_TBL,
    BLOCK_TRAINER_TBL,
    FEATURES_DIR,
    JT_STATS_META_SUBDIR,
)
from src.pipeline.features.id_value_policy import sanitize_netkeiba_string_id

logger = logging.getLogger(__name__)

JT_STATS_DIR = Path("data/features") / JT_STATS_META_SUBDIR
MANIFEST_NAME = "_manifest.json"
MERGE_SPEC_NAME = "_merge_spec.json"

# 過去データ（race_result 行）と ``left`` マージする際の主キー（LayerA・学習母表の horse_id と整合）
JT_RACE_FEATURES_PRIMARY_KEYS: tuple[str, ...] = ("race_id", "horse_id")
# 主キーに加え、検証・突合用に同梱する列（出馬表の jockey_id / trainer_id / date と照合可能）
JT_RACE_FEATURES_CONTEXT_COLS: tuple[str, ...] = (
    "jt_result_date",
    "jt_race_datetime",
    "jt_row_jockey_id",
    "jt_row_trainer_id",
)


def _norm_entity_id(s: pd.Series) -> pd.Series:
    """欠損・空・プレースホルダを欠損に。"""
    out = s.astype(str).str.strip()
    out = out.mask(out == "", pd.NA)
    out = out.mask(out.isin(["nan", "None", "0", "-1"]), pd.NA)
    return out


def _jra_mask(df: pd.DataFrame) -> pd.Series:
    if "venue_code" not in df.columns:
        return pd.Series(True, index=df.index)
    vc = pd.to_numeric(df["venue_code"], errors="coerce")
    return vc.between(1, 10, inclusive="both")


def _parse_dt(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["date"], errors="coerce")
    if "start_time" in df.columns:
        st = df["start_time"].astype(str).str.strip()
        st = st.replace({"", "nan", "None"}, pd.NA)
        # "12:35" or "12:35:00"
        tpart = pd.to_datetime(st, format="%H:%M", errors="coerce")
        tpart2 = pd.to_datetime(st, format="%H:%M:%S", errors="coerce")
        tt = tpart.fillna(tpart2)
        if tt.notna().any():
            d = d + pd.to_timedelta(
                tt.dt.hour.fillna(0) * 3600
                + tt.dt.minute.fillna(0) * 60
                + tt.dt.second.fillna(0),
                unit="s",
            )
    return d


def _grade_bucket(g: pd.Series, rc: pd.Series) -> pd.Series:
    g = g.fillna("").astype(str).str.upper()
    rc = rc.fillna("").astype(str)
    out = pd.Series("other", index=g.index, dtype="string")
    out = out.mask(g.str.contains("G1"), "G1")
    out = out.mask(g.str.contains("G2") & ~g.str.contains("G1"), "G2")
    out = out.mask(g.str.contains("G3") & ~g.str.contains("G[12]"), "G3")
    out = out.mask(g.str.contains("L|Ｌ", regex=True), "L")
    out = out.mask((out == "other") & g.str.contains("OP|オープン|公開", regex=True), "OP")
    out = out.mask((out == "other") & rc.str.contains("勝クラ|3勝|2勝|1勝|未勝|新馬", regex=True), "class_nov_cond")
    return out


def _weight_bin_kg(wj: pd.Series, whorse: pd.Series) -> pd.Series:
    """斤量帯（騎手斤量優先、欠損時は負担重量っぽい int 列）。"""
    jw = pd.to_numeric(wj, errors="coerce")
    wh = pd.to_numeric(whorse, errors="coerce")
    w = jw.fillna(wh)
    bins = [-np.inf, 52.0, 54.0, 56.0, 58.0, 60.0, np.inf]
    labels = ["<=52", "53-54", "55-56", "57-58", "59-60", ">=61"]
    return pd.cut(w, bins=bins, labels=labels).astype("string")


def _season_quarter(dt: pd.Series) -> pd.Series:
    m = dt.dt.month
    q = ((m - 1) // 3 + 1).astype("Int64").astype("string")
    return "Q" + q


def _surface_coarse(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str)
    out = pd.Series("other", index=x.index, dtype="string")
    out = out.mask(x.str.contains("芝|草", regex=True), "turf")
    out = out.mask(x.str.contains("ダ|障", regex=True), "dirt_ob")
    return out


def _track_bucket(tc: pd.Series) -> pd.Series:
    x = tc.fillna("").astype(str)
    out = pd.Series("unknown", index=x.index, dtype="string")
    out = out.mask(x.str.contains("良|稍重", regex=True), "firm_good")
    out = out.mask(x.str.contains("重|不", regex=True), "yielding_heavy")
    return out


def _distance_band(dist: pd.Series, surf: pd.Series) -> pd.Series:
    d = pd.to_numeric(dist, errors="coerce")
    sc = _surface_coarse(surf)
    out = pd.Series("unknown", index=d.index, dtype="string")
    sprint = d < 1600
    mile = (d >= 1600) & (d < 2000)
    mid = (d >= 2000) & (d < 2400)
    long_ = d >= 2400
    out = out.mask(sprint, "S_<1600")
    out = out.mask(mile, "M_1600_1999")
    out = out.mask(mid, "I_2000_2399")
    out = out.mask(long_, "L_ge2400")
    return sc.astype(str) + "_" + out.astype(str)


def _finish_numeric(fp: pd.Series) -> pd.Series:
    return pd.to_numeric(fp, errors="coerce")


def _win_top3(fp: pd.Series) -> tuple[pd.Series, pd.Series]:
    fp = _finish_numeric(fp)
    win = (fp == 1).astype(np.float64)
    t3 = ((fp >= 1) & (fp <= 3) & fp.notna()).astype(np.float64)
    return win, t3


def _parse_pass_first_corner(passing_order: Any) -> float:
    """通過順セル先頭（1コーナー相当）を整数位置として返す。欠損は nan。"""
    if passing_order is None or (isinstance(passing_order, float) and np.isnan(passing_order)):
        return float("nan")
    s = str(passing_order).strip()
    if not s or s in ("-", "nan", "None"):
        return float("nan")
    first = s.split("-")[0].strip()
    try:
        return float(int(first))
    except (TypeError, ValueError):
        return float("nan")


def _parse_pass_mean_segments(passing_order: Any) -> float:
    """通過順 ``3-4-4-2`` の数値平均（コーナー間の平均位置の目安）。"""
    if passing_order is None or (isinstance(passing_order, float) and np.isnan(passing_order)):
        return float("nan")
    s = str(passing_order).strip()
    if not s or s in ("-", "nan", "None"):
        return float("nan")
    vals: list[float] = []
    for p in s.split("-"):
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(float(int(p)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _prior_mean_exclude_self(
    work: pd.DataFrame,
    group_key: str,
    val_col: str,
) -> np.ndarray:
    """``group_key`` 内は時刻昇順であること。当行前の有限 ``val_col`` の平均（無ければ nan）。"""
    if val_col not in work.columns:
        return np.full(len(work), np.nan)
    v = pd.to_numeric(work[val_col], errors="coerce")
    valid = v.notna()
    sv = v.where(valid, 0.0)
    cv = valid.astype(np.float64)
    cs_s = sv.groupby(work[group_key]).cumsum().to_numpy()
    cs_c = cv.groupby(work[group_key]).cumsum().to_numpy()
    cur_s = np.where(valid.to_numpy(), v.to_numpy(dtype=np.float64), 0.0)
    cur_c = valid.astype(np.float64).to_numpy()
    prior_s = cs_s - cur_s
    prior_c = cs_c - cur_c
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(prior_c > 0, prior_s / prior_c, np.nan)
    return out


def _apply_no_prior_zeros_to_na_for_prefix(df: pd.DataFrame, col_prefix: str) -> None:
    """各 ``*_starts`` が 0 または欠損の行では、関連する率・着差・通過指標を欠損にする（0 を「データなし」と混同しない）。"""
    starts_cols = [c for c in df.columns if c.startswith(col_prefix) and c.endswith("_starts")]
    suffixes = (
        "_wins",
        "_top3",
        "_win_rate",
        "_top3_rate",
        "_avg_finish",
        "_avg_pass_first",
        "_avg_pass_seg_mean",
        "_avg_pass_norm_first",
    )
    for sc in starts_cols:
        stem = sc[: -len("_starts")]
        m = df[sc].isna() | (pd.to_numeric(df[sc], errors="coerce") == 0)
        df.loc[m, sc] = np.nan
        for suf in suffixes:
            tc = stem + suf
            if tc in df.columns:
                df.loc[m, tc] = np.nan


def _prior_in_sorted_group(
    work: pd.DataFrame,
    group_key: str,
    out_prefix: str,
) -> pd.DataFrame:
    """work は group_key ごとに日付昇順に並んでいること。"""
    fp = _finish_numeric(work["finish_position"])
    work = work.copy()
    work["_win"] = (fp == 1).astype(np.float64)
    work["_t3"] = ((fp >= 1) & (fp <= 3) & fp.notna()).astype(np.float64)
    work["_fp"] = fp.astype(np.float64)

    gb = work.groupby(group_key, sort=False)
    cs_w = gb["_win"].cumsum()
    cs_t3 = gb["_t3"].cumsum()
    cs_fp = gb["_fp"].cumsum()
    cnt = gb.cumcount()

    prior_w = cs_w - work["_win"]
    prior_t3 = cs_t3 - work["_t3"]
    prior_fp = cs_fp - work["_fp"].fillna(0.0)
    prior_n = cnt

    out = work[["race_id", "horse_id"]].copy()
    out[f"{out_prefix}_starts"] = prior_n.astype(np.float64)
    out[f"{out_prefix}_wins"] = prior_w
    out[f"{out_prefix}_top3"] = prior_t3
    pr = prior_n.astype(np.float64).replace(0, np.nan)
    out[f"{out_prefix}_win_rate"] = np.where(np.isfinite(pr) & (pr > 0), prior_w / pr, np.nan)
    out[f"{out_prefix}_top3_rate"] = np.where(np.isfinite(pr) & (pr > 0), prior_t3 / pr, np.nan)
    out[f"{out_prefix}_avg_finish"] = np.where(np.isfinite(pr) & (pr > 0), prior_fp / pr, np.nan)
    return out


def _merge_prior_on_sorted(
    base_sorted: pd.DataFrame,
    group_col: str,
    sub_col: str,
    prefix: str,
) -> pd.DataFrame:
    """base_sorted に sub_col を連結した複合キーで直前累計を付与し、entity 時系列順に戻す。"""
    work = base_sorted.copy()
    work["_sub"] = work[sub_col].astype(str).fillna("_na")
    work["_gk"] = work[group_col].astype(str) + "|" + work["_sub"]
    work = work.sort_values(["_gk", "_dt", "race_id", "horse_id"], kind="mergesort")
    prior = _prior_in_sorted_group(work, "_gk", prefix)
    work[prefix + "_starts"] = prior[f"{prefix}_starts"].to_numpy()
    work[prefix + "_wins"] = prior[f"{prefix}_wins"].to_numpy()
    work[prefix + "_top3"] = prior[f"{prefix}_top3"].to_numpy()
    work[prefix + "_win_rate"] = prior[f"{prefix}_win_rate"].to_numpy()
    work[prefix + "_top3_rate"] = prior[f"{prefix}_top3_rate"].to_numpy()
    work[prefix + "_avg_finish"] = prior[f"{prefix}_avg_finish"].to_numpy()
    work = work.drop(columns=["_sub", "_gk"], errors="ignore")
    return work.sort_values(["_eid", "_dt", "race_id", "horse_id"], kind="mergesort")


def _same_day_prior(base_sorted: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    work = base_sorted.copy()
    dkey = work[group_col].astype(str) + "|" + work["_date_only"].astype(str)
    work["_sdk"] = dkey
    work = work.sort_values(["_sdk", "_dt", "race_id", "horse_id"], kind="mergesort")
    prior = _prior_in_sorted_group(work, "_sdk", prefix)
    for c in list(prior.columns):
        if c not in ("race_id", "horse_id"):
            work[c] = prior[c].to_numpy()
    work = work.drop(columns=["_sdk"], errors="ignore")
    return work.sort_values(["_eid", "_dt", "race_id", "horse_id"], kind="mergesort")


def _rolling_count_starts_shifted(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).count()


def _rolling_sum_shifted(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).sum()


def _rolling_mean_finish_shifted(fp: pd.Series, window: int) -> pd.Series:
    return fp.shift(1).rolling(window, min_periods=1).mean()


def _calendar_roll(
    sub: pd.DataFrame,
    days: int,
    prefix: str,
) -> pd.DataFrame:
    """sub: 単一 entity の行。同一タイムスタンプを避けるため微シフトした索引で rolling。"""
    if sub.empty:
        return pd.DataFrame(
            {
                f"{prefix}_starts": np.array([], dtype=float),
                f"{prefix}_wins": np.array([], dtype=float),
                f"{prefix}_top3": np.array([], dtype=float),
            },
            index=sub.index,
        )
    tmp = sub.sort_values("_dt").copy()
    idx_sorted = tmp.index
    tmp["_dtu"] = tmp["_dt"] + pd.to_timedelta(np.arange(len(tmp), dtype=np.int64), unit="us")
    z = tmp.set_index("_dtu").sort_index()
    w = z["win"].astype(float).shift(1)
    t3 = z["top3"].astype(float).shift(1)
    td = f"{days}D"
    cnt = w.rolling(td, min_periods=1).count()
    wins = w.rolling(td, min_periods=1).sum()
    t3s = t3.rolling(td, min_periods=1).sum()
    block = pd.DataFrame(
        {
            f"{prefix}_starts": cnt.to_numpy(),
            f"{prefix}_wins": wins.to_numpy(),
            f"{prefix}_top3": t3s.to_numpy(),
        },
        index=idx_sorted,
    )
    return block.reindex(sub.index)


def _entity_block_features(
    base: pd.DataFrame,
    entity_col: str,
    col_prefix: str,
) -> pd.DataFrame:
    """base: 全列付き race_result 行、ソート済みで _dt, _date_only, 各種 slice 列あり。"""
    eid = _norm_entity_id(base[entity_col])
    work = base.loc[eid.notna()].copy()
    work["_eid"] = eid[eid.notna()]
    work["_dt"] = base.loc[work.index, "_dt"]
    work["_date_only"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    win, t3 = _win_top3(work["finish_position"])
    work["win"] = win
    work["top3"] = t3
    work["_fp"] = _finish_numeric(work["finish_position"])

    work = work.sort_values(["_eid", "_dt", "race_id", "horse_id"], kind="mergesort")

    # --- 通算直前（entity 全体）---
    prior_all = _prior_in_sorted_group(work, "_eid", f"{col_prefix}_prior_all")
    for c in prior_all.columns:
        if c not in ("race_id", "horse_id"):
            work[c] = prior_all[c].to_numpy()

    # --- 通過位置（当該レースの通過順セル由来。集計は常に当該レースより前のみ）---
    for c in ("_pass1", "_pass_mean_seg", "_pass_norm_first"):
        if c not in work.columns:
            work[c] = np.nan
    work[f"{col_prefix}_prior_all_avg_pass_first"] = _prior_mean_exclude_self(work, "_eid", "_pass1")
    work[f"{col_prefix}_prior_all_avg_pass_seg_mean"] = _prior_mean_exclude_self(
        work, "_eid", "_pass_mean_seg"
    )
    work[f"{col_prefix}_prior_all_avg_pass_norm_first"] = _prior_mean_exclude_self(
        work, "_eid", "_pass_norm_first"
    )

    # --- ローリング N 走 ---
    g = work.groupby("_eid", sort=False, group_keys=False)
    for n in (5, 10, 30):
        work[f"{col_prefix}_roll{n}_starts"] = g["win"].transform(
            lambda s, nn=n: _rolling_count_starts_shifted(s, nn)
        )
        work[f"{col_prefix}_roll{n}_wins"] = g["win"].transform(
            lambda s, nn=n: _rolling_sum_shifted(s, nn)
        )
        work[f"{col_prefix}_roll{n}_top3"] = g["top3"].transform(
            lambda s, nn=n: _rolling_sum_shifted(s, nn)
        )
        work[f"{col_prefix}_roll{n}_avg_finish"] = g["_fp"].transform(
            lambda s, nn=n: _rolling_mean_finish_shifted(s, nn)
        )
        work[f"{col_prefix}_roll{n}_avg_pass_first"] = g["_pass1"].transform(
            lambda s, nn=n: s.shift(1).rolling(nn, min_periods=1).mean()
        )
        work[f"{col_prefix}_roll{n}_avg_pass_seg_mean"] = g["_pass_mean_seg"].transform(
            lambda s, nn=n: s.shift(1).rolling(nn, min_periods=1).mean()
        )
        work[f"{col_prefix}_roll{n}_avg_pass_norm_first"] = g["_pass_norm_first"].transform(
            lambda s, nn=n: s.shift(1).rolling(nn, min_periods=1).mean()
        )

    # --- カレンダー 90D / 365D ---
    cal_parts: list[pd.DataFrame] = []
    for e, sub in work.groupby("_eid", sort=False):
        r90 = _calendar_roll(sub, 90, f"{col_prefix}_cal90")
        r365 = _calendar_roll(sub, 365, f"{col_prefix}_cal365")
        cal_parts.append(r90.join(r365))
    if cal_parts:
        cal_df = pd.concat(cal_parts).reindex(work.index)
        for c in cal_df.columns:
            work[c] = cal_df[c].to_numpy()
    else:
        for tag in ("cal90", "cal365"):
            work[f"{col_prefix}_{tag}_starts"] = np.nan
            work[f"{col_prefix}_{tag}_wins"] = np.nan
            work[f"{col_prefix}_{tag}_top3"] = np.nan

    # --- 当日直前（同一日複数レースの前のみ）---
    work_sd = _same_day_prior(work, "_eid", f"{col_prefix}_same_day")
    sd_cols = [c for c in work_sd.columns if c.startswith(f"{col_prefix}_same_day")]
    work = work.drop(columns=sd_cols, errors="ignore").merge(
        work_sd[["race_id", "horse_id"] + sd_cols],
        on=["race_id", "horse_id"],
        how="left",
    )

    # --- 条件別直前累計 ---
    work = _merge_prior_on_sorted(work, "_eid", "venue_code", f"{col_prefix}_at_venue")
    work = _merge_prior_on_sorted(work, "_eid", "_surface_c", f"{col_prefix}_at_surface")
    work = _merge_prior_on_sorted(work, "_eid", "_dist_band", f"{col_prefix}_at_dist")
    work = _merge_prior_on_sorted(work, "_eid", "_grade_b", f"{col_prefix}_at_grade")
    work = _merge_prior_on_sorted(work, "_eid", "_wbin", f"{col_prefix}_at_weight_bin")
    work = _merge_prior_on_sorted(work, "_eid", "_season_q", f"{col_prefix}_at_season")
    work = _merge_prior_on_sorted(work, "_eid", "_track_b", f"{col_prefix}_at_track_cond")

    # 過去データが無く 0 だけが立つ行は欠損扱い（学習時に「本当のゼロ」と混同しない）
    _apply_no_prior_zeros_to_na_for_prefix(work, col_prefix)

    drop_cols = [
        "_eid",
        "win",
        "top3",
        "_fp",
        "_date_only",
    ]
    feat_cols = [c for c in work.columns if c.startswith(f"{col_prefix}_")]
    out = work[["race_id", "horse_id"] + feat_cols].copy()
    out = out.drop_duplicates(subset=["race_id", "horse_id"], keep="last")
    return out


def attach_jt_race_metadata(feats: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    """主キー行に race_result 由来の日付・時刻・騎手/調教師 ID を付与（任意マージ・検証用）。"""
    if feats.empty:
        return feats
    rr = rr.copy()
    if "_dt" not in rr.columns:
        rr["_dt"] = _parse_dt(rr)
    keys = list(JT_RACE_FEATURES_PRIMARY_KEYS)
    meta_parts = list(dict.fromkeys(keys + [c for c in ("date", "_dt", "jockey_id", "trainer_id") if c in rr.columns]))
    meta = rr[meta_parts].drop_duplicates(subset=keys, keep="last")
    out = feats.merge(meta, on=keys, how="left", validate="one_to_one")
    ren: dict[str, str] = {}
    if "date" in out.columns:
        ren["date"] = "jt_result_date"
    if "_dt" in out.columns:
        ren["_dt"] = "jt_race_datetime"
    if "jockey_id" in out.columns:
        ren["jockey_id"] = "jt_row_jockey_id"
    if "trainer_id" in out.columns:
        ren["trainer_id"] = "jt_row_trainer_id"
    out = out.rename(columns=ren)
    jk_cols = sorted(c for c in out.columns if c.startswith("jk_"))
    tr_cols = sorted(c for c in out.columns if c.startswith("tr_"))
    head = [c for c in (*JT_RACE_FEATURES_PRIMARY_KEYS, *JT_RACE_FEATURES_CONTEXT_COLS) if c in out.columns]
    ordered = head + jk_cols + tr_cols
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest].reset_index(drop=True)


def build_merge_spec() -> dict[str, Any]:
    """各 Parquet のマージキー仕様（リポジトリにコミット可能な静的契約）。"""
    return {
        "schema_version": 1,
        "missing_prior_policy": (
            "各ブロックの *_starts が 0 または欠損のとき、対応する *_wins/*_top3/*_rate/*_avg_finish/*_avg_pass_* を欠損にする。"
            "「過去が無いのに 0」が本当の数値 0 と混ざらないようにする。"
        ),
        "passing_features": (
            "通過順 race_result.passing_order から先頭コーナー位置・セグメント平均・(先頭/頭数) を算出。"
            "jk_/tr_ の *_avg_pass_first|*_avg_pass_seg_mean|*_avg_pass_norm_first を参照。"
        ),
        "jt_race_features": {
            "file_pattern": "race_horse_tbl/{year}/jt_race_features.parquet",
            "file_legacy": "race_horse_tbl/jt_race_features.parquet",
            "row_semantics": "race_result の 1 行（1 馬）に対応。統計は jt_race_datetime より前の完了レースのみ使用。",
            "primary_merge_keys": list(JT_RACE_FEATURES_PRIMARY_KEYS),
            "context_columns": {
                "jt_result_date": "race_result.date（YYYY-MM-DD）",
                "jt_race_datetime": "race_result の日付＋発走時刻に基づく並び用タイムスタンプ",
                "jt_row_jockey_id": "当該結果行の jockey_id（出馬表側 ID との一致確認用）",
                "jt_row_trainer_id": "当該結果行の trainer_id",
            },
            "feature_prefixes": {"jockey": "jk_", "trainer": "tr_"},
        },
        "lookup_jockey": {
            "file": "jockey_tbl/lookup_jockey.parquet",
            "primary_merge_keys": ["jockey_id"],
            "note": "学習用の時点厳密マージは jt_race_features を使用。こちらは推論スナップショット向け。",
        },
        "lookup_trainer": {
            "file": "trainer_tbl/lookup_trainer.parquet",
            "primary_merge_keys": ["trainer_id"],
            "note": "同上",
        },
        "lookup_jockey_slices": {
            "file": "jockey_tbl/lookup_jockey_slices.parquet",
            "primary_merge_keys": ["jockey_id", "slice_kind", "slice_key"],
            "note": "通算スナップショット。学習で条件一致させる場合はリークに注意。",
        },
        "lookup_trainer_slices": {
            "file": "trainer_tbl/lookup_trainer_slices.parquet",
            "primary_merge_keys": ["trainer_id", "slice_kind", "slice_key"],
            "note": "同上",
        },
    }


def load_jt_race_features_from_disk(
    base_dir: str | Path,
    *,
    years: list[str] | None = None,
    jt_path: Path | None = None,
) -> pd.DataFrame:
    """``race_horse_tbl/<YYYY>/jt_race_features.parquet`` を結合して読む。単一ファイルの旧配置もフォールバック。"""
    b = Path(base_dir)
    if jt_path is not None:
        p = Path(jt_path)
        if p.is_file():
            return pd.read_parquet(p)
        raise FileNotFoundError(str(p))

    rhd = b / FEATURES_DIR / BLOCK_RACE_HORSE_TBL
    frames: list[pd.DataFrame] = []
    if years:
        ys = [str(y) for y in years]
    else:
        ys = sorted(
            p.name
            for p in rhd.iterdir()
            if p.is_dir() and len(p.name) == 4 and p.name.isdigit()
        )
    for y in ys:
        p = rhd / y / "jt_race_features.parquet"
        if p.is_file():
            frames.append(pd.read_parquet(p))
    if frames:
        return pd.concat(frames, ignore_index=True)

    flat = rhd / "jt_race_features.parquet"
    if flat.is_file():
        return pd.read_parquet(flat)
    leg = b / JT_STATS_DIR / "jt_race_features.parquet"
    if leg.is_file():
        return pd.read_parquet(leg)
    return pd.DataFrame()


def validate_jt_race_features(df: pd.DataFrame) -> list[str]:
    """``jt_race_features`` のスキーマ検証。問題がなければ空リスト。"""
    errs: list[str] = []
    if df.empty:
        return errs
    for k in JT_RACE_FEATURES_PRIMARY_KEYS:
        if k not in df.columns:
            errs.append(f"missing_primary_key:{k}")
    for k in JT_RACE_FEATURES_CONTEXT_COLS:
        if k not in df.columns:
            errs.append(f"missing_context_col:{k}")
    if not errs:
        dup = df.duplicated(subset=list(JT_RACE_FEATURES_PRIMARY_KEYS), keep=False)
        if dup.any():
            errs.append(f"duplicate_keys:{int(dup.sum())}_rows")
    return errs


def build_jockey_trainer_race_features(race_result_df: pd.DataFrame) -> pd.DataFrame:
    """race_result_flat 相当の DataFrame から (race_id, horse_id) 行の特徴を構築。"""
    if race_result_df.empty:
        return pd.DataFrame()

    rr = race_result_df.copy()
    if "horse_id" not in rr.columns:
        return pd.DataFrame()
    rr["horse_id"] = sanitize_netkeiba_string_id(rr["horse_id"])
    rr = rr.loc[rr["horse_id"].notna()].copy()
    if rr.empty:
        return pd.DataFrame()
    rr["_dt"] = _parse_dt(rr)
    rr["_surface_c"] = _surface_coarse(rr["surface"]) if "surface" in rr.columns else "other"
    rr["_grade_b"] = (
        _grade_bucket(rr["grade"], rr["race_class"])
        if "grade" in rr.columns
        else pd.Series("other", index=rr.index)
    )
    if "race_class" not in rr.columns:
        rr["race_class"] = ""
    rr["_wbin"] = _weight_bin_kg(
        rr["jockey_weight"] if "jockey_weight" in rr.columns else pd.Series(np.nan, index=rr.index),
        rr["weight"] if "weight" in rr.columns else pd.Series(np.nan, index=rr.index),
    )
    rr["_season_q"] = _season_quarter(rr["_dt"])
    rr["_track_b"] = (
        _track_bucket(rr["track_condition"])
        if "track_condition" in rr.columns
        else pd.Series("unknown", index=rr.index)
    )
    rr["_dist_band"] = (
        _distance_band(rr["distance"], rr["surface"])
        if "distance" in rr.columns
        else pd.Series("unknown", index=rr.index)
    )

    if "passing_order" in rr.columns:
        rr["_pass1"] = rr["passing_order"].map(_parse_pass_first_corner)
        rr["_pass_mean_seg"] = rr["passing_order"].map(_parse_pass_mean_segments)
        fs = (
            pd.to_numeric(rr["field_size"], errors="coerce")
            if "field_size" in rr.columns
            else pd.Series(np.nan, index=rr.index)
        )
        p1 = rr["_pass1"].to_numpy(dtype=float)
        fsv = fs.to_numpy(dtype=float)
        rr["_pass_norm_first"] = np.where(
            np.isfinite(p1) & np.isfinite(fsv) & (fsv > 1.0),
            p1 / fsv,
            np.nan,
        )
    else:
        rr["_pass1"] = np.nan
        rr["_pass_mean_seg"] = np.nan
        rr["_pass_norm_first"] = np.nan

    rr = rr.sort_values(["_dt", "race_id", "horse_id"], kind="mergesort")

    jk = _entity_block_features(rr, "jockey_id", "jk")
    tr = _entity_block_features(rr, "trainer_id", "tr")

    base_keys = rr[["race_id", "horse_id"]].drop_duplicates()
    out = base_keys.merge(jk, on=["race_id", "horse_id"], how="left").merge(
        tr, on=["race_id", "horse_id"], how="left"
    )
    return attach_jt_race_metadata(out, rr)


def _lookup_last_row_entity_stats(
    race_result_df: pd.DataFrame,
    feats: pd.DataFrame,
    entity_col: str,
    prefix: str,
    id_out: str,
) -> pd.DataFrame:
    """各 ID の時系列最終レース行に付いた prior 特徴（そのレースより前の累計・ローリング）。"""
    rr = race_result_df.copy()
    rr["_dt"] = _parse_dt(rr)
    eid = _norm_entity_id(rr[entity_col])
    rr = rr.loc[eid.notna()].copy()
    rr["_eid"] = eid[eid.notna()]
    m = rr.merge(feats, on=["race_id", "horse_id"], how="inner")
    m = m.sort_values(["_eid", "_dt", "race_id", "horse_id"], kind="mergesort")
    tail = m.groupby("_eid", sort=False).tail(1)
    cols = [c for c in tail.columns if c.startswith(f"{prefix}_")]
    out = tail[["_eid"] + cols].rename(columns={"_eid": id_out})
    return out.reset_index(drop=True)


def _slice_long_table(
    race_result_df: pd.DataFrame,
    entity_col: str,
    entity_key_out: str,
) -> pd.DataFrame:
    """通算（リーク注意: 推論スナップショット用）の slice_kind / slice_key 別集計。"""
    rr = race_result_df.copy()
    eid = _norm_entity_id(rr[entity_col])
    rr = rr.loc[eid.notna()].copy()
    rr["_eid"] = eid[eid.notna()]
    win, t3 = _win_top3(rr["finish_position"])
    rr["win"] = win
    rr["top3"] = t3
    rr["_fp"] = _finish_numeric(rr["finish_position"])
    rc = rr["race_class"] if "race_class" in rr.columns else pd.Series("", index=rr.index)

    def _agg_slice(kind: str, keys: pd.Series) -> pd.DataFrame:
        tmp = rr.assign(_sk=keys.astype(str).fillna("_na"))
        g = tmp.groupby(["_eid", "_sk"], dropna=False, sort=False)
        out = g.agg(
            starts=("win", "count"),
            wins=("win", "sum"),
            top3=("top3", "sum"),
            avg_finish=("_fp", "mean"),
        ).reset_index()
        out = out.rename(columns={"_eid": entity_key_out, "_sk": "slice_key"})
        out["slice_kind"] = kind
        return out[[entity_key_out, "slice_kind", "slice_key", "starts", "wins", "top3", "avg_finish"]]

    parts: list[pd.DataFrame] = []
    if "venue_code" in rr.columns:
        parts.append(_agg_slice("venue_code", rr["venue_code"]))
    if "surface" in rr.columns:
        parts.append(_agg_slice("surface_coarse", _surface_coarse(rr["surface"])))
    if "grade" in rr.columns:
        parts.append(_agg_slice("grade_bucket", _grade_bucket(rr["grade"], rc)))
    if "jockey_weight" in rr.columns or "weight" in rr.columns:
        jw = rr["jockey_weight"] if "jockey_weight" in rr.columns else pd.Series(np.nan, index=rr.index)
        wh = rr["weight"] if "weight" in rr.columns else pd.Series(np.nan, index=rr.index)
        parts.append(_agg_slice("weight_bin", _weight_bin_kg(jw, wh)))
    parts.append(_agg_slice("season_quarter", _season_quarter(_parse_dt(rr))))
    if "distance" in rr.columns:
        parts.append(_agg_slice("distance_band", _distance_band(rr["distance"], rr["surface"])))
    if "track_condition" in rr.columns:
        parts.append(_agg_slice("track_bucket", _track_bucket(rr["track_condition"])))
    out = pd.concat(parts, ignore_index=True)
    return out


def build_lookup_tables(race_result_df: pd.DataFrame, feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "lookup_jockey": _lookup_last_row_entity_stats(
            race_result_df, feats, "jockey_id", "jk", "jockey_id"
        ),
        "lookup_trainer": _lookup_last_row_entity_stats(
            race_result_df, feats, "trainer_id", "tr", "trainer_id"
        ),
        "lookup_jockey_slices": _slice_long_table(race_result_df, "jockey_id", "jockey_id"),
        "lookup_trainer_slices": _slice_long_table(race_result_df, "trainer_id", "trainer_id"),
    }


def load_race_result_for_stats(
    *,
    years: list[str] | None,
    base_dir: str | Path = ".",
    jra_only: bool = True,
) -> pd.DataFrame:
    from src.pipeline.features.feature_store import FeatureStore

    store = FeatureStore(base_dir=base_dir)
    avail = store.available_years()
    use_years = [str(y) for y in (years or avail)]
    use_years = [y for y in use_years if y in avail]
    cols = [
        "race_id",
        "horse_id",
        "date",
        "start_time",
        "venue_code",
        "surface",
        "distance",
        "direction",
        "grade",
        "race_class",
        "track_condition",
        "finish_position",
        "jockey_id",
        "jockey_name",
        "trainer_id",
        "trainer_name",
        "jockey_weight",
        "weight",
        "passing_order",
        "field_size",
    ]
    frames: list[pd.DataFrame] = []
    for y in use_years:
        sch = store.source_schema("race_result", y)
        want = [c for c in cols if c in sch]
        df = store.load_source("race_result", years=[y], columns=want)
        if not df.empty:
            for c in cols:
                if c not in df.columns:
                    df[c] = pd.NA
            frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    rr = pd.concat(frames, ignore_index=True)
    rr["horse_id"] = sanitize_netkeiba_string_id(rr["horse_id"])
    fp = _finish_numeric(rr["finish_position"])
    rr = rr.loc[fp.notna() & (fp > 0)].copy()
    rr = rr.loc[rr["horse_id"].notna()].copy()
    rr = rr.drop(columns=["horse_number"], errors="ignore")
    if jra_only and "venue_code" in rr.columns:
        rr = rr.loc[_jra_mask(rr)].copy()
    return rr.reset_index(drop=True)


def write_jockey_trainer_stats(
    *,
    years: list[str] | None = None,
    base_dir: str | Path = ".",
    jra_only: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Parquet 群と manifest を書き出す。

    直前累計・ローリングは **tables にある全年** の race_result で計算する（リークなし）。
    ``years`` を指定した場合は ``jt_race_features`` の **行だけ** その年に限定して書き出す。
    ``lookup_*`` は常に全年ベースの最終スナップショット。
    """
    base = Path(base_dir)
    if out_dir is not None:
        meta_dir = Path(out_dir)
        race_horse_dir = Path(out_dir)
        jockey_dir = Path(out_dir)
        trainer_dir = Path(out_dir)
    else:
        feat = base / FEATURES_DIR
        meta_dir = feat / JT_STATS_META_SUBDIR
        race_horse_dir = feat / BLOCK_RACE_HORSE_TBL
        jockey_dir = feat / BLOCK_JOCKEY_TBL
        trainer_dir = feat / BLOCK_TRAINER_TBL
    meta_dir.mkdir(parents=True, exist_ok=True)
    race_horse_dir.mkdir(parents=True, exist_ok=True)
    jockey_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir.mkdir(parents=True, exist_ok=True)

    rr_full = load_race_result_for_stats(years=None, base_dir=base, jra_only=jra_only)
    if rr_full.empty:
        raise RuntimeError("race_result が空のため jockey/trainer 統計を生成できません。")

    feats_full = build_jockey_trainer_race_features(rr_full)
    if years:
        yset = {str(y) for y in years}
        feats = feats_full[
            feats_full["race_id"].astype(str).str[:4].isin(yset)
        ].reset_index(drop=True)
    else:
        feats = feats_full

    if feats.empty:
        raise RuntimeError(
            "jt_race_features が 0 行です。--years の範囲を見直すか、race_result を確認してください。"
        )

    lookups = build_lookup_tables(rr_full, feats_full)

    vjt = validate_jt_race_features(feats)
    if vjt:
        raise ValueError("jt_race_features 検証失敗: " + "; ".join(vjt))

    paths: dict[str, Any] = {}
    yseries = feats["race_id"].astype(str).str.slice(0, 4)
    out_years = sorted({y for y in yseries.unique() if len(str(y)) == 4 and str(y).isdigit()})
    by_year: dict[str, str] = {}
    for y in out_years:
        sub = feats.loc[yseries == y].copy()
        if sub.empty:
            continue
        ydir = race_horse_dir / y
        ydir.mkdir(parents=True, exist_ok=True)
        p_y = ydir / "jt_race_features.parquet"
        sub.to_parquet(p_y, index=False)
        by_year[y] = str(p_y)
    if not by_year:
        raise RuntimeError("jt_race_features の年別書き出しに失敗しました（race_id 年抽出が空）。")
    paths["jt_race_features_by_year"] = by_year
    paths["jt_race_features"] = by_year[sorted(by_year.keys())[0]]

    legacy_flat = race_horse_dir / "jt_race_features.parquet"
    if legacy_flat.is_file():
        legacy_flat.unlink()

    spec_path = meta_dir / MERGE_SPEC_NAME
    spec_path.write_text(
        json.dumps(build_merge_spec(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["merge_spec"] = str(spec_path)

    for name, df in lookups.items():
        if name in ("lookup_jockey", "lookup_jockey_slices"):
            dest = jockey_dir
        elif name in ("lookup_trainer", "lookup_trainer_slices"):
            dest = trainer_dir
        else:
            dest = race_horse_dir
        p = dest / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = str(p)

    mp = meta_dir / MANIFEST_NAME
    paths["manifest"] = str(mp)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "years_all_computed": sorted(rr_full["race_id"].astype(str).str[:4].unique().tolist()),
        "years_jt_race_features_output": sorted(feats["race_id"].astype(str).str[:4].unique().tolist())
        if len(feats)
        else [],
        "years_filter_requested": list(years) if years else None,
        "jra_only": jra_only,
        "rows_race_result_computed": len(rr_full),
        "rows_jt_race_features": len(feats),
        "paths": paths,
        "merge_spec_file": MERGE_SPEC_NAME,
        "jt_race_features_primary_merge_keys": list(JT_RACE_FEATURES_PRIMARY_KEYS),
        "notes": [
            "jt_race_features は当該レースより前のみを使った条件別・ローリング統計（学習用）。",
            "累計の参照期間は rows_race_result_computed に含まれる全年（--years は出力行フィルタのみ）。",
            "lookup_* はデータ全体の最終状態スナップショット（当日予測で ID 参照用）。",
            "lookup_*_slices は条件別通算（学習時の条件一致マージは jt_race_features 側を推奨）。",
        ],
    }
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "jockey/trainer stats 出力: meta=%s jt_years=%s race_rows=%d feature_rows=%d",
        meta_dir,
        ",".join(sorted(by_year.keys())),
        len(rr_full),
        len(feats),
    )
    return manifest


def merge_jt_race_features_into_layer_a(
    layer_df: pd.DataFrame,
    *,
    base_dir: str | Path = ".",
    jt_path: Path | None = None,
    jt_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """jt_race_features（年別 Parquet または ``jt_df``）を主キーで ``left`` マージ。

    ``jt_df`` を渡すとディスクを読まず（テスト・メモリ結合用）。
    ``jt_path`` に単一ファイルを指定した場合はそのみ読み込む（後方互換）。
    """
    if jt_df is not None:
        jt = jt_df
    else:
        b = Path(base_dir)
        layer_years: list[str] | None = None
        if jt_path is None and "race_id" in layer_df.columns:
            layer_years = sorted(
                {y for y in layer_df["race_id"].astype(str).str.slice(0, 4).unique() if len(y) == 4 and y.isdigit()}
            )
        try:
            jt = load_jt_race_features_from_disk(b, years=layer_years, jt_path=jt_path)
        except FileNotFoundError:
            logger.warning("jt_race_features 指定パスが見つかりません（スキップ）")
            return layer_df
        if jt.empty:
            logger.warning("jt_race_features なし（スキップ）")
            return layer_df
    if jt.empty:
        logger.warning("jt_race_features が空です（マージスキップ）")
        return layer_df
    keys = list(JT_RACE_FEATURES_PRIMARY_KEYS)
    for k in keys:
        if k not in layer_df.columns:
            raise ValueError(f"layer_df にマージキー '{k}' がありません")
        if k not in jt.columns:
            raise ValueError(f"jt にマージキー '{k}' がありません")
    verr = validate_jt_race_features(jt)
    if verr:
        raise ValueError("jt_race_features 検証失敗: " + "; ".join(verr))
    return layer_df.merge(jt, on=keys, how="left", validate="many_to_one")
