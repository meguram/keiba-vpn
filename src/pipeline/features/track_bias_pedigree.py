"""
当日・前日の馬場傾向（芝/ダ別）と血統（父・母父）の preprocess 特徴。

- 馬場傾向: 同一開催日×場×芝/ダで、当該レースより前に終了したレースの勝者上がり3F平均（当日）と、
  前開催日の同条件での勝者上がり3F平均（前日）。
- 血統: 父・母父について、芝/ダ別のリークなし累計勝率（当該レースより過去のみ）。
- 交互作用: 上記勝率 ×（当日傾向 − 前日傾向）。
- 加重ブレンド: 当日・当該レース前の傾向に高い重み、前日に低い重みを付けた合成値
  × 血統勝率（例: 日曜10R 予測では日曜1–9R 芝の情報を強く、土曜全日を弱く反映）。

raw: race_result_flat + race_shutuba_flat（sire / dam_sire）
preprocess: 本モジュール → LayerA 母表 merge

新馬・未出走馬・レース前予測:
  父・母父の芝/ダ別勝率は「産駒全体の過去」から計算するため、当該馬に戦績がなくても定義可能。
  馬場傾向（当日・前日の勝者上がり3F）は、同一開催日内の「確定済みレースの勝者」から累積する。
  レース前に当該レースが race_result に無い場合は、build_track_bias_pedigree_features(..., include_shutuba_entries_without_result=True)
  で出馬表を主軸にマージし、未確定レースにプレースホルダ行を足して race 単位の傾向を付与する（学習用 LayerA の既定は False）。

馬場状態・距離帯:
  - 集計キーに 馬場3区分（良 / 稍重・重 / 不良）・芝・ダ・障・距離4帯（短/マイル/中/長）を含め、
    細かい条件で当日・前日の傾向を取り、欠損時は場×芝ダ障のみの粗い集計でフォールバック。
  - チューニング JSON で馬場区分・距離帯ごとの乗算重みを指定可能。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 馬場傾向の合成: 当日（該当レースより前の同一場・芝/ダ）を強く、前日を弱く
# 例: 東京日曜10R 芝では、日曜1–9R の勝者上がり3F 平均に近い重み、土曜芝は補助
DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT = 0.75
DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT = 0.25


def compute_weighted_blend_series(
    same_day_prior: pd.Series,
    prev_day: pd.Series,
    same_day_weight: float,
    prev_day_weight: float,
) -> pd.Series:
    """
    当日（該当レース前）と前日の勝者上がり3F系統計から加重合成。
    先頭レース（当日分 NaN）は前日のみ。当日のみある場合は当日のみ。
    """
    sdp = same_day_prior
    pdv = prev_day
    if not sdp.index.equals(pdv.index):
        pdv = pdv.reindex(sdp.index)

    sw = float(same_day_weight)
    pw = float(prev_day_weight)
    ssum = sw + pw
    if ssum <= 0:
        sw = DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT
        pw = DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT
        ssum = sw + pw
    sw_n = sw / ssum
    pw_n = pw / ssum

    blended = sw_n * sdp + pw_n * pdv
    blended = blended.where(sdp.notna(), pdv)
    blended = blended.where(~(sdp.notna() & pdv.isna()), sdp)
    return blended


COND_BUCKET_KEYS = ("good", "yielding_soft", "heavy_bad", "unknown")
DIST_BAND_KEYS = ("sprint", "mile", "middle", "long", "unknown")


def track_condition_bucket(tc: Any) -> str:
    """
    馬場3区分: 良 / 稍重・重 / 不良（データ上の 良・稍重・重・不良 を割当）。
    """
    s = str(tc or "").strip()
    if not s:
        return "unknown"
    if s == "良":
        return "good"
    if s in ("稍重", "重"):
        return "yielding_soft"
    if "不" in s or s == "不良":
        return "heavy_bad"
    return "unknown"


def distance_band(distance: Any) -> str:
    """距離4帯（メートル）。"""
    try:
        d = int(float(distance))
    except (TypeError, ValueError):
        return "unknown"
    if d <= 1400:
        return "sprint"
    if d <= 1800:
        return "mile"
    if d <= 2200:
        return "middle"
    return "long"


def stage_key(venue_code: Any, surface: Any, distance: Any) -> str:
    """競馬場 + 芝/ダ/障 + 距離帯（説明・ログ用）。"""
    return f"{_norm_venue(venue_code)}|{surface_group(surface)}|{distance_band(distance)}"


def default_cond_multipliers() -> dict[str, float]:
    return {k: 1.0 for k in COND_BUCKET_KEYS}


def default_dist_multipliers() -> dict[str, float]:
    return {k: 1.0 for k in DIST_BAND_KEYS}


def apply_context_multipliers(
    series: pd.Series,
    cond_bucket: pd.Series,
    dist_band: pd.Series,
    cond_mult: dict[str, float] | None,
    dist_mult: dict[str, float] | None,
) -> pd.Series:
    """馬場区分・距離帯の乗算重み（幾何平均1を想定、unknown は 1.0）。"""
    cm = cond_mult or default_cond_multipliers()
    dm = dist_mult or default_dist_multipliers()
    cb = cond_bucket.astype(str).fillna("unknown")
    db = dist_band.astype(str).fillna("unknown")
    mc = cb.map(lambda x: float(cm.get(x, cm.get("unknown", 1.0))))
    md = db.map(lambda x: float(dm.get(x, dm.get("unknown", 1.0))))
    return series * mc * md


def load_track_bias_weights_config(path: str | Path) -> dict[str, Any]:
    """チューニング JSON 全文（v1/v2 互換）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ver = int(raw.get("version", 1))
    cfg: dict[str, Any] = {
        "version": ver,
        "best_same_day_weight": float(raw["best_same_day_weight"]),
        "best_prev_day_weight": float(raw["best_prev_day_weight"]),
        "cond_multipliers": dict(raw.get("cond_multipliers") or default_cond_multipliers()),
        "dist_multipliers": dict(raw.get("dist_multipliers") or default_dist_multipliers()),
    }
    for k in COND_BUCKET_KEYS:
        cfg["cond_multipliers"].setdefault(k, 1.0)
    for k in DIST_BAND_KEYS:
        cfg["dist_multipliers"].setdefault(k, 1.0)
    return cfg


def load_track_bias_weights_json(path: str | Path) -> tuple[float, float]:
    """後方互換: (same_day, prev_day) のみ。"""
    cfg = load_track_bias_weights_config(path)
    return float(cfg["best_same_day_weight"]), float(cfg["best_prev_day_weight"])


def recompute_weighted_interactions(
    df: pd.DataFrame,
    same_day_weight: float,
    prev_day_weight: float,
    *,
    cond_multipliers: dict[str, float] | None = None,
    dist_multipliers: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    LayerA 行に対し、加重ブレンドと血統×加重馬場の列だけを上書き（他列はそのまま）。
    チューニング済み重みで評価・再ビルドなしで試すときに使う。
    """
    out = df.copy()
    sdp = out["track_bias_winner_3f_same_day_prior"]
    pdv = out["track_bias_winner_3f_prev_day"]
    diff = out["track_bias_winner_3f_diff"]
    wcol = compute_weighted_blend_series(sdp, pdv, same_day_weight, prev_day_weight)
    cb = out.get("track_bias_cond_bucket")
    db = out.get("track_bias_dist_band")
    if cb is not None and db is not None:
        wcol = apply_context_multipliers(
            wcol, cb, db, cond_multipliers, dist_multipliers
        )
    out["track_bias_winner_3f_weighted"] = wcol
    sp = out.get("sire_prior_win_rate_surface")
    dp = out.get("dam_sire_prior_win_rate_surface")
    if sp is not None:
        out["sire_x_track_bias_weighted"] = sp * wcol
    if dp is not None:
        out["dam_sire_x_track_bias_weighted"] = dp * wcol
    if sp is not None and diff is not None:
        out["sire_x_track_bias_diff"] = sp * diff
    if dp is not None and diff is not None:
        out["dam_sire_x_track_bias_diff"] = dp * diff
    return out


def surface_group(surface: Any) -> str:
    """芝 / ダート / 障害 / unknown。"""
    s = str(surface or "").strip()
    if "障" in s:
        return "obstacle"
    if "芝" in s:
        return "turf"
    if "ダ" in s:
        return "dirt"
    return "unknown"


def _norm_venue(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _norm_pedigree_name(v: Any) -> str:
    if pd.isna(v):
        return "__UNKNOWN__"
    t = str(v).strip()
    return t if t else "__UNKNOWN__"


def _time_sort_seconds(t: Any) -> float:
    if pd.isna(t):
        return float("nan")
    s = str(t).strip()
    if not s:
        return float("nan")
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2))
        sec = int(m.group(3) or 0)
        return float(h * 3600 + mi * 60 + sec)
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 4:
        return float(int(digits[:2]) * 3600 + int(digits[2:4]) * 60)
    return float("nan")


def _prior_win_rate_by_id_surface(
    work: pd.DataFrame,
    id_col: str,
    prefix: str,
) -> pd.DataFrame:
    """(id, surface_group) ごとに日付・発走時刻順の累計勝率（当行より前のみ）。"""
    if id_col not in work.columns:
        return pd.DataFrame()

    w = work[
        [
            "race_id",
            "horse_number",
            id_col,
            "date",
            "_d",
            "_st_sec",
            "surface_group",
            "finish_position",
        ]
    ].copy()
    w["_pid"] = w[id_col].map(_norm_pedigree_name)
    w = w.sort_values(
        ["_pid", "surface_group", "_d", "_st_sec", "race_id", "horse_number"],
        kind="mergesort",
    )
    fp = pd.to_numeric(w["finish_position"], errors="coerce")
    w["_win"] = (fp == 1).astype(np.float64)
    g = w.groupby(["_pid", "surface_group"], sort=False)
    cs = g["_win"].cumsum()
    prior_wins = cs - w["_win"]
    prior_races = g.cumcount()
    pr = prior_races.astype(float).replace(0, np.nan).to_numpy()
    pw = prior_wins.to_numpy()
    rate = np.where(np.isfinite(pr) & (pr > 0), pw / pr, np.nan)

    out = w[["race_id", "horse_number"]].copy()
    out[f"{prefix}_prior_win_rate_surface"] = rate
    out = out.drop_duplicates(subset=["race_id", "horse_number"], keep="last")
    return out


def build_track_bias_pedigree_features(
    race_result_df: pd.DataFrame,
    shutuba_df: pd.DataFrame,
    *,
    same_day_weight: float = DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT,
    prev_day_weight: float = DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT,
    cond_multipliers: dict[str, float] | None = None,
    dist_multipliers: dict[str, float] | None = None,
    include_shutuba_entries_without_result: bool = False,
) -> pd.DataFrame:
    """
    race_id + horse_number 単位の特徴1表。
    race_result は全利用年を渡す（履歴・傾向計算のため）。
    shutuba は race_id, horse_number, sire, dam_sire を含むこと。

    include_shutuba_entries_without_result:
      True のとき出馬表を左主軸にマージし、race_result にまだ無い馬（新馬戦のレース前予測など）も行を持つ。
      馬場傾向の「当日累積」には、未確定レース用のプレースホルダ（勝者3F=0・SDP 計算では当行の値は使わない）を 1 行足す。
      前日平均（日次集計）は実勝者のみ（プレースホルダ除外）で計算する。

    same_day_weight / prev_day_weight は正規化して用い、
    track_bias_winner_3f_weighted = w_s×当日該当レース前 + w_p×前日（当日分が無い先頭レースは前日のみ）。
    馬場傾向の集計は 場×芝/ダ/障×馬場3区分×距離4帯で行い、欠損は場×芝/ダ/障のみの集計で補完。
    cond_multipliers / dist_multipliers はチューニング済み乗算（既定 1.0）。
    """
    if race_result_df.empty:
        return pd.DataFrame()
    if shutuba_df.empty:
        return pd.DataFrame()

    rr_cols = [
        c
        for c in (
            "race_id",
            "horse_number",
            "date",
            "venue_code",
            "surface",
            "start_time",
            "finish_position",
            "last_3f",
            "track_condition",
            "distance",
        )
        if c in race_result_df.columns
    ]
    if "race_id" not in rr_cols or "finish_position" not in rr_cols:
        return pd.DataFrame()

    rr_p = race_result_df[rr_cols].drop_duplicates(subset=["race_id", "horse_number"], keep="first")
    base_sb = ("race_id", "horse_number", "sire", "dam_sire")
    extra_sb = ("date", "venue_code", "surface", "start_time", "track_condition", "distance")
    sb_cols = [c for c in base_sb + extra_sb if c in shutuba_df.columns]
    if "sire" not in sb_cols:
        logger.warning("track_bias_pedigree: shutuba に sire が無いため血統特徴をスキップします")
        sb = pd.DataFrame(columns=["race_id", "horse_number"])
    else:
        sb = shutuba_df[sb_cols].drop_duplicates(subset=["race_id", "horse_number"], keep="first")

    if include_shutuba_entries_without_result:
        rr = sb.merge(rr_p, on=["race_id", "horse_number"], how="left")
    else:
        rr = rr_p.merge(sb, on=["race_id", "horse_number"], how="left")
    if "sire" not in rr.columns:
        rr["sire"] = np.nan
    if "dam_sire" not in rr.columns:
        rr["dam_sire"] = np.nan

    for col in ("date", "venue_code", "surface", "start_time", "track_condition", "distance"):
        if col in rr.columns:
            rr[col] = rr.groupby("race_id", sort=False)[col].transform(lambda s: s.ffill().bfill())

    rr["surface_group"] = rr["surface"].map(surface_group)
    rr["venue_code"] = rr["venue_code"].map(_norm_venue)
    rr["_d"] = pd.to_datetime(rr["date"], errors="coerce").dt.normalize()
    rr["_st_sec"] = rr["start_time"].map(_time_sort_seconds) if "start_time" in rr.columns else float("nan")
    tc_col = rr["track_condition"] if "track_condition" in rr.columns else pd.Series("", index=rr.index)
    dist_col = rr["distance"] if "distance" in rr.columns else pd.Series(np.nan, index=rr.index)
    rr["cond_bucket"] = tc_col.map(track_condition_bucket)
    rr["dist_band"] = dist_col.map(distance_band)
    rr["stage_key"] = [
        stage_key(v, s, d)
        for v, s, d in zip(rr["venue_code"], rr["surface"], dist_col)
    ]

    fp_all = pd.to_numeric(rr["finish_position"], errors="coerce")
    w_win = rr.loc[fp_all == 1].copy()
    if not w_win.empty:
        w_win = w_win.drop_duplicates(subset=["race_id"], keep="first")
    w_win["win_3f"] = pd.to_numeric(w_win.get("last_3f"), errors="coerce")

    g_fine = ["_d", "venue_code", "surface_group", "cond_bucket", "dist_band"]
    g_coarse = ["_d", "venue_code", "surface_group"]

    if w_win.empty and not include_shutuba_entries_without_result:
        return pd.DataFrame()

    w_ext = w_win.copy()
    if include_shutuba_entries_without_result and not sb.empty:
        have = set(w_win["race_id"].astype(str).unique()) if not w_win.empty else set()
        need = set(rr["race_id"].astype(str).unique()) - have
        for rid in need:
            sub = rr.loc[rr["race_id"].astype(str) == str(rid)]
            if "horse_number" in sub.columns:
                sub = sub.assign(
                    _hn=pd.to_numeric(sub["horse_number"], errors="coerce")
                ).sort_values("_hn", kind="mergesort").drop(columns=["_hn"])
            if sub.empty:
                continue
            row = sub.iloc[:1].copy()
            row["win_3f"] = 0.0
            w_ext = pd.concat([w_ext, row], ignore_index=True)

    if w_ext.empty:
        return pd.DataFrame()

    w_ext = w_ext.sort_values(
        g_fine + ["_st_sec", "race_id"],
        kind="mergesort",
    )

    def _same_day_prior_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding().mean()

    w_ext["sdp_fine"] = w_ext.groupby(g_fine, sort=False)["win_3f"].transform(_same_day_prior_mean)
    w_ext["sdp_coarse"] = w_ext.groupby(g_coarse, sort=False)["win_3f"].transform(_same_day_prior_mean)
    w_ext["track_bias_winner_3f_same_day_prior"] = w_ext["sdp_fine"].fillna(w_ext["sdp_coarse"])

    if not w_win.empty:
        daily_fine = (
            w_win.groupby(g_fine, as_index=False)["win_3f"]
            .mean()
            .rename(columns={"win_3f": "_daily_mean_win_3f"})
        )
        daily_coarse = (
            w_win.groupby(g_coarse, as_index=False)["win_3f"]
            .mean()
            .rename(columns={"win_3f": "_daily_mean_win_3f_coarse"})
        )
    else:
        daily_fine = pd.DataFrame(columns=g_fine + ["_daily_mean_win_3f"])
        daily_coarse = pd.DataFrame(columns=g_coarse + ["_daily_mean_win_3f_coarse"])

    daily_fine_prev = daily_fine.copy()
    if not daily_fine_prev.empty:
        daily_fine_prev["_d"] = daily_fine_prev["_d"] + pd.Timedelta(days=1)
        daily_fine_prev = daily_fine_prev.rename(columns={"_daily_mean_win_3f": "pdv_fine"})

    daily_coarse_prev = daily_coarse.copy()
    if not daily_coarse_prev.empty:
        daily_coarse_prev["_d"] = daily_coarse_prev["_d"] + pd.Timedelta(days=1)
        daily_coarse_prev = daily_coarse_prev.rename(
            columns={"_daily_mean_win_3f_coarse": "pdv_coarse"}
        )

    w = w_ext
    if not daily_fine_prev.empty:
        w = w.merge(daily_fine_prev, on=g_fine, how="left")
    else:
        w["pdv_fine"] = np.nan
    if not daily_coarse_prev.empty:
        w = w.merge(daily_coarse_prev, on=g_coarse, how="left")
    else:
        w["pdv_coarse"] = np.nan
    w["track_bias_winner_3f_prev_day"] = w["pdv_fine"].fillna(w["pdv_coarse"])

    w["track_bias_winner_3f_diff"] = w["track_bias_winner_3f_same_day_prior"] - w[
        "track_bias_winner_3f_prev_day"
    ]

    w["track_bias_winner_3f_weighted"] = compute_weighted_blend_series(
        w["track_bias_winner_3f_same_day_prior"],
        w["track_bias_winner_3f_prev_day"],
        same_day_weight,
        prev_day_weight,
    )
    w["track_bias_winner_3f_weighted"] = apply_context_multipliers(
        w["track_bias_winner_3f_weighted"],
        w["cond_bucket"],
        w["dist_band"],
        cond_multipliers,
        dist_multipliers,
    )

    race_bias = w[
        [
            "race_id",
            "track_bias_winner_3f_same_day_prior",
            "track_bias_winner_3f_prev_day",
            "track_bias_winner_3f_diff",
            "track_bias_winner_3f_weighted",
            "cond_bucket",
            "dist_band",
            "stage_key",
        ]
    ].drop_duplicates(subset=["race_id"], keep="first")
    race_bias = race_bias.rename(
        columns={
            "cond_bucket": "track_bias_cond_bucket",
            "dist_band": "track_bias_dist_band",
            "stage_key": "track_bias_stage_key",
        }
    )

    out = rr[["race_id", "horse_number"]].drop_duplicates().merge(race_bias, on="race_id", how="left")

    sire_f = _prior_win_rate_by_id_surface(rr, "sire", "sire")
    dam_f = _prior_win_rate_by_id_surface(rr, "dam_sire", "dam_sire")
    if not sire_f.empty:
        out = out.merge(sire_f, on=["race_id", "horse_number"], how="left")
    if not dam_f.empty:
        out = out.merge(dam_f, on=["race_id", "horse_number"], how="left")

    diff = out["track_bias_winner_3f_diff"]
    sw = out.get("sire_prior_win_rate_surface")
    dw = out.get("dam_sire_prior_win_rate_surface")
    if sw is not None:
        out["sire_x_track_bias_diff"] = sw * diff
    if dw is not None:
        out["dam_sire_x_track_bias_diff"] = dw * diff

    wcol = out.get("track_bias_winner_3f_weighted")
    if wcol is not None and sw is not None:
        out["sire_x_track_bias_weighted"] = sw * wcol
    if wcol is not None and dw is not None:
        out["dam_sire_x_track_bias_weighted"] = dw * wcol

    feat_cols = [c for c in out.columns if c not in ("race_id", "horse_number")]
    logger.info(
        "track_bias_pedigree: %d 行, columns=%s",
        len(out),
        feat_cols,
    )
    return out


def merge_track_bias_pedigree_into_layer_a(
    layer_df: pd.DataFrame,
    race_result_full: pd.DataFrame,
    shutuba_full: pd.DataFrame,
    *,
    same_day_weight: float = DEFAULT_TRACK_BIAS_SAME_DAY_WEIGHT,
    prev_day_weight: float = DEFAULT_TRACK_BIAS_PREV_DAY_WEIGHT,
    cond_multipliers: dict[str, float] | None = None,
    dist_multipliers: dict[str, float] | None = None,
    include_shutuba_entries_without_result: bool = False,
) -> pd.DataFrame:
    feat = build_track_bias_pedigree_features(
        race_result_full,
        shutuba_full,
        same_day_weight=same_day_weight,
        prev_day_weight=prev_day_weight,
        cond_multipliers=cond_multipliers,
        dist_multipliers=dist_multipliers,
        include_shutuba_entries_without_result=include_shutuba_entries_without_result,
    )
    if feat.empty:
        return layer_df
    drop_cols = [c for c in feat.columns if c not in ("race_id", "horse_number")]
    layer_df = layer_df.drop(columns=[c for c in drop_cols if c in layer_df.columns], errors="ignore")
    return layer_df.merge(feat, on=["race_id", "horse_number"], how="left")
