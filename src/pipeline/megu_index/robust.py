"""めぐ指数の頑健化（外れ値対策）。

従来の ``megu_index`` は、2 着 +2 秒超の走を ``out_of_range`` として NaN にし、極端値はそのまま残していた。
ここでは元の ``megu_index`` / ``computation_status`` を変えずに、下流（成長曲線など）が使う列を足す。

  megu_index_raw      補正後タイムと基準タイムから素直に計算した値（範囲外だった走も含む。丸めなし）
  megu_index_robust   raw を [0, 100] に丸めた値
  megu_index_status   ok / clipped / restored_out_of_range / restored_clipped / no_par / invalid
  low_confidence      基準タイムのサンプルが少ない（< ``MIN_PAR_SAMPLES``）セル、または階層フォールバックで他の条件に寄せた基準
  megu_version        この頑健化を含む計算の版
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.megu_index.common import MEGU_BASE, MEGU_POINTS_PER_SEC

MEGU_VERSION = "v3-base50-robust2-w55"
CLIP_MIN = 0.0
CLIP_MAX = 100.0
MIN_PAR_SAMPLES = 30
# 基準が L1（場×回り×クラス）または L2（回りを問わない×クラス）で取れた行だけを『高信頼』とみなす
HIGH_CONFIDENCE_LEVELS = ("L1_class", "L2_no_direction")

W55_COLUMNS = ["megu_index_w55", "megu_index_w55_robust", "megu_index_w55_half_robust"]

ROBUST_COLUMNS = [
    "megu_index_raw", "megu_index_robust", "megu_index_status", "low_confidence", "megu_version",
    *W55_COLUMNS,
]

# ── 斤量を 55kg に一般化した指数（成長曲線用）──────────────────────────────────────
# 現行の斤量補正は『性別ごとの基準斤量（牝 53 / 牡・セン 55）に換算したタイム』を作る。基準タイム(par)はその換算タイムの
# 2 着馬の平均なので、『全馬 55kg』に変えるには、馬のタイムと par の両方を 55kg に直す必要がある。
#   t_nw      = 走破 − ペース補正 − 馬場補正             （斤量補正の前）
#   corr_w55  = t_nw − β·(斤量 − 55)·(距離/2000)          （全馬 55kg に換算したタイム。斤量欠損は NaN）
#   par_w55   = par + 2β·(距離/2000)·p_牝                  （par の 2 着馬のうち牝の割合 p_牝 だけ、基準が 53→55kg で遅くなる）
#   指数 w55  = 50 + (par_w55 − corr_w55)·10
# 性差の項: 同じ条件なら 牡 は +20β·d·p_牝 点、牝 は −20β·d·(1−p_牝) 点だけ現行の指数から動く（牡と牝の差は 20β·d 点）。
# 『半分版』は β を 0.5 倍にした比較用（par も同じ β で作り直した場合に等しくなるよう、par の 2 着馬の平均斤量偏差 m_dev を使う）。
W55_BASE_KG = 55.0
# 2 着馬（平地、2020〜2025）の 牝の割合 p と 平均斤量偏差 m_dev（斤量−基準）を 面×距離帯で集計した値
W55_PAR_STATS = {
    "ダート|~1200": {"p": 0.4657, "m_dev": 0.9492},
    "ダート|1400-1600": {"p": 0.3220, "m_dev": 1.2531},
    "ダート|1800-2000": {"p": 0.2877, "m_dev": 0.9706},
    "ダート|2200-2400": {"p": 0.0601, "m_dev": 1.2523},
    "ダート|2500~": {"p": 0.0598, "m_dev": 4.5043},
    "芝|~1200": {"p": 0.5787, "m_dev": 0.9924},
    "芝|1400-1600": {"p": 0.5029, "m_dev": 1.1751},
    "芝|1800-2000": {"p": 0.3977, "m_dev": 1.0097},
    "芝|2200-2400": {"p": 0.1987, "m_dev": 1.2428},
    "芝|2500~": {"p": 0.1252, "m_dev": 3.0254},
}
W55_PAR_STATS_ALL = {"p": 0.3782, "m_dev": 1.1682}
_W55_BANDS = ((0, 1300, "~1200"), (1300, 1700, "1400-1600"), (1700, 2100, "1800-2000"), (2100, 2500, "2200-2400"), (2500, 10**9, "2500~"))
_USABLE_STATUS = ("ok", "clipped", "restored_out_of_range", "restored_clipped")


def _w55_stats(surface, distance) -> tuple[float, float]:
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return W55_PAR_STATS_ALL["p"], W55_PAR_STATS_ALL["m_dev"]
    for lo, hi, name in _W55_BANDS:
        if lo <= d < hi:
            s = W55_PAR_STATS.get(f"{surface}|{name}")
            if s:
                return s["p"], s["m_dev"]
    return W55_PAR_STATS_ALL["p"], W55_PAR_STATS_ALL["m_dev"]


def add_w55_columns(df: pd.DataFrame, *, beta_weight: float) -> pd.DataFrame:
    """斤量を 55kg に一般化した指数の列（W55_COLUMNS）を足す。``add_robust_columns`` の後に呼ぶ（行は増減しない）。

    df に必要な列: finish_time_sec, delta_pace_sec, delta_track_sec, jockey_weight_num, dist_scale, par_time_final,
    surface, distance, megu_index_status。指数を付けない行（no_par・invalid・障害 など）と斤量欠損の行は NaN。
    """
    out = df.copy()

    def num(c):
        return pd.to_numeric(out[c], errors="coerce") if c in out.columns else pd.Series(np.nan, index=out.index)

    t_nw = num("finish_time_sec") - num("delta_pace_sec").fillna(0) - num("delta_track_sec").fillna(0)
    w = num("jockey_weight_num")
    ds = num("dist_scale").fillna(1.0)
    par = num("par_time_final")
    st = [_w55_stats(s, d) for s, d in zip(out.get("surface", [None] * len(out)), out.get("distance", [None] * len(out)))]
    p = pd.Series([a for a, _ in st], index=out.index)
    m = pd.Series([b for _, b in st], index=out.index)

    corr55 = t_nw - beta_weight * (w - W55_BASE_KG) * ds
    corr55h = t_nw - 0.5 * beta_weight * (w - W55_BASE_KG) * ds
    par55 = par + 2.0 * beta_weight * ds * p
    par55h = par + beta_weight * ds * (0.5 * m + p)
    raw55 = MEGU_BASE + (par55 - corr55) * MEGU_POINTS_PER_SEC
    raw55h = MEGU_BASE + (par55h - corr55h) * MEGU_POINTS_PER_SEC

    usable = out["megu_index_status"].astype(str).isin(_USABLE_STATUS) & w.notna() & par.notna()
    raw55 = raw55.where(usable)
    raw55h = raw55h.where(usable)
    out["megu_index_w55"] = raw55
    out["megu_index_w55_robust"] = raw55.clip(CLIP_MIN, CLIP_MAX)
    out["megu_index_w55_half_robust"] = raw55h.clip(CLIP_MIN, CLIP_MAX)
    return out


def add_robust_columns(
    df: pd.DataFrame,
    *,
    bad_par: pd.Series | None = None,
    finish_pos: pd.Series | None = None,
) -> pd.DataFrame:
    """``compute_for_dataframe`` の途中結果に頑健化の列を足して返す（行は増減しない）。

    df に必要な列: corrected_time_sec, par_time_final, computation_status。あれば使う列: par_sample_count, par_match_level。
    bad_par: 基準タイムが使えない行（None のとき par_time_final の欠損のみ）。
    finish_pos: 着順（数値）。NaN や 0 以下は『着順なし』として指数を付けない（失格・降着の取り扱い不能な行）。
    """
    out = df.copy()
    corrected = pd.to_numeric(out["corrected_time_sec"], errors="coerce")
    par = pd.to_numeric(out["par_time_final"], errors="coerce")
    raw = MEGU_BASE + (par - corrected) * MEGU_POINTS_PER_SEC

    no_par = par.isna() | corrected.isna()
    if bad_par is not None:
        no_par = no_par | bad_par.reindex(out.index).fillna(True).astype(bool)
    invalid = pd.Series(False, index=out.index)
    if finish_pos is not None:
        fp = pd.to_numeric(finish_pos.reindex(out.index), errors="coerce")
        invalid = fp.isna() | (fp <= 0)

    usable = ~no_par & ~invalid
    raw = raw.where(usable)
    clipped = raw.clip(CLIP_MIN, CLIP_MAX)
    was_clipped = usable & ((raw < CLIP_MIN) | (raw > CLIP_MAX))
    was_oor = out["computation_status"].astype(str).eq("out_of_range")

    status = pd.Series("ok", index=out.index, dtype=object)
    status[was_clipped & ~was_oor] = "clipped"
    status[usable & was_oor & ~was_clipped] = "restored_out_of_range"
    status[usable & was_oor & was_clipped] = "restored_clipped"
    status[no_par] = "no_par"
    status[invalid & ~no_par] = "invalid"

    low = pd.Series(False, index=out.index)
    if "par_sample_count" in out.columns:
        sc = pd.to_numeric(out["par_sample_count"], errors="coerce")
        low = low | (sc < MIN_PAR_SAMPLES)
    if "par_match_level" in out.columns:
        lv = out["par_match_level"].astype(str)
        low = low | (usable & ~lv.isin(HIGH_CONFIDENCE_LEVELS))
    low = low & usable

    out["megu_index_raw"] = raw
    out["megu_index_robust"] = clipped
    out["megu_index_status"] = status
    out["low_confidence"] = low
    out["megu_version"] = MEGU_VERSION
    return out
