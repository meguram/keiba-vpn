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

MEGU_VERSION = "v3-base50-robust1"
CLIP_MIN = 0.0
CLIP_MAX = 100.0
MIN_PAR_SAMPLES = 30
# 基準が L1（場×回り×クラス）または L2（回りを問わない×クラス）で取れた行だけを『高信頼』とみなす
HIGH_CONFIDENCE_LEVELS = ("L1_class", "L2_no_direction")

ROBUST_COLUMNS = [
    "megu_index_raw", "megu_index_robust", "megu_index_status", "low_confidence", "megu_version",
]


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
