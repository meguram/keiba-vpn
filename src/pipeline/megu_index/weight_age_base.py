"""斤量補正の基準斤量（base_weight）。

牡・セン固定 55 kg / 牝固定 53 kg。
年齢・月・出生年によらず一定値とすることで、
weight_dev_kg = jockey_weight − base_weight の解釈を単純化する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_MALE: float = 55.0   # 牡・セン
BASE_FEMALE: float = 53.0  # 牝


def jra_base_weight(sex_group: str) -> float:
    """基準斤量を返す（固定値）。

    Parameters
    ----------
    sex_group : '牡' or '牝'（セン馬は '牡' 扱い）

    Returns
    -------
    float  基準斤量 (kg)
    """
    return BASE_FEMALE if sex_group == "牝" else BASE_MALE


def attach_base_weight(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame に base_weight_kg 列を付与して返す。

    前提列:
      - sex_group : str（'牡' or '牝'）

    付与列:
      - base_weight_kg : float（55.0 or 53.0）
    """
    out = df.copy()
    sex = out["sex_group"].fillna("牡")
    out["base_weight_kg"] = np.where(sex == "牝", BASE_FEMALE, BASE_MALE)
    return out
