"""tp-58: カテゴリの掛け合わせ（騎手×コース など）の as-of 目的変数統計 + 階層縮約。

- 各行について『その行の開催日より前（同日は含めない）』の同じキーの行だけから集計する（同日の別レースも見ない）。
- 子キー（例: 騎手×コース）の平均は親キー（例: 騎手）の平均へ縮約: (S_c + m·p_parent) / (N_c + m)。親も同様に根の事前平均へ縮約。
- 出力: 縮約平均 `<name>_m`、過去件数 `<name>_n`、親との差 `<name>_dev`（交互作用そのもの）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _codes(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros(len(df), dtype="int64")
    return pd.MultiIndex.from_frame(df[cols].astype("string").fillna("∅")).factorize()[0].astype("int64")


def asof_sum_count(keys: np.ndarray, date: np.ndarray, y: np.ndarray, valid: np.ndarray):
    """同じ keys で date が『より前』の行の y の合計と件数（同日を含めない）。valid=False の行は集計に入れない（値は返す）。"""
    d = pd.DataFrame({"k": keys, "d": date, "y": np.where(valid, y, 0.0), "c": valid.astype("float64")})
    day = d.groupby(["k", "d"], sort=True).agg(ys=("y", "sum"), cs=("c", "sum")).reset_index()
    cum_y = day.groupby("k")["ys"].cumsum().to_numpy() - day["ys"].to_numpy()
    cum_c = day.groupby("k")["cs"].cumsum().to_numpy() - day["cs"].to_numpy()
    day["sy"], day["sc"] = cum_y, cum_c
    out = d[["k", "d"]].merge(day[["k", "d", "sy", "sc"]], on=["k", "d"], how="left")
    return out["sy"].to_numpy(), out["sc"].to_numpy()


def shrunk_cross(df: pd.DataFrame, cols: list[str], parent_cols: list[str], y: np.ndarray, valid: np.ndarray, date_col: str = "date",
                 m: float = 20.0, m_parent: float = 50.0, prior: float = 0.1) -> dict[str, np.ndarray]:
    """cols の組を子、parent_cols の組を親として縮約した as-of 平均・件数・親との差を返す。"""
    date = pd.to_datetime(df[date_col]).to_numpy("datetime64[D]").astype("int64")
    y = np.asarray(y, dtype="float64")
    valid = np.asarray(valid, dtype=bool)
    root_y, root_c = asof_sum_count(np.zeros(len(df), "int64"), date, y, valid)
    root = np.where(root_c > 0, root_y / np.maximum(root_c, 1), prior)
    py, pc = asof_sum_count(_codes(df, parent_cols), date, y, valid)
    parent = (py + m_parent * root) / (pc + m_parent)
    cy, cc = asof_sum_count(_codes(df, cols), date, y, valid)
    child = (cy + m * parent) / (cc + m)
    return {"m": child, "n": cc, "dev": child - parent, "parent": parent}
