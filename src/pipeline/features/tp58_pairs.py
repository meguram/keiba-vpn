"""tp-58: 数値列 2 本の掛け合わせ列（積・差・対数比）。定数（平均・標準偏差）は 2021 年以前の行だけから決め、推論時に同じ式で再現できる。

基の列名 `rel(<col>)` は『<col> の標準化値 − そのレースの平均』（同じレースの出走馬の同時点の値だけを使う）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLIP = 5.0


def _parse(name: str):
    if name.startswith("rel(") and name.endswith(")"):
        return name[4:-1], True
    return name, False


def fit_constants(df: pd.DataFrame, names: list[str], mask: np.ndarray) -> dict:
    """mask（2021 年以前の行）だけで列ごとの平均・標準偏差・対数の平均・標準偏差を決める。"""
    out = {}
    for n in names:
        col, _ = _parse(n)
        if col in out:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")[mask]
        sd = float(np.nanstd(x))
        d = {"mu": float(np.nanmean(x)), "sd": sd if sd > 1e-12 else 1.0}
        if np.nanmin(x) > 0:
            lx = np.log(x)
            d["lmu"], d["lsd"] = float(np.nanmean(lx)), float(np.nanstd(lx)) or 1.0
        out[col] = d
    return out


def _std(df, name, consts, race_ids):
    col, rel = _parse(name)
    c = consts[col]
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
    z = np.nan_to_num(np.clip((x - c["mu"]) / c["sd"], -CLIP, CLIP), nan=0.0)
    if rel:
        z = z - pd.Series(z).groupby(race_ids).transform("mean").to_numpy()
    return z


def pair_column(df: pd.DataFrame, op: str, a: str, b: str, consts: dict) -> np.ndarray:
    race_ids = df["race_id"].to_numpy()
    if op == "logratio":
        ca, cb = consts[_parse(a)[0]], consts[_parse(b)[0]]
        xa = pd.to_numeric(df[_parse(a)[0]], errors="coerce").to_numpy(dtype="float64")
        xb = pd.to_numeric(df[_parse(b)[0]], errors="coerce").to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            za = np.clip((np.log(xa) - ca["lmu"]) / ca["lsd"], -CLIP, CLIP)
            zb = np.clip((np.log(xb) - cb["lmu"]) / cb["lsd"], -CLIP, CLIP)
        return np.nan_to_num(za - zb, nan=0.0)
    za, zb = _std(df, a, consts, race_ids), _std(df, b, consts, race_ids)
    if op == "prod":
        return za * zb
    if op == "diff":
        return za - zb
    raise ValueError(op)


def column_name(op: str, a: str, b: str) -> str:
    return f"x58_{op}__{a}__{b}"
