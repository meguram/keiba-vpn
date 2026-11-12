"""V2: コース別の枠×脚質の 3 着内率を、前週までの開催の結果から算出して出走馬に当てる（記事の主張 C）。

- 窓: 当日の週（月曜始まり）より前の結果だけ。短い窓（同じ場・面、直前 21 日）と長い窓（同じコース＝場・面・距離、直前 365 日）。
- セル: 枠 3 分割（馬番の相対位置）× 脚質 3 分割（a_style の固定の境目）。
- 値: 3 着内かどうか − 頭数ごとの期待（min(3,n)/n）の窓内の和 ÷ (件数 + K)。0 に縮約するので、データの無いセルは 0。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STYLE_CUTS = (0.33, 0.57)   # a_style の下位・中位・上位の境目（固定）
GATE_CUTS = (1 / 3, 2 / 3)
SHORT_DAYS, LONG_DAYS = 21, 365
K_SHORT, K_LONG = 20.0, 40.0
OUT_COLS = ["wkb_cell", "wkb_cell_dev", "wkb_gate", "wkb_style", "wkb_c_cell", "wkb_c_cell_dev", "wkb_c_gate", "wkb_c_style", "wkb_recent_minus_course"]


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    """d: race_id・horse_id・date・venue_code・surface・distance・field_size・horse_number・finish_position・a_style。"""
    d = d.copy()
    g = (d["horse_number"] - 1.0) / (d["field_size"] - 1.0).clip(lower=1.0)
    d["g3"] = np.where(g < GATE_CUTS[0], 0, np.where(g < GATE_CUTS[1], 1, 2))
    d["s3"] = np.where(d["a_style"].isna(), -1, np.where(d["a_style"] < STYLE_CUTS[0], 0, np.where(d["a_style"] < STYLE_CUTS[1], 1, 2)))
    d["exc"] = (d["finish_position"] <= 3).astype(float) - np.minimum(3.0, d["field_size"]) / d["field_size"]
    d.loc[d["finish_position"].isna(), "exc"] = np.nan
    wk = d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")
    d["qdate"] = wk - pd.Timedelta(days=1)   # 前週の日曜まで
    return d


def _window(d: pd.DataFrame, keys: list[str], days: int, k: float) -> np.ndarray:
    """keys のセルごとの、(qdate - days, qdate] の exc の和 ÷ (件数 + k)。d の行順で返す。"""
    h = d[d["exc"].notna() & (d["s3"] >= 0)]
    daily = h.groupby(keys + ["date"], sort=False).agg(s=("exc", "sum"), n=("exc", "size")).reset_index().sort_values("date")
    keyed = daily.sort_values(keys + ["date"])
    keyed[["cs", "cn"]] = keyed.groupby(keys)[["s", "n"]].cumsum()
    keyed = keyed.sort_values("date")
    q = d[keys + ["qdate"]].drop_duplicates().sort_values("qdate")
    q["lo"] = q["qdate"] - pd.Timedelta(days=days)
    cum = keyed[keys + ["date", "cs", "cn"]]
    hi = pd.merge_asof(q, cum.rename(columns={"date": "dh", "cs": "s_hi", "cn": "n_hi"}), left_on="qdate", right_on="dh", by=keys, direction="backward")
    lo = pd.merge_asof(q, cum.rename(columns={"date": "dl", "cs": "s_lo", "cn": "n_lo"}), left_on="lo", right_on="dl", by=keys, direction="backward")
    q["ws"] = hi["s_hi"].fillna(0.0).to_numpy() - lo["s_lo"].fillna(0.0).to_numpy()
    q["wn"] = hi["n_hi"].fillna(0.0).to_numpy() - lo["n_lo"].fillna(0.0).to_numpy()
    q["v"] = q["ws"] / (q["wn"] + k)
    m = d[keys + ["qdate"]].merge(q[keys + ["qdate", "v"]], on=keys + ["qdate"], how="left")
    return m["v"].to_numpy()


def build(d: pd.DataFrame) -> pd.DataFrame:
    """prepare() 済みの全走（履歴を含む）から、各行の OUT_COLS を作る。脚質が無い行の脚質を含む値は欠損。"""
    d = d.reset_index(drop=True)
    vs = ["venue_code", "surface"]
    course = vs + ["distance"]
    f = d[["race_id", "horse_id"]].copy()
    cell = _window(d, vs + ["g3", "s3"], SHORT_DAYS, K_SHORT)
    style = _window(d, vs + ["s3"], SHORT_DAYS, K_SHORT)
    c_cell = _window(d, course + ["g3", "s3"], LONG_DAYS, K_LONG)
    c_style = _window(d, course + ["s3"], LONG_DAYS, K_LONG)
    nos = (d["s3"] < 0).to_numpy()
    for name, v in (("wkb_cell", cell), ("wkb_style", style), ("wkb_c_cell", c_cell), ("wkb_c_style", c_style)):
        f[name] = np.where(nos, np.nan, v)
    f["wkb_cell_dev"] = f["wkb_cell"] - f["wkb_style"]
    f["wkb_c_cell_dev"] = f["wkb_c_cell"] - f["wkb_c_style"]
    f["wkb_gate"] = _window(d, vs + ["g3"], SHORT_DAYS, K_SHORT)
    f["wkb_c_gate"] = _window(d, course + ["g3"], LONG_DAYS, K_LONG)
    f["wkb_recent_minus_course"] = f["wkb_cell"] - f["wkb_c_cell"]
    return f
