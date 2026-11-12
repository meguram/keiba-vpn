"""成績表からレース単位の要約を作り、「前日まで」と「同日の前の R まで」の集計を返す（先読みを仕組みで防ぐ）。

- 前日まで（prior）: 同じ場・同じ面の、直近 PRIOR_DAYS 開催日の日平均。前の開催日が GAP_DAYS 日より前なら欠損（開催が空いた）。
- 同日の途中（day）: 同じ日・同じ場・同じ面で、R が前のレースの平均。同日の他の場・自分の R 以降は使わない。
- 基準値（par）: セルごとの、それより前の日付（同日を含まない）の累積平均。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TABLES_DIR = Path(__file__).resolve().parents[4] / "data/page_reference/tables"
SURFACES = ("芝", "ダート")
PRIOR_DAYS = 3
GAP_DAYS = 14
PAR_MIN_N = 20
KEY = ["venue_code", "surface"]
_COLS = ["race_id", "date", "venue_code", "surface", "distance", "direction", "grade", "round", "field_size", "finish_position", "time_sec", "last_3f", "passing_order",
         "pace", "race_class", "race_name"]


def load_results(years, tables_dir: Path = TABLES_DIR) -> pd.DataFrame:
    parts = [pd.read_parquet(Path(tables_dir) / str(y) / "race_result_flat.parquet", columns=_COLS) for y in years]
    d = pd.concat(parts, ignore_index=True)
    d = d[d["surface"].isin(SURFACES)].copy()
    # 障害は surface が「芝」で記録されることがあるため、クラス名・レース名で除く（race_result_flat は年によって列の値が違う）
    jump = d["race_class"].astype(str).str.contains("障害") | d["race_name"].astype(str).str.contains("障害|ジャンプ")
    d = d[~d["race_id"].isin(d.loc[jump, "race_id"])].drop(columns=["race_class", "race_name"]).copy()
    d["race_id"] = d["race_id"].astype(str)
    d["venue_code"] = d["venue_code"].astype(str)
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
    # race_result_flat の round は 2020〜2023 年が全部 0。同日の R の順序は race_id の末尾 2 桁で決める。
    d["round"] = d["race_id"].str[-2:].astype(int)
    return d


def _corner4_rel(order: str | None, field_size: float) -> float:
    if not isinstance(order, str) or not field_size:
        return np.nan
    m = re.findall(r"\d+", order)
    return int(m[-1]) / field_size if m else np.nan


def _pace_diff(pace: str | None) -> float:
    """前半 3F − 後半 3F（正なら前が遅い＝後傾）。"""
    try:
        p = json.loads(pace) if isinstance(pace, str) else None
        return float(p["first_half_3f"]) - float(p["second_half_3f"]) if p else np.nan
    except Exception:
        return np.nan


def race_summary(res: pd.DataFrame) -> pd.DataFrame:
    """1 レース 1 行: 勝ち馬と 2 着の平均走破時計・上がり、前半の時計、上位 3 頭の 4 角の相対位置、ペース差。"""
    r = res.dropna(subset=["finish_position", "time_sec", "last_3f"]).copy()
    r["c4"] = [_corner4_rel(o, f) for o, f in zip(r["passing_order"], r["field_size"])]
    top2 = r[r["finish_position"].isin([1, 2])]
    g = top2.groupby("race_id")
    base = g.agg(top2_time=("time_sec", "mean"), top2_l3f=("last_3f", "mean"), n_top2=("time_sec", "size"))
    base["top2_front"] = base["top2_time"] - base["top2_l3f"]
    top3 = r[r["finish_position"].isin([1, 2, 3])].groupby("race_id").agg(front_idx=("c4", "mean"))
    win = r[r["finish_position"] == 1].drop_duplicates("race_id").set_index("race_id")
    meta = win[["date", "venue_code", "surface", "distance", "direction", "grade", "round", "field_size"]].copy()
    meta["pace_diff"] = [_pace_diff(p) for p in win["pace"]]
    out = meta.join(base).join(top3).reset_index()
    out = out[out["n_top2"] == 2].drop(columns="n_top2")
    return out.sort_values(["date", "venue_code", "surface", "round", "race_id"]).reset_index(drop=True)


def prior_cell_mean(df: pd.DataFrame, keys: list[str], col: str) -> tuple[pd.Series, pd.Series]:
    """セル（keys）ごとの、それより前の日付（同日を含まない）の累積平均と件数。df の行順で返す。"""
    daily = df.groupby(keys + ["date"], sort=True).agg(s=(col, lambda x: x.sum(min_count=1)), n=(col, "count")).fillna({"s": 0.0}).reset_index()
    daily = daily.sort_values(keys + ["date"])
    cs = daily.groupby(keys)[["s", "n"]].cumsum()
    daily["ps"] = cs["s"] - daily["s"]
    daily["pn"] = cs["n"] - daily["n"]
    m = df[keys + ["date"]].merge(daily[keys + ["date", "ps", "pn"]], on=keys + ["date"], how="left")
    mean = (m["ps"] / m["pn"].where(m["pn"] > 0)).to_numpy()
    return pd.Series(mean, index=df.index), pd.Series(m["pn"].fillna(0).to_numpy(), index=df.index)


def with_par(rs: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    """rs に、列 col の as-of の基準値 par_<name> と、基準との差 dev_<name> を足す。細かいセルで件数が足りなければ粗いセルに落とす。"""
    rs = rs.copy()
    fine = ["venue_code", "surface", "distance", "direction", "grade"]
    coarse = ["surface", "distance", "grade"]
    mf, nf = prior_cell_mean(rs, fine, col)
    mc, _ = prior_cell_mean(rs, coarse, col)
    par = mf.where(nf >= PAR_MIN_N, mc)
    rs[f"par_{name}"] = par
    rs[f"dev_{name}"] = rs[col] - par
    return rs


def prior_and_day(rs: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    """cols の各列について {prefix}_{col}_prior（前日まで）と {prefix}_{col}_day（同日の前の R まで）、_day_n、_prior_gap を返す（race_id 付き）。

    rs は race_summary 相当（date・venue_code・surface・round と cols）。行の順序には依存しない。
    """
    rs = rs.sort_values(["date", "venue_code", "surface", "round", "race_id"]).reset_index(drop=True)
    out = rs[["race_id"]].copy()
    # 同日の前の R まで（自分を含まない累積）
    grp = rs.groupby(KEY + ["date"], sort=False)
    out[f"{prefix}_day_n"] = grp.cumcount()
    for c in cols:
        v = rs[c]
        csum = v.fillna(0.0).groupby([rs[k] for k in KEY] + [rs["date"]], sort=False).cumsum() - v.fillna(0.0)
        cnt = v.notna().astype(int).groupby([rs[k] for k in KEY] + [rs["date"]], sort=False).cumsum() - v.notna().astype(int)
        out[f"{prefix}_{c}_day"] = (csum / cnt.where(cnt > 0)).to_numpy()
    # 前日まで（直近 PRIOR_DAYS 開催日の日平均）
    day = rs.groupby(KEY + ["date"], sort=True)[cols].mean().reset_index().sort_values(KEY + ["date"])
    dg = day.groupby(KEY, sort=False)
    prev = dg["date"].shift(1)
    gap = (day["date"] - prev).dt.days
    # 開催が GAP_DAYS 日より空いたら別の区間とし、窓は区間の中だけ（空いた後の日が、前の開催の日を平均に入れない）
    day["_seg"] = (gap > GAP_DAYS).groupby([day["venue_code"], day["surface"]], sort=False).cumsum()
    sg = day.groupby(KEY + ["_seg"], sort=False)
    prior = pd.DataFrame({"venue_code": day["venue_code"], "surface": day["surface"], "date": day["date"]})
    for c in cols:
        prior[f"{prefix}_{c}_prior"] = sg[c].transform(lambda s: s.rolling(PRIOR_DAYS, min_periods=1).mean().shift(1)).to_numpy()
    prior[f"{prefix}_prior_gap"] = gap.to_numpy()
    m = rs[["race_id", "venue_code", "surface", "date"]].merge(prior, on=["venue_code", "surface", "date"], how="left")
    out = out.merge(m.drop(columns=["venue_code", "surface", "date"]), on="race_id", how="left")
    return out
