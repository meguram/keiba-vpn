"""芝道悪適性・ダートタフ適正（走法マップの軸。芝とダートは完全に分けて計算する）。

差 = パワー走の平均成績 − 非パワー走の平均成績（成績 = −(着順−1)/(頭数−1)。大きいほど良い）。
パワー走: 芝はクッション値 ≤ 9.2（水分が多い）、ダートは含水率 < 6.5%（乾いている）。
パワー走と非パワー走がそれぞれ MIN_SIDE 走以上ない馬は「判定不可」で、値は 0 でも 50 でもなく NaN（不明）にする。
予測力は確認できていない（docs/ideas/running_map_axes_fixed.md 10章）。タイプ分類用の軸。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

POWER_TURF_MAX = 9.2        # クッション値がこれ以下ならパワー走
POWER_DIRT_BELOW = 6.5      # 含水率（%）がこれ未満ならパワー走
MIN_SIDE = 3                # パワー走・非パワー走それぞれの最低走数
SHRINK_C = 2.0              # 走数に応じた 0 方向への縮め（有効走数 m に対し m/(m+C)）
LOW_RELIABILITY_BELOW = 6   # 有効走数がこれ未満なら低信頼

AXES = {"芝": ("going_turf", "芝道悪適性"), "ダート": ("going_dirt", "ダートタフ適正")}


def power_flag(surface: pd.Series, x: pd.Series) -> pd.Series:
    """パワー走か。水分の値が無い走は NA（どちら側にも数えない）。"""
    dirt = (surface == "ダート").to_numpy()
    vals = np.where(dirt, x.to_numpy(float) < POWER_DIRT_BELOW, x.to_numpy(float) <= POWER_TURF_MAX)
    return pd.Series(vals, index=x.index, dtype="boolean").mask(x.isna())


def load_moisture(root: Optional[Path] = None, extra_files: Optional[list[str]] = None) -> pd.DataFrame:
    """日×競馬場の馬場水分。列: date, venue, cushion_value, dirt_m（ゴール前と4角の含水率の平均）。extra_files は同形式の補完ファイル（後勝ち）。"""
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    recs: list[dict] = []
    for f in sorted(glob.glob(str(root / "data/page_reference/cushion/cushion_20*.json"))):
        recs += json.load(open(f, encoding="utf-8"))
    for f in extra_files or []:
        recs += json.load(open(f, encoding="utf-8"))
    cu = pd.DataFrame(recs)
    cu["date"] = pd.to_datetime(cu["date"].astype(str), errors="coerce").dt.normalize()
    cu = cu.dropna(subset=["date"]).rename(columns={"venue_name": "venue"}).drop_duplicates(["date", "venue"], keep="last")
    for c in ["cushion_value", "dirt_moisture_goal", "dirt_moisture_4corner"]:
        cu[c] = pd.to_numeric(cu.get(c), errors="coerce")
    cu["dirt_m"] = cu[["dirt_moisture_goal", "dirt_moisture_4corner"]].mean(axis=1)
    return cu[["date", "venue", "cushion_value", "dirt_m"]].reset_index(drop=True)


def attach_moisture(runs: pd.DataFrame, moisture: pd.DataFrame) -> pd.DataFrame:
    """走の表（date, venue, surface, finish_position, field_size）に x（水分）と perf（−着順率）を足す。水分が無い走は x が NaN。"""
    df = runs.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.merge(moisture, on=["date", "venue"], how="left")
    df["x"] = np.where(df["surface"] == "ダート", df["dirt_m"], df["cushion_value"])
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    n = pd.to_numeric(df["field_size"], errors="coerce")
    df["perf"] = np.where((n >= 2) & (fp >= 1), -(fp - 1) / (n - 1).where(n >= 2), np.nan)
    return df


def _horse_diff(g: pd.DataFrame) -> dict:
    flag = power_flag(g["surface"], g["x"])
    is_p = flag.fillna(False).astype(bool)
    is_n = (~flag.fillna(True)).astype(bool)
    p, n = g.loc[is_p, "perf"].dropna(), g.loc[is_n, "perf"].dropna()
    if len(p) < MIN_SIDE or len(n) < MIN_SIDE:
        return {"diff": np.nan, "n_power": len(p), "n_normal": len(n), "m_eff": np.nan}
    return {"diff": p.mean() - n.mean(), "n_power": len(p), "n_normal": len(n), "m_eff": 1.0 / (1.0 / len(p) + 1.0 / len(n))}


def going_scores(runs: pd.DataFrame, horse_ids: list[str], as_of: str | pd.Timestamp) -> pd.DataFrame:
    """対象日 as_of より前の走だけで、馬ごと・面ごとの道悪適性スコア。runs は attach_moisture 済み（horse_id 列つき）。

    列: horse_id, 面ごとに <key>（50+10z。判定不可は NaN）, n_power_<key>, n_normal_<key>, low_<key>（低信頼）。
    標準化は同じ時点で判定可能な全馬の平均・標準偏差（縮めた後）で行う。判定可能な馬が 30 頭未満なら全員 NaN。
    """
    as_of = pd.Timestamp(as_of)
    past = runs[(runs["date"] < as_of) & runs["x"].notna() & runs["perf"].notna()]
    out = pd.DataFrame({"horse_id": [str(h) for h in horse_ids]})
    for surface, (key, _) in AXES.items():
        d = past[past["surface"] == surface]
        rows = {h: _horse_diff(g) for h, g in d.groupby("horse_id")}
        t = pd.DataFrame.from_dict(rows, orient="index", columns=["diff", "n_power", "n_normal", "m_eff"]).astype(float)
        shrunk = t["diff"] * t["m_eff"] / (t["m_eff"] + SHRINK_C)
        ok = shrunk.dropna()
        if len(ok) >= 30 and ok.std() > 0:
            score = 50 + 10 * (shrunk - ok.mean()) / ok.std()
        else:
            score = pd.Series(np.nan, index=t.index)
        ids = out["horse_id"]
        out[key] = ids.map(score)
        out[f"n_power_{key}"] = ids.map(t["n_power"]).fillna(0).astype(int)
        out[f"n_normal_{key}"] = ids.map(t["n_normal"]).fillna(0).astype(int)
        m = ids.map(t["m_eff"])
        out[f"low_{key}"] = m.notna() & (m < LOW_RELIABILITY_BELOW)
    return out
