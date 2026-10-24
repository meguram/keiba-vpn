"""馬別調教（db.netkeiba.com/horse/training）の追い切りを表にし、時計の特徴を作る（tp-19 用）。

キーは horse_id。調教の各行は「次に出走したレース」に紐づく（race_info の日付）ので、race_date が馬ごとのレース日になる。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

RACE_DATE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


def normalize_course(raw: str) -> tuple[str, str, str]:
    """調教コース表記 → (正規化名, 種別 slope|wood|poly|other, 所属 栗|美|不明)。"""
    s = unicodedata.normalize("NFKC", str(raw or "")).strip()
    if not s:
        return "", "unknown", "不明"
    center = "栗" if s.startswith("栗") or s.startswith("CW") else ("美" if s.startswith(("美", "南")) else "不明")
    if s.endswith("坂"):
        kind = "slope"
    elif "W" in s.upper() or "ウッド" in s:
        kind = "wood"
    elif s.endswith("P") or "ポリ" in s:
        kind = "poly"
    else:
        kind = "other"
    return s, kind, center


def assign_laps(values: list[float], kind: str) -> dict[str, float]:
    """末尾から割り当てる累積タイム。坂路 = 4F,3F,2F,1F / ウッド = 6F,5F,4F,3F,1F（2F は載らない）。矛盾する行は空にする。"""
    v = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    out: dict[str, float] = {}
    order = {"slope": ["t1", "t2", "t3", "t4"], "wood": ["t1", "t3", "t4", "t5", "t6"], "poly": ["t1", "t3", "t4", "t5", "t6"]}.get(kind)
    if order is None or not v:
        return out
    for k, x in zip(order, v[::-1]):
        out[k] = x
    chain = [out[k] for k in ("t6", "t5", "t4", "t3", "t2", "t1") if k in out]
    if any(b >= a for a, b in zip(chain, chain[1:])) or not (8.0 <= out.get("t1", 12.0) <= 20.0):
        return {}
    return out


def entry_to_row(horse_id: str, e: dict[str, Any]) -> dict[str, Any] | None:
    course, kind, center = normalize_course(e.get("course"))
    m = RACE_DATE.search(str(e.get("race_info", "")))
    if not m or not e.get("date"):
        return None
    laps = assign_laps(e.get("lap_times") or [], kind)
    pos = re.sub(r"\D", "", str(e.get("position", "")))
    row: dict[str, Any] = {
        "horse_id": str(horse_id), "train_date": e["date"], "race_date": f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
        "course": course, "kind": kind, "center": center, "cond": unicodedata.normalize("NFKC", str(e.get("track_condition", ""))),
        "rider": str(e.get("rider", "")), "position": int(pos) if pos else np.nan, "leg_color": unicodedata.normalize("NFKC", str(e.get("leg_color", ""))),
        "rank": str(e.get("rank", "")), "evaluation": str(e.get("evaluation", "")),
    }
    for k in ("t1", "t2", "t3", "t4", "t5", "t6"):
        row[k] = laps.get(k, np.nan)
    return row


def record_to_rows(rec: dict | None) -> list[dict[str, Any]]:
    if not rec:
        return []
    hid = rec.get("horse_id", "")
    rows = [entry_to_row(hid, e) for e in rec.get("entries") or []]
    return [r for r in rows if r]


def add_lap_features(t: pd.DataFrame) -> pd.DataFrame:
    """区間ラップと加速度。l1 = 終い1F、lmid = 直前2F の 1F あたり平均（坂路・ウッド共通）、accel = l1 − lmid（負 = 加速）。"""
    t = t.copy()
    t["days_before"] = (pd.to_datetime(t["race_date"]) - pd.to_datetime(t["train_date"])).dt.days
    t["l1"] = t["t1"]
    t["lmid"] = (t["t3"] - t["t1"]) / 2.0
    t["accel"] = t["l1"] - t["lmid"]
    t["l4"] = t["t4"] - t["t3"]                      # 4F 目の 1F ラップ（坂路・ウッド共通）
    t["slope_l2"] = np.where(t["kind"].eq("slope"), t["t2"] - t["t1"], np.nan)
    return t
