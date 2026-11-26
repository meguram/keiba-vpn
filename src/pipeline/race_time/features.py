"""基準の走破時計の関数の入力（レースごと）。読むのは手元の確定結果の表（page_reference/tables/<年>/race_result_flat.parquet）だけ。

- 目的の値: 1〜5 着の走破時計の中央値（主）と、勝ち時計（副）。着順の無い馬（取消・中止・除外）は使わない。
- 前半の速さ: ラップを累積し、600m の通過時計を線形の補間で求める。結果表の pace.first_half_3f は使わない
  （1700・1900・2100m などは最初の区間が 100m で、500m 分の時計になっている。2026-11-26 の調べ）。
- クラス: 未勝利 1・1勝 2・2勝 3・3勝 4・OP/L 5・G3 6・G2 7・G1 8 の順序。新馬は未勝利と同じ 1 にして、別の印（newcomer）を立てる。
- 年齢の条件: 2 歳 0・3 歳 1・3 歳以上と 4 歳以上 2 の順序（race_class の文字から）。
- 障害は除く。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

CLASS_ORD = {"新馬": 1, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "L": 5, "G3": 6, "G2": 7, "G1": 8}
CLASS_LABEL = {1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝", 5: "OP/L", 6: "G3", 7: "G2", 8: "G1"}
AGE_LABEL = {0: "2歳", 1: "3歳", 2: "古馬（3歳以上・4歳以上）"}
SURF = {"芝": "芝", "ダート": "ダ", "ダ": "ダ"}


def first_lap_len(distance: int) -> int:
    """最初の区間の長さ（m）。JRA のラップは 200m ごとで、端数（100m・50m）は最初の区間になる。"""
    r = int(distance) % 200
    return r if r else 200


def front_time_at(lap_times: Sequence[float] | None, distance: int, at_m: float = 600.0) -> float | None:
    """ラップから、スタートから at_m（m）の通過時計（秒）を求める。区間の中は等速とみなして線形に補間する。
    ラップの数が距離と合わない・足りないときは None。"""
    if lap_times is None:
        return None
    try:
        laps = [float(x) for x in lap_times]
    except (TypeError, ValueError):
        return None
    if not laps or any(not math.isfinite(x) or x <= 0 for x in laps):
        return None
    d = int(distance)
    lens = [first_lap_len(d)] + [200] * (len(laps) - 1)
    if sum(lens) != d or at_m > d:
        return None
    dist = t = 0.0
    for ln, lt in zip(lens, laps):
        if dist + ln >= at_m - 1e-9:
            return round(t + lt * (at_m - dist) / ln, 4)
        dist += ln
        t += lt
    return None


def age_ord(race_class: str | None) -> int | None:
    s = str(race_class or "")
    if "２歳" in s or "2歳" in s:
        return 0
    if "以上" in s:
        return 2
    if "３歳" in s or "3歳" in s:
        return 1
    return None


def _laps(v: Any) -> list | None:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return None
    try:
        return list(v)
    except TypeError:
        return None


def race_table(years: Iterable[int], tables_root: Path) -> pd.DataFrame:
    """レースごとの入力の表。列: race_id, date, venue, surf, distance, course, class_ord, newcomer, age_ord, field_size,
    front600, y_top5, y_win, y_2nd, n_finish, track_condition。"""
    cols = ["race_id", "date", "venue", "surface", "distance", "grade", "race_class", "field_size", "lap_times",
            "finish_position", "time_sec", "track_condition"]
    rows = []
    for y in years:
        p = Path(tables_root) / str(y) / "race_result_flat.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=cols)
        df = df[df["surface"].isin(["芝", "ダート"])]
        for rid, g in df.groupby("race_id", sort=False):
            f = g[(g["finish_position"] > 0) & (g["time_sec"] > 0)].sort_values("finish_position")
            if len(f) < 3:
                continue
            h = g.iloc[0]
            co = CLASS_ORD.get(str(h["grade"]))
            ao = age_ord(h["race_class"])
            dist = int(h["distance"])
            fr = front_time_at(_laps(h["lap_times"]), dist, 600.0)
            if co is None or ao is None:
                continue
            surf = SURF[str(h["surface"])]
            rows.append({
                "race_id": str(rid), "date": str(h["date"])[:10].replace("-", ""), "venue": str(h["venue"]), "surf": surf,
                "distance": dist, "course": f"{h['venue']}|{surf}|{dist}", "class_ord": co, "newcomer": int(str(h["grade"]) == "新馬"),
                "age_ord": ao, "field_size": int(h["field_size"] or len(g)), "front600": fr,
                "y_top5": float(np.median(f["time_sec"].head(5))), "y_win": float(f["time_sec"].iloc[0]),
                "y_2nd": float(f["time_sec"].iloc[1]), "n_finish": int(len(f)),
                "track_condition": str(h["track_condition"] or ""),
            })
    return pd.DataFrame(rows)


def distance_band(distance: int) -> str:
    d = int(distance)
    return "~1300" if d <= 1300 else "1400-1600" if d <= 1600 else "1700-2000" if d <= 2000 else "2100-2500" if d <= 2500 else "2600~"
