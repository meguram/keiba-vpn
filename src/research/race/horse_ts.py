"""実測の馬別ラップから、コース単位の TS 偏差値を作る。

馬ごとのピーク = ラスト 3 区間（200m）の最速ラップ。中止・取消（着順 0・区間が 0 秒）は除く。
レースごとに lap_correction のモデル D（同日の残差 + ペース）で水準のずれ（秒）を推定して引き、
コース（競馬場・芝ダ・距離）の補正後ピークの平均・標準偏差から偏差値にする（速いほど高い）。
馬場状態の区分は使わない。クラスも使わない。

python -m src.research.race.horse_ts --laps /home/jovyan/acq/horse_lap/horse_lap_2026.jsonl ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.research.race import lap_correction as lc
except ImportError:     # ホストでは未コミットのため /home/jovyan/acq に並べて置く
    import lap_correction as lc

COURSE = ["venue", "surface", "distance"]
SHRINK_MEAN = 100.0     # コースの頭数が少ないとき、同じ芝ダ・距離帯の平均へ寄せる強さ（頭数換算）
SHRINK_SD = 200.0


def _shrunk_dev(d: pd.DataFrame, col: str) -> pd.Series:
    """コースの平均を引いた偏差。レース数が少ないコースは、同じ芝ダ・距離帯の平均へ寄せた基準で引く。"""
    g = d.groupby(COURSE)[col]
    n, mu = g.transform("size"), g.transform("mean")
    pool = d.groupby(["surface", d["distance"] // 400])[col].transform("mean")
    return d[col] - (n * mu + lc.MIN_COURSE_RACES * pool) / (n + lc.MIN_COURSE_RACES)


def race_offsets(years: list[int]) -> pd.DataFrame:
    """レースごとの水準のずれ（秒）= b_day × 同日の残差 + b_pace × ペース偏差。芝ダ別に全期間で当てる。
    レース数が少ないコースも対象にする（基準を距離帯の平均へ寄せる）。係数はレース数が十分あるコースだけで当てる。"""
    parts = []
    for surf in ("芝", "ダート"):
        d = lc.race_table(years).query("surface == @surf").copy()
        d["peak_dev"], d["pace_dev"] = _shrunk_dev(d, "peak"), _shrunk_dev(d, "pace")
        big = d.groupby(COURSE)["race_id"].transform("size") >= lc.MIN_COURSE_RACES
        bp = lc.ols(d[big], ["pace_dev"])
        d["res_pace"] = d["peak_dev"] - bp[0] - bp[1] * d["pace_dev"]
        d["day_pace"] = lc.loo_day(d, "res_pace")
        b = lc.ols(d[big], ["day_pace", "pace_dev"])
        d["offset"] = b[1] * d["day_pace"] + b[2] * d["pace_dev"]
        parts.append(d[["race_id", "offset"]])
    return pd.concat(parts, ignore_index=True)


def horse_peaks(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            for e in r.get("entries", []):
                laps = [s["lap_time"] for s in e.get("sections", [])]
                if not e.get("finish_order") or len(laps) < 6 or not all(v and v > 0 for v in laps):
                    continue
                rows.append({"race_id": str(r["race_id"]), "horse_id": str(e["horse_id"]), "finish_order": e["finish_order"],
                             "peak": min(laps[-3:])})
    return pd.DataFrame(rows)


def course_ts(horses: pd.DataFrame) -> pd.DataFrame:
    """horses: race_id, horse_id, peak, offset, venue, surface, distance → ts（コース単位の偏差値）を足して返す。"""
    h = horses.dropna(subset=["offset"]).copy()
    h["x"] = h["peak"] - h["offset"]
    h["band"] = (h["distance"] // 400 * 400).astype(int)
    g = h.groupby(COURSE)["x"]
    c = pd.DataFrame({"n": g.size(), "mu": g.mean(), "sd": g.std()}).reset_index()
    c["band"] = (c["distance"] // 400 * 400).astype(int)
    pool = h.groupby(["surface", "band"])["x"].agg(mu_p="mean", sd_p="std").reset_index()
    c = c.merge(pool, on=["surface", "band"], how="left")
    c["mu_s"] = (c["n"] * c["mu"] + SHRINK_MEAN * c["mu_p"]) / (c["n"] + SHRINK_MEAN)
    c["sd_s"] = np.sqrt((c["n"] * c["sd"].fillna(c["sd_p"]) ** 2 + SHRINK_SD * c["sd_p"] ** 2) / (c["n"] + SHRINK_SD))
    h = h.merge(c[COURSE + ["mu_s", "sd_s"]], on=COURSE, how="left")
    h["ts"] = 50 - 10 * (h["x"] - h["mu_s"]) / h["sd_s"]
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", nargs="+", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2027)))
    a = ap.parse_args()
    races = lc.race_table(a.years)[["race_id", "venue", "surface", "distance"]]
    off = race_offsets(a.years)
    hp = horse_peaks([Path(p) for p in a.laps])
    h = hp.merge(races.assign(race_id=races["race_id"].astype(str)), on="race_id", how="inner")
    h = h.merge(off.assign(race_id=off["race_id"].astype(str)), on="race_id", how="left")
    print(f"馬 {len(hp)} 頭・レース {hp['race_id'].nunique()} / 補正できた馬 {h['offset'].notna().sum()}")
    t = course_ts(h)
    print("補正後ピークのコース内の標準偏差 / 補正前:", round(t["x"].std(), 3), round(t["peak"].std(), 3))
    print("着順との相関（偏差値 ts, レース内順位）:", round(t.groupby("race_id").apply(lambda g: g["ts"].rank(ascending=False).corr(g["finish_order"])).mean(), 3))
    print(t["ts"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
