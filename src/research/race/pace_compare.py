"""ペース偏差の定義を比べる（モデル D の検証 R²）。

cur  : 最初の 1 区間とラスト 3 区間を除く平均ラップ（現行）
f3   : 発走から 600m の通過タイム
f5   : 発走から 1000m の通過タイム
mix  : 1400m 以下は f3、それより長いと f5
f4   : 発走から 800m の通過タイム
なし : ペース項なし（同日の残差だけ）

python -m src.research.race.pace_compare
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.research.race import lap_correction as lc
except ImportError:
    import lap_correction as lc
from src.research.race import running_profile as rp

KEY = ["venue", "surface", "distance"]


def table(years: list[int]) -> pd.DataFrame:
    runs = rp.load_runs(years)
    r = runs.drop_duplicates("race_id")[["race_id", "date", "venue", "surface", "distance", "lap_times"]].copy()
    laps = [rp._laps(s) for s in r["lap_times"]]
    ok = np.array([len(x) >= 6 and all(v > 0 for v in x) for x in laps])
    r, laps = r[ok].copy(), [x for x, k in zip(laps, ok) if k]
    r["peak"] = [min(x[-3:]) for x in laps]
    r["cur"] = [float(np.mean(x[1:-3])) for x in laps]
    for name, m in (("f3", 600), ("f4", 800), ("f5", 1000)):
        r[name] = [lc.pass_time(x, int(d), m) for x, d in zip(laps, r["distance"])]
    r["mix"] = np.where(r["distance"] <= 1400, r["f3"], r["f5"])
    return r.drop(columns="lap_times").dropna().reset_index(drop=True)


def evaluate(d: pd.DataFrame, pcols: list[str]) -> dict:
    d = d[d.groupby(KEY)["race_id"].transform("size") >= lc.MIN_COURSE_RACES].copy()
    d["peak_dev"] = d["peak"] - d.groupby(KEY)["peak"].transform("mean")
    cols = []
    for c in pcols:
        d[c + "_dev"] = d[c] - d.groupby(KEY)[c].transform("mean")
        cols.append(c + "_dev")
    train = d["date"] < lc.SPLIT_DATE
    bp = lc.ols(d[train], cols)
    d["res"] = d["peak_dev"] - bp[0] - (d[cols].to_numpy() @ bp[1:] if cols else 0)
    d["day"] = lc.loo_day(d, "res")
    b = lc.ols(d[train], ["day"] + cols)
    sprint = d["distance"] <= 1400
    return {"train": lc.r2(d[train], ["day"] + cols, b), "test": lc.r2(d[~train], ["day"] + cols, b),
            "sprint": lc.r2(d[~train & sprint], ["day"] + cols, b), "long": lc.r2(d[~train & ~sprint], ["day"] + cols, b),
            "coef": dict(zip(["const", "day"] + cols, [round(float(x), 3) for x in b]))}


def main() -> None:
    t = table(list(range(2020, 2027)))
    print(f"レース {len(t)} 件")
    variants = {"なし": [], "cur": ["cur"], "f3": ["f3"], "f4": ["f4"], "f5": ["f5"], "mix": ["mix"]}
    for surf in ("芝", "ダート"):
        d = t[t["surface"] == surf]
        print(f"\n== {surf}")
        for name, pc in variants.items():
            m = evaluate(d, pc)
            print(f"  {name:4s} 学習={m['train']:.3f} 検証={m['test']:.3f} (短距離≤1400 {m['sprint']:.3f} / 長め {m['long']:.3f})  {m['coef']}")


if __name__ == "__main__":
    main()
