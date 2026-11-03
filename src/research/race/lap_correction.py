"""走法 TS の補正係数（レース単位のラップから）。

レースの最速ラップ（ラスト 3 区間の最小）の水準が、馬場状態・ペース・その日のバイアスでどれだけ動くかを、
コース（競馬場・芝ダ・距離）内の偏差で回帰して求める。クラスは使わない。

  peak_dev = b_cond[馬場状態] + b_pace × pace_dev + b_day × day

pace_dev: 序盤の通過タイム（1400m 以下は 600m 通過、それより長いと 1000m 通過）のコース内偏差。遅い序盤 = 速い上がり。
day: 同じ日・同じ競馬場・同じ芝ダの他レースの偏差の合計を（他レース数 + DAY_SHRINK）で割ったもの（自分は除く）。
     何を引いた偏差にするかで 3 通り（raw=コース平均だけ引く / pace=ペースも引く / full=馬場状態とペースも引く）。

馬場状態の区分なしで、day だけで足りるかを、2020-24 年で係数を出し 2025-26 年で当てはまりを見て比べる。

python -m src.research.race.lap_correction [--years 2020 ... 2026]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.research.race import running_profile as rp

MIN_COURSE_RACES = 30
DAY_SHRINK = 2.0
SPLIT_DATE = pd.Timestamp("2025-01-01")
CONDS = ["稍重", "重", "不良"]


def pass_time(laps: list[float], dist: int, x: float) -> float:
    """発走から x[m] 地点の通過タイム（区間内は直線補間）。最初の区間は端数（dist - 200*(n-1)）。"""
    n = len(laps)
    l0 = dist - 200 * (n - 1)
    if not (0 < l0 <= 200 + 1e-6) or x >= dist:
        return float("nan")
    b = l0 + 200 * np.arange(n)       # 各区間の終端位置
    t = np.cumsum(laps)
    i = int(np.searchsorted(b, x))
    t0, b0 = (0.0, 0.0) if i == 0 else (t[i - 1], b[i - 1])
    return float(t0 + (x - b0) / (b[i] - b0) * laps[i])


def early_pace(laps: list[float], dist: int) -> float:
    return pass_time(laps, dist, 600 if dist <= 1400 else 1000)


def race_table(years: list[int]) -> pd.DataFrame:
    runs = rp.load_runs(years)
    r = runs.drop_duplicates("race_id")[["race_id", "date", "venue", "surface", "distance", "track_condition", "lap_times"]].copy()
    laps = [rp._laps(s) for s in r["lap_times"]]
    ok = np.array([len(x) >= 6 and all(v > 0 for v in x) for x in laps])
    r, laps = r[ok].copy(), [x for x, k in zip(laps, ok) if k]
    r["peak"] = [min(x[-3:]) for x in laps]
    r["pace"] = [early_pace(x, int(d)) for x, d in zip(laps, r["distance"])]
    return r.drop(columns="lap_times").dropna(subset=["pace"]).reset_index(drop=True)


def loo_day(d: pd.DataFrame, col: str | pd.Series) -> pd.Series:
    """同じ日・競馬場の他レースの値の合計 / (他レース数 + DAY_SHRINK)。他レースが無ければ 0。"""
    v = d[col] if isinstance(col, str) else col
    g = v.groupby([d["date"], d["venue"]])
    n, s = g.transform("size"), g.transform("sum")
    return (s - v) / (n - 1 + DAY_SHRINK)


def ols(d: pd.DataFrame, cols: list[str], fit: np.ndarray | None = None):
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
    y = d["peak_dev"].to_numpy()
    if fit is None:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b
    return None


def r2(d: pd.DataFrame, cols: list[str], b: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
    res = d["peak_dev"].to_numpy() - X @ b
    return float(1 - (res ** 2).sum() / ((d["peak_dev"] - d["peak_dev"].mean()) ** 2).sum())


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    key = ["venue", "surface", "distance"]
    d = d[d.groupby(key)["race_id"].transform("size") >= MIN_COURSE_RACES].copy()
    d["peak_dev"] = d["peak"] - d.groupby(key)["peak"].transform("mean")
    d["pace_dev"] = d["pace"] - d.groupby(key)["pace"].transform("mean")
    for c in CONDS:
        d[c] = (d["track_condition"] == c).astype(float)
    return d


def compare(d: pd.DataFrame) -> dict:
    d = prepare(d)
    train = d["date"] < SPLIT_DATE
    d["day_raw"] = loo_day(d, "peak_dev")
    # pace / full 版の day は、学習期間で当てた係数の残差から作る
    bp = ols(d[train], ["pace_dev"])
    d["res_pace"] = d["peak_dev"] - bp[0] - bp[1] * d["pace_dev"]
    d["day_pace"] = loo_day(d, "res_pace")
    bf = ols(d[train], CONDS + ["pace_dev"])
    d["res_full"] = d["peak_dev"] - bf[0] - d[CONDS + ["pace_dev"]].to_numpy() @ bf[1:]
    d["day_full"] = loo_day(d, "res_full")
    models = {
        "A 馬場+ペース": CONDS + ["pace_dev"],
        "B day(raw)のみ": ["day_raw"],
        "C day(raw)+ペース": ["day_raw", "pace_dev"],
        "D day(ペース引き)+ペース": ["day_pace", "pace_dev"],
        "E 馬場+ペース+day(full)": CONDS + ["pace_dev", "day_full"],
    }
    out = {"n_train": int(train.sum()), "n_test": int((~train).sum()), "models": {}}
    for name, cols in models.items():
        b = ols(d[train], cols)
        out["models"][name] = {"train_r2": round(r2(d[train], cols, b), 4), "test_r2": round(r2(d[~train], cols, b), 4),
                               "coef": dict(zip(["const"] + cols, [round(float(x), 3) for x in b]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2027)))
    a = ap.parse_args()
    t = race_table(a.years)
    print(f"レース {len(t)} 件（{t['date'].min().date()}〜{t['date'].max().date()}）")
    for surf in ("芝", "ダート"):
        res = compare(t[t["surface"] == surf])
        print(f"\n== {surf} 学習 {res['n_train']} / 検証 {res['n_test']}")
        for name, m in res["models"].items():
            print(f"  {name}: 学習R2={m['train_r2']} 検証R2={m['test_r2']}  {m['coef']}")


if __name__ == "__main__":
    main()
