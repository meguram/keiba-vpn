"""過去走の風補正（候補 14）の素材づくり（再学習 2026-11）。

    python -m src.scripts.data.wind_adjust_2016 table      # レースごとの向かい風成分 → OUT/race_wind.parquet
    python -m src.scripts.data.wind_adjust_2016 beta       # 上がり3F・道中・全体への係数を 2016〜2025 年で当てはめ → OUT/beta.json

- 風: 気象庁の時別値（2016〜2019 年は /tmp/jma_exp/w2016_2019.parquet、2020 年〜は jma_weather_hourly_flat.parquet）。GCS は読まない。
- 発走の直前の正時の観測だけを使う（発走後は混ぜない）。向かい風成分 = 風速 × cos(風向 − 場ごとの直線の向き)。静穏は 0。
- 係数 β は 2016〜2025 年（学習に使う年だけ。2026 年は使わない）。補正は「過去走の値」にだけ適用する。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.columns import wind_venue as wv

OUT = Path(os.environ.get("WIND_OUT") or "/home/jovyan/work/retrain_out/base_2016/wind")
W_OLD = Path(os.environ.get("WIND_OLD") or "/tmp/jma_exp/w2016_2019.parquet")
YEARS = range(2016, 2026)
STRONG = 4.0   # 「強い向かい風」の閾値（評価前に固定）


def _wind() -> pd.DataFrame:
    cols = ["venue_code", "observed_at", "wind_dir", "wind_dir_deg", "wind_speed_ms"]
    w = pd.concat([pd.read_parquet(W_OLD, columns=cols), pd.read_parquet(wv.WIND, columns=cols)], ignore_index=True)
    w["venue_code"] = w["venue_code"].astype(str).str.zfill(2)
    w["obs_time"] = pd.to_datetime(w["observed_at"].astype(str).str[:19])
    return w.drop(columns="observed_at").drop_duplicates(["venue_code", "obs_time"])


def table() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "venue_code", "surface", "distance", "track_condition", "grade", "start_time",
            "last_3f", "time_sec", "horse_id", "passing_order", "race_name", "race_class", "finish_position"]
    r = pd.concat([pd.read_parquet(wv.TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    from src.research.columns.compare_series import jump_mask
    r = r[r["surface"].isin(["芝", "ダート"]) & ~jump_mask(r)].copy()
    r["venue_code"] = r["venue_code"].astype(str).str.zfill(2)
    r["date"] = pd.to_datetime(r["date"].astype(str).str[:10])
    r["l3"] = pd.to_numeric(r["last_3f"], errors="coerce")
    r["t"] = pd.to_numeric(r["time_sec"], errors="coerce")
    g = r.groupby("race_id")
    rr = g.agg(date=("date", "first"), venue=("venue", "first"), venue_code=("venue_code", "first"), surface=("surface", "first"),
               distance=("distance", "first"), tc=("track_condition", "first"), grade=("grade", "first"), start_time=("start_time", "first"),
               l3=("l3", lambda s: s[s.between(30, 45)].mean()), t=("t", lambda s: s[s > 40].median())).reset_index()
    rr = rr.merge(wv.home_headings(), on="venue_code", how="left")
    ts = pd.to_datetime(rr["date"].dt.strftime("%Y-%m-%d") + " " + rr["start_time"].astype(str), errors="coerce")
    rr["obs_time"] = ts.dt.floor("h")
    rr = rr.merge(_wind(), on=["venue_code", "obs_time"], how="left")
    rel = np.radians(rr["wind_dir_deg"] - rr["home_heading"])
    rr["head"] = rr["wind_speed_ms"] * np.cos(rel)
    rr["cross"] = rr["wind_speed_ms"] * np.sin(rel)
    calm = rr["wind_dir"] == "静穏"
    rr.loc[calm, ["head", "cross"]] = 0.0
    rr["year"] = rr["date"].dt.year
    rr["key"] = rr["venue"] + rr["surface"] + rr["distance"].astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    rr.to_parquet(OUT / "race_wind.parquet", index=False)
    cov = rr.groupby("year")["head"].apply(lambda s: round(float(s.notna().mean()), 4)).to_dict()
    print("races", len(rr), "wind coverage by year", cov, flush=True)
    return rr


def beta() -> None:
    import statsmodels.formula.api as smf

    rr = pd.read_parquet(OUT / "race_wind.parquet")
    d = rr[rr["head"].notna() & rr["l3"].notna()].copy()
    res = {}
    for name, y in (("l3", "l3"), ("t", "t")):
        dd = d[d[y].notna()].copy()
        m = smf.ols(f"{y} ~ head + cross + C(key) + C(tc) + C(grade) + C(year) + C(venue):C(surface):C(tc)", dd).fit(cov_type="HC1")
        res[name] = {"head": float(m.params["head"]), "se": float(m.bse["head"]), "cross": float(m.params["cross"]),
                     "n": int(m.nobs), "r2": float(m.rsquared)}
        print(name, res[name], flush=True)
    base = smf.ols("l3 ~ C(key) + C(tc) + C(grade) + C(year) + C(venue):C(surface):C(tc)", d).fit()
    res["l3_resid_sd"] = float(np.std(base.resid))
    res["head_sd"] = float(d["head"].std())
    res["strong_share"] = float((d["head"] >= STRONG).mean())
    (OUT / "beta.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    {"table": table, "beta": beta}[sys.argv[1]]()
