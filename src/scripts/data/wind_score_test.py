"""過去走の風が、今の親モデルの予測の残差に残っているか（スコア検定）。再学習 候補 14 の第 1 段（2026-11）。

    python -m src.scripts.data.wind_score_test [--oof <ens/v6.parquet>] [--out <dir>]

- 残差 = 勝ったか(0/1) − 予測勝率（段 9 の LOYO、検証 2022〜2024 年）。説明変数 x は前走(まで)の風だけ。今回のレースの風は使わない。
- スコア統計 = Σ(y−p)·x を、レース単位で足してから分散を取る（同じレースの出走は独立でない）。z = 合計 / √(レース別合計の二乗和)。
- 候補は評価前に固定: 前走が強い向かい風(≥4m/s)の旗、前走の向かい風成分（連続）、前 3 走の平均、前走の向かい風 × 前走の脚質(前/後)、前走の横風の大きさ。
  5 本なので Bonferroni で p<0.01 を合格線にする。2026 年は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.columns import wind_venue as wv

BASE = Path("/home/jovyan/work/retrain_out/base_2016")
WIND = Path(os.environ.get("WIND_OUT") or "/home/jovyan/work/retrain_out/wind16")


def runs() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "date", "passing_order", "finish_position", "field_size"]
    r = pd.concat([pd.read_parquet(wv.TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in range(2016, 2026)], ignore_index=True)
    r["date"] = pd.to_datetime(r["date"].astype(str).str[:10])
    r["horse_id"] = r["horse_id"].astype(str)
    po = r["passing_order"].astype(str).str.split("-")
    first = pd.to_numeric(po.str[0], errors="coerce")
    r["q_first"] = (first - 1) / (pd.to_numeric(r["field_size"], errors="coerce") - 1)
    w = pd.read_parquet(WIND / "race_wind.parquet", columns=["race_id", "head", "cross"])
    r = r.merge(w, on="race_id", how="left").sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = r.groupby("horse_id")
    r["prev_head"] = g["head"].shift()
    r["prev_cross"] = g["cross"].shift().abs()
    r["prev_q"] = g["q_first"].shift()
    r["prev3_head"] = g["head"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    r["prev_strong"] = (r["prev_head"] >= 4.0).astype(float).where(r["prev_head"].notna())
    r["front"] = (r["prev_q"] < 0.3).astype(float).where(r["prev_q"].notna())
    r["back"] = (r["prev_q"] > 0.6).astype(float).where(r["prev_q"].notna())
    r["prev_head_front"] = r["prev_head"] * r["front"]
    r["prev_head_back"] = r["prev_head"] * r["back"]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(BASE / "tp27_v6e/ens/v6.parquet"))
    ap.add_argument("--out", default=str(WIND))
    a = ap.parse_args()
    o = pd.read_parquet(a.oof)[["race_id", "horse_id", "year", "p_win"]]
    o["horse_id"] = o["horse_id"].astype(str)
    r = runs()
    d = o.merge(r[["race_id", "horse_id", "finish_position", "prev_strong", "prev_head", "prev3_head", "prev_head_front", "prev_head_back",
                   "prev_cross"]], on=["race_id", "horse_id"], how="left")
    d["y"] = (pd.to_numeric(d["finish_position"], errors="coerce") == 1).astype(float)
    d["res"] = d["y"] - d["p_win"]
    d["v"] = d["p_win"] * (1 - d["p_win"])
    rows = []
    for x in ["prev_strong", "prev_head", "prev3_head", "prev_head_front", "prev_head_back", "prev_cross"]:
        s = d[d[x].notna()].copy()
        xc = s[x] - s[x].mean()
        s["sx"] = s["res"] * xc
        per_race = s.groupby("race_id")["sx"].sum()
        z = float(per_race.sum() / np.sqrt((per_race ** 2).sum()))
        # 年ごとの符号（同符号か）
        yz = {}
        for y, sy in s.assign(xc=xc).groupby("year"):
            pr = (sy["res"] * sy["xc"]).groupby(sy["race_id"]).sum()
            yz[int(y)] = round(float(pr.sum() / np.sqrt((pr ** 2).sum())), 2)
        rows.append({"x": x, "n": int(len(s)), "share_nonzero": round(float((s[x] != 0).mean()), 3), "z": round(z, 2), "z_by_year": yz})
        print(rows[-1], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "score_test.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def boost(oof: str | None = None) -> None:
    """残差に小さな木を当てはめる（init_score = 段 9 の予測）。風の列あり／なし（対照）の、年を除いた検証の LL を比べる。交互作用も拾う。"""
    import lightgbm as lgb

    o = pd.read_parquet(oof or str(BASE / "tp27_v6e/ens/v6.parquet"))[["race_id", "horse_id", "year", "p_win"]]
    o["horse_id"] = o["horse_id"].astype(str)
    r = runs()
    w = pd.read_parquet(WIND / "race_wind.parquet", columns=["race_id", "venue_code", "surface", "distance", "tc", "grade"])
    d = o.merge(r[["race_id", "horse_id", "finish_position", "prev_strong", "prev_head", "prev3_head", "prev_q", "prev_cross"]],
                on=["race_id", "horse_id"], how="left").merge(w, on="race_id", how="left")
    d["y"] = (pd.to_numeric(d["finish_position"], errors="coerce") == 1).astype(int)
    for c in ("venue_code", "surface", "tc", "grade"):
        d[c] = d[c].astype("category")
    d["init"] = np.log(d["p_win"].clip(1e-6, 1 - 1e-6) / (1 - d["p_win"].clip(1e-6, 1 - 1e-6)))
    ctx = ["venue_code", "surface", "distance", "tc", "grade", "prev_q"]
    wnd = ["prev_strong", "prev_head", "prev3_head", "prev_cross"]
    prm = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=4, seed=7)

    def ll(y, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    out = {}
    for name, cols in (("control", ctx), ("wind", ctx + wnd)):
        gain = []
        for hold in (2022, 2023, 2024):
            tr, te = d[d["year"] != hold], d[d["year"] == hold]
            m = lgb.train(prm, lgb.Dataset(tr[cols], tr["y"], init_score=tr["init"]), num_boost_round=300)
            pr = 1 / (1 + np.exp(-(te["init"].to_numpy() + m.predict(te[cols], raw_score=True))))
            gain.append(ll(te["y"].to_numpy(), pr) - ll(te["y"].to_numpy(), te["p_win"].to_numpy()))
        out[name] = [round(g, 6) for g in gain]
        print(name, out[name], "mean", round(float(np.mean(gain)), 6), flush=True)
    (WIND / "boost.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    import sys

    boost() if len(sys.argv) > 1 and sys.argv[1] == "boost" else main()
