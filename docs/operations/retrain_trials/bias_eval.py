"""馬場の当日予測の評価（README「馬場の当日予測」の定義どおり）。

    PYTHONPATH=. python docs/operations/retrain_trials/bias_eval.py base            # 基準線・現行
    PYTHONPATH=. python docs/operations/retrain_trials/bias_eval.py grid            # 重みの探索（試行）

レースごとの実現傾向 b_F（初角位置の傾き）・b_D（枠の傾き）を、そのレースより前のレースだけから予測する。
出力: retrain_out/bias/（races.parquet・結果 JSON）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(os.environ.get("BIAS_HORSE", "/home/jovyan/work/retrain_out/tp53_2016/horse.parquet"))
OUT = Path(os.environ.get("BIAS_OUT", "/home/jovyan/work/retrain_out/bias"))
EVAL_YEARS = [2022, 2023, 2024]
DAY_RACES = 12
TARGETS = ["b_F", "b_D"]


def race_table() -> pd.DataFrame:
    f = OUT / "races.parquet"
    if f.exists():
        return pd.read_parquet(f)
    d = pd.read_parquet(H, columns=["race_id", "year", "venue", "surface", "distance", "track_condition", "finish_position", "act_early_norm", "horse_number"])
    d = d[(d["surface"] != "障") & (d["finish_position"] > 0) & d["act_early_norm"].notna()].copy()
    d["race_id"] = d["race_id"].astype(str)
    d["n"] = d.groupby("race_id")["horse_id" if "horse_id" in d else "horse_number"].transform("size")
    d = d[d["n"] >= 6]
    d["rk"] = d.groupby("race_id")["finish_position"].rank(method="average")
    d["r"] = (d["rk"] - 1) / (d["n"] - 1)
    d["g"] = (d.groupby("race_id")["horse_number"].rank(method="average") - 1) / (d["n"] - 1)
    rows = []
    for rid, x in d.groupby("race_id", sort=False):
        A = np.column_stack([np.ones(len(x)), x["act_early_norm"].to_numpy(float), x["g"].to_numpy(float)])
        coef, *_ = np.linalg.lstsq(A, x["r"].to_numpy(float), rcond=None)
        f0 = x.iloc[0]
        rows.append((rid, int(f0["year"]), f0["venue"], f0["surface"], int(f0["distance"]), f0["track_condition"], len(x), coef[1], coef[2]))
    R = pd.DataFrame(rows, columns=["race_id", "year", "venue", "surface", "distance", "going", "n", "b_F", "b_D"])
    R["day_key"] = R["race_id"].str[:10]
    R["meet"] = R["race_id"].str[:8]
    R["day"] = R["race_id"].str[8:10].astype(int)
    R["rnum"] = R["race_id"].str[10:12].astype(int)
    R["course"] = R["venue"] + R["surface"] + R["distance"].astype(str)
    R["vs"] = R["venue"] + R["surface"]
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_parquet(f, index=False)
    return R


def course_prior(R: pd.DataFrame, m: float = 30.0) -> pd.DataFrame:
    """年ごとに、その年より前のレースからコース平均（場×芝ダの平均へ m 本分縮める）。列 mu_b_F, mu_b_D。"""
    out = pd.DataFrame(index=R.index, columns=[f"mu_{t}" for t in TARGETS], dtype=float)
    for Y in sorted(R["year"].unique()):
        past = R[R["year"] < Y]
        te = R["year"] == Y
        if past.empty:
            continue
        for t in TARGETS:
            g0 = past[t].mean()
            vs = past.groupby("vs")[t].agg(["sum", "count"])
            vs_mu = (vs["sum"] + m * g0) / (vs["count"] + m)
            co = past.groupby(["vs", "course"])[t].agg(["sum", "count"]).reset_index()
            co["mu"] = (co["sum"] + m * co["vs"].map(vs_mu)) / (co["count"] + m)
            x = R.loc[te, ["vs", "course"]].merge(co[["vs", "course", "mu"]], on=["vs", "course"], how="left")
            mu = x["mu"].to_numpy()
            mu = np.where(np.isnan(mu), R.loc[te, "vs"].map(vs_mu).fillna(g0).to_numpy(), mu)
            out.loc[te, f"mu_{t}"] = mu
    return out


def window_pairs(R: pd.DataFrame) -> pd.DataFrame:
    """(i, j, gap, prev_day, cond_changed)。j は i より前に結果が出た、同じ場・同じ芝ダのレース（同じ日の前のレース＋同じ開催の前日）。"""
    key = R[["race_id", "day_key", "meet", "day", "rnum", "surface", "going"]].reset_index().rename(columns={"index": "i"})
    same = key.merge(key, on=["day_key", "surface"], suffixes=("", "_j"))
    same = same[same["rnum_j"] < same["rnum"]]
    same = same.assign(gap=same["rnum"] - 1 - same["rnum_j"], prev=False, chg=False)
    k2 = key.assign(day=key["day"] - 1)
    prev = k2.merge(key, on=["meet", "day", "surface"], suffixes=("", "_j"))
    prev = prev[prev["day"] >= 1]
    prev = prev.assign(gap=(DAY_RACES - prev["rnum_j"]) + (prev["rnum"] - 1), prev=True, chg=prev["going"] != prev["going_j"])
    cols = ["i", "i_j", "gap", "prev", "chg"]
    return pd.concat([same[cols], prev[cols]], ignore_index=True).rename(columns={"i_j": "j"})


def predict(R: pd.DataFrame, P: pd.DataFrame, mu: pd.DataFrame, half: float, prev_w: float, chg_w: float, K: float,
            same_going_only: bool = False) -> pd.DataFrame:
    w = 0.5 ** (P["gap"].to_numpy() / half)
    w = w * np.where(P["prev"], prev_w * np.where(P["chg"], chg_w, 1.0), 1.0)
    if same_going_only:
        w = w * np.where(P["chg"], 0.0, 1.0)
    out = pd.DataFrame(index=R.index)
    out["n_prev"] = pd.Series(1.0, index=P.index).groupby(P["i"]).sum().reindex(R.index).fillna(0).to_numpy()
    for t in TARGETS:
        dev = R[t].to_numpy()[P["j"].to_numpy()] - mu[f"mu_{t}"].to_numpy()[P["j"].to_numpy()]
        ok = ~np.isnan(dev)
        sw = np.bincount(P["i"].to_numpy()[ok], weights=w[ok], minlength=len(R))
        swd = np.bincount(P["i"].to_numpy()[ok], weights=(w * dev)[ok], minlength=len(R))
        adj = swd / (sw + K)
        out[f"base_{t}"] = mu[f"mu_{t}"].to_numpy()
        out[f"pred_{t}"] = mu[f"mu_{t}"].to_numpy() + adj
        out[f"sw_{t}"] = sw
    return out


def skill(R: pd.DataFrame, pr: pd.DataFrame, col: str = "pred", boot: int = 0, seed: int = 0) -> dict:
    m = R["year"].isin(EVAL_YEARS) & (pr["n_prev"] > 0) & pr["base_b_F"].notna()
    d = R[m]
    p = pr[m]
    res = {"n_races": int(m.sum())}
    per = {}
    for t in TARGETS:
        e_b = (d[t] - p[f"base_{t}"]) ** 2
        e_p = (d[t] - p[f"{col}_{t}"]) ** 2
        per[t] = (e_b, e_p)
        res[f"skill_{t}"] = round(float(1 - e_p.mean() / e_b.mean()), 5)
    res["skill"] = round((res["skill_b_F"] + res["skill_b_D"]) / 2, 5)
    res["by_year"] = {}
    for Y in EVAL_YEARS:
        yy = d["year"] == Y
        res["by_year"][Y] = round(float(np.mean([1 - per[t][1][yy].mean() / per[t][0][yy].mean() for t in TARGETS])), 5)
    if boot:
        rng = np.random.default_rng(seed)
        days = d["day_key"].to_numpy()
        ud, inv = np.unique(days, return_inverse=True)
        agg = {t: (np.bincount(inv, per[t][0].to_numpy()), np.bincount(inv, per[t][1].to_numpy())) for t in TARGETS}
        vals = []
        for _ in range(boot):
            c = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))
            vals.append(np.mean([1 - (c * agg[t][1]).sum() / (c * agg[t][0]).sum() for t in TARGETS]))
        res["ci95"] = [round(float(np.quantile(vals, 0.025)), 5), round(float(np.quantile(vals, 0.975)), 5)]
    return res


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "base"
    R = race_table()
    mu = course_prior(R)
    P = window_pairs(R)
    res = {}
    if cmd == "base":
        cur = predict(R, P, mu, half=6.0, prev_w=0.7, chg_w=0.5, K=3.0)
        res["current"] = skill(R, cur, boot=1000)
        b = R[TARGETS].describe().to_dict()
        res["target_stats"] = {t: {k: round(v, 4) for k, v in b[t].items()} for t in TARGETS}
    elif cmd == "grid":
        for half in (2.0, 4.0, 6.0, 12.0, 1e9):
            for prev_w in (0.0, 0.4, 0.7, 1.0):
                for K in (1.0, 3.0, 6.0, 12.0, 24.0):
                    pr = predict(R, P, mu, half=half, prev_w=prev_w, chg_w=0.5, K=K)
                    s = skill(R, pr)
                    res[f"h{half}_p{prev_w}_K{K}"] = s
                    print(f"h{half} p{prev_w} K{K}: {s['skill']:.4f} F {s['skill_b_F']:.4f} D {s['skill_b_D']:.4f} {s['by_year']}", flush=True)
    (OUT / f"bias_{cmd}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res if cmd == "base" else {}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
