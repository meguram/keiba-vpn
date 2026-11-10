"""走法軸の簡易版（全年の入力だけで作る）と、親モデルへの履歴としての評価。再学習 候補 15（2026-11）。

    python -m src.scripts.data.run_axes_simple build      # 2016〜2026 年の簡易版 → simple_axes.parquet
    python -m src.scripts.data.run_axes_simple fidelity   # 関門 1: 実測の軸との一致 → fidelity.json
    python -m src.scripts.data.run_axes_simple test       # 関門 2: 残差のスコア検定と木（関門 1 を通った軸だけ）

合格線は docs/operations/retrain_trials/README.md の「走法軸の履歴（候補 15）」に測る前に固定したもの。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.race.run_axes import _band_resid, _band_std, band_of, course_dev

ROOT = Path("/home/jovyan/work/keiba-vpn/data/page_reference/tables")
BASE = Path("/home/jovyan/work/retrain_out/base_2016")
OUT = Path(os.environ.get("AXES_OUT") or "/home/jovyan/work/retrain_out/axes16")
REAL = Path("/home/jovyan/work/keiba-vpn/data/local/running_profile/run_axes.parquet")
AXES = ["s_split", "s_front", "s_cr", "s_makuri", "s_endure"]
REAL_OF = {"s_split": "split", "s_front": "front", "s_cr": "cr", "s_makuri": "makuri", "s_endure": "endure"}
BANDS = ("短", "1600", "1800+")


def _entries(po: str) -> list[int]:
    try:
        return [int(x) for x in str(po).split("-")]
    except ValueError:
        return []


def load(years=range(2016, 2027)) -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "surface", "distance", "field_size", "horse_id", "finish_position", "time_sec",
            "passing_order", "last_3f", "lap_times"]
    r = pd.concat([pd.read_parquet(ROOT / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    r["date"] = pd.to_datetime(r["date"].astype(str).str[:10])
    r["horse_id"] = r["horse_id"].astype(str)
    r["race_id"] = r["race_id"].astype(str)
    r["pos"] = pd.to_numeric(r["finish_position"], errors="coerce")
    return r[r["surface"].isin(["芝", "ダート"]) & r["pos"].notna()].reset_index(drop=True)


def build() -> pd.DataFrame:
    t = load()
    n = len(t)
    dist = t["distance"].to_numpy(float)
    band = band_of(dist)
    fs = pd.to_numeric(t["field_size"], errors="coerce").to_numpy(float)
    l3 = pd.to_numeric(t["last_3f"], errors="coerce").to_numpy(float)
    ts = pd.to_numeric(t["time_sec"], errors="coerce").to_numpy(float)

    # レース単位のラップ（1 レース 1 回だけ読む）
    rl = t.drop_duplicates("race_id")[["race_id", "lap_times", "distance"]]
    first_lap, pass_t = {}, {}
    for rid, lp, d in zip(rl["race_id"], rl["lap_times"], rl["distance"]):
        try:
            laps = json.loads(lp) if isinstance(lp, str) else list(lp)
            laps = np.array([float(x) for x in laps])
        except Exception:
            continue
        if len(laps) < 5 or not np.isfinite(laps).all():
            continue
        len1 = d - 200.0 * (len(laps) - 1)
        if not (50 <= len1 <= 200):
            continue
        marks = np.concatenate([[0.0], len1 + 200.0 * np.arange(len(laps))])
        cum = np.concatenate([[0.0], np.cumsum(laps)])
        first_lap[rid] = (laps[0], len1)
        pass_t[rid] = float(np.interp(600.0 if d <= 1400 else 1000.0, marks, cum))
    f1 = t["race_id"].map(lambda k: first_lap.get(k, (np.nan, np.nan))[0]).to_numpy(float)
    len1 = t["race_id"].map(lambda k: first_lap.get(k, (np.nan, np.nan))[1]).to_numpy(float)
    t["pass_race"] = t["race_id"].map(pass_t)

    # 前後半比: 自分の上がり 3F の 200m 当たり ÷ 中盤の 200m 当たり
    mid_len = dist - 600.0 - len1
    mid_t = ts - l3 - f1
    ok = (mid_len >= 400) & (mid_t > 0) & (l3 > 0)
    ratio = np.where(ok, (l3 / 3.0) / (mid_t / (mid_len / 200.0)), np.nan)
    # 入力の誤り（上がり 3F が極端など）を除く。範囲は最初の相関確認で外れ値（400 など）が出たので置いた（勝敗は見ていない）
    t["split_raw"] = np.where((l3 >= 30) & (l3 <= 50) & (ratio >= 0.6) & (ratio <= 1.6), ratio, np.nan)

    # 通過順（右そろえ: 末尾 = 4 角、その前 = 3 角）
    ent = [_entries(x) for x in t["passing_order"]]
    ne = np.array([len(e) for e in ent])
    den = np.maximum(fs - 1, 1)

    def rel(vals):
        return (np.asarray(vals, float) - 1) / den

    first = np.array([e[0] if e else np.nan for e in ent], float)
    c3 = np.array([e[-2] if len(e) >= 2 else np.nan for e in ent], float)
    c4 = np.array([e[-1] if e else np.nan for e in ent], float)
    second = np.array([e[1] if len(e) >= 2 else (e[0] if e else np.nan) for e in ent], float)
    t["rel_first"], t["rel_c3"], t["rel_second"] = rel(first), rel(c3), rel(second)
    t["cr_raw"] = rel(c3) - rel(c4)
    t["mk_raw"] = np.where(ne >= 3, rel(first) - rel(c3), np.nan)
    t["ne"] = ne
    t["band"] = band
    return t


def finish_axes(t: pd.DataFrame, front_choice: dict[str, str]) -> pd.DataFrame:
    """front_choice = 距離帯 → 使う列（rel_first / rel_c3 / rel_second）。2025 年だけで選んだもの。"""
    band = t["band"].to_numpy()
    t = t.copy()
    t["rel_sel"] = np.nan
    for b, c in front_choice.items():
        m = band == b
        t.loc[m, "rel_sel"] = t.loc[m, c]
    out = t[["race_id", "horse_id", "date", "venue", "surface", "distance", "band"]].copy()
    out["s_split"] = course_dev(t, "split_raw")
    t["front_raw"] = -t["rel_sel"]
    out["s_front"] = 50 + 10 * _band_std(course_dev(t, "front_raw"), band)
    out["s_cr"] = 50 + 10 * _band_std(course_dev(t, "cr_raw"), band)
    out["s_makuri"] = 50 + 10 * _band_std(course_dev(t, "mk_raw"), band)
    # 消耗耐性: 前後半比を、レースの前半ペースと自分の位置で揃えた残差
    rp = course_dev(t.assign(pass_race=t["pass_race"]), "pass_race")
    X = np.column_stack([rp.to_numpy(float), t["rel_sel"].to_numpy(float)])
    res = _band_resid(course_dev(t, "split_raw"), X, band, np.zeros(len(t), bool))
    res = res - res.groupby(t["race_id"].to_numpy()).transform("mean")
    out["s_endure"] = 50 - 10 * _band_std(res, band)
    return out


def pick_front(t: pd.DataFrame, real: pd.DataFrame) -> dict[str, str]:
    """2025 年の実測 front との相関が最大の列を、距離帯ごとに選ぶ（選択に使うのは 2025 年だけ）。"""
    m = t[t["date"].dt.year == 2025].merge(real[["race_id", "horse_id", "front"]], on=["race_id", "horse_id"])
    choice = {}
    for b in BANDS:
        s = m[(m["band"] == b) & m["front"].notna()]
        best = max(("rel_first", "rel_c3", "rel_second"), key=lambda c: np.corrcoef(
            s[c].fillna(s[c].mean()), s["front"])[0, 1] * -1)
        choice[b] = best
    return choice


def fidelity() -> None:
    t = pd.read_parquet(OUT / "base_runs.parquet")
    real = pd.read_parquet(REAL)
    real["race_id"], real["horse_id"] = real["race_id"].astype(str), real["horse_id"].astype(str)
    choice = pick_front(t, real)
    print("front 選択(2025):", choice, flush=True)
    a = finish_axes(t, choice)
    a.to_parquet(OUT / "simple_axes.parquet", index=False)
    a["year"] = a["date"].dt.year
    j = a.merge(real[["race_id", "horse_id"] + list(REAL_OF.values())], on=["race_id", "horse_id"], how="inner")
    j["year"] = j["date"].dt.year
    res = {"front_choice": choice, "axes": {}}
    ref = a[a["year"].isin([2025, 2026])]
    for ax in AXES:
        rc = REAL_OF[ax]
        r = {}
        for y in (2025, 2026):
            s = j[(j["year"] == y) & j[ax].notna() & j[rc].notna()]
            r[y] = {"n": int(len(s)), "pearson": round(float(s[ax].corr(s[rc])), 3),
                    "spearman": round(float(s[ax].corr(s[rc], method="spearman")), 3), "cover": round(float(j.loc[j.year == y, ax].notna().mean()), 3)}
        z = (a[ax] - a[ax].mean()) / a[ax].std()
        ym = z.groupby(a["year"]).mean()
        ysd = a[ax].groupby(a["year"]).std()
        ref_m, ref_sd = float(z[a["year"].isin([2025, 2026])].mean()), float(a.loc[a["year"].isin([2025, 2026]), ax].std())
        miss = a[ax].isna().groupby(a["year"]).mean()
        old = [y for y in range(2016, 2024)]
        g1 = r[2026]["pearson"] >= 0.70
        g2 = abs(r[2025]["pearson"] - r[2026]["pearson"]) <= 0.05
        g3 = bool((ym.loc[old] - ref_m).abs().max() <= 0.10 and (ysd.loc[old] / ref_sd - 1).abs().max() <= 0.10)
        g4 = bool((miss.loc[2016:2025] - miss.loc[2016:2025].mean()).abs().max() < 0.02)
        res["axes"][ax] = {"corr": r, "gate1_r>=0.70": bool(g1), "gate2_year_gap<=0.05": bool(g2), "gate3_no_drift": g3, "gate4_missing_flat": g4,
                           "year_mean_z": {int(k): round(float(v), 3) for k, v in ym.items()},
                           "year_sd_ratio": {int(k): round(float(v / ref_sd), 3) for k, v in ysd.items()},
                           "missing": {int(k): round(float(v), 3) for k, v in miss.items()},
                           "pass": bool(g1 and g2 and g3 and g4)}
        print(ax, json.dumps({k: v for k, v in res["axes"][ax].items() if k not in ("year_mean_z", "year_sd_ratio", "missing")}, ensure_ascii=False), flush=True)
    res["passed"] = [k for k, v in res["axes"].items() if v["pass"]]
    (OUT / "fidelity.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("通過:", res["passed"])


def histories(a: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """as-of の履歴: 前走の値(p1)、前 3 走の平均(p3)。値が無い走は飛ばして直前の有効な値を使う。"""
    a = a.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    out = a[["race_id", "horse_id"]].copy()
    g = a.groupby("horse_id", sort=False)
    for ax in axes:
        v = a[ax]
        ff = v.groupby(a["horse_id"]).ffill()
        out[f"{ax}_p1"] = ff.groupby(a["horse_id"]).shift()
        valid = v.notna()
        m3 = pd.Series(np.nan, index=a.index)
        sub = a.loc[valid, ["horse_id"]].assign(v=v[valid])
        m3[valid] = sub.groupby("horse_id")["v"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out[f"{ax}_p3"] = m3.groupby(a["horse_id"]).ffill().groupby(a["horse_id"]).shift()
    return out


def score_test() -> None:
    from scipy.stats import norm

    fid = json.loads((OUT / "fidelity.json").read_text(encoding="utf-8"))
    axes = fid["passed"]
    if not axes:
        print("関門 1 を通った軸がない → 関門 2 は行わない")
        (OUT / "score_test.json").write_text(json.dumps({"passed_axes": [], "rows": []}), encoding="utf-8")
        return
    a = pd.read_parquet(OUT / "simple_axes.parquet")
    h = histories(a, axes)
    h.to_parquet(OUT / "simple_axes_hist.parquet", index=False)
    cand = [c for c in h.columns if c.endswith(("_p1", "_p3"))]
    K = len(cand)
    thr = float(norm.isf(0.005 / K))
    o = pd.read_parquet(BASE / "tp27_v6e/ens/v6.parquet")[["race_id", "horse_id", "year", "p_win"]]
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    y = load(range(2022, 2025))[["race_id", "horse_id", "pos"]]
    d = o.merge(y, on=["race_id", "horse_id"], how="left").merge(h, on=["race_id", "horse_id"], how="left")
    d["res"] = (d["pos"] == 1).astype(float) - d["p_win"]
    rows = []
    for x in cand:
        s = d[d[x].notna()].copy()
        xc = s[x] - s[x].mean()
        pr = (s["res"] * xc).groupby(s["race_id"]).sum()
        z = float(pr.sum() / np.sqrt((pr ** 2).sum()))
        yz = {}
        for yr, sy in s.assign(xc=xc).groupby("year"):
            q = (sy["res"] * sy["xc"]).groupby(sy["race_id"]).sum()
            yz[int(yr)] = round(float(q.sum() / np.sqrt((q ** 2).sum())), 2)
        sign_ok = len({np.sign(v) for v in yz.values()}) == 1
        rows.append({"x": x, "n": int(len(s)), "z": round(z, 2), "z_by_year": yz, "pass": bool(abs(z) >= thr and sign_ok)})
        print(rows[-1], flush=True)
    res = {"passed_axes": axes, "K": K, "threshold": round(thr, 2), "rows": rows}
    (OUT / "score_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("K =", K, "合格線 |z| >=", round(thr, 2))


def boost() -> None:
    import lightgbm as lgb

    fid = json.loads((OUT / "fidelity.json").read_text(encoding="utf-8"))
    axes = fid["passed"]
    if not axes:
        return
    h = pd.read_parquet(OUT / "simple_axes_hist.parquet")
    o = pd.read_parquet(BASE / "tp27_v6e/ens/v6.parquet")[["race_id", "horse_id", "year", "p_win"]]
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    y = load(range(2022, 2025))[["race_id", "horse_id", "pos"]]
    w = pd.read_parquet("/home/jovyan/work/retrain_out/wind16/race_wind.parquet", columns=["race_id", "venue_code", "surface", "distance", "tc", "grade"])
    w["race_id"] = w["race_id"].astype(str)
    runs = load(range(2016, 2025))[["race_id", "horse_id", "passing_order", "field_size", "date"]]
    runs = runs.sort_values(["horse_id", "date", "race_id"])
    first = pd.to_numeric(runs["passing_order"].astype(str).str.split("-").str[0], errors="coerce")
    runs["q"] = (first - 1) / (pd.to_numeric(runs["field_size"], errors="coerce") - 1)
    runs["prev_q"] = runs.groupby("horse_id")["q"].shift()
    d = (o.merge(y, on=["race_id", "horse_id"], how="left").merge(h, on=["race_id", "horse_id"], how="left")
         .merge(w, on="race_id", how="left").merge(runs[["race_id", "horse_id", "prev_q"]], on=["race_id", "horse_id"], how="left"))
    d["y"] = (d["pos"] == 1).astype(int)
    for c in ("venue_code", "surface", "tc", "grade"):
        d[c] = d[c].astype("category")
    p = d["p_win"].clip(1e-6, 1 - 1e-6)
    d["init"] = np.log(p / (1 - p))
    ctx = ["venue_code", "surface", "distance", "tc", "grade", "prev_q"]
    ax_cols = [c for c in h.columns if c.endswith(("_p1", "_p3"))]
    prm = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=4, seed=7)

    def ll(yy, pp):
        pp = np.clip(pp, 1e-9, 1 - 1e-9)
        return float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).mean())

    out = {}
    for name, cols in (("control", ctx), ("axes", ctx + ax_cols)):
        gain = []
        for hold in (2022, 2023, 2024):
            tr, te = d[d["year"] != hold], d[d["year"] == hold]
            m = lgb.train(prm, lgb.Dataset(tr[cols], tr["y"], init_score=tr["init"]), num_boost_round=300)
            pr = 1 / (1 + np.exp(-(te["init"].to_numpy() + m.predict(te[cols], raw_score=True))))
            gain.append(ll(te["y"].to_numpy(), pr) - ll(te["y"].to_numpy(), te["p_win"].to_numpy()))
        out[name] = [round(g, 6) for g in gain]
        print(name, out[name], "mean", round(float(np.mean(gain)), 6), flush=True)
    out["diff_axes_minus_control"] = [round(a - b, 6) for a, b in zip(out["axes"], out["control"])]
    out["diff_mean"] = round(float(np.mean(out["diff_axes_minus_control"])), 6)
    (OUT / "boost.json").write_text(json.dumps(out), encoding="utf-8")
    print("axes - control:", out["diff_axes_minus_control"], "mean", out["diff_mean"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build().to_parquet(OUT / "base_runs.parquet", index=False)
        print("done")
    elif cmd == "fidelity":
        fidelity()
    elif cmd == "test":
        score_test()
        boost()
    else:
        print(__doc__)
