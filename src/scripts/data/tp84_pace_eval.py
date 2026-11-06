"""tp-84: レース全体のペース（前半の時計・RPCI）を、メンバー構成・馬場・同日の傾向でどこまで事前に当てられるか。tp-55 の学習表に特徴量を足して比べる。ホストで実行。

python -m src.scripts.data.tp84_pace_eval [--first-year 2022] [--last-year 2025] [--out data/local/modeling/tp84]
目的変数（どれも『コース別の平年値（par）からのずれ』を当てる）:
  F3  前半 3F / F5 前半 5F / HYB 1400m 以下は F3・それより長いレースは F5 / RPCI 50×(上がり3Fを除く1F平均÷上がり3Fの1F平均)
指標: par 比の説明率 = 1 − SSE(モデル) / SSE(par)。学習は評価年より前の全年（拡張窓）。2026 年は使わない。市場オッズ・人気は使わない。
特徴量の束: base（tp-55 の表）, cush（その日のクッション値・含水率・前開催日からの変化）, sd2（同日・前の開催日の前半5F/RPCI のずれ）,
mem（出走馬の過去走が走ったレースのペースの癖・先行勢の分布の追加）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf
from src.scripts.data import tp55_build as tb

ROOT = Path(__file__).resolve().parents[3]
FEAT = Path("/home/jovyan/work/tp55_out/races_feat.parquet")
CUSHION = ROOT / "data/jra_baba"
PAR_END = pd.Timestamp("2021-12-31")        # par・ずれの基準を作る期間（ここまでの走だけ）
TARGETS = ["F3", "F5", "HYB", "RPCI"]
M_SHRINK = 8.0


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def race_table(d: pd.DataFrame) -> pd.DataFrame:
    L = d["laps_json"].map(lambda s: np.asarray(json.loads(s), float) if s else None)
    ok = L.map(lambda a: a is not None and len(a) >= 5 and a.min() > 8 and a.max() < 20)
    d = d[ok].copy()
    L = L[ok]
    d["F3"] = L.map(lambda a: a[:3].sum())
    d["F5"] = L.map(lambda a: a[:5].sum())
    d["L3"] = L.map(lambda a: a[-3:].sum())
    d["RPCI"] = L.map(lambda a: 50.0 * a[:-3].mean() / a[-3:].mean())
    d["HYB"] = np.where(d["distance"] <= 1400, d["F3"], d["F5"])
    return d


def par_scalar(fit: pd.DataFrame, tgt: pd.DataFrame, col: str) -> np.ndarray:
    """コース別の平均（件数が少ないコースは面×距離の平均へ寄せる）。fit の行だけで作る。"""
    g = fit.groupby(["surface", "distance"], observed=True)[col].mean()
    gm = pd.MultiIndex.from_arrays([tgt["surface"], tgt["distance"]]).map(g).to_numpy(float)
    gm = np.where(np.isnan(gm), fit[col].mean(), gm)
    c = fit.groupby("course_key", observed=True)[col].agg(["sum", "count"])
    s = tgt["course_key"].map(c["sum"]).astype(float).fillna(0).to_numpy()
    n = tgt["course_key"].map(c["count"]).astype(float).fillna(0).to_numpy()
    return (s + M_SHRINK * gm) / (n + M_SHRINK)


# ───────────────────────── 特徴量の束
def cushion_features(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in range(2020, 2026):
        p = CUSHION / f"cushion_{y}.json"
        if p.exists():
            rows += json.loads(p.read_text(encoding="utf-8"))
    c = pd.DataFrame(rows)
    c["date"] = pd.to_datetime(c["date"])
    c["venue"] = c["venue_name"]
    c = c.drop_duplicates(["date", "venue"]).sort_values(["venue", "date"])
    for k in ("cushion_value", "turf_moisture_goal", "turf_moisture_4corner", "dirt_moisture_goal", "dirt_moisture_4corner"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c["turf_m"] = c[["turf_moisture_goal", "turf_moisture_4corner"]].mean(1)
    c["dirt_m"] = c[["dirt_moisture_goal", "dirt_moisture_4corner"]].mean(1)
    g = c.groupby("venue")
    for k in ("cushion_value", "turf_m", "dirt_m"):
        c[k + "_prev"] = g[k].shift(1)      # 前の開催日（同じ会場）の値
    x = d[["race_id", "date", "venue", "surface"]].copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.merge(c[["date", "venue", "cushion_value", "turf_m", "dirt_m", "cushion_value_prev", "turf_m_prev", "dirt_m_prev"]], on=["date", "venue"], how="left")
    turf = (x["surface"] == "芝").to_numpy()
    out = pd.DataFrame({"race_id": x["race_id"]})
    out["cu_cushion"] = x["cushion_value"].where(turf)
    out["cu_cushion_d"] = (x["cushion_value"] - x["cushion_value_prev"]).where(turf)
    out["cu_moist"] = np.where(turf, x["turf_m"], x["dirt_m"])
    out["cu_moist_d"] = np.where(turf, x["turf_m"] - x["turf_m_prev"], x["dirt_m"] - x["dirt_m_prev"])
    return out


def sameday2_features(d: pd.DataFrame) -> pd.DataFrame:
    """同日・同会場で前のラウンドのレースと、同じ会場・面の前の開催日のレースの、前半5F／RPCI のずれ（par 基準は ≤2021 年）。"""
    x = d[["race_id", "date", "venue", "surface", "round_no", "distance", "course_key", "F5", "RPCI", "F3", "L3"]].copy()
    x["date"] = pd.to_datetime(x["date"])
    fit = x[x["date"] <= PAR_END]
    for c in ("F5", "RPCI", "F3", "L3"):
        x["d_" + c] = x[c].to_numpy() - par_scalar(fit, x, c)
    x["d_bal"] = x["d_L3"] - x["d_F3"]
    x = x.sort_values(["date", "venue", "round_no"], kind="stable").reset_index(drop=True)
    res = {}
    day_mean: dict = {}            # (venue, surface) → その日までの [(date, mean d_F5, mean d_RPCI, mean d_bal)]
    for (dt, v), g in x.groupby(["date", "venue"], sort=False):
        for sf in g["surface"].unique():
            hist = day_mean.get((v, sf), [])
            prev = hist[-1] if hist else (None, np.nan, np.nan, np.nan)
            prev2 = hist[-2] if len(hist) > 1 else (None, np.nan, np.nan, np.nan)
            sub = g[g["surface"] == sf]
            for i, r in enumerate(g.itertuples()):
                if r.surface != sf:
                    continue
                before = g.iloc[:i]
                before = before[before["surface"] == sf].tail(4)
                res[r.race_id] = (len(before), before["d_F5"].mean(), before["d_RPCI"].mean(), before["d_bal"].mean(),
                                  before["d_F5"].iloc[-1] if len(before) else np.nan, before["d_RPCI"].iloc[-1] if len(before) else np.nan,
                                  prev[1], prev[2], prev[3], prev2[1], prev2[2])
            hist.append((dt, sub["d_F5"].mean(), sub["d_RPCI"].mean(), sub["d_bal"].mean()))
            day_mean[(v, sf)] = hist
    cols = ["s2_n", "s2_f5", "s2_rpci", "s2_bal", "s2_f5_last", "s2_rpci_last", "s2_pd_f5", "s2_pd_rpci", "s2_pd_bal", "s2_pd2_f5", "s2_pd2_rpci"]
    return pd.DataFrame.from_dict(res, orient="index", columns=cols).rename_axis("race_id").reset_index()


def member_features(d: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """出走馬の過去走が走ったレースのペースの癖（前の3走のレースの RPCI／前半5F のずれ）と、先行勢の分布の追加列。"""
    res, _, _ = tb.load_inputs(years, None)
    res = res[res["surface"].isin(["芝", "ダート"])].copy()
    res["date"] = pd.to_datetime(res["date"])
    res["early"] = [lf.first_corner_norm(p, f) for p, f in zip(res["passing_order"], res["field_size"])]
    x = d[["race_id", "date", "course_key", "surface", "distance", "F5", "RPCI"]].copy()
    x["date"] = pd.to_datetime(x["date"])
    fit = x[x["date"] <= PAR_END]
    x["d_F5"] = x["F5"].to_numpy() - par_scalar(fit, x, "F5")
    x["d_RPCI"] = x["RPCI"].to_numpy() - par_scalar(fit, x, "RPCI")
    r = res[["race_id", "horse_id", "date", "early"]].dropna(subset=["horse_id"]).merge(x[["race_id", "d_F5", "d_RPCI"]], on="race_id", how="left")
    r = r.sort_values(["horse_id", "date", "race_id"], kind="stable")
    g = r.groupby("horse_id", sort=False)
    for c in ("d_F5", "d_RPCI", "early"):
        r["h_" + c] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    r["h_n"] = g.cumcount()
    ent = r[r["race_id"].isin(d["race_id"])]
    rows = []
    for rid, e in ent.groupby("race_id", sort=False):
        ee = e["h_early"].to_numpy(float)
        k = np.isfinite(ee)
        o = np.argsort(np.where(k, ee, np.inf))[:4]
        o = o[k[o]]
        rows.append({
            "race_id": rid,
            "m_rpci_mean": e["h_d_RPCI"].mean(), "m_f5_mean": e["h_d_F5"].mean(),
            "m_rpci_front": e["h_d_RPCI"].to_numpy(float)[o].mean() if len(o) else np.nan,
            "m_f5_front": e["h_d_F5"].to_numpy(float)[o].mean() if len(o) else np.nan,
            "m_rpci_min": e["h_d_RPCI"].min(), "m_f5_min": e["h_d_F5"].min(),
            "m_n_e25": int((ee[k] <= 0.25).sum()), "m_n_e50": int((ee[k] <= 0.5).sum()),
            "m_share_e25": float((ee[k] <= 0.25).mean()) if k.any() else np.nan,
            "m_sum_top6": float((1 - np.sort(ee[k])[:6]).sum()) if k.any() else np.nan,
            "m_e_gap14": float(np.sort(ee[k])[3] - np.sort(ee[k])[0]) if k.sum() >= 4 else np.nan,
            "m_new_share": float((e["h_n"] == 0).mean()),
        })
    return pd.DataFrame(rows)


# ───────────────────────── 評価
def fit_predict(tr, te, feats, t, seed=0):
    import lightgbm as lgb

    p_tr, p_te = par_scalar(tr, tr, t), par_scalar(tr, te, t)
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=5.0, n_jobs=3, verbose=-1, random_state=seed)
    m.fit(tr[feats], tr[t] - p_tr)
    return p_te, p_te + m.predict(te[feats]), m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-year", type=int, default=2022)
    ap.add_argument("--last-year", type=int, default=2025)
    ap.add_argument("--out", default=str(ROOT / "data/local/modeling/tp84"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(FEAT)
    d["year"] = d["date"].str[:4].astype(int)
    d = d[d["surface"].isin(["芝", "ダート"]) & (d["year"] <= a.last_year)].copy()
    d = race_table(d).reset_index(drop=True)
    _log(f"races {len(d):,}")
    ex = {"cush": cushion_features(d), "sd2": sameday2_features(d), "mem": member_features(d, list(range(2019, a.last_year + 1)))}
    for k, v in ex.items():
        d = d.merge(v, on="race_id", how="left")
        _log(f"{k}: {len(v.columns) - 1} cols, 被覆 {float(v.drop(columns='race_id').notna().mean().mean()):.2f}")
    cat = ["venue", "surface", "direction", "grade", "race_class", "weather", "track_condition", "course_key"]
    for c in cat:
        d[c] = d[c].astype("category")
    drop = {"race_id", "date", "laps_json", "year", "F3", "F5", "L3", "RPCI", "HYB"}
    base = [c for c in pd.read_parquet(FEAT).columns if c not in drop and c in d.columns]
    bundles = {"base": base}
    for k, v in ex.items():
        bundles["+" + k] = base + [c for c in v.columns if c != "race_id"]
    allf = base + [c for k in ex for c in ex[k].columns if c != "race_id"]
    bundles["+all"] = list(dict.fromkeys(allf))
    comp = set(sum((lf.FEATURE_GROUPS[g] for g in ("style", "gate", "ten", "jockey", "field")), [])) | set(ex["mem"].columns) - {"race_id"}
    cond = set(lf.FEATURE_GROUPS["cond"][:2] + lf.FEATURE_GROUPS["sameday"] + lf.FEATURE_GROUPS["recent"]) | set(ex["cush"].columns) | set(ex["sd2"].columns) - {"race_id"}
    cond -= {"grade_c", "class_c"}
    cond |= {"track_condition", "weather"}
    full = bundles["+all"]
    bundles["-comp"] = [c for c in full if c not in comp]
    bundles["-cond"] = [c for c in full if c not in cond]
    bundles["-comp-cond"] = [c for c in full if c not in comp and c not in cond]
    _log(f"comp {len(comp)} cols / cond {len(cond)} cols / full {len(full)}")
    rows, imp = [], {}
    for y in range(a.first_year, a.last_year + 1):
        tr, te = d[d.year < y], d[d.year == y]
        for bn, feats in bundles.items():
            for t in TARGETS:
                p_te, pred, m = fit_predict(tr, te, feats, t)
                yv = te[t].to_numpy()
                if bn == "+all" and y == a.last_year:
                    imp[t] = dict(sorted(zip(feats, m.booster_.feature_importance("gain").round(0).tolist()), key=lambda z: -z[1])[:25])
                for grp, mask in (("全体", np.ones(len(te), bool)), ("<=1400", (te["distance"] <= 1400).to_numpy()), (">1400", (te["distance"] > 1400).to_numpy())):
                    sp, sm = ((yv - p_te)[mask] ** 2).sum(), ((yv - pred)[mask] ** 2).sum()
                    rows.append({"year": y, "bundle": bn, "target": t, "grp": grp, "n": int(mask.sum()), "rmse_par": float(np.sqrt(sp / mask.sum())),
                                 "rmse_model": float(np.sqrt(sm / mask.sum())), "gain": float(1 - sm / sp)})
            _log(f"{y} {bn} done")
    r = pd.DataFrame(rows)
    r.to_csv(out / "bundles.csv", index=False)
    (out / "importance.json").write_text(json.dumps(imp, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220)
    for grp in ("全体", "<=1400", ">1400"):
        s = r[r.grp == grp].pivot_table(index="bundle", columns="target", values="gain", aggfunc="mean").round(3)
        print(f"\n[{grp}] par比の説明率（{a.first_year}〜{a.last_year} 平均）\n", s.loc[list(bundles)][TARGETS].to_string())
    s = r[(r.grp == "全体")].pivot_table(index="bundle", columns=["target", "year"], values="gain").round(3)
    print("\n年別（全体）\n", s.loc[list(bundles)].to_string())


if __name__ == "__main__":
    main()
