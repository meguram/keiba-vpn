"""tp-77 追加2: ペース軸の特徴量（学習なしの統計量。すべて過去のレースだけから作る）。

python -m src.scripts.data.ptrain_pace build   # PTRAIN_OUT/pace_rows.parquet（race_id, horse_id キー）

P77_C*  コース別の前半 3F・前半 5F の通過時間の分布（トラックバイアス予想家向け。「何番手」の区切りを使わない連続値）
        コース = 場|面|距離|回り。平均・標準偏差・馬場悪化時の平均・クラス別の平均・5F−3F。直前の開催日までの累積。
P77_R*  馬ごとの得意な RPCI レンジとの合致度（ラップ予想家向け）
        過去走の RPCI を (面×距離帯) の z にし、成績で重み付けした分布（中心・幅）と、今回の予想 RPCI（過去の年だけで学習した
        ridge）の距離・カーネルで重み付けした条件付き成績。
市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.features.horse_history.common import FIT_END, _rpci_from_laps
from src.scripts.data import tp27_eval as t
from src.scripts.data import v6_eval as ev

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
TABLES = Path(os.environ.get("PTRAIN_TABLES", ROOT / "data/page_reference/tables"))
KEY = ["race_id", "horse_id"]
YEARS = list(range(2019, 2026))
BAD_GOING = {"重", "不良"}
BW, KAPPA, KAPPA_K, W_SLOPE = 0.75, 3.0, 2.0, 2.0


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_runs() -> pd.DataFrame:
    cols = ["race_id", "date", "horse_id", "finish_position", "field_size", "distance", "surface", "direction", "venue",
            "track_condition", "grade", "lap_times", "passing_order", "time_sec", "last_3f"]
    parts = []
    for y in YEARS:
        p = TABLES / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=cols))
    r = pd.concat(parts, ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"])].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"])
    return r


def race_table(r: pd.DataFrame) -> pd.DataFrame:
    """1 行 = 1 レース。RPCI・前半 3F/5F の時間・コースキー。ラップ数が距離と合わないレースは除く。"""
    g = r.drop_duplicates("race_id")[["race_id", "date", "distance", "surface", "direction", "venue", "track_condition", "grade", "lap_times"]].copy()
    laps = g["lap_times"].map(lf.parse_laps)
    ok = [(l is not None) and len(l) == lf.n_laps_of(d) and len(l) >= 5 for l, d in zip(laps, g["distance"])]
    g, laps = g[ok].copy(), laps[ok]
    g["lap_times_c3"] = [sum(l[:3]) for l in laps]
    g["lap_times_c5"] = [sum(l[:5]) for l in laps]
    g["lap_times_cl3"] = [sum(l[:-3]) for l in laps]
    g["f3"] = g["lap_times_c3"]
    g["f5"] = [sum(l[:5]) for l in laps]
    g["rpci"] = g["lap_times"].map(_rpci_from_laps)
    g["ckey"] = [lf.course_key(v, s, d, di) for v, s, d, di in zip(g["venue"], g["surface"], g["distance"], g["direction"])]
    g["bad"] = g["track_condition"].isin(BAD_GOING).astype(int)
    g["dist_band"] = pd.cut(g["distance"].astype(float), [0, 1200, 1500, 1900, 2400, 99999], labels=False)
    return g.drop(columns=["lap_times"])


def _asof_stats(g: pd.DataFrame, keys: list[str], val: str, pre: str) -> pd.DataFrame:
    """keys の群ごとに、より前の開催日までの累積 n・平均・標準偏差を返す（同日のレースは含めない）。"""
    d = g.groupby(keys + ["date"], sort=True)[val].agg(["size", "sum"]).reset_index()
    d["s2"] = g.assign(_v2=g[val] ** 2).groupby(keys + ["date"], sort=True)["_v2"].sum().to_numpy()
    for c in ("size", "sum", "s2"):
        d[c] = d.groupby(keys)[c].cumsum() - d[c]
    n = d["size"].replace(0, np.nan)
    d[f"{pre}_mean"] = d["sum"] / n
    d[f"{pre}_sd"] = np.sqrt(np.maximum(d["s2"] / n - d[f"{pre}_mean"] ** 2, 0))
    d[f"{pre}_n"] = d["size"]
    return d[keys + ["date", f"{pre}_mean", f"{pre}_sd", f"{pre}_n"]]


def course_pace(g: pd.DataFrame) -> pd.DataFrame:
    out = g[["race_id", "date", "ckey", "bad", "grade", "dist_band", "surface"]].copy()
    for v, p in (("f3", "c3"), ("f5", "c5")):
        a = _asof_stats(g, ["ckey"], v, f"P77_{p}")
        out = out.merge(a, on=["ckey", "date"], how="left")
        b = _asof_stats(g, ["ckey", "bad"], v, f"P77_{p}_going").drop(columns=[f"P77_{p}_going_sd"])
        out = out.merge(b, on=["ckey", "bad", "date"], how="left")
        c = _asof_stats(g, ["ckey", "grade"], v, f"P77_{p}_cls").drop(columns=[f"P77_{p}_cls_sd"])
        out = out.merge(c, on=["ckey", "grade", "date"], how="left")
    out["P77_c5m3"] = out["P77_c5_mean"] - out["P77_c3_mean"]
    out["P77_c3_going_gap"] = out["P77_c3_going_mean"] - out["P77_c3_mean"]
    out["P77_c5_going_gap"] = out["P77_c5_going_mean"] - out["P77_c5_mean"]
    return out.drop(columns=["ckey", "bad", "grade", "dist_band", "surface"])


def rpci_z(g: pd.DataFrame) -> pd.DataFrame:
    fit = g[(g["date"] <= pd.Timestamp(FIT_END)) & g["rpci"].notna()]
    st = fit.groupby(["surface", "dist_band"])["rpci"].agg(["mean", "std"]).rename(columns={"mean": "_m", "std": "_s"}).reset_index()
    g = g.merge(st, on=["surface", "dist_band"], how="left")
    g["z"] = (g["rpci"] - g["_m"]) / g["_s"]
    return g.drop(columns=["_m", "_s"])


def predict_race_z(g: pd.DataFrame, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """今回のレースの予想 RPCI（z）。その年より前の年のレースだけで学習した ridge（欠損は学習側の中央値）。"""
    from sklearn.linear_model import Ridge

    keep = [c for c in ["frp_course_rpci_mean", "frp_course_cls_rpci_mean", "f2_pressure", "f2_nfront", "f2_nfront40", "field_size", "distance"] if c in X.columns]
    rf = pd.concat([df[["race_id"]].astype(str).reset_index(drop=True), X[keep].reset_index(drop=True).astype(float)], axis=1).drop_duplicates("race_id")
    cp = course_pace(g)
    cp = cp[["race_id"] + [c for c in cp.columns if c.startswith("P77_c") and c.endswith(("_mean", "_sd"))]]
    rf = rf.merge(g[["race_id", "date", "z"]], on="race_id", how="inner").merge(cp, on="race_id", how="left")
    rf["year"] = rf["date"].dt.year
    feats = keep + [c for c in cp.columns if c != "race_id"]
    rf["P77_pred_z"] = np.nan
    for y in sorted(rf["year"].unique()):
        tr = rf[(rf["year"] < y) & rf["z"].notna()]
        if len(tr) < 500:
            continue
        med = tr[feats].median()
        A, mu, sd = tr[feats].fillna(med), tr[feats].fillna(med).mean(), tr[feats].fillna(med).std().replace(0, 1)
        m = Ridge(alpha=30.0).fit((A - mu) / sd, tr["z"])
        te = rf["year"] == y
        rf.loc[te, "P77_pred_z"] = m.predict((rf.loc[te, feats].fillna(med) - mu) / sd)
    return rf[["race_id", "P77_pred_z"]]


def horse_fit(r: pd.DataFrame, g: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """targets: race_id, horse_id, date, P77_pred_z。過去走（date が厳密に前）だけで馬の得意レンジとの合致度を作る。"""
    run = r[["race_id", "horse_id", "date", "finish_position", "field_size"]].merge(g[["race_id", "z"]], on="race_id", how="inner")
    fp = pd.to_numeric(run["finish_position"], errors="coerce")
    n = pd.to_numeric(run["field_size"], errors="coerce")
    run["res"] = ((n - fp) / (n - 1).replace(0, np.nan)) - 0.5
    run = run.dropna(subset=["z", "res"]).sort_values(["horse_id", "date"])
    hist = {h: (x["date"].to_numpy("datetime64[D]"), x["z"].to_numpy(float), x["res"].to_numpy(float)) for h, x in run.groupby("horse_id", sort=False)}
    cols = ["P77_R_n", "P77_R_mu", "P77_R_sd", "P77_R_gap", "P77_R_zd", "P77_R_logm", "P77_R_kfit", "P77_R_kn", "P77_R_krel"]
    out = np.full((len(targets), len(cols)), np.nan, dtype="float32")
    tdate = targets["date"].to_numpy("datetime64[D]")
    tz = targets["P77_pred_z"].to_numpy(float)
    for h, idx in targets.groupby("horse_id", sort=False).indices.items():
        if h not in hist:
            continue
        d, z, res = hist[h]
        M = d[None, :] < tdate[idx][:, None]
        w = np.exp(W_SLOPE * res)[None, :] * M
        sw = w.sum(1)
        mu = (w * z[None, :]).sum(1) / (sw + KAPPA)
        var = (w * (z[None, :] - mu[:, None]) ** 2).sum(1) + KAPPA * 1.0
        sd = np.sqrt(var / (sw + KAPPA))
        p = tz[idx]
        gap = p - mu
        zd = gap / sd
        K = np.exp(-0.5 * ((z[None, :] - p[:, None]) / BW) ** 2) * M
        kn = K.sum(1)
        kf = (K * res[None, :]).sum(1) / (kn + KAPPA_K)
        hm = (M * res[None, :]).sum(1) / (M.sum(1) + KAPPA_K)
        out[idx] = np.column_stack([M.sum(1), mu, sd, gap, zd, -0.5 * zd**2, kf, kn, kf - hm]).astype("float32")
    res_df = pd.DataFrame(out, columns=cols)
    res_df.loc[np.isnan(tz), [c for c in cols if c not in ("P77_R_n", "P77_R_mu", "P77_R_sd")]] = np.nan
    return res_df


# ---------------------------------------------------------------- E: 馬ごとの前半 3F/5F 通過の推定
def _pos_feats(r: pd.DataFrame) -> pd.DataFrame:
    def pos(v):
        try:
            return [int(x) for x in str(v).split("-")]
        except ValueError:
            return []
    P = r["passing_order"].map(pos)
    fs = pd.to_numeric(r["field_size"], errors="coerce")
    out = pd.DataFrame({"race_id": r["race_id"].to_numpy(), "horse_id": r["horse_id"].to_numpy(), "fs": fs.to_numpy(),
                        "p_first": [p[0] if p else np.nan for p in P], "p_last": [p[-1] if p else np.nan for p in P], "ncor": P.map(len).to_numpy()})
    out["q_first"] = (out["p_first"] - 1) / (out["fs"] - 1)
    out["q_last"] = (out["p_last"] - 1) / (out["fs"] - 1)
    return out


def run_estimates(r: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """走ごとの『前半 3F・5F 地点での、レース時計に対する遅れ（秒）』の推定。ゴール前 3F 地点の遅れは実測（タイム−上り3F−ラップ累計）。
    早い地点は、実測できる地点（距離−600m）の遅れを、通過順位・頭数・地点の距離から当てはめた回帰（年ごとに前の年のラベルだけで学習）で推定する。
    1200m の 3F 地点、1600m の 5F 地点は実測そのものを使う。"""
    from sklearn.ensemble import HistGradientBoostingRegressor as H

    lp = g[["race_id", "distance", "lap_times_c3", "lap_times_c5", "lap_times_cl3"]]
    x = _pos_feats(r).merge(r[["race_id", "horse_id", "time_sec", "last_3f"]].assign(race_id=lambda d: d["race_id"].astype(str), horse_id=lambda d: d["horse_id"].astype(str)),
                            on=["race_id", "horse_id"]).merge(lp, on="race_id")
    x["time_sec"] = pd.to_numeric(x["time_sec"], errors="coerce")
    x["last_3f"] = pd.to_numeric(x["last_3f"], errors="coerce")
    x = x[(x["time_sec"] > 0) & x["last_3f"].between(30, 50)].copy()
    x["lagL3"] = (x["time_sec"] - x["last_3f"]) - x["lap_times_cl3"]
    x["xpt"] = x["distance"].astype(float) - 600.0
    x = x.merge(g[["race_id", "date"]], on="race_id")
    x["year"] = x["date"].dt.year
    F = ["xpt", "fs", "q_first", "p_first", "q_last", "p_last", "ncor"]
    x["est_lag3"], x["est_lag5"] = np.nan, np.nan
    for y in sorted(x["year"].unique()):
        tr = x[x["year"] == 2021] if y <= 2020 else x[x["year"] < y]
        tr = tr.dropna(subset=["lagL3"])
        if len(tr) < 2000:
            continue
        m = H(max_iter=200, learning_rate=0.08, random_state=0).fit(tr[F], tr["lagL3"])
        te = x["year"] == y
        for pt, col in ((600.0, "est_lag3"), (1000.0, "est_lag5")):
            z = x.loc[te, F].copy()
            z["xpt"] = pt
            x.loc[te, col] = m.predict(z)
    x.loc[x["distance"] == 1200, "est_lag3"] = x["lagL3"]       # 実測の地点
    x.loc[x["distance"] == 1600, "est_lag5"] = x["lagL3"]
    return x[["race_id", "horse_id", "date", "est_lag3", "est_lag5"]]


def horse_pass(est: pd.DataFrame, targets: pd.DataFrame, half_life: float = 3.0, kappa: float = 1.0) -> pd.DataFrame:
    """targets: race_id, horse_id, date, P77_c3_mean/sd, P77_c5_mean/sd。過去走（date が厳密に前）の推定遅れの直近重み付き平均。"""
    e = est.dropna(subset=["est_lag3", "est_lag5"]).sort_values(["horse_id", "date"])
    pre = e[e["date"] <= pd.Timestamp(FIT_END)]       # 事前平均は固定の期間だけから（未来の走を使わない）
    m3, m5 = (float(pre["est_lag3"].mean()), float(pre["est_lag5"].mean())) if len(pre) else (0.0, 0.0)
    hist = {h: (x["date"].to_numpy("datetime64[D]"), x["est_lag3"].to_numpy(float), x["est_lag5"].to_numpy(float)) for h, x in e.groupby("horse_id", sort=False)}
    cols = ["P77_E_n", "P77_E_lag3_ewm", "P77_E_lag5_ewm", "P77_E_lag3_last", "P77_E_lag5_last", "P77_E_d53_ewm"]
    out = np.full((len(targets), len(cols)), np.nan, dtype="float32")
    tdate = targets["date"].to_numpy("datetime64[D]")
    for h, idx in targets.groupby("horse_id", sort=False).indices.items():
        if h not in hist:
            continue
        d, a3, a5 = hist[h]
        M = d[None, :] < tdate[idx][:, None]
        c = np.cumsum(M, axis=1)
        n = M.sum(1)
        w = M * 0.5 ** ((n[:, None] - c) / half_life)
        sw = w.sum(1) + kappa
        l3 = ((w * a3[None, :]).sum(1) + kappa * m3) / sw
        l5 = ((w * a5[None, :]).sum(1) + kappa * m5) / sw
        last = M & (c == n[:, None])
        out[idx] = np.column_stack([n, l3, l5, (last * a3[None, :]).sum(1), (last * a5[None, :]).sum(1), l5 - l3]).astype("float32")
    o = pd.DataFrame(out, columns=cols)
    o.loc[o["P77_E_n"] == 0, cols[1:]] = np.nan
    o["P77_E_T3_exp"] = targets["P77_c3_mean"].to_numpy() + o["P77_E_lag3_ewm"]
    o["P77_E_T5_exp"] = targets["P77_c5_mean"].to_numpy() + o["P77_E_lag5_ewm"]
    o["P77_E_z3"] = o["P77_E_lag3_ewm"] / targets["P77_c3_sd"].to_numpy()
    o["P77_E_z5"] = o["P77_E_lag5_ewm"] / targets["P77_c5_sd"].to_numpy()
    rid = targets["race_id"].to_numpy()
    for k in ("3", "5"):
        v = o[f"P77_E_lag{k}_ewm"]
        gr = v.groupby(rid)
        o[f"P77_E_T{k}_rel"] = v - gr.transform("mean")
        o[f"P77_E_T{k}_min"] = v - gr.transform("min")
    return o


def cmd_build(a) -> None:
    r = load_runs()
    g = rpci_z(race_table(r))
    _log(f"レース {len(g)}（RPCI あり {int(g['z'].notna().sum())}）")
    df, base, meta, folds, lp = t.setup()
    X = ev.build_frame(ev.config_of("z:RB@e"), df)
    rows = df[KEY].astype(str).reset_index(drop=True)
    rows["date"] = pd.to_datetime(df["date"]).to_numpy()
    cp = course_pace(g)
    pz = predict_race_z(g, X, df)
    rows = rows.merge(pz, on="race_id", how="left")
    hf = horse_fit(r, g, rows)
    est = run_estimates(r, g)
    rows = rows.merge(cp[["race_id", "P77_c3_mean", "P77_c3_sd", "P77_c5_mean", "P77_c5_sd"]], on="race_id", how="left")
    he = horse_pass(est, rows)
    cpc = [c for c in cp.columns if c.startswith("P77_")]
    out = pd.concat([rows[KEY], rows[["race_id"]].merge(cp[["race_id"] + cpc], on="race_id", how="left")[cpc], rows[["P77_pred_z"]], hf, he], axis=1)
    for c in out.columns[2:]:
        out[c] = out[c].astype("float32")
    out.to_parquet(OUT / "pace_rows.parquet", index=False)
    cov = {c: round(float(out[c].notna().mean()), 3) for c in out.columns[2:]}
    (OUT / "pace_rows.meta.json").write_text(json.dumps({"rows": len(out), "coverage": cov}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"pace_rows: {out.shape} 被覆 {cov}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
