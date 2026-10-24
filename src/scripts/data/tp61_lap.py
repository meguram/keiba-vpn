"""tp-61: 実測ラップの形からペースの『種類』を作る（前半3F−後半3F だけでなく、中盤の緩み・加速の始まり・ラスト2F/1F の形）。

python -m src.scripts.data.tp61_lap feats                 # 区間ラップ → 形の特徴（par との差で正規化）
python -m src.scripts.data.tp61_lap cluster [--k 3 .. 9]  # k-means の数 k の比較（安定性・検証年の LogLoss）と、採用 k のタイプ付け
python -m src.scripts.data.tp61_lap types --k K           # タイプ別の頻度（距離帯・芝ダ）と脚質別の成績
入力: /home/jovyan/work/tp55_out/races_feat.parquet（laps_json）, tp61 の race_cls / horse。
tp-55 の B（前半3F・中盤平均・後半3F の 3 次元 k-means）の拡張で、中盤の最遅の位置と緩み幅、加速の長さ、ラスト2F/1F の変化を足す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp61"
RF = Path("/home/jovyan/work/tp55_out/races_feat.parquet")
REF = (2020, 2021, 2022, 2023)
FEATS = ["f_front", "f_mid", "f_last3", "mid_max", "mid_slow_pos", "accel_len", "peak_back", "last_drop", "l2_step"]
STYLE_CODES = ["front", "stalker", "mid", "closer"]


def lap_shape(laps: np.ndarray, par: np.ndarray) -> dict:
    n = len(laps)
    if n < 5 or len(par) != n or np.isnan(par).any():
        return {}
    dev = laps - par
    mid_lo, mid_hi = (3, n - 3) if n >= 8 else (2, n - 2)
    mid = dev[mid_lo:mid_hi]
    rec = {"n_laps": n, "f_front": float(dev[:3].mean()), "f_last3": float(dev[-3:].mean()),
           "f_mid": float(mid.mean()) if len(mid) else np.nan}
    if len(mid):
        j = int(np.argmax(mid))
        rec["mid_max"] = float(mid[j])
        rec["mid_slow_pos"] = float((mid_lo + j) / (n - 1))
    else:
        rec["mid_max"], rec["mid_slow_pos"] = np.nan, np.nan
    # 加速の長さ: 後半(後ろ半分)のラップのうち、中盤の中央値より 0.2 秒以上速いものの数 / 全ラップ数。ピークの位置: 後半で最も速いラップが最終ラップから何ラップ前か
    mid_ref = float(np.median(laps[mid_lo:mid_hi])) if len(mid) else float(np.median(laps))
    half = n // 2
    rec["accel_len"] = float((laps[half:] <= mid_ref - 0.2).sum() / n)
    rec["peak_back"] = float((n - 1) - (half + int(np.argmin(laps[half:]))))
    rec["last_drop"] = float(laps[-1] - laps[-2])                     # +: 最後が落ちた
    rec["l2_step"] = float(laps[-2] - laps[-3]) if n >= 3 else np.nan   # +: ラスト 2F 目で落ちた
    return rec


def cmd_feats() -> int:
    d = pd.read_parquet(RF)
    d["year"] = d["date"].str[:4].astype(int)
    laps = [json.loads(s) for s in d["laps_json"]]
    ref = d[d["year"].isin(REF)]
    par = lf.fit_par(pd.DataFrame({"venue": ref["venue"], "surface": ref["surface"], "distance": ref["distance"], "direction": ref["direction"],
                                   "laps": [json.loads(s) for s in ref["laps_json"]]}))
    rows = []
    for rid, v, s, di, dr, la in zip(d["race_id"], d["venue"], d["surface"], d["distance"], d["direction"], laps):
        p = par.vec(v, s, di, dr)
        rec = {"race_id": rid}
        if p is not None and la:
            rec.update(lap_shape(np.asarray(la, float), np.asarray(p, float)))
        rows.append(rec)
    f = pd.DataFrame(rows)
    f.to_parquet(D / "lapshape.parquet", index=False)
    print(f.shape, f[FEATS].notna().mean().round(3).to_dict())
    print(f[FEATS].describe().round(3).to_string())
    return 0


def _frame() -> pd.DataFrame:
    f = pd.read_parquet(D / "lapshape.parquet")
    r = pd.read_parquet(D / "race_cls.parquet")[["race_id", "year", "surface", "band", "distance", "pace_cls", "shape_cls", "bal"]]
    return f.merge(r, on="race_id", how="inner").dropna(subset=FEATS)


def _kmeans(X: np.ndarray, k: int, seed: int = 0):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def cmd_cluster(ks: list[int]) -> int:
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    df = _frame()
    tr = df[df["year"].isin(REF)]
    mu, sd = tr[FEATS].mean(), tr[FEATS].std() + 1e-9
    Xtr = ((tr[FEATS] - mu) / sd).to_numpy()
    rng = np.random.default_rng(0)
    out = []
    for k in ks:
        km = _kmeans(Xtr, k)
        idx = rng.choice(len(Xtr), 4000, replace=False)
        sil = float(silhouette_score(Xtr[idx], km.predict(Xtr[idx])))
        # 安定性: 2 つの半分標本で学習したクラスタの ARI（全標本の割り当て同士）
        aris = []
        for s in range(4):
            a, b = rng.choice(len(Xtr), len(Xtr) // 2, replace=False), rng.choice(len(Xtr), len(Xtr) // 2, replace=False)
            la, lb = _kmeans(Xtr[a], k, s).predict(Xtr), _kmeans(Xtr[b], k, s + 10).predict(Xtr)
            aris.append(adjusted_rand_score(la, lb))
        out.append({"k": k, "inertia": float(km.inertia_), "silhouette": sil, "ari_stability": float(np.mean(aris))})
    print(pd.DataFrame(out).round(3).to_string())
    return 0


def type_names_from_centroids(cen: pd.DataFrame) -> list[str]:
    names = []
    for i, c in cen.iterrows():
        bal = c["f_last3"] - c["f_front"]
        if bal <= -0.6 and c["f_front"] >= 0.5:
            nm = "超スロー（前半が遅く後半で上がる）"
        elif bal <= -0.3:
            nm = "スロー"
        elif bal >= 0.6:
            nm = "ハイ（前半が速く後半が落ちる）"
        else:
            nm = "平均"
        names.append(nm)
    return names


def cmd_types(k: int) -> int:
    df = _frame()
    tr = df[df["year"].isin(REF)]
    mu, sd = tr[FEATS].mean(), tr[FEATS].std() + 1e-9
    km = _kmeans(((tr[FEATS] - mu) / sd).to_numpy(), k)
    cen = pd.DataFrame(km.cluster_centers_ * sd.to_numpy() + mu.to_numpy(), columns=FEATS)
    cen["bal"] = cen["f_last3"] - cen["f_front"]
    order = np.argsort(cen["bal"].to_numpy())                   # スロー寄り → ハイ寄り
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    cen = cen.iloc[order].reset_index(drop=True)
    df["lap_type"] = remap[km.predict(((df[FEATS] - mu) / sd).to_numpy())]
    share = df[df["year"].isin(REF)]["lap_type"].value_counts(normalize=True).sort_index()
    cen["share"] = share.to_numpy()
    print(cen.round(3).to_string())
    df.to_parquet(D / f"laptype_k{k}.parquet", index=False)
    json.dump({"k": k, "mu": mu.to_dict(), "sd": sd.to_dict(), "centroids": cen.round(4).to_dict(orient="list"), "feats": FEATS},
              open(D / f"laptype_k{k}.json", "w"), ensure_ascii=False, indent=1)
    print(pd.crosstab(df["lap_type"], df["pace_cls"], normalize="index").round(2))
    print(pd.crosstab([df["surface"], df["band"]], df["lap_type"], normalize="index").round(2).to_string())
    return 0


def cmd_eval(k: int) -> int:
    """タイプ別の脚質の成績（レース単位ブートストラップ）と、馬単位 LogLoss（学習 2020〜2023 → 検証 2024・2025）の比較。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    from src.scripts.data.tp61_analysis import STYLE_CODES as SC, _boot, horse_frame, poly_design

    t = pd.read_parquet(D / f"laptype_k{k}.parquet")[["race_id", "lap_type"] + FEATS]
    h = horse_frame().merge(t, on="race_id", how="inner")
    h = h[h["field_size"] >= 8].dropna(subset=["early_norm", "bal", "c4_gap_per_horse", "c4_lead_group_frac"])
    ref = h.groupby("style")[["is_top3", "is_win"]].mean()
    rows = []
    for ty, g in h.groupby("lap_type"):
        for st in SC:
            gs = g[g["style"] == st]
            per = gs.groupby("race_id").agg(top3=("is_top3", "sum"), win=("is_win", "sum"), n=("is_top3", "size"))
            b3 = _boot(pd.DataFrame({"a": per["top3"], "b": ref.loc[st, "is_top3"] * per["n"]}), ["a", "b"])
            bw = _boot(pd.DataFrame({"a": per["win"], "b": ref.loc[st, "is_win"] * per["n"]}), ["a", "b"])
            rows.append({"type": int(ty), "style": st, "n_races": len(per), "top3_lift": b3["mean"], "t3_lo": b3["lo"], "t3_hi": b3["hi"],
                         "win_lift": bw["mean"], "w_lo": bw["lo"], "w_hi": bw["hi"], "l3_rank_pct": float(gs["l3f_rank_pct"].mean()), "pos_gain": float(gs["pos_gain"].mean())})
    ct = pd.DataFrame(rows)
    ct.to_csv(D / f"laptype_k{k}_cells.csv", index=False)
    pd.set_option("display.width", 200)
    print(ct.round(3).to_string())
    for c in range(k):
        h[f"ty_{c}"] = (h["lap_type"] == c).astype(float)
    for c in ("slow", "high"):
        h[f"pc_{c}"] = (h["pace_cls"] == c).astype(float)
    h["gap_z"] = (h["c4_gap_per_horse"] - h["c4_gap_per_horse"].mean()) / h["c4_gap_per_horse"].std()
    h["lead_z"] = (h["c4_lead_group_frac"] - h["c4_lead_group_frac"].mean()) / h["c4_lead_group_frac"].std()
    h["bal_z"] = (h["bal"] - h["bal"].mean()) / h["bal"].std()
    for f in FEATS:
        h[f"z_{f}"] = (h[f] - h[f].mean()) / (h[f].std() + 1e-9)
    tyc = [f"ty_{c}" for c in range(1, k)]
    zc = [f"z_{f}" for f in FEATS]
    specs = {"M0 位置のみ": ([], []), "M1 ペース3段階": (["pc_slow", "pc_high"], ["pc_slow", "pc_high"]),
             "M4 連続bal+凝縮度": (["bal_z", "gap_z", "lead_z"], ["bal_z", "gap_z", "lead_z"]),
             f"M7 ラップ型{k}": (tyc, tyc), f"M8 ラップ型{k}+凝縮度": (tyc + ["gap_z", "lead_z"], tyc + ["gap_z", "lead_z"]),
             "M9 ラップ連続9+凝縮度": (zc + ["gap_z", "lead_z"], zc + ["gap_z", "lead_z"])}
    tr, va = h[h["year"].isin(REF)], h[h["year"].isin([2024, 2025])]
    res = {}
    for tgt in ("is_win", "is_top3"):
        for name, (ex, it) in specs.items():
            Xtr, Xva = poly_design(tr, ex, it), poly_design(va, ex, it)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=500).fit((Xtr - mu) / sd, tr[tgt])
            res[(tgt, name)] = log_loss(va[tgt], m.predict_proba((Xva - mu) / sd)[:, 1])
    for (tgt, name), v in res.items():
        print(f"{tgt:8s} {name:24s} LL={v:.5f} vs M0 {v - res[(tgt, 'M0 位置のみ')]:+.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feats", "cluster", "types", "eval"])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9])
    a = ap.parse_args()
    if a.cmd == "feats":
        return cmd_feats()
    if a.cmd == "cluster":
        return cmd_cluster(a.k)
    if a.cmd == "eval":
        return cmd_eval(a.k[0])
    return cmd_types(a.k[0])


if __name__ == "__main__":
    sys.exit(main())
