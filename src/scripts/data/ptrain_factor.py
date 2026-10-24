"""tp-77 追加6（試行）: 種牡馬の『因子』分析 — 距離軸の試行。系統（○○系）の平均と、その系統に属する種牡馬の差分（δ）を求め、
δ が大きく負（距離が短くなる）／正の種牡馬に共通する祖先を調べる。学習ではなく調査・分析。結果は as-of ではなく全期間（2020–2025）の記述。

python -m src.scripts.data.ptrain_factor distance --surface 芝
軸の測り方: 産駒（馬ごと）の勝利距離の平均（勝った馬だけ）。種牡馬ごとの平均を、同じ系統の平均へ縮約（経験ベイズ）し、δ=縮約後の種牡馬の値−系統の値。
共通祖先: 種牡馬自身の血統表（産駒の 10 世代 JSON の F 側）から、5 世代以内の祖先の出現を、δ の下位／上位の種牡馬で Fisher 検定（BH 補正）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
MINING = Path(os.environ.get("PTRAIN_MINING", "/home/jovyan/work/tp72_out/mining.parquet"))
LINEAGE = Path(os.environ.get("PTRAIN_LINEAGE", ROOT / "data/page_reference/pedigree_race_index/stallion_lineage.parquet"))
PED = Path(os.environ.get("PTRAIN_PED10", ROOT / "data/local/horse_pedigree_10gen"))
MIN_HORSES = int(os.environ.get("PTRAIN_FACTOR_MIN", 10))


def eb_shrink(df: pd.DataFrame, group: str = "line") -> pd.DataFrame:
    """df: sire, line, n, ybar, var。系統の平均へ経験ベイズで縮約。tau2=系統内の種牡馬間の分散（個体内分散の寄与を除く）。"""
    d = df.copy()
    sig2 = float((d["var"] * (d["n"] - 1)).sum() / max((d["n"] - 1).sum(), 1))
    lm = (d.assign(_w=d["ybar"] * d["n"]).groupby(group)["_w"].sum() / d.groupby(group)["n"].sum()).rename("theta_line")
    d = d.merge(lm, on=group)
    between = ((d["ybar"] - d["theta_line"]) ** 2 - sig2 / d["n"]).clip(lower=0)
    tau2 = max(float(between.mean()), 1e-6)
    k = sig2 / tau2
    d["theta"] = (d["n"] * d["ybar"] + k * d["theta_line"]) / (d["n"] + k)
    d["delta"] = d["theta"] - d["theta_line"]
    d["se"] = np.sqrt(1.0 / (d["n"] / sig2 + 1.0 / tau2))
    d["z"] = d["delta"] / d["se"]
    d.attrs.update({"sigma2": sig2, "tau2": tau2, "k": k})
    return d


def sire_pedigree(sire: str, horses: list[str], maxg: int = 5) -> dict[str, int]:
    """産駒の 10 世代 JSON の F 側から、種牡馬自身の血統表（祖先ID → 世代）。"""
    for h in horses[:3]:
        p = PED / h[:4] / f"{h}.json"
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out: dict[str, int] = {}
        for a in j.get("ancestors") or []:
            pf = a.get("path_fm") or ""
            if pf.startswith("F") and 2 <= len(pf) <= maxg + 1 and a.get("horse_id"):
                out.setdefault(a["horse_id"], len(pf) - 1)
        if out:
            return out
    return {}


def fisher_enrich(P: dict[str, dict[str, int]], low: list[str], high: list[str], min_support: int = 6) -> pd.DataFrame:
    from scipy.stats import fisher_exact

    allc = pd.Series([a for s in P.values() for a in s]).value_counts()
    rows = []
    for a, n in allc.items():
        if n < min_support:
            continue
        in_low = sum(a in P[s] for s in low); in_high = sum(a in P[s] for s in high)
        t = [[in_low, len(low) - in_low], [in_high, len(high) - in_high]]
        p = fisher_exact(t)[1]
        rows.append({"anc": a, "support": int(n), "in_low": in_low, "n_low": len(low), "in_high": in_high, "n_high": len(high), "p": p})
    d = pd.DataFrame(rows)
    if len(d):
        o = np.argsort(d["p"].to_numpy()); m = len(d); q = np.empty(m); prev = 1.0
        for rank, i in zip(range(m, 0, -1), o[::-1]):
            prev = min(prev, d["p"].iloc[i] * m / rank); q[i] = prev
        d["q"] = q
    return d


def ridge_ancestors(P: dict[str, dict[str, int]], d: pd.DataFrame, min_support: int = 5, n_boot: int = 200, seed: int = 0) -> dict:
    """種牡馬のδを、その血統表の祖先の有無（世代が近いほど重い 2^-世代）でリッジ回帰（重み=1/se²）。種牡馬単位の5分割CVで当てはまりを見る。
    係数の符号がブートストラップで安定したものだけを『候補』にする。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    sires = d["sire"].tolist()
    cnt = pd.Series([a for s in sires for a in P[s]]).value_counts()
    anc = cnt[cnt >= min_support].index.tolist()
    if len(anc) < 5 or len(sires) < 30:
        return {"note": "祖先または種牡馬が少なすぎる"}
    X = np.array([[0.5 ** P[s].get(a, 99) if a in P[s] else 0.0 for a in anc] for s in sires])
    y = d["delta"].to_numpy(); w = 1.0 / (d["se"].to_numpy() ** 2); w = w / w.mean()
    rng = np.random.default_rng(seed)
    best = None
    for alpha in (0.3, 1, 3, 10, 30):
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            pred[te] = Ridge(alpha=alpha).fit(X[tr], y[tr], sample_weight=w[tr]).predict(X[te])
        r2 = 1 - np.average((y - pred) ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2, weights=w)
        if best is None or r2 > best[1]:
            best = (alpha, r2)
    coef = Ridge(alpha=best[0]).fit(X, y, sample_weight=w).coef_
    sign = np.zeros(len(anc))
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        c = Ridge(alpha=best[0]).fit(X[i], y[i], sample_weight=w[i]).coef_
        sign += np.sign(c) == np.sign(coef)
    out = pd.DataFrame({"anc": anc, "support": cnt[anc].to_numpy(), "coef": coef, "stable": sign / n_boot})
    return {"alpha": best[0], "cv_r2": round(float(best[1]), 3), "n_sires": len(sires), "n_ancestors": len(anc), "table": out}


def cmd_distance(a) -> None:
    m = pd.read_parquet(MINING, columns=["race_id", "horse_id", "date", "distance", "surface", "finish_position", "sire"])
    m["fp"] = pd.to_numeric(m["finish_position"], errors="coerce")
    dts = pd.to_datetime(m["date"])
    w = m[(m["fp"] == 1) & (m["surface"] == a.surface) & (dts >= "2020-01-01") & (dts < "2026-01-01")].copy()       # 2026 年は使わない
    w["horse_id"], w["sire"] = w["horse_id"].astype(str), w["sire"].astype(str)
    h = w.groupby(["horse_id", "sire"])["distance"].mean().rename("y").reset_index()
    lin = pd.read_parquet(LINEAGE)
    line = dict(zip(lin["stallion_id"].astype(str), lin["sub_group_label"]))
    name = dict(zip(lin["stallion_id"].astype(str), lin["stallion_name"]))
    h["line"] = h["sire"].map(line).fillna("系統不明")
    g = h.groupby(["sire", "line"])["y"].agg(n="size", ybar="mean", var="var").reset_index()
    g = g[g["n"] >= MIN_HORSES].copy()
    g["var"] = g["var"].fillna(g["var"].median())
    d = eb_shrink(g)
    d["name"] = d["sire"].map(name)
    offs = h.groupby("sire")["horse_id"].apply(list)
    P = {s: sire_pedigree(s, offs[s]) for s in d["sire"]}
    d = d[d["sire"].map(lambda s: len(P[s]) > 0)]
    low = d.nsmallest(max(int(len(d) * 0.2), 8), "delta")["sire"].tolist()
    high = d.nlargest(max(int(len(d) * 0.2), 8), "delta")["sire"].tolist()
    en = fisher_enrich({s: P[s] for s in d["sire"]}, low, high)
    nm = lambda x: name.get(x) or x
    if len(en):
        en["name"] = en["anc"].map(nm)
        en["direction"] = np.where(en["in_low"] / en["n_low"] > en["in_high"] / en["n_high"], "短くする側", "長くする側")
    rg = ridge_ancestors(P, d.reset_index(drop=True))
    rtab = rg.pop("table", None)
    if rtab is not None:
        rtab["name"] = rtab["anc"].map(nm)
        stab = rtab[rtab["stable"] >= 0.9]
        rg["short_side"] = stab.nsmallest(12, "coef")[["name", "support", "coef", "stable"]].round(3).to_dict("records")
        rg["long_side"] = stab.nlargest(12, "coef")[["name", "support", "coef", "stable"]].round(3).to_dict("records")
        rg["n_stable_0.9"] = int(len(stab))
    rep = {"ridge": rg, "axis": f"{a.surface}の勝利距離（馬ごとの平均）", "n_sires": int(len(d)), **{k: round(float(v), 4) for k, v in d.attrs.items()},
           "lines": d.groupby("line").agg(n_sires=("sire", "size"), theta=("theta_line", "first"), sd_delta=("delta", "std")).round(1).reset_index().sort_values("n_sires", ascending=False).head(15).to_dict("records"),
           "shorter": d.nsmallest(15, "delta")[["name", "line", "n", "ybar", "theta_line", "theta", "delta", "z"]].round(2).to_dict("records"),
           "longer": d.nlargest(15, "delta")[["name", "line", "n", "ybar", "theta_line", "theta", "delta", "z"]].round(2).to_dict("records"),
           "enrich": en.sort_values("p").head(25).round(4).to_dict("records") if len(en) else [],
           "enrich_q_lt_0.1": int((en["q"] < 0.1).sum()) if len(en) else 0, "enrich_tested": int(len(en))}
    (OUT / f"factor_distance_{a.surface}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("axis", "n_sires", "sigma2", "tau2", "k", "enrich_q_lt_0.1", "enrich_tested")}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("distance")
    d.add_argument("--surface", default="芝")
    d.set_defaults(fn=cmd_distance)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
