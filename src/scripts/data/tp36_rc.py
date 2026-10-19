"""tp-36 多重比較の補正: White の Reality Check（スチューデント化した最大統計量のレース単位ブートストラップ）。

python -m src.scripts.data.tp36_rc --out data/local/modeling/betting/tp36/rc.json
帰無仮説 = 『試した方策のどれも平均利益が 0 以下（回収率 100% 以下）』。p 値が小さいときだけ『一番良かった方策は偶然ではない』と言える。
族 = 単勝の均等買い方策（市場込み混合・市場なしモデル・市場だけの傾きの補正、それぞれ EV のしきい値違い）。評価年 2023〜2025。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp36_edge2 import market_only

ROOT = Path(__file__).resolve().parents[3]
TP = ROOT / "data/local/modeling/betting/tp36"


def reality_check(profit: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """profit: [レース数, 方策数]（レースごとの利益）。戻り値: 各方策の t 値、最大 t、Reality Check の p 値。"""
    R, K = profit.shape
    mu = profit.mean(0)
    sd = profit.std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.inf)
    t = np.sqrt(R) * mu / sd
    rng = np.random.default_rng(seed)
    tmax = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, R, R)
        pb = profit[idx]
        sdb = pb.std(0, ddof=1)
        sdb = np.where(sdb > 1e-12, sdb, np.inf)
        tmax[b] = (np.sqrt(R) * (pb.mean(0) - mu) / sdb).max()
    return {"t": t.tolist(), "t_max": float(t.max()), "best": int(t.argmax()), "p_value": float((tmax >= t.max()).mean()), "n_strategies": int(K)}


def build_profit_matrix(d: pd.DataFrame, evs: dict[str, pd.Series]) -> tuple[np.ndarray, list[str]]:
    rids = d["race_id"].unique()
    ridx = pd.Series(np.arange(len(rids)), index=rids)
    cols, names = [], []
    for name, ev in evs.items():
        m = ev >= float(name.rsplit("_", 1)[1])
        prof = (d["ret_win"] - 100.0).where(m, 0.0)
        agg = prof.groupby(d["race_id"]).sum().reindex(rids).fillna(0.0).to_numpy()
        cols.append(agg)
        names.append(name)
    return np.stack(cols, 1), names


def main_single(raw: str, out: str) -> None:
    """1 点買い・k 点掃引の全方策（券種 × 確率 × オッズ源 × {k 掃引, k=1 の最小エッジ}）を一つの族とした Reality Check。"""
    import pickle

    from src.scripts.data.tp36_single import KS, key_arrays

    d = pickle.loads(Path(raw).read_bytes())
    recs, meta = d["recs"], d["meta"]
    order = sorted(range(len(meta)), key=lambda i: (meta[i][1], meta[i][0]))
    recs = [recs[i] for i in order]
    R = len(recs)
    keys = sorted({k for r in recs for k in r if k[0] != "all"})
    cols, names = [], []
    for key in keys:
        ev, _, py = key_arrays(recs, key, R)
        for k in KS:
            bought = np.isfinite(ev[:, :k])
            cols.append(((py[:, :k] - 100.0) * bought).sum(1).astype(np.float32))
            names.append("|".join(key) + f"|k{k}")
        for x in (1.0, 1.2, 1.5, 2.0):
            bought = np.isfinite(ev[:, :1]) & (ev[:, :1] >= x)
            cols.append(((py[:, :1] - 100.0) * bought).sum(1).astype(np.float32))
            names.append("|".join(key) + f"|k1|ev{x}")
    P = np.stack(cols, 1)
    ok = P.std(0) > 0
    P, names = P[:, ok], [n for n, o in zip(names, ok) if o]
    res = {"n_races": R, "n_strategies": len(names), "all": reality_check(P, n_boot=1000)}
    fam = {"real_odds": [i for i, n in enumerate(names) if n.split("|")[2] == "real"], "est_odds": [i for i, n in enumerate(names) if n.split("|")[2] == "est"],
           "model_only": [i for i, n in enumerate(names) if n.split("|")[1] == "model"], "mixed": [i for i, n in enumerate(names) if n.split("|")[1] == "mixed"]}
    for f, idx in fam.items():
        r = reality_check(P[:, idx], n_boot=1000)
        r["best_name"] = names[idx[r["best"]]]
        r["n_strategies"] = len(idx)
        r.pop("t", None)
        res[f] = r
    res["best_name_all"] = names[res["all"]["best"]]
    res["all"].pop("t", None)
    res["top_t"] = sorted(((n, round(float(t), 2)) for n, t in zip(names, (P.mean(0) / P.std(0, ddof=1)) * np.sqrt(R))), key=lambda x: -x[1])[:12]
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TP / "rc.json"))
    ap.add_argument("--frame", default=str(TP / "edge_frame.parquet"))
    ap.add_argument("--single-raw", default="", help="指定すると 1 点買いの族の Reality Check を行う")
    a = ap.parse_args()
    if a.single_raw:
        main_single(a.single_raw, a.out)
        return
    d = pd.read_parquet(a.frame)
    d = d[d["year"] >= 2022].reset_index(drop=True)
    pm = market_only(d)
    d = d.assign(p_mkt=pm)
    dd = d[d["year"] >= 2023].reset_index(drop=True)
    taus = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5]
    fam: dict[str, pd.Series] = {}
    for kind, col in (("mixed", "p_blend"), ("model", "p_win"), ("market_only", "p_mkt")):
        ev = dd[col] * dd["win_odds"]
        for t in taus:
            fam[f"{kind}_{t}"] = ev
    P, names = build_profit_matrix(dd, fam)
    out: dict = {"n_races": int(P.shape[0]), "strategies": names}
    out["all"] = reality_check(P)
    sel = [i for i, n in enumerate(names) if n.startswith("mixed_")]
    out["mixed_only"] = reality_check(P[:, sel])
    sel2 = [i for i, n in enumerate(names) if n.startswith("model_")]
    out["model_only_no_market"] = reality_check(P[:, sel2])
    out["best_name_all"] = names[out["all"]["best"]]
    out["best_name_mixed"] = [names[i] for i in sel][out["mixed_only"]["best"]]
    out["t_by_strategy"] = {n: round(t, 3) for n, t in zip(names, out["all"]["t"])}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
