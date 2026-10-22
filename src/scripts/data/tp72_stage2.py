"""tp-72 第 2 段: 記述子のペア（39 個 → 741 ペア）の交互作用を、モデルの残差に対して系統的に走査する（学習を伴わない集計版）。

python -m src.scripts.data.tp72_stage2 plan                 # ペア数・計算量の見積りだけ
python -m src.scripts.data.tp72_stage2 run [--out DIR]      # セル集計による走査（木・GBM の学習はしない）

各ペア (i,j)・各目的変数（勝ち残差 / 複勝残差 / 穴馬 odds>=12 の複勝残差）について:
  主効果だけ: logit(p) に水準ごとのずれ δ_i + δ_j を足す（残差の 1 ステップ Newton、リッジ λ=20 で縮約。2 回の往復で当てはめ）
  主効果+交互作用: さらにセル (i,j) ごとのずれ δ_ij（縮約 λ=20）を足す
  発見期間 2022〜2024 を 1 年ずつ外し（LOYO）、外側の年での対数損失の改善 ΔLL = LL(主効果) − LL(主効果+交互作用) をレース単位クラスタ z で検定
  経験的な帰無（z の中央値・MAD）で補正し、BH（741 × 3 目的変数）。
上位ペアのセルは、発見期間で当てはめた δ_ij の大きい正のセル（n>=300）を test 2025 で 1 回だけ確認し、2026 は記述統計のみ。
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp72_desc import descriptors
from src.scripts.data.tp72_eval import CONF, DESC, DISC, OUT, bh, boot_roi, cluster_z, phi, prepare

LAM = 20.0
EPS = 1e-6
MIN_CELL_N = 300


def _lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sg(z):
    return 1.0 / (1.0 + np.exp(-z))


def _shift(code: np.ndarray, k: int, y: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    g = np.bincount(code, weights=y - p, minlength=k)
    h = np.bincount(code, weights=p * (1 - p), minlength=k)
    return g / (h + lam)


def fit_pair(ci: np.ndarray, cj: np.ndarray, ki: int, kj: int, y: np.ndarray, p: np.ndarray):
    z0 = _lg(p)
    di = np.zeros(ki)
    dj = np.zeros(kj)
    for _ in range(2):
        pp = _sg(z0 + dj[cj])
        di = _shift(ci, ki, y, pp, LAM)
        pp = _sg(z0 + di[ci])
        dj = _shift(cj, kj, y, pp, LAM)
    pm = _sg(z0 + di[ci] + dj[cj])
    cell = ci * kj + cj
    dc = _shift(cell, ki * kj, y, pm, LAM)
    return di, dj, dc


def logloss_row(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def plan(nd: int = 43, rows: int = 140_000) -> dict:
    pairs = nd * (nd - 1) // 2
    return {"n_descriptors": nd, "n_pairs": pairs, "targets": 3, "folds": 3, "cell_fits": pairs * 3 * 3,
            "est_seconds_single_core": pairs * 3 * 3 * 0.03, "note": "bincount のみ（学習なし）。GBM/木での走査は保留"}


def run(out_dir: Path) -> dict:
    d = prepare()
    Dm, L = descriptors(d)
    names = list(Dm.columns)
    K = [len(L[n]) for n in names]
    codes = {n: Dm[n].to_numpy() for n in names}
    yr = d["year"].to_numpy()
    rc = d["rc"].to_numpy()
    nrace = int(rc.max()) + 1
    targets = {
        "win": (d["y1"].to_numpy(), d["p_win"].to_numpy(), np.ones(len(d), bool)),
        "top3": (d["y3"].to_numpy(), d["p_top3"].to_numpy(), np.ones(len(d), bool)),
        "ls_top3": (d["y3"].to_numpy(), d["p_top3"].to_numpy(), (d["win_odds"] >= 12).to_numpy()),
    }
    disc = np.isin(yr, DISC)
    results = []
    for (a, b) in combinations(range(len(names)), 2):
        na, nb = names[a], names[b]
        ca, cb = codes[na], codes[nb]
        ok = (ca >= 0) & (cb >= 0)
        for tn, (y, p, sub) in targets.items():
            use = ok & sub & disc
            gains = np.zeros(len(d))
            for h in DISC:
                tr = use & (yr != h)
                te = use & (yr == h)
                if tr.sum() < 2000 or te.sum() < 500:
                    continue
                di, dj, dc = fit_pair(ca[tr], cb[tr], K[a], K[b], y[tr], p[tr])
                z0 = _lg(p[te])
                pm = _sg(z0 + di[ca[te]] + dj[cb[te]])
                pf = _sg(z0 + di[ca[te]] + dj[cb[te]] + dc[ca[te] * K[b] + cb[te]])
                gains[te] = logloss_row(y[te], pm) - logloss_row(y[te], pf)
            m = use & (gains != 0)
            R = np.bincount(rc[use], weights=gains[use], minlength=nrace)
            den = math.sqrt(float((R ** 2).sum()))
            z = float(R.sum() / den) if den > 0 else float("nan")
            results.append({"i": na, "j": nb, "target": tn, "n": int(use.sum()), "mean_gain": float(gains[use].mean()) if use.any() else float("nan"), "z": z})
    df = pd.DataFrame(results)
    out = {"plan": plan(len(names)), "n_tests": int(len(df))}
    for tn in targets:
        s = df[df["target"] == tn]["z"].dropna()
        med = float(s.median())
        sd = float(1.4826 * (s - med).abs().median())
        out[f"null_{tn}"] = {"median": med, "mad_sd": sd}
    df["z_adj"] = df.apply(lambda r: (r["z"] - out[f"null_{r['target']}"]["median"]) / max(out[f"null_{r['target']}"]["mad_sd"], 1e-6), axis=1)
    df["p"] = df["z_adj"].map(lambda z: 1.0 if np.isnan(z) else 1 - phi(z))
    df["p_raw"] = df["z"].map(lambda z: 1.0 if np.isnan(z) else 1 - phi(z))
    df["q"] = bh(df["p"].to_numpy())
    df["q_raw"] = bh(df["p_raw"].to_numpy())
    out["n_q_lt_0.10"] = int((df["q"] < 0.10).sum())
    out["n_q_raw_lt_0.10"] = int((df["q_raw"] < 0.10).sum())
    out["pairs"] = df.sort_values("z_adj", ascending=False).head(60).to_dict("records")
    # 上位ペア（q<0.10 または z_adj 上位 15）のセルを 2025 で確認
    top = df.sort_values("z_adj", ascending=False)
    pick = pd.concat([top[top["q"] < 0.10], top.head(15)]).drop_duplicates(["i", "j", "target"])
    cells = []
    for _, r in pick.iterrows():
        a, b = names.index(r["i"]), names.index(r["j"])
        ca, cb = codes[r["i"]], codes[r["j"]]
        y, p, sub = targets[r["target"]]
        use = (ca >= 0) & (cb >= 0) & sub & disc
        di, dj, dc = fit_pair(ca[use], cb[use], K[a], K[b], y[use], p[use])
        cn = np.bincount(ca[use] * K[b] + cb[use], minlength=K[a] * K[b])
        order = np.argsort(-dc)
        for c in order[:3]:
            if dc[c] <= 0 or cn[c] < MIN_CELL_N:
                continue
            la, lb = c // K[b], c % K[b]
            m = (ca == la) & (cb == lb) & sub
            entry = {"i": r["i"], "j": r["j"], "target": r["target"], "cell": f"{L[r['i']][la]} × {L[r['j']][lb]}", "delta": float(dc[c]), "n_disc": int(cn[c])}
            for tag, sel in (("y2025", yr == CONF), ("y2026", yr == DESC)):
                mm = m & sel
                n = int(mm.sum())
                e = {"n": n}
                if n:
                    for k in ("m1", "k1", "m3", "k3", "r1", "r3"):
                        z, tot, _ = cluster_z(d, mm, "e_" + k, nrace)
                        e["z_" + k] = z
                        e["mean_" + k] = tot / n
                    e["roi_win"] = boot_roi(d, mm, "r1")
                    e["roi_place"] = boot_roi(d, mm, "r3")
                entry[tag] = e
            cells.append(entry)
    out["cells"] = cells
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    df.to_parquet(out_dir / "stage2_pairs.parquet", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan(), ensure_ascii=False, indent=1))
    else:
        r = run(Path(a.out))
        print(json.dumps({k: v for k, v in r.items() if k not in ("pairs", "cells")}, ensure_ascii=False, indent=1))
