"""連系の想定オッズの換算: 市場の単勝の確率から Harville で組の確率 P_hat を作り、log(オッズ) = poly2(log(1/P_hat)) で想定オッズにする。

既存の作り（src/pipeline/inference/target_bets.py の all_candidates と、係数の当てはめ src/scripts/data/odds_pred_eval.py）と同じ形。
既存の係数（odds_pred_config.json の poly2）は 2022〜25 の払戻で当てはめてあり、入力も sec77 の想定単複から MB60 で作った確率なので、
ここでは訓練（2019〜22）の確定の単勝オッズからの Harville の P_hat と、当たった組の確定の払戻で当てはめ直す（3連複・3連単の既存の当てはめと同じく、価格は当たった組の払戻だけ）。
判定のときは、同じ係数を T-60 の再現の単勝オッズからの P_hat に当てる。費用 0 円。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import bet_cycles as bc

TYPES = ("fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan", "wakuren")
ORIG_CONFIG = Path("/home/jovyan/work/keiba-vpn/data/local/modeling/odds_pred/odds_pred_config.json")


def winners_table(years, frames: dict, waku: dict) -> pd.DataFrame:
    """訓練の各レースの当たった組の (券種, P_hat, 払戻)。P_hat は確定の単勝オッズからの Harville。"""
    from src.pipeline.inference.exotic_probabilities import RankDistribution

    fl = bc.load_runners(years)
    fl = fl[~fl["scratched"]]
    pay = bc.load_payoffs(set(fl["race_id"]))
    bc.fill_payoffs_from_flat(pay, years)
    rows = []
    for rid, g in fl.groupby("race_id", sort=False):
        pr = pay.get(rid)
        if not pr or len(g) < 3 or g["odds"].isna().any():
            continue
        hn = g["horse_number"].astype(int).to_numpy()
        q = 1.0 / g["odds"].to_numpy(float)
        d = RankDistribution(win=q / q.sum())
        P2, P3, n = d._p2, d._p3, len(hn)
        idx = {h: i for i, h in enumerate(hn)}
        t2, t3 = d.top2(), d.top3()

        def ix(c, sep):
            return [idx.get(int(x), -1) for x in c.split(sep)]

        for c, v in pr.get("fukusho", {}).items():
            i = idx.get(int(c), -1)
            if i >= 0 and n >= 5:
                rows.append(("fukusho", t3[i] if n >= 8 else t2[i], v))
        for c, v in pr.get("umaren", {}).items():
            a, b = ix(c, "-")
            if min(a, b) >= 0:
                rows.append(("umaren", P2[a, b] + P2[b, a], v))
        for c, v in pr.get("wide", {}).items():
            a, b = ix(c, "-")
            if min(a, b) >= 0:
                s = P3[a, b, :] + P3[b, a, :] + P3[a, :, b] + P3[b, :, a] + P3[:, a, b] + P3[:, b, a]
                rows.append(("wide", float(s.sum()), v))
        for c, v in pr.get("umatan", {}).items():
            a, b = ix(c, ">")
            if min(a, b) >= 0:
                rows.append(("umatan", P2[a, b], v))
        for c, v in pr.get("sanrenpuku", {}).items():
            a, b, e = ix(c, "-")
            if min(a, b, e) >= 0:
                rows.append(("sanrenpuku", P3[a, b, e] + P3[a, e, b] + P3[b, a, e] + P3[b, e, a] + P3[e, a, b] + P3[e, b, a], v))
        for c, v in pr.get("sanrentan", {}).items():
            a, b, e = ix(c, ">")
            if min(a, b, e) >= 0:
                rows.append(("sanrentan", P3[a, b, e], v))
        wk = waku.get(rid)
        if wk:
            fr = np.array([frames.get((rid, int(h)), 0) for h in hn])
            if (fr > 0).all():
                U = P2 + P2.T
                for c, v in wk.items():
                    fa, fb = (int(x) for x in c.split("-"))
                    A_, B_ = np.where(fr == fa)[0], np.where(fr == fb)[0]
                    s = U[np.ix_(A_, B_)].sum() / (2.0 if fa == fb else 1.0)
                    if s > 0:
                        rows.append(("wakuren", float(s), v))
    t = pd.DataFrame(rows, columns=["bet_type", "P_hat", "payout"])
    t = t[(t["P_hat"] > 0) & (t["payout"] > 0)]
    t["x"] = np.log(1.0 / t["P_hat"])
    t["y"] = np.log(t["payout"] / 100.0)
    return t


def fit(t: pd.DataFrame) -> dict:
    out = {}
    for bt in TYPES:
        d = t[t["bet_type"] == bt]
        co = np.polyfit(d["x"], d["y"], 2)
        out[bt] = {"coef": [float(c) for c in co], "n": int(len(d)), "rmse_log": float(np.sqrt(np.mean((np.polyval(co, d["x"]) - d["y"]) ** 2)))}
    return out


def compare_original(t: pd.DataFrame) -> dict:
    """元の係数（2022〜25 の当てはめ）を同じ訓練の当たりに当てたときの誤差（log オッズの RMSE と平均のずれ）。"""
    co = json.loads(ORIG_CONFIG.read_text(encoding="utf-8")).get("poly2") or {}
    out = {}
    for bt, c in co.items():
        d = t[t["bet_type"] == bt]
        if d.empty:
            continue
        e = np.polyval(c, d["x"]) - d["y"]
        out[bt] = {"rmse_log": float(np.sqrt(np.mean(e ** 2))), "bias_log": float(e.mean()), "n": int(len(d))}
    return out


def original_coefs() -> dict:
    co = json.loads(ORIG_CONFIG.read_text(encoding="utf-8")).get("poly2") or {}
    return {k: {"coef": list(v)} for k, v in co.items()}


def est_odds(coefs: dict, bt: str, p_hat: np.ndarray) -> np.ndarray:
    c = coefs[bt]["coef"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        o = np.exp(np.polyval(c, np.log(1.0 / np.clip(p_hat, 1e-12, None))))
    return np.where(p_hat > 0, np.clip(o, 1.0, 1e6), np.inf).astype(np.float32)
