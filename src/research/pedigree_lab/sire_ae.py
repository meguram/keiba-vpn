"""市場が血統を読み違える型の記述（1 号の指示 2・log.md「市場が血統を読み違える型の記述」）。尤度の A/E だけ・回収率は見ない。

父ごとの A/E = 勝ち ÷ 確率の和。確率は (i) 市場（確定オッズに温度 1 つを当てた条件付きロジット）、(ii) Benter（市場 ＋ 血統の点）。
開発の行を 2017〜20 と 2021〜24 に分け、各期に産駒が 300 走以上の父で、2 期の相関（件数の重み）・真の sd・上位 10% の持続を出す。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.sire_ae
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, gates
from src.research.pedigree_lab.hyp_tests import clogit


def probs(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    rid = X["race_id"].to_numpy()
    b = clogit(X[cols].to_numpy(), X["win"].to_numpy(float), pd.factorize(rid)[0])
    eta = X[cols].to_numpy() @ b
    e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
    return e / pd.Series(e).groupby(rid).transform("sum").to_numpy()


def table(X: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    T = pd.DataFrame({"sire": X["sire"], "per": np.where(X["date"].dt.year <= 2020, 1, 2), "win": X["win"], "p": p})
    g = T.groupby(["sire", "per"]).agg(w=("win", "sum"), e=("p", "sum"), n=("win", "size"), v=("p", lambda s: float((s * (1 - s)).sum())))
    g["ae"] = g["w"] / g["e"]
    g["noise"] = g["v"] / g["e"] ** 2            # 二項の雑音の分散（A/E の尺度）
    w = g.unstack("per")
    w = w[(w[("n", 1)] >= 300) & (w[("n", 2)] >= 300)]
    return w


def stats(w: pd.DataFrame) -> dict:
    a1, a2 = w[("ae", 1)].to_numpy(), w[("ae", 2)].to_numpy()
    n = (w[("n", 1)] + w[("n", 2)]).to_numpy()
    wt = n / n.sum()
    m1, m2 = np.sum(wt * a1), np.sum(wt * a2)
    cov = np.sum(wt * (a1 - m1) * (a2 - m2))
    var1, var2 = np.sum(wt * (a1 - m1) ** 2), np.sum(wt * (a2 - m2) ** 2)
    corr = cov / np.sqrt(var1 * var2)
    noise = np.sum(wt * (w[("noise", 1)].to_numpy() + w[("noise", 2)].to_numpy()) / 2)
    var_obs = (var1 + var2) / 2
    sd_true = float(np.sqrt(max(var_obs - noise, 0)))
    top = a1 >= np.quantile(a1, 0.9)
    direct = float(np.average(a2[top], weights=n[top]))
    # 見込みの倍率: 観測の A/E で上位 10% を選んだ次の期の期待値 ≈ m2 + 1.755·相関·観測の sd
    # （2 期の相関はすでに雑音の分だけ下がっているので、sd は観測の値を使う。critic 3 で直した）
    implied = m2 + 1.755 * max(corr, 0) * np.sqrt(var2)
    return {"sires": int(len(w)), "corr": round(float(corr), 3), "sd_obs": round(float(np.sqrt(var_obs)), 4),
            "sd_noise": round(float(np.sqrt(noise)), 4), "sd_true": round(sd_true, 4),
            "top10_next_period_AE": round(direct, 4), "implied_top10": round(implied, 4),
            "pass_1_10": bool(direct >= 1.10 and implied >= 1.10)}


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "sire_ae", "父ごとの A/E の記述（尤度だけ・回収率は見ない）"))
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "win", "odds", "horse_number", "sire"])
    X = O.merge(F, on=["race_id", "horse_id"])
    X = X[X["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = X["race_id"].to_numpy()
    inv = 1 / X["odds"].to_numpy()
    X["lq"] = np.log(inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy())
    X["lp"] = np.log(np.clip(X["p_win"], 1e-5, 1))
    res = {}
    wm = table(X, probs(X, ["lq"]))
    wb = table(X, probs(X, ["lp", "lq"]))
    res["market"] = stats(wm)
    res["benter"] = stats(wb)
    top = wm[("ae", 1)].sort_values(ascending=False).head(10)
    res["top10_sires_market"] = [{"sire": s, "ae_2017_20": round(float(wm.loc[s, ("ae", 1)]), 3), "ae_2021_24": round(float(wm.loc[s, ("ae", 2)]), 3)}
                                 for s in top.index]
    res["judge"] = {"corr_in_0.3_0.5": bool(0.3 <= res["market"]["corr"] <= 0.5),
                    "benter_corr_lower": bool(res["benter"]["corr"] < res["market"]["corr"]),
                    "close_1_10": not res["market"]["pass_1_10"]}
    path = data.OUT / "sire_ae.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
