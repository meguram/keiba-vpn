"""段 3: 事前登録の規則（R0-PL1 の R3 で 1 号が決めたもの）を、開発の行（split 0・1）で確かめる。

- 規則はこのファイルの RULES に固定する（数を見る前に commit して sha を log に残す）。選ぶ点は血統のモデルの点だけで、オッズは使わない。
- 払戻は確定（100 円定額）。区間は開催の単位のブートストラップ。基準線は同じ人気 × 年 × 頭数の帯。
- 帰無は bets.perm_null（点を log q・頭数に回帰した残差を、年 × オッズの 20 分位の中で場面の中だけ並べ替え）。
  登録した全規則の「同じ人気との差」の最大の統計量で Westfall-Young の p を出す。
- 2 つの OOF（開催の単位 = 主・2 代母の単位 = 覚え込みの確かめ）で流す。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.prereg --part dev
保留（--part hold）は 1 号の指示が出てから 1 回だけ。出力: OUT/prereg_<part>.json。費用 0 円。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import bets, data

FIRST_DIRT = lambda X: ((X["first_surf"] == "Y") & (X["surf"] == "D")).to_numpy()


def _race_rank(X: pd.DataFrame, col: str) -> np.ndarray:
    return X.groupby("race_id")[col].rank(ascending=False, method="first").to_numpy()


def rule_p1(X: pd.DataFrame, max_rank: int = 3) -> np.ndarray:
    """初ダートの馬のうち、そのレースで p_win が最も高い 1 頭。レースの全頭の中で p_win が max_rank 位以内。"""
    fd = FIRST_DIRT(X)
    rk = _race_rank(X, "p_win")
    v = np.where(fd, X["p_win"].to_numpy(), -1.0)
    best = pd.Series(v).groupby(X["race_id"].to_numpy()).transform("max").to_numpy()
    return fd & (v == best) & (rk <= max_rank)


RULES: dict = {}     # R3 の決定で埋める: 名前 → (選ぶ関数, 券種, 場面の関数)


def load(part: str, kf: str, reason: str) -> pd.DataFrame:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select" if part == "dev" else "confirm",
                                   f"prereg_{kf}_{part}", reason))
    O = pd.read_parquet(data.OUT / f"{kf}.parquet")
    O = O[O["part"] == part]
    keep = ["race_id", "horse_id", "meeting", "date", "win", "top3", "odds", "popularity", "n_run", "debut", "first_surf",
            "surf", "dchg", "going", "cls", "horse_number", "payoff", "split", "distance"]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=keep)
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    X["year"] = X["date"].dt.year
    X = X[X["odds"] > 0].copy()
    inv = 1.0 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(X["race_id"].to_numpy()).transform("sum").to_numpy()
    X = X.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    return bets.attach_pay(X)


def evaluate(X: pd.DataFrame, n_perm: int) -> dict:
    out = {}
    nulls = {}
    for name, (fn, kind, scene) in RULES.items():
        sel = fn(X)
        r = bets.roi(X, sel, kind)
        yrs = {}
        for t, g in X.groupby("year"):
            m = sel[g.index]
            if m.sum() >= 5:
                rr = bets.roi(g, m, kind, n_boot=50)
                yrs[int(t)] = rr["vs_same_pop"]
        r["years_pos"] = f"{sum(v > 0 for v in yrs.values())}/{len(yrs)}"
        r["by_year_vs_same_pop"] = yrs
        S = X[scene(X)] if scene is not None else X
        nulls[name] = bets.perm_null(S.reset_index(drop=True), "p_win", lambda Y, f=fn: f(Y), kind, n_perm=n_perm)
        r["null_mean"] = round(float(nulls[name].mean()), 4)
        r["p_single"] = round(float((nulls[name] >= r["roi"]).mean()), 4)
        out[name] = r
        print(name, json.dumps(r, ensure_ascii=False), flush=True)
    # Westfall-Young: 規則ごとの（回収率 − 帰無の平均）の最大
    names = list(RULES)
    obs = np.array([out[n]["roi"] - nulls[n].mean() for n in names])
    mx = np.max(np.vstack([nulls[n] - nulls[n].mean() for n in names]), axis=0)
    for n, o in zip(names, obs):
        out[n]["p_wy"] = round(float((mx >= o).mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="dev", choices=["dev", "hold"])
    ap.add_argument("--perm", type=int, default=200)
    a = ap.parse_args()
    res = {}
    for kf in ("kf_v4_pure_meeting", "kf_v4_pure_dd"):
        X = load(a.part, kf, f"R0-PL1 の事前登録の規則の確かめ（{a.part}）")
        res[kf] = evaluate(X, a.perm)
    (data.OUT / f"prereg_{a.part}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(data.OUT / f"prereg_{a.part}.json"))


if __name__ == "__main__":
    main()
