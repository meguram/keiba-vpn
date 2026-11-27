"""米の札 × ダートの道悪の形の確かめ（log.md「調べ K3」。記述・買い方の設計ではない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.us_dirt_wet

- 行: ダートの全部の行（split 0・1）。父の札は national_type の 2020〜24 年の表（系統で決まる固定の属性）。
- 式: レースの中の条件付きロジット。勝ちの確率 ∝ exp(η)、η = log q の 3 次 + 札の印（米・欧。日が基準）
  + 米 × 稍重・米 × 重・米 × 不良 + 欧 × 道悪。q は確定の単勝オッズの逆数をレースの中で和 1 に直した値（2 号）。
- 期間: 2016〜18 年（確かめ）と 2019〜24 年（見つけた期間）。年ごとの「米 × 道悪（稍重・重・不良をまとめた 1 本）」の係数も出す（2 号）。
- 区間: 開催日の単位のブートストラップ（300 回・レースを丸ごと引き直す）。種牡馬の単位は作りが壊れていたのでやめた（critic）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.research.pedigree_lab import data
from src.research.pedigree_lab.national_type import add_cols


def clogit(Z: np.ndarray, y: np.ndarray, rid: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """レースの中の条件付きロジット（重みつき）。rid は 0 始まりの連番。"""
    nr = rid.max() + 1
    w = np.ones(len(y)) if w is None else w

    def f(b):
        eta = Z @ b
        m = np.full(nr, -np.inf); np.maximum.at(m, rid, eta)
        e = np.exp(eta - m[rid])
        s = np.bincount(rid, weights=e, minlength=nr)
        p = e / s[rid]
        wr = np.bincount(rid, weights=w * y, minlength=nr)                 # 勝ち馬の重み（レースの重み）
        ll = (w * y * (eta - m[rid])).sum() - (wr * np.log(np.maximum(s, 1e-300))).sum()
        g = Z.T @ (w * y) - Z.T @ (p * wr[rid])
        return -ll, -g
    r = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    if not r.success:
        raise RuntimeError(f"条件付きロジットが収束しない: {r.message}")         # 収束を確かめる（critic）
    return r.x


def design(D: pd.DataFrame, pooled_wet: bool) -> tuple[np.ndarray, list[str]]:
    lq = np.log(D["q"].clip(1e-4, 1).to_numpy())
    lq = (lq - (-2.6)) / 1.0                                                  # 尺度をそろえる（3 次の項が大きくなり過ぎないように。critic）
    us, eu = D["_cty_s"].eq("米").to_numpy(float), D["_cty_s"].eq("欧").to_numpy(float)
    tc = D["track_condition"].astype(str)
    s_, h_, b_ = tc.eq("稍重").to_numpy(float), tc.eq("重").to_numpy(float), tc.eq("不良").to_numpy(float)
    wet = s_ + h_ + b_
    cols = [lq, lq ** 2, lq ** 3, us, eu, eu * wet]
    names = ["lq", "lq2", "lq3", "米", "欧", "欧×道悪"]
    if pooled_wet:
        cols.append(us * wet); names.append("米×道悪")
    else:
        cols += [us * s_, us * h_, us * b_]; names += ["米×稍重", "米×重", "米×不良"]
    return np.column_stack(cols), names


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "us_dirt_wet", "米の札 × ダートの道悪の確かめ（記述）"))
    T = pd.read_parquet(data.OUT / "national_type_table.parquet")
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["split"].isin([0, 1]) & (X["date"] <= pd.Timestamp(data.DEV_END)) & X["surf"].eq("D")].reset_index(drop=True)
    add_cols(X, T)
    od = pd.to_numeric(X["odds"], errors="coerce").where(lambda v: v > 0)
    X["q"] = (1 / od) / (1 / od).groupby(X["race_id"]).transform("sum")
    ok = X.groupby("race_id")["q"].transform(lambda s: s.notna().all()) & X.groupby("race_id")["_cty_s"].transform(lambda s: s.notna().all())
    X = X[ok].reset_index(drop=True)                                        # レースの全頭に q と札がそろうレースだけ
    X = X[X.groupby("race_id")["win"].transform("sum") == 1].reset_index(drop=True)   # 勝ち馬が 1 頭のレース（同着を除く）
    rng = np.random.default_rng(3)
    res = {}
    for per, (a, b) in (("2016〜18 年（確かめ）", (2016, 2018)), ("2019〜24 年（見つけた期間）", (2019, 2024))):
        D = X[X["date"].dt.year.between(a, b)].reset_index(drop=True)
        rid = pd.factorize(D["race_id"])[0]
        y = D["win"].to_numpy(float)
        Z, names = design(D, False)
        b0 = clogit(Z, y, rid)
        tc = D["track_condition"].astype(str)
        us = D["_cty_s"].eq("米")
        cnt = {g_: {"走り": int((tc == g_).sum()), "レース": int(D.loc[tc == g_, "race_id"].nunique()), "米の走り": int(((tc == g_) & us).sum()),
                    "米の勝ち": int(D.loc[(tc == g_) & us, "win"].sum())} for g_ in ("良", "稍重", "重", "不良")}
        out = {"走り": int(len(D)), "レース": int(rid.max() + 1), "馬場ごとの数": cnt, "係数": dict(zip(names, np.round(b0, 3).tolist())), "区間": {}}
        # 区間は開催日の単位だけ（レースを丸ごと引き直す）。種牡馬の単位で馬に重みを付けると、勝ち馬の重みにしか効かず、
        # 結果で行を選ぶ再抽出になる（critic）
        for unit, key in (("開催日", D["date"].dt.strftime("%Y%m%d").to_numpy()),):
            uu, inv = np.unique(key, return_inverse=True)
            bs = []
            for _ in range(300):
                cnt = np.bincount(rng.integers(0, len(uu), len(uu)), minlength=len(uu)).astype(float)
                bs.append(clogit(Z, y, rid, cnt[inv]))
            bs = np.array(bs)
            out["区間"][unit] = {n: np.percentile(bs[:, i], [2.5, 97.5]).round(3).tolist() for i, n in enumerate(names) if n.startswith(("米", "欧"))}
        Zp, np_ = design(D, True)
        yrs = {}
        for yy in range(a, b + 1):
            m = (D["date"].dt.year == yy).to_numpy()
            r2 = pd.factorize(D.loc[m, "race_id"])[0]
            yrs[int(yy)] = round(float(clogit(Zp[m], y[m], r2)[np_.index("米×道悪")]), 3)
        out["年ごとの米×道悪"] = yrs
        res[per] = out
        print(per, json.dumps(out, ensure_ascii=False), flush=True)
    path = data.OUT / "us_dirt_wet.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
