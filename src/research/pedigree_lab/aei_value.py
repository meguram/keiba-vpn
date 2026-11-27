"""平均獲得賞金（AEI 型・1 走あたり）で見た種牡馬・母父・牝系・組み合わせの価値（log.md「平均獲得賞金で見た…」の「作りの直し」。開発の行）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.aei_value [--null 1] [--fit]

- 各走の b = 本賞金 ÷ 前の年 × 年齢の区分（rage）の全走りの 1 走あたりの平均本賞金（公開の AEI と同じく年の平均で割る。格が残る）。
  本賞金は prize_race.parquet の着順ごとの実額（無ければ 1 着 × 比）。同着は、占める着順の額の平均を分け合う（critic）。
  重い裾は、前の年の全走りの 99.5% 点で切り詰める（2018 年は 2017 年の点。2 号）。集計には 2018 年以降の走りだけを使う。
- 列（as-of・その日より前・自分の過去走を引く。public_rules.sums を使う）:
  縮める先は、その日より前の全走りの b の平均 c（約 0.92。累計 100 走以下の最初の日だけ 1）。
  av_sire・av_bms: 父・母父の産駒の b の平均（擬似件数 200 走で c へ）。
  av_fam: 母・2 代母・3 代母の牝系の子孫（母の筋でその牝馬を持つ馬）の b の平均（擬似件数 30 走で c へ）を 2^−代 で重み付けた値。
  av_ci: 同じ母の、ほかの父の子の b の平均（母の全部の子 − 同じ父 × 同じ母の子。擬似件数 10 走で c へ）。
  av_lift: 父の産駒の (b − その走りの av_ci) の平均（擬似件数 200 走で 0 へ）。母の質を引いた父の上積み。
  av_nick_add・av_nick_mul: 父 × 母父の組の b の平均を、見込み（av_sire + av_bms − c・av_sire × av_bms ÷ c）へ擬似件数 100 走で縮め、見込みを引いた値。
- 帰無（--null）: 借り手の血統（祖先と組の集計は本物の血統表で作り、各馬に同じ生年の別の馬の祖先の並びを貸す）。
- --fit: (A) ＋ 列を時間の順（学習は 2018 年から）で当て、行ごとの予測を aei_wf{tag}.parquet に残す。列は aei_cols{tag}.parquet に残す。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.public_rules import parse, slot_join, sums

RATIO = {1: 1.0, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.1}
START = pd.Timestamp("2018-01-01")
COLS = ["av_sire", "av_bms", "av_fam", "av_ci", "av_lift", "av_nick_add", "av_nick_mul"]


def run_prize(X: pd.DataFrame) -> np.ndarray:
    """各走の本賞金（万円）。同着は占める着順の額の平均。"""
    T = pd.read_parquet(data.OUT / "prize_race.parquet")
    p1 = T["pz1"].fillna(T["p1"])
    E = pd.DataFrame({"race_id": T["race_id"]})
    for k, r in RATIO.items():
        E[k] = T[f"pz{k}"].fillna(p1 * r)
    M = X[["race_id", "finish_position"]].copy()
    M["fp"] = pd.to_numeric(M["finish_position"], errors="coerce")
    M["cnt"] = M.groupby(["race_id", "fp"])["fp"].transform("size")
    Mm = M.merge(E, on="race_id", how="left")
    fp, cnt = Mm["fp"].to_numpy(), Mm["cnt"].to_numpy()
    # 実額があればそのまま使う（同着なら、すでに分けた後の額。critic）。無い着順だけ、占める着順の比の額の平均を分け合う
    P = T.set_index("race_id")
    act = np.full(len(M), np.nan)
    for k in RATIO:
        v = Mm["race_id"].map(P[f"pz{k}"]).to_numpy()
        act = np.where(fp == k, v, act)
    tot = np.zeros(len(M))
    p1m = Mm["race_id"].map(p1.set_axis(T["race_id"])).to_numpy()
    for k, r in RATIO.items():
        inside = (fp <= k) & (k <= fp + cnt - 1)
        tot += np.where(inside, p1m * r, 0.0)
    e = np.where(np.isfinite(act), act, tot / cnt)
    e[~np.isfinite(e)] = np.nan
    has = X["race_id"].isin(T["race_id"]).to_numpy()
    e[~has] = np.nan
    return e


def run_b(X: pd.DataFrame) -> np.ndarray:
    e = run_prize(X)
    yr = X["date"].dt.year.to_numpy()
    D = pd.DataFrame({"y": yr, "g": X["rage"].astype(str).to_numpy(), "e": e})
    # 割る数は前の年 × 区分の 1 走あたりの平均（as-of。critic）。前の年が無い年（2016 年）は同じ年
    m = D.groupby(["y", "g"])["e"].mean()
    mu = np.array([m.get((y - 1, g), m.get((y, g), np.nan)) for y, g in zip(D["y"], D["g"])])
    b = e / mu
    cap = pd.Series(b).groupby(yr).quantile(0.995)
    c = pd.Series(yr).map(lambda y: cap.get(y - 1, cap.get(y))).to_numpy()
    return np.minimum(b, c)


def extend(A: pd.DataFrame) -> pd.DataFrame:
    """組の札（gen 98 = 父 × 母父）と、父 × 母（gen 97・CI 用）を祖先の表に足す。"""
    W = A[A["gen"] <= 2].pivot_table(index="horse_id", columns=["gen", "pos"], values="anc", aggfunc="first")
    s, d, b = W.get((1, 0)), W.get((1, 1)), W.get((2, 2))
    E = [A]
    for gen, x, y in ((98, s, b), (97, s, d)):
        ok = x.notna() & y.notna()
        E.append(pd.DataFrame({"horse_id": W.index[ok], "gen": gen, "pos": 0, "anc": (x[ok] + "|" + y[ok]).to_numpy()}))
    return pd.concat(E, ignore_index=True)


def build(X: pd.DataFrame, A: pd.DataFrame, Ab: pd.DataFrame | None) -> pd.DataFrame:
    b = run_b(X)
    use = (X["date"] >= START).to_numpy() & np.isfinite(b)
    R = X[["date", "horse_id"]].reset_index().rename(columns={"index": "row"})
    R["_b"], R["_one"] = np.where(use, b, 0.0), use.astype(float)
    n = len(X)
    out = pd.DataFrame(index=np.arange(n))

    def pick(Z, slots):
        return None if Z is None else Z[pd.MultiIndex.from_frame(Z[["gen", "pos"]]).isin(pd.MultiIndex.from_tuples(slots))]

    def one(slot, vals=("_b", "_one")):
        L, s = sums(R, pick(A, [slot]), pick(Ab, [slot]), ["anc"], list(vals))
        S = pd.DataFrame(s, columns=list(vals)).assign(row=L["row"].to_numpy()).groupby("row").first().reindex(out.index)
        return S

    # 縮める先は、その日より前の全走りの b の平均 c（1 ではない。走りの単位の平均は約 0.92。critic）
    cs, _ = asof(R.assign(_k="1"), ["_k"], ["_b", "_one"])
    c_row = np.where(cs[:, 1] > 100, cs[:, 0] / np.maximum(cs[:, 1], 1), 1.0)
    cser = pd.Series(c_row, index=out.index)
    S = one((1, 0)); out["av_sire"] = (S["_b"].fillna(0) + 200 * cser) / (S["_one"].fillna(0) + 200)
    S = one((2, 2)); out["av_bms"] = (S["_b"].fillna(0) + 200 * cser) / (S["_one"].fillna(0) + 200)
    fam = [(1, 1), (2, 3), (3, 7)]
    # 子孫は牝系（母 → 母 → …の筋）でその牝馬を持つ馬だけ（息子の種牡馬を通した孫は入れない。critic）
    tail = [(1, 1), (2, 3), (3, 7), (4, 15), (5, 31)]
    L, s = sums(R, pick(A, tail), pick(Ab, tail), ["anc"], ["_b", "_one"])
    v = (s[:, 0] + 30 * c_row[L["row"].to_numpy()]) / (s[:, 1] + 30)
    J = slot_join(L, v, A if Ab is None else Ab, fam)
    w = 0.5 ** J["gen"].to_numpy()
    d = pd.DataFrame({"row": J["row"], "w": w, "wv": w * J["v"]}).groupby("row")[["w", "wv"]].sum()
    out["av_fam"] = (d["wv"] / d["w"]).reindex(out.index).fillna(cser).to_numpy()
    Sd, Ssd = one((1, 1)), one((97, 0))
    nb = Sd["_b"].fillna(0) - Ssd["_b"].fillna(0)
    nn = Sd["_one"].fillna(0) - Ssd["_one"].fillna(0)
    out["av_ci"] = ((nb + 10 * cser) / (nn + 10)).to_numpy()
    # 父の上積み: 産駒の (b − av_ci) の as-of の平均（CI のある走りだけ）
    has_ci = (nn.to_numpy() > 0) & use
    R2 = R.copy()
    R2["_d"] = np.where(has_ci, b - out["av_ci"].to_numpy(), 0.0)
    R2["_one"] = has_ci.astype(float)
    L, s = sums(R2, pick(A, [(1, 0)]), pick(Ab, [(1, 0)]), ["anc"], ["_d", "_one"])
    S = pd.DataFrame(s, columns=["_d", "_one"]).assign(row=L["row"].to_numpy()).groupby("row").first().reindex(out.index)
    out["av_lift"] = (S["_d"].fillna(0) / (S["_one"].fillna(0) + 200)).to_numpy()
    S = one((98, 0))
    x, y = out["av_sire"].to_numpy(), out["av_bms"].to_numpy()
    for kind, prior in (("add", x + y - c_row), ("mul", x * y / c_row)):
        out[f"av_nick_{kind}"] = (S["_b"].fillna(0).to_numpy() + 100 * prior) / (S["_one"].fillna(0).to_numpy() + 100) - prior
    out["b"] = b                                              # 各走の b（記述と回収率の集計用。学習の列には入れない）
    return out


def main() -> None:
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    fit = "--fit" in sys.argv
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "aei_value" + ("_null" if null_seed is not None else ""),
                                   "平均獲得賞金で見た血統の価値の列（as-of・尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    ped = data.load_ped()
    A = extend(parse(ped[ped["horse_id"].isin(X["horse_id"].unique())]))
    Ab = None
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        H = pd.Series(A["horse_id"].unique())
        hb = H.str[:4].to_numpy()
        tgt = H.to_numpy().copy()
        for bb in np.unique(hb):
            ii = np.flatnonzero(hb == bb)
            tgt[ii] = H.to_numpy()[rng.permutation(ii)]
        Ab = A.assign(horse_id=A["horse_id"].map(dict(zip(tgt, H.to_numpy()))))
    C = build(X, A, Ab)
    tag = "" if null_seed is None else f"_null{null_seed}"
    print("coverage", C[COLS].notna().mean().round(3).to_dict(), flush=True)
    print("mean", C[COLS].mean().round(4).to_dict(), "sd", C[COLS].std().round(4).to_dict(), flush=True)
    path = data.OUT / f"aei_cols{tag}.parquet"
    X[["race_id", "horse_id", "date"]].join(C).to_parquet(path, index=False)
    if fit:
        from src.research.pedigree_lab.walkforward import fit_year
        for c in COLS:
            X[c] = C[c].to_numpy()
        X = X[X["date"].dt.year >= 2018].reset_index(drop=True)   # 集計が 2018 年からなので、学習もそろえる（critic・2 号）
        use = [c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + COLS
        yr = X["date"].dt.year.to_numpy()
        p = np.full(len(X), np.nan)
        for t in range(2020, 2025):
            p[yr == t] = fit_year(X, use, "win", t, 2000)
            print(t, flush=True)
        te = yr >= 2020
        o = X.loc[te, ["race_id", "horse_id"]].copy()
        o["p_TB"] = p[te]
        path2 = data.OUT / f"aei_wf{tag}.parquet"
        o.to_parquet(path2, index=False)
        path = f"{path} {path2}"
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
