"""公開の手法を規則で作った列（docs/operations/pedigree_lab/public_approaches.md の未試行の分。開発の行・時間の順・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.public_rules --kind D [--null 1]

- D（Dosage 型の血統の期待距離）: 4 代までの牡の祖先（15 頭）ごとに、子孫（5 代に持つ馬）の 3 着内の走りの平均の距離を as-of で出し、
  全体の平均へ擬似件数 200 で縮める。代の重み 16・8・4・2（Dosage と同じ）で平均して期待距離 E を作る。
  芝ダ別に作る（1 号）。列は pr_E・pr_G（子孫の 3 着内の平均の距離 − 出走の平均の距離の重み付きの平均）・pr_dz（(今回の距離 − E) / 400）・pr_adz。chef-de-race の一覧の代わりをデータで作る。
- RF（Rasmussen 型の前半・「優れた」の条件なし）: 5 代以内に同じ牝馬が 2 回以上、別々の子を通して出る印。基準に ib_ を入れる（pr_rf）と、最も近い組の代の和（pr_rf_g。無ければ 11）。
- X（X 因子）: 父の母の父（3:2）の子孫の 3 着内率の偏差を、牝の馬だけに入れる（牝では X の通り道）。
- XP（X の偽薬）: 同じ列を、牡・騸の馬だけに入れる（牡では通り道でない。同じ位置で性だけが違う）。X − XP が X 因子の主張の分。
  母の側の通り道（3:6 など）は、母の父の列（bms_*）の有無で比べにくいので使わない（critic）。
- 子孫は、5 代のどこかにその祖先を持つ馬（重なりは 1 馬 1 行）。値は、指定の枠に出るたびに重みを付けて足す。
- 帰無（--null）: 祖先の集計は本物の血統表のまま作り、各馬に同じ生年の別の馬の祖先の並びを馬の単位で貸して、
  その日より前の集計を引く（2 号の「借り手の血統」。血統表を入れ替えてから集計し直すと、値が全体へ縮んで甘い帰無になる）。
- (A)（第 4 版の 189 列）＋ 列を時間の順（walkforward.fit_year）で当て、行ごとの予測を parquet に残す（compare_null で比べる）。
- 血統表の位置: g 代の p 番（0 始まり）の親は g+1 代の 2p（父）と 2p+1（母）。偶数が牡。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.walkforward import fit_year

K = 200.0
D_SLOTS = [(1, 0)] + [(2, p) for p in (0, 2)] + [(3, p) for p in (0, 2, 4, 6)] + [(4, p) for p in range(0, 16, 2)]
D_W = {1: 16.0, 2: 8.0, 3: 4.0, 4: 2.0}
X_SLOT = (3, 2)     # 父の母の父。牝の馬では X の通り道で、牡の馬では通り道でない（同じ位置で性だけが違う偽薬。critic）


def parse(ped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                rows.append((h, int(g), int(p), i))
    return pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc"])


def runs(X: pd.DataFrame) -> pd.DataFrame:
    R = X[["date", "horse_id", "surf"]].reset_index().rename(columns={"index": "row"})
    d, t3 = X["distance"].astype(float).to_numpy(), X["top3"].astype(float).to_numpy()
    R["_d3"], R["_t3"], R["_dr"], R["_one"] = d * t3, t3, d, 1.0
    return R


def link(R: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    U = A.drop_duplicates(["horse_id", "anc"])[["horse_id", "anc"]]   # data.asof は同じ鍵・同じ馬・同じ日で 1 行が前提
    return R.merge(U, on="horse_id", how="inner").sort_values(["date", "row"], kind="stable").reset_index(drop=True)


def sums(R: pd.DataFrame, A: pd.DataFrame, Ab: pd.DataFrame | None, keys, vals):
    """祖先 × 条件の as-of の和。Ab（借りた血統表）があれば、集計は本物の血統表で作り、値だけを借りた祖先で引く（2 号）。"""
    L = link(R, A)
    s, _ = asof(L, keys, vals)
    if Ab is None:
        return L, s
    Lb = link(R, Ab)
    G = L.groupby(keys + ["date"])[vals].sum().groupby(level=list(range(len(keys)))).cumsum().reset_index().sort_values("date")
    q = Lb.reset_index().rename(columns={"index": "_i"}).sort_values("date")
    m = pd.merge_asof(q[["_i", "date"] + keys], G, on="date", by=keys, allow_exact_matches=False).sort_values("_i")
    sb = m[vals].fillna(0.0).to_numpy()
    # 借りた祖先が本当の祖先でもある行は、自分の過去の走りを引いた本物の値を使う
    T = pd.DataFrame(s, columns=vals).assign(row=L["row"].to_numpy(), anc=L["anc"].to_numpy())
    J = Lb[["row", "anc"]].merge(T, on=["row", "anc"], how="left")
    hit = J[vals[0]].notna().to_numpy()
    sb[hit] = J.loc[hit, vals].to_numpy()
    return Lb, sb


def slot_join(L: pd.DataFrame, vals: np.ndarray, A: pd.DataFrame, slots) -> pd.DataFrame:
    """祖先の値を、指定の枠の行（重なった祖先は出るたびに 1 行）へ戻す。"""
    S = A[pd.MultiIndex.from_frame(A[["gen", "pos"]]).isin(pd.MultiIndex.from_tuples(slots))]
    V = L[["row", "horse_id", "anc"]].assign(v=vals)
    return V.merge(S[["horse_id", "anc", "gen", "pos"]], on=["horse_id", "anc"], how="inner")


def col_D(X: pd.DataFrame, A: pd.DataFrame, Ab=None) -> pd.DataFrame:
    """芝ダ別（1 号）。m3 = 子孫の 3 着内の平均の距離、mr = 子孫の出走の平均の距離。G = m3 − mr は厩舎の距離の選び方を除いた偏り。"""
    R = runs(X)
    vals = ["_d3", "_t3", "_dr", "_one"]
    L, s = sums(R, A, Ab, ["anc", "surf"], vals)
    bs, _ = asof(R.assign(_k="1"), ["_k", "surf"], vals)
    b3 = np.where(bs[:, 1] > 30, bs[:, 0] / np.maximum(bs[:, 1], 1), 1600.0)[L["row"].to_numpy()]
    br = np.where(bs[:, 3] > 30, bs[:, 2] / np.maximum(bs[:, 3], 1), 1600.0)[L["row"].to_numpy()]
    m3 = (s[:, 0] + K * b3) / (s[:, 1] + K)
    mr = (s[:, 2] + K * br) / (s[:, 3] + K)
    AA = A if Ab is None else Ab
    J = slot_join(L, m3, AA, D_SLOTS)                        # Dosage と同じく、重なった祖先は出るたびに数える（critic）
    J["g"] = slot_join(L, m3 - mr, AA, D_SLOTS)["v"].to_numpy()
    w = J["gen"].map(D_W).to_numpy()
    d = pd.DataFrame({"row": J["row"], "w": w, "wm": w * J["v"], "wg": w * J["g"]})
    g = d.groupby("row")[["w", "wm", "wg"]].sum()
    out = pd.DataFrame(index=np.arange(len(X)))
    out["pr_E"] = (g["wm"] / g["w"]).reindex(out.index).to_numpy()
    out["pr_G"] = (g["wg"] / g["w"]).reindex(out.index).to_numpy()
    out["pr_dz"] = (X["distance"].astype(float).to_numpy() - out["pr_E"].to_numpy()) / 400.0
    out["pr_adz"] = np.abs(out["pr_dz"])
    return out


def col_X(X: pd.DataFrame, A: pd.DataFrame, female: bool, Ab=None) -> pd.DataFrame:
    """父の母の父の子孫の 3 着内率の偏差。X は牝の馬だけ・XP は牡・騸の馬だけに値を入れる。"""
    R = runs(X)
    L, s = sums(R, A, Ab, ["anc"], ["_t3", "_one"])
    bs, _ = asof(R.assign(_k="1"), ["_k"], ["_t3", "_one"])
    base = np.where(bs[:, 1] > 30, bs[:, 0] / np.maximum(bs[:, 1], 1), 0.2)[L["row"].to_numpy()]
    rate = (s[:, 0] + K * base) / (s[:, 1] + K) - base
    J = slot_join(L, rate, A if Ab is None else Ab, [X_SLOT]).drop_duplicates("row")
    v = J.set_index("row")["v"].reindex(np.arange(len(X))).to_numpy()
    is_f = X["sex"].astype(str).eq("牝").to_numpy()
    return pd.DataFrame({"pr_x": np.where(is_f == female, v, np.nan)})


def col_RF(A: pd.DataFrame) -> pd.DataFrame:
    F = A[(A["pos"] % 2 == 1)].copy()                       # 牝の祖先
    F["cg"], F["cp"] = F["gen"] - 1, F["pos"] // 2           # その牝馬を通る子の位置
    C = A.rename(columns={"gen": "cg", "pos": "cp", "anc": "child_id"})
    F = F.merge(C, on=["horse_id", "cg", "cp"], how="left")
    F.loc[F["gen"] == 1, "child_id"] = F.loc[F["gen"] == 1, "horse_id"]
    F = F[F.duplicated(["horse_id", "anc"], keep=False)]
    rows = []
    for (h, a), d in F.groupby(["horse_id", "anc"]):
        gs, ch = d["gen"].to_numpy(), d["child_id"].to_numpy()
        best = min((gs[i] + gs[j] for i in range(len(d)) for j in range(i + 1, len(d)) if ch[i] != ch[j]), default=None)
        if best is not None:                                 # 別々の子の組だけで代の和の最小を取る（critic）
            rows.append((h, best))
    R = pd.DataFrame(rows, columns=["horse_id", "g"]).groupby("horse_id")["g"].min()
    out = pd.DataFrame({"horse_id": A["horse_id"].unique()})
    out["pr_rf_g"] = out["horse_id"].map(R).fillna(11).astype(float)
    out["pr_rf"] = (out["pr_rf_g"] < 11).astype(float)
    return out


def main() -> None:
    kind = sys.argv[sys.argv.index("--kind") + 1]
    assert kind in ("D", "RF", "X", "XP"), kind
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"public_rules_{kind}" + ("_null" if null_seed is not None else ""),
                                   "公開の手法を規則で作った列の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    ped = data.load_ped()
    A = parse(ped[ped["horse_id"].isin(X["horse_id"].unique())])
    Ab = None
    if null_seed is not None:
        # 帰無（2 号の「借り手の血統」）: 祖先の集計は本物の血統表のまま作り、馬 h には同じ生年の別の馬 h' の祖先の並びを
        # 馬の単位で貸して、その日より前の集計を引く。RF は借りた血統表から印を作る
        rng = np.random.default_rng(null_seed)
        H = pd.Series(A["horse_id"].unique())
        hb = H.str[:4].to_numpy()
        tgt = H.to_numpy().copy()
        for b in np.unique(hb):
            ii = np.flatnonzero(hb == b)
            tgt[ii] = H.to_numpy()[rng.permutation(ii)]
        Ab = A.assign(horse_id=A["horse_id"].map(dict(zip(tgt, H.to_numpy()))))
    if kind == "D":
        C = col_D(X, A, Ab)
    elif kind in ("X", "XP"):
        C = col_X(X, A, female=(kind == "X"), Ab=Ab)
    else:
        C = X[["horse_id"]].merge(col_RF(A if Ab is None else Ab), on="horse_id", how="left").drop(columns="horse_id")
    new = list(C.columns)
    for c in new:
        X[c] = C[c].to_numpy()
    print("coverage", {c: round(float(pd.Series(X[c]).notna().mean()), 3) for c in new},
          "mean", {c: round(float(np.nanmean(X[c].to_numpy(float))), 4) for c in new}, flush=True)
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    # RF の比べだけは近交の係数（ib_）を基準に入れ、近交一般の効きを除く（1 号）
    use = [c for c in ablate.col_sets(X)["pure"] if kind == "RF" or not c.startswith("ib_")] + new
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    out = X.loc[te, ["race_id", "horse_id"]].copy()
    out["p_TB"] = p[te]
    path = data.OUT / f"public_{kind}_wf{'' if null_seed is None else f'_null{null_seed}'}.parquet"
    out.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
