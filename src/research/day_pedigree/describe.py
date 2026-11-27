"""記述（合否に使わない・探す 70% の側だけ）。画面の形の表と、族ごとの記述の α。

    KEIBA_SESSION=day-pedigree OMP_NUM_THREADS=4 nice -n 19 python -m src.research.day_pedigree.describe

- 族: A（6 次元）・S 特殊（芝ダごと 10 頭・血量 × log(1/p)）・S 規則（δ = log 2 の 5 頭）・C（共通の血の濃さ 1 本）・M（ss の 5 札）・B（母父の系統の 5 札）。
  S・C の祖先と M・B の札は features_band.json（着順を見る前に commit した一覧）から読む。
- 時点は 2 本: (a) 表示（発走の時点で確定しているレース全部）、(b) 検定（T-60）。
- 出す表:
  1. k ごとの、使った前のレースの数・事後の 80% 区間の幅（特徴の平均）・縮めの係数（事前の分散に対する事後の分散の比）。
  2. 族ごと・k の帯ごとの記述の α（T-60・確定オッズ）。試した数に入れ、合否には使わない。
  3. 例の日（探す側の日を 3 つ・無作為）の、レースごとの特殊な祖先の事後の平均と 80% 区間（画面の形）。
- 5 着内・確定オッズの α は市場の上の情報の記述。回収率・買い目は出さない。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.day_pedigree import check as ck
from src.research.day_pedigree import data as dd
from src.research.day_pedigree import features as fe
from src.research.day_pedigree import posterior as po
from src.research.pedigree_lab import data as pl

Z80 = 1.2816
K_BANDS = {"2〜3": (2, 3), "4〜5": (4, 5), "6〜": (6, 99)}


def family_cols(R: pd.DataFrame, band: dict) -> dict[str, dict[str, np.ndarray]]:
    """芝ダごとの族の列（行 × 特徴）。祖先の札は芝ダで違うので、芝ダごとに作る。"""
    ped = pl.load_ped()
    ped = ped[ped["horse_id"].isin(R["horse_id"].unique())]
    L = fe.parse_anc(ped)
    P = fe.prevalence(R, L)
    LB = fe.label_table(ped).set_index("horse_id")
    out: dict[str, dict[str, np.ndarray]] = {}
    for s in ("T", "D"):
        b = band["S"][s]
        fams = {}
        for nm, lst in (("S_special", b["special"]), ("S_rule", b["chosen"])):
            ids = [r["anc"] for r in lst]
            M = np.zeros((len(R), len(ids)))
            J = R[["horse_id", "surf", "year"]].reset_index().merge(L[L["anc"].isin(ids)], on="horse_id")
            J = J.merge(P, on=["anc", "surf", "year"], how="left")
            # 2016 年の行は前の年の登場率が無い（データは 2016 年から）。その祖先の最初の年の値（2016 年の出走馬の割合）で埋める（critic の 3）。
            # 着順は使わない。2016 年は燃やしの年で、事前（μ・τ²）にだけ入る
            first = P.sort_values("year").groupby(["anc", "surf"])["p"].first().rename("p0").reset_index()
            J = J.merge(first, on=["anc", "surf"], how="left")
            J["p"] = J["p"].fillna(J["p0"])
            J["v"] = J["bw"] * np.log(1.0 / J["p"].clip(lower=1e-3).fillna(1e-3))
            for f, a in enumerate(ids):
                jj = J[J["anc"] == a]
                M[jj["index"].to_numpy(), f] = jj["v"].to_numpy()
            fams[nm] = M
        com = [r["anc"] for r in b["common_by_count"]]
        J = R[["horse_id"]].reset_index().merge(L[L["anc"].isin(com)], on="horse_id")
        C = np.zeros((len(R), 1))
        np.add.at(C[:, 0], J["index"].to_numpy(), J["bw"].to_numpy())
        fams["C"] = C
        for nm, key in (("M", "M_ss"), ("B", "B_ds4")):
            ids = [r["anc"] for r in band[key][s]["chosen"]]
            lab = R["horse_id"].map(LB[key]).fillna("").to_numpy()
            fams[nm] = np.stack([(lab == a).astype(float) for a in ids], axis=1)
        fams["A"] = R[ck.A_COLS].to_numpy(float)
        out[s] = fams
    return out


def run_family(R, Rc, X, phi, part_rows, lag):
    rid = R["rid"].to_numpy()
    U, I, Xc = po.race_stats(X, R["y"].to_numpy(), R["pt"].to_numpy(), rid)
    pri = po.priors(Rc, U, I, phi, range(2017, 2025))
    mean, var, nprev = po.day_posterior(Rc, U, I, phi, pri, lag)
    mu = ck.mu_by_race(Rc, pri, X.shape[1])
    tau2 = np.full_like(mu, np.nan)
    for i, (s, y) in enumerate(zip(Rc["surf"], Rc["year"])):
        p = pri.get((s, y))
        if p is not None:
            tau2[i] = p["tau2"]
    return {"mean": mean, "var": var, "nprev": nprev, "mu": mu, "tau2": tau2, "Xc": Xc, "pri": pri}


def main() -> None:
    dd.open_token("select", "describe_v1", "記述（探す 70% だけ・族ごとの α と画面の形の表。合否に使わない）")
    band = json.loads((dd.OUT / "features_band.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261129)
    R, Rc, _ = ck.build(0)
    _, _, _, phi, _, _ = ck.prepare(R, Rc, rng)
    cols = family_cols(R, band)
    y, pt, cl = R["y"].to_numpy(), R["pt"].to_numpy(), R["meeting"].to_numpy()
    # φ は A 族の 6 次元から出したものを全部の族に使う（0/1 の札・祖先の列の過分散は見ていない。記述の注。critic の 8）
    res: dict = {"phi": phi, "phi_note": "A 族の φ を全部の族に使い回した", "width": {}, "alpha": {}, "examples": {}}
    rk = Rc["k"].to_numpy()
    for s in ("T", "D"):
        ms = (R["surf"] == s).to_numpy()
        msc = (Rc["surf"] == s).to_numpy()
        for fam, X in cols[s].items():
            X = np.where(ms[:, None], X, 0.0)          # 他の芝ダの行は 0（日は芝ダで分かれるので、事後に入らない）
            for lag, tag in ((ck.SHOW_LAG, "a_show"), (ck.TEST_LAG, "b_t60")):
                o = run_family(R, Rc, X, phi, None, lag)
                ok = msc & (Rc["year"] >= 2017).to_numpy() & np.isfinite(o["var"]).all(axis=1)
                tab = []
                for k in range(1, 9):
                    m = ok & (rk == k)
                    if m.sum() < 30:
                        continue
                    w = 2 * Z80 * np.sqrt(o["var"][m]).mean(axis=1)
                    # 比は τ² > 0 の次元だけで平均する（τ² = 0 の次元は動けず比がいつも 1 で、狭まりを薄めるため。critic の 2）
                    live = o["tau2"][m] > 1e-9
                    ratio = np.where(live, o["var"][m] / np.maximum(o["tau2"][m], 1e-12), np.nan)
                    shrink = np.nanmean(np.where(live.any(axis=1)[:, None], ratio, 1.0), axis=1)
                    tab.append({"k": k, "races": int(m.sum()), "prev_mean": float(o["nprev"][m].mean()),
                                "width80": float(w.mean()), "var_ratio": float(np.mean(shrink)),
                                "live_dims": float(live.sum(axis=1).mean())})
                res["width"][f"{s}_{fam}_{tag}"] = tab
                if tag == "b_t60":
                    dev = o["mean"] - o["mu"]
                    sc = ck.scores(R, dev, o["Xc"])
                    sel = ms & (R["year"] >= 2017).to_numpy() & (R["k"] >= 2).to_numpy()
                    al = {"all": po.offset_alpha(y[sel], pt[sel], sc[sel], cl[sel])}
                    for bn, (a, b) in K_BANDS.items():
                        m = sel & R["k"].between(a, b).to_numpy()
                        al[bn] = po.offset_alpha(y[m], pt[m], sc[m], cl[m])
                    al["tau2_mean"] = float(np.nanmean(o["tau2"][msc]))
                    res["alpha"][f"{s}_{fam}"] = al
                    print(s, fam, json.dumps({k: (round(v["alpha"], 3), round(v["se"], 3)) if isinstance(v, dict) else v
                                              for k, v in al.items()}, ensure_ascii=False), flush=True)
                if fam == "S_special" and tag == "a_show":
                    days = Rc.loc[ok & (Rc["k_last"] >= 6).to_numpy(), "day"].unique()
                    pick = rng.choice(days, size=min(3, len(days)), replace=False)
                    names = [r["name"] for r in band["S"][s]["special"]]
                    ex = []
                    for d in pick:
                        ii = np.flatnonzero((Rc["day"] == d).to_numpy())
                        ii = ii[np.argsort(rk[ii])]
                        rows = []
                        for i in ii:
                            sd = np.sqrt(o["var"][i])
                            rows.append({"k": int(rk[i]), "prev": int(o["nprev"][i]),
                                         "mean": (o["mean"][i] - o["mu"][i]).round(4).tolist(),
                                         "lo": (o["mean"][i] - o["mu"][i] - Z80 * sd).round(4).tolist(),
                                         "hi": (o["mean"][i] - o["mu"][i] + Z80 * sd).round(4).tolist()})
                        ex.append({"day": str(d), "names": names, "races": rows})
                    res["examples"][s] = ex
    path = dd.OUT / "describe.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("書いた", path)
    dd.finish_all(str(path))


if __name__ == "__main__":
    main()
