"""早熟度合い（ユーザーの言葉どおりの版。log.md「早熟度合い（ユーザーの言葉どおりの版）」。開発の行・探す側・未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.maturity_curve [--boot 200]

- 早熟の指数（祖先ごと・as-of・自分を引く）: 子孫の Σ(2 歳の勝ち − r_c × 2〜3 歳の勝ち) ÷ (Σ 2〜3 歳の勝ち + 20)。
  r_c は世代 c の全体の比（Σ 2 歳の勝ち ÷ Σ 2〜3 歳の勝ち）。世代 c はレースの年 Y ≥ c+4 で使う（c+3 年の年末までに窓が終わる）。
- 晩成の指数: 3 歳と 4〜5 歳の両方を走った子孫の (4〜5 歳の −分位の平均 − 3 歳の −分位の平均) から世代の平均を引き、Σ ÷ (頭数 + 20)。
  世代 c は Y ≥ c+6 で使う（c+5 年の年末までに窓が終わる）。生き残りの偏りがある。
- 父の指数は父ごと、牝系は母の側の牡の祖先（2〜5 代）ごとに作り、血量で重み付けて平均する（aptitude.maternal_long の表。凍結のファイルは読むだけ）。
- 主の検定: [lp, 指数, 指数 × レースの年齢] の「指数 × 年齢」の傾き（父の早熟・牝系の早熟の 2 本・同時の区間）。年齢はレースの出走馬の月齢（生まれた年の 1/1 から）の平均を中心化し、年の単位にした値。
- 表: 指数の三分位 × 時期の「勝ち ÷ Σ(1/頭数)」と「勝ち ÷ Σq」。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import data
from src.research.pedigree_lab.aptitude_period import attach_part
from src.research.pedigree_lab.explore2 import fit_boot
from src.research.pedigree_lab.explore5 import sim_ci

K_E, K_L = 20.0, 20.0
YEARS = list(range(2016, 2025))


def horse_parts(F: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの世代・早熟の材料（2 歳の勝ち・2〜3 歳の勝ち）・晩成の材料（3 歳と 4〜5 歳の −分位の平均）。"""
    F = F.assign(by=pd.to_numeric(F["horse_id"].str[:4], errors="coerce"), y=F["date"].dt.year)
    F["age"] = F["y"] - F["by"]
    g = -F["fpct"]
    H = pd.DataFrame({
        "w2": F[F["age"] == 2].groupby("horse_id")["win"].sum(),
        "w23": F[F["age"].isin([2, 3])].groupby("horse_id")["win"].sum(),
        "m3": g[F["age"] == 3].groupby(F.loc[F["age"] == 3, "horse_id"]).mean(),
        "m45": g[F["age"].isin([4, 5])].groupby(F.loc[F["age"].isin([4, 5]), "horse_id"]).mean(),
    })
    H = H.reset_index().rename(columns={"index": "horse_id"})
    H["c"] = pd.to_numeric(H["horse_id"].str[:4], errors="coerce")
    H[["w2", "w23"]] = H[["w2", "w23"]].fillna(0)
    # 2 歳が手元の期間（2016 年〜）に入る世代だけ（c ≥ 2014）。2 歳〜3 歳の窓が 2024 年末までに終わる世代（c ≤ 2021）
    ok_e = (H["c"] >= 2014) & (H["c"] <= 2021)
    r_c = H[ok_e].groupby("c")["w2"].sum() / H[ok_e].groupby("c")["w23"].sum()
    H["e_num"] = np.where(ok_e, H["w2"] - H["c"].map(r_c).fillna(0) * H["w23"], 0.0)
    H["e_den"] = np.where(ok_e, H["w23"], 0.0)
    H["e_avail"] = H["c"] + 4                      # この年からの行で使える
    both = H["m3"].notna() & H["m45"].notna() & (H["c"] >= 2013) & (H["c"] <= 2019)
    d = (H["m45"] - H["m3"]).where(both)
    H["l_num"] = (d - d.groupby(H["c"]).transform("mean")).fillna(0.0)
    H["l_den"] = both.astype(float)
    H["l_avail"] = H["c"] + 6
    return H


def cum_index(P: pd.DataFrame, key: str) -> pd.DataFrame:
    """鍵 × 年 Y の累積（その年に使える世代までの和）。P は [key, e_num, e_den, e_avail, l_num, l_den, l_avail]。"""
    out = []
    for pre in ("e", "l"):
        g = P.groupby([key, f"{pre}_avail"])[[f"{pre}_num", f"{pre}_den"]].sum().reset_index().rename(columns={f"{pre}_avail": "Y"})
        piv_n = g.pivot(index=key, columns="Y", values=f"{pre}_num").reindex(columns=YEARS, fill_value=0).fillna(0)
        piv_d = g.pivot(index=key, columns="Y", values=f"{pre}_den").reindex(columns=YEARS, fill_value=0).fillna(0)
        # avail が 2016 より前の世代は 2016 列に寄せる
        early_n = g[g["Y"] < YEARS[0]].groupby(key)[f"{pre}_num"].sum()
        early_d = g[g["Y"] < YEARS[0]].groupby(key)[f"{pre}_den"].sum()
        piv_n[YEARS[0]] = piv_n[YEARS[0]].add(early_n.reindex(piv_n.index).fillna(0), fill_value=0)
        piv_d[YEARS[0]] = piv_d[YEARS[0]].add(early_d.reindex(piv_d.index).fillna(0), fill_value=0)
        cn, cd = piv_n.cumsum(axis=1), piv_d.cumsum(axis=1)
        s = cn.stack().rename(f"{pre}_cn").to_frame().join(cd.stack().rename(f"{pre}_cd"))
        out.append(s)
    T = out[0].join(out[1], how="outer").fillna(0).reset_index()
    T.columns = [key, "Y", "e_cn", "e_cd", "l_cn", "l_cd"]
    return T


def row_index(R: pd.DataFrame, key: str, T: pd.DataFrame, H: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """行（key・Y・horse_id）ごとの指数。自分の世代が使える年なら、自分の分を引く。"""
    X = R[[key, "Y", "horse_id"]].merge(T, on=[key, "Y"], how="left").merge(
        H[["horse_id", "e_num", "e_den", "e_avail", "l_num", "l_den", "l_avail"]], on="horse_id", how="left")
    X = X.fillna({"e_cn": 0, "e_cd": 0, "l_cn": 0, "l_cd": 0, "e_num": 0, "e_den": 0, "l_num": 0, "l_den": 0, "e_avail": 9999, "l_avail": 9999})
    se = (X["Y"] >= X["e_avail"]).to_numpy()
    sl = (X["Y"] >= X["l_avail"]).to_numpy()
    en = X["e_cn"].to_numpy() - np.where(se, X["e_num"], 0)
    ed = X["e_cd"].to_numpy() - np.where(se, X["e_den"], 0)
    ln = X["l_cn"].to_numpy() - np.where(sl, X["l_num"], 0)
    ld = X["l_cd"].to_numpy() - np.where(sl, X["l_den"], 0)
    return en / (ed + K_E), ln / (ld + K_L), ed, ld


def zc(S: pd.DataFrame, v: np.ndarray) -> np.ndarray:
    rid = S["race_id"].to_numpy()
    c = v - pd.Series(v).groupby(rid).transform("mean").to_numpy()
    sd = c.std()
    return c / (sd if sd > 0 else 1.0)


def period(m: np.ndarray) -> np.ndarray:
    lab = np.full(len(m), "", dtype=object)
    # agem = (年 − 生まれた年) × 12 + 月 − 1。2 歳の 6 月が 29（critic の指摘で 12 か月のずれを直した）
    for lo, hi, nm in ((29, 32, "2 歳 6〜9 月"), (33, 35, "2 歳 10〜12 月"), (36, 40, "3 歳 1〜5 月"), (41, 47, "3 歳 6〜12 月"),
                       (48, 59, "4 歳"), (60, 999, "5 歳以上")):
        lab[(m >= lo) & (m <= hi)] = nm
    return lab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=200)
    a = p.parse_args()
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "maturity_curve",
                                   "早熟度合い（ユーザーの言葉どおりの版・父と牝系 5 代・時期の表。尤度だけ・記述）"))
    F = ap_.base_frame()
    H = horse_parts(F)
    F["Y"] = F["date"].dt.year
    # 父
    Ps = H.merge(F.drop_duplicates("horse_id")[["horse_id", "sire"]], on="horse_id", how="left")
    Ts = cum_index(Ps, "sire")
    F["e_sire"], F["l_sire"], _, _ = row_index(F, "sire", Ts, H)
    # 牝系: 母の側の牡の祖先（2〜5 代）。子孫の材料を祖先ごとに集め、行には血量で重み付けて平均する
    L = ap_.maternal_long(F, drop_common=False)
    Anc = L.drop_duplicates(["horse_id", "anc"])[["horse_id", "anc", "bw"]]
    Pa = Anc.merge(H, on="horse_id", how="inner")
    Ta = cum_index(Pa, "anc")
    LR = L[["row", "anc", "bw", "horse_id"]].assign(Y=F["Y"].to_numpy()[L["row"].to_numpy()])
    ea, la, eda, lda = row_index(LR, "anc", Ta, H)
    # 材料のある祖先（その年までに子孫の窓が終わった勝ち・頭数がある）だけで重みの和を取る（critic の指摘）
    bw = LR["bw"].to_numpy()
    we_w, wl_w = bw * (eda > 0), bw * (lda > 0)
    agg = pd.DataFrame({"row": LR["row"], "we": we_w * ea, "wl": wl_w * la, "wE": we_w, "wL": wl_w}).groupby("row")[["we", "wl", "wE", "wL"]].sum()
    n = np.arange(len(F))
    with np.errstate(invalid="ignore", divide="ignore"):
        F["e_mat"] = (agg["we"] / agg["wE"]).reindex(n).fillna(0).to_numpy()
        F["l_mat"] = (agg["wl"] / agg["wL"]).reindex(n).fillna(0).to_numpy()
    by = pd.to_numeric(F["horse_id"].str[:4], errors="coerce")
    F["agem"] = (F["Y"] - by) * 12 + F["date"].dt.month - 1
    S = attach_part(F, "dev").sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    S = S[S["Y"] >= 2019].reset_index(drop=True)            # 早熟は 2018 年から、晩成は 2019 年から付く。両方そろう 2019 年からにする（critic の指摘）
    rid = S["race_id"].to_numpy()
    ager = pd.Series(S["agem"].to_numpy(float)).groupby(rid).transform("mean").to_numpy()
    agec = (ager - ager.mean()) / 12.0
    Z = {k: zc(S, S[k].to_numpy()) for k in ("e_sire", "l_sire", "e_mat", "l_mat")}
    res = {"rows": int(len(S)), "sd_raw": {k: round(float(S[k].std()), 5) for k in Z},
           "corr_e_l_sire": round(float(np.corrcoef(Z["e_sire"], Z["l_sire"])[0, 1]), 3),
           "corr_e_sire_mat": round(float(np.corrcoef(Z["e_sire"], Z["e_mat"])[0, 1]), 3)}
    for market in (False, True):
        base = [S["lp"].to_numpy(), S["lq"].to_numpy()] if market else [S["lp"].to_numpy()]
        cols = [Z["e_sire"], Z["e_sire"] * agec, Z["e_mat"], Z["e_mat"] * agec, Z["l_sire"], Z["l_sire"] * agec, Z["l_mat"], Z["l_mat"] * agec]
        names = ["e_sire", "e_sire_x_age", "e_mat", "e_mat_x_age", "l_sire", "l_sire_x_age", "l_mat", "l_mat_x_age"]
        b, bs = fit_boot(S, np.c_[tuple(base + cols)], a.boot)
        nb = len(base)
        c_main, main = sim_ci(b, bs, {"e_sire_x_age": nb + 1, "e_mat_x_age": nb + 3})
        _, rest = sim_ci(b, bs, {k: nb + i for i, k in enumerate(names)})
        tag = "mkt" if market else "nomkt"
        res[tag] = {"main_c": round(c_main, 3), "main": {k: {"gamma": v["gamma"], "sim_ci": v["sim_ci"]} for k, v in main.items()},
                    "all": {k: {"gamma": v["gamma"], "ci": v["ci"]} for k, v in rest.items()}}
        print(tag, json.dumps(res[tag], ensure_ascii=False), flush=True)
    # 表: 三分位 × 時期
    per = period(S["agem"].to_numpy())
    inv_n = 1.0 / S.groupby("race_id")["horse_id"].transform("size").to_numpy()
    tables = {}
    for k in ("e_sire", "e_mat", "l_sire", "l_mat"):
        v = S[k].to_numpy()
        has = v != 0
        # 三分位は年ごとに切る（年が進むと縮みが弱まり、後の年ほど端に入りやすいため。critic の指摘）
        terc = np.full(len(v), "", dtype=object)
        yy = S["Y"].to_numpy()
        for y in np.unique(yy):
            m = (yy == y) & has
            if m.sum() < 100:
                continue
            q1, q2 = np.quantile(v[m], [1 / 3, 2 / 3])
            terc[m] = np.where(v[m] <= q1, "晩成寄り（下 1/3）", np.where(v[m] >= q2, "早熟寄り（上 1/3）", "普通（中 1/3）"))
        T = pd.DataFrame({"t": terc, "p": per, "w": S["win"].to_numpy(float), "u": inv_n, "q": S["q"].to_numpy()})
        T = T[(T["t"] != "") & (T["p"] != "")]
        g = T.groupby(["t", "p"]).agg(rows=("w", "size"), wins=("w", "sum"), u=("u", "sum"), q=("q", "sum"))
        g["ae_uniform"] = (g["wins"] / g["u"]).round(3)
        g["ae_market"] = (g["wins"] / g["q"]).round(3)
        tables[k] = {f"{t}|{pp}": {"rows": int(r.rows), "wins": int(r.wins), "ae_uniform": float(r.ae_uniform), "ae_market": float(r.ae_market)}
                     for (t, pp), r in g.iterrows()}
    res["tables"] = tables
    # 父の早熟の指数の 2 期の持続（終わった世代で分ける: 2014〜16 世代 と 2017〜20 世代）
    Hs = Ps[(Ps["e_den"] > 0)]
    A1 = Hs[Hs["c"] <= 2016].groupby("sire")[["e_num", "e_den"]].sum()
    A2 = Hs[(Hs["c"] >= 2017) & (Hs["c"] <= 2020)].groupby("sire")[["e_num", "e_den"]].sum()
    J = A1.join(A2, lsuffix="1", rsuffix="2", how="inner")
    J = J[(J["e_den1"] >= 30) & (J["e_den2"] >= 30)]
    res["persist_e_sire"] = {"sires": int(len(J)), "corr": round(float((J["e_num1"] / J["e_den1"]).corr(J["e_num2"] / J["e_den2"])), 3)}
    path = data.OUT / "maturity_curve.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "tables"}, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
