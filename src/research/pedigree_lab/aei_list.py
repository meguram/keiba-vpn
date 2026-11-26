"""平均獲得賞金の価値の一覧・2 期の持続・回収率（記述。log.md「平均獲得賞金で見た…」の「作りの直し」）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.aei_list

- 行: 2018〜24 年の開発の行（split 0・1）。b は aei_cols.parquet（年 × 年齢の区分の平均で割り、前の年の 99.5% 点で切った値）。
- 一覧（1 頭 1 票）: 馬ごとの b の平均（全馬の平均で割って 1 を平均の馬にする）を、父・母父・父 × 母父ごとに頭数で平均し、擬似件数 10 頭で縮める（父・母父は 1 へ、組は見込みへ）。
  見込みは足し算（父の値 + 母父の値 − 1）。下限: 父・母父は 30 頭以上かつ 300 走以上、組は 10 頭以上かつ 50 走以上。
  区間は馬の単位のブートストラップ（1,000 回・選んだ後に付けるので甘い）。勝者の呪いは、馬を id の偶奇で分け、
  偶数の馬で上位 30 を選び、奇数の馬で測った値も並べる。名前は手元の 10 代の血統表から引く。
- 持続: 2014〜16 年生まれと 2017〜20 年生まれ（馬が重ならない）で、父の値・組の残差の Spearman。帰無は 200 回
  （父: 父の札を同じ生年の中で馬の単位で入れ替え。組: 母父の札を同じ生年 × 父の中で馬ごとに 1 回入れ替え、両期で同じ札）。
- 回収率（記述・確定オッズの単勝）: 組の残差（as-of の av_nick_add）が、前の年の分布の 90% 点より上の走り。
  基準線は同じ人気の全走りの回収率。比・上位 1% の払い戻しを除いた値・50 倍で切った値・的中の件数を出す。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research.pedigree_lab import data

PED10 = Path("/home/jovyan/work/keiba-vpn/data/local/horse_pedigree_10gen")
KH = 10.0


def names(ids: set[str], horses: pd.DataFrame) -> dict[str, str]:
    """父・母父の id → 名前。その id を父・母父に持つ馬の 10 代の表を 1 頭ずつ読む。"""
    out: dict[str, str] = {}
    for col, pos in (("sire", (1, 0)), ("ds", (2, 2))):
        sub = horses[horses[col].isin(ids - set(out))].drop_duplicates(col)
        for h, a in zip(sub["horse_id"], sub[col]):
            f = PED10 / h[:4] / f"{h}.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text()).get("ancestors", []):
                if (int(r.get("generation", 0)), int(r.get("position", 0))) == pos:
                    out[a] = r.get("name", a)
                    break
    return out


def perm_within(groups: np.ndarray, values: np.ndarray, rng) -> np.ndarray:
    """同じ群の中で値を入れ替える（ベクトルで）。"""
    g = pd.factorize(groups)[0]
    a = np.lexsort((np.arange(len(g)), g))
    b = np.lexsort((rng.random(len(g)), g))
    out = values.copy()
    out[a] = values[b]
    return out


def horse_table(T: pd.DataFrame) -> pd.DataFrame:
    H = T.groupby("horse_id").agg(b=("b", "mean"), n=("b", "size"), sire=("sire", "first"), ds=("ds", "first"), by=("by", "first")).reset_index()
    # 馬ごとの平均の全体の平均（約 0.65）で割り、1 が平均の馬になるようにする（縮める先 1・見込み vs + vb − 1 の中心を合わせる。critic）
    H["b"] = H["b"] / H["b"].mean()
    return H


def value(H: pd.DataFrame, key, prior=None) -> pd.DataFrame:
    g = H.groupby(key).agg(sb=("b", "sum"), nh=("b", "size"), nr=("n", "sum"))
    pr = 1.0 if prior is None else prior.reindex(g.index).to_numpy()
    g["v"] = (g["sb"] + KH * pr) / (g["nh"] + KH)
    g["raw"] = g["sb"] / g["nh"]
    g["prior"] = pr
    return g


def nick_table(H: pd.DataFrame, vs: pd.Series, vb: pd.Series) -> pd.DataFrame:
    idx = H.groupby(["sire", "ds"]).size().index
    pr = pd.Series(vs.reindex(idx.get_level_values(0)).fillna(1).to_numpy() + vb.reindex(idx.get_level_values(1)).fillna(1).to_numpy() - 1, index=idx)
    g = value(H, ["sire", "ds"], pr)
    g["resid"] = g["v"] - g["prior"]
    return g


def boot_ci(H: pd.DataFrame, key, keys_sel, nboot=1000, seed=3, prior=None):
    """選んだ鍵の値の、馬の単位のブートストラップの 95% 区間。"""
    rng = np.random.default_rng(seed)
    sub = H[pd.MultiIndex.from_frame(H[key]).isin(keys_sel) if isinstance(key, list) else H[key].isin(keys_sel)].reset_index(drop=True)
    grp = sub.groupby(key).indices
    res = {k: [] for k in grp}
    b = sub["b"].to_numpy()
    for _ in range(nboot):
        for k, ii in grp.items():
            jj = rng.choice(ii, len(ii))
            pr = 1.0 if prior is None else prior.get(k, 1.0)
            res[k].append((b[jj].sum() + KH * pr) / (len(jj) + KH) - (0 if prior is None else pr))
    return {k: np.percentile(v, [2.5, 97.5]).round(3).tolist() for k, v in res.items()}


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "aei_list", "平均獲得賞金の価値の一覧・持続・回収率（記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "sire", "ds", "win", "odds", "popularity"])
    C = pd.read_parquet(data.OUT / "aei_cols.parquet", columns=["race_id", "horse_id", "b", "av_nick_add"])
    T = X.merge(C, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & (T["date"] >= "2018-01-01") & (T["date"] <= data.DEV_END) & T["b"].notna()].reset_index(drop=True)
    T = T[T["sire"].fillna("").ne("") & T["ds"].fillna("").ne("")]
    T["by"] = T["horse_id"].str[:4]
    H = horse_table(T)
    res: dict = {"rows": int(len(T)), "horses": int(len(H))}

    # 一覧
    vs = value(H, "sire"); vb = value(H, "ds")
    nk = nick_table(H, vs["v"], vb["v"])
    sel_s = vs[(vs["nh"] >= 30) & (vs["nr"] >= 300)].sort_values("v", ascending=False)
    sel_b = vb[(vb["nh"] >= 30) & (vb["nr"] >= 300)].sort_values("v", ascending=False)
    sel_n = nk[(nk["nh"] >= 10) & (nk["nr"] >= 50)].sort_values("resid", ascending=False)
    ev = H[H["horse_id"].str[-1].astype(int) % 2 == 0]; od = H[H["horse_id"].str[-1].astype(int) % 2 == 1]
    vs_e, vs_o = value(ev, "sire"), value(od, "sire")
    vb_e, vb_o = value(ev, "ds"), value(od, "ds")
    nk_e = nick_table(ev, vs_e["v"], vb_e["v"]); nk_o = nick_table(od, vs_o["v"], vb_o["v"])
    top_s, top_b, top_n = sel_s.head(30), sel_b.head(30), sel_n.head(30)
    res["組の残差の偏り"] = {"下限を満たす組の残差の平均": round(float(sel_n["resid"].mean()), 3), "正の割合": round(float((sel_n["resid"] > 0).mean()), 3),
                       "奇数の馬の下限の半分を満たす組の残差の平均": round(float(nk_o[(nk_o["nh"] >= 5) & (nk_o["nr"] >= 25)]["resid"].mean()), 3)}
    ids = set(top_s.index) | set(top_b.index) | {a for a, _ in top_n.index} | {b for _, b in top_n.index}
    nm = names(ids, T[["horse_id", "sire", "ds"]].drop_duplicates("horse_id"))
    ci_s = boot_ci(H, "sire", list(top_s.index)); ci_b = boot_ci(H, "ds", list(top_b.index))
    ci_n = boot_ci(H, ["sire", "ds"], list(top_n.index), prior=top_n["prior"].to_dict())

    def split_half(sel, ve, vo, col, k):
        # 選ぶ半分・測る半分とも、下限の半分を満たす鍵だけ（critic）
        ok_o = vo[(vo["nh"] >= k[0] / 2) & (vo["nr"] >= k[1] / 2)].index
        e = ve[(ve["nh"] >= k[0] / 2) & (ve["nr"] >= k[1] / 2)]
        e = e[e.index.isin(ok_o)].sort_values(col, ascending=False).head(30)
        return {"選んだ半分の平均": round(float(e[col].mean()), 3), "測る半分の平均": round(float(vo.loc[e.index, col].mean()), 3), "数": int(len(e))}
    res["一覧"] = {
        "種牡馬": {"候補の数": int(len(sel_s)), "上位": [[nm.get(k, k), int(r.nh), int(r.nr), round(r.v, 3), ci_s[k]] for k, r in top_s.iterrows()],
                 "勝者の呪い": split_half(sel_s, vs_e, vs_o, "v", (30, 300))},
        "母父": {"候補の数": int(len(sel_b)), "上位": [[nm.get(k, k), int(r.nh), int(r.nr), round(r.v, 3), ci_b[k]] for k, r in top_b.iterrows()],
               "勝者の呪い": split_half(sel_b, vb_e, vb_o, "v", (30, 300))},
        "組（父 × 母父・残差）": {"候補の数": int(len(sel_n)),
                          "上位": [[nm.get(a, a) + " × " + nm.get(b, b), int(r.nh), int(r.nr), round(r.resid, 3), round(r.prior, 3), ci_n[(a, b)]] for (a, b), r in top_n.iterrows()],
                          "勝者の呪い": split_half(sel_n, nk_e, nk_o, "resid", (10, 50))},
    }

    # 持続（生年の群で分ける）
    byi = H["by"].astype(int)
    P1, P2 = H[byi.between(2014, 2016)], H[byi.between(2017, 2020)]

    def cen(Z):
        # 期ごとの部分集合で中心を 1 に合わせる（critic）
        return Z.assign(b=Z["b"] / Z["b"].mean())

    def rho_sire(a, b):
        a, b = cen(a), cen(b)
        va, vb_ = value(a, "sire"), value(b, "sire")
        k = va[(va["nh"] >= 20)].index.intersection(vb_[vb_["nh"] >= 20].index)
        return spearmanr(va.loc[k, "v"], vb_.loc[k, "v"])[0], len(k)

    def rho_nick(a, b):
        out = []
        for Z in (cen(a), cen(b)):
            s_, b_ = value(Z, "sire")["v"], value(Z, "ds")["v"]
            out.append(nick_table(Z, s_, b_))
        k = out[0][out[0]["nh"] >= 5].index.intersection(out[1][out[1]["nh"] >= 5].index)
        return spearmanr(out[0].loc[k, "resid"], out[1].loc[k, "resid"])[0], len(k)
    rs, ns = rho_sire(P1, P2); rn, nn = rho_nick(P1, P2)
    rng = np.random.default_rng(3)
    null_s, null_n = [], []
    for _ in range(200):
        Hs = H.copy()
        Hs["sire"] = perm_within(Hs["by"].to_numpy(), Hs["sire"].to_numpy(), rng)
        null_s.append(rho_sire(Hs[byi.between(2014, 2016)], Hs[byi.between(2017, 2020)])[0])
        Hn = H.copy()
        Hn["ds"] = perm_within((Hn["by"] + "|" + Hn["sire"]).to_numpy(), Hn["ds"].to_numpy(), rng)
        null_n.append(rho_nick(Hn[byi.between(2014, 2016)], Hn[byi.between(2017, 2020)])[0])
    res["持続"] = {"種牡馬": {"rho": round(float(rs), 3), "数": ns, "帰無の 95% 点": round(float(np.nanpercentile(null_s, 95)), 3), "帰無の平均": round(float(np.nanmean(null_s)), 3), "帰無が上回る割合": round(float(np.mean(np.array(null_s) >= rs)), 3)},
                 "組の残差": {"rho": round(float(rn), 3), "数": nn, "帰無の 95% 点": round(float(np.nanpercentile(null_n, 95)), 3), "帰無の平均": round(float(np.nanmean(null_n)), 3), "帰無が上回る割合": round(float(np.mean(np.array(null_n) >= rn)), 3)}}
    print(res["持続"], flush=True)

    # 回収率（記述）
    T["y"] = T["date"].dt.year
    thr = T.groupby("y")["av_nick_add"].quantile(0.9)
    T["hi"] = T["av_nick_add"] > T["y"].map(lambda y: thr.get(y - 1, np.nan))
    res["線より上の割合（年ごと）"] = T[T["y"] >= 2019].groupby("y")["hi"].mean().round(3).to_dict()   # 組の履歴の無い行は 0 で、線が低く出うる（critic）
    U = T[T["y"] >= 2019].copy()
    U["ret"] = U["win"] * U["odds"]
    U["ret50"] = U["win"] * np.minimum(U["odds"], 50)
    pop = U.groupby("popularity")["ret"].mean()
    hi = U[U["hi"]]
    q99 = hi["ret"].quantile(0.99)
    base = float(hi["popularity"].map(pop).mean())
    res["回収率"] = {"走り": int(len(hi)), "的中": int(hi["win"].sum()), "回収率": round(float(hi["ret"].mean()), 3),
                  "同じ人気の基準線": round(base, 3), "比": round(float(hi["ret"].mean() / base), 3),
                  "上位 1% を除く": round(float(hi.loc[hi["ret"] <= q99, "ret"].mean()), 3), "50 倍で切る": round(float(hi["ret50"].mean()), 3),
                  "年ごと": hi.groupby("y")["ret"].mean().round(3).to_dict()}
    print(res["回収率"], flush=True)
    path = data.OUT / "aei_list.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    for k in ("種牡馬", "母父", "組（父 × 母父・残差）"):
        print(k, res["一覧"][k]["候補の数"], res["一覧"][k]["勝者の呪い"], res["一覧"][k]["上位"][:10], flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
