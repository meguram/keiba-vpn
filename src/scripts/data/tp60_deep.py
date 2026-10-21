"""tp-60 10 世代の血量（存在割合）による種牡馬の分析。1 頭 1 指標（総血量）にまとめ、位置の重複を残差で分け、近縁な種牡馬は代表に寄せる。

python -m src.scripts.data.tp60_deep select     # 出現率の高い男系祖先の候補、近縁（父子・相関）の整理 → 代表
python -m src.scripts.data.tp60_deep effect     # 代表ごとの血量帯別成績（勝率・複勝率・回収率）と、V4 残差への回帰（深い血量 6〜10 世代の上乗せ）
python -m src.scripts.data.tp60_deep group      # 特徴量の群（オッズ非依存）を horse 単位で書き出す: DEEP_TOT（総血量）, DEEP_6_10（6〜10 世代だけ）

位置の重複の扱い:
  (a) 祖先 a の血量 = Σ_{出現位置} 2^-世代（1〜10 世代）を 1 つの数にする（父・父父・母父…で別々の指標を並べない）。
  (b) 位置別に見るときは、『父である／母父である』の指標を回帰に入れて、血量の残りの部分（その他の位置の 1〜5 世代、6〜10 世代）の係数を見る。
  (c) 父子・全兄弟・相関 0.5 以上の種牡馬は 1 頭に寄せる（産駒数の多い方を代表）。
市場オッズは分析（残差の基準）にだけ使い、特徴量の群には入れない。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.scripts.data import tp60_lib as L
from src.scripts.data.tp60_roi import _save

MIN_PRESENCE = float(os.environ.get("TP60_MIN_PRESENCE", 0.03))      # 5 世代以内に現れる競走馬の割合の下限（10 世代の出現率は古い創始者で 97% に飽和するため、基準は 5 世代以内の出現率）
N_REP = int(os.environ.get("TP60_N_REP", 40))
CORR_MERGE = 0.5


def _share_wide(bs: pd.DataFrame, ancs: list[str], horses: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """horse × 祖先 の w5, wdeep（密。祖先は数十〜数百に絞った後だけ使う）。"""
    ai = {a: i for i, a in enumerate(ancs)}
    sub = bs[bs["anc_id"].isin(ai)]
    r = horses.get_indexer(sub["horse_id"])
    c = sub["anc_id"].map(ai).to_numpy()
    ok = r >= 0
    W5 = np.zeros((len(horses), len(ancs)), dtype=np.float32)
    WD = np.zeros((len(horses), len(ancs)), dtype=np.float32)
    W5[r[ok], c[ok]] = sub["w5"].to_numpy()[ok]
    WD[r[ok], c[ok]] = sub["wdeep"].to_numpy()[ok]
    return W5, WD


def cmd_select(a) -> None:
    m = L.load_master()
    bs = pd.read_parquet(L.PED / "blood_share.parquet")
    names = json.loads((L.PED / "anc_names.json").read_text(encoding="utf-8"))
    cnt = m.groupby("horse_id").size()
    tot = cnt.sum()
    bs = bs.assign(cnt=bs["horse_id"].map(cnt).fillna(0))
    bs["p10"] = ((bs["w5"] + bs["wdeep"]) > 0)
    bs["p5"] = bs["w5"] > 0
    bs["pdeep_only"] = (bs["w5"] == 0) & (bs["wdeep"] > 0)
    g = bs.groupby("anc_id")
    cand = pd.DataFrame({"presence10": g.apply(lambda d: (d["cnt"] * d["p10"]).sum() / tot), "presence5": g.apply(lambda d: (d["cnt"] * d["p5"]).sum() / tot),
                         "presence_deep_only": g.apply(lambda d: (d["cnt"] * d["pdeep_only"]).sum() / tot), "mean_share10": g.apply(lambda d: (d["cnt"] * (d["w5"] + d["wdeep"])).sum() / tot)})
    cand["name"] = cand.index.map(lambda i: names.get(i, ""))
    cand = cand.sort_values("presence10", ascending=False)
    thr_table = {str(t): int((cand["presence10"] >= t).sum()) for t in (0.5, 0.3, 0.2, 0.1, 0.05, 0.03)}
    thr_table5 = {str(t): int((cand["presence5"] >= t).sum()) for t in (0.5, 0.3, 0.2, 0.1, 0.05, 0.03)}
    keep = cand[(cand["presence5"] >= MIN_PRESENCE) & (cand["presence10"] <= 0.97)]
    sat = cand[cand["presence10"] > 0.97]
    L.log(f"候補（presence5 ≥ {MIN_PRESENCE}）{len(keep)}、10 世代の出現率が飽和（>97%）{len(sat)}")
    # 相関・父子の整理
    ancs = list(keep.index)
    horses = pd.Index(sorted(set(bs["horse_id"])))
    W5, WD = _share_wide(bs, ancs, horses)
    W = W5 + WD
    # 馬ごとの出走数で重み付けした相関
    w = horses.map(cnt).fillna(0).to_numpy(dtype=np.float64)
    Wc = (W - (W * w[:, None]).sum(0) / w.sum()) * np.sqrt(w)[:, None]
    nrm = np.sqrt((Wc ** 2).sum(0)) + 1e-12
    C = (Wc.T @ Wc) / np.outer(nrm, nrm)
    G = pd.read_parquet(L.OUTD / "embed" / "graph.parquet") if (L.OUTD / "embed" / "graph.parquet").exists() else None
    father = dict(zip(G["node_id"], G["sire_id"])) if G is not None else {}
    idx = {x: i for i, x in enumerate(ancs)}
    order = [x for x in cand.sort_values("presence5", ascending=False).index if x in set(keep.index)]
    reps, assigned = [], {}
    for s in sorted(order, key=lambda x: -cand.loc[x, "presence5"]):
        if s in assigned:
            continue
        reps.append(s)
        assigned[s] = s
        for t in order:
            if t in assigned:
                continue
            rel = (father.get(t) == s) or (father.get(s) == t)
            if rel or C[idx[s], idx[t]] >= CORR_MERGE:
                assigned[t] = s
        if len(reps) >= N_REP:
            break
    out = {"n_runner_rows": int(tot), "thr_presence10": thr_table, "thr_presence5": thr_table5, "n_saturated_gt97": int(len(sat)),
           "saturated": [{"id": i, "name": cand.loc[i, "name"], "presence10": float(cand.loc[i, "presence10"])} for i in sat.index[:12]],
           "reps": [{"id": r, "name": cand.loc[r, "name"], "presence10": float(cand.loc[r, "presence10"]), "presence5": float(cand.loc[r, "presence5"]),
                     "presence_deep_only": float(cand.loc[r, "presence_deep_only"]), "mean_share10": float(cand.loc[r, "mean_share10"]),
                     "merged": [{"id": t, "name": cand.loc[t, "name"], "corr": float(C[idx[r], idx[t]])} for t, rr in assigned.items() if rr == r and t != r][:6]} for r in reps],
           "n_after_dedup": len(reps), "n_before_dedup": int(len(keep))}
    # 位置別を並べたときの重複の見積り: 同じ祖先が複数の位置に出る馬の割合（上位 reps）
    _save("deep_select.json", out)
    L.log(f"代表 {len(reps)}（整理前 {len(keep)}）: {thr_table}")


def _ols_cluster(X: np.ndarray, y: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    codes, uniq = pd.factorize(cl)
    S = np.zeros((X.shape[1], X.shape[1]))
    Xe = X * e[:, None]
    sums = np.zeros((len(uniq), X.shape[1]))
    np.add.at(sums, codes, Xe)
    S = sums.T @ sums
    g = len(uniq)
    adj = g / max(g - 1, 1)
    V = adj * XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def cmd_effect(a) -> None:
    sel = json.loads((L.RES / "deep_select.json").read_text(encoding="utf-8"))
    reps = [r["id"] for r in sel["reps"]]
    nm = {r["id"]: r["name"] for r in sel["reps"]}
    m = L.load_master()
    oof_p = os.environ.get("TP60_V5_OOF", "/home/jovyan/work/keiba-vpn/data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet")
    if os.path.exists(oof_p):
        o = pd.read_parquet(oof_p)[["race_id", "horse_id", "p_win"]].rename(columns={"p_win": "p_v5"})
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        m = m.merge(o.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    else:
        m["p_v5"] = np.nan
    bs = pd.read_parquet(L.PED / "blood_share.parquet")
    horses = pd.Index(sorted(set(bs["horse_id"])))
    W5, WD = _share_wide(bs, reps, horses)
    hrow = horses.get_indexer(m["horse_id"])
    ok = hrow >= 0
    m = m[ok].reset_index(drop=True)
    hrow = hrow[ok]
    tr = m["year"] <= 2025
    out: dict = {"reps": {}}
    BANDS = [("0", 0, 0), ("(0,3.1%]", 1e-9, 0.03125), ("(3.1,6.3%]", 0.03125, 0.0625), ("(6.3,12.5%]", 0.0625, 0.125), (">12.5%", 0.125, 9)]
    coefs = []
    for k, s in enumerate(reps):
        w5, wd = W5[hrow, k].astype(float), WD[hrow, k].astype(float)
        tot = w5 + wd
        e: dict = {"name": nm[s], "bands": {}}
        for bn, lo, hi in BANDS:
            mk = ((tot == 0) if bn == "0" else ((tot > lo) & (tot <= hi))) & tr.to_numpy()
            if mk.sum() < 200:
                continue
            cw, cp = L.roi_ci(m, mk, "ret_win", B=300), L.roi_ci(m, mk, "ret_place", B=300)
            d = m[mk]
            e["bands"][bn] = {"horses": int(mk.sum()), "win_rate": float(d["win"].mean()), "top3_rate": float(d["top3"].mean()), "roi_win": cw["roi"], "roi_win_lo": cw["lo"], "roi_win_hi": cw["hi"],
                              "roi_place": cp["roi"], "roi_place_lo": cp["lo"], "roi_place_hi": cp["hi"], "ex_mkt": float((d["win"] - d["q_win"]).mean())}
        mk = ((w5 == 0) & (wd > 0)) & tr.to_numpy()
        if mk.sum() >= 200:
            cw, cp = L.roi_ci(m, mk, "ret_win", B=300), L.roi_ci(m, mk, "ret_place", B=300)
            d = m[mk]
            e["bands"]["deep_only(6-10世代のみ)"] = {"horses": int(mk.sum()), "win_rate": float(d["win"].mean()), "top3_rate": float(d["top3"].mean()), "roi_win": cw["roi"], "roi_win_lo": cw["lo"], "roi_win_hi": cw["hi"],
                                                     "roi_place": cp["roi"], "roi_place_lo": cp["lo"], "roi_place_hi": cp["hi"], "ex_mkt": float((d["win"] - d["q_win"]).mean())}
        # 回帰: 血量を持つ馬に限り、y ~ 1 + 父 + 母父 + 1〜5 世代のその他 + 6〜10 世代。y は V4 の勝率との差（2022 年以降）と市場の勝率との差（2020 年以降）
        pres = (tot > 0) & tr.to_numpy()
        is_sire, is_dsire = (m["sire_id"] == s).to_numpy(), (m["dsire_id"] == s).to_numpy()
        other5 = np.clip(w5 - 0.5 * is_sire - 0.25 * is_dsire, 0, None)
        for yname, ycol, y0 in (("resid_v4", "p_win", 2022), ("resid_v5", "p_v5", 2022), ("resid_mkt", "q_win", 2020)):
            sel_r = pres & (m["year"] >= y0).to_numpy() & m[ycol].notna().to_numpy()
            if yname == "resid_v5":
                sel_r &= (m["year"] <= 2024).to_numpy()
            if sel_r.sum() < 800:
                continue
            X = np.column_stack([np.ones(sel_r.sum()), is_sire[sel_r], is_dsire[sel_r], other5[sel_r], wd[sel_r]])
            y = (m.loc[sel_r, "win"] - m.loc[sel_r, ycol]).to_numpy()
            b, se = _ols_cluster(X, y, m.loc[sel_r, "race_id"].to_numpy())
            e[yname] = {"n": int(sel_r.sum()), "coef": {"sire": float(b[1]), "dsire": float(b[2]), "other5": float(b[3]), "deep": float(b[4])},
                        "t": {"sire": float(b[1] / se[1]) if se[1] > 0 else None, "dsire": float(b[2] / se[2]) if se[2] > 0 else None, "other5": float(b[3] / se[3]) if se[3] > 0 else None,
                              "deep": float(b[4] / se[4]) if se[4] > 0 else None}, "deep_mean_share_among_present": float(wd[sel_r].mean())}
            # 前半・後半での深い血量の係数（安定性）
            hf = {}
            halves = (("2022-2023", (2022, 2023)), ("2024", (2024, 2024))) if yname == "resid_v5" else (("2020-2022", (2020, 2022)), ("2023-2025", (2023, 2025)))
            for lab, ys in halves:
                rr = sel_r & m["year"].between(*ys).to_numpy()
                if rr.sum() < 400:
                    continue
                X2 = np.column_stack([np.ones(rr.sum()), is_sire[rr], is_dsire[rr], other5[rr], wd[rr]])
                b2, se2 = _ols_cluster(X2, (m.loc[rr, "win"] - m.loc[rr, ycol]).to_numpy(), m.loc[rr, "race_id"].to_numpy())
                hf[lab] = {"deep": float(b2[4]), "t": float(b2[4] / se2[4]) if se2[4] > 0 else None, "other5": float(b2[3])}
            e[yname]["halves"] = hf
        out["reps"][s] = e
        coefs.append((s, e))
    # 集計: 深い血量・その他 5 世代の係数の符号と t の分布
    summ = {}
    from scipy.stats import spearmanr

    for yname in ("resid_v4", "resid_v5", "resid_mkt"):
        ts_deep = [e[yname]["t"]["deep"] for _, e in coefs if yname in e and e[yname]["t"]["deep"] is not None]
        ts_o5 = [e[yname]["t"]["other5"] for _, e in coefs if yname in e and e[yname]["t"]["other5"] is not None]
        if not ts_deep:
            continue
        ts_deep, ts_o5 = np.array(ts_deep), np.array(ts_o5)
        k1, k2 = ("2022-2023", "2024") if yname == "resid_v5" else ("2020-2022", "2023-2025")
        h1 = [e[yname]["halves"].get(k1, {}).get("deep") for _, e in coefs if yname in e]
        h2 = [e[yname]["halves"].get(k2, {}).get("deep") for _, e in coefs if yname in e]
        pr = [(x, y_) for x, y_ in zip(h1, h2) if x is not None and y_ is not None]
        summ[yname] = {"n_reps": int(len(ts_deep)), "deep_|t|>2": int((np.abs(ts_deep) > 2).sum()), "deep_|t|>3": int((np.abs(ts_deep) > 3).sum()), "deep_mean_t": float(ts_deep.mean()), "deep_sd_t": float(ts_deep.std()),
                       "other5_|t|>2": int((np.abs(ts_o5) > 2).sum()), "other5_|t|>3": int((np.abs(ts_o5) > 3).sum()), "other5_mean_t": float(ts_o5.mean()),
                       "deep_half_corr": float(np.corrcoef(*zip(*pr))[0, 1]) if len(pr) > 5 else None, "deep_half_spearman": float(spearmanr(*zip(*pr)).correlation) if len(pr) > 5 else None, "expected_|t|>2_under_null": float(0.0455 * len(ts_deep))}
    out["summary"] = summ
    _save("deep_effect.json", out)
    L.log(f"effect 完了: {summ}")


def cmd_group(a) -> None:
    sel = json.loads((L.RES / "deep_select.json").read_text(encoding="utf-8"))
    reps = [r["id"] for r in sel["reps"]]
    bs = pd.read_parquet(L.PED / "blood_share.parquet")
    horses = pd.Index(sorted(set(bs["horse_id"])))
    W5, WD = _share_wide(bs, reps, horses)
    hs = pd.read_parquet(L.PED / "horse_static.parquet").drop_duplicates("horse_id").set_index("horse_id").reindex(horses)
    tot = pd.DataFrame(W5 + WD, index=horses, columns=[f"d60_tot_{i}" for i in range(len(reps))]).astype("float32")
    deep = pd.DataFrame(WD, index=horses, columns=[f"d60_deep_{i}" for i in range(len(reps))]).astype("float32")
    deep["d60_deep_sum"] = WD.sum(1).astype("float32")
    deep["d60_deep_cov"] = hs["deep_cov"].astype("float32").to_numpy()
    out = L.OUTD / "groups"
    out.mkdir(parents=True, exist_ok=True)
    tot.reset_index().rename(columns={"index": "horse_id"}).to_parquet(out / "DEEP_TOT.parquet", index=False)
    deep.reset_index().rename(columns={"index": "horse_id"}).to_parquet(out / "DEEP_6_10.parquet", index=False)
    # 家系の静的特徴（オッズ・結果を使わない。生年までに分かる数）: 父の産駒世代、父・母父の産駒数（その馬の生年まで）、母の何番目の産駒か
    h2 = hs.reset_index().rename(columns={"index": "horse_id"})
    h2 = h2.dropna(subset=["birth_year"]).copy()
    first_by = h2.groupby("sire_id")["birth_year"].min()
    h2["sire_first_by"] = h2["sire_id"].map(first_by)
    crop = h2["birth_year"] - h2["sire_first_by"]
    h2["fam_sire_crop"] = np.where(h2["sire_first_by"] >= 2017, crop, 7.0)           # 0=初年度産駒、7=既存の種牡馬（2017 年以前に産駒がいる）
    for nm, key in (("sire", "sire_id"), ("dsire", "dsire_id"), ("dam", "dam_id")):
        g = h2.groupby([key, "birth_year"]).size().rename("n").reset_index().sort_values([key, "birth_year"])
        g["cum"] = g.groupby(key)["n"].cumsum()
        h2 = h2.merge(g[[key, "birth_year", "cum"]].rename(columns={"cum": f"fam_{nm}_foals"}), on=[key, "birth_year"], how="left")
    h2["fam_dam_prev"] = h2["fam_dam_foals"] - h2.groupby(["dam_id", "birth_year"])["horse_id"].transform("size")       # 前の年までの産駒数（競走馬になった馬）
    fam = h2[["horse_id", "fam_sire_crop", "fam_sire_foals", "fam_dsire_foals", "fam_dam_foals", "fam_dam_prev"]].astype({c: "float32" for c in ("fam_sire_crop", "fam_sire_foals", "fam_dsire_foals", "fam_dam_foals", "fam_dam_prev")})
    fam.to_parquet(out / "PED_FAM.parquet", index=False)
    L.log(f"PED_FAM {fam.shape}")
    (out / "deep_reps.json").write_text(json.dumps([{"col": i, "id": r} for i, r in enumerate(reps)], ensure_ascii=False), encoding="utf-8")
    L.log(f"群を書き出し: DEEP_TOT {tot.shape}, DEEP_6_10 {deep.shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("select", "effect", "group"):
        sub.add_parser(n)
    a = ap.parse_args()
    {"select": cmd_select, "effect": cmd_effect, "group": cmd_group}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
