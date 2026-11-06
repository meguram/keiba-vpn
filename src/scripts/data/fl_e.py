"""最終層 E: データ派。似たレース（会場 × 芝ダ × 距離帯）の統計で、市場の確率を補正する。

E0 市場確率 × 条件別の補正比（単勝オッズ帯 × 条件セルで、実際の勝ち数 ÷ 市場が見込む勝ち数。粗いセルへ縮約）
E1 条件付きロジット [親モデル, 市場, 補正比の対数]（補正比は学習年から入れ子で計算して漏れを防ぐ）
E2 荒れるレースの分類に、条件セルの『荒れる率』を足す（C2 との比較）
python -m src.scripts.data.fl_e
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.scripts.data import fl_core as C
from src.scripts.data import fl_c
from src.scripts.data.fl_b import pick_stats, soft

K_PRIOR = 30.0
EDGES = [3, 7, 15]


def keys(D):
    db = np.digitize(D.distance, [1500, 1900, 2300])
    coarse = np.array([f"{s}|{d}" for s, d in zip(D.surface, db)])
    fine = np.array([f"{v}|{c}" for v, c in zip(D.venue, coarse)])
    return coarse, fine


def table(D, idx, coarse, fine):
    m = D.mask[idx]
    rr = np.broadcast_to(np.arange(len(idx))[:, None], m.shape)[m]
    df = pd.DataFrame({"c": coarse[idx][rr], "f": fine[idx][rr], "b": np.digitize(D.win_odds[idx][m], EDGES), "w": D.win[idx][m].astype(float), "q": D.q[idx][m]})
    gc = df.groupby(["c", "b"])[["w", "q"]].sum()
    rc = (gc["w"] + K_PRIOR) / (gc["q"] + K_PRIOR)
    gf = df.groupby(["f", "c", "b"])[["w", "q"]].sum().reset_index()
    gf["rc"] = [rc.get((c, b), 1.0) for c, b in zip(gf["c"], gf["b"])]
    gf["r"] = (gf["w"] + K_PRIOR * gf["rc"]) / (gf["q"] + K_PRIOR)
    return rc, gf.set_index(["f", "b"])["r"]


def lookup(D, idx, coarse, fine, tb):
    rc, rf = tb
    m = D.mask[idx]
    rr = np.broadcast_to(np.arange(len(idx))[:, None], m.shape)
    b = np.digitize(D.win_odds[idx], EDGES)
    f_h, c_h = fine[idx][rr], coarse[idx][rr]
    ix = pd.MultiIndex.from_arrays([f_h.ravel(), b.ravel()])
    v = rf.reindex(ix).to_numpy().reshape(m.shape)
    ixc = pd.MultiIndex.from_arrays([c_h.ravel(), b.ravel()])
    vc = rc.reindex(ixc).to_numpy().reshape(m.shape)
    v = np.where(np.isnan(v), np.where(np.isnan(vc), 1.0, vc), v)
    return np.log(np.clip(v, 1e-3, None))


def nested_lrc(D, years, yv, coarse, fine):
    """評価年 yv 以外の学習年は『その年を除いた学習年』の表で、評価年は全学習年の表で、補正比の対数を作る。"""
    out = np.zeros((D.R, D.N))
    T = [y for y in years if y != yv]
    for yt in T:
        others = [y for y in T if y != yt]
        te = C.races_of(D, [yt])
        out[te] = lookup(D, te, coarse, fine, table(D, C.races_of(D, others), coarse, fine))
    te = C.races_of(D, [yv])
    out[te] = lookup(D, te, coarse, fine, table(D, C.races_of(D, T), coarse, fine))
    return out


def main() -> None:
    D = C.load()
    ev = C.races_of(D, C.OOF_YEARS)
    coarse, fine = keys(D)
    pop = np.where(D.mask, -D.pop, -np.inf)
    cand = D.mask & (D.win_odds >= C.LONG_ODDS)
    res: dict = {"prob": {}, "pick": {}, "policies": [], "rough": {}}

    # 補正比そのものの確認（全年の表）
    rc, rf = table(D, ev, coarse, fine)
    res["ratio_coarse"] = {f"{k[0]}|b{k[1]}": round(float(v), 3) for k, v in rc.items()}
    print("粗いセル×オッズ帯（<3,3-7,7-15,15+）の 実際の勝ち ÷ 市場期待:", {k: round(v, 2) for k, v in res["ratio_coarse"].items()}, flush=True)

    # E0 / E1（評価年ごと、入れ子）
    sE0, sE1 = np.full((D.R, D.N), -np.inf), np.full((D.R, D.N), -np.inf)
    for yv in C.OOF_YEARS:
        T = [y for y in C.OOF_YEARS if y != yv]
        te = C.races_of(D, [yv])
        lrc = nested_lrc(D, C.OOF_YEARS, yv, coarse, fine)
        sE0[te] = np.where(D.mask[te], D.lq[te] + lrc[te], -np.inf)
        C.FEATS["lrc"] = lambda D_, lrc=lrc: lrc
        mdl = C.CLogit(["lp", "lq", "lrc"]).fit(D, C.races_of(D, T))
        sE1[te] = mdl.score(D, te)
        print(f"E1 θ({yv}) = {mdl.theta.round(3).tolist()}", flush=True)
    scores = {"E0": sE0, "E1": sE1}
    for nm, s in list(scores.items()):
        p = soft(s)
        res["prob"][nm] = C.prob_metrics(D, p)
        v = res["prob"][nm]
        print(f"{nm} win_ll {v['win_ll']:.4f} | 10倍以上 ll {v['long10_ll']:.4f} Σp {v['long10_sum_p']:.1f}/実 {v['long10_wins']:.0f}", flush=True)
        scores[nm + "_p"] = p

    uni = {bt: C.gen_uniform(D, bt, cand, 1.0, ridx=ev) for bt in ("tansho", "fukusho")}
    for nm in ("E0", "E1"):
        sc = scores[nm]
        top_sc = np.where(cand, sc, -np.inf).max(1)
        thr = np.quantile(top_sc[ev][np.isfinite(top_sc[ev])], 0.70)
        for tag, sel in (("all", None), ("top30", top_sc >= thr)):
            res["pick"][f"{nm}/{tag}"] = pick_stats(D, sc, cand, ev, sel)
            for bt in ("tansho", "fukusho"):
                tk = C.gen_single(D, sc, bt, 1, sel=sel, cand=cand, ridx=ev)
                ct = C.gen_single(D, pop, bt, 1, sel=sel, ridx=ev)
                o = C.summarize(D, f"E/{nm}/{tag}/{bt}_long1", tk, ct, group="E", params={"model": nm, "sel": tag, "bt": bt})
                if tag == "all":
                    o2 = C.summarize(D, "tmp", tk, uni[bt], group="E", log=False, detail=False)
                    o["vs_uniform_long"] = {"ctrl_roi": o2.get("ctrl_roi"), "diff": o2.get("diff"), "diff_ci": o2.get("diff_ci")}
                res["policies"].append(o)
                ps = res["pick"][f"{nm}/{tag}"]
                print(C.fmt(o) + f" | 人気薄の的中/市場期待 {ps['wins']}/{ps['market_expected']:.1f}" + (f" 対 一様買い差 {o['vs_uniform_long']['diff']:+.3f}" if "vs_uniform_long" in o else ""), flush=True)
        p = scores[nm + "_p"]
        for th in (1.0, 1.2):
            sel_c = cand & (p * D.win_odds >= th)
            sc2 = np.where(sel_c, p * D.win_odds, -np.inf)
            tk = C.gen_single(D, sc2, "tansho", 1, cand=sel_c, ridx=ev)
            ct = C.gen_single(D, pop, "tansho", 1, sel=sel_c.any(1), ridx=ev)
            o = C.summarize(D, f"E/{nm}/ev{th}/tansho_long1", tk, ct, group="E", params={"model": nm, "ev": th})
            res["policies"].append(o)
            print(C.fmt(o), flush=True)

    # E2: 荒れるレースの分類 + 条件セルの荒れる率
    nlong3 = (D.top3 & D.mask & (D.win_odds >= 10)).sum(1)
    y_r = nlong3 >= 1
    base = fl_c.race_features(D)
    z0, _ = fl_c.rough_loyo(D, y_r)
    rate = np.zeros(D.R)
    for yv in C.OOF_YEARS:
        T = [y for y in C.OOF_YEARS if y != yv]

        def rate_of(src, dst):
            s = pd.DataFrame({"c": coarse[src], "f": fine[src], "y": y_r[src].astype(float)})
            b0 = s["y"].mean()
            gc = s.groupby("c")["y"].agg(["sum", "count"])
            rcm = ((gc["sum"] + 50 * b0) / (gc["count"] + 50))
            gf = s.groupby(["f", "c"])["y"].agg(["sum", "count"]).reset_index()
            gf["r"] = (gf["sum"] + 50 * gf["c"].map(rcm)) / (gf["count"] + 50)
            rf_ = gf.set_index("f")["r"]
            return pd.Series(fine[dst]).map(rf_).fillna(pd.Series(coarse[dst]).map(rcm)).fillna(b0).to_numpy()

        for yt in T:
            rate[C.races_of(D, [yt])] = rate_of(C.races_of(D, [x for x in T if x != yt]), C.races_of(D, [yt]))
        rate[C.races_of(D, [yv])] = rate_of(C.races_of(D, T), C.races_of(D, [yv]))
    z1, _ = fl_c.rough_loyo(D, y_r, Xr=np.concatenate([base, rate[:, None]], 1))
    res["rough"] = {"auc_C2": fl_c.auc(z0[ev], y_r[ev]), "auc_E2": fl_c.auc(z1[ev], y_r[ev])}
    print(f"荒れるレース分類 AUC: 市場・親モデル特徴のみ {res['rough']['auc_C2']:.3f} → 条件セル荒れ率を追加 {res['rough']['auc_E2']:.3f}", flush=True)

    (C.OUT / "stage_E.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    np.savez_compressed(C.OUT / "scores_E.npz", E0=np.where(np.isfinite(sE0), sE0, np.nan).astype(np.float32), E1=np.where(np.isfinite(sE1), sE1, np.nan).astype(np.float32), rough_z_E2=z1)
    C.save_trials("E")


if __name__ == "__main__":
    main()
