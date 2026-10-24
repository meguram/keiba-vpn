"""tp-37 の馬券側と残差の分析（レース単位の母数）。

resid: V4 の OOF 予測（valid 2022〜24。leave-one-year-out）の『勝ち残差』（実際の勝ち − 予測確率）が、追い切りの条件・交互作用で説明できるか。
       条件に当てはまる馬がいるレースを 1 単位として、レース内の残差の合計をブートストラップする。BH 補正と年別の符号の一致を見る。
bet:   V4 の予測 × 追い切りの条件で絞り込んだ買い方のレース単位の回収率（レース均等 / 最上位人気 1 頭 / モデル最上位 1 頭）。
       絞り込み = 条件のみ / EV≥1.0 / EV≥1.2（モデル単体）/ 混合（Benter 型）EV≥1.0 / モデル上位 3 頭。
       選択と集計は 2022〜24 年だけ。2025 年は参考列。市場オッズは馬券の判断にだけ使う（特徴量には入れない）。
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.scripts.data import tp35_eval as E
from src.scripts.data import tp35_race_level as RL
from src.scripts.data import tp37_eval as M

OUT = M.OUT
B = int(os.environ.get("TP37_BOOT", 400))
YEARS_SEL = (2022, 2024)
MODES = ("none", "ev1.0", "ev1.2", "bev1.0", "top3")
MODE_NAMES = {"none": "条件のみ", "ev1.0": "モデルEV≥1.0", "ev1.2": "モデルEV≥1.2", "bev1.0": "混合EV≥1.0", "top3": "モデル上位3頭"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────── 土台の表
def load_frame() -> pd.DataFrame:
    """tp-35 のレース単位の表（払戻・確定オッズ・V4 の p_win を含む）に、tp-37 の群と離散キーを足し、混合確率（Benter 型）を足す。"""
    d = pd.read_parquet(E.OUT / "race_level/frame.parquet")
    df, *_ = M.t.setup()
    assert (d["race_id"].astype(str).to_numpy() == df["race_id"].astype(str).to_numpy()).all() and (d["horse_id"].astype(str).to_numpy() == df["horse_id"].astype(str).to_numpy()).all()
    for c in ("distance", "surface", "a_style", "field_size"):
        d[c] = df[c].to_numpy()
    d["surface"] = d["surface"].astype(str)
    for g, cols in (("TRZ", ["f_ztr_t4", "f_ztr_t1", "f_ztr_acc", "f_ztrs_t4", "f_ztrs_t1", "f_ztrs_acc", "f_dev_t4", "f_z0_t4", "m_ztr_t4", "min_ztr_t1"]), ("D1_between", ["hard_3d", "n_all", "first_days", "n_distinct"]),
                    ("D2_change", ["d_ztr_t4", "d_ztr_t1", "d_int", "d_nall"]), ("D3_rider", ["f_rider_bias_t4"])):
        x = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
        assert (x["race_id"].to_numpy() == d["race_id"].astype(str).to_numpy()).all()
        for c in cols:
            d[c] = x[c].to_numpy()
    tf = OUT / "groups/D7_text_flags.parquet"
    if tf.exists():
        x = pd.read_parquet(tf)
        assert (x["race_id"].to_numpy() == d["race_id"].astype(str).to_numpy()).all()
        for c in ("fc_net", "fc_sensen", "fc_okure", "fc_doonyu", "fc_yusei", "fc_ressei"):
            d[c] = x[c].to_numpy()
    k = pd.read_parquet(OUT / "groups/KEYS.parquet")
    assert (k["race_id"].to_numpy() == d["race_id"].astype(str).to_numpy()).all()
    for c in k.columns:
        if c not in ("race_id", "horse_id"):
            d[c] = k[c].to_numpy()
    d = d.rename(columns={"clu": "clu_k"})
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    # 混合確率（Benter 型）: 評価年より前の年の勝者の対数尤度で当てはめる（tp-36 と同じ。2022 年は前の年が無いので欠損）
    from src.scripts.data import tp36_edge as T36

    d36 = T36.load_frame()
    d36["race_id"], d36["horse_id"] = d36["race_id"].astype(str), d36["horse_id"].astype(str)
    d36["p_blend"] = np.nan
    for ey in (2023, 2024, 2025):
        a_, b_ = T36.fit_benter(d36[d36["year"] < ey])
        m = d36["year"] == ey
        d36.loc[m, "p_blend"] = T36.blend(d36[m], a_, b_)
    d = d.merge(d36[["race_id", "horse_id", "p_blend"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    return d


def conditions(d: pd.DataFrame) -> "OrderedDict[str, np.ndarray]":
    """追い切りの条件（各行に True/False）。欠損は False。"""
    C: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def put(name, m):
        C[name] = np.asarray(pd.Series(m).fillna(False).to_numpy(dtype=bool))

    cls = d["F_cls"]
    for c in ("A1", "A2", "A3", "B1", "B2", "B3", "X"):
        put(f"型={c}", cls == c)
    put("型=A(加速)", cls.isin(["A1", "A2", "A3"]))
    put("型=B(減速)", cls.isin(["B1", "B2", "B3"]))
    for lab in ("坂路54-56", "坂路<54", "坂路56-58"):
        put(f"前週土日={lab}", d["wk_label"] == lab)
    for m, nm in (("t4", "4F時計"), ("t1", "終い1F"), ("acc", "加速度")):
        z = d[f"f_ztr_{m}"]
        put(f"調教師比 {nm} かなり速い(<-1)", z < -1.0)
        put(f"調教師比 {nm} 速い(<-0.35)", z < -0.35)
        put(f"調教師比 {nm} 遅い(>0.35)", z > 0.35)
        put(f"調教師比 {nm} かなり遅い(>1)", z > 1.0)
    put("調教師比 4F速い かつ 終い速い", (d["f_ztr_t4"] < -0.35) & (d["f_ztr_t1"] < -0.35))
    put("調教師比(標準偏差で割る) 4F<-1", d["f_ztrs_t4"] < -1.0)
    put("全体比 4F速い(z0<-1)", d["f_z0_t4"] < -1.0)
    clu = d["clu_k"]
    for k in range(5):
        put(f"クラスタ{k}", clu == k)
        put(f"クラスタ{k} かつ 4F速い", (clu == k) & (d["f_ztr_t4"] < -0.35))
    dist = pd.to_numeric(d["distance"], errors="coerce")
    st = pd.to_numeric(d["a_style"], errors="coerce")
    put("型=A3 かつ 短距離(<1400)", (cls == "A3") & (dist < 1400))
    put("型=A3 かつ ダート", (cls == "A3") & (d["surface"] == "ダ"))
    put("型=B(減速) かつ 芝", cls.isin(["B1", "B2", "B3"]) & (d["surface"] == "芝"))
    put("4F速い かつ 逃げ先行寄り(a_style<0.27)", (d["f_ztr_t4"] < -0.35) & (st < 0.27))
    put("終い速い かつ 差し追込寄り(a_style>0.64)", (d["f_ztr_t1"] < -0.35) & (st > 0.64))
    put("加速度速い かつ 長距離(>=2000)", (d["f_ztr_acc"] < -0.35) & (dist >= 2000))
    put("直前3日に強め以上", d["hard_3d"] >= 1)
    put("追い切り本数多い(>=8)", d["n_all"] >= 8)
    put("始まりが早い(21日以上前)", d["first_days"] >= 21)
    put("前走比で4F大きく改善(<-0.7)", d["d_ztr_t4"] < -0.7)
    put("追い切り騎手=助手", d["f_rider_class"] == 1)
    put("追い切り騎手=騎手名あり", d["f_rider_class"] == 4)
    put("短評=動き良好", d["f_eval"].isin(["動き軽快", "動き上々", "好気配", "キビキビ", "活気十分", "動き良好"]))
    put("短評=目立たず", d["f_eval"].isin(["目立たず", "動き平凡"]))
    put("追い切りランクA", d["f_rank"] == "A")
    put("追い切りランクD", d["f_rank"] == "D")
    if "fc_net" in d.columns:
        put("コメント=好意的(net>=1)", d["fc_net"] >= 1)
        put("コメント=否定的(net<=-1)", d["fc_net"] <= -1)
        put("コメント=併せ馬で先着", d["fc_sensen"] == 1)
        put("コメント=併せ馬で遅れ", d["fc_okure"] == 1)
        put("コメント=併せ馬で優勢", d["fc_yusei"] == 1)
        put("コメント=併せ馬で劣勢", d["fc_ressei"] == 1)
    return C


def mode_masks(d: pd.DataFrame) -> dict[str, np.ndarray]:
    wo = d["win_odds"].to_numpy(dtype=float)
    ev = d["p_win"].to_numpy(dtype=float) * wo
    bev = d["p_blend"].to_numpy(dtype=float) * wo
    rank = d.groupby("race_id")["p_win"].rank(ascending=False, method="first").to_numpy()
    return {"none": np.ones(len(d), dtype=bool), "ev1.0": np.nan_to_num(ev) >= 1.0, "ev1.2": np.nan_to_num(ev) >= 1.2, "bev1.0": np.nan_to_num(bev) >= 1.0, "top3": np.nan_to_num(rank, nan=99) <= 3}


# ───────────────────────── resid（V4 の勝ち残差）
def _bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    thr = q * (np.arange(1, n + 1)) / n
    ok = p[o] <= thr
    k = (np.max(np.flatnonzero(ok)) + 1) if ok.any() else 0
    keep = np.zeros(n, dtype=bool)
    keep[o[:k]] = True
    return keep


def cmd_resid(a) -> None:
    d0 = load_frame()
    d = RL.prep(d0)
    d = d[d["year"].between(*YEARS_SEL) & d["p_win"].notna()].reset_index(drop=True)
    d["resid"] = d["win"].astype(float) - d["p_win"].astype(float)
    C = conditions(d)
    base_c = [k for k in C if k.startswith(("型=", "調教師比", "クラスタ")) and "かつ" not in k][:24]
    cells = OrderedDict(C)
    for bc in base_c:
        for ca, g in d.groupby("cond_a"):
            if (C[bc] & (d["cond_a"] == ca).to_numpy()).sum() >= 300:
                cells[f"{bc} × {ca}"] = C[bc] & (d["cond_a"] == ca).to_numpy()
    rng = np.random.default_rng(11)
    rows = []
    rid_codes, races = pd.factorize(d["race_id"])
    nR = len(races)
    res = d["resid"].to_numpy()
    yr = d["year"].to_numpy()
    for name, m in cells.items():
        if m.sum() < 300:
            continue
        r = np.bincount(rid_codes[m], weights=res[m], minlength=nR)
        n = np.bincount(rid_codes[m], minlength=nR).astype(float)
        pw = np.bincount(rid_codes[m], weights=d["p_win"].to_numpy()[m], minlength=nR)
        has = n > 0
        rr, nn, pp = r[has], n[has], pw[has]
        est = rr.sum() / nn.sum()
        idx = rng.integers(0, len(rr), (B, len(rr)))
        bs = rr[idx].sum(1) / nn[idx].sum(1)
        se = float(bs.std())
        z = est / se if se > 0 else 0.0
        yrs = {}
        for y in range(YEARS_SEL[0], YEARS_SEL[1] + 1):
            mm = m & (yr == y)
            yrs[str(y)] = float(res[mm].mean()) if mm.sum() >= 50 else None
        signs = [np.sign(v) for v in yrs.values() if v is not None]
        rows.append({"cond": name, "races": int(has.sum()), "horses": int(nn.sum()), "mean_n": float(nn.sum() / has.sum()), "resid_per_horse": float(est), "rel_to_p": float(rr.sum() / pp.sum()),
                     "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "z": float(z), "p": float(2 * (1 - ndtr(abs(z)))), "years": yrs, "same_sign_all_years": bool(len(signs) == 3 and len(set(signs)) == 1)})
    R = pd.DataFrame(rows)
    R["bh10"] = _bh(R["p"].to_numpy(), 0.10)
    R["bh05"] = _bh(R["p"].to_numpy(), 0.05)
    summ = {"cells_tested": int(len(R)), "base_conditions": len(C), "p_lt_005_raw": int((R["p"] < 0.05).sum()), "expected_by_chance": float(0.05 * len(R)), "bh10": int(R["bh10"].sum()), "bh05": int(R["bh05"].sum()),
            "bh10_and_same_sign": int((R["bh10"] & R["same_sign_all_years"]).sum()), "boot": B,
            "top": json.loads(R.sort_values("p").head(15).to_json(orient="records", force_ascii=False))}
    M._jsave(OUT / "resid.json", {"summary": summ, "all": json.loads(R.to_json(orient="records", force_ascii=False))})
    _log(f"残差: {summ['cells_tested']} 条件 / raw p<0.05 {summ['p_lt_005_raw']}（偶然の期待 {summ['expected_by_chance']:.1f}）/ BH10 {summ['bh10']} / BH05 {summ['bh05']} / BH10 かつ 3 年同符号 {summ['bh10_and_same_sign']}")
    for r in summ["top"][:8]:
        _log(f"  {r['cond']}: 残差/頭 {r['resid_per_horse']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] レース {r['races']} p={r['p']:.4f} 年別 {r['years']}")


# ───────────────────────── bet
def _label_table(d: pd.DataFrame, C, MK) -> tuple[pd.DataFrame, dict]:
    parts, names = [], {}
    for mode in MODES:
        mk = MK[mode]
        for cname, cm in [("ALL", np.ones(len(d), dtype=bool))] + list(C.items()):
            s = cm & mk
            if s.sum() < 150:
                continue
            lab = f"{cname}|{mode}"
            parts.append(RL.race_table(d, np.where(s, lab, None).astype(object)))
            names[lab] = (cname, mode)
    return pd.concat(parts), names


def _rc(rt, races, bases, min_den=300, nb=500, seed=0):
    nm, N_, D_ = RL.build_matrix(rt, races, bases)
    r = RL.reality_check(N_, D_, min_den=min_den, nb=nb, seed=seed)
    o, dt = r.pop("obs"), r.pop("den_tot")
    return nm, o, dt, r


def cmd_bet(a) -> None:
    d0 = load_frame()
    d = RL.prep(d0)
    d = d[d["p_win"].notna() & d["win_odds"].notna()].reset_index(drop=True)
    dv = d[d["year"].between(*YEARS_SEL)].reset_index(drop=True)
    C = conditions(dv)
    MK = mode_masks(dv)
    races = pd.Index(sorted(dv["race_id"].unique()))
    res: dict = {"rows": int(len(dv)), "races": int(len(races)), "years": list(YEARS_SEL), "conditions": len(C), "modes": list(MODES), "boot": B}
    rt, names = _label_table(dv, C, MK)
    S = RL.summarize(rt)
    S = S[S["races"] >= 300]
    rt = rt[rt.index.get_level_values(0).isin(S.index)]
    S = RL.with_ci(rt, S, seed=1, metrics=("roi_race_w", "roi_fav_w", "roi_race_p", "roi_fav_p", "roi_horse_w"))
    S["cond"] = [names[k][0] for k in S.index]
    S["mode"] = [names[k][1] for k in S.index]
    # モード単独（条件なし）との差
    base_mode = {m: S[(S["cond"] == "ALL") & (S["mode"] == m)].iloc[0] for m in MODES if ((S["cond"] == "ALL") & (S["mode"] == m)).any()}
    res["mode_alone"] = {m: json.loads(base_mode[m].to_json()) for m in base_mode}
    for m in base_mode:
        _log(f"[{MODE_NAMES[m]}] レース {int(base_mode[m]['races'])} 頭/レース {base_mode[m]['mean_n']:.2f} 単勝 レース均等 {base_mode[m]['roi_race_w']:.3f} 最上位人気 {base_mode[m]['roi_fav_w']:.3f}")
    # reality check ① 『回収率 100% 超』（基準 = 1.0）。② 『モード単独より上がるか』（基準 = モード単独の回収率）
    one = {(b, s): 1.0 for b in ("w", "p") for s in ("horse", "race", "fav")}
    nm, o, dt, r1 = _rc(rt, races, one, seed=3)
    top = np.argsort(-np.where(dt >= 300, o, -9))[:12]
    res["rc_over100"] = {"rules_tried": int((dt >= 300).sum()), **r1, "best_rule": [names[nm[r1['best']][0]][0], names[nm[r1['best']][0]][1], nm[r1['best']][1], nm[r1['best']][2]],
                         "top": [{"cond": names[nm[i][0]][0], "mode": names[nm[i][0]][1], "bet": nm[i][1], "scheme": nm[i][2], "excess": float(o[i]), "den": float(dt[i])} for i in top]}
    _log(f"RC①(100%超): 規則 {res['rc_over100']['rules_tried']} 最良 {res['rc_over100']['best_rule']} 超過 {res['rc_over100']['best_excess']:+.3f} p={res['rc_over100']['p']:.3f}")
    inc_tried, inc_best = 0, None
    res["rc_incremental"] = {}
    for mode in MODES:
        if mode not in base_mode:
            continue
        rtm = rt[rt.index.get_level_values(0).map(lambda k: names[k][1] == mode and names[k][0] != "ALL")]
        if len(rtm) == 0:
            continue
        bases = RL.base_rois(dv[MK[mode]].reset_index(drop=True))
        nm2, o2, dt2, r2 = _rc(rtm, races, bases, seed=5)
        top2 = np.argsort(-np.where(dt2 >= 300, o2, -9))[:6]
        res["rc_incremental"][mode] = {"rules_tried": int((dt2 >= 300).sum()), **r2, "best_rule": [names[nm2[r2['best']][0]][0], nm2[r2['best']][1], nm2[r2['best']][2]],
                                       "top": [{"cond": names[nm2[i][0]][0], "bet": nm2[i][1], "scheme": nm2[i][2], "excess": float(o2[i]), "den": float(dt2[i])} for i in top2],
                                       "base": {f"{b}_{s}": float(v) for (b, s), v in bases.items()}}
        _log(f"RC②[{MODE_NAMES[mode]}]: 規則 {res['rc_incremental'][mode]['rules_tried']} 最良 {res['rc_incremental'][mode]['best_rule']} 超過 {res['rc_incremental'][mode]['best_excess']:+.3f} p={res['rc_incremental'][mode]['p']:.3f}")
    # 上位の条件（単勝・レース均等 / 最上位人気 の回収率の高い順）。100% 超の区間下限があるか
    S2 = S[S["races"] >= 300].copy()
    res["max_lower_bound"] = {}
    for col in ("roi_race_w", "roi_fav_w", "roi_race_p", "roi_fav_p", "roi_horse_w"):
        lo = S2[f"{col}_lo"]
        i = lo.idxmax()
        res["max_lower_bound"][col] = {"rule": f"{S2.loc[i, 'cond']}|{S2.loc[i, 'mode']}", "lo": float(lo.max()), "roi": float(S2.loc[i, col]), "races": int(S2.loc[i, "races"])}
        res.setdefault("n_lower_ge_1", {})[col] = int((lo >= 1.0).sum())
    _log("区間下限の最大: " + json.dumps(res["max_lower_bound"], ensure_ascii=False))
    res["n_lower_ge_1"]["rules"] = int(len(S2))
    # 年別（上位 25 規則 + モード単独）。2025 は参考
    top_rules = S2.sort_values("roi_race_w", ascending=False).head(25).index.tolist() + [f"ALL|{m}" for m in base_mode]
    yearly = []
    for lab in dict.fromkeys(top_rules):
        cname, mode = names[lab]
        for y in range(YEARS_SEL[0], 2026):
            dy = d[d["year"] == y].reset_index(drop=True)
            Cy, MKy = conditions(dy), mode_masks(dy)
            cm = np.ones(len(dy), dtype=bool) if cname == "ALL" else Cy[cname]
            s = cm & MKy[mode]
            if s.sum() < 30:
                continue
            ry = RL.race_table(dy, np.where(s, "x", None).astype(object))
            Sy = RL.summarize(ry).iloc[0]
            yearly.append({"rule": lab, "year": y, "races": int(Sy["races"]), "roi_race_w": float(Sy["roi_race_w"]), "roi_fav_w": float(Sy["roi_fav_w"]), "roi_race_p": float(Sy["roi_race_p"]),
                           "race_win": float(Sy["race_win"]), "exp_win": float(Sy["exp_win"]), "ref_only": bool(y == 2025)})
    res["yearly"] = yearly
    res["summary"] = json.loads(S2.reset_index().rename(columns={"index": "rule", "_k": "rule"}).to_json(orient="records", force_ascii=False))
    M._jsave(OUT / "bet.json", res)
    _log("saved bet.json")


# ───────────────────────── followup（上位に出た条件が EV の水準をまたいで一貫するか。年別・2025 は参考）
FOLLOW_CONDS = ("型=A1", "型=A(加速)", "短評=目立たず", "追い切りランクD", "直前3日に強め以上")
EV_BINS = ((0.8, 1.0), (1.0, 1.2), (1.2, 99.0))


def cmd_followup(a) -> None:
    d0 = load_frame()
    d = RL.prep(d0)
    d = d[d["p_win"].notna() & d["win_odds"].notna() & (d["year"] >= 2022)].reset_index(drop=True)
    C = conditions(d)
    wo = d["win_odds"].to_numpy(dtype=float)
    evs = {"model": d["p_win"].to_numpy(dtype=float) * wo, "blend": d["p_blend"].to_numpy(dtype=float) * wo}
    yr = d["year"].to_numpy()
    rng = np.random.default_rng(21)
    rows = []
    for cname in FOLLOW_CONDS:
        if cname not in C:
            continue
        for evname, ev in evs.items():
            for lo, hi in EV_BINS:
                inbin = np.isfinite(ev) & (ev >= lo) & (ev < hi)
                for part, years in (("2022-24", (2022, 2024)), ("2025(参考)", (2025, 2025))):
                    ym = (yr >= years[0]) & (yr <= years[1])
                    for flag in (True, False):
                        s = inbin & ym & (C[cname] == flag)
                        if s.sum() < 50:
                            continue
                        rt = RL.race_table(d, np.where(s, "x", None).astype(object))
                        S = RL.summarize(rt).iloc[0]
                        sub = rt
                        nu, de = RL._metric_arrays(sub)["roi_race_w"]
                        _, lo_, hi_ = RL._ratio_boot(nu, de, rng)
                        rows.append({"cond": cname, "ev": evname, "bin": f"{lo}-{hi}", "part": part, "cond_true": flag, "races": int(S["races"]), "horses": int(S["horses"]),
                                     "roi_race_w": float(S["roi_race_w"]), "lo": float(lo_), "hi": float(hi_), "roi_fav_w": float(S["roi_fav_w"]), "roi_race_p": float(S["roi_race_p"]), "race_win": float(S["race_win"]), "exp_win": float(S["exp_win"])})
    M._jsave(OUT / "followup.json", rows)
    R = pd.DataFrame(rows)
    for (c, e, b), g in R.groupby(["cond", "ev", "bin"]):
        t_ = g[g["cond_true"]].set_index("part")
        f_ = g[~g["cond_true"]].set_index("part")
        _log(f"{c} / {e} EV{b}: " + " ".join(f"{p}: 該当 {t_.loc[p, 'roi_race_w']:.2f}({int(t_.loc[p, 'races'])}R) 非該当 {f_.loc[p, 'roi_race_w']:.2f}({int(f_.loc[p, 'races'])}R)" for p in ("2022-24", "2025(参考)") if p in t_.index and p in f_.index))


# ───────────────────────── final2026（2022〜25 年のデータから事前に決めた 1 つの方策を、未使用の 2026 年で 1 回だけ評価）
def _race_equal(new: pd.DataFrame, mask: np.ndarray, rng, nb: int = 2000) -> dict:
    sub = new[mask]
    if len(sub) == 0:
        return {"races": 0}
    g = sub.assign(r=sub["ret_win"] / 100.0).groupby("race_id").agg(n=("r", "size"), mean=("r", "mean"), win=("win", "max"), fav=("win_odds", "min"))
    roi_race = g["mean"].to_numpy()
    n = len(roi_race)
    est = float(roi_race.mean())
    idx = rng.integers(0, n, (nb, n))
    bs = roi_race[idx].mean(1)
    hs = sub.assign(r=sub["ret_win"] / 100.0)
    return {"races": int(n), "horses": int(len(sub)), "mean_n": float(len(sub) / n), "roi_race": est, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "race_hit": float((g["win"] > 0).mean()), "roi_horse": float(hs["r"].sum() / len(hs))}


def cmd_final2026(a) -> None:
    from src.pipeline.features import tp35_lap_groups as L35
    from src.scripts.data import tp36_final2026 as F26
    from src.scripts.data.tp36_edge import blend, fit_benter

    TP = M.BET / "tp36"
    old = pd.read_parquet(TP / "edge_frame.parquet")
    old = old[old["year"] >= 2022]
    new = F26.build_frame2026(TP / "pred2026", E.ROOT / "data/page_reference/tables/2026/race_result_flat.parquet")
    ab = fit_benter(old)                                                       # 混合の重みは 2022〜25 年だけで当てはめる
    new["p_blend"] = blend(new, *ab)
    tp = L35.prepare_laps(pd.read_parquet(E.LAPS))
    Fs = L35.pick_slot(tp, L35.SLOT_F, "F")
    new["race_date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
    new["horse_id"] = new["horse_id"].astype(str)
    new = new.merge(Fs[["horse_id", "race_date", "F_cls"]], on=["horse_id", "race_date"], how="left")
    new["win_odds"] = pd.to_numeric(new["win_odds"], errors="coerce")
    ev_b = new["p_blend"] * new["win_odds"]
    ev_m = new["p_win"] * new["win_odds"]
    a1 = (new["F_cls"] == "A1").to_numpy()
    rng = np.random.default_rng(31)
    pol = OrderedDict([
        ("混合EV≥1.0 かつ 型=A1（2022〜25 年から事前に決めた方策）", (ev_b >= 1.0).to_numpy() & a1),
        ("混合EV 1.0〜1.2 かつ 型=A1", ((ev_b >= 1.0) & (ev_b < 1.2)).to_numpy() & a1),
        ("対照: 混合EV≥1.0（型を問わない）", (ev_b >= 1.0).to_numpy()),
        ("対照: 混合EV≥1.0 かつ 型≠A1", (ev_b >= 1.0).to_numpy() & ~a1 & new["F_cls"].notna().to_numpy()),
        ("対照: モデルEV≥1.2 かつ 型=A1", (ev_m >= 1.2).to_numpy() & a1),
        ("対照: 型=A1 のみ", a1),
    ])
    out = {"benter_ab": ab, "races": int(new["race_id"].nunique()), "rows": int(len(new)), "a1_share_of_rows": float(a1.mean()), "has_cls_share": float(new["F_cls"].notna().mean()),
           "policies": {k: _race_equal(new, m, rng) for k, m in pol.items()}}
    M._jsave(OUT / "final2026.json", out)
    for k, v in out["policies"].items():
        _log(f"{k}: {v}")


# ───────────────────────── tilt（V4 の確率を、コメントの数個のフラグで傾け直す校正層。GBM を再学習しない）
TILT_FLAGS = ("neg", "pos", "sensen", "okure", "yusei", "ressei")


def tilt_matrix(d: pd.DataFrame) -> np.ndarray:
    net = d["fc_net"].to_numpy(dtype=float)
    cols = [np.nan_to_num(net <= -1).astype(float), np.nan_to_num(net >= 1).astype(float)] + [np.nan_to_num(d[f"fc_{k}"].to_numpy(dtype=float)) for k in ("sensen", "okure", "yusei", "ressei")]
    return np.column_stack(cols)


def _race_groups(d: pd.DataFrame):
    codes, _ = pd.factorize(d["race_id"])
    order = np.argsort(codes, kind="stable")
    return codes, order


def fit_tilt(logp: np.ndarray, X: np.ndarray, win: np.ndarray, codes: np.ndarray, lam: float = 5.0) -> np.ndarray:
    """勝者の条件付き対数尤度（softmax(log p + θ·x)）の最大化。θ の L2 罰則 lam。"""
    from scipy.optimize import minimize

    nR = codes.max() + 1
    k = X.shape[1]

    def f(th):
        z = logp + X @ th
        m = np.full(nR, -np.inf)
        np.maximum.at(m, codes, z)
        e = np.exp(z - m[codes])
        den = np.bincount(codes, weights=e, minlength=nR)
        p = e / den[codes]
        ll = np.log(np.clip(p[win == 1], 1e-12, 1)).sum()
        grad = X[win == 1].sum(0) - (X * p[:, None]).sum(0)
        return -ll + lam * (th @ th), -grad + 2 * lam * th

    r = minimize(f, np.zeros(k), jac=True, method="L-BFGS-B")
    return r.x


def tilted(logp: np.ndarray, X: np.ndarray, th: np.ndarray, codes: np.ndarray) -> np.ndarray:
    nR = codes.max() + 1
    z = logp + X @ th
    m = np.full(nR, -np.inf)
    np.maximum.at(m, codes, z)
    e = np.exp(z - m[codes])
    return e / np.bincount(codes, weights=e, minlength=nR)[codes]


def cmd_tilt(a) -> None:
    d0 = load_frame()
    d = RL.prep(d0)
    d = d[d["p_win"].notna() & (d["year"] >= 2022)].copy()
    ok_races = d.groupby("race_id")["win"].transform("sum") == 1
    d = d[ok_races].reset_index(drop=True)                                              # コメントの無い馬も入れる（fc_net が NaN → フラグ 0）
    s = d.groupby("race_id")["p_win"].transform("sum")
    d = d[(s > 0.97) & (s < 1.03)].reset_index(drop=True)
    d["p_win"] = d["p_win"] / d.groupby("race_id")["p_win"].transform("sum")
    X = tilt_matrix(d)
    logp = np.log(np.clip(d["p_win"].to_numpy(dtype=float), 1e-9, 1))
    win = d["win"].to_numpy()
    yr = d["year"].to_numpy()
    rng = np.random.default_rng(5)
    res = {"flags": list(TILT_FLAGS), "coverage_comment": float(d["fc_net"].notna().mean()), "folds": {}}
    for ey, train_years in ((2023, (2022,)), (2024, (2022, 2023)), (2025, (2022, 2023, 2024))):
        tr = np.isin(yr, train_years)
        te = yr == ey
        codes_tr, _ = pd.factorize(d.loc[tr, "race_id"])
        th = fit_tilt(logp[tr], X[tr], win[tr], codes_tr)
        codes_te, races = pd.factorize(d.loc[te, "race_id"])
        p1 = tilted(logp[te], X[te], th, codes_te)
        w = win[te] == 1
        base_ll = -np.log(np.clip(np.exp(logp[te])[w], 1e-9, 1))
        new_ll = -np.log(np.clip(p1[w], 1e-9, 1))
        diff = new_ll - base_ll
        nrace = len(diff)
        idx = rng.integers(0, nrace, (2000, nrace))
        bs = diff[idx].mean(1)
        res["folds"][str(ey)] = {"theta": dict(zip(TILT_FLAGS, map(float, th))), "races": int(nrace), "base_logloss": float(base_ll.mean()), "tilted_logloss": float(new_ll.mean()),
                                 "delta": float(diff.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "ref_only": ey == 2025,
                                 "n_flagged_horses": int((X[te].sum(1) > 0).sum())}
        _log(f"傾け直し {ey}（学習 {train_years}）: θ={dict(zip(TILT_FLAGS, np.round(th, 3)))} 勝率 LogLoss {base_ll.mean():.5f} → {new_ll.mean():.5f}（差 {diff.mean():+.5f} [{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}]）")
    M._jsave(OUT / "tilt.json", res)


def _jl(name: str):
    return M._jload(OUT / name)


def _f(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x:+.{nd}f}"


def _roi(r, k, nd=1):
    v = r.get(k)
    lo, hi = r.get(f"{k}_lo"), r.get(f"{k}_hi")
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v * 100:.{nd}f}%" + (f" [{lo * 100:.0f}–{hi * 100:.0f}]" if lo is not None and hi is not None and np.isfinite(lo) else "")


def cmd_report(a) -> None:
    """結果の要約を標準出力に出す（章の本文を書く前の確認用）。"""
    s1, s2 = _jl("stage1/results.json").get("groups", {}), _jl("stage2/results.json").get("groups", {})
    print("stage1:", {g: (round(v["vs_V4"]["Δwin"], 5), round(v["vs_V4"]["win_hi"], 5)) for g, v in s1.items()})
    print("stage2:", {g: (round(v["vs_V4"]["Δwin"], 5), round(v["vs_V4"]["win_hi"], 5), v["vs_V4"]["採用"]) for g, v in s2.items()})



DEFINITIONS_HTML = """<ul>
<li><b>調教師ごとの時計の標準化（TRZ・TRZ_SELF・CLZ）</b>。時計は 3 種類: 4F の総時計 t4、終い 1F t1、加速度 acc（= 終い 1F − 直前 2F の 1F 平均。負 = 加速）。
まず（種別 × 所属 × 強度 × 馬場）ごとの『その追い切りの日より前の追い切り』の平均・標準偏差で標準化する（z0。小さい = 速い。TL_A の標準化は強度までで馬場を含まず、直前 12 か月の窓）。
次に調教師ごとの『いつもの速さ』dev を、調教師 × 種別 × 強度の z0 の過去平均として、調教師全体の平均（縮約の強さ 30）→ 全体（15）へ階層縮約して出す。過去の追い切りが 20 本未満の調教師は欠損にする。
調教師比 ztr = z0 − dev、ztrs = ztr ÷ 調教師の過去のばらつき（縮約）。<b>TRZ</b> は最終追い切り・1 週前の z0 / dev / ztr / ztrs と 14 日の集約（30 列）、<b>TRZ_SELF</b> は馬自身の過去・前走との差（6 列）。
<b>CLZ</b> は調教師を『時計の出し方』（強め以上の割合・一杯の割合・坂路の割合・t4/t1/加速度の標準化平均）で 5 クラスタに分け（2020 年末までの調教師で k-means、番号は強めの割合の小さい順）、クラスタ平均を縮約の親にした版 zh と、クラスタ番号を足す（11 列）。</li>
<li><b>交互作用（I1〜I11）</b>。追い切りの型（A1〜B3）・調教師比の時計の 5 階級・調教師の時計の出し方クラスタ・強度・追い切りの日数を、レース条件（面 × 距離帯、場 × 面 × 距離、クラス、脚質 4 区分）と組にして、日付が厳密に前のレースの成績（勝・複勝・着順残差）を階層縮約した as-of の lift にする。各 4 列。</li>
<li><b>本数・間隔・前走比（D1・D2）</b>。前のレースから今回までの追い切り本数・始まりの早さ・日数の間隔・直前 3 日の強め以上、最終追い切りの標準化時計・強度・本数・日数の前走との差、前走と同じ調教コースか。</li>
<li><b>追い切り騎手（D3）</b>。追い切りに乗った人の区分（助手・師・見習・騎手名あり）と、騎手名ごとの『調教師比の時計』の過去平均（癖）。追い切りの騎手は名前しかなく ID が無い。名前は追い切り表の中のカテゴリとしてだけ使い、レースの騎手とは結合しない。</li>
<li><b>短評・追い切りランク・曜日・走路位置（D4・D5）</b>。追い切りごとの短評（『順調』『動き軽快』など、語彙は 2022 年より前の出現数で決める）と A〜D のランク、調教師 × 曜日、種別 × 走路位置の as-of の lift。</li>
<li><b>記者コメント（D7・D8）</b>。馬別調教のページには、最終週の追い切りに『記者コメント』が付いている（追い切り全体の約 8%、出走馬の約 30%）。従来の追い切り表には落としていたので、読み取り専用で取り出した（<code>tp37_comments</code>）。
前日夕方（eve）の取得にも同じ割合で載っていることを 2026-10-18 の出走馬で確認した。<b>D7_text_flags</b> = 語の有無（好意・否定の語の数、併せ馬の『先着・遅れ・同入・優勢・劣勢・追走・併せ』、『初戦・叩き』）、<b>D7_text_lift</b> = コメントの最後の一文（『好仕上がり』『平行線』『初戦から』などの定型の見立て）と好意否定の区分の as-of の lift、
<b>D8_text_ctx</b> = 調教師ごとのコメントの口調との差、馬自身の前走・過去のコメントとの差。生のコメント文は公開リポジトリに入れない（保存先は data/local）。</li>
</ul>"""

EXTRA = {
    "summary": """追い切りの残りの余地を V4（1,062 列）に対して全部試したが、<b>採用基準を満たした特徴量の群は無かった</b>。試したのは追加 22 群（調教師ごとの時計の標準化 3、追い切り × 条件の交互作用 11、本数・間隔・前走比 2、追い切り騎手・短評・曜日 3、記者コメント 3）と、TL_A の時計列 15 列を置き換える 2 通り。
第 1 段（valid 2022+2023）を通ったのは 7 件（TRZ_SELF、CLZ、I2、記者コメントの語と最後の一文、D1+D2、D3+D4+D5）だが、第 2 段（valid 2024・新 seed）では勝率が +0.00019〜−0.00011（上限はいずれも +0.00013 以上）に戻り、すべて不採用。調教師ごとの標準化は既存の TL_A と相関 0.92〜0.99 で、ほぼ同じ情報だった。<br>
<b>新しく分かったこと.</b> ① 馬別調教ページには『記者コメント』が最終週の追い切りに付き（出走馬の約 30%。前日夕方の取得にも同じ割合で載る）、否定的なコメントは V4 の勝ち予測を相対 20% 過大にし（残差 −0.0068/頭、p=0.0007、3 年同符号）、併せ馬で先着の記述は 20% 過小にしていた（+0.0157/頭、p=0.018）。ただし対象が少なく、特徴量にしても傾け直しても V4 への上乗せは基準未満。
② 馬券側は、V4 の予測 × 追い切りの条件で絞った 1,512 規則に、reality check を通るものは無い（p=0.22）。『型=A1（終い 1F だけ 12 秒台で加速）かつ混合 EV≥1.0』は 2023〜24 年 144%（95% 区間 101〜212%、654 レース）、2025 年（参考）133%、2026 年（1 回のみ）120%（42〜215%、107 レース）と一貫して 100% を超えるが、多重比較の補正後は偶然を排除できず（p=0.32）、2026 年は型を問わない混合 EV≥1.0（116%）や型≠A1（128%）と差がない。""",
    "final2026": """<p>『混合 EV≥1.0 かつ 型=A1』を 2022〜25 年のデータだけから事前に固定し（混合の重みも 2022〜25 年で当てはめた）、未使用の 2026 年を評価した。最初の評価は旧い追い切り表（最終追い切りの型が分かる行 31%）で行い、同じ方策を再取得後の追い切り表（34%）で 1 回やり直した（方策は変えていない）。以下は後者（前者の方策は 90 レース・126%・区間 37〜239%）。対照を同時に示す。</p>""",
    "final2026_note": "2026 年の被覆が低い原因（馬別調教ページに 2026 年前半の追い切りが十分に載っていない可能性）は未調査。型が分かる馬だけの結果なので、どの方策も区間は広く、判断には使えない（方策間の差は偶然の範囲）。",
    "conclusion": """<ul>
<li><b>特徴量</b>: 追い切りの残りの余地は、V4 に足して勝率の LogLoss を下げる形では見つからなかった。第 1 段（2 年・seed 1 本）で通っても第 2 段で戻る群が 7 件あり、第 1 段の通過は偶然を含むと見る（試した群数 24、第 1 段の基準は緩い）。調教師ごとの標準化・クラスタ・交互作用は、TL_A と V4 の既存列（調教師の力量・コース適性）に吸収されている。</li>
<li><b>記者コメント</b>: 新しい情報源として確かに V4 の残差と関連する（否定的 −20%、併せ馬で先着 +20%、3 年同符号、BH 通過）が、効くのは出走馬の一部（約 30% × 稀な記述）で、特徴量にしても（第 1 段 −0.0003 → 第 2 段 +0.0001、プール −0.00014〜−0.00018）、傾け直しても（勝ち馬の対数損失 +0.0005／−0.0011／−0.0003、区間は 0 を跨ぐ）基準の 0.0002 に届かない。v5 の再学習の材料にはしない。コメントの取得は前日夕方の取得に載っているので、将来 V4 の残差補正を作る場合に使える。</li>
<li><b>複勝率</b>: D1（本数・間隔）が第 1 段（D1+D2）と第 2 段（D1）で複勝の LogLoss を −0.0012〜−0.0017 下げたが、区間の上限は 0 を跨ぐ（+0.00018、+0.00005）。複勝専用の判断としては保留。</li>
<li><b>馬券</b>: 追い切りの条件を V4 の予測に重ねても、区間下限が 100% を超える規則は reality check を通らない。最有望は『型=A1 × 混合 EV≥1.0』だが、2026 年で上乗せが確認できなかった。実オッズ（T-60/T-15）の蓄積と 2026 年のデータが増えた後に、この 1 規則だけを再検証する。</li>
<li><b>tp-40（調教サイクル内の推移）との重複</b>: 本数・間隔・前走比（D1・D2）と馬自身の過去・前走との標準化時計の差（TRZ_SELF）は、調教サイクル内の推移を見る点で同じ情報の系統。ここでは第 2 段で再現しなかったので、CY が通る場合も別の列を足す根拠にはならない（CY を V4 に足す評価で重複を再確認する）。</li>
<li><b>運用上の発見</b>: 馬別調教のコメントは eve（前日夕方）の取得で約 34%（10/18 の出走馬 450 頭中 151 頭）の馬に載っている。追い切りの短評・ランクは全頭に載る。</li>
</ul>""",
}

CHAPTER = E.ROOT / "docs/html/modeling/chapters/workout_remaining_ideas.html"


def cmd_chapter(a) -> None:
    import html as H

    bm, cov, rs, bt, fu = _jl("build_meta.json"), _jl("coverage.json"), _jl("resid.json"), _jl("bet.json"), _jl("followup.json")
    f26, f26o = _jl("final2026.json"), _jl("final2026_oldlaps.json")
    s1d, s2d = _jl("stage1/results.json"), _jl("stage2/results.json")
    s1, s2 = s1d.get("groups", {}), s2d.get("groups", {})
    extra = {**EXTRA, **_jl("chapter_extra.json")}
    head = open(E.ROOT / "docs/html/modeling/chapters/lap_groups_and_trainer_stats.html", encoding="utf-8").read().split("<body>")[0]
    head = head.replace("ラップ種別と調教師別・コース別の調教成績（tp-35）", "追い切りの残りの余地（調教師ごとの標準化・交互作用・本数/間隔・騎手・短評コメント）と馬券への応用（tp-37）")
    order = list(s1.keys()) + [g for g in s2 if g not in s1]
    rows = ""
    for g in order:
        a1, b1 = s1.get(g), s2.get(g)
        c1, c2 = (a1 or {}).get("vs_V4", {}), (b1 or {}).get("vs_V4", {})
        cols = (a1 or b1)["cols"]
        verdict = ("<b>採用</b>" if c2.get("採用") else "不採用") if b1 else "第 1 段で不通過"
        rows += (f"<tr><td>{H.escape(g)}</td><td class=num>{cols}</td><td class=num>{_f(c1.get('Δwin'), 5)}（{_f(c1.get('win_hi'), 5)}）</td><td class=num>{_f(c1.get('Δtop2'), 5)}</td><td class=num>{_f(c1.get('Δtop3'), 5)}</td>"
                 f"<td class=num>{_f(c2.get('Δwin'), 5) if b1 else '-'}（{_f(c2.get('win_hi'), 5) if b1 else '-'}）</td><td class=num>{_f(c2.get('Δtop3'), 5) if b1 else '-'}</td><td>{verdict}</td></tr>")
    base1 = s1d.get("base", {}).get("levels", {})
    base2 = s2d.get("base", {}).get("levels", {})
    # 残差
    rsum = rs.get("summary", {})
    rres = "".join(f"<tr><td>{H.escape(r['cond'])}</td><td class=num>{r['races']:,}</td><td class=num>{r['mean_n']:.2f}</td><td class=num>{r['resid_per_horse']:+.4f}（{r['lo']:+.4f}〜{r['hi']:+.4f}）</td><td class=num>{r['rel_to_p'] * 100:+.1f}%</td><td class=num>{r['p']:.4f}</td>"
                   f"<td class=num>{' / '.join('-' if v is None else f'{v:+.4f}' for v in r['years'].values())}</td><td>{'BH10 通過' if r.get('bh10') else ''}</td></tr>" for r in rsum.get("top", [])[:12])
    tilt = _jl("tilt.json")
    trows = "".join(f"<tr><td>{k}{'（参考）' if v.get('ref_only') else ''}</td><td class=num>{v['races']:,}</td><td class=num>{v['base_logloss']:.5f} → {v['tilted_logloss']:.5f}</td><td class=num>{v['delta']:+.5f}（{v['lo']:+.5f}〜{v['hi']:+.5f}）</td>"
                    f"<td>{', '.join(f'{a} {b:+.2f}' for a, b in v['theta'].items())}</td></tr>" for k, v in tilt.get("folds", {}).items())
    # 馬券
    ma = bt.get("mode_alone", {})
    mrows = "".join(f"<tr><td>{H.escape(MODE_NAMES.get(m, m))}</td><td class=num>{int(v['races']):,}</td><td class=num>{v['mean_n']:.2f}</td><td class=num>{v['race_win'] * 100:.1f}% / {v['exp_win'] * 100:.1f}%</td>"
                    f"<td class=num>{_roi(v, 'roi_race_w')}</td><td class=num>{_roi(v, 'roi_fav_w')}</td><td class=num>{_roi(v, 'roi_race_p')}</td></tr>" for m, v in ma.items())
    summ = pd.DataFrame(bt.get("summary", []))
    top_rows = ""
    if len(summ):
        summ["lo"] = summ["roi_race_w_lo"]
        for _, r in summ.sort_values("roi_race_w_lo", ascending=False).head(10).iterrows():
            top_rows += (f"<tr><td>{H.escape(r['cond'])}</td><td>{H.escape(MODE_NAMES.get(r['mode'], r['mode']))}</td><td class=num>{int(r['races']):,}</td><td class=num>{r['mean_n']:.2f}</td>"
                         f"<td class=num>{_roi(r, 'roi_race_w')}</td><td class=num>{_roi(r, 'roi_fav_w')}</td><td class=num>{_roi(r, 'roi_race_p')}</td></tr>")
    rc1 = bt.get("rc_over100", {})
    rc2 = "".join(f"<tr><td>{H.escape(MODE_NAMES.get(m, m))}</td><td class=num>{v['rules_tried']}</td><td>{H.escape(' / '.join(map(str, v['best_rule'])))}</td><td class=num>{v['best_excess']:+.3f}</td><td class=num>{v['p']:.3f}</td></tr>" for m, v in bt.get("rc_incremental", {}).items())
    yr = pd.DataFrame(bt.get("yearly", []))
    yrows = ""
    if len(yr):
        for rule, g in yr.groupby("rule", sort=False):
            if g["races"].max() < 150:
                continue
            cells = " ".join(f"{int(r.year)}{'*' if r.ref_only else ''}: {r.roi_race_w * 100:.0f}%({int(r.races)}R)" for r in g.itertuples())
            yrows += f"<tr><td>{H.escape(rule.replace('|', ' ｜ '))}</td><td>{cells}</td></tr>"
            if yrows.count("<tr>") >= 14:
                break
    furows = ""
    if fu:
        fdf = pd.DataFrame(fu)
        for (c, e, b), g in fdf.groupby(["cond", "ev", "bin"], sort=False):
            t_, f_ = g[g["cond_true"]].set_index("part"), g[~g["cond_true"]].set_index("part")
            def cell(part):
                if part in t_.index and part in f_.index:
                    return (f"該当 {t_.loc[part, 'roi_race_w'] * 100:.0f}% [{t_.loc[part, 'lo'] * 100:.0f}–{t_.loc[part, 'hi'] * 100:.0f}]（{int(t_.loc[part, 'races'])}R）／ "
                            f"非該当 {f_.loc[part, 'roi_race_w'] * 100:.0f}%（{int(f_.loc[part, 'races'])}R）")
                return "-"
            furows += f"<tr><td>{H.escape(c)}</td><td>{'混合' if e == 'blend' else 'モデル'}</td><td>{b}</td><td>{cell('2022-24')}</td><td>{cell('2025(参考)')}</td></tr>"
    f26rows = ""
    for k, v in f26.get("policies", {}).items():
        f26rows += f"<tr><td>{H.escape(k)}</td><td class=num>{v.get('races', 0):,}</td><td class=num>{v.get('mean_n', float('nan')):.2f}</td><td class=num>{v.get('roi_race', float('nan')) * 100:.1f}% [{v.get('lo', float('nan')) * 100:.0f}–{v.get('hi', float('nan')) * 100:.0f}]</td><td class=num>{v.get('roi_horse', float('nan')) * 100:.1f}%</td></tr>"
    cov_rows = "".join(f"<tr><td>{H.escape(k)}</td><td class=num>{v['mean']:+.3f}</td><td class=num>{v['std']:.3f}</td><td class=num>{v['p05']:+.2f}</td><td class=num>{v['p95']:+.2f}</td></tr>" for k, v in cov.get("trz_stats", {}).items())
    corr_rows = "".join(f"<tr><td>{H.escape(k)}</td><td class=num>{v:+.3f}</td></tr>" for k, v in cov.get("corr_with_tl", {}).items())
    n_groups = len(order)
    body = f"""<body><a href="../../index.html" class="muted">← ドキュメント一覧</a><h1>追い切りの残りの余地と馬券への応用（tp-37）</h1>
<p class="muted">追い切りの分析（tp-19 / tp-28 / tp-35）で残っていた余地を、V4 全体（1,062 列）に対して全部試した。調教師ごとの時計の標準化（と、時計の出し方のクラスタ）、追い切り × コース・距離・面・クラス・脚質の交互作用、本数・間隔・前走比、追い切り騎手、短評・記者コメント、そして V4 の予測と追い切りの条件を組み合わせた馬券。すべて as-of（レース日より前の情報だけ）。市場オッズ・人気は特徴量に使っていない（馬券の判断にだけ使う）。</p>
<div class="note"><b>要点.</b> {extra.get('summary', '（結論は結果の確定後に書く）')}</div>
<h2>1. 何を作ったか</h2>
{DEFINITIONS_HTML}
<h2>2. 被覆と基本統計</h2>
<p>標準化した時計（小さい = 速い）の最終追い切りの分布。調教師比 = 全体の基準で標準化した値 − その調教師の『いつもの速さ』（同じ種別・強度の過去の追い切りの平均、階層縮約）。</p>
<table><tr><th>列</th><th>平均</th><th>標準偏差</th><th>5%点</th><th>95%点</th></tr>{cov_rows}</table>
<table><tr><th>新しい列 ~ 既存の TL_A の列</th><th>相関</th></tr>{corr_rows}</table>
<p class="muted">新しい標準化は既存の TL_A の標準化と相関 0.92〜0.99 で、ほぼ同じ情報を別の基準で見ているだけ（調教師比にしても 0.92〜0.94 が残る）。</p>
<h2>3. 特徴量としての増分（V4 全体に対する勝率 LogLoss の差、95% CI 上限）</h2>
<p>基準 = V4（1,062 列）。第 1 段 = valid 2022+2023・seed 1 本（V4 の勝率 LogLoss {base1.get('logloss_win', float('nan')):.5f}）、第 2 段 = 通過群を valid 2024・新 seed 2 本で確認（V4 の勝率 LogLoss {base2.get('logloss_win', float('nan')):.5f}）。試した群は 24（追加 22 群と、TL_A の時計列 15 列を置き換える 2 通り。下の表には合成した ALL_TEXT・ALL[5]・バッチの行も含む）。採用の基準は『CI 上限 &lt; 0・効果 0.0002 以上・連対と複勝の悪化は 0.0002 以内』。交互作用 I3〜I11 と D1〜D5 は、ホストの負荷の制約で 3〜4 群ずつのバッチ（BATCH[…]）にして第 1 段にかけ、通過したバッチだけ構成群を個別に（と、バッチ全体を）第 2 段にかけた。</p>
<table><tr><th>群</th><th>列数</th><th>第 1 段 Δwin（上限）</th><th>Δtop2</th><th>Δtop3</th><th>第 2 段 Δwin（上限）</th><th>第 2 段 Δtop3</th><th>採否</th></tr>{rows}</table>
<h2>4. V4 の勝ち残差は追い切りの条件で説明できるか（レース単位）</h2>
<p>V4 の OOF 予測（valid 2022〜24、leave-one-year-out）の『実際の勝ち − 予測確率』を、条件に当てはまる馬がいるレースを 1 単位としてブートストラップ（{rsum.get('boot', '-')} 回）。条件は {rsum.get('base_conditions', '-')} 通りと、その一部 × 距離・面の組（芝/ダ × 距離帯）を足した合計 {rsum.get('cells_tested', '-')} セル（件数 300 頭以上）。</p>
<p>生の p&lt;0.05 は {rsum.get('p_lt_005_raw', '-')} セル（偶然の期待 {rsum.get('expected_by_chance', float('nan')):.1f}）、BH（FDR 10%）を通ったのは {rsum.get('bh10', '-')} セル、うち 3 年同符号 {rsum.get('bh10_and_same_sign', '-')} セル。</p>
<table><tr><th>条件</th><th>レース</th><th>1R平均頭数</th><th>残差/頭（95% 区間）</th><th>予測に対する相対</th><th>p</th><th>年別（2022/23/24）</th><th></th></tr>{rres}</table>
<h3>4-1. コメントのフラグで V4 の確率を傾け直す（再学習なしの校正）</h3>
<p>V4 の勝ち確率に、コメントの 6 個のフラグ（否定的・好意的・先着・遅れ・優勢・劣勢）で exp(θ·x) を掛けてレース内で正規化する。θ は『その年より前の年の OOF』だけで当てはめる（2023 年は 2022 年、2024 年は 2022〜23 年、2025 年は 2022〜24 年。2025 年は参考）。下の LogLoss は『勝ち馬の対数確率（レース単位）』で、馬単位の二値 LogLoss（0.21 前後）とは尺度が違う（おおよそ頭数 14 で割ると馬単位に近い）。コメントがある出走馬は全体の {tilt.get('coverage_comment', float('nan')) * 100:.0f}%。</p>
<table><tr><th>評価年</th><th>レース</th><th>勝ち馬の対数損失 基準 → 傾け後</th><th>差（95% 区間）</th><th>θ</th></tr>{trows}</table>
<h2>5. V4 の予測 × 追い切りの条件で絞った馬券（レース単位）</h2>
<p>選択と集計は 2022〜24 年（混合は 2023〜24 年）。絞り込み = 条件のみ / モデルの EV（予測勝率 × 確定単勝オッズ）≥1.0 / ≥1.2 / 混合（Benter 型）EV≥1.0 / モデル上位 3 頭。条件 {bt.get('conditions', '-')} 通り × 絞り込み 5 通り。回収率は単勝・複勝それぞれで『レース均等』（該当馬を均等額で全頭）と『最上位人気 1 頭』。下は絞り込みだけ（条件なし）の回収率。</p>
<table><tr><th>絞り込み</th><th>レース</th><th>1R平均頭数</th><th>レース勝率 / 期待</th><th>単勝 レース均等</th><th>単勝 最上位人気</th><th>複勝 レース均等</th></tr>{mrows}</table>
<p>回収率の区間下限が 100% 以上の規則: {', '.join(f'{k}={v}' for k, v in bt.get('n_lower_ge_1', {}).items())}（規則 = 条件 × 絞り込みの組で {bt.get('n_lower_ge_1', {}).get('rules', '-')} 通り）。</p>
<h3>5-1. 区間下限の高い規則の上位（単勝・レース均等の下限順）</h3>
<table><tr><th>条件</th><th>絞り込み</th><th>レース</th><th>1R平均頭数</th><th>単勝 レース均等</th><th>単勝 最上位人気</th><th>複勝 レース均等</th></tr>{top_rows}</table>
<h3>5-2. reality check（多重比較の補正）</h3>
<p>① 『回収率が 100% を超える規則があるか』: 規則 {rc1.get('rules_tried', '-')} 通り（条件 × 絞り込み × 単複 × 買い方）。最良は {H.escape(' / '.join(map(str, rc1.get('best_rule', []))))}（超過 {rc1.get('best_excess', float('nan')):+.3f}）で p = {rc1.get('p', float('nan')):.3f}。</p>
<p>② 『絞り込み単独より回収率が上がるか』（基準 = 同じ絞り込みで条件なしの回収率）:</p>
<table><tr><th>絞り込み</th><th>規則</th><th>最良の規則</th><th>超過</th><th>p</th></tr>{rc2}</table>
<h3>5-3. 年別（2025 は参考。選択には使っていない）</h3>
<table><tr><th>規則（条件 ｜ 絞り込み）</th><th>年別の単勝回収率（レース均等）</th></tr>{yrows}</table>
<h3>5-4. 上位に出た条件が EV の水準をまたいで一貫するか</h3>
<table><tr><th>条件</th><th>EV</th><th>EV の帯</th><th>2022〜24 年（単勝・レース均等）</th><th>2025 年（参考）</th></tr>{furows}</table>
<h2>6. 未使用の 2026 年での 1 回だけの評価</h2>
{extra.get('final2026', '')}
<table><tr><th>方策（2022〜25 年のデータから事前に固定）</th><th>レース</th><th>1R平均頭数</th><th>単勝 レース均等（95% 区間）</th><th>単勝 馬単位</th></tr>{f26rows}</table>
<p class="muted">2026 年の追い切りの被覆は {f26.get('has_cls_share', float('nan')) * 100:.0f}%（最終追い切りの型が分かる行の割合）。{extra.get('final2026_note', '')}</p>
<h2>7. 結論と今後</h2>
{extra.get('conclusion', '')}
</body></html>"""
    CHAPTER.write_text(head + body, encoding="utf-8")
    _log(f"章を書いた: {CHAPTER}")
