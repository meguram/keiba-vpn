"""tp-60 血統を使った馬券の規則と、現行の期待値（V4 の確率 × 確定オッズ）への補正・絞り込み。

python -m src.scripts.data.tp60_rules rules      # 規則の網羅（選ぶのは各年より前のデータだけ）。2022〜2024 で選び、2025 は確認、2026 はテストセット②
python -m src.scripts.data.tp60_rules evfilter   # EV≥閾値の買い目に、血統セルの成績（過去）で絞り込み・補正を足したときの回収率・対数損失

購入: 該当馬を 100 円ずつ（単勝 or 複勝）。回収率は確定オッズ・確定払戻での値（購入時点のオッズとは違う）。
選抜バイアスの扱い: 規則の数（試行数）を数え、2022〜2024 のレース単位の損益で reality check（White）の p を出す。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.scripts.data import tp60_lib as L
from src.scripts.data.tp60_roi import K_SHRINK, _save

VAL_YEARS = [2022, 2023, 2024]
CONFIRM = 2025
TEST2 = 2026

KEYCONDS = [("sire_id", "all"), ("dsire_id", "all"), ("ssire_id", "all"), ("sire_dsire", "all"), ("sire_id", "sd"), ("dsire_id", "sd"),
            ("sire_line", "sd"), ("dsire_line", "sd"), ("line_pair", "sd"), ("sire_id", "going"), ("sire_id", "pop_band")]
NMIN = {"all": 80, "sd": 40, "going": 40, "pop_band": 40}
FRACS = [0.05, 0.10, 0.20]
ODDS_FILTERS = {"none": (0.0, 1e9), "le30": (0.0, 30.0), "p10to60": (10.0, 60.0)}   # 単勝オッズの範囲（選ぶ対象の絞り込み。購入時点では確定オッズ前の値に近いもので置き換わる）


def _cells(m: pd.DataFrame, key: str, cond: str):
    d = m.dropna(subset=[key])
    c = L.collapse(d, key, cond)
    c["cell"] = c["key"].astype(str) + "#" + c["cond"].astype(str)
    cell_row = pd.Series(d[key].astype(str).to_numpy() + "#" + d[cond].astype(str).to_numpy(), index=d.index)
    return c, cell_row


def rule_masks(m: pd.DataFrame, years: list[int]) -> dict[str, np.ndarray]:
    """規則名 → 買う馬の mask（m の行位置）。選抜は『その年より前』の累積成績。"""
    out: dict[str, np.ndarray] = {}
    yr = m["year"].to_numpy()
    for key, cond in KEYCONDS:
        c, cell_row = _cells(m, key, cond)
        nmin = NMIN[cond] if cond != "all" or key not in ("sire_dsire",) else 40
        cell_all = pd.Series(cell_row.reindex(m.index).to_numpy(), index=m.index)
        for bt, scol in (("win", "rw"), ("place", "rp")):
            for frac in FRACS:
                sel_rows = np.zeros(len(m), dtype=bool)
                for y in years:
                    pri = L.agg_cells(c[c["year"] < y])
                    pri["cell"] = pri["key"].astype(str) + "#" + pri["cond"].astype(str)
                    pri = pri[pri["horses"] >= nmin]
                    if len(pri) < 20:
                        continue
                    mu = pri[scol].sum() / pri["horses"].sum()
                    s = L.eb_shrink(pri[scol].to_numpy(), pri["horses"].to_numpy(), mu, K_SHRINK)
                    thr = np.quantile(s, 1 - frac)
                    chosen = set(pri.loc[s >= thr, "cell"])
                    sel_rows |= (yr == y) & cell_all.isin(chosen).to_numpy()
                for of, (lo, hi) in ODDS_FILTERS.items():
                    msk = sel_rows & (m["win_odds"].to_numpy() >= lo) & (m["win_odds"].to_numpy() <= hi)
                    out[f"{key}@{cond}|{bt}|top{int(frac * 100)}|{of}"] = msk
    return out


def bet_vectors(m: pd.DataFrame, masks: dict[str, np.ndarray], years: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """規則ごとの『レースごとの損益（円）』ベクトル（対象年のレース順）と、そのレース ID。"""
    sub = m[m["year"].isin(years)]
    codes, races = pd.factorize(sub["race_id"])
    k = len(races)
    prof = {}
    pos = np.flatnonzero(m["year"].isin(years).to_numpy())
    for name, msk in masks.items():
        bt = name.split("|")[1]
        col = "ret_win" if bt == "win" else "ret_place"
        mm = msk[pos]
        v = (sub[col].to_numpy(dtype=float) - 100.0) * mm
        prof[name] = np.bincount(codes, weights=v, minlength=k)
    return prof, races


def n_bets(m: pd.DataFrame, msk: np.ndarray, years: list[int]) -> int:
    return int((msk & m["year"].isin(years).to_numpy()).sum())


def reality_check(P: np.ndarray, B: int = 2000, seed: int = L.RNG_SEED) -> dict:
    """White の reality check（t 値で正規化した最大値）。P: レース × 規則 の損益。H0: どの規則も期待損益が 0 を超えない。tp-59 と同じ形式。"""
    n, K = P.shape
    mu = P.mean(0)
    sd = P.std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.inf)
    t = np.sqrt(n) * mu / sd
    rng = np.random.default_rng(seed)
    tmax = np.empty(B)
    for b in range(B):
        pb = P[rng.integers(0, n, n)]
        sdb = pb.std(0, ddof=1)
        sdb = np.where(sdb > 1e-12, sdb, np.inf)
        tmax[b] = (np.sqrt(n) * (pb.mean(0) - mu) / sdb).max()
    return {"V": float(t.max()), "p": float((tmax >= t.max()).mean()), "K": int(K), "best_idx": int(t.argmax()), "t_max": float(t.max())}


def cmd_rules(a) -> None:
    m = L.load_master()
    L.log(f"規則の mask を作る（{len(KEYCONDS)} キー × 2 券種 × {len(FRACS)} × {len(ODDS_FILTERS)}）")
    masks = rule_masks(m, VAL_YEARS + [CONFIRM, TEST2])
    names = list(masks)
    L.log(f"試行数 {len(names)}")
    prof_val, races_val = bet_vectors(m, masks, VAL_YEARS)
    P = np.column_stack([prof_val[n] for n in names])
    rc = reality_check(P)
    res: dict = {"n_trials": len(names), "rc_val": rc, "rules": {}}
    # 規則ごとの要約（2022〜2024 の合計、2025、2026）
    sub = m
    rows = []
    for n in names:
        msk = masks[n]
        bt = n.split("|")[1]
        col = "ret_win" if bt == "win" else "ret_place"
        e = {}
        for lab, ys in (("val", VAL_YEARS), ("2025", [CONFIRM]), ("2026", [TEST2])):
            mm = msk & sub["year"].isin(ys).to_numpy()
            k = int(mm.sum())
            e[lab] = {"bets": k, "roi": float(sub.loc[mm, col].sum() / (100.0 * k)) if k else np.nan, "hit": float((sub.loc[mm, col] > 0).mean()) if k else np.nan}
        rows.append((n, e))
    rows.sort(key=lambda t: -(t[1]["val"]["roi"] if np.isfinite(t[1]["val"]["roi"]) and t[1]["val"]["bets"] >= 300 else -9))
    # 上位 15 件は区間を付ける（val / 2025 / 2026）
    for n, e in rows[:15]:
        msk = masks[n]
        bt = n.split("|")[1]
        col = "ret_win" if bt == "win" else "ret_place"
        for lab, ys in (("val", VAL_YEARS), ("2025", [CONFIRM]), ("2026", [TEST2])):
            mm = (msk & sub["year"].isin(ys).to_numpy())
            ci = L.roi_ci(sub, mm, col, B=1000)
            e[lab]["lo"], e[lab]["hi"], e[lab]["race_roi"] = ci["lo"], ci["hi"], ci["roi"]
        # 最大ドローダウン（val の損益の累積）
        pv = prof_val[n]
        cum = np.cumsum(pv)
        e["max_dd_yen"] = float((np.maximum.accumulate(cum) - cum).max())
    res["top_by_val"] = [{"rule": n, **e} for n, e in rows[:15]]
    # 全規則の分布
    vr = np.array([e["val"]["roi"] for _, e in rows if e["val"]["bets"] >= 300 and np.isfinite(e["val"]["roi"])])
    cr = np.array([e["2025"]["roi"] for _, e in rows if e["val"]["bets"] >= 300 and e["2025"]["bets"] >= 100 and np.isfinite(e["2025"]["roi"])])
    res["dist"] = {"val_roi_quantiles": {str(q): float(np.quantile(vr, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)}, "n_val": int(len(vr)),
                   "roi2025_median": float(np.median(cr)), "n_2025": int(len(cr)), "share_val_ge_100": float((vr >= 1.0).mean()),
                   "share_val_top10pct_then_2025_ge_100": None}
    top_idx = [n for n, e in rows if e["val"]["bets"] >= 300][:10]
    res["top10_by_val_then_2025"] = [{"rule": n, "val_roi": e["val"]["roi"], "roi2025": e["2025"]["roi"], "roi2026": e["2026"]["roi"], "bets_val": e["val"]["bets"], "bets_2025": e["2025"]["bets"]}
                                     for n, e in rows if n in top_idx]
    res["bets_val_min"] = 300
    _save("rules.json", res)
    L.log(f"試行 {len(names)} 規則、RC p={rc['p']:.3f}、val ROI 分位 {res['dist']['val_roi_quantiles']}")


def _logloss_win(p: np.ndarray, y: np.ndarray, race_ids: np.ndarray) -> float:
    """勝率のレース内正規化後の対数損失（勝ち馬の確率 −log）。勝ち馬のいるレースのみ。"""
    d = pd.DataFrame({"r": race_ids, "p": np.clip(p, 1e-6, None), "y": y})
    d["pn"] = d["p"] / d.groupby("r")["p"].transform("sum")
    w = d[d["y"] == 1]
    return float(-np.log(w["pn"]).mean())


def cmd_evfilter(a) -> None:
    mf = L.load_master()
    m = mf[mf["p_win"].notna() & (mf["year"] >= 2022)].reset_index(drop=True)
    m["ev_win"] = m["p_win"] * m["win_odds"]
    m["ev_place"] = m["p_top3"] * m["place_eff"]
    out: dict = {"thresholds": {}, "recal": {}}
    # 血統セルの成績（過去・縮約）: 各馬の行に『その年より前の累積』で付ける（年単位の更新）。回収率の縮約は 2020 年からの全データ、V4 の偏りは p のある 2022 年以降
    stats = {}
    for key, cond in (("sire_id", "all"), ("dsire_id", "all"), ("sire_id", "sd"), ("dsire_id", "sd"), ("sire_line", "sd"), ("line_pair", "sd")):
        c = L.collapse(mf.dropna(subset=[key]), key, cond)
        c["cell"] = c["key"].astype(str) + "#" + c["cond"].astype(str)
        nm = f"{key}@{cond}"
        col_roi, col_bias = np.full(len(m), np.nan), np.full(len(m), np.nan)
        col_roi_p = np.full(len(m), np.nan)
        cell_all = pd.Series(m[key].astype(str).to_numpy() + "#" + m[cond].astype(str).to_numpy(), index=m.index).where(m[key].notna())
        for y in (2023, 2024, 2025, 2026):
            pri = L.agg_cells(c[c["year"] < y])
            pri["cell"] = pri["key"].astype(str) + "#" + pri["cond"].astype(str)
            mu_w = pri["rw"].sum() / pri["horses"].sum()
            mu_p = pri["rp"].sum() / pri["horses"].sum()
            pri["roi_s"] = L.eb_shrink(pri["rw"].to_numpy(), pri["horses"].to_numpy(), mu_w, 120.0) / 100.0
            pri["roip_s"] = L.eb_shrink(pri["rp"].to_numpy(), pri["horses"].to_numpy(), mu_p, 120.0) / 100.0
            pb = L.agg_cells(c[(c["year"] >= 2022) & (c["year"] < y)])
            pb["cell"] = pb["key"].astype(str) + "#" + pb["cond"].astype(str)
            pb["bias"] = (pb["win"] + 60.0) / (pb["p"] + 60.0)          # 1 が偏りなし
            mp, mpp, mb = dict(zip(pri["cell"], pri["roi_s"])), dict(zip(pri["cell"], pri["roip_s"])), dict(zip(pb["cell"], pb["bias"]))
            idx = np.flatnonzero((m["year"] == y).to_numpy())
            cs = cell_all.iloc[idx]
            col_roi[idx] = cs.map(mp).to_numpy(dtype=float)
            col_roi_p[idx] = cs.map(mpp).to_numpy(dtype=float)
            col_bias[idx] = cs.map(mb).to_numpy(dtype=float)
        stats[nm] = (col_roi, col_roi_p, col_bias)
    yrs_eval = [2023, 2024, 2025]
    # 1) EV≥T の買い目に、血統セルの縮約回収率が『中央値以上』/『下位 25% を除く』で絞った効果
    for bt, evc, retc, T in (("win", "ev_win", "ret_win", 1.0), ("win", "ev_win", "ret_win", 1.2), ("place", "ev_place", "ret_place", 1.0), ("place", "ev_place", "ret_place", 1.1)):
        base = (m[evc] >= T).to_numpy()
        ent = {"T": T, "bt": bt, "base": {}, "filters": {}}
        for lab, ys in (("2023-2024", [2023, 2024]), ("2025", [2025]), ("2026", [2026])):
            mm = base & m["year"].isin(ys).to_numpy()
            ci = L.roi_ci(m, mm, retc, B=800, race_equal=False)
            ent["base"][lab] = ci
        for nm, (roi_w, roi_p, bias) in stats.items():
            col = roi_w if bt == "win" else roi_p
            for fname, fmask in (("上位半分", col >= np.nanmedian(col)), ("下位25%を除く", col >= np.nanquantile(col, 0.25)), ("下位25%のみ", col < np.nanquantile(col, 0.25))):
                sel = base & np.nan_to_num(fmask, nan=False).astype(bool) & ~np.isnan(col)
                f = {}
                for lab, ys in (("2023-2024", [2023, 2024]), ("2025", [2025]), ("2026", [2026])):
                    yy = m["year"].isin(ys).to_numpy()
                    elig = base & yy & ~np.isnan(col)
                    d = L.boot_diff(m, sel & yy, elig, retc, B=800)
                    f[lab] = {k: d[k] for k in ("sel_roi", "all_roi", "diff", "diff_lo", "diff_hi", "p_le0", "n_sel") if k in d}
                ent["filters"][f"{nm}:{fname}"] = f
        out["thresholds"][f"{bt}_T{T}"] = ent
    # 2) V4 の勝率の偏りを血統セルで補正した確率の対数損失（as-of）
    for nm, (roi_w, roi_p, bias) in stats.items():
        res = {}
        for y in yrs_eval:
            yy = (m["year"] == y).to_numpy() & ~np.isnan(bias)
            d = m[yy]
            p0 = d["p_win"].to_numpy()
            ll0 = _logloss_win(p0, d["win"].to_numpy(), d["race_id"].to_numpy())
            best = None
            for lam in (0.25, 0.5, 1.0):
                p1 = p0 * np.power(bias[yy], lam)
                ll1 = _logloss_win(p1, d["win"].to_numpy(), d["race_id"].to_numpy())
                if best is None or ll1 < best[1]:
                    best = (lam, ll1)
            # 全 lam の結果も
            res[str(y)] = {"ll_v4": ll0, "best_lambda": best[0], "ll_adj": best[1], "delta": best[1] - ll0,
                           **{f"delta_l{lam}": _logloss_win(p0 * np.power(bias[yy], lam), d["win"].to_numpy(), d["race_id"].to_numpy()) - ll0 for lam in (0.25, 0.5, 1.0)}}
        out["recal"][nm] = res
    _save("evfilter.json", out)
    L.log("evfilter 完了")


K_RES = 1000.0
TAUS = (1.0, 1.1, 1.2, 1.3, 1.5)


def cmd_evfinal(a) -> None:
    """tp-59 の ev_final と同じ形式: V4 の EV に『市場に対する残差が正』の条件を重ねる規則の族（血統・騎手・調教師）を reality check にかける。"""
    mf = L.load_master()
    m = mf[(mf["year"] >= 2022) & mf["p_win"].notna()].reset_index(drop=True)
    m["ev_win"] = m["p_win"] * m["win_odds"]
    m["ev_place"] = m["p_top3"] * m["place_eff"]
    for nm, key in (("s", "sire_id"), ("d", "dsire_id"), ("j", "jockey_id"), ("t", "trainer_id"), ("l", "line_pair")):
        m[f"{nm}_res_pos"] = np.nan
        for y in range(2022, 2027):
            h = mf[mf["year"] < y].dropna(subset=[key])
            g = h.groupby(key).agg(n=("win", "size"), w=("win", "sum"), q=("q_win", "sum"))
            r = (g["w"] - g["q"]) / (g["n"] + K_RES)
            cur = (m["year"] == y).to_numpy()
            m.loc[cur, f"{nm}_res_pos"] = (m.loc[cur, key].map(r) > 0).astype(float)
            m.loc[cur & m[key].map(r).isna().to_numpy(), f"{nm}_res_pos"] = np.nan
    f = lambda c: (m[c] == 1)
    conds = {"none": np.ones(len(m), dtype=bool), "s": f("s_res_pos").to_numpy(), "d": f("d_res_pos").to_numpy(), "sd": (f("s_res_pos") & f("d_res_pos")).to_numpy(),
             "s_or_d": (f("s_res_pos") | f("d_res_pos")).to_numpy(), "line": f("l_res_pos").to_numpy(),
             "jt": (f("j_res_pos") & f("t_res_pos")).to_numpy(), "jt+s": (f("j_res_pos") & f("t_res_pos") & f("s_res_pos")).to_numpy(),
             "jt+d": (f("j_res_pos") & f("t_res_pos") & f("d_res_pos")).to_numpy(), "jt+sd": (f("j_res_pos") & f("t_res_pos") & f("s_res_pos") & f("d_res_pos")).to_numpy()}
    codes, uniq = pd.factorize(m["race_id"])
    sel = m["year"].isin(VAL_YEARS).to_numpy()
    sel_races = np.unique(codes[sel])
    names, masks = [], []
    for bet, evc in (("win", "ev_win"), ("place", "ev_place")):
        for tau in TAUS:
            for cn, cm in conds.items():
                names.append((bet, tau, cn))
                masks.append((m[evc] >= tau).to_numpy() & cm)
    prof = np.zeros((len(uniq), len(names)))
    for i, ((bet, _, _), mk) in enumerate(zip(names, masks)):
        mm = mk & sel
        np.add.at(prof[:, i], codes[mm], m.loc[mm, f"ret_{bet}"].to_numpy() - 100.0)
    rc = reality_check(prof[sel_races])
    rows = []
    for i, ((bet, tau, cn), mk) in enumerate(zip(names, masks)):
        retc = f"ret_{bet}"
        row = {"bet": bet, "tau": tau, "cond": cn}
        for lab, ys in (("sel_2022_2024", VAL_YEARS), ("2025", [CONFIRM]), ("2026", [TEST2])):
            mm = mk & m["year"].isin(ys).to_numpy()
            if mm.sum() < 20:
                row[lab] = {"n": int(mm.sum())}
                continue
            ci = L.roi_ci(m, mm, retc, B=1000, race_equal=False)
            row[lab] = {"n": ci["n"], "roi": ci["roi"], "lo": ci["lo"], "hi": ci["hi"], "hit": float((m.loc[mm, retc] > 0).mean())}
        rows.append(row)
    # 増分: 騎手・調教師の条件（jt）を満たす EV 買い目の中で、血統の条件（s, d, sd）の有無による回収率の差
    inc = []
    for bet, evc in (("win", "ev_win"), ("place", "ev_place")):
        for tau in ((1.2, 1.5) if bet == "win" else (1.0, 1.3)):
            base = (m[evc] >= tau).to_numpy() & conds["jt"]
            for cn in ("s", "d", "sd", "s_or_d"):
                for lab, ys in (("2022-2024", VAL_YEARS), ("2025", [CONFIRM]), ("2026", [TEST2])):
                    yy = m["year"].isin(ys).to_numpy()
                    elig = base & yy & ~m[["s_res_pos", "d_res_pos"]].isna().any(axis=1).to_numpy()
                    d_ = L.boot_diff(m, conds[cn] & elig, elig, f"ret_{bet}", B=800)
                    inc.append({"bet": bet, "tau": tau, "cond": cn, "period": lab, **{k: d_.get(k) for k in ("sel_roi", "all_roi", "diff", "diff_lo", "diff_hi", "p_le0", "n_sel")}})
    _save("evfinal.json", {"n_rules": len(names), "rc": rc, "best": rows[rc["best_idx"]], "rows": rows, "increment_over_jt": inc})
    L.log(f"evfinal: 規則 {len(names)}、RC t_max={rc['t_max']:.2f} p={rc['p']:.3f}、最良 {rows[rc['best_idx']]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rules")
    sub.add_parser("evfilter")
    sub.add_parser("evfinal")
    a = ap.parse_args()
    {"rules": cmd_rules, "evfilter": cmd_evfilter, "evfinal": cmd_evfinal}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
