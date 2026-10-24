"""tp-59 (4): 騎手・調教師の過去の回収率で買う規則 / 推奨 EV に騎手・調教師の条件を重ねる効果。

python -m src.scripts.data.tp59_rules scan      # 規則の族（関係者 11 種 × 指標 × 件数下限 × 上位 K）を全部評価。選択年 2022〜2024 の Reality Check、2025 年の確認、2026 年の最終確認
python -m src.scripts.data.tp59_rules ev        # V4 の EV に条件を重ねたときの差（選択年 / 2025 / 2026）

選び方は『当年より前の全期間（2020 年〜前年）の成績』だけで決める（as-of。当年の結果は使わない）。選択年は 2022〜2024 年。2025 年は確認で 1 回、2026 年はテストセット②として最後に 1 回。
回収率は 1 出走（1 買い目）100 円。区間はレース単位ブートストラップ。市場オッズは買い目の評価（払戻・EV）にだけ使い、特徴量には使わない。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp59_lib as L
from src.scripts.data import tp59_profile as P

OUT = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
SEL = (2022, 2023, 2024)
FAMILIES = ["j", "t", "jt", "jv", "jsd", "tv", "tsd", "tc", "jo", "to", "jc"]
N_LIST = (30, 60, 120)
K_LIST = (5, 10, 20, 50)
SCORES = ("roi", "ex")           # roi = 払戻の縮約平均、ex = オッズ帯の期待払戻を引いた超過
BETS = ("win", "place")
K_SHR = 100.0


def add_keys(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["surf_dist"] = d["surface"].astype(str) + "_" + d["dist_band"].astype(str)
    d["og"] = np.select([d["win_odds"] < 5, d["win_odds"] < 15], ["fav", "mid"], "long")
    j, t = d["jockey_id"].astype(str), d["trainer_id"].astype(str)
    d["k_j"], d["k_t"] = j, t
    d["k_jt"] = j + "|" + t
    d["k_jv"] = j + "|" + d["venue_code"].astype(str)
    d["k_jsd"] = j + "|" + d["surf_dist"]
    d["k_tv"] = t + "|" + d["venue_code"].astype(str)
    d["k_tsd"] = t + "|" + d["surf_dist"]
    d["k_tc"] = t + "|" + d["cls"].astype(str)
    d["k_jo"] = j + "|" + d["og"]
    d["k_to"] = t + "|" + d["og"]
    d["k_jc"] = j + "|" + d["cls"].astype(str)
    return d


def family_rank(d: pd.DataFrame, fam: str, year: int, bet: str, score: str, n_min: int) -> pd.Series:
    """year の各出走について、その関係者セルの順位（0 が最良、対象外は NaN）。成績は year より前だけ。"""
    key = f"k_{fam}"
    h = d[d["year"] < year]
    col_ret, col_ex = f"ret_{bet}", f"ex_{bet}"
    g = h.groupby(key).agg(n=("win", "size"), s=(col_ret, "sum"), e=(col_ex, "sum"))
    if fam in ("jo", "to"):
        og = g.index.str.rsplit("|", n=1).str[-1]
        prior = h.groupby("og")[col_ret].mean().reindex(og).to_numpy()
    else:
        prior = np.full(len(g), h[col_ret].mean())
    if score == "roi":
        g["score"] = (g["s"] + K_SHR * prior) / (100.0 * (g["n"] + K_SHR))
    else:
        g["score"] = g["e"] / (100.0 * (g["n"] + K_SHR))
    g = g[g["n"] >= n_min].sort_values("score", ascending=False)
    rk = pd.Series(np.arange(len(g)), index=g.index)
    cur = d[d["year"] == year]
    return cur[key].map(rk)


def rule_name(fam, bet, score, n_min, k) -> str:
    return f"{fam}|{bet}|{score}|n{n_min}|k{k}"


def scan() -> dict:
    d = P.load()
    d = add_keys(d)
    d = d[d["year"] >= 2020].reset_index(drop=True)
    race_codes, race_uniq = pd.factorize(d["race_id"])
    d["rc"] = race_codes
    years = list(SEL) + [2025, 2026]
    rules: dict[str, dict] = {}
    profit_rows: dict[str, np.ndarray] = {}
    base = {}
    for y in years:
        cur = d[d["year"] == y]
        base[y] = {b: L.boot_roi(cur["race_id"].to_numpy(), cur[f"ret_{b}"].to_numpy(), 300) for b in BETS}
    n_races = len(race_uniq)
    for fam, bet, score, n_min in itertools.product(FAMILIES, BETS, SCORES, N_LIST):
        ranks = {y: family_rank(d, fam, y, bet, score, n_min) for y in years}
        for k in K_LIST:
            name = rule_name(fam, bet, score, n_min, k)
            per_year = {}
            prof = np.zeros(n_races)
            for y in years:
                cur = d[d["year"] == y]
                m = (ranks[y] < k).to_numpy()
                sel = cur[m]
                if len(sel) == 0:
                    per_year[y] = {"n": 0}
                    continue
                ret = sel[f"ret_{bet}"].to_numpy()
                per_year[y] = {"n": int(len(sel)), "roi": float(ret.sum() / (100.0 * len(sel))) * 100, "hit": float((ret > 0).mean())}
                if y in SEL:
                    np.add.at(prof, sel["rc"].to_numpy(), ret - 100.0)
            rules[name] = {"fam": fam, "bet": bet, "score": score, "n_min": n_min, "k": k, "per_year": per_year}
            profit_rows[name] = prof
    names = list(rules)
    sel_races = np.unique(d.loc[d["year"].isin(SEL), "rc"].to_numpy())
    M = np.stack([profit_rows[n][sel_races] for n in names], 1)
    rc = L.reality_check(M, n_boot=2000)
    for i, n in enumerate(names):
        rules[n]["t_sel"] = float(rc["t"][i])
        rules[n]["profit_sel"] = float(M[:, i].sum())
    # 族ごとの RC（その族の中で最良を選んだ場合の p）
    fam_rc = {}
    for fam in FAMILIES:
        idx = [i for i, n in enumerate(names) if rules[n]["fam"] == fam]
        fam_rc[fam] = {k: v for k, v in L.reality_check(M[:, idx], n_boot=1000).items() if k != "t"}
        fam_rc[fam]["best_rule"] = names[idx[int(fam_rc[fam]["best"])]]
    ranked = sorted(names, key=lambda n: -rules[n]["t_sel"])
    # 確認用: 全体の上位 10 と各族の最良
    confirm = list(dict.fromkeys(ranked[:10] + [fam_rc[f]["best_rule"] for f in FAMILIES]))
    detail = {}
    for n in confirm:
        r = rules[n]
        fam, bet, score, n_min, k = r["fam"], r["bet"], r["score"], r["n_min"], r["k"]
        detail[n] = {}
        for y_name, ys in (("sel_2022_2024", SEL), ("2025", (2025,)), ("2026", (2026,))):
            rows = []
            for y in ys:
                cur = d[d["year"] == y]
                rk = family_rank(d, fam, y, bet, score, n_min)
                rows.append(cur[(rk < k).to_numpy()])
            s = pd.concat(rows)
            if len(s) == 0:
                detail[n][y_name] = {"n": 0}
                continue
            ret = s[f"ret_{bet}"].to_numpy()
            detail[n][y_name] = {"pooled": L.boot_roi(s["race_id"].to_numpy(), ret, 1000), "race_equal": L.race_equal_roi(s["race_id"].to_numpy(), ret, 1000),
                                 "hit": float((ret > 0).mean()), "dd": L.max_drawdown(s["date"].to_numpy(), s["race_id"].to_numpy(), ret - 100.0)}
            # 同じ期間に同じ券種を全出走に買った回収率との差
            allp = pd.concat([d[d["year"] == y] for y in ys])
            detail[n][y_name]["diff_vs_all"] = L.boot_diff(s["race_id"].to_numpy(), ret, allp["race_id"].to_numpy(), allp[f"ret_{bet}"].to_numpy(), 500)
    out = {"n_rules": len(names), "reality_check": {k: v for k, v in rc.items() if k != "t"}, "best_rule": names[rc["best"]], "family_rc": fam_rc,
           "top10": [{"rule": n, "t_sel": rules[n]["t_sel"], "per_year": rules[n]["per_year"]} for n in ranked[:10]],
           "confirm": detail, "base": base,
           "dist_t": {"mean": float(np.mean(rc["t"])), "q05": float(np.quantile(rc["t"], 0.05)), "q95": float(np.quantile(rc["t"], 0.95)), "frac_t_gt_2": float(np.mean(rc["t"] > 2))},
           "all_rules": {n: {"t_sel": rules[n]["t_sel"], "per_year": rules[n]["per_year"]} for n in names}}
    (OUT / "rules.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


# ------------------------------------------------------------------ EV に条件を重ねる
TAUS = (1.0, 1.1, 1.2, 1.3, 1.5)
K_RES = 1000.0
K_EX = 500.0


def add_asof_conditions(d: pd.DataFrame) -> pd.DataFrame:
    """当年より前の成績だけで、騎手・調教師の条件を各出走に付ける（True/False）。"""
    d = d.copy()
    for c in ("j_str_hi", "t_str_hi", "j_ex_pos", "t_ex_pos", "jt_ex_pos", "j_res_pos", "t_res_pos", "j_up", "j_down", "j_same"):
        d[c] = np.nan
    for y in range(2022, 2027):
        h = d[d["year"] < y]
        cur = d["year"] == y
        for nm, key in (("j", "jockey_id"), ("t", "trainer_id")):
            g = h.groupby(key).agg(n=("win", "size"), w=("win", "sum"), ex=("ex_win", "sum"), q=("q_win", "sum"))
            r0 = g["w"].sum() / g["n"].sum()
            strength = (g["w"] + P.K_WIN * r0) / (g["n"] + P.K_WIN)
            d.loc[cur, f"{nm}_str"] = d.loc[cur, key].map(strength)
            if nm == "j":
                jstr = strength
            d.loc[cur, f"{nm}_ex_pos"] = (d.loc[cur, key].map(g["ex"] / (100.0 * (g["n"] + K_EX))) > 0).astype(float)
            d.loc[cur, f"{nm}_res_pos"] = (d.loc[cur, key].map((g["w"] - g["q"]) / (g["n"] + K_RES)) > 0).astype(float)
        pair = h.assign(k=h["jockey_id"].astype(str) + "|" + h["trainer_id"].astype(str)).groupby("k").agg(n=("win", "size"), ex=("ex_win", "sum"))
        kk = d.loc[cur, "jockey_id"].astype(str) + "|" + d.loc[cur, "trainer_id"].astype(str)
        d.loc[cur, "jt_ex_pos"] = (kk.map(pair["ex"] / (100.0 * (pair["n"] + K_EX))) > 0).astype(float)
        prev = d.loc[cur, "prev_jockey_id"].map(jstr)
        up = d.loc[cur, "j_str"] - prev
        chg = d.loc[cur, "chg_jockey"]
        d.loc[cur, "j_up"] = ((chg == 1) & (up > 0.01)).astype(float)
        d.loc[cur, "j_down"] = ((chg == 1) & (up < -0.01)).astype(float)
        d.loc[cur, "j_same"] = (chg == 0).astype(float)
    for nm in ("j", "t"):
        med = d.groupby("year")[f"{nm}_str"].transform("median")
        d[f"{nm}_str_hi"] = (d[f"{nm}_str"] >= med).astype(float).where(d["year"] >= 2022)
    return d


CONDS = ["j_str_hi", "t_str_hi", "j_ex_pos", "t_ex_pos", "jt_ex_pos", "j_res_pos", "t_res_pos", "j_up", "j_down", "j_same"]


def ev_overlay() -> dict:
    d = P.load()
    d = add_asof_conditions(d)
    d = d[(d["year"] >= 2022) & d["p_win_raw"].notna()].copy()
    d["ev_win"] = d["p_win_raw"] * d["win_odds"]
    d["ev_place"] = d["p_top3"] * d["place_eff"]
    periods = {"sel_2022_2024": d[d["year"].isin(SEL)], "2025": d[d["year"] == 2025], "2026": d[d["year"] == 2026]}
    out: dict = {"taus": list(TAUS), "conds": CONDS, "base": {}, "overlay": []}
    for bet, evc, retc in (("win", "ev_win", "ret_win"), ("place", "ev_place", "ret_place")):
        for tau in TAUS:
            for per, g in periods.items():
                s = g[g[evc] >= tau]
                out["base"][f"{bet}|{tau}|{per}"] = {**L.boot_roi(s["race_id"].to_numpy(), s[retc].to_numpy(), 500), "hit": float((s[retc] > 0).mean()) if len(s) else None}
                for c in CONDS:
                    s_in = s[s[c] == 1]
                    s_out = s[s[c] == 0]
                    if len(s_in) < 30 or len(s_out) < 30:
                        continue
                    dif = L.boot_diff(s_in["race_id"].to_numpy(), s_in[retc].to_numpy(), s_out["race_id"].to_numpy(), s_out[retc].to_numpy(), 500)
                    out["overlay"].append({"bet": bet, "tau": tau, "period": per, "cond": c, "n_in": int(len(s_in)), "n_out": int(len(s_out)),
                                           "roi_in": float(100 * s_in[retc].mean() / 100.0), "roi_out": float(100 * s_out[retc].mean() / 100.0), **dif})
    (P.OUT / "ev_overlay.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


def ev_final() -> dict:
    """V4 の EV に『騎手 / 調教師の市場に対する残差が正（過去の勝ち数が市場の見込みを上回る）』を重ねた規則の族（2 券種 × τ 5 通り × 条件 4 通り + 条件なし）を、
    100% を超えるかの Reality Check（選択年 2022〜2024）にかけ、2025・2026 を別に見る。"""
    d = P.load()
    d = add_asof_conditions(d)
    d = d[(d["year"] >= 2022) & d["p_win_raw"].notna()].copy().reset_index(drop=True)
    d["ev_win"] = d["p_win_raw"] * d["win_odds"]
    d["ev_place"] = d["p_top3"] * d["place_eff"]
    d["j_or_t_res"] = ((d["j_res_pos"] == 1) | (d["t_res_pos"] == 1)).astype(float)
    d["j_and_t_res"] = ((d["j_res_pos"] == 1) & (d["t_res_pos"] == 1)).astype(float)
    conds = {"none": None, "j_res_pos": "j_res_pos", "t_res_pos": "t_res_pos", "either": "j_or_t_res", "both": "j_and_t_res"}
    rc_codes, rc_uniq = pd.factorize(d["race_id"])
    d["rc"] = rc_codes
    names, flags = [], []
    for bet, evc in (("win", "ev_win"), ("place", "ev_place")):
        for tau in TAUS:
            for cn, col in conds.items():
                m = (d[evc] >= tau) & ((d[col] == 1) if col else True)
                names.append((bet, tau, cn))
                flags.append(m.to_numpy())
    sel_mask = d["year"].isin(SEL).to_numpy()
    sel_races = np.unique(d.loc[sel_mask, "rc"].to_numpy())
    pos = pd.Series(np.arange(len(rc_uniq)))
    prof = np.zeros((len(rc_uniq), len(names)))
    for i, ((bet, _, _), m) in enumerate(zip(names, flags)):
        mm = m & sel_mask
        np.add.at(prof[:, i], d.loc[mm, "rc"].to_numpy(), d.loc[mm, f"ret_{bet}"].to_numpy() - 100.0)
    rc = L.reality_check(prof[sel_races], n_boot=2000)
    rows = []
    for i, ((bet, tau, cn), m) in enumerate(zip(names, flags)):
        row = {"bet": bet, "tau": tau, "cond": cn, "t_sel": float(rc["t"][i])}
        for per, yrs in (("sel_2022_2024", SEL), ("2025", (2025,)), ("2026", (2026,))):
            g = d[m & d["year"].isin(yrs).to_numpy()]
            if len(g) < 20:
                row[per] = {"n": int(len(g))}
                continue
            ret = g[f"ret_{bet}"].to_numpy()
            row[per] = {"pooled": L.boot_roi(g["race_id"].to_numpy(), ret, 1000), "race_equal": L.race_equal_roi(g["race_id"].to_numpy(), ret, 500), "hit": float((ret > 0).mean()),
                        "dd": L.max_drawdown(g["date"].to_numpy(), g["race_id"].to_numpy(), ret - 100.0)}
        rows.append(row)
    out = {"n_rules": len(names), "reality_check": {k: v for k, v in rc.items() if k != "t"}, "best": rows[rc["best"]], "rows": rows}
    # 診断（族の外）: モデルの勝率に対する残差の符号（V4 が騎手・調教師の力をすでに織り込んでいるか）
    d2 = d.copy()
    out["diag_model_resid"] = []
    for y in range(2023, 2027):
        h = d2[(d2["year"] < y) & (d2["year"] >= 2022)]
        for nm, key in (("j", "jockey_id"), ("t", "trainer_id")):
            g = (h["win"] - h["p_win_raw"]).groupby(h[key]).sum()
            d2.loc[d2["year"] == y, f"{nm}_resp_pos"] = (d2.loc[d2["year"] == y, key].map(g) > 0).astype(float)
    for bet, evc in (("win", "ev_win"),):
        for tau in (1.0, 1.2, 1.5):
            for c in ("j_resp_pos", "t_resp_pos"):
                for per, yrs in (("2023_2024", (2023, 2024)), ("2025", (2025,)), ("2026", (2026,))):
                    g = d2[(d2[evc] >= tau) & d2["year"].isin(yrs)]
                    a, b = g[g[c] == 1], g[g[c] == 0]
                    if len(a) < 30 or len(b) < 30:
                        continue
                    out["diag_model_resid"].append({"bet": bet, "tau": tau, "cond": c, "period": per, "n_in": int(len(a)), "n_out": int(len(b)),
                                                    **L.boot_diff(a["race_id"].to_numpy(), a[f"ret_{bet}"].to_numpy(), b["race_id"].to_numpy(), b[f"ret_{bet}"].to_numpy(), 500)})
    (P.OUT / "ev_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


def ev_robust(tau: float = 1.5, n_perm: int = 2000) -> dict:
    """候補『単勝 EV ≥ τ かつ 騎手・調教師の市場に対する残差がともに正』の頑健性。
    (a) 年別 (b) 的中の大きい順に k 本を除いた回収率 (c) オッズの上限を付けた回収率 (d) 条件の付け替え検定:
    その年の騎手（調教師）に付いた条件（正/負）を、その年の騎手どうし（調教師どうし）で無作為に入れ替えても同じ回収率が出るか。"""
    d = P.load()
    d = add_asof_conditions(d)
    d = d[(d["year"] >= 2022) & d["p_win_raw"].notna()].copy().reset_index(drop=True)
    d["ev_win"] = d["p_win_raw"] * d["win_odds"]
    both = (d["ev_win"] >= tau) & (d["j_res_pos"] == 1) & (d["t_res_pos"] == 1)
    out: dict = {"tau": tau, "rule": f"win EV>={tau} & j_res_pos & t_res_pos"}
    g = d[both]
    out["by_year"] = {int(y): {**L.boot_roi(x["race_id"].to_numpy(), x["ret_win"].to_numpy(), 500), "hit": float((x["ret_win"] > 0).mean()), "median_odds": float(x["win_odds"].median())}
                      for y, x in g.groupby("year")}
    periods = {"sel_2022_2024": g[g["year"].isin(SEL)], "2025": g[g["year"] == 2025], "2026": g[g["year"] == 2026], "2025_2026": g[g["year"] >= 2025]}
    out["pooled"] = {k: {**L.boot_roi(x["race_id"].to_numpy(), x["ret_win"].to_numpy(), 1000), "hit": float((x["ret_win"] > 0).mean())} for k, x in periods.items()}
    out["drop_top"] = {}
    for k, x in periods.items():
        r = np.sort(x["ret_win"].to_numpy())[::-1]
        out["drop_top"][k] = {f"drop{m}": float((r[m:].sum()) / (100.0 * (len(r) - m)) * 100) if len(r) > m else None for m in (0, 1, 3, 5, 10)}
    out["odds_cap"] = {}
    for cap in (20, 30, 50, 100):
        out["odds_cap"][str(cap)] = {}
        for k, x in periods.items():
            y = x[x["win_odds"] <= cap]
            out["odds_cap"][str(cap)][k] = {**L.boot_roi(y["race_id"].to_numpy(), y["ret_win"].to_numpy(), 300)} if len(y) else None
    # 付け替え検定
    rng = np.random.default_rng(0)
    cand = d[d["ev_win"] >= tau]
    res = {}
    for per, ys in (("sel_2022_2024", SEL), ("2025", (2025,)), ("2026", (2026,))):
        c = cand[cand["year"].isin(ys)]
        obs = c[(c["j_res_pos"] == 1) & (c["t_res_pos"] == 1)]
        obs_roi = float(obs["ret_win"].sum() / (100.0 * len(obs)) * 100) if len(obs) else float("nan")
        codes_j = {y: pd.factorize(c.loc[c["year"] == y, "jockey_id"]) for y in ys}
        codes_t = {y: pd.factorize(c.loc[c["year"] == y, "trainer_id"]) for y in ys}
        flag_j = {y: c.loc[c["year"] == y].groupby("jockey_id")["j_res_pos"].first().reindex(codes_j[y][1]).to_numpy() for y in ys}
        flag_t = {y: c.loc[c["year"] == y].groupby("trainer_id")["t_res_pos"].first().reindex(codes_t[y][1]).to_numpy() for y in ys}
        ret_y = {y: c.loc[c["year"] == y, "ret_win"].to_numpy() for y in ys}
        sims = np.empty(n_perm)
        for b in range(n_perm):
            tot, n = 0.0, 0
            for y in ys:
                fj = rng.permutation(flag_j[y])[codes_j[y][0]]
                ft = rng.permutation(flag_t[y])[codes_t[y][0]]
                m = (fj == 1) & (ft == 1)
                tot += ret_y[y][m].sum()
                n += int(m.sum())
            sims[b] = tot / (100.0 * n) * 100 if n else np.nan
        res[per] = {"observed_roi": obs_roi, "n_obs": int(len(obs)), "perm_mean": float(np.nanmean(sims)), "perm_sd": float(np.nanstd(sims)), "perm_q95": float(np.nanquantile(sims, 0.95)),
                    "p_ge_obs": float(np.nanmean(sims >= obs_roi))}
    out["flag_permutation"] = res
    (P.OUT / "ev_robust.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


def flag_modelfree() -> dict:
    """モデルを使わない確認: 騎手・調教師の『勝ち数 − 市場の見込み』が（当年より前に）ともに正の馬を、単勝オッズの帯ごとに、条件なしと比べる。
    2021 年は 2020 年だけで条件を作る（これまでの選択年・確認年と無関係の標本）。年別・オッズ帯別の単勝・複勝の回収率と、差の区間。"""
    d = P.load()
    d = d.sort_values(["date", "race_id"]).reset_index(drop=True)
    for c in ("j_pos", "t_pos"):
        d[c] = np.nan
    for y in range(2021, 2027):
        h = d[d["year"] < y]
        cur = d["year"] == y
        for nm, key in (("j_pos", "jockey_id"), ("t_pos", "trainer_id")):
            g = (h["win"] - h["q_win"]).groupby(h[key]).sum()
            d.loc[cur, nm] = (d.loc[cur, key].map(g) > 0).astype(float)
    d = d[d["year"] >= 2021].copy()
    d["both"] = (d["j_pos"] == 1) & (d["t_pos"] == 1)
    bands = {"all": (0, 1e9), "<10": (0, 10), "10-30": (10, 30), "30-100": (30, 100), ">=100": (100, 1e9), ">=30": (30, 1e9)}
    out: dict = {"rows": []}
    for y in list(range(2021, 2027)) + ["2021-2024", "2025-2026", "2021-2026"]:
        if isinstance(y, int):
            g = d[d["year"] == y]
        elif y == "2021-2024":
            g = d[d["year"].between(2021, 2024)]
        elif y == "2025-2026":
            g = d[d["year"] >= 2025]
        else:
            g = d
        for bn, (lo, hi) in bands.items():
            gg = g[(g["win_odds"] >= lo) & (g["win_odds"] < hi)]
            a, b = gg[gg["both"]], gg[~gg["both"]]
            if len(a) < 30:
                continue
            row = {"period": str(y), "band": bn, "n_flag": int(len(a)), "n_other": int(len(b))}
            for bet in ("win", "place"):
                rc = f"ret_{bet}"
                row[bet] = {"flag": L.boot_roi(a["race_id"].to_numpy(), a[rc].to_numpy(), 500), "other_roi": float(100 * b[rc].mean() / 100.0),
                            "diff": L.boot_diff(a["race_id"].to_numpy(), a[rc].to_numpy(), b["race_id"].to_numpy(), b[rc].to_numpy(), 500)}
            out["rows"].append(row)
    (P.OUT / "flag_modelfree.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "ev", "evfinal", "evrobust", "flagfree"])
    a = ap.parse_args()
    if a.cmd == "flagfree":
        flag_modelfree()
        print("saved")
    elif a.cmd == "evrobust":
        r = ev_robust()
        print(json.dumps(r["pooled"], ensure_ascii=False, default=float)[:600])
    elif a.cmd == "evfinal":
        r = ev_final()
        print("RC", r["reality_check"], "best", r["best"]["bet"], r["best"]["tau"], r["best"]["cond"])
    elif a.cmd == "scan":
        r = scan()
        print("rules", r["n_rules"], "RC", r["reality_check"], "best", r["best_rule"])
    else:
        ev_overlay()
        print("saved")


if __name__ == "__main__":
    main()
