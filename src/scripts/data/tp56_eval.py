"""tp-56 の評価: (1) V5 への増分（OOF への条件付きロジット/ロジスティックの上乗せ、LOYO）、(2) 印の質、(3) 買い目設計のバックテスト。

python -m src.scripts.data.tp56_eval screen            # V5 の OOF に特徴量群を足したときの勝率・連対・複勝 LogLoss の増分（LOYO、レース単位ブートストラップ）
python -m src.scripts.data.tp56_eval marks             # 統合した印の質（◎の勝率・複勝圏、カバー率）と既存の印との比較
python -m src.scripts.data.tp56_eval bet               # 買い目設計のバックテスト（2022〜2025 年、2026 年は別コマンド）

入出力はホストの TP56_OUT（既定 /home/jovyan/work/tp56_out）。V5 の OOF は training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet（2022〜2024 年、LOYO）。
増分の評価は『V5 のロジットに特徴量の線形項を足す』一次選別（勝率=レース内 softmax の条件付きロジット、連対・複勝=ロジスティック。
係数は評価年を含まない 2 年で学習、L2 正則化）。2 段階: 評価年 = 2022 / 2023 / 2024 の LOYO。test 2025 は選択に使わない。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.pipeline.features import tp56_yosou as ty

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP56_OUT", "/home/jovyan/work/tp56_out"))
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
YEARS = (2022, 2023, 2024)
EPS = 1e-9


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_base() -> pd.DataFrame:
    feats = pd.read_parquet(OUT / "features.parquet")
    oof = pd.read_parquet(OOF)
    d = feats.merge(oof, on=["race_id", "horse_id"], how="inner")
    d = d.sort_values(["date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    d["y_win"] = (d["finish_position"] == 1).astype(float)
    d["y_top2"] = (d["finish_position"] <= 2).astype(float)
    d["y_top3"] = (d["finish_position"] <= 3).astype(float)
    nwin = d.groupby("race_id")["y_win"].transform("sum")
    d = d[nwin == 1].reset_index(drop=True)          # 同着 1 着・不成立のレースは除く（条件付きロジットの前提）
    d["ridx"] = pd.factorize(d["race_id"])[0]
    return d


def starts_of(ridx: np.ndarray):
    st = np.flatnonzero(np.r_[True, ridx[1:] != ridx[:-1]])
    return st, np.diff(np.r_[st, len(ridx)])


def zscore(Xtr: np.ndarray, *others):
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    out = []
    for X in (Xtr,) + others:
        Z = (X - mu) / sd
        out.append(np.nan_to_num(Z, nan=0.0))
    return out


def fit_clogit(X, off, y, ridx, l2=1.0):
    """レース内 softmax（ロジット = off + Xθ）の勝者の対数尤度を最大化。"""
    st, cnt = starts_of(ridx)
    k = X.shape[1]

    def f(th):
        s = off + X @ th
        m = np.maximum.reduceat(s, st)
        e = np.exp(s - np.repeat(m, cnt))
        Z = np.add.reduceat(e, st)
        p = e / np.repeat(Z, cnt)
        ll = (s * y).sum() - (m + np.log(Z)).sum()   # 勝者が 1 頭のレースを前提
        g = X.T @ (y - p)
        return -ll + l2 * (th @ th), -g + 2 * l2 * th

    r = minimize(f, np.zeros(k), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return r.x


def fit_logit(X, off, y, l2=1.0):
    k = X.shape[1]

    def f(th):
        s = off + X @ th
        p = expit(s)
        ll = (y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)).sum()
        return -ll + l2 * (th @ th), X.T @ (p - y) + 2 * l2 * th

    r = minimize(f, np.zeros(k), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return r.x


def softmax_p(s, ridx):
    st, cnt = starts_of(ridx)
    m = np.maximum.reduceat(s, st)
    e = np.exp(s - np.repeat(m, cnt))
    return e / np.repeat(np.add.reduceat(e, st), cnt)


def row_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def race_sums(loss, ridx):
    st, _ = starts_of(ridx)
    return np.add.reduceat(loss, st)


def boot_delta(Lnew, Lbase, cnt, n_boot=1000, seed=0):
    """レース単位ブートストラップ。差 = Σ(新−基準) / Σ行数。"""
    rng = np.random.default_rng(seed)
    n = len(Lnew)
    d = Lnew - Lbase
    est = d.sum() / cnt.sum()
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = d[i].sum() / cnt[i].sum()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def loyo(d: pd.DataFrame, cols: list[str], l2=3.0) -> dict:
    """評価年ごとに、他の 2 年で係数を学習し、評価年の LogLoss を基準（V5 の OOF）と比べる。全評価年のプールで CI を出す。"""
    out = {}
    pooled = {t: {"new": [], "base": [], "cnt": []} for t in ("win", "top2", "top3")}
    peryear = {}
    for ev in YEARS:
        tr = d[d["year"] != ev]
        te = d[d["year"] == ev]
        Xtr, Xte = zscore(tr[cols].to_numpy(float), te[cols].to_numpy(float))
        res = {}
        # 勝率
        offtr, offte = np.log(np.clip(tr["p_win"].to_numpy(), 1e-6, 1)), np.log(np.clip(te["p_win"].to_numpy(), 1e-6, 1))
        th = fit_clogit(Xtr, offtr, tr["y_win"].to_numpy(), tr["ridx"].to_numpy(), l2)
        pn = softmax_p(offte + Xte @ th, te["ridx"].to_numpy())
        pb = softmax_p(offte, te["ridx"].to_numpy())
        pooled["win"]["new"].append(race_sums(row_loss(pn, te["y_win"].to_numpy()), te["ridx"].to_numpy()))
        pooled["win"]["base"].append(race_sums(row_loss(te["p_win"].to_numpy(), te["y_win"].to_numpy()), te["ridx"].to_numpy()))
        pooled["win"]["cnt"].append(starts_of(te["ridx"].to_numpy())[1])
        res["win_theta"] = th.round(3).tolist()
        for t in ("top2", "top3"):
            off_tr, off_te = logit(tr[f"p_{t}"].to_numpy()), logit(te[f"p_{t}"].to_numpy())
            th2 = fit_logit(Xtr, off_tr, tr[f"y_{t}"].to_numpy(), l2)
            pnew = expit(off_te + Xte @ th2)
            pooled[t]["new"].append(race_sums(row_loss(pnew, te[f"y_{t}"].to_numpy()), te["ridx"].to_numpy()))
            pooled[t]["base"].append(race_sums(row_loss(te[f"p_{t}"].to_numpy(), te[f"y_{t}"].to_numpy()), te["ridx"].to_numpy()))
            pooled[t]["cnt"].append(starts_of(te["ridx"].to_numpy())[1])
        peryear[ev] = res
    for t in ("win", "top2", "top3"):
        Ln = np.concatenate(pooled[t]["new"])
        Lb = np.concatenate(pooled[t]["base"])
        cn = np.concatenate(pooled[t]["cnt"])
        est, lo, hi = boot_delta(Ln, Lb, cn)
        out[t] = {"delta": est, "lo": lo, "hi": hi}
        # 年別
        yr = {}
        o = 0
        for ev, a, b, c in zip(YEARS, pooled[t]["new"], pooled[t]["base"], pooled[t]["cnt"]):
            yr[ev] = float((a.sum() - b.sum()) / c.sum())
        out[t]["by_year"] = yr
    out["theta"] = {str(k): v["win_theta"] for k, v in peryear.items()}
    return out


def cmd_screen(a) -> None:
    d = load_base()
    print("rows", len(d), "races", d["ridx"].nunique(), "years", d["year"].value_counts().sort_index().to_dict(), flush=True)
    G = dict(ty.GROUPS)
    # 個別の列
    singles = {c: [c] for c in ["fit_rate", "fit_lift", "fit_pct", "fit_best", "m_front", "m_kick", "m_att", "elo_rel_c", "elo_pw_c", "elo_rel_c_adj", "elo_pw_c_adj",
                                "elo_rel_g", "front_t_rel", "kick_t_rel", "stam_t_rel", "trouble_last"]}
    groups = {**G, **{f"s:{k}": v for k, v in singles.items()}}
    groups["FIT+ELO"] = G["FIT"] + G["ELO"]
    groups["FIT+ELO_ADJ"] = G["FIT"] + G["ELO_ADJ"]
    groups["FIT+HT"] = G["FIT"] + G["HT"]
    base_fin = {t: float(row_loss(d[f"p_{t}"].to_numpy(), d[f"y_{t}"].to_numpy()).mean()) for t in ("win", "top2", "top3")}
    print("基準 V5 OOF LogLoss", base_fin, flush=True)
    res = {"base": base_fin, "n_groups_tried": len(groups), "groups": {}}
    for name, cols in groups.items():
        r = loyo(d, cols)
        res["groups"][name] = {"cols": cols, **{k: v for k, v in r.items() if k != "theta"}}
        w, t2, t3 = r["win"], r["top2"], r["top3"]
        ok = (w["hi"] < 0 and w["delta"] <= -0.0002 and t2["delta"] <= 0.0002 and t3["delta"] <= 0.0002)
        print(f"{name:14s} win {w['delta']:+.5f} [{w['lo']:+.5f},{w['hi']:+.5f}] top2 {t2['delta']:+.5f} top3 {t3['delta']:+.5f} 採用={ok}", flush=True)
        res["groups"][name]["adopt"] = bool(ok)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")



# ───────────────────────── 統合した印（ユーザーの手順）と買い目設計 ─────────────────────────
BET = ROOT / "data/local/modeling/betting"
FINAL_V5 = ROOT / "data/local/modeling/training_pipeline/final_eval_v5/test2025_predictions.parquet"
INT_COLS = ty.GROUPS["ALL_ADJ"]


def with_2025(d: pd.DataFrame) -> pd.DataFrame:
    """2025 年（参考）: bundle_v5_2024 の test 2025 予測 + 特徴量。"""
    feats = pd.read_parquet(OUT / "features.parquet")
    p25 = pd.read_parquet(FINAL_V5).drop(columns=["finish_position"])
    x = feats.merge(p25, on=["race_id", "horse_id"], how="inner")
    x["year"] = 2025
    x["y_win"] = (x["finish_position"] == 1).astype(float)
    x["y_top2"] = (x["finish_position"] <= 2).astype(float)
    x["y_top3"] = (x["finish_position"] <= 3).astype(float)
    nwin = x.groupby("race_id")["y_win"].transform("sum")
    x = x[nwin == 1]
    return pd.concat([d, x], ignore_index=True).sort_values(["date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)


def integrated_scores(d: pd.DataFrame, cols: list[str], l2=3.0) -> pd.DataFrame:
    """統合スコア: 2022〜2024 年は LOYO、2025 年は 2022〜2024 年で学習。pw_int（勝率）と p3_int（複勝）。"""
    d = d.copy()
    d["ridx"] = pd.factorize(d["race_id"])[0]
    d["pw_int"] = np.nan
    d["p3_int"] = np.nan
    for ev in sorted(d["year"].unique()):
        tr = d[(d["year"] != ev) & (d["year"] <= 2024)]
        te_i = d["year"] == ev
        te = d[te_i]
        Xtr, Xte = zscore(tr[cols].to_numpy(float), te[cols].to_numpy(float))
        th = fit_clogit(Xtr, np.log(np.clip(tr["p_win"].to_numpy(), 1e-6, 1)), tr["y_win"].to_numpy(), tr["ridx"].to_numpy(), l2)
        d.loc[te_i, "pw_int"] = softmax_p(np.log(np.clip(te["p_win"].to_numpy(), 1e-6, 1)) + Xte @ th, te["ridx"].to_numpy())
        th3 = fit_logit(Xtr, logit(tr["p_top3"].to_numpy()), tr["y_top3"].to_numpy(), l2)
        d.loc[te_i, "p3_int"] = expit(logit(te["p_top3"].to_numpy()) + Xte @ th3)
    return d


def make_marks(g: pd.DataFrame, variant: str) -> list[int] | None:
    """1 レースの印の順序 [◎, ○, ▲, △1, △2, △3, △4]（馬の行番号）。"""
    n = len(g)
    if n < 6:
        return None
    if variant == "base":
        sw, s3 = g["p_win"].to_numpy(), g["p_top3"].to_numpy()
        pool = np.arange(n)
    elif variant == "int":
        sw, s3 = g["pw_int"].to_numpy(), g["p3_int"].to_numpy()
        pool = np.arange(n)
    else:   # fit: 適合上位 2 グループの中から（足りなければ全馬）
        sw, s3 = g["pw_int"].to_numpy(), g["p3_int"].to_numpy()
        pool = np.flatnonzero(g["fit_rank"].fillna(99).to_numpy() <= 2)
        if len(pool) < 4:
            pool = np.arange(n)
    order = pool[np.argsort(-sw[pool], kind="stable")]
    top3 = list(order[:3])
    rest = [i for i in pool[np.argsort(-s3[pool], kind="stable")] if i not in top3][:4]
    if len(rest) < 4:   # プールが小さいときは、プール外から複勝確率順で補う
        used = set(top3) | set(rest)
        rest += [i for i in np.argsort(-s3, kind="stable") if i not in used][: 4 - len(rest)]
    return top3 + rest


def mark_quality(d: pd.DataFrame, variants: list[str]) -> dict:
    res = {}
    for v in variants:
        rows = []
        for rid, g in d.groupby("race_id", sort=False):
            m = make_marks(g, v)
            if m is None:
                continue
            fin = g["finish_position"].to_numpy()
            t3 = fin <= 3
            rows.append({"year": int(g["year"].iloc[0]), "honmei_win": float(fin[m[0]] == 1), "honmei_t3": float(t3[m[0]]), "o_t3": float(t3[m[1]]), "a_t3": float(t3[m[2]]),
                         "cov3": float(t3[m[:3]].sum()), "cov7": float(t3[m].sum()), "all3": float(t3[m].sum() == 3), "all3_oab": float(t3[m[:3]].sum() == 3), "n_marks": float(len(m))})
        r = pd.DataFrame(rows)
        res[v] = {"all": r.drop(columns="year").mean().round(4).to_dict(), "by_year": r.groupby("year").mean().round(4).to_dict("index"), "n_races": int(len(r))}
        res[v]["_rows"] = r
    return res


def cmd_marks(a) -> None:
    d = with_2025(load_base())
    d = integrated_scores(d, INT_COLS)
    d.to_parquet(OUT / "integrated.parquet", index=False)
    q = mark_quality(d, ["base", "int", "fit"])
    out = {}
    for v, r in q.items():
        rr = r.pop("_rows")
        out[v] = r
        print(v, r["n_races"], r["all"], flush=True)
    # 差の信頼区間（◎の複勝圏率、◎○▲のカバー数）
    rng = np.random.default_rng(0)
    base_rows = mark_quality(d, ["base"])["base"]["_rows"]
    for v in ("int", "fit"):
        rv = mark_quality(d, [v])[v]["_rows"]
        for m in ("honmei_win", "honmei_t3", "cov3", "cov7"):
            diff = (rv[m].to_numpy() - base_rows[m].to_numpy())
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
            out[v].setdefault("diff_vs_base", {})[m] = [round(float(diff.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        print(v, "diff_vs_base", out[v]["diff_vs_base"], flush=True)
    (OUT / "marks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")




LS_TH = (8.0, 12.0, 20.0)


def sc(*x):
    return tuple(sorted(int(i) for i in x))


def rule_names() -> list[str]:
    base = ["wide1", "uma1", "wide2", "wide3", "tri4box", "tri_axis6"]
    ls = [f"ls{k}_{th:g}_{fb}" for th in LS_TH for k in ("A", "B", "C") for fb in ("a", "b")]
    return base + ls


def tickets_for(rule: str, M: list[int], odds: dict) -> list[tuple]:
    """M: 印の馬番 [◎, ○, ▲, △1..△4]。戻り値: [(券種, 組合せ)]。"""
    o, a, b = M[0], M[1], M[2]
    tri = M[3:7]
    if rule == "wide1":
        return [("wide", sc(o, a))]
    if rule == "uma1":
        return [("umaren", sc(o, a))]
    if rule == "wide2":
        return [("wide", sc(o, a)), ("wide", sc(o, b))]
    if rule == "wide3":
        return [("wide", sc(o, a)), ("wide", sc(o, b)), ("wide", sc(a, b))]
    if rule == "tri4box":
        c = [o, a, b, tri[0]]
        return [("sanrenpuku", sc(*t)) for t in ((c[0], c[1], c[2]), (c[0], c[1], c[3]), (c[0], c[2], c[3]), (c[1], c[2], c[3]))]
    if rule == "tri_axis6":
        p = [a, b, tri[0], tri[1]]
        return [("sanrenpuku", sc(o, p[i], p[j])) for i in range(4) for j in range(i + 1, 4)]
    # ls{A,B,C}_{th}_{a,b}: 印の中に高オッズ馬（th 以上）がいればその馬から、いなければ人気側の絞った買い
    _, kind_th = rule.split("ls", 1)
    kind, th, fb = kind_th.split("_")
    th = float(th)
    cand = [h for h in M[:5] if odds.get(h, 0.0) >= th]
    if cand:
        L = cand[0]
        partners = [h for h in M[:4] if h != L][:3]
        W = [("wide", sc(L, h)) for h in partners]
        T = [("sanrenpuku", sc(L, partners[i], partners[j])) for i in range(len(partners)) for j in range(i + 1, len(partners))]
        return W if kind == "A" else T if kind == "B" else W + T
    return [("wide", sc(o, a))] if fb == "a" else [("umaren", sc(o, a)), ("wide", sc(o, a))]


def settle(hits_r: dict, tk: list[tuple]) -> float | None:
    """100 円あたりの払戻倍率の合計。券種が壊れているレースは None。"""
    broken = hits_r.get("_broken", set())
    r = 0.0
    for bt, combo in tk:
        if bt in broken:
            return None
        r += hits_r.get(bt, {}).get(combo, 0.0) / 100.0
    return r


def ruin_prob(S, R, n_races=300, n_sim=2000, bank_mult=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n = len(S)
    B0 = bank_mult * S.mean()
    ruined = 0
    for _ in range(n_sim):
        i = rng.integers(0, n, n_races)
        path = B0 + np.cumsum(R[i] - S[i])
        ruined += (path.min() <= 0)
    return ruined / n_sim


def bet_inputs(d: pd.DataFrame):
    """印の表に馬番・単勝オッズを足し、オッズ・払戻が壊れたレースを除く（tp48_2026 と同じ除外）。"""
    from src.scripts.data import tp48_eval as t48

    frame = pd.read_parquet(BET / "tp36/frame_all.parquet", columns=["race_id", "horse_id", "horse_number", "win_odds"])
    d = d.merge(frame, on=["race_id", "horse_id"], how="inner")
    inv = (1.0 / d["win_odds"].where(d["win_odds"] >= 1.0)).groupby(d["race_id"]).sum()
    bad_odds = set(inv[~inv.between(1.0, 1.6)].index) | set(d.loc[~(d["win_odds"] >= 1.0), "race_id"])
    hits, bad = t48.load_hits_all()
    pay26 = pd.read_parquet(ROOT / "data/local/modeling/betting_2026/payoff.parquet")
    bad = set(bad) | set(pay26.loc[pay26["ambiguous"] | ~pay26["parsed"], "race_id"]) | bad_odds
    return d, hits, bad


def cmd_bet(a) -> None:
    from src.scripts.data.tp50_lib import reality_check, roi_ci

    d = pd.read_parquet(OUT / "integrated.parquet")
    d, hits, bad = bet_inputs(d)
    names = rule_names()
    variants = ["base", "int", "fit"]
    rows = {(v, r): [] for v in variants for r in names}   # (variant, rule) → [(year, race_id, S, R)]
    for rid, g in d.groupby("race_id", sort=False):
        if rid in bad or rid not in hits:
            continue
        g = g.reset_index(drop=True)
        odds = {int(h): float(o) for h, o in zip(g["horse_number"], g["win_odds"]) if o >= 1.0}
        yr = int(g["year"].iloc[0])
        for v in variants:
            m = make_marks(g, v)
            if m is None:
                continue
            M = [int(g["horse_number"].iloc[i]) for i in m]
            for r in names:
                tk = tickets_for(r, M, odds)
                ret = settle(hits[rid], tk)
                if ret is None:
                    continue
                rows[(v, r)].append((yr, rid, float(len(tk)), ret))
    res = {}
    prim = {}
    for (v, r), lst in rows.items():
        arr = pd.DataFrame(lst, columns=["year", "race_id", "S", "R"])
        pr = arr[arr["year"].isin((2022, 2023, 2024))]
        if len(pr) < 500:
            continue
        S, R = pr["S"].to_numpy(), pr["R"].to_numpy()
        roi, lo, hi = roi_ci(S, R, 500)
        ex = (R.sum() - np.sort(R)[-3:].sum()) / S.sum()
        by_year = {int(y): round(float(g["R"].sum() / g["S"].sum()), 3) for y, g in arr.groupby("year")}
        res[f"{v}|{r}"] = {"roi": round(roi, 4), "lo": round(lo, 4), "hi": round(hi, 4), "hit": round(float((R > 0).mean()), 4),
                           "reach500": round(float((R / S >= 5).mean()), 4), "roi_ex_top3": round(float(ex), 4), "ruin": ruin_prob(S, R),
                           "points_per_race": round(float(S.mean()), 2), "n_races": int(len(pr)), "by_year": by_year}
        prim[f"{v}|{r}"] = pr.set_index("race_id")["R"] - pr.set_index("race_id")["S"]
    # reality check（共通のレース集合）
    keys = list(prim)
    common = set.intersection(*[set(prim[k].index) for k in keys])
    P = np.stack([prim[k].loc[sorted(common)].to_numpy() for k in keys], axis=1)
    rc = reality_check(P, 1000)
    best = keys[rc["best"]]
    out = {"n_strategies": len(keys), "reality_check": {**rc, "best_name": best}, "strategies": res}
    (OUT / "bet.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    top = sorted(res.items(), key=lambda kv: -kv[1]["lo"])[:12]
    print("戦略数", len(keys), "RC", rc, best, flush=True)
    for k, v in top:
        print(k, v, flush=True)



# ───────────────────────── 各段階の単独の予測力 ─────────────────────────
def auc(score: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(score)
    r = pd.Series(score[m]).rank().to_numpy()
    yy = y[m]
    n1, n0 = yy.sum(), (1 - yy).sum()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_race_corr(d: pd.DataFrame, col: str, sign: float = 1.0) -> tuple[float, int]:
    """レース内順位相関（col が高いほど着順が良い想定、sign=-1 で反転）の平均。レース単位。"""
    x = d[["race_id", col, "finish_position"]].dropna()
    x = x[x.groupby("race_id")[col].transform("size") >= 6]
    a = x.groupby("race_id")[col].rank()
    b = -x.groupby("race_id")["finish_position"].rank()
    xa = pd.DataFrame({"r": x["race_id"], "a": a * sign, "b": b})
    g = xa.groupby("r")
    xa["a"] -= g["a"].transform("mean")
    xa["b"] -= g["b"].transform("mean")
    num = (xa["a"] * xa["b"]).groupby(xa["r"]).sum()
    den = np.sqrt((xa["a"] ** 2).groupby(xa["r"]).sum() * (xa["b"] ** 2).groupby(xa["r"]).sum())
    c = (num / den).dropna()
    return float(c.mean()), int(len(c))


def cmd_stages(a) -> None:
    feats = pd.read_parquet(OUT / "features.parquet")
    feats["year"] = pd.to_datetime(feats["date"]).dt.year
    f = feats[(feats["year"] >= 2022) & (feats["year"] <= 2025) & (feats["field_size"] >= 6)].copy()
    f["y3"] = (f["finish_position"] <= 3).astype(float)
    f["y1"] = (f["finish_position"] == 1).astype(float)
    out: dict = {}
    # (1) レースタイプ: 直前までの流れ R_* が、そのレース自身の実現値（fb/kb/chalk/inside/attr）をどれだけ予測するか
    runs = ty.prepare_runs(pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "data/page_reference/tables").glob("*/race_result_flat.parquet"))], ignore_index=True))
    rl = ty.race_level(runs)
    rl["year"] = pd.to_datetime(rl["date"]).dt.year
    rr = f.groupby("race_id").first()[[f"R_{c}" for c in ty.RACE_SIG]].reset_index()
    rl = rl.merge(rr, on="race_id", how="inner")
    rl = rl[(rl["year"] >= 2022) & (rl["year"] <= 2025)]
    rt = {}
    for c in ty.RACE_SIG:
        x = rl[[c, f"R_{c}"]].dropna()
        # 同じ競馬場・面の長期平均との比較: 流れ（R_）が長期平均より当たるか
        base = rl.groupby(["venue", "surface"])[c].transform("mean").loc[x.index]
        rho = float(np.corrcoef(x[c], x[f"R_{c}"])[0, 1])
        rho_b = float(np.corrcoef(x[c], base)[0, 1])
        rt[c] = {"corr_stream": round(rho, 4), "corr_venue_mean": round(rho_b, 4), "n_races": int(len(x))}
    out["race_type"] = rt
    # (2) 馬タイプ: 6 タイプ + 不明の複勝率・勝率
    out["horse_type"] = {int(k): {"n": int(len(g)), "top3": round(float(g["y3"].mean()), 4), "win": round(float(g["y1"].mean()), 4)} for k, g in f.groupby("tp")}
    # (3) 適合: 適合率の AUC、適合順位別の複勝率、レース内相関
    fr = {"auc_top3_fit_rate": round(auc(f["fit_rate"].to_numpy(), f["y3"].to_numpy()), 4), "auc_top3_fit_lift": round(auc(f["fit_lift"].to_numpy(), f["y3"].to_numpy()), 4),
          "within_race_corr_fit_rate": round(within_race_corr(f, "fit_rate")[0], 4), "by_rank": {}}
    for k, g in f.groupby(f["fit_rank"].clip(upper=5)):
        fr["by_rank"][int(k)] = {"n": int(len(g)), "top3": round(float(g["y3"].mean()), 4), "win": round(float(g["y1"].mean()), 4)}
    # 馬自身の力を揃えた適合（同じ elo_rel_c の十分位の中で fit_best か否か）
    f["elo_dec"] = pd.qcut(f["elo_rel_c"].rank(method="first"), 10, labels=False)
    fr["top3_gain_given_elo_decile"] = round(float((f[f.fit_best == 1].groupby("elo_dec")["y3"].mean() - f[f.fit_best == 0].groupby("elo_dec")["y3"].mean()).mean()), 4)
    out["fit"] = fr
    # (4) 条件付き対戦比較（Elo）と不利補正
    el = {}
    for c in ("elo_rel_g", "elo_rel_c", "elo_pw_c", "elo_rel_g_adj", "elo_rel_c_adj", "elo_pw_c_adj"):
        el[c] = {"auc_top3": round(auc(f[c].to_numpy(), f["y3"].to_numpy()), 4), "within_race_corr": round(within_race_corr(f, c)[0], 4)}
    sub = f[(f["trouble_n3"] > 0)]
    el["subset_trouble_n3>0"] = {"n_rows": int(len(sub)), **{c: round(within_race_corr(sub.assign(), c)[0], 4) for c in ("elo_rel_c", "elo_rel_c_adj")}}
    out["pairwise"] = el
    # 参考: V5 の複勝確率（2025 は最終版、2022〜24 は OOF）の AUC、人気（逆順）
    out["reference"] = {"auc_top3_popularity_inv": round(auc(-f["popularity"].to_numpy(float), f["y3"].to_numpy()), 4),
                        "within_race_corr_popularity_inv": round(within_race_corr(f, "popularity", -1.0)[0], 4)}
    (OUT / "stages.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str), flush=True)


# ───────────────────────── 2026 年（未使用）の 1 回だけの評価 ─────────────────────────
RULES_FILE = ROOT / "src/scripts/data/tp56_2026_rules.json"
RULES_2026 = [
    {"id": "Y1", "variant": "int", "rule": "lsB_12_b", "note": "統合した印 + 穴(単勝12倍以上)軸の3連複、いなければ馬連+ワイド（2022〜24 年で平均・下限とも最良）"},
    {"id": "Y2", "variant": "int", "rule": "lsC_12_b", "note": "統合した印 + 穴軸のワイド+3連複、いなければ馬連+ワイド"},
    {"id": "Y3", "variant": "fit", "rule": "lsC_12_b", "note": "ユーザー手順を文字どおり（適合上位2グループ内から印 + 穴軸のワイド+3連複）"},
    {"id": "Y4", "variant": "base", "rule": "lsB_12_b", "note": "既存の印（V5 の確率順）で同じ買い方（印の効果の切り分け）"},
]


def rules_hash(payload: dict) -> str:
    import hashlib

    body = json.dumps({"rules": payload["rules"], "periods": payload["periods"]}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def cmd_register(a) -> None:
    from datetime import datetime

    p = {"registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "purpose": "2026 年（未使用）の評価。評価前にこの一覧を固定する。確率は frame_all の 2026 年（V4 系）、係数は 2022〜24 年で学習。",
         "periods": {"design": "2022〜2025", "final": "2026（1 回だけ）"}, "rules": RULES_2026}
    p["hash"] = rules_hash(p)
    RULES_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    print("registered", RULES_FILE, p["hash"])


def cmd_run2026(a) -> None:
    from src.scripts.data.tp50_lib import roi_ci

    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} が既にある。2026 年の評価は 1 回だけ")
    payload = json.loads(RULES_FILE.read_text(encoding="utf-8"))
    if rules_hash(payload) != payload.get("hash"):
        raise SystemExit("規則のハッシュが固定時と違う")
    d = with_2025(load_base())
    feats = pd.read_parquet(OUT / "features.parquet")
    feats["year"] = pd.to_datetime(feats["date"]).dt.year
    f26 = feats[feats["year"] == 2026]
    fr = pd.read_parquet(BET / "tp36/frame_all.parquet", columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3", "year"])
    fr = fr[fr["year"] == 2026].drop(columns="year")
    x = f26.merge(fr, on=["race_id", "horse_id"], how="inner")
    x["year"] = 2026
    for t, c in (("y_win", 1), ("y_top2", 2), ("y_top3", 3)):
        x[t] = (x["finish_position"] <= c).astype(float) if t != "y_win" else (x["finish_position"] == 1).astype(float)
    x = x[x.groupby("race_id")["y_win"].transform("sum") == 1]
    d = pd.concat([d, x], ignore_index=True).sort_values(["date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    d = integrated_scores(d, INT_COLS)
    d, hits, bad = bet_inputs(d[d["year"].isin((2024, 2025, 2026))])
    res = {"hash": payload["hash"], "registered_at": payload["registered_at"], "rules": {}}
    for r in payload["rules"]:
        rows = {2024: [], 2025: [], 2026: []}
        for rid, g in d.groupby("race_id", sort=False):
            if rid in bad or rid not in hits:
                continue
            g = g.reset_index(drop=True)
            m = make_marks(g, r["variant"])
            if m is None:
                continue
            M = [int(g["horse_number"].iloc[i]) for i in m]
            odds = {int(h): float(o) for h, o in zip(g["horse_number"], g["win_odds"]) if o >= 1.0}
            tk = tickets_for(r["rule"], M, odds)
            ret = settle(hits[rid], tk)
            if ret is None:
                continue
            rows[int(g["year"].iloc[0])].append((float(len(tk)), ret))
        o = {}
        for y, lst in rows.items():
            if len(lst) < 50:
                continue
            S, R = np.array([a for a, _ in lst]), np.array([b for _, b in lst])
            roi, lo, hi = roi_ci(S, R, 500)
            o[str(y)] = {"n_races": len(lst), "roi": round(roi, 4), "lo": round(lo, 4), "hi": round(hi, 4), "hit": round(float((R > 0).mean()), 4),
                         "reach500": round(float((R / S >= 5).mean()), 4), "roi_ex_top3": round(float((R.sum() - np.sort(R)[-3:].sum()) / S.sum()), 4),
                         "points_per_race": round(float(S.mean()), 2), "ruin": ruin_prob(S, R)}
        res["rules"][r["id"]] = {**r, "result": o}
        print(r["id"], r["variant"], r["rule"], o, flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "eval_2026.json"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("cmd", choices=["screen", "marks", "bet", "stages", "register", "run2026"])
    a = ap.parse_args()
    {"screen": cmd_screen, "marks": cmd_marks, "bet": cmd_bet, "stages": cmd_stages, "register": cmd_register, "run2026": cmd_run2026}.get(a.cmd, lambda x: print("未実装"))(a)


if __name__ == "__main__":
    main()
