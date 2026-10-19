"""tp-35 の補足: 条件別の勝率・回収率を『レース単位』で集計し直す。

馬単位の母数（該当馬の頭数）で率を出すと、同じレースに該当馬が複数いるほど率が薄まって見える。
ここでは該当馬が 1 頭以上いるレースを 1 単位として、次を出す。
  レース勝率   = 該当馬のいずれかが 1 着のレース / 該当馬がいるレース（連対・複勝も同様）。比較用に『該当馬を無作為に選んだ場合の期待値』
                （超幾何: 頭数 n・出走 F 頭で 1 - C(F-n,k)/C(F,k)）と、その比（レース勝率の比）も出す
  回収率       = ①馬単位（該当馬を 1 頭ずつ同額で全頭買う）②レース均等（1 レースの賭け金を一定にし、該当馬に等分）
                ③最上位人気 1 頭（該当馬のうち確定単勝オッズ最小の馬だけ）④モデル最上位 1 頭（V4 の p_win 最大。2022 年以降のみ）
  信頼区間     = レース単位のブートストラップ。多重比較は reality check（レース単位ブートストラップの最大値）

python -m src.scripts.data.tp35_race_level build   # 馬ごとの表（ラップ種別・前週土日・オッズ・予測）を tp35/race_level/frame.parquet に保存
python -m src.scripts.data.tp35_race_level run     # 全集計 → tp35/race_level/results.json
python -m src.scripts.data.tp35_race_level chapter # 章 lap_groups_and_trainer_stats.html に『レース単位で見直す』節を挿入（冪等）
test 2025・2026 は条件の選択に使わない（2025 は参考列のみ）。市場オッズは条件の選択・買い方の評価にだけ使い、特徴量には使わない。
"""

from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb

from src.pipeline.features import tp35_lap_groups as L
from src.scripts.data import tp35_eval as E

OUT = E.OUT / "race_level"
BET = E.ROOT / "data/local/modeling/betting"
CHAPTER = E.ROOT / "docs/html/modeling/chapters/lap_groups_and_trainer_stats.html"
Y0, Y1 = 2020, 2024            # 条件の選択・集計に使う年（2025 は参考）
B = int(os.environ.get("TP35RL_BOOT", 400))
WK_NAMES = ["坂路<54", "坂路54-56", "坂路56-58", "坂路58-60", "坂路>=60"]
SCHEMES = {"horse": "馬単位", "race": "レース均等", "fav": "最上位人気1頭"}
ROI_COLS = {("w", "horse"): "roi_horse_w", ("w", "race"): "roi_race_w", ("w", "fav"): "roi_fav_w",
            ("p", "horse"): "roi_horse_p", ("p", "race"): "roi_race_p", ("p", "fav"): "roi_fav_p"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────── build
def cmd_build(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, *_ = E.t.setup()
    ret = E._returns(df)
    tp = L.prepare_laps(pd.read_parquet(E.LAPS))
    F, wk = L.pick_slot(tp, L.SLOT_F, "F"), L.prev_weekend(tp)
    d = df[["race_id", "horse_id", "date", "trainer_id", "finish_position", "year", "horse_number"]].copy()
    d["race_date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["horse_id"] = d["horse_id"].astype(str)
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce")
    for part in (F, wk):
        d = d.merge(part, on=["horse_id", "race_date"], how="left")
    assert len(d) == len(df) and (d["race_id"].to_numpy() == df["race_id"].to_numpy()).all()
    d["win_ret"], d["place_ret"] = ret["win_ret"].to_numpy(), ret["place_ret"].to_numpy()
    w = pd.read_parquet(BET / "odds_win_place.parquet")[["race_id", "horse_number", "win_odds"]]
    w["race_id"], w["horse_number"] = w["race_id"].astype(str), pd.to_numeric(w["horse_number"], errors="coerce")
    d = d.merge(w.drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left")
    pr = pd.concat([pd.read_parquet(E.WORK / "tp27/ens/v4_e2.parquet")[["race_id", "horse_id", "p_win"]],
                    pd.read_parquet(E.WORK / "final_eval_v4/test2025_predictions.parquet")[["race_id", "horse_id", "p_win"]]], ignore_index=True)
    pr["race_id"], pr["horse_id"] = pr["race_id"].astype(str), pr["horse_id"].astype(str)
    d = d.merge(pr.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    G = pd.concat([pd.read_parquet(E.OUT / "groups/LG_roi.parquet")[["lg_tr_course_cls_roiw", "lg_tr_course_cls_roip", "lg_tr_cls_roiw", "lg_tr_cls_roip"]],
                   pd.read_parquet(E.OUT / "groups/LG_stat.parquet")[["lg_tr_course_cls_logn", "lg_tr_cls_logn"]]], axis=1)
    assert len(G) == len(d)
    for c in G.columns:
        d[c] = G[c].to_numpy()
    wt = d["wk_t4"].to_numpy(dtype=float)
    wbin = np.where(np.isfinite(wt), np.digitize(wt, L.WK_BINS), -1)
    d["wk_label"] = np.where(np.isfinite(wt), np.array(WK_NAMES, dtype=object)[np.clip(wbin, 0, 4)], np.where(d["wk_n_all"].notna(), "土日に坂路なし(他調教のみ)", "土日に調教なし"))
    d.drop(columns=["date"]).to_parquet(OUT / "frame.parquet", index=False)
    _log(f"frame {d.shape} → {OUT / 'frame.parquet'}")


# ───────────────────────── 集計の道具
def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["year"].between(Y0, 2025) & d["finish_position"].gt(0) & d["win_ret"].notna() & d["place_ret"].notna()].copy()
    fin = d["finish_position"].to_numpy()
    d["win"], d["t2"], d["t3"] = (fin == 1).astype("int8"), (fin <= 2).astype("int8"), (fin <= 3).astype("int8")
    d["fs"] = d.groupby("race_id")["win"].transform("size")
    d["win_odds"] = d["win_odds"].where(d["win_odds"] >= 1.0)
    return d.reset_index(drop=True)


def race_table(d: pd.DataFrame, key) -> pd.DataFrame:
    """(条件, レース) ごとに 1 行。key は d と同じ並びの Series（欠損の行は条件に入らない）。"""
    x = d.assign(_k=pd.Series(np.asarray(key, dtype=object), index=d.index)).dropna(subset=["_k"])
    x = x.sort_values(["_k", "race_id"], kind="stable")
    g = x.groupby(["_k", "race_id"], sort=False)
    rt = g.agg(year=("year", "first"), n=("win", "size"), win_any=("win", "max"), top2_any=("t2", "max"), top3_any=("t3", "max"),
               sum_wr=("win_ret", "sum"), sum_pr=("place_ret", "sum"), win_hs=("win", "sum"), top3_hs=("t3", "sum"), fs=("fs", "first"))
    rt["mean_wr"], rt["mean_pr"] = rt["sum_wr"] / rt["n"], rt["sum_pr"] / rt["n"]
    n, fs = rt["n"].to_numpy(dtype=float), rt["fs"].to_numpy(dtype=float)
    rt["exp_win"] = n / fs
    for k in (2, 3):
        with np.errstate(invalid="ignore", divide="ignore"):
            rt[f"exp_top{k}"] = np.where(fs - n < k, 1.0, 1.0 - comb(np.maximum(fs - n, 0), k) / comb(fs, k))
    for tag, col, asc in (("fav", "win_odds", True), ("mod", "p_win", False)):
        xo = x.sort_values(["_k", "race_id", col], kind="stable", ascending=[True, True, asc], na_position="last").drop_duplicates(["_k", "race_id"]).set_index(["_k", "race_id"]).reindex(rt.index)
        has = xo[col].notna().to_numpy()
        rt[f"{tag}_wr"] = np.where(has, xo["win_ret"].to_numpy(), np.nan)
        rt[f"{tag}_pr"] = np.where(has, xo["place_ret"].to_numpy(), np.nan)
    return rt


def summarize(rt: pd.DataFrame) -> pd.DataFrame:
    g = rt.groupby(level=0)
    hs = g["n"].sum()
    S = pd.DataFrame({"races": g.size(), "horses": hs})
    S["mean_n"] = hs / S["races"]
    S["share_multi"] = (rt["n"] >= 2).groupby(level=0).mean()
    S["horse_win"], S["horse_top3"] = g["win_hs"].sum() / hs, g["top3_hs"].sum() / hs
    for k, c in (("win", "win_any"), ("top2", "top2_any"), ("top3", "top3_any")):
        S[f"race_{k}"] = g[c].mean()
        S[f"exp_{k}"] = g[f"exp_{k}"].mean()
        S[f"lift_{k}"] = S[f"race_{k}"] / S[f"exp_{k}"]
    S["roi_horse_w"], S["roi_horse_p"] = g["sum_wr"].sum() / hs, g["sum_pr"].sum() / hs
    S["roi_race_w"], S["roi_race_p"] = g["mean_wr"].mean(), g["mean_pr"].mean()
    for tag in ("fav", "mod"):
        S[f"roi_{tag}_w"], S[f"roi_{tag}_p"] = g[f"{tag}_wr"].mean(), g[f"{tag}_pr"].mean()
        S[f"n_{tag}"] = g[f"{tag}_wr"].count()
    return S


def _ratio_boot(num: np.ndarray, den: np.ndarray, rng, nb: int = B) -> tuple[float, float, float]:
    """num/den の点推定と 95% 区間（単位 = 要素。NaN の要素は除外）。"""
    m = np.isfinite(num) & np.isfinite(den) & (den > 0)
    num, den = num[m], den[m]
    n = len(num)
    if n < 20:
        return (float(num.sum() / den.sum()) if n else np.nan, np.nan, np.nan)
    est = num.sum() / den.sum()
    out = np.empty(nb)
    for s in range(0, nb, 100):
        e = min(nb, s + 100)
        idx = rng.integers(0, n, (e - s, n))
        out[s:e] = num[idx].sum(1) / den[idx].sum(1)
    return float(est), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _metric_arrays(sub: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    one = np.ones(len(sub))
    n = sub["n"].to_numpy(dtype=float)
    m = {"roi_horse_w": (sub["sum_wr"].to_numpy(), n), "roi_horse_p": (sub["sum_pr"].to_numpy(), n),
         "roi_race_w": (sub["mean_wr"].to_numpy(), one), "roi_race_p": (sub["mean_pr"].to_numpy(), one),
         "roi_fav_w": (sub["fav_wr"].to_numpy(), one), "roi_fav_p": (sub["fav_pr"].to_numpy(), one),
         "race_win": (sub["win_any"].to_numpy(dtype=float), one), "race_top3": (sub["top3_any"].to_numpy(dtype=float), one)}
    return m


def with_ci(rt: pd.DataFrame, S: pd.DataFrame, seed: int = 0, metrics: tuple[str, ...] | None = None, min_races: int = 1) -> pd.DataFrame:
    """S の行ごとに、レース単位ブートストラップの 95% 区間を {指標}_lo / _hi として付ける。"""
    rng = np.random.default_rng(seed)
    rows = {}
    for k, sub in rt.groupby(level=0):
        if len(sub) < min_races:
            continue
        r = {}
        for name, (nu, de) in _metric_arrays(sub).items():
            if metrics and name not in metrics:
                continue
            _, lo, hi = _ratio_boot(nu, de, rng)
            r[f"{name}_lo"], r[f"{name}_hi"] = lo, hi
        rows[k] = r
    return S.join(pd.DataFrame.from_dict(rows, orient="index"))


def reality_check(NUM: np.ndarray, DEN: np.ndarray, min_den: float, nb: int = 500, seed: int = 0) -> dict:
    """行 = 規則、列 = レース。NUM = 基準値に対する超過払戻の合計、DEN = 賭け口数。統計量 = ΣNUM/ΣDEN。"""
    rng = np.random.default_rng(seed)
    den_tot = DEN.sum(1)
    obs = NUM.sum(1) / np.maximum(den_tot, 1)
    valid = den_tot >= min_den
    nR = NUM.shape[1]
    best = int(np.argmax(np.where(valid, obs, -np.inf)))
    mx = []
    for _ in range(nb):
        cnt = np.bincount(rng.integers(0, nR, nR), minlength=nR).astype(float)
        sm, cm = NUM @ cnt, DEN @ cnt
        mx.append(np.max(np.where(valid, sm / np.maximum(cm, 1) - obs, -np.inf)))
    return {"best": best, "best_excess": float(obs[best]), "p": float((np.array(mx) >= obs[best]).mean()), "n_valid": int(valid.sum()), "obs": obs, "den_tot": den_tot}


def build_matrix(rt: pd.DataFrame, races: pd.Index, bases: dict) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """rt の条件ごと × (単勝/複勝) × (馬単位/レース均等/最上位人気) の規則を、レース列の行列に展開する。bases は基準回収率。"""
    names, NUM, DEN = [], [], []
    for k, sub in rt.groupby(level=0):
        ridx = races.get_indexer(sub.index.get_level_values(1))
        for bet in ("w", "p"):
            sw, mean, fav = ("sum_wr", "mean_wr", "fav_wr") if bet == "w" else ("sum_pr", "mean_pr", "fav_pr")
            for sch, (nu, de) in {"horse": (sub[sw].to_numpy() - bases[(bet, "horse")] * sub["n"].to_numpy(), sub["n"].to_numpy(dtype=float)),
                                  "race": (sub[mean].to_numpy() - bases[(bet, "race")], np.ones(len(sub))),
                                  "fav": (sub[fav].to_numpy() - bases[(bet, "fav")], sub[fav].notna().to_numpy(dtype=float))}.items():
                row_n, row_d = np.zeros(len(races)), np.zeros(len(races))
                nu = np.nan_to_num(nu)
                np.add.at(row_n, ridx, nu)
                np.add.at(row_d, ridx, de)
                names.append((k, bet, sch))
                NUM.append(row_n)
                DEN.append(row_d)
    return names, np.vstack(NUM), np.vstack(DEN)


def base_rois(d: pd.DataFrame) -> dict:
    ALL = race_table(d, np.full(len(d), "ALL", dtype=object))
    S = summarize(ALL).iloc[0]
    return {("w", "horse"): S["roi_horse_w"], ("w", "race"): S["roi_race_w"], ("w", "fav"): S["roi_fav_w"],
            ("p", "horse"): S["roi_horse_p"], ("p", "race"): S["roi_race_p"], ("p", "fav"): S["roi_fav_p"]}


# ───────────────────────── 条件の族
def families(d: pd.DataFrame) -> dict[str, np.ndarray]:
    cls, kc, crs = d["F_cls"], d["F_kind_c"], d["F_course"]
    kind = kc.str.split("|").str[0]
    tr = d["trainer_id"].astype(str)
    has = cls.notna()
    fam = {
        "cls": cls,
        "accel": cls.str[0].map({"A": "A(加速)", "B": "B(減速)", "X": "X(終い2Fが13秒以上)"}),
        "kind_accel": (kind + "|" + cls.str[0]).where(has),
        "kind_cls": (kind + "|" + cls).where(has),
        "course_cls": (crs.astype(str) + "|" + cls).where(has),
        "tbin": (kc.astype(str) + "|時計階級" + d["F_tbin"].astype("Int64").astype(str)).where(d["F_tbin"].fillna(-1) >= 0),
        "wk": d["wk_label"],
        "wk_center": (d["wk_kind_c"].astype(str) + "|" + d["wk_label"]).where(d["wk_kind_c"].notna()),
        "trainer_cls": (tr + "|" + cls).where(has),
        "trainer_course_cls": (tr + "|" + crs.astype(str) + "|" + cls).where(has),
        "trainer_wk": (tr + "|" + d["wk_label"]),
    }
    return {k: v.to_numpy(dtype=object) for k, v in fam.items()}


POOLED = ("cls", "accel", "kind_accel", "kind_cls", "course_cls", "tbin", "wk", "wk_center")
CELLS = ("course_cls", "trainer_cls", "trainer_course_cls", "trainer_wk")


# ───────────────────────── 再現性（前年の上位 10% 条件を翌年買う）
def repro(rt: pd.DataFrame, fam: str, nmin: int, bet: str, rng) -> dict:
    rt0 = rt[rt["year"].between(Y0, Y1)]
    ry = rt0.groupby([rt0.index.get_level_values(0), "year"]).agg(n=("n", "size"), horses=("n", "sum"), win_any=("win_any", "mean"), top3_any=("top3_any", "mean"), exp_win=("exp_win", "mean"),
                                                                 hw=("win_hs", "sum"), h3=("top3_hs", "sum"), sw=("sum_wr", "sum"), sp=("sum_pr", "sum"), rw=("mean_wr", "mean"), rp=("mean_pr", "mean"), fw=("fav_wr", "mean"), fp=("fav_pr", "mean"))
    ry.index.names = ["cond", "year"]
    ry = ry.reset_index()
    ry["horse_win"], ry["horse_top3"] = ry["hw"] / ry["horses"], ry["h3"] / ry["horses"]
    ry["roi_horse_w"], ry["roi_horse_p"] = ry["sw"] / ry["horses"], ry["sp"] / ry["horses"]
    ry["race_win_ex"] = ry["win_any"] - ry["exp_win"]
    sel_col = {"w": "rw", "p": "rp"}[bet]
    mets = {"horse_win": "horse_win", "race_win": "win_any", "race_win_ex": "race_win_ex", "horse_top3": "horse_top3", "race_top3": "top3_any", "roi_horse_w": "roi_horse_w", "roi_race_w": "rw", "roi_fav_w": "fw", "roi_race_p": "rp"}
    sp_rows, top_rows, all_rows = [], [], []
    rti = rt0.reset_index().rename(columns={"level_0": "cond", "_k": "cond"})
    for y in range(Y0, Y1):
        a, b = ry[(ry.year == y) & (ry.n >= nmin)], ry[(ry.year == y + 1) & (ry.n >= nmin)]
        m = a.merge(b, on="cond", suffixes=("_0", "_1"))
        if len(m) < 20:
            continue
        sp_rows.append({"year": y, "cells": len(m), **{k: float(m[f"{c}_0"].rank().corr(m[f"{c}_1"].rank())) for k, c in mets.items()}})
        kk = max(int(len(m) * 0.1), 3)
        top = set(m.sort_values(f"{sel_col}_0", ascending=False).head(kk)["cond"])
        nxt = rti[(rti["year"] == y + 1) & rti["cond"].isin(set(m["cond"]))]
        top_rows.append(nxt[nxt["cond"].isin(top)])
        all_rows.append(nxt)
    if not sp_rows:
        return {}
    T, A = pd.concat(top_rows), pd.concat(all_rows)
    res = {"pairs": len(sp_rows), "cells_mean": float(np.mean([r["cells"] for r in sp_rows])),
           "spearman": {k: float(np.nanmean([r[k] for r in sp_rows])) for k in mets}}
    sw, mean, fav = ("sum_wr", "mean_wr", "fav_wr") if bet == "w" else ("sum_pr", "mean_pr", "fav_pr")
    Tcodes = pd.factorize(pd.concat([T["race_id"], A["race_id"]]))[0]
    tc, ac = Tcodes[:len(T)], Tcodes[len(T):]
    nR = Tcodes.max() + 1
    out = {}
    for sch in ("horse", "race", "fav"):
        col = {"horse": sw, "race": mean, "fav": fav}[sch]
        tn, td = (T[col].to_numpy(), T["n"].to_numpy(dtype=float)) if sch == "horse" else (T[col].to_numpy(), np.ones(len(T)))
        an, ad = (A[col].to_numpy(), A["n"].to_numpy(dtype=float)) if sch == "horse" else (A[col].to_numpy(), np.ones(len(A)))
        tm, am = np.isfinite(tn), np.isfinite(an)
        est_t, est_a = tn[tm].sum() / td[tm].sum(), an[am].sum() / ad[am].sum()
        diffs, tb = [], []
        for _ in range(B):
            cnt = np.bincount(rng.integers(0, nR, nR), minlength=nR).astype(float)
            wt_, wa_ = cnt[tc], cnt[ac]
            t_ = (wt_[tm] * tn[tm]).sum() / max((wt_[tm] * td[tm]).sum(), 1e-9)
            a_ = (wa_[am] * an[am]).sum() / max((wa_[am] * ad[am]).sum(), 1e-9)
            tb.append(t_)
            diffs.append(t_ - a_)
        out[sch] = {"top_roi": float(est_t), "all_roi": float(est_a), "top_lo": float(np.percentile(tb, 2.5)), "top_hi": float(np.percentile(tb, 97.5)),
                    "diff_p": float(np.mean(np.array(diffs) <= 0)), "top_rows": int(tm.sum()), "all_rows": int(am.sum())}
    res["next_year"] = out
    return res


# ───────────────────────── run
def cmd_run(a) -> None:
    d0 = pd.read_parquet(OUT / "frame.parquet")
    d = prep(d0)
    d["model_ok"] = d["p_win"].notna()
    dp = d[d["year"].between(Y0, Y1)].reset_index(drop=True)
    rng = np.random.default_rng(7)
    res: dict = {"rows": int(len(dp)), "races": int(dp["race_id"].nunique()), "years": [Y0, Y1], "boot": B}
    bases = base_rois(dp)
    res["bases"] = {f"{b}_{s}": float(v) for (b, s), v in bases.items()}
    _log(f"基準回収率 {res['bases']}")
    fam = families(dp)
    fam_all = families(d)
    pooled_rt, rt_store = {}, {}
    # ---- 1) 条件ごとの表（2020〜24 まとめ、年別）
    for f in POOLED:
        rt = race_table(dp, fam[f])
        pooled_rt[f] = rt
        S = summarize(rt)
        keep = S[S["races"] >= 200].index
        rt_k = rt[rt.index.get_level_values(0).isin(keep)]
        S = S.loc[keep]
        if f in ("course_cls",):
            S = S[S["races"] >= 500]
            rt_k = rt_k[rt_k.index.get_level_values(0).isin(S.index)]
        S = with_ci(rt_k, S, seed=1, metrics=("roi_horse_w", "roi_race_w", "roi_fav_w", "roi_race_p", "roi_fav_p", "race_win", "race_top3"))
        res[f"pooled_{f}"] = json.loads(S.reset_index().rename(columns={"_k": "cond", "index": "cond"}).to_json(orient="records", force_ascii=False))
        _log(f"{f}: {len(S)} 条件")
    res["all_conditions"] = {"races_ge_200": {f: len(res[f"pooled_{f}"]) for f in POOLED}}
    # 全体の参照
    res["reference_all"] = json.loads(summarize(race_table(dp, np.full(len(dp), "ALL", dtype=object))).reset_index().to_json(orient="records"))[0]
    # 年別（2025 は参考）
    yearly = {}
    for f in ("cls", "kind_cls", "wk"):
        rows = []
        keys = [r["cond"] for r in res[f"pooled_{f}"]]
        for y in range(Y0, 2026):
            dy = d[d["year"] == y].reset_index(drop=True)
            rty = race_table(dy, families(dy)[f])
            rty = rty[rty.index.get_level_values(0).isin(keys)]
            Sy = with_ci(rty, summarize(rty), seed=2, metrics=("roi_race_w", "roi_fav_w", "roi_horse_w", "race_win"))
            Sy["year"] = y
            rows.append(Sy.reset_index().rename(columns={"_k": "cond"}))
        yearly[f] = json.loads(pd.concat(rows).to_json(orient="records", force_ascii=False))
    res["yearly"] = yearly
    # ---- 2) 条件族ごとの reality check（2020〜24 のプール。規則 = 条件 × 単複 × 3 通りの買い方）
    races = pd.Index(sorted(dp["race_id"].unique()))
    rc = {}
    allnames, allN, allD = [], [], []
    for f in POOLED:
        rt = pooled_rt[f]
        keep = summarize(rt)
        keep = keep[keep["races"] >= 300].index
        rtk = rt[rt.index.get_level_values(0).isin(keep)]
        nm, N_, D_ = build_matrix(rtk, races, bases)
        allnames += [(f, *x) for x in nm]
        allN.append(N_)
        allD.append(D_)
    N_, D_ = np.vstack(allN), np.vstack(allD)
    r = reality_check(N_, D_, min_den=300, nb=500, seed=3)
    o = r.pop("obs")
    dt = r.pop("den_tot")
    top = np.argsort(-np.where(dt >= 300, o, -9))[:8]
    rc["pooled"] = {"rules_tried": int((dt >= 300).sum()), "rules_total": int(len(dt)), **r, "best_rule": list(allnames[r["best"]]),
                    "top": [{"rule": list(allnames[i]), "excess": float(o[i]), "den": float(dt[i])} for i in top]}
    rb = reality_check(N_, D_, min_den=3000, nb=500, seed=5)
    ob = rb.pop("obs")
    dtb = rb.pop("den_tot")
    rc["pooled_big"] = {"rules_tried": int((dtb >= 3000).sum()), **rb, "best_rule": list(allnames[rb["best"]]),
                        "top": [{"rule": list(allnames[i]), "excess": float(ob[i]), "den": float(dtb[i])} for i in np.argsort(-np.where(dtb >= 3000, ob, -9))[:8]]}
    _log(f"pooled RC: {rc['pooled']['rules_tried']} 規則 最良 {rc['pooled']['best_rule']} 超過 {rc['pooled']['best_excess']:+.3f} p={rc['pooled']['p']:.3f}")
    # ---- 3) as-of 規則（valid 2022〜24、tp-35 bet と同じ 48 規則 × 3 通りの買い方）
    dv = dp[dp["year"].between(2022, 2024)].reset_index(drop=True)
    bases_v = base_rois(dv)
    racesv = pd.Index(sorted(dv["race_id"].unique()))
    keys_all, parts = [], []
    for col, ncol in (("lg_tr_course_cls_roiw", "lg_tr_course_cls_logn"), ("lg_tr_course_cls_roip", "lg_tr_course_cls_logn"), ("lg_tr_cls_roiw", "lg_tr_cls_logn"), ("lg_tr_cls_roip", "lg_tr_cls_logn")):
        x = dv[col].to_numpy(dtype=float)
        n = np.expm1(dv[ncol].to_numpy(dtype=float))
        for delta in (0.1, 0.2, 0.3):
            for nmin in (10, 20):
                s = np.isfinite(x) & (x >= delta) & (n >= nmin)
                lab = f"{col}>={delta}&n>={nmin}"
                keys_all.append(lab)
                parts.append(race_table(dv, np.where(s, lab, None).astype(object)))
    rtv = pd.concat(parts)
    nm, Nv, Dv = build_matrix(rtv, racesv, bases_v)
    # 規則の買い対象（単勝規則は単勝、複勝規則は複勝のみ）に絞る: roiw 列は単勝、roip 列は複勝
    keepi = [i for i, (k, bet, sch) in enumerate(nm) if (("roiw" in k) == (bet == "w"))]
    nm2, Nv2, Dv2 = [nm[i] for i in keepi], Nv[keepi], Dv[keepi]
    rv = reality_check(Nv2, Dv2, min_den=300, nb=500, seed=4)
    ov = rv.pop("obs")
    dtv = rv.pop("den_tot")
    topv = np.argsort(-np.where(dtv >= 300, ov, -9))[:8]
    Sv = summarize(rtv)
    rc["asof"] = {"rules_total": len(nm2), "rules_tried": int((dtv >= 300).sum()), **rv, "best_rule": list(nm2[rv["best"]]),
                  "top": [{"rule": list(nm2[i]), "excess": float(ov[i]), "den": float(dtv[i])} for i in topv],
                  "base": {f"{b}_{s}": float(v) for (b, s), v in bases_v.items()},
                  "summary": json.loads(Sv.reset_index().rename(columns={"_k": "rule"}).to_json(orient="records", force_ascii=False))}
    _log(f"as-of RC: {rc['asof']['rules_tried']} 規則 最良 {rc['asof']['best_rule']} 超過 {rc['asof']['best_excess']:+.3f} p={rc['asof']['p']:.3f}")
    res["reality_check"] = rc
    # ---- 4) 再現性（前年の上位 10% 条件を翌年買う）
    rp = {}
    ntests = 0
    for f in CELLS:
        rt = race_table(dp, fam[f])
        rt_store[f] = rt
        for nmin in (10, 20):
            for bet in ("w", "p"):
                r_ = repro(rt, f, nmin, bet, rng)
                if r_:
                    rp[f"{f}|nmin{nmin}|{bet}"] = r_
                    ntests += 3
                    _log(f"repro {f} n>={nmin} {bet}: " + json.dumps({k: round(v["top_roi"], 3) for k, v in r_["next_year"].items()}) + f" (全体 " + json.dumps({k: round(v["all_roi"], 3) for k, v in r_["next_year"].items()}) + ")")
    res["repro"] = rp
    res["repro_tests"] = ntests
    # セル全体の頭数/レース
    res["cells_mean_n"] = {f: {"cells": int(summarize(rt_store[f]).shape[0]), "mean_n_weighted": float(rt_store[f]["n"].mean()), "share_multi": float((rt_store[f]["n"] >= 2).mean())} for f in CELLS}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log("saved results.json")


# ───────────────────────── レース内の同型頭数を特徴量にできるか（第 1 段だけ）
RACE_CLS_LEVEL = {"A3": 6.0, "A2": 5.0, "A1": 4.0, "B3": 3.0, "B2": 2.0, "B1": 1.0, "X": 0.0}


def cmd_racefeat(a) -> None:
    """LG_race: 同じレースの出走馬の追い切り型（レース前に分かる情報だけ）。結果は一切使わない。"""
    d = pd.read_parquet(OUT / "frame.parquet")
    g = d.groupby("race_id", sort=False)
    known = d["F_cls"].notna()
    n_known = known.groupby(d["race_id"]).transform("sum")
    same = d.groupby(["race_id", "F_cls"])["horse_id"].transform("size")
    f = pd.DataFrame({"race_id": d["race_id"].astype(str), "horse_id": d["horse_id"].astype(str)})
    f["rc_n_same_cls"] = np.where(known, same - 1, np.nan)
    f["rc_share_same_cls"] = np.where(known & (n_known > 1), (same - 1) / np.maximum(n_known - 1, 1), np.nan)
    isA = d["F_cls"].isin(["A1", "A2", "A3"])
    f["rc_n_A"] = np.where(known, isA.groupby(d["race_id"]).transform("sum"), np.nan)
    f["rc_share_A"] = np.where(known & (n_known > 0), isA.groupby(d["race_id"]).transform("sum") / np.maximum(n_known, 1), np.nan)
    lvl = d["F_cls"].map(RACE_CLS_LEVEL)
    f["rc_self_level_pct"] = lvl.groupby(d["race_id"]).rank(pct=True).where(known & (n_known > 1))
    wt = d["wk_t4"]
    nw = wt.notna().groupby(d["race_id"]).transform("sum")
    f["rc_wk_fast_n"] = (wt < 56.0).groupby(d["race_id"]).transform("sum").astype(float).where(wt.notna())
    f["rc_self_wk_rank_pct"] = (-wt).groupby(d["race_id"]).rank(pct=True).where(wt.notna() & (nw > 1))
    for c in f.columns[2:]:
        f[c] = f[c].astype("float32")
    (E.OUT / "groups").mkdir(parents=True, exist_ok=True)
    f.to_parquet(E.OUT / "groups" / "LG_race.parquet", index=False)
    _log(f"LG_race {f.shape} 被覆 " + json.dumps({c: round(float(f[c].notna().mean()), 3) for c in f.columns[2:]}))


def cmd_stage(a) -> None:
    """第 1 段（valid 2022+2023、seed 1 本）。基準 V4 の OOF は stage1/base.parquet を流用する。"""
    import shutil

    d = E.OUT / "stage1_race"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "base.parquet").exists():
        shutil.copy(E.OUT / "stage1" / "base.parquet", d / "base.parquet")
        base = E._jload(E.OUT / "stage1" / "results.json")["base"]
        E._jsave(d / "results.json", {"base": base, "groups": {}})
    E._stage("stage1_race", [2022, 2023], [0], ["LG_race"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "racefeat", "stage", "chapter"])
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "racefeat": cmd_racefeat, "stage": cmd_stage, "chapter": lambda a: cmd_chapter(a)}[a.cmd](a)


# ───────────────────────── chapter
MARK0, MARK1 = "<!-- tp35-race-level:begin -->", "<!-- tp35-race-level:end -->"


def _p(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}%"


def _ci(r, k):
    lo, hi = r.get(f"{k}_lo"), r.get(f"{k}_hi")
    return "" if lo is None or hi is None or not np.isfinite(lo) else f" <span class=muted>[{100 * lo:.0f}–{100 * hi:.0f}]</span>"


def _tab(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td class=num>{c}</td>" if i else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def _cond_rows(recs: list[dict], order: list[str] | None = None) -> list[list[str]]:
    if order:
        recs = sorted(recs, key=lambda r: order.index(r["cond"]) if r["cond"] in order else 99)
    out = []
    for r in recs:
        out.append([_html.escape(r["cond"]), f"{int(r['races']):,}", f"{r['mean_n']:.2f}", _p(r["horse_win"]), _p(r["race_win"]), _p(r["exp_win"]), f"{r['lift_win']:.2f}",
                    _p(r["race_top3"]), _p(r["exp_top3"]), f"{r['lift_top3']:.2f}",
                    _p(r["roi_horse_w"]) + _ci(r, "roi_horse_w"), _p(r["roi_race_w"]) + _ci(r, "roi_race_w"), _p(r["roi_fav_w"]) + _ci(r, "roi_fav_w"), _p(r.get("roi_mod_w")),
                    _p(r["roi_race_p"]) + _ci(r, "roi_race_p"), _p(r["roi_fav_p"]) + _ci(r, "roi_fav_p")])
    return out


HEAD = ["条件", "該当レース", "1R平均頭数", "馬単位勝率", "レース勝率", "期待値(無作為)", "比", "レース複勝率", "期待値", "比",
        "単勝: 馬単位", "単勝: レース均等", "単勝: 最上位人気1頭", "単勝: モデル最上位", "複勝: レース均等", "複勝: 最上位人気1頭"]


def chapter_html(r: dict) -> str:
    ref = r["reference_all"]
    bs = r["bases"]
    cls_order = ["A1", "A2", "A3", "B1", "B2", "B3", "X"]
    parts = [f"<h2>8. レース単位で見直す（母数を『該当馬の頭数』から『該当馬がいるレース』に揃えた集計）</h2>",
             "<p>前の節までの勝率・回収率は、該当馬を 1 頭ずつ数える『馬単位』の集計だった。同じレースに該当馬が複数いると、馬単位の勝率はそのぶん薄まって見える"
             "（参考書籍の指摘どおり。例: 8 枠が 2 頭出ているレースで 10 回とも 8 枠が勝つと、馬単位では 50% と表示される）。そこで、条件に当てはまる馬が 1 頭以上いるレースを 1 単位として取り直した。</p>",
             "<ul><li><b>レース勝率</b> = 該当馬のいずれかが 1 着のレース ÷ 該当馬がいるレース（複勝は 3 着以内）。そのまま比べると頭数が多い条件ほど高く出るので、"
             "『該当馬を無作為に選んだ場合の期待値』（超幾何分布。出走 F 頭・該当 n 頭で 1−C(F−n,k)/C(F,k)）と、その比も出した。比が 1 を超えれば、頭数では説明できない上積みがある。</li>"
             "<li><b>回収率</b>は買い方を 4 通りで出した: <b>馬単位</b>＝該当馬を 1 頭 100 円ずつ全頭買う／<b>レース均等</b>＝1 レースの賭け金を一定にして該当馬に等分／"
             "<b>最上位人気 1 頭</b>＝該当馬のうち確定単勝オッズが最小の 1 頭だけ／<b>モデル最上位</b>＝V4 の勝率予測が最大の 1 頭（2022 年以降のみ。参考）。"
             "区間（[ ]）はレース単位のブートストラップの 95%。</li></ul>",
             f"<p class=muted>対象: {r['years'][0]}〜{r['years'][1]} 年の {r['races']:,} レース・{r['rows']:,} 頭（払戻を解釈できたもののみ）。全馬・全レースの基準回収率: 単勝 馬単位 {_p(bs['w_horse'])} ／ レース均等 {_p(bs['w_race'])} ／ 最上位人気 {_p(bs['w_fav'])}、"
             f"複勝 馬単位 {_p(bs['p_horse'])} ／ レース均等 {_p(bs['p_race'])} ／ 最上位人気 {_p(bs['p_fav'])}。オッズは race_odds 由来の確定値（2025 年の race_result_flat のオッズは使わない）。2025 年は条件の選択に使わず、参考列だけに出す。</p>"]
    parts.append("<h3>8-1. 最終追い切りのラップ種別（加速 A／減速 B／終い 2F が 13 秒以上 X）</h3>")
    parts.append(_tab(HEAD, _cond_rows(r["pooled_accel"]) + _cond_rows(r["pooled_cls"], cls_order)))
    a, b = (next(x for x in r["pooled_accel"] if x["cond"].startswith(k)) for k in ("A(", "B("))
    parts.append(f"<p>加速（A）の馬が 1 頭以上いるレースの {_p(a['race_win'])} で、そのいずれかが勝つ。馬単位の勝率 {_p(a['horse_win'])} の約 {a['race_win'] / a['horse_win']:.0f} 倍の見え方になるが、"
                 f"これは 1 レースに平均 {a['mean_n']:.1f} 頭が該当するため。無作為に選んだ場合の期待値 {_p(a['exp_win'])} と比べると比は {a['lift_win']:.2f} で、減速（B）は {b['lift_win']:.2f}。"
                 f"頭数の影響を除いても A は B より勝ちやすい向きは変わらないが、差は馬単位で見たときと同程度（馬単位勝率 {_p(a['horse_win'])} 対 {_p(b['horse_win'])}）。"
                 f"回収率はレース均等でも、A {_p(a['roi_race_w'])}{_ci(a, 'roi_race_w')}・B {_p(b['roi_race_w'])}{_ci(b, 'roi_race_w')} と、100% には届かない。</p>")
    parts.append("<h3>8-2. 調教コース種別 × ラップ種別（200 レース以上）</h3>")
    parts.append(_tab(HEAD, _cond_rows(sorted(r["pooled_kind_cls"], key=lambda x: x["cond"]))))
    parts.append("<h3>8-3. 前週の土日の坂路最速 4F</h3>")
    wk_order = ["坂路<54", "坂路54-56", "坂路56-58", "坂路58-60", "坂路>=60", "土日に坂路なし(他調教のみ)", "土日に調教なし"]
    parts.append(_tab(HEAD, _cond_rows(r["pooled_wk"], wk_order)))
    parts.append("<h3>8-4. 回収率の高かった条件は偶然か（reality check）</h3>")
    rc = r["reality_check"]
    pb, pg, asof = rc["pooled"], rc["pooled_big"], rc["asof"]

    def rule(x):
        k, cond, bet, sch = x["best_rule"] if "best_rule" in x else x["rule"]
        return f"{_html.escape(cond)}（{'単勝' if bet == 'w' else '複勝'}・{SCHEMES[sch]}）"

    base_name = {("w", "race"): bs["w_race"]}
    parts.append(_tab(["試した規則の集合", "規則数", "最良の規則", "基準からの超過", "reality check p"],
                      [["全条件（300 レース以上の条件 × 単勝/複勝 × 3 通りの買い方、2020〜24 年のプール）", f"{pb['rules_tried']}", rule(pb), f"{pb['best_excess']:+.3f}", f"{pb['p']:.3f}"],
                       ["上のうち 3,000 口以上の条件だけ", f"{pg['rules_tried']}", rule(pg), f"{pg['best_excess']:+.3f}", f"{pg['p']:.3f}"],
                       ["過去の調教師別 ROI が高い組を買う規則（valid 2022〜24、各レース日より前の推定）", f"{asof['rules_tried']}",
                        f"{asof['best_rule'][0]}（{'単勝' if asof['best_rule'][1] == 'w' else '複勝'}・{SCHEMES[asof['best_rule'][2]]}）", f"{asof['best_excess']:+.3f}", f"{asof['p']:.3f}"]]))
    lows = [(x[f"{m}_lo"], f, x["cond"], m) for f in POOLED for x in r[f"pooled_{f}"] for m in ("roi_horse_w", "roi_race_w", "roi_fav_w", "roi_race_p", "roi_fav_p") if np.isfinite(x.get(f"{m}_lo", np.nan))]
    lo_best = max(lows)
    mname = {"roi_horse_w": "単勝・馬単位", "roi_race_w": "単勝・レース均等", "roi_fav_w": "単勝・最上位人気", "roi_race_p": "複勝・レース均等", "roi_fav_p": "複勝・最上位人気"}
    n_hi = sum(1 for x in lows if x[0] > 1.0)
    parts.append(f"<p><b>区間の下限が 100% を超えた条件は {n_hi} 個</b>（全条件 × 5 通りの買い方 {len(lows)} 通りのうち）。下限の最大は {_p(lo_best[0], 0)}（{_html.escape(lo_best[2])}・{mname[lo_best[3]]}）。</p>")
    parts.append(f"<p>最良の規則でも p は 0.15 以上で、5% では偶然を排除できない。小さい条件（例: 南W の B1、545 レース）の高い回収率は区間が極端に広い（{_p(next(x for x in r['pooled_course_cls'] if x['cond'] == '南W|B1')['roi_race_w_lo'], 0)}〜"
                 f"{_p(next(x for x in r['pooled_course_cls'] if x['cond'] == '南W|B1')['roi_race_w_hi'], 0)}）。「基準からの超過」は各買い方の全体の回収率との差（買い方ごとに基準が違う）。</p>")
    # 年別の確認（目立った 2 条件）
    ydefs = [("kind_cls", "slope|A3", "坂路の A3（最終追い切り）"), ("wk", "坂路54-56", "前週土日の坂路 54〜56 秒")]
    yrows = []
    for fam, cond, lab in ydefs:
        ys = {x["year"]: x for x in r["yearly"][fam] if x["cond"] == cond}
        yrows.append([_html.escape(lab)] + [(_p(ys[y]["roi_race_w"], 0) + _ci(ys[y], "roi_race_w")) if y in ys else "-" for y in range(2020, 2026)])
    parts.append("<p>目立った 2 条件（レース均等の単勝回収率）の年別。2025 年は条件を選んだ後の参考値。</p>")
    parts.append(_tab(["条件"] + [f"{y}{'（参考）' if y == 2025 else ''}" for y in range(2020, 2026)], yrows))
    parts.append("<h3>8-5. 調教師別・コース別の傾向は翌年も続くか（レース単位）</h3>")
    fam_names = {"course_cls": "調教コース×種別", "trainer_cls": "調教師×種別", "trainer_course_cls": "調教師×調教コース×種別", "trainer_wk": "調教師×前週土日の時計階級"}
    rows5 = []
    for fam in ("course_cls", "trainer_cls", "trainer_course_cls", "trainer_wk"):
        k = f"{fam}|nmin10|w"
        v = r["repro"].get(k)
        cm = r["cells_mean_n"][fam]
        if not v:
            continue
        nx = v["next_year"]
        rows5.append([fam_names[fam], f"{cm['mean_n_weighted']:.2f}", _p(cm["share_multi"]), f"{v['spearman']['horse_win']:+.2f}", f"{v['spearman']['race_win_ex']:+.2f}", f"{v['spearman']['roi_horse_w']:+.2f}", f"{v['spearman']['roi_race_w']:+.2f}",
                      _p(nx["race"]["top_roi"]) + f" <span class=muted>[{100 * nx['race']['top_lo']:.0f}–{100 * nx['race']['top_hi']:.0f}]</span>", _p(nx["race"]["all_roi"]), f"{nx['race']['diff_p']:.2f}"])
    parts.append(_tab(["条件の作り方", "セルあたり平均頭数", "複数頭のレース割合", "順位相関 馬単位勝率", "順位相関 レース勝率(期待値差)", "順位相関 馬単位回収率", "順位相関 レース均等回収率",
                       "前年の回収率上位10%を翌年買う(レース均等)", "全セル平均", "上位−全体 ≤0 の p"], rows5))
    best_p = min((x["diff_p"], k, s) for k, v in r["repro"].items() for s, x in v["next_year"].items())
    parts.append(f"<p>調教師を含むセルは 1 レースに平均 1.01 頭しか該当しない（複数頭は 1〜2%）ので、馬単位とレース単位はほぼ同じ数字になる（順位相関も同じ）。レース単位に直して意味が出るのは、"
                 f"種別・時計階級のように 1 レースに複数頭が入る条件（8-1〜8-3）。前年の上位を翌年買っても全体を上回らず（検定 {r['repro_tests']} 通りで最小の片側 p = {best_p[0]:.2f}、"
                 f"補正後は 1 に近い）、回収率の傾向は翌年に続かない。</p>")
    st = Path(E.OUT / "stage1_race" / "results.json")
    if st.exists():
        sg = json.loads(st.read_text(encoding="utf-8")).get("groups", {}).get("LG_race")
        if sg:
            c = sg["vs_V4"]
            parts.append(f"<h3>8-6. 『同じレースに同型が何頭いるか』を特徴量にできるか</h3><p>同じレースの出走馬の追い切り型（同型の他馬の頭数・割合、加速型の頭数、自分の型の相対順位、前週土日が速い馬の頭数など 7 列。レース前に分かる情報だけ）を足して、"
                         f"V4 全体に対する勝率 LogLoss の差を測った（第 1 段: valid 2022+2023、seed 1 本）。Δwin {c['Δwin']:+.5f}（CI 上限 {c['win_hi']:+.5f}）、Δtop2 {c['Δtop2']:+.5f}、Δtop3 {c['Δtop3']:+.5f} で、"
                         f"{'採用基準を満たさない（不採用）' if not (c['Δwin'] < -0.00012 and c['win_hi'] < 0.0001) else '第 2 段へ進む'}。</p>")
    parts.append("<div class=note><b>結論（レース単位）.</b> 母数をレースに揃えると、条件に複数頭が入る種別（A1 など）のレース勝率は馬単位の 3〜4 倍に見えるが、頭数で説明できる分を除いた上積みは馬単位で見たときと同じ向きと大きさだった。"
                 "レース単位の回収率（均等・最上位人気）も、区間の下限が 100% を超える条件はなく、reality check でも偶然を排除できない。特徴量・馬券規則の採用/不採用の判断は変わらない。</div>")
    return MARK0 + "\n" + "\n".join(parts) + "\n" + MARK1


def cmd_chapter(a) -> None:
    r = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    html = CHAPTER.read_text(encoding="utf-8")
    block = chapter_html(r)
    if MARK0 in html:
        html = re.sub(re.escape(MARK0) + ".*?" + re.escape(MARK1), lambda m: block, html, flags=re.S)
    else:
        html = html.replace("</body></html>", block + "\n</body></html>")
    tail = "（7 節）。</div>"
    add = " 母数を『該当馬の頭数』から『該当馬がいるレース』に取り直しても（8 節）、区間の下限が 100% を超える条件はなく、判断は同じです。"
    if "8 節）" not in html and tail in html:
        html = html.replace(tail, "（7 節）。" + add + "</div>", 1)
    CHAPTER.write_text(html, encoding="utf-8")
    _log(f"chapter 更新 {CHAPTER}")


if __name__ == "__main__":
    main()
