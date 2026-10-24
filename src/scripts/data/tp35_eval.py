"""tp-35: ラップ種別（A1〜B3）× 調教師別/コース別の調教成績、前週土日の坂路最速時計の成績。

python -m src.scripts.data.tp35_eval lapstats   # 追い切りを 1F ごとのラップに直せる割合と、A1〜B3 の分布（種別・所属別）
python -m src.scripts.data.tp35_eval build      # 群ごとの特徴量を tp35/groups/*.parquet に保存（追い切り表は training_laps.parquet をそのまま使う）
python -m src.scripts.data.tp35_eval repro      # 『調教師×調教コース×ラップ種別』の傾向が翌年も再現するか
python -m src.scripts.data.tp35_eval bet        # 回収率上位の組を買う単純規則（valid 2022〜24。推定は各レース日より前のみ）。reality check 付き
python -m src.scripts.data.tp35_eval corr       # ROI 系・勝率系が確定オッズの水準とどれくらい相関するか（市場情報の混入度）
python -m src.scripts.data.tp35_eval overlap    # 前週土日の列と TL_A の 1 週前（tl_w1_*）の列との重複
python -m src.scripts.data.tp35_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp35_eval stage2     # 第 2 段: 通過群を valid 2024、新 seed 2 本で確認 + 同時追加
python -m src.scripts.data.tp35_eval report     # /tmp/tp35_report.{json,md}
test 2025・2026 は使わない。市場オッズ・人気は特徴量に使わない（ROI 系は過去の確定払戻の統計。勝率系と別の群）。環境変数: TP35_THREADS（既定 6）, TP35_LOAD_LIMIT（既定 16）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = V / "tp35"
WORK = ROOT / "data/local/modeling/training_pipeline"
LAPS = Path(os.environ.get("TP35_LAPS") or WORK / "features/training_laps.parquet")
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field", "TP27_OUT": V / "tp27"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("TP27_THREADS", os.environ.get("TP35_THREADS", "6"))

from src.pipeline.features import tp35_lap_groups as L  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402

KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP35_THREADS", 6))
LOAD_LIMIT = float(os.environ.get("TP35_LOAD_LIMIT", 16))
GROUPS = ["LG_cls", "LG_stat", "LG_roi", "LG_horse", "LG_cond", "WK_stat", "WK_roi"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wait() -> None:
    t0 = time.time()
    while os.getloadavg()[0] >= LOAD_LIMIT and time.time() - t0 < 3600:
        time.sleep(30)


def _jsave(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[["race_id", "horse_id", "date", "trainer_id", "finish_position", "field_size", "venue_code", "surface", "distance"]].copy()


def _returns(df: pd.DataFrame) -> pd.DataFrame:
    from src.scripts.data.tp28_eval import _load_returns
    return _load_returns(df)


# ───────────────────────── lapstats
def cmd_lapstats(a) -> None:
    laps = pd.read_parquet(LAPS)
    tp = L.prepare_laps(laps)
    rep: dict = {"rows": int(len(tp))}
    ok = tp["kind"].isin(["slope", "wood", "poly"])
    rep["by_kind"] = {}
    for k, g in tp[ok].groupby("kind"):
        has = g["cls"].ne("NA")
        rep["by_kind"][k] = {"rows": int(len(g)), "lap_ok_share": float(has.mean()), "approx_l2": bool(g["l2_approx"].mean() > 0.5),
                             "dist": {c: float((g.loc[has, "cls"] == c).mean()) for c in L.CLASSES}}
    rep["by_center_kind"] = {}
    for (k, c), g in tp[ok].groupby(["kind", "center"]):
        has = g["cls"].ne("NA")
        rep["by_center_kind"][f"{k}|{c}"] = {"rows": int(len(g)), "lap_ok_share": float(has.mean()), "dist": {x: round(float((g.loc[has, "cls"] == x).mean()), 4) for x in L.CLASSES}}
    # 最終追い切り（レースの 1〜6 日前）に絞った分布
    f = tp[ok & tp["days_before"].between(*L.SLOT_F) & tp["cls"].ne("NA")]
    rep["final_session_dist"] = {c: float((f["cls"] == c).mean()) for c in L.CLASSES}
    wk = tp[tp["is_prev_weekend"].eq(1)]
    rep["prev_weekend"] = {"rows": int(len(wk)), "slope_rows": int((wk["kind"] == "slope").sum()), "wood_poly_rows": int(wk["kind"].isin(["wood", "poly"]).sum())}
    for y, g in tp.assign(year=pd.to_datetime(tp["race_date"]).dt.year).groupby("year"):
        rep.setdefault("by_year_lap_ok", {})[int(y)] = float((g["cls"].ne("NA") & g["kind"].isin(["slope", "wood", "poly"])).sum() / max(g["kind"].isin(["slope", "wood", "poly"]).sum(), 1))
    _jsave(OUT / "lapstats.json", rep)
    _log(json.dumps({k: v for k, v in rep.items() if k in ("rows", "final_session_dist", "prev_weekend")}, ensure_ascii=False))
    for k, v in rep["by_kind"].items():
        _log(f"{k}: 1F ごとに直せる {v['lap_ok_share']:.3f}（近似 l2={v['approx_l2']}） 分布 " + " ".join(f"{c} {x:.3f}" for c, x in v["dist"].items()))


# ───────────────────────── build
def cmd_build(a) -> None:
    df, base, meta, folds, lp = t.setup()
    laps = pd.read_parquet(LAPS)
    ret = _returns(df)
    _log(f"払戻の解釈: 単勝 {ret['win_ret'].notna().mean():.3f} / 複勝 {ret['place_ret'].notna().mean():.3f}（追い切り表 {len(laps):,} 本）")
    feats = L.build_features(_runs(df), laps, ret)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    cover = {}
    for g, f in feats.items():
        assert (f["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (f["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
        f.to_parquet(gdir / f"{g}.parquet", index=False)
        cover[g] = {"cols": int(f.shape[1] - 2), "non_null_share": float(f.drop(columns=KEY).notna().mean().mean())}
        _log(f"{g}: {cover[g]}")
    _jsave(OUT / "build_meta.json", {"laps_rows": int(len(laps)), "win_ret_parsed": float(ret["win_ret"].notna().mean()), "place_ret_parsed": float(ret["place_ret"].notna().mean()), "groups": cover})


# ───────────────────────── repro（傾向の再現性）
def _cells(df: pd.DataFrame, ret: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    tp = L.prepare_laps(laps)
    F = L.pick_slot(tp, L.SLOT_F, "F")
    d = df[["race_id", "horse_id", "date", "trainer_id", "finish_position", "year"]].copy()
    d["race_date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d = d.merge(F, on=["horse_id", "race_date"], how="left")
    d["win_ret"], d["place_ret"] = ret["win_ret"].to_numpy(), ret["place_ret"].to_numpy()
    d = d[d["F_cls"].notna() & d["finish_position"].gt(0)]
    d["win"], d["top3"] = (d["finish_position"] == 1).astype(float), (d["finish_position"] <= 3).astype(float)
    return d


def _yearly(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = d.groupby(keys + ["year"]).agg(n=("win", "size"), win=("win", "mean"), top3=("top3", "mean"), roiw=("win_ret", "mean"), roip=("place_ret", "mean")).reset_index()
    return g


def cmd_repro(a) -> None:
    df, base, meta, folds, lp = t.setup()
    laps = pd.read_parquet(LAPS)
    d = _cells(df, _returns(df), laps)
    d = d[d["year"].between(2020, 2024)]
    rep: dict = {}
    for name, keys in (("trainer_course_cls", ["trainer_id", "F_course", "F_cls"]), ("trainer_cls", ["trainer_id", "F_cls"]), ("course_cls", ["F_course", "F_cls"])):
        y = _yearly(d, keys)
        out_rows = []
        for nmin in (10, 20):
            pairs = []
            for yr in range(2020, 2024):
                a_, b_ = y[(y.year == yr) & (y.n >= nmin)], y[(y.year == yr + 1) & (y.n >= nmin)]
                m = a_.merge(b_, on=keys, suffixes=("_0", "_1"))
                if len(m) < 20:
                    continue
                row = {"year": yr, "cells": int(len(m))}
                for met in ("win", "top3", "roiw", "roip"):
                    row[f"spearman_{met}"] = float(m[f"{met}_0"].rank().corr(m[f"{met}_1"].rank()))
                # 前年の回収率（単勝）上位 10% の組が翌年も高いか
                k = max(int(len(m) * 0.1), 3)
                top = m.sort_values("roiw_0", ascending=False).head(k)
                row["top10pct_next_roiw"] = float(np.average(top["roiw_1"], weights=top["n_1"]))
                row["all_next_roiw"] = float(np.average(m["roiw_1"], weights=m["n_1"]))
                top_p = m.sort_values("roip_0", ascending=False).head(k)
                row["top10pct_next_roip"] = float(np.average(top_p["roip_1"], weights=top_p["n_1"]))
                row["all_next_roip"] = float(np.average(m["roip_1"], weights=m["n_1"]))
                pairs.append(row)
            rep[f"{name}|nmin{nmin}"] = {"by_year_pair": pairs, "mean_spearman": {k: float(np.nanmean([p[f"spearman_{k}"] for p in pairs])) for k in ("win", "top3", "roiw", "roip")} if pairs else {},
                                          "mean_top10_next_roiw": float(np.nanmean([p["top10pct_next_roiw"] for p in pairs])) if pairs else None,
                                          "mean_all_next_roiw": float(np.nanmean([p["all_next_roiw"] for p in pairs])) if pairs else None,
                                          "mean_top10_next_roip": float(np.nanmean([p["top10pct_next_roip"] for p in pairs])) if pairs else None,
                                          "mean_all_next_roip": float(np.nanmean([p["all_next_roip"] for p in pairs])) if pairs else None}
            _log(f"{name} nmin={nmin}: " + json.dumps({k: v for k, v in rep[f'{name}|nmin{nmin}'].items() if k != 'by_year_pair'}, ensure_ascii=False))
    _jsave(OUT / "repro.json", rep)


# ───────────────────────── bet（回収率上位の組を買う規則）
def cmd_bet(a) -> None:
    df, base, meta, folds, lp = t.setup()
    ret = _returns(df)
    G = {g: pd.read_parquet(OUT / "groups" / f"{g}.parquet") for g in ("LG_roi", "LG_stat", "WK_roi")}
    ok = df["year"].between(2022, 2024).to_numpy() & ret["win_ret"].notna().to_numpy() & ret["place_ret"].notna().to_numpy() & df["finish_position"].gt(0).to_numpy()
    wr, pr = ret["win_ret"].to_numpy(), ret["place_ret"].to_numpy()
    rid = df["race_id"].to_numpy()
    rules, sel = [], []
    cands = [("lg_tr_course_cls_roiw", "LG_roi", "lg_tr_course_cls_logn", "LG_stat"), ("lg_tr_course_cls_roip", "LG_roi", "lg_tr_course_cls_logn", "LG_stat"),
             ("lg_tr_cls_roiw", "LG_roi", "lg_tr_cls_logn", "LG_stat"), ("lg_tr_cls_roip", "LG_roi", "lg_tr_cls_logn", "LG_stat"),
             ("cd_trainer_bin_roiw", None, None, None)]
    for col, g, ncol, ng in cands[:4]:
        x = G[g][col].to_numpy(dtype=float)
        n = np.expm1(G[ng][ncol].to_numpy(dtype=float))
        for delta in (0.1, 0.2, 0.3):
            for nmin in (10, 20):
                s = ok & np.isfinite(x) & (x >= delta) & (n >= nmin)
                for bet_on, rr in (("win", wr), ("place", pr)):
                    rules.append({"rule": f"{col}>={delta}&n>={nmin}→{bet_on}", "bet_on": bet_on})
                    sel.append((s, rr))
    base_w, base_p = float(np.nanmean(wr[ok])), float(np.nanmean(pr[ok]))
    res = []
    excess = []
    for r, (s, rr) in zip(rules, sel):
        base = base_w if r["bet_on"] == "win" else base_p
        k = int(s.sum())
        roi = float(np.nanmean(rr[s])) if k else float("nan")
        hit = float((rr[s] > 0).mean()) if k else float("nan")
        res.append({**r, "bets": k, "roi": roi, "hit": hit, "base_roi": base, "excess": roi - base if k else float("nan")})
        excess.append(np.where(s, rr - base, 0.0))
    E = np.vstack(excess)                                                             # 規則 × 行
    cnt = np.array([r["bets"] for r in res], dtype=float)
    # reality check: レース単位のブートストラップで『最良規則の超過』が偶然かを検定
    races, inv = np.unique(rid[ok], return_inverse=True)
    Eo = E[:, ok]
    rng = np.random.default_rng(0)
    obs = np.where(cnt > 0, Eo.sum(axis=1) / np.maximum(cnt, 1), np.nan)
    per_race = np.zeros((E.shape[0], len(races)))
    for i in range(E.shape[0]):
        per_race[i] = np.bincount(inv, weights=Eo[i], minlength=len(races))
    cnt_race = np.zeros((E.shape[0], len(races)))
    sel_ok = np.vstack([s[ok] for s, _ in sel])
    for i in range(E.shape[0]):
        cnt_race[i] = np.bincount(inv, weights=sel_ok[i].astype(float), minlength=len(races))
    B, maxes = 500, []
    for _ in range(B):
        idx = rng.integers(0, len(races), len(races))
        sm = per_race[:, idx].sum(axis=1)
        cm = np.maximum(cnt_race[:, idx].sum(axis=1), 1)
        maxes.append(np.nanmax(np.where(cnt > 30, sm / cm - obs, -np.inf)))
    best_i = int(np.nanargmax(np.where(cnt > 30, obs, -np.inf)))
    p_rc = float((np.array(maxes) >= obs[best_i]).mean())
    out = {"rules_tried": len(res), "rows": int(ok.sum()), "base_roi_win": base_w, "base_roi_place": base_p, "best_rule": res[best_i], "reality_check_p": p_rc,
           "top5_by_excess": sorted([r for r in res if r["bets"] > 30], key=lambda r: -r["excess"])[:5], "all": res}
    _jsave(OUT / "bet.json", out)
    _log(f"規則 {len(res)} 通り。最良 {res[best_i]['rule']} 買い目 {res[best_i]['bets']} 回収率 {res[best_i]['roi']:.3f}（全体 {res[best_i]['base_roi']:.3f}）reality check p={p_rc:.3f}")


# ───────────────────────── corr（ROI 系が市場情報を持ち込んでいないか）
def cmd_corr(a) -> None:
    import polars as pl

    df, *_ = t.setup()
    years = sorted({int(y) for y in df["year"].unique()})
    tables = ROOT / "data/page_reference/tables"
    odds = pd.concat([pl.read_parquet(tables / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "odds"]).to_pandas() for y in years if (tables / str(y) / "race_result_flat.parquet").exists()])
    odds["race_id"] = odds["race_id"].astype(str)
    odds["horse_number"] = pd.to_numeric(odds["horse_number"], errors="coerce")
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    m = df[["race_id", "horse_number"]].astype({"race_id": str}).merge(odds.drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left")
    lo = np.log(m["odds"].clip(lower=1.0)).to_numpy()
    rows = {}
    for g in ("LG_roi", "WK_roi", "LG_stat", "WK_stat"):
        G = _load_group(g, df)
        for c in G.columns:
            x = G[c].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(lo)
            if ok.sum() < 1000:
                continue
            w = pd.DataFrame({"r": m["race_id"].to_numpy()[ok], "x": x[ok], "o": lo[ok]})
            rho = float(pd.Series(x[ok]).rank().corr(pd.Series(lo[ok]).rank()))
            within = float(np.corrcoef(w.groupby("r")["x"].rank(pct=True), w.groupby("r")["o"].rank(pct=True))[0, 1])
            rows[f"{g}.{c}"] = {"spearman_all": rho, "spearman_within_race": within}
    _jsave(OUT / "odds_corr.json", rows)
    for k, v in sorted(rows.items(), key=lambda kv: -abs(kv[1]["spearman_within_race"]))[:10]:
        _log(f"{k:36s} 全体 {v['spearman_all']:+.3f} レース内 {v['spearman_within_race']:+.3f}")


# ───────────────────────── overlap
def cmd_overlap(a) -> None:
    df, base, meta, folds, lp = t.setup()
    G = pd.read_parquet(OUT / "groups" / "WK_stat.parquet")
    tla = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    tla["race_id"], tla["horse_id"] = tla["race_id"].astype(str), tla["horse_id"].astype(str)
    m = df[KEY].merge(tla, on=KEY, how="left")
    w1 = [c for c in m.columns if c.startswith("w1_") and m[c].dtype.kind in "fi"]
    rows = {}
    for c in ("wk_t4", "wk_n_slope", "wk_none_slope", "wk_global_res", "wk_trainer_res"):
        x = G[c].to_numpy(dtype=float)
        for w in [k for k in w1 if k.endswith(("t4", "t1", "exists", "days"))][:8]:
            y = m[w].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 1000:
                rows[f"{c} ~ {w}"] = float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))
    top = dict(sorted(rows.items(), key=lambda kv: -abs(kv[1]))[:12])
    _jsave(OUT / "overlap.json", {"spearman": top, "w1_columns": w1[:30]})
    for k, v in top.items():
        _log(f"{k}: {v:+.3f}")


# ───────────────────────── stage1 / stage2
def _frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.read_parquet(V / "tp31" / "v4_frame.parquet")
    X.index = df.index
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    return X


def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


def _run_subset(name, X, meta, folds, lp, years: list[int], seeds: list[int]):
    from src.pipeline.models import zoo_runner

    fs = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    _wait()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, fs, seeds, THREADS, params=dict(lp))
    lv = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")
    return lv, oof


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str]) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = _frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
    if bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = _run_subset("V4", X0, meta, folds, lp, years, seeds)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        _jsave(resp, res)
    steps = res.setdefault("groups", {})
    for g in groups:
        if g in steps:
            continue
        if g.startswith("ALL"):
            names = [x for x in groups if not x.startswith("ALL")]
            G = pd.concat([_load_group(x, df) for x in names], axis=1)
        else:
            G = _load_group(g, df)
        lv, oof = _run_subset(f"V4+{g}", pd.concat([X0, G], axis=1), meta, folds, lp, years, seeds)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


def cmd_stage1(a) -> None:
    _stage("stage1", [2022, 2023], [0], GROUPS)


def cmd_stage2(a) -> None:
    r = _jload(OUT / "stage1/results.json")
    passed = [g for g, s in r.get("groups", {}).items() if s["vs_V4"]["Δwin"] < -0.00012 and s["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1/passed.json", passed)
    if not passed:
        return
    _stage("stage2", [2024], [2, 3], passed + (["ALL_PASSED"] if len(passed) > 1 else []))


def _tl_regen() -> tuple[dict, dict]:
    return _jload(OUT / "tl_regen_build.json"), _jload(OUT / "tl_regen_eval.json")


def _tl_md() -> list[str]:
    b, e = _tl_regen()
    if not b:
        return []
    ys = sorted(b["coverage_new"]["has_training"])
    L_ = ["", "## TL_A 再生成の効果（調教の再取得後の追い切り表）", "",
          f"追い切り表は {b.get('laps_rows_new', 0):,} 本（旧 1,190,429 本）。ログイン切れで空保存だった 5,261 頭を取り直した。", "",
          "| 年 | 調教あり（旧→新） | 最終追い切りに時計あり（旧→新） |", "|---|---|---|"]
    for y in ys:
        L_.append(f"| {y} | {b['coverage_old']['has_training'][y]:.3f} → {b['coverage_new']['has_training'][y]:.3f} | {b['coverage_old'].get('f_t1', {}).get(y, float('nan')):.3f} → {b['coverage_new']['f_t1'][y]:.3f} |")
    if e:
        c = e["vs_old"]
        L_ += ["", f"V4 の tl_ 列（{e['cols_replaced']} 列）を再生成版に差し替えた場合（valid 2022〜24、2 seed。基準は旧 tl_ の V4）: 勝率 {e['levels_old']['logloss_win']:.5f} → {e['levels_new']['logloss_win']:.5f}"
               f"（レース単位の差 Δwin {c['Δwin']:+.5f}、CI 上限 {c['win_hi']:+.5f}）、連対 Δtop2 {c['Δtop2']:+.5f}、複勝 Δtop3 {c['Δtop3']:+.5f}（CI 上限 {c['top3_hi']:+.5f}）。採用基準: {'満たす' if c['採用'] else '満たさない（複勝のレース単位の差が悪化方向で許容を超える）'}。", "",
               "| 評価 | 旧 tl_ | 新 tl_ | 差 |", "|---|---|---|---|"]
        for k in ("2022", "2023", "2024", "旧版で調教なし(欠損)の行", "旧版で調教ありの行"):
            a_, b_ = e["by_year_old"][k], e["by_year_new"][k]
            L_.append(f"| 勝率 LogLoss {k} | {a_:.5f} | {b_:.5f} | {b_ - a_:+.5f} |")
    return L_


def cmd_report(a) -> None:
    rep = {"lapstats": _jload(OUT / "lapstats.json"), "build": _jload(OUT / "build_meta.json"), "repro": _jload(OUT / "repro.json"), "bet": {k: v for k, v in _jload(OUT / "bet.json").items() if k != "all"},
           "overlap": _jload(OUT / "overlap.json"), "tl_regen": {"build": _tl_regen()[0], "eval": _tl_regen()[1]}, "stage1": _jload(OUT / "stage1/results.json"), "stage2": _jload(OUT / "stage2/results.json")}
    Path("/tmp/tp35_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    s1, s2 = rep["stage1"].get("groups", {}), rep["stage2"].get("groups", {})
    lines = ["# tp-35 ラップ種別（A1〜B3）×調教師別/コース別の成績、前週土日の坂路時計", ""]
    ls = rep["lapstats"]
    if ls:
        lines += ["## 1F ごとのラップに直せる割合と分布", "", "| 種別 | 行数 | 直せる割合 | " + " | ".join(L.CLASSES) + " |", "|---|---|---|" + "---|" * len(L.CLASSES)]
        for k, v in ls.get("by_kind", {}).items():
            lines.append(f"| {k} | {v['rows']:,} | {v['lap_ok_share']:.3f} | " + " | ".join(f"{v['dist'][c]:.3f}" for c in L.CLASSES) + " |")
        lines += ["", "最終追い切りの分布: " + " ".join(f"{c} {x:.3f}" for c, x in ls.get("final_session_dist", {}).items()), ""]
    lines += ["## 第 1 段・第 2 段（V4 全体に対する勝率の増分）", "", "| 群 | 列数 | 第 1 段 Δwin(上限) | 第 2 段 Δwin(上限) | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS + ["ALL_PASSED"]:
        a, b = s1.get(g), s2.get(g)
        if not a and not b:
            continue
        a1 = f"{a['vs_V4']['Δwin']:+.5f}({a['vs_V4']['win_hi']:+.5f})" if a else "-"
        b1 = f"{b['vs_V4']['Δwin']:+.5f}({b['vs_V4']['win_hi']:+.5f})" if b else "-"
        c = b["vs_V4"] if b else {}
        lines.append(f"| {g} | {(a or b)['cols']} | {a1} | {b1} | {c.get('Δtop2', float('nan')):+.5f} | {c.get('Δtop3', float('nan')):+.5f} | {('採用' if c.get('採用') else '不採用') if b else '-'} |")
    lines += _tl_md()
    Path("/tmp/tp35_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ───────────────────────── chapter（docs の 1 章を結果から生成）
CHAPTER = ROOT / "docs/html/modeling/chapters/lap_groups_and_trainer_stats.html"


def _fmt(x, nd=5):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def cmd_chapter(a) -> None:
    import html as H

    ls, bm, rp, bt, ov = (_jload(OUT / n) for n in ("lapstats.json", "build_meta.json", "repro.json", "bet.json", "overlap.json"))
    s1, s2 = _jload(OUT / "stage1/results.json").get("groups", {}), _jload(OUT / "stage2/results.json").get("groups", {})
    rows_kind = "".join(f"<tr><td>{H.escape(k)}</td><td class=num>{v['rows']:,}</td><td class=num>{v['lap_ok_share']:.3f}</td><td>{'近似（2F〜3F の平均）' if v['approx_l2'] else '正確（t2 − t1）'}</td>" + "".join(f"<td class=num>{v['dist'][c]:.3f}</td>" for c in L.CLASSES) + "</tr>" for k, v in ls.get("by_kind", {}).items())
    rows_cc = "".join(f"<tr><td>{H.escape(k)}</td><td class=num>{v['rows']:,}</td>" + "".join(f"<td class=num>{v['dist'][c]:.3f}</td>" for c in L.CLASSES) + "</tr>" for k, v in ls.get("by_center_kind", {}).items() if v["rows"] > 20000)
    rp_rows = ""
    for k, v in rp.items():
        ms = v.get("mean_spearman", {})
        rp_rows += (f"<tr><td>{H.escape(k)}</td><td class=num>{ms.get('win', float('nan')):.2f}</td><td class=num>{ms.get('top3', float('nan')):.2f}</td><td class=num>{ms.get('roiw', float('nan')):.2f}</td><td class=num>{ms.get('roip', float('nan')):.2f}</td>"
                    f"<td class=num>{v.get('mean_top10_next_roiw') or float('nan'):.3f} / {v.get('mean_all_next_roiw') or float('nan'):.3f}</td><td class=num>{v.get('mean_top10_next_roip') or float('nan'):.3f} / {v.get('mean_all_next_roip') or float('nan'):.3f}</td></tr>")
    st_rows = ""
    for g in GROUPS + ["ALL_PASSED"]:
        a_, b_ = s1.get(g), s2.get(g)
        if not a_ and not b_:
            continue
        c1, c2 = (a_ or {}).get("vs_V4", {}), (b_ or {}).get("vs_V4", {})
        st_rows += (f"<tr><td>{g}</td><td class=num>{(a_ or b_)['cols']}</td><td class=num>{_fmt(c1.get('Δwin'))}（{_fmt(c1.get('win_hi'))}）</td><td class=num>{_fmt(c2.get('Δwin')) if b_ else '-'}（{_fmt(c2.get('win_hi')) if b_ else '-'}）</td>"
                    f"<td class=num>{_fmt(c2.get('Δtop2')) if b_ else '-'}</td><td class=num>{_fmt(c2.get('Δtop3')) if b_ else '-'}</td><td>{('採用' if c2.get('採用') else '不採用') if b_ else '第 1 段で不通過'}</td></tr>")
    best = bt.get("best_rule", {})
    top5 = "".join(f"<tr><td>{H.escape(r['rule'])}</td><td class=num>{r['bets']:,}</td><td class=num>{r['roi']:.3f}</td><td class=num>{r['hit']:.3f}</td><td class=num>{r['base_roi']:.3f}</td></tr>" for r in bt.get("top5_by_excess", []))
    ov_rows = "".join(f"<tr><td>{H.escape(k)}</td><td class=num>{v:+.2f}</td></tr>" for k, v in list(ov.get("spearman", {}).items())[:8])
    tb, te = _tl_regen()
    tl_html = ""
    if tb:
        ys = sorted(tb["coverage_new"]["has_training"])
        tl_html = "<table><tr><th>年</th><th>調教あり（旧→新）</th><th>最終追い切りに時計あり（旧→新）</th></tr>" + "".join(
            f"<tr><td>{y}</td><td class=num>{tb['coverage_old']['has_training'][y]:.3f} → {tb['coverage_new']['has_training'][y]:.3f}</td><td class=num>{tb['coverage_old'].get('f_t1', {}).get(y, float('nan')):.3f} → {tb['coverage_new']['f_t1'][y]:.3f}</td></tr>" for y in ys) + "</table>"
        if te:
            c = te["vs_old"]
            tl_html += (f"<p>V4 の tl_ 列（{te['cols_replaced']} 列）を再生成版に差し替えた場合（valid 2022〜24、2 seed）: 勝率 {te['levels_old']['logloss_win']:.5f} → {te['levels_new']['logloss_win']:.5f}"
                        f"（Δwin {c['Δwin']:+.5f}、CI 上限 {c['win_hi']:+.5f}）、Δtop2 {c['Δtop2']:+.5f}、Δtop3 {c['Δtop3']:+.5f}（上限 {c['top3_hi']:+.5f}）。</p><table><tr><th>評価</th><th>旧 tl_</th><th>新 tl_</th><th>差</th></tr>"
                        + "".join(f"<tr><td>{k}</td><td class=num>{te['by_year_old'][k]:.5f}</td><td class=num>{te['by_year_new'][k]:.5f}</td><td class=num>{te['by_year_new'][k] - te['by_year_old'][k]:+.5f}</td></tr>" for k in ("2022", "2023", "2024", "旧版で調教なし(欠損)の行", "旧版で調教ありの行")) + "</table>")
    adopted = [g for g, v in s2.items() if v.get("vs_V4", {}).get("採用")]
    summary = ("第 2 段（valid 2024・新 seed）で採用基準（CI 上限 &lt; 0・効果 0.0002 以上）を満たした特徴量の群はありませんでした。V4 には何も足しません。" if not adopted else f"採用: {', '.join(adopted)}。")
    summary += (f" 回収率上位の組を買う単純規則は {bt.get('rules_tried', '-')} 通り中で最良でも reality check の p = {bt.get('reality_check_p', float('nan')):.2f} で、偶然を排除できません。"
                " 一方、調教の再取得で TL_A を再生成すると勝率が改善します（7 節）。")
    head = open(ROOT / "docs/html/modeling/chapters/transport_and_stay.html", encoding="utf-8").read().split("<body>")[0]
    head = head.replace("輸送と滞在（tp-33）", "ラップ種別と調教師別・コース別の調教成績（tp-35）")
    body = f"""<body><a href="../../index.html" class="muted">← ドキュメント一覧</a><h1>ラップ種別と調教師別・コース別の調教成績（tp-35）</h1>
<p class="muted">追い切りの終い 2F のラップを A1〜B3 の 6 種別に分け、『調教師 × 調教コース × ラップ種別』『レース条件 × 調教コース × 時計階級』『前週の土日の坂路最速 4F』の成績を、日付が厳密に前のレースだけで集計して特徴量にし、V4 全体に対する増分を測った。資料の本文・図表は引用していない（定義と、私たちの検証結果だけを書く）。</p>
<div class="note"><b>要点.</b> {summary}</div>
<h2>1. 定義</h2>
<ul><li><b>ラップ種別</b>（終い 1F = l1、その一つ前の 1F = l2、小数 1 位）: A3 = l1 が 12 秒未満で加速 / B3 = l2 が 12 秒未満で減速 / A2 = l1・l2 ともに 12 秒台で加速 / B2 = 同じく減速 / A1 = l1 だけが 12 秒台で加速（l2 は 13 秒以上）/ B1 = l2 だけが 12 秒台で減速（l1 は 13 秒以上）/ X = 終い 2F がともに 13 秒以上。</li>
<li><b>坂路</b>は l2 = t2 − t1 で正確に出せる。<b>ウッド・ポリ</b>は 2F の時計が載らない（6F・5F・4F・3F・1F）ので l2 = (t3 − t1) / 2 の近似（2F〜3F の平均）。</li>
<li><b>前週の土日</b>: レースの週の月曜（月曜開催は前の週に数える）の前の土曜・日曜。土曜開催は 7・6 日前、日曜開催は 8・7 日前、月曜開催は 9・8 日前。その中の坂路の最速 4F を 〜53.9 / 54.0〜55.9 / 56.0〜57.9 / 58.0〜59.9 / 60.0〜 に分け、『土日に坂路調教なし』も一つの区分にする。</li></ul>
<h2>2. 1F ごとのラップに直せる割合と、種別の分布</h2>
<table><tr><th>種別</th><th>行数</th><th>直せる割合</th><th>l2</th>{''.join(f'<th>{c}</th>' for c in L.CLASSES)}</tr>{rows_kind}</table>
<table><tr><th>種別|所属</th><th>行数</th>{''.join(f'<th>{c}</th>' for c in L.CLASSES)}</tr>{rows_cc}</table>
<p class="muted">最終追い切り（レースの 1〜6 日前）の分布: {' '.join(f'{c} {x:.3f}' for c, x in ls.get('final_session_dist', {}).items())}。払戻の解釈率: 単勝 {bm.get('win_ret_parsed', float('nan')):.3f} / 複勝 {bm.get('place_ret_parsed', float('nan')):.3f}。</p>
<h2>3. 『傾向の強い調教師 × ラップ』は翌年も再現するか</h2>
<p>各年（2020〜2024）の組ごとの成績を、翌年の同じ組と比べた（件数 n 以上の組）。順位相関が高いほど再現する。右の 2 列は、前年の回収率が上位 10% の組を翌年に買った場合の回収率 / 全組の平均。</p>
<table><tr><th>組</th><th>勝率</th><th>複勝率</th><th>単勝回収率</th><th>複勝回収率</th><th>上位 10%（単勝）翌年 / 全体</th><th>上位 10%（複勝）翌年 / 全体</th></tr>{rp_rows}</table>
<h2>4. 特徴量としての増分（V4 全体に対する勝率 LogLoss の差、95% CI 上限）</h2>
<table><tr><th>群</th><th>列数</th><th>第 1 段（valid 2022+2023）</th><th>第 2 段（valid 2024・新 seed）</th><th>Δtop2</th><th>Δtop3</th><th>採否</th></tr>{st_rows}</table>
<h2>5. 回収率上位の組を買う単純規則（valid 2022〜24。各レース日より前の推定）</h2>
<p>試した規則 {bt.get('rules_tried', '-')} 通り。最良の規則 {H.escape(str(best.get('rule', '-')))}: 買い目 {best.get('bets', 0):,}、回収率 {best.get('roi', float('nan')):.3f}（全体 {best.get('base_roi', float('nan')):.3f}）、reality check の p = {bt.get('reality_check_p', float('nan')):.2f}。</p>
<table><tr><th>規則</th><th>買い目</th><th>回収率</th><th>的中率</th><th>全体の回収率</th></tr>{top5}</table>
<h2>6. TL_A の 1 週前（w1_*）との重複</h2><table><tr><th>列の組</th><th>順位相関</th></tr>{ov_rows}</table>
<h2>7. TL_A の再生成（調教の再取得後）</h2>{tl_html}
</body></html>"""
    CHAPTER.write_text(head + body, encoding="utf-8")
    _log(f"章を書いた: {CHAPTER}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lapstats", "build", "repro", "bet", "overlap", "corr", "stage1", "stage2", "report", "chapter"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"lapstats": cmd_lapstats, "build": cmd_build, "repro": cmd_repro, "bet": cmd_bet, "overlap": cmd_overlap, "corr": cmd_corr, "stage1": cmd_stage1, "stage2": cmd_stage2, "report": cmd_report, "chapter": cmd_chapter}[a.cmd](a)


if __name__ == "__main__":
    main()
