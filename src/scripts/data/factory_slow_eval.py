"""特徴量ファクトリー G6「出遅れ率」の生成と評価。

python -m src.scripts.data.factory_slow_eval remarks               # 取り込み済みの horse_race_remarks を読んで remarks_labels.parquet にする
python -m src.scripts.data.factory_slow_eval labels  [proxy|real]   # ラベル作成と妥当性の検証 → /tmp/fac_slow_label_<src>.json
python -m src.scripts.data.factory_slow_eval build   [proxy|real]   # 群を作って factory/slowstart/<src>_<群>.parquet
python -m src.scripts.data.factory_slow_eval pred    [proxy|real]   # 出遅れ確率の OOF（年ごとの expanding）→ <src>_SS_pred.parquet
python -m src.scripts.data.factory_slow_eval stage1  [proxy|real]   # V0 / V0+S3 に各群を足す。valid 2022+2023、seed 0
python -m src.scripts.data.factory_slow_eval stage2  [proxy|real]   # 通過群を valid 2022〜2024、seed 2・3 で確認 + 採否
python -m src.scripts.data.factory_slow_eval report                 # /tmp/fac_slow_report.{json,md}

基準: 修正後の学習表の tp-03 採用 175 列（V0）と、現行の前段 3 列を足した V0+S3。LightGBM 固定、valid 2022・2023・2024 expanding。
test 2025・2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
SMART = WORK / "smartrc_ten1f"
OUT = WORK / "factory/slowstart"
KEY = ["race_id", "horse_id"]
TABLES = ROOT / "data/page_reference/tables"
STAGE_COLS = ["s_time_pred", "s_time_gap", "s_rank_pred"]
STAGE1_MARGIN = -0.0001
COLS = ["race_id", "horse_id", "date", "jockey_id", "trainer_id", "venue_code", "surface", "distance", "horse_number", "field_size", "bracket_number",
        "sex_age", "weight", "finish_position"]


def _frame() -> pd.DataFrame:
    import polars as pl

    f = pl.read_parquet(FIXED / "features/train_table.parquet", columns=COLS).to_pandas()
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    f["date"] = pd.to_datetime(f["date"].astype(str))
    return f


def _parents() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/local/modeling/training_pipeline_ped/pedigree_keys.parquet").astype(str)


def _labels(src: str) -> tuple[pd.DataFrame, str]:
    from src.pipeline.features import factory_slowstart as fs

    if src == "proxy":
        lab = fs.proxy_labels(pd.read_parquet(SMART / "career.parquet"))
        return lab, "slow_b"
    rem = pd.read_parquet(OUT / "remarks_labels.parquet")
    return fs.real_labels(rem), "slow_rem"                      # ｽﾀｰﾄ指数は 2024 年以降の一部の走にしか入っていないので、備考の『出遅』を実ラベルにする


def cmd_remarks() -> None:
    """取り込み済みの horse_race_remarks（GCS）を読んで、1 走 1 行の表にする。生の備考テキストは保存せず、フラグと指数だけ残す。"""
    import polars as pl
    from concurrent.futures import ThreadPoolExecutor

    from src.scraper.storage import HybridStorage

    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(pl.read_parquet(FIXED / "features/train_table.parquet", columns=["horse_id"])["horse_id"].cast(pl.Utf8).to_list()))
    if os.environ.get("SLOW_REMARKS_FROM_CACHE") == "1":          # 取り込み途中: ディスクキャッシュにある馬だけ（GCS の空振りを避ける）
        have = {p.stem for p in (ROOT / "data/cache/horse_race_remarks").glob("*/*.json")}
        ids = [h for h in ids if h in have]
    st = HybridStorage()

    def one(h):
        try:
            d = st.load("horse_race_remarks", h) or {}
        except Exception:
            d = {}
        return [{"horse_id": h, "race_id": r.get("race_id"), "date": r.get("date"), "start_index": r.get("start_index"), "chase_index": r.get("chase_index"),
                 "closing_index": r.get("closing_index"), "remarks": r.get("remarks") or ""} for r in d.get("race_history", [])]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [x for part in ex.map(one, ids) for x in part]
    df = pd.DataFrame(rows)
    df["has_rem"] = df["remarks"].astype(str).str.len() > 0
    n_h = df["horse_id"].nunique() if len(df) else 0
    df.to_parquet(OUT / "remarks_labels.parquet", index=False)
    print(f"馬 {n_h} 頭 / {len(df)} 走 / start_index あり {df['start_index'].notna().mean():.3f} / 備考あり {df['has_rem'].mean():.3f} / 出遅 {df['remarks'].astype(str).str.contains('出遅').mean():.4f} ({time.time() - t0:.0f}s)", flush=True)


def cmd_compare() -> None:
    """代理ラベルと実ラベル（取り込み済みの馬だけ）の突き合わせ。"""
    from sklearn.metrics import roc_auc_score

    from src.pipeline.features import factory_slowstart as fs

    prox = pd.read_parquet(OUT / "proxy_labels.parquet") if (OUT / "proxy_labels.parquet").exists() else fs.proxy_labels(pd.read_parquet(SMART / "career.parquet"))
    rem = pd.read_parquet(OUT / "remarks_labels.parquet")
    real = fs.real_labels(rem)
    m = real.merge(prox.drop_duplicates(["horse_id", "date"]), on=["horse_id", "date"], how="inner", suffixes=("_real", "_proxy")).dropna(subset=["resid"])
    out = {"horses_real": int(real["horse_id"].nunique()), "runs_real": int(len(real)), "runs_matched_with_proxy": int(len(m)),
           "real_rates": {c: float(real[c].mean()) for c in ("slow_idx", "slow_rem", "trouble_rem")},
           "start_index_threshold": real.attrs.get("thresholds")}
    out["spearman_resid_vs_start_index"] = float(m["resid"].corr(m["start_index"], method="spearman"))
    mi = m.dropna(subset=["slow_idx"])
    out["auc_resid_vs_real_slow_idx"] = float(roc_auc_score(mi["slow_idx"], mi["resid"])) if mi["slow_idx"].nunique() > 1 else None
    out["auc_proxy_slow_b_vs_real_slow_idx"] = float(roc_auc_score(mi["slow_idx"], mi["slow_b"])) if mi["slow_idx"].nunique() > 1 and mi["slow_b"].nunique() > 1 else None
    if m["slow_rem"].nunique() > 1:
        out["auc_resid_vs_real_remarks_slow"] = float(roc_auc_score(m["slow_rem"], m["resid"]))
        out["share_flagged_slow_b_among_remarks_slow"] = float(m.loc[m["slow_rem"] > 0, "slow_b"].mean())
        out["share_flagged_slow_b_among_not_remarks"] = float(m.loc[m["slow_rem"] == 0, "slow_b"].mean())
    out["start_index_by_remarks_slow"] = {"remarks_slow_mean": float(real.loc[real["slow_rem"] > 0, "start_index"].astype(float).mean()), "other_mean": float(real.loc[real["slow_rem"] == 0, "start_index"].astype(float).mean())}
    Path("/tmp/fac_slow_label_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


# ───────────────────────── ラベルの妥当性
def cmd_labels(src: str) -> None:
    from src.pipeline.features import factory_slowstart as fs

    OUT.mkdir(parents=True, exist_ok=True)
    lab, val = _labels(src)
    lab.to_parquet(OUT / f"{src}_labels.parquet", index=False)
    f = _frame()
    m = f.merge(lab.drop_duplicates(["horse_id", "date"]), on=["horse_id", "date"], how="left")
    rep: dict = {"source": src, "primary_label": val, "thresholds": lab.attrs.get("thresholds"), "rows_frame": len(f)}
    flags = [c for c in lab.columns if c.startswith("slow")]
    rep["coverage_labeled"] = {c: float(m[c].notna().mean()) for c in flags}
    rep["rate"] = {c: float(m[c].mean()) for c in flags}
    rep["rate_by_year"] = {c: {int(y): float(v) for y, v in m.groupby(m["date"].dt.year)[c].mean().items()} for c in flags[:2]}
    # (b) 出遅れ判定の走で着順がどれだけ悪いか（馬ごとの差。出遅れた走 − 同じ馬の出遅れでない走）
    m["fin_pct"] = (m["finish_position"].astype(float) - 1) / (m["field_size"].astype(float) - 1).clip(lower=1)
    m = m[m["finish_position"].astype(float) > 0]
    for c in flags:
        d = m.dropna(subset=[c])
        pos = d[d[c] > 0]
        hm = d.groupby("horse_id")["fin_pct"].transform("mean")
        rep.setdefault("finish_pct_flagged_minus_not", {})[c] = {"raw": float(pos["fin_pct"].mean() - d[d[c] == 0]["fin_pct"].mean()),
                                                                 "within_horse": float((pos["fin_pct"] - hm[pos.index]).mean()), "n_flagged": int(len(pos))}
    # (c) 個体差: 走数が多い馬の奇数番目/偶数番目の走の出遅れ率の相関と、二項分布からの過分散
    d = m.dropna(subset=[val]).sort_values(["horse_id", "date"])
    d["k"] = d.groupby("horse_id").cumcount()
    cnt = d.groupby("horse_id")[val].transform("size")
    d = d[cnt >= 12]
    a = d[d["k"] % 2 == 0].groupby("horse_id")[val].mean()
    b = d[d["k"] % 2 == 1].groupby("horse_id")[val].mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    rep["split_half"] = {"horses": int(len(j)), "corr": float(j["a"].corr(j["b"])), "spearman": float(j["a"].corr(j["b"], method="spearman"))}
    g = d.groupby("horse_id")[val].agg(["mean", "size"])
    p = float(d[val].mean())
    exp_var = float((p * (1 - p) / g["size"]).mean())
    rep["overdispersion"] = {"var_of_horse_rates": float(g["mean"].var()), "expected_if_no_horse_effect": exp_var, "ratio": float(g["mean"].var() / exp_var)}
    # (a) バロメーターのｽﾀｰﾄ指数（2024〜25、上位着順のみ）との整合
    bar = []
    import polars as pl

    for y in (2023, 2024, 2025):
        fp = TABLES / str(y) / "race_barometer_flat.parquet"
        if fp.exists():
            bar.append(pl.read_parquet(fp, columns=["race_id", "horse_id", "index_start"]).to_pandas())
    if bar:
        bb = pd.concat(bar)
        bb["race_id"], bb["horse_id"] = bb["race_id"].astype(str), bb["horse_id"].astype(str)
        z = f.merge(lab.drop_duplicates(["horse_id", "date"]), on=["horse_id", "date"], how="left").merge(bb, on=KEY, how="inner")
        z = z.dropna(subset=[val])
        low = (z["index_start"] <= z["index_start"].quantile(0.1)).astype(int)
        from sklearn.metrics import roc_auc_score

        score = z["resid"] if "resid" in z.columns else -z["start_z"]
        rep["barometer_check"] = {"rows": int(len(z)), "auc_label_vs_low_start_index": float(roc_auc_score(low, z[val])), "auc_resid_vs_low_start_index": float(roc_auc_score(low, score)) if "resid" in z.columns else None,
                                  "spearman_resid_vs_start_index": float(z["resid"].corr(z["index_start"], method="spearman")) if "resid" in z.columns else None,
                                  "note": "バロメーターの指数は上位着順の馬だけ（選択バイアスあり）"}
    Path(f"/tmp/fac_slow_label_{src}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


# ───────────────────────── 特徴量
def cmd_build(src: str) -> None:
    from src.pipeline.features import factory_slowstart as fs

    OUT.mkdir(parents=True, exist_ok=True)
    lab, val = _labels(src)
    f = fs.prepare(_frame())
    t0 = time.time()
    groups = fs.build_groups(f, lab, val, _parents(), p0=0.08 if src == "proxy" else 0.025)
    groups["SS_damage"] = fs.with_damage(f, lab, val)
    meta = {}
    for name, g in groups.items():
        g = g.copy()
        nn = g.notna().mean()
        keep = [c for c in g.columns if nn[c] > 0.02 and g[c].nunique(dropna=True) > 1]
        out = pd.concat([f[KEY].reset_index(drop=True), g[keep].reset_index(drop=True)], axis=1)
        out.to_parquet(OUT / f"{src}_{name}.parquet", index=False)
        meta[name] = {"cols": len(keep), "coverage": float(nn[keep].mean()) if keep else 0.0}
        print(f"[{time.time() - t0:.0f}s] {name}: {len(keep)} 列 coverage {meta[name]['coverage']:.2f}", flush=True)
    (OUT / f"{src}_build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_pred(src: str) -> None:
    import lightgbm as lgb

    from src.pipeline.features import factory_slowstart as fs

    lab, val = _labels(src)
    f = fs.prepare(_frame())
    y = f[["horse_id", "date"]].merge(lab.drop_duplicates(["horse_id", "date"])[["horse_id", "date", val]], on=["horse_id", "date"], how="left")[val].to_numpy()
    names = ["SS_horse", "SS_horse_ctx", "SS_gate", "SS_jockey", "SS_trainer", "SS_sire", "SS_misc", "SS_damage"]
    parts = [pd.read_parquet(OUT / f"{src}_{n}.parquet").drop(columns=KEY) for n in names if (OUT / f"{src}_{n}.parquet").exists()]
    num = f[["horse_number", "field_size", "distance", "bracket_number", "weight", "age", "dband", "lband", "fband"]].reset_index(drop=True)
    X = pd.concat(parts + [num, f[["surface", "venue_code", "sex", "gate3"]].astype("category").reset_index(drop=True)], axis=1)
    year = f["date"].dt.year.to_numpy()
    date = f["date"].to_numpy()
    p = np.full(len(f), np.nan)
    threads = int(os.environ.get("FAC_THREADS", "2"))
    for Y in range(2021, 2026):
        tr = np.flatnonzero((date < np.datetime64(f"{Y}-01-01") - np.timedelta64(7, "D")) & np.isfinite(y))
        te = np.flatnonzero(year == Y)
        if len(tr) < 5000:
            continue
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               reg_lambda=5.0, n_jobs=threads, verbose=-1, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        print(f"pred {Y}: train {len(tr)} rows, mean p {np.nanmean(p[te]):.4f}", flush=True)
    d = pd.DataFrame({"race_id": f["race_id"].to_numpy(), "p": p})
    out = pd.DataFrame({"ss_p": p})
    out["ss_p_rank"] = d.groupby("race_id")["p"].rank(ascending=False, method="average").to_numpy()
    out["ss_p_racesum"] = d.groupby("race_id")["p"].transform("sum").to_numpy()
    out["ss_p_gap"] = (d["p"] - d.groupby("race_id")["p"].transform("mean")).to_numpy()
    import polars as pl

    sty = pl.read_parquet(FIXED / "features/train_table.parquet", columns=["a_style"]).to_pandas()["a_style"].to_numpy(dtype=float)
    out["ss_p_x_style"] = p * (1.0 - sty) if np.nanmax(sty) <= 1.0 else p * sty
    out.insert(0, "horse_id", f["horse_id"].to_numpy())
    out.insert(0, "race_id", f["race_id"].to_numpy())
    out.to_parquet(OUT / f"{src}_SS_pred.parquet", index=False)
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(p) & np.isfinite(y)
    print("OOF AUC", {int(Y): round(float(roc_auc_score(y[ok & (year == Y)], p[ok & (year == Y)])), 4) for Y in range(2021, 2026) if (ok & (year == Y)).sum() > 100}, flush=True)


# ───────────────────────── 評価
def _setup():
    from src.pipeline.models import ablation as ab, zoo
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    threads = int(os.environ.get("FAC_THREADS", min(2, detect_compute(reserve_cores=6).threads)))
    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((WORK / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    st = pd.read_parquet(WORK / "oof/stage_features_expanding.parquet")[KEY + STAGE_COLS]
    st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
    df = df.merge(st, on=KEY, how="left")
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    return df, base, meta, list(yearly_expanding_folds(df)), lp, threads


def _group(df: pd.DataFrame, src: str, name: str) -> pd.DataFrame:
    """SS_all = 統計の群すべて（SS_pred を除く）、SS_all_pred = SS_all + SS_pred。"""
    if name in ("SS_all", "SS_all_pred"):
        parts = sorted(p.stem[len(src) + 1:] for p in OUT.glob(f"{src}_SS_*.parquet"))
        parts = [x for x in parts if x not in ("SS_all", "SS_all_pred") and (x != "SS_pred" or name == "SS_all_pred")]
        return pd.concat([_group(df, src, x) for x in parts], axis=1)
    g = pd.read_parquet(OUT / f"{src}_{name}.parquet")
    g["race_id"], g["horse_id"] = g["race_id"].astype(str), g["horse_id"].astype(str)
    m = df[KEY].merge(g, on=KEY, how="left")
    assert len(m) == len(df)
    m = m.drop(columns=KEY)
    m.index = df.index
    return m


def _wait_for_load(limit: float = 15.0, max_wait: int = 3600) -> None:
    t0 = time.time()
    while os.getloadavg()[0] >= limit and time.time() - t0 < max_wait:
        time.sleep(30)


def _jl(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _js(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _by_year(res: pd.DataFrame) -> dict:
    return {str(r["fold"]): {k: float(r[k]) for k in ("logloss_win", "logloss_top2", "logloss_top3")} for _, r in res.iterrows()}


def cmd_stage(which: str, src: str) -> None:
    from src.pipeline.models import feature_bakeoff as fb, zoo_runner

    df, base, meta, folds, lp, threads = _setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    names = sorted(p.stem[len(src) + 1:] for p in OUT.glob(f"{src}_SS_*.parquet"))
    if os.environ.get("SLOW_GROUPS"):
        names = os.environ["SLOW_GROUPS"].split(",")
    s1p, s2p = OUT / f"{src}_stage1.json", OUT / f"{src}_stage2.json"
    s1, s2 = _jl(s1p), _jl(s2p)
    if which == "stage1":
        use_folds, seeds, store, op = folds[:2], [0], s1, OUT / f"{src}_oof1"
    else:
        use_folds, seeds, store, op = folds, [2, 3], s2, OUT / f"{src}_oof2"
        names = [n for n in names if s1.get(n, {}).get("pass")]
    op.mkdir(exist_ok=True)
    t0 = time.time()
    bases = {"V0": df[base], "V0S3": pd.concat([df[base], df[STAGE_COLS]], axis=1)}
    base_oof, base_res = {}, {}
    for bn, X0 in bases.items():
        if (op / f"{bn}.parquet").exists():
            base_oof[bn], base_res[bn] = pd.read_parquet(op / f"{bn}.parquet"), _jl(op / f"{bn}.json")
        else:
            _wait_for_load()
            r, o = zoo_runner.run_oof(bn, "lgb_binary", X0, meta, use_folds, seeds, threads, params=dict(lp))
            o.to_parquet(op / f"{bn}.parquet", index=False)
            base_oof[bn], base_res[bn] = o, _by_year(r)
            _js(op / f"{bn}.json", base_res[bn])
        print(f"{bn} {which}: " + " ".join(f"{k} {v['logloss_win']:.5f}" for k, v in base_res[bn].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    for n in names:
        for bn in ("V0", "V0S3"):
            key = f"{n}@{bn}"
            if key in store and "win_diff" in store[key]:
                continue
            _wait_for_load()
            G = _group(df, src, n)
            r, oof = zoo_runner.run_oof(f"{bn}+{n}", "lgb_binary", pd.concat([bases[bn], G], axis=1), meta, use_folds, seeds, threads, params=dict(lp))
            ry = _by_year(r)
            c = fb.compare_oof(base_oof[bn], oof, fin_map, key)
            per = {k: ry[k]["logloss_win"] - base_res[bn][k]["logloss_win"] for k in ry if k in base_res[bn]}
            rec = {"cols": int(G.shape[1]), "base": bn, "win_diff": c["Δwin"], "win_hi": c["win_hi"], "top2_diff": c["Δtop2"], "top3_diff": c["Δtop3"], "per_year_win_diff": per}
            if which == "stage1":
                rec["pass"] = bool(c["Δwin"] < STAGE1_MARGIN and all(v <= 0 for v in per.values()))
            else:
                rec["adopt"] = bool(c["採用"])
            store[key] = rec
            if which == "stage1" and bn == "V0":                          # 通過判定は V0 に対して。V0S3 は参考
                store[n] = rec
            _js(s1p if which == "stage1" else s2p, store)
            print(f"{key:24s} cols {G.shape[1]:3d} Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} "
                  f"{'PASS' if rec.get('pass') or rec.get('adopt') else '-'} ({time.time() - t0:.0f}s)", flush=True)
            if which == "stage1" and bn == "V0" and not rec["pass"]:
                break                                                       # V0 で通らない群は V0S3 を測らない（時間の節約）


def cmd_report() -> None:
    rep = {"tested_groups": {}, "label": {}}
    L = ["# 特徴量ファクトリー G6 出遅れ率\n"]
    for src in ("proxy", "real"):
        s1, s2 = _jl(OUT / f"{src}_stage1.json"), _jl(OUT / f"{src}_stage2.json")
        lab = _jl(Path(f"/tmp/fac_slow_label_{src}.json"))
        if not (s1 or s2):
            continue
        rep["tested_groups"][src] = len([k for k in s1 if "@" not in k])
        rep[src] = {"stage1": s1, "stage2": s2}
        rep["label"][src] = lab
        L += [f"## ラベル: {src}\n", "| 群 | 基準 | 列 | stage1 Δwin | 通過 | stage2 Δwin (CI上限) | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|---|---|---|"]
        for k, r in sorted(s1.items()):
            if "@" not in k:
                continue
            r2 = s2.get(k)
            L.append(f"| {k.split('@')[0]} | {r['base']} | {r['cols']} | {r['win_diff']:+.5f} | {'○' if r.get('pass') else '-'} | "
                     + (f"{r2['win_diff']:+.5f} ({r2['win_hi']:+.5f}) | {r2['top2_diff']:+.5f} | {r2['top3_diff']:+.5f} | {'採用' if r2.get('adopt') else '不採用'} |" if r2 else "- | - | - | - |"))
    Path("/tmp/fac_slow_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    Path("/tmp/fac_slow_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    src = sys.argv[2] if len(sys.argv) > 2 else "proxy"
    {"remarks": cmd_remarks, "compare": cmd_compare, "labels": lambda: cmd_labels(src), "build": lambda: cmd_build(src), "pred": lambda: cmd_pred(src), "stage1": lambda: cmd_stage("stage1", src),
     "stage2": lambda: cmd_stage("stage2", src), "report": cmd_report}[cmd]()
