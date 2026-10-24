"""tp-30: 荒れやすいレースの分析（事後指標・事前予測・予想への寄与・温度調整）。

python -m src.scripts.data.tp30_eval prep       # レース単位の表（事後指標・事前入力・オッズ構造）を作る
python -m src.scripts.data.tp30_eval describe   # 荒れやすい条件の記述統計と有意性
python -m src.scripts.data.tp30_eval predict    # 荒れ度の事前予測（モデル用 / 馬券用）の expanding OOF と精度
python -m src.scripts.data.tp30_eval feature    # 予測した荒れ度を勝率モデルの文脈列に足した増分
python -m src.scripts.data.tp30_eval temper     # 荒れ度による確率の温度調整（v3 の OOF）
test 2025・2026 年は使わない。
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
OUT = WORK / "tp30"
YEARS = [2020, 2021, 2022, 2023, 2024]
VALID = [2022, 2023, 2024]
THREADS = int(os.environ.get("TP30_THREADS", "4"))
CAT = ["surface", "venue_code", "grade", "race_class", "weather", "track_condition"]


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def load_races() -> pd.DataFrame:
    return pd.read_parquet(OUT / "races.parquet")


# ───────────────────────── prep
def prep() -> None:
    from src.pipeline.features import tp30_volatility as tv
    from src.pipeline.models import ablation as ab

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = tv.load_results(YEARS)
    ph = tv.posthoc(res)
    print(f"[{time.time() - t0:.0f}s] 事後指標 {len(ph)} レース", flush=True)
    meta = res.groupby("race_id").agg(date=("date", "first"), venue_code=("venue_code", "first"), surface=("surface", "first"), distance=("distance", "first"),
                                      grade=("grade", "first"), race_class=("race_class", "first"), track_condition=("track_condition", "first")).reset_index()
    meta["year"] = pd.to_datetime(meta["date"]).dt.year
    ph = ph.merge(meta[["race_id", "year"]], on="race_id", how="left")
    ref = ph[ph["year"].isin([2020, 2021])].copy()
    ph = tv.add_composite(ph, ref)
    tt = ab.load_table(FIXED / "features/train_table.parquet")
    tt["race_id"] = tt["race_id"].astype(str)
    ctx = tv.race_context(tt)
    print(f"[{time.time() - t0:.0f}s] 事前入力 {ctx.shape}", flush=True)
    mk = tv.odds_cols(res)
    races = ph.merge(ctx.drop(columns=["date"]), on="race_id", how="inner", suffixes=("", "_ctx")).merge(mk, on="race_id", how="left")
    races["date"] = races["race_id"].map(meta.set_index("race_id")["date"])
    races.to_parquet(OUT / "races.parquet", index=False)
    print(f"[{time.time() - t0:.0f}s] 保存 {OUT / 'races.parquet'} {races.shape}", flush=True)


# ───────────────────────── describe
def describe() -> None:
    from scipy import stats

    from src.pipeline.features.course_variant import add_course_variant

    r = load_races()
    out: dict = {"races": len(r)}
    corr_cols = ["vol", "log_win_pop", "log_pay_sanrenpuku", "log_pay_tansho", "log_pay_umaren", "log_pay_sanrentan", "log_mean_pop_top3", "rho", "fav3_all_top3", "n_long_top3", "manba"]
    out["corr_spearman"] = r[corr_cols].corr(method="spearman").round(3).to_dict()
    r["top20"] = (r["vol"] >= r["vol"].quantile(0.8)).astype(int)
    out["overall"] = {"vol_mean": r["vol"].mean(), "manba_rate": r["manba"].mean(), "fav3_all_top3": r["fav3_all_top3"].mean(), "win_pop_median": r["win_pop"].median()}
    r["dist_band"] = pd.cut(r["distance"], [0, 1300, 1500, 1900, 2400, 9999], labels=["~1300", "1400-1500", "1600-1900", "2000-2400", "2500~"]).astype(str)
    r["fs_band"] = pd.cut(r["field_size"], [0, 11, 14, 16, 18], labels=["~11", "12-14", "15-16", "17-18"]).astype(str)
    r["variant"] = add_course_variant(r.rename(columns={"venue_code": "venue_code"}), parsed_col=None).astype(str)
    r["kai"] = r["race_id"].str[6:8]
    r["nichi"] = r["race_id"].str[8:10].astype(int).clip(upper=8).astype(str)
    r["course"] = r["venue_code"].astype(str) + "|" + r["surface"].astype(str) + "|" + r["distance"].astype(int).astype(str) + "|" + r["variant"]
    levels = {}
    for f in ["year", "venue_code", "surface", "dist_band", "grade", "race_class", "fs_band", "track_condition", "month", "nichi", "variant", "weather", "round"]:
        g = r.groupby(f)["vol"].agg(["count", "mean", "std"]).reset_index()
        g["se"] = g["std"] / np.sqrt(g["count"])
        g["share_top20"] = r.groupby(f)["top20"].mean().to_numpy()
        g["manba"] = r.groupby(f)["manba"].mean().to_numpy()
        groups = [x["vol"].to_numpy() for _, x in r.groupby(f) if len(x) >= 30]
        kw = stats.kruskal(*groups) if len(groups) > 1 else None
        levels[f] = {"kruskal_p": float(kw.pvalue) if kw else None, "table": g.round(4).to_dict("records")}
    out["levels"] = levels
    # コース別（荒れやすい/堅いコースの一覧と有意性）。Welch t（残りのレースとの差）+ BH-FDR
    rows = []
    for c, x in r.groupby("course"):
        if len(x) < 120:
            continue
        rest = r.loc[r["course"] != c, "vol"]
        t = stats.ttest_ind(x["vol"], rest, equal_var=False)
        rows.append({"course": c, "n": len(x), "vol_mean": x["vol"].mean(), "diff": x["vol"].mean() - rest.mean(), "p": float(t.pvalue), "manba": x["manba"].mean(),
                     "fav3_all_top3": x["fav3_all_top3"].mean()})
    cd = pd.DataFrame(rows).sort_values("p")
    m = len(cd)
    cd["q_bh"] = np.minimum.accumulate((cd["p"] * m / np.arange(1, m + 1))[::-1])[::-1].clip(upper=1)
    out["courses"] = cd.round(4).to_dict("records")
    out["n_course_tests"] = m
    # 年による安定性: 2020-21 と 2022-24 でコースの荒れ度の相関
    a = r[r["year"] <= 2021].groupby("course")["vol"].mean()
    b = r[r["year"] >= 2022].groupby("course")["vol"].mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    out["course_vol_year_stability_spearman"] = float(j.corr(method="spearman").iloc[0, 1])
    (OUT / "describe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("races", "overall", "course_vol_year_stability_spearman", "n_course_tests")}, ensure_ascii=False, default=_jsonable))
    print("荒れやすい上位:", cd.head(6)[["course", "n", "diff", "q_bh"]].round(3).to_dict("records"))
    print("堅い上位:", cd.sort_values("diff").head(6)[["course", "n", "diff", "q_bh"]].round(3).to_dict("records"))
    for f, v in levels.items():
        print(f, "Kruskal p", v["kruskal_p"])


# ───────────────────────── predict
def features_for(r: pd.DataFrame, kind: str) -> list[str]:
    base = [c for c in r.columns if c in ("field_size", "distance", "round", "n_debut", "month", "style_mean", "style_sd") or
            any(c.endswith(s) for s in ("_sd", "_gap", "_top12", "_ent"))] + CAT
    base = [c for c in dict.fromkeys(base) if c in r.columns and not c.startswith("mk_") and c != "vol_sd"]
    if kind == "market":
        base += [c for c in r.columns if c.startswith("mk_")]
    return base


def _prep_X(r: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = r[cols].copy()
    for c in CAT:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def predict() -> None:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    r = load_races().sort_values("date").reset_index(drop=True)
    r["race_id"] = r["race_id"].astype(str)
    targets = {"vol": "vol", "log_pay_sanrenpuku": "log_pay_sanrenpuku", "log_win_pop": "log_win_pop", "neg_rho": "neg_rho"}
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
                  n_jobs=THREADS, verbose=-1)
    oof = r[["race_id", "year"]].copy()
    rep: dict = {}
    for kind in ("model", "market"):
        cols = features_for(r, kind)
        X = _prep_X(r, cols)
        rep[kind] = {"n_features": len(cols), "targets": {}}
        for tname, tcol in targets.items():
            y = r[tcol].to_numpy(float)
            pred = np.full(len(r), np.nan)
            for Y in [2021, 2022, 2023, 2024]:
                tr = (r["year"] < Y).to_numpy() & np.isfinite(y)
                te = (r["year"] == Y).to_numpy()
                ps = []
                for seed in (0, 1):
                    m = lgb.LGBMRegressor(random_state=seed, **params).fit(X[tr], y[tr])
                    ps.append(m.predict(X[te]))
                pred[te] = np.mean(ps, axis=0)
            oof[f"{kind}_{tname}"] = pred
            ev = np.isin(r["year"], VALID) & np.isfinite(y)
            yy, pp = y[ev], pred[ev]
            r2 = 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            rho = pd.Series(yy).corr(pd.Series(pp), method="spearman")
            row = {"r2": float(r2), "spearman": float(rho)}
            if tname == "vol":
                # 荒れの上位 20%（年ごとに、その年より前の学習年の分位でラベル化）を、予測の上位 20% で拾えるか
                aucs, caps, base_rate = [], [], []
                for Y in VALID:
                    tr_thr = np.quantile(r.loc[r["year"] < Y, "vol"].dropna(), 0.8)
                    te = (r["year"] == Y).to_numpy()
                    lab = (r.loc[te, "vol"].to_numpy() >= tr_thr).astype(int)
                    sc = pred[te]
                    aucs.append(roc_auc_score(lab, sc))
                    k = int(0.2 * te.sum())
                    top = np.argsort(-sc)[:k]
                    caps.append(lab[top].mean())      # 予測上位 20% のうち実際に荒れた割合（precision）
                    base_rate.append(lab.mean())
                row.update({"auc_top20": float(np.mean(aucs)), "precision_at20": float(np.mean(caps)), "base_rate": float(np.mean(base_rate)),
                            "lift": float(np.mean(caps) / np.mean(base_rate))})
            rep[kind]["targets"][tname] = row
            print(kind, tname, {k: round(v, 4) for k, v in row.items()}, flush=True)
        if kind == "model":
            imp = pd.Series(lgb.LGBMRegressor(random_state=0, **params).fit(X[(r["year"] < 2024).to_numpy()], r.loc[r["year"] < 2024, "vol"]).feature_importances_, index=cols)
            rep[kind]["importance_top"] = imp.sort_values(ascending=False).head(12).to_dict()
    oof.to_parquet(OUT / "vol_oof.parquet", index=False)
    (OUT / "predict.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")


# ───────────────────────── feature（勝率モデルの文脈列に足した増分）
def feature() -> None:
    from src.pipeline.models import ablation as ab, feature_bakeoff as fb, zoo, zoo_runner
    from src.pipeline.models.time_split import yearly_expanding_folds

    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((WORK / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    stage = pd.read_parquet(WORK / "oof/stage_features_expanding.parquet")
    S3 = ["s_time_pred", "s_time_gap", "s_rank_pred"]
    df = df.merge(stage[["race_id", "horse_id"] + S3], on=["race_id", "horse_id"], how="left")
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    vo = pd.read_parquet(OUT / "vol_oof.parquet")
    vcols = {"vol_m": "model_vol", "vol_m_pay": "model_log_pay_sanrenpuku", "vol_m_pop": "model_log_win_pop", "vol_m_rho": "model_neg_rho"}
    vo = vo.rename(columns={v: k for k, v in vcols.items()})[["race_id"] + list(vcols)]
    df = df.merge(vo, on="race_id", how="left")
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    fin_map = meta.set_index(["race_id", "horse_id"])["finish_position"]
    variants = {"V0": df[base], "V0+S3": df[base + S3],
                "V0+vol": df[base + ["vol_m"]], "V0+vol4": df[base + list(vcols)],
                "V0+S3+vol": df[base + S3 + ["vol_m"]], "V0+S3+vol4": df[base + S3 + list(vcols)]}
    # 荒れ度とレース内の相対（荒れ度 × 馬の強さの相互作用は木が作る）
    oofs, levels = {}, {}
    t0 = time.time()
    for name, X in variants.items():
        res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, [0, 1], THREADS, params=dict(lp))
        oofs[name] = oof
        levels[name] = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
        print(f"{name}: win {levels[name]['logloss_win']:.5f} top2 {levels[name]['logloss_top2']:.5f} top3 {levels[name]['logloss_top3']:.5f} ({time.time() - t0:.0f}s)", flush=True)
    pairs = [("V0+vol", "V0"), ("V0+vol4", "V0"), ("V0+S3+vol", "V0+S3"), ("V0+S3+vol4", "V0+S3")]
    cmp_ = {}
    for new, b in pairs:
        c = fb.compare_oof(oofs[b], oofs[new], fin_map, f"{new} − {b}")
        cmp_[f"{new} − {b}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
        print(f"{new} − {b}", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items() if k in ("Δwin", "win_hi", "Δtop2", "Δtop3", "採用")}, flush=True)
    (OUT / "feature.json").write_text(json.dumps({"levels": levels, "compare": cmp_, "n_variants": len(variants)}, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")


# ───────────────────────── temper（v3 OOF の確率の温度調整）
def temper() -> None:
    from scipy.optimize import minimize

    from src.pipeline.features import tp30_volatility as tv
    from src.pipeline.models import win_models as wm

    o = pd.read_parquet(WORK / "oof/model_oof/ensemble_v3.parquet")
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    fin = pd.read_parquet(FIXED / "features/train_table.parquet", columns=["race_id", "horse_id", "finish_position"])
    fin["race_id"], fin["horse_id"] = fin["race_id"].astype(str), fin["horse_id"].astype(str)
    o = o.merge(fin[["race_id", "horse_id", "finish_position"]], on=["race_id", "horse_id"], how="inner")
    vo = pd.read_parquet(OUT / "vol_oof.parquet")
    kinds = {"model": "model_vol", "market": "market_vol"}
    ks = {"p_win": 1.0, "p_top2": 2.0, "p_top3": 3.0}
    res: dict = {}
    for kname, vcol in kinds.items():
        v = vo.set_index("race_id")[vcol]
        o["v"] = o["race_id"].map(v)
        o = o[o["v"].notna()].copy()
        base_parts, tem_parts, params_by_year = [], [], {}
        for Y in (2023, 2024):
            fit = o[o["year"] < Y].reset_index(drop=True)
            te = o[o["year"] == Y].reset_index(drop=True)
            mu, sd = fit["v"].mean(), fit["v"].std()
            params = {}

            def pack(d, k):
                rid, mask, (pp, yy), pos = tv.pad_races(d["race_id"].to_numpy(), d["p_win"].to_numpy(), (d["finish_position"] <= int(k)).to_numpy(float))
                return rid, mask, pos

            for pc, k in ks.items():
                rid, mask, (P, Yb), pos = tv.pad_races(fit["race_id"].to_numpy(), fit[pc].to_numpy(), (fit["finish_position"] <= int(k)).to_numpy(float))
                zr = ((fit.groupby("race_id", sort=False)["v"].first().loc[rid].to_numpy()) - mu) / sd

                def loss(ab_, P=P, Yb=Yb, mask=mask, zr=zr, k=k):
                    a, b = ab_
                    q = np.clip(tv.temper_batch(P, mask, np.exp(a + b * zr), k), 1e-9, 1 - 1e-9)
                    ll = -(Yb * np.log(q) + (1 - Yb) * np.log(1 - q))
                    return float((ll * mask).sum() / mask.sum())

                params[pc] = minimize(loss, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 80}).x
            params_by_year[Y] = {k_: [float(x) for x in v_] for k_, v_ in params.items()}
            new = te[["race_id", "horse_id", "year", "finish_position"]].copy()
            for pc, k in ks.items():
                rid, mask, (P,), pos = tv.pad_races(te["race_id"].to_numpy(), te[pc].to_numpy())
                zr = ((te.groupby("race_id", sort=False)["v"].first().loc[rid].to_numpy()) - mu) / sd
                a, b = params[pc]
                Q = tv.temper_batch(P, mask, np.exp(a + b * zr), k)
                out = np.zeros(len(te))
                out[pos[mask]] = Q[mask]
                new[pc] = out
            base_parts.append(te)
            tem_parts.append(new)
        base = pd.concat(base_parts)
        tem = pd.concat(tem_parts)
        out_ = {}
        for pc, k in ks.items():
            yy = (base["finish_position"] <= int(k)).to_numpy(float)
            rid = base["race_id"].to_numpy()
            d = wm.paired_bootstrap_diff(wm.per_race_loss(rid, base[pc].to_numpy(), yy), wm.per_race_loss(rid, tem[pc].to_numpy(), yy), n_boot=1000)
            out_[pc] = {"base": wm.binary_logloss(base[pc].to_numpy(), yy), "tempered": wm.binary_logloss(tem[pc].to_numpy(), yy), "diff": d["diff"], "hi": d["hi"]}
            print(kname, pc, {k_: round(v_, 5) for k_, v_ in out_[pc].items()}, flush=True)
        res[kname] = {"by_metric": out_, "rows": len(base), "params": params_by_year}
    (OUT / "temper.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")


if __name__ == "__main__":
    {"prep": prep, "describe": describe, "predict": predict, "feature": feature, "temper": temper}[sys.argv[1]]()
