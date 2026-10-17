"""tp-29: 期待とずれた馬（期待外れ / 期待以上）のグループ差分から作るコース適性の評価。

python -m src.scripts.data.tp29_eval prep   # ラベル（市場版・モデル期待版）・属性・特徴量を作る（年 ≤ 2024 のみ。結果は data/local/modeling/training_pipeline/tp29/）
python -m src.scripts.data.tp29_eval win    # LightGBM 固定・valid 2022〜2024 expanding・2 seed。test 2025 は使わない
python -m src.scripts.data.tp29_eval lr     # 人気を制御したときの適性スコアの予測力（同じ人気の馬どうしで期待以上/期待外れを当てる）
python -m src.scripts.data.tp29_eval doc    # コース別の向き/不向きの要因を docs の 1 章にする

人気は過去レースのラベルを作る材料としてだけ使う（特徴量に当日の人気は入れない）。モデル期待版は v3 の OOF 勝率の順位（2022〜2024 年）。
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
OLD = ROOT / "data/local/modeling/training_pipeline"
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
PED = ROOT / "data/local/modeling/training_pipeline_ped"
OUT = OLD / "tp29"
MAX_YEAR = 2024            # test 2025・2026 年は使わない
FEAT_PREFIXES = ("apt", "hz_")


def _tables(years=range(2020, MAX_YEAR + 1)) -> pd.DataFrame:
    import polars as pl

    cols = ["race_id", "horse_id", "popularity", "time_sec"]
    parts = [pl.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet", columns=cols).to_pandas() for y in years]
    rr = pd.concat(parts, ignore_index=True)
    rr["race_id"], rr["horse_id"] = rr["race_id"].astype(str), rr["horse_id"].astype(str)
    return rr.drop_duplicates(["race_id", "horse_id"])


def load_frame() -> pd.DataFrame:
    from src.pipeline.features import course_variant as cv
    from src.pipeline.models import ablation as ab

    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    for c in ("venue_code", "surface", "jockey_id", "trainer_id", "course_type"):
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "").astype(str)
    df = df[df["year"] <= MAX_YEAR].copy()
    df = df.merge(_tables(), on=["race_id", "horse_id"], how="left")
    keys = pd.read_parquet(PED / "pedigree_keys.parquet")
    df = df.merge(keys.drop_duplicates("horse_id"), on="horse_id", how="left")
    df["sire_id"], df["dam_sire_id"] = df["sire_id"].fillna(""), df["dam_sire_id"].fillna("")
    df["variant"] = cv.add_course_variant(df, parsed_col="course_type", parsed_available=False)
    oof = pd.read_parquet(OLD / "oof/model_oof/ensemble_v3.parquet", columns=["race_id", "horse_id", "p_win"])
    oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
    oof["mrank"] = oof.groupby("race_id")["p_win"].rank(ascending=False, method="first")
    df = df.merge(oof[["race_id", "horse_id", "mrank"]], on=["race_id", "horse_id"], how="left")
    return df.reset_index(drop=True)


def prep() -> None:
    from src.pipeline.features import tp29_course_aptitude as ca

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    print(f"[{time.time() - t0:.0f}s] 行 {len(df)} / 年 {sorted(df['year'].unique())}", flush=True)
    tok = ca.attribute_tokens(df)
    kk = ca.course_keys(df, df["variant"])
    base = pd.DataFrame({"race_id": df["race_id"], "horse_id": df["horse_id"], "year": df["year"], "date": df["date"], "surf": df["surface"].astype(str),
                         "popularity": df["popularity"], "mrank": df["mrank"], "finish_position": df["finish_position"]})
    F = pd.concat([base, kk, tok], axis=1)
    counts = {}
    for mode, rank_col in (("pop", "popularity"), ("model", "mrank")):
        g = ca.make_groups(df.assign(rank=df[rank_col]), "rank")
        Fm = F.assign(g1=g["g1"].to_numpy(), g2=g["g2"].to_numpy(), labelable=g["labelable"].to_numpy(), gap3=g["gap3"].to_numpy())
        feats, tables = ca.build_features(Fm)
        out = pd.concat([Fm[["race_id", "horse_id", "year"]], feats], axis=1)
        out.to_parquet(OUT / f"feats_{mode}.parquet", index=False)
        Fm[["race_id", "horse_id", "year", "k1", "k2", "k3", "g1", "g2", "labelable", "gap3"]].to_parquet(OUT / f"labels_{mode}.parquet", index=False)
        by_year = Fm.groupby("year").agg(rows=("g1", "size"), labelable=("labelable", "sum"), g1=("g1", "sum"), g2=("g2", "sum")).reset_index()
        counts[mode] = by_year.to_dict("records")
        print(f"[{time.time() - t0:.0f}s] {mode}: ", by_year.to_dict("records"), flush=True)
        # 最終の統計（年 ≤ 2024 のすべてで作る。ドキュメント用。モデルには使わない）
        S = Fm[Fm["g1"] | Fm["g2"]]
        fin = ca.build_tables(S)
        for lv, t in fin.items():
            t.to_parquet(OUT / f"tables_final_{mode}_{lv}.parquet", index=False)
        print(f"  最終統計 有意: " + ", ".join(f"{lv}={int(t['sig'].sum())}/{len(t)}" for lv, t in fin.items()), flush=True)
        # 窓ごとの有意数
        win_sig = {int(y): {lv: int(t["sig"].sum()) for lv, t in tb.items()} for y, tb in tables.items()}
        counts[f"{mode}_window_sig"] = win_sig
        print("  窓ごとの有意数", win_sig, flush=True)
        # コース別の件数（k2）
        ck = Fm.groupby("k2").agg(rows=("g1", "size"), labelable=("labelable", "sum"), g1=("g1", "sum"), g2=("g2", "sum")).reset_index()
        ck.to_csv(OUT / f"counts_k2_{mode}.csv", index=False)
    (OUT / "counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _load_eval_frame():
    from src.pipeline.models import ablation as ab

    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    return df


def evaluate() -> None:
    from src.pipeline.models import feature_bakeoff as fb, zoo, zoo_runner
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    out = OUT / "impact"
    out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("TP29_THREADS", min(5, detect_compute(reserve_cores=6).threads)))
    seeds = [0, 1]
    adopted = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    df = _load_eval_frame()
    key = ["race_id", "horse_id"]
    stage = pd.read_parquet(OLD / "oof/stage_features_expanding.parquet")
    s3cols = ["s_time_pred", "s_time_gap", "s_rank_pred"]
    df = df.merge(stage[key + s3cols].astype({"race_id": str, "horse_id": str}), on=key, how="left")
    fp = pd.read_parquet(OUT / "feats_pop.parquet")
    fm = pd.read_parquet(OUT / "feats_model.parquet")
    for f in (fp, fm):
        f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    cols = [c for c in fp.columns if c.startswith(FEAT_PREFIXES)]
    raw_cols = [c for c in cols if c.startswith("apt") and not c.startswith("aptx")]
    x_cols = [c for c in cols if c.startswith("aptx")]
    hz_cols = [c for c in cols if c.startswith("hz_")]
    P = df[key].merge(fp[key + cols], on=key, how="left")[cols].astype("float32")
    M = df[key].merge(fm[key + cols], on=key, how="left")[cols].astype("float32")
    P.index = M.index = df.index
    meta = df[key + ["finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    fin_map = meta.set_index(key)["finish_position"]
    B0, B3 = df[base], df[base + s3cols]
    variants = {
        "V0": B0, "V0S3": B3,
        "P_all": pd.concat([B0, P[cols]], axis=1), "P_raw": pd.concat([B0, P[raw_cols]], axis=1), "P_x": pd.concat([B0, P[x_cols]], axis=1), "P_hz": pd.concat([B0, P[hz_cols]], axis=1),
        "PS_all": pd.concat([B3, P[cols]], axis=1),
        "M_all": pd.concat([B0, M[cols]], axis=1), "MS_all": pd.concat([B3, M[cols]], axis=1),
    }
    only = os.environ.get("VARIANTS")
    if only:
        variants = {k: v for k, v in variants.items() if k in only.split(",")}
    oofs, res_all = {}, {}
    t0 = time.time()
    for name, X in variants.items():
        res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, seeds, threads, params=dict(lp))
        oofs[name], res_all[name] = oof, res
        print(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} cols {X.shape[1]} ({time.time() - t0:.0f}s)", flush=True)
    pairs = [("P_all", "V0", "市場版 全部（P_all − V0）"), ("P_raw", "V0", "市場版 G2−G1 の差のみ（P_raw − V0）"), ("P_x", "V0", "市場版 コース固有の超過のみ（P_x − V0）"), ("P_hz", "V0", "市場版 馬自身の履歴のみ（P_hz − V0）"),
             ("PS_all", "V0S3", "市場版 全部（PS_all − V0S3）"), ("M_all", "V0", "モデル期待版 全部（M_all − V0）"), ("MS_all", "V0S3", "モデル期待版 全部（MS_all − V0S3）")]
    yr = meta.set_index(key)["year"]
    cmp_ = {}
    for new, basev, label in pairs:
        if new not in oofs or basev not in oofs:
            continue
        c = fb.compare_oof(oofs[basev], oofs[new], fin_map, label)
        cmp_[label] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
        print(label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
        if label.startswith("モデル期待版"):             # ラベルのある 2023・2024 年だけでも比べる
            sel = lambda o: o[yr.loc[list(zip(o["race_id"], o["horse_id"]))].isin([2023, 2024]).to_numpy()]
            c2 = fb.compare_oof(sel(oofs[basev]), sel(oofs[new]), fin_map, label + " [2023-24のみ]")
            cmp_[label + " [2023-24のみ]"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c2.items()}
            print(label + " [2023-24のみ]", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c2.items()}, flush=True)
    rep = {"levels": {k: {m: float(v[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")} for k, v in res_all.items()},
           "by_fold": {k: v[["fold", "logloss_win", "logloss_top2", "logloss_top3"]].round(5).to_dict("records") for k, v in res_all.items()},
           "compare": cmp_, "seeds": seeds, "rows": len(df), "n_cols": len(cols), "cols": cols}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("保存:", out / "report.json", flush=True)


def lr() -> None:
    """人気を制御した適性スコアの予測力。(1) 人気 6 位以降の馬の中で G2（期待以上）を当てる、(2) 人気 1〜3 位の馬の中で G1（期待外れ）を当てる。
    年 Y の検証 = 年 2021..Y-1 で学習した LR を年 Y で評価（スコアは年 Y より前のラベルだけで作ってある）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    lab = pd.read_parquet(OUT / "labels_pop.parquet")
    ft = pd.read_parquet(OUT / "feats_pop.parquet")
    df = load_frame()[["race_id", "horse_id", "year", "popularity", "field_size"]]
    for d in (lab, ft):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    D = df.merge(lab[["race_id", "horse_id", "g1", "g2", "labelable"]], on=["race_id", "horse_id"]).merge(ft.drop(columns=["year"]), on=["race_id", "horse_id"])
    D = D[D["labelable"]]
    D["lpop"] = np.log(D["popularity"].clip(lower=1))
    D["rel_pop"] = D["popularity"] / D["field_size"].clip(lower=1)
    D["hz_g2_rate"] = (D["hz_g2_all"] + 0.5) / (D["hz_n_all"] + 5.0)
    D["hz_g1_rate"] = (D["hz_g1_all"] + 0.5) / (D["hz_n_all"] + 5.0)
    ctrl = ["lpop", "field_size", "rel_pop", "hz_n_all"]
    sets = {"course_raw": ["apt1_sum", "apt2_sum", "apt3_sum"], "course_excess": ["aptx1_sum", "aptx2_sum", "aptx3_sum"],
            "course_all": ["apt1_sum", "apt2_sum", "apt3_sum", "aptx1_sum", "aptx2_sum", "aptx3_sum"],
            "horse_hist": ["hz_g2_rate", "hz_g1_rate", "hz_g2_k2", "hz_g1_k2"],
            "all": ["apt1_sum", "apt2_sum", "apt3_sum", "aptx1_sum", "aptx2_sum", "aptx3_sum", "hz_g2_rate", "hz_g1_rate", "hz_g2_k2", "hz_g1_k2"]}
    rep: dict = {}
    for name, sub, ycol in (("G2_among_pop6plus", D[D["popularity"] >= 6], "g2"), ("G1_among_pop1to3", D[D["popularity"] <= 3], "g1")):
        rep[name] = {}
        for Y in (2022, 2023, 2024):
            tr = sub[(sub["year"] < Y) & sub["apt2_sum"].notna()]
            te = sub[(sub["year"] == Y) & sub["apt2_sum"].notna()]
            if len(tr) < 2000 or te[ycol].sum() < 30:
                rep[name][Y] = {"skip": f"train {len(tr)} / test positives {int(te[ycol].sum())}"}
                continue

            def fit_pred(cols):
                sc = StandardScaler().fit(tr[cols].fillna(0.0))
                m = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(tr[cols].fillna(0.0)), tr[ycol])
                return m, m.predict_proba(sc.transform(te[cols].fillna(0.0)))[:, 1]

            y = te[ycol].to_numpy()
            races = te["race_id"].to_numpy()
            ur = pd.unique(races)
            idx = {r: np.flatnonzero(races == r) for r in ur}
            boots = [np.concatenate([idx[r] for r in rng.choice(ur, len(ur))]) for _ in range(150)]
            boots = [ii for ii in boots if 0 < y[ii].sum() < len(ii)]
            _, p0 = fit_pred(ctrl)
            a0 = roc_auc_score(y, p0)
            res = {"n_train": int(len(tr)), "n_test": int(len(te)), "pos_test": int(y.sum()), "auc_ctrl": round(a0, 4)}
            for sname, cols in sets.items():
                m1, p1 = fit_pred(ctrl + cols)
                d = [roc_auc_score(y[ii], p1[ii]) - roc_auc_score(y[ii], p0[ii]) for ii in boots]
                res[sname] = {"auc": round(roc_auc_score(y, p1), 4), "dAUC": round(roc_auc_score(y, p1) - a0, 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                              "coef_std": dict(zip(cols, m1.coef_[0][len(ctrl):].round(3)))}
            rep[name][Y] = res
            print(name, Y, {k: (v if not isinstance(v, dict) else (v["dAUC"], v["ci95"])) for k, v in res.items()}, flush=True)
    (OUT / "lr_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    {"prep": prep, "win": evaluate, "lr": lr}[sys.argv[1]]()
