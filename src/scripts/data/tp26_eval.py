"""tp-26: 馬の成績が連動するコース（コース転移）の対応づけと、勝率予測への増分の評価。

python -m src.scripts.data.tp26_eval prep     # 走ごとの表・年ごとの類似度（置換検定つき）・行ごとの特徴量を data/local/modeling/training_pipeline/tp26/ に作る
python -m src.scripts.data.tp26_eval eval     # LightGBM 固定・valid 2022〜2024 expanding・2 seed。test 2025 は使わない
python -m src.scripts.data.tp26_eval report   # 類似ペアの表（有意/概形/両方）と /tmp/tp26_report.md|json
環境変数: TP26_THREADS（既定 3）、TP26_VARIANTS（カンマ区切りで評価する変種を絞る）
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp26_course_transfer as t

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
OUT = OLD / "tp26"
SNAP_YEARS = [2021, 2022, 2023, 2024, 2025]            # 年 y の行は『y 年 1 月 1 日より前』の走で作った類似度を使う
VARIANTS = ["ALL", "STAT", "GEO", "AND", "OR"]
STAGE_COLS = ["s_time_pred", "s_time_gap", "s_rank_pred"]


def load_df() -> pd.DataFrame:
    from src.pipeline.models import ablation as ab

    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    return df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)


def arch_courses(df: pd.DataFrame) -> pd.Series:
    """会場を問わない原型: 面 | 200m 刻みの距離 | 回り。"""
    surf = df["surface"].astype(str).map(lambda s: "ダート" if s.startswith("ダ") else s)
    dist = (pd.to_numeric(df["distance"], errors="coerce").fillna(0) / 200).round().astype(int) * 200
    return surf + "|" + dist.astype(str) + "|" + df["direction"].astype(str).replace("nan", "")


def prep() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_df()
    labels = pd.read_parquet(OLD / "oof/stage_labels.parquet")
    runs = t.build_runs(df, labels)
    cnt = runs["course"].value_counts()
    courses = sorted(cnt[cnt >= 300].index.tolist())
    print(f"[{time.time() - t0:.0f}s] 走 {len(runs):,} / コース {len(courses)}（300 走以上）", flush=True)
    runs.to_parquet(OUT / "runs.parquet", index=False)
    geo = t.geo_table(courses)
    d = t.geo_distance(geo)
    close = t.geo_close(d, geo)
    knn_d = d[close & (d > 0)]
    tau = float(np.median(knn_d)) if knn_d.size else 1.0
    kernel = np.exp(-(d ** 2) / (2 * tau ** 2))
    geo.to_parquet(OUT / "geo.parquet", index=False)
    np.savez(OUT / "geo.npz", d=d, close=close, kernel=kernel, tau=tau)
    print(f"[{time.time() - t0:.0f}s] 概形: master にあるコース {int(geo['in_master'].sum())}/{len(courses)} / 近傍の対 {int(close.sum() // 2)} / τ={tau:.2f}", flush=True)

    snaps: dict = {}
    for m in t.METRICS:
        for y in SNAP_YEARS + [2026]:
            snaps[(m, y)] = t.similarity_snapshot(runs, courses, m, pd.Timestamp(f"{y}-01-01"), n_perm=40, seed=y)
            s = snaps[(m, y)]
            sig = s["valid"] & (s["q"] < t.Q_MAX) & (s["r_exc"] >= t.R_MIN)
            print(f"[{time.time() - t0:.0f}s] snapshot {m} <{y}: 馬 {s['n_horses']:,} / 有効な対 {int(s['valid'].sum() // 2)} / 有意 {int(sig.sum() // 2)} / a_hat {s['a_hat']:.2f}", flush=True)
    (OUT / "snapshots.pkl").write_bytes(pickle.dumps(snaps))

    feats = {}
    for m in t.METRICS:
        XG = t.asof_vectors(runs, courses, m)
        for v in VARIANTS:
            W = {y: t.weight_matrix(snaps.get((m, y)), v, close, kernel) for y in sorted(runs["year"].unique())}
            W = {y: (w if y in SNAP_YEARS else np.zeros_like(w)) for y, w in W.items()}       # 2020 年は類似度が作れない
            f = t.transfer_features(runs, courses, m, W, XG=XG)
            for c in f.columns:
                feats[(v, c)] = f[c].to_numpy()
        # 対照: 自分のコースの実績（再中心化版）
        c2i = {c: i for i, c in enumerate(courses)}
        ci = runs["course"].map(c2i).fillna(-1).astype(int).to_numpy()
        X, G = XG
        rows = np.arange(len(runs))
        ok = ci >= 0
        ex = np.full(len(runs), np.nan, dtype=np.float32)
        ex[ok] = X[rows[ok], ci[ok]] * G[rows[ok], ci[ok]]
        feats[("EXACT", f"ex_{m}")] = ex
        feats[("EXACT", f"exn_{m}")] = np.where(ok, G[rows, np.maximum(ci, 0)], 0.0).astype(np.float32)
        del X, G, XG
        print(f"[{time.time() - t0:.0f}s] 特徴量 {m}", flush=True)
    key = runs[["race_id", "horse_id"]].copy()
    for (v, c), arr in feats.items():
        key[f"{v}__{c}"] = arr
    key.to_parquet(OUT / "features.parquet", index=False)

    # 原型（会場を問わない）の粒度: STAT と ALL だけ
    r2 = runs.drop(columns=["course"]).merge(df[["race_id", "horse_id"]].assign(course=arch_courses(df)), on=["race_id", "horse_id"], how="left")
    r2 = r2.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    cnt2 = r2["course"].value_counts()
    courses2 = sorted(cnt2[cnt2 >= 300].index.tolist())
    close2 = np.zeros((len(courses2), len(courses2)), dtype=bool)
    kernel2 = np.zeros_like(close2, dtype=float)
    key2 = r2[["race_id", "horse_id"]].copy()
    snaps2 = {}
    for m in t.METRICS:
        XG = t.asof_vectors(r2, courses2, m)
        for y in SNAP_YEARS + [2026]:
            snaps2[(m, y)] = t.similarity_snapshot(r2, courses2, m, pd.Timestamp(f"{y}-01-01"), n_perm=40, seed=100 + y)
        for v in ("ALL", "STAT"):
            W = {y: (t.weight_matrix(snaps2.get((m, y)), v, close2, kernel2) if y in SNAP_YEARS else np.zeros((len(courses2),) * 2)) for y in sorted(r2["year"].unique())}
            f = t.transfer_features(r2, courses2, m, W, XG=XG)
            for c in f.columns:
                key2[f"A{v}__{c}"] = f[c].to_numpy()
        del XG
    key2.to_parquet(OUT / "features_arch.parquet", index=False)
    (OUT / "snapshots_arch.pkl").write_bytes(pickle.dumps(snaps2))
    json.dump({"courses": courses, "courses_arch": courses2}, open(OUT / "courses.json", "w"), ensure_ascii=False)
    print(f"[{time.time() - t0:.0f}s] prep 完了", flush=True)


def evaluate() -> None:
    from src.pipeline.models import feature_bakeoff as fb, zoo, zoo_runner
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    out = OUT / "impact"
    out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("TP26_THREADS", min(3, detect_compute(reserve_cores=6).threads)))
    seeds = [0, 1]
    adopted = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    df = load_df()
    key = ["race_id", "horse_id"]
    f1 = pd.read_parquet(OUT / "features.parquet")
    f2 = pd.read_parquet(OUT / "features_arch.parquet")
    for f in (f1, f2):
        f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    F = df[key].merge(f1, on=key, how="left").merge(f2, on=key, how="left")
    assert len(F) == len(df)
    stage = pd.read_parquet(OLD / "oof/stage_features_expanding.parquet")[key + STAGE_COLS]
    stage["race_id"], stage["horse_id"] = stage["race_id"].astype(str), stage["horse_id"].astype(str)
    S3 = df[key].merge(stage, on=key, how="left")[STAGE_COLS]
    meta = df[key + ["finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    fin_map = meta.set_index(key)["finish_position"]

    def cols(prefix: str, names: list[str]) -> pd.DataFrame:
        return F[[f"{prefix}__{n}" for n in names]].rename(columns=lambda c: c.split("__", 1)[1]).astype("float32")

    ms = list(t.METRICS)
    ct_names = [n for m in ms for n in (f"ct_{m}", f"cn_{m}")]
    variants: dict[str, pd.DataFrame] = {"V0": df[base], "V0+S3": pd.concat([df[base], S3], axis=1)}
    for v in VARIANTS:
        variants[f"V0+{v}"] = pd.concat([df[base], cols(v, ct_names)], axis=1)
    variants["V0+EXACT"] = pd.concat([df[base], cols("EXACT", [n for m in ms for n in (f"ex_{m}", f"exn_{m}")])], axis=1)
    for v in ("ALL", "STAT"):
        variants[f"V0+ARCH_{v}"] = pd.concat([df[base], cols(f"A{v}", ct_names)], axis=1)
    for m in ms:                         # 指標別（STAT だけ）: どの成績の転移が効くか
        variants[f"V0+STAT_{m}"] = pd.concat([df[base], cols("STAT", [f"ct_{m}", f"cn_{m}"])], axis=1)
    only = [x for x in os.environ.get("TP26_VARIANTS", "").split(",") if x]
    names = [n for n in variants if (not only or n in only or n in ("V0", "V0+S3"))]
    res_path = out / "oof.pkl"
    store = pickle.loads(res_path.read_bytes()) if res_path.exists() else {"oof": {}, "res": {}}
    t0 = time.time()
    for name in names:
        if name in store["oof"]:
            continue
        res, oof = zoo_runner.run_oof(name, "lgb_binary", variants[name], meta, folds, seeds, threads, params=dict(lp))
        store["oof"][name], store["res"][name] = oof, res
        res_path.write_bytes(pickle.dumps(store))
        print(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
    cmp_ = {}
    pairs = [(n, "V0") for n in names if n.startswith("V0+") and n != "V0+S3"] + [("V0+S3", "V0")]
    for new, basev in pairs:
        if new in store["oof"] and basev in store["oof"]:
            c = fb.compare_oof(store["oof"][basev], store["oof"][new], fin_map, f"{new} − {basev}")
            cmp_[f"{new} − {basev}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
            print(f"{new} − {basev}", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
    # S3 に重ねた増分（最良の変種と全変種）
    for new in [n for n in names if n.startswith("V0+") and n not in ("V0+S3",)]:
        k = new.replace("V0+", "")
        both = f"V0+S3+{k}"
        if both not in store["oof"] and os.environ.get("TP26_WITH_S3", "1") == "1" and (not only or both.replace("V0+S3+", "V0+") in only):
            Xb = pd.concat([variants["V0+S3"], variants[new].drop(columns=base)], axis=1)
            res, oof = zoo_runner.run_oof(both, "lgb_binary", Xb, meta, folds, seeds, threads, params=dict(lp))
            store["oof"][both], store["res"][both] = oof, res
            res_path.write_bytes(pickle.dumps(store))
            print(f"{both}: win {res['logloss_win'].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
        if both in store["oof"]:
            c = fb.compare_oof(store["oof"]["V0+S3"], store["oof"][both], fin_map, f"{both} − V0+S3")
            cmp_[f"{both} − V0+S3"] = {k2: (float(v) if isinstance(v, (float, np.floating)) else v) for k2, v in c.items()}
            print(f"{both} − V0+S3", {k2: (round(v, 5) if isinstance(v, float) else v) for k2, v in c.items()}, flush=True)
    rep = {"levels": {k: {m: float(v[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")} for k, v in store["res"].items()},
           "by_fold": {k: v[["fold", "logloss_win", "logloss_top2", "logloss_top3"]].round(5).to_dict("records") for k, v in store["res"].items()},
           "compare": cmp_, "seeds": seeds, "rows": len(df), "n_variants_tested": len([n for n in names if n not in ("V0", "V0+S3")])}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("保存:", out / "report.json", flush=True)


def compare_all() -> None:
    """保存済みの OOF（oof.pkl）から全ての比較を作り直して report.json に書く（評価を分けて回したときの統合用）。"""
    from src.pipeline.models import feature_bakeoff as fb

    out = OUT / "impact"
    store = pickle.loads((out / "oof.pkl").read_bytes())
    df = load_df()
    meta = df[["race_id", "horse_id", "finish_position"]]
    fin_map = meta.set_index(["race_id", "horse_id"])["finish_position"]
    names = list(store["oof"])
    cmp_ = {}
    for n in names:
        if n in ("V0",):
            continue
        basev = "V0+S3" if n.startswith("V0+S3+") else "V0"
        if n == "V0+S3":
            basev = "V0"
        c = fb.compare_oof(store["oof"][basev], store["oof"][n], fin_map, f"{n} − {basev}")
        cmp_[f"{n} − {basev}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
    rep = {"levels": {k: {m: float(v[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")} for k, v in store["res"].items()},
           "by_fold": {k: v[["fold", "logloss_win", "logloss_top2", "logloss_top3"]].round(5).to_dict("records") for k, v in store["res"].items()},
           "compare": cmp_, "seeds": [0, 1], "rows": len(df), "n_variants_tested": len([n for n in names if n not in ("V0", "V0+S3") and not n.startswith("V0+S3+")])}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k, v in cmp_.items():
        print(k, f"Δwin {v['Δwin']:+.5f} hi {v['win_hi']:+.5f} Δtop2 {v['Δtop2']:+.5f} Δtop3 {v['Δtop3']:+.5f}", "採用" if v["採用"] else "")


if __name__ == "__main__":
    {"prep": prep, "eval": evaluate, "compare": compare_all}[sys.argv[1]]()
