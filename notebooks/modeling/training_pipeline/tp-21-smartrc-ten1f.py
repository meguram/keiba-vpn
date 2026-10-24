#!/usr/bin/env python3
"""tp-21 SmartRC テン1F（スタート後 1F）特徴量の ablation（保存済みデータのみ。追加の外部アクセスなし）。

実行場所: ホスト PC（stg）。ホストのリポジトリには手を入れず、このファイルと src/pipeline/features/smartrc_ten1f.py を
/tmp/smartrc_ablation/ に置いて実行する（モジュールは src. からの import に失敗したら同じ場所のものを使う）。

  python tp-21-smartrc-ten1f.py extract            # GCS の smartrc_race(2020〜2025) を 1 回ずつ読み、runners / career を parquet に保存
  python tp-21-smartrc-ten1f.py coverage           # カバレッジ（空レース・テン欠損の理由別）と、先読みの確認
  python tp-21-smartrc-ten1f.py win                # p_win の LogLoss: ベース vs ＋SmartRC（tp-03 と同じ分割・基準）
  python tp-21-smartrc-ten1f.py front              # 前段(序盤): 当該レースのテン1F(相対)・最初のコーナー位置の予測精度
  python tp-21-smartrc-ten1f.py oof                # 前段の予測を年ごとの expanding OOF で作り、p_win に足した効果
  環境変数: SEEDS=0,1,2（既定）/ SPECS=BASE,+SX_all（win で評価する spec を絞る）/ THREADS=4 / SMARTRC_OUT=出力先

分割: 学習 2020〜2025、valid 2022/2023/2024 の expanding window（purge 7 日 + embargo 3 日）、seed 3 通り。test 2025 は使わない。
採否: 共通の全 fold で悪化せず、平均の改善が max(0.0001, 2×seed 間のばらつき) 以上（tp-03 と同じ）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("KEIBA_REPO_ROOT") or "/home/jovyan/work/keiba-vpn")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from src.pipeline.features import smartrc_ten1f as S
except ImportError:                      # ホストのリポジトリにはまだ無い
    import smartrc_ten1f as S

TP = REPO / "data/local/modeling/training_pipeline"
OUT = Path(os.environ.get("SMARTRC_OUT") or TP / "smartrc_ten1f")
THREADS = int(os.environ.get("THREADS", "4"))
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))
BASE_PREFIX = "chuou/data/preprocessed/netkeiba/pc/smartrc_race"
YEARS = ("2020", "2021", "2022", "2023", "2024", "2025")


# ── 抽出 ─────────────────────────────────────────────────────────────────────

def _bucket():
    os.chdir(REPO)
    from src.scraper.storage import HybridStorage

    return HybridStorage()._get_bucket()


def _work(args):
    names = args
    b = _bucket()
    runners, careers, empty = [], [], []
    for nm in names:
        rid = nm.rsplit("/", 1)[-1][:-5]
        d = json.loads(b.blob(nm).download_as_bytes())
        r, c = S.extract_race(d, rid)
        if not r:
            empty.append(rid)
        runners += r
        careers += c
    return pd.DataFrame(runners), S.career_table(careers), empty


def cmd_extract():
    from multiprocessing import Pool

    OUT.mkdir(parents=True, exist_ok=True)
    b = _bucket()
    names = []
    for y in YEARS:
        names += [x.name for x in b.list_blobs(prefix=f"{BASE_PREFIX}/{y}/") if x.name.endswith(".json")]
    print("files", len(names), flush=True)
    chunks = [names[i:i + 100] for i in range(0, len(names), 100)]
    R, C, E = [], [], []
    t0 = time.time()
    with Pool(int(os.environ.get("PROCS", "4"))) as p:
        for i, (r, c, e) in enumerate(p.imap(_work, chunks), 1):
            R.append(r); C.append(c); E += e
            if i % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
    runners = pd.concat(R, ignore_index=True)
    career = pd.concat(C, ignore_index=True).drop_duplicates(["horse_id", "rcode"]).reset_index(drop=True)
    runners.to_parquet(OUT / "runners.parquet", index=False)
    career.to_parquet(OUT / "career.parquet", index=False)
    json.dump({"empty_races": sorted(E), "n_runner_rows": len(runners), "n_career_rows": len(career)}, open(OUT / "extract_meta.json", "w"))
    print("runners", runners.shape, "career", career.shape, "empty races", len(E))


# ── 共通: 学習用テーブルに SmartRC 特徴量を付ける ────────────────────────────────────

def load_frame(with_unverified: bool = False):
    from src.pipeline.models import ablation as ab
    from src.pipeline.models.time_split import restrict_to_training_years

    df = ab.load_table(TP / "features/train_table.parquet")
    df = restrict_to_training_years(df).reset_index(drop=True)
    runners = pd.read_parquet(OUT / "runners.parquet")
    career = pd.read_parquet(OUT / "career.parquet")
    ent = df[["race_id", "horse_id", "date", "distance", "surface"]].copy()
    ent["surface"] = ent["surface"].astype(str)
    feats = S.build_features(ent, career, runners)
    cols = S.CORE + S.RELATIVE + S.RUNNER_RANKS
    assert (feats["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (feats["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
    for c in cols:
        df[c] = feats[c].to_numpy()
    tg = S.race_targets(df[["race_id", "horse_id", "date"]], career)
    for c in ("ten_actual", "ten_rel_actual", "c1_rel_actual"):
        df[c] = tg[c].to_numpy()
    if with_unverified:
        r = runners.drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])
        idx = pd.MultiIndex.from_arrays([df["race_id"], df["horse_id"]])
        for c in S.UNVERIFIED:
            df["sxu_" + c] = pd.to_numeric(r[c].astype(str).str.strip(), errors="coerce").reindex(idx).to_numpy()
    return df, ab


def base_columns(ab):
    a = json.loads((TP / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    assert a["best"] == "ALL"
    return list(a["b0_features"]) + list(a["features"])


# ── カバレッジ ───────────────────────────────────────────────────────────────

def cmd_coverage():
    df, _ = load_frame()
    meta = json.loads((OUT / "extract_meta.json").read_text())
    rows = []
    df["year"] = pd.to_datetime(df["date"]).dt.year
    for y, g in df.groupby("year"):
        debut = g["sx_n_prior"] == 0
        no_ten = (g["sx_n_prior"] > 0) & (g["sx_ten_n"] == 0)
        rows.append({"year": int(y), "rows": len(g), "races": g["race_id"].nunique(),
                     "初出走(過去走なし)": float(debut.mean()), "過去走ありテンなし": float(no_ten.mean()), "テンあり(特徴量が付く)": float((g["sx_ten_n"] > 0).mean()),
                     "平均テン本数": float(g.loc[g["sx_ten_n"] > 0, "sx_ten_n"].mean()), "正解(当該レースのテン)あり": float(g["ten_actual"].notna().mean()),
                     "正解 ten_rel あり": float(g["ten_rel_actual"].notna().mean()), "corner1 正解あり": float(g["c1_rel_actual"].notna().mean())})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "coverage_by_year.csv", index=False)
    print(cov.round(3).to_string(index=False))
    print("runners が空のレース:", len(meta["empty_races"]), meta["empty_races"][:5])
    # 先読みの確認: 特徴量の最新走の日付は常に当該レース日より前
    career = pd.read_parquet(OUT / "career.parquet")
    runners = pd.read_parquet(OUT / "runners.parquet")
    bad = 0; tot = 0
    for i in range(1, 6):
        h = runners[f"h{i}_rcode"].astype(str).str.strip()
        ok = h.str.len() >= 8
        tot += int(ok.sum()); bad += int((h[ok].str[:8] >= runners.loc[ok, "date"].astype(str)).sum())
    print(f"h1〜h5 の過去走日付 >= 当該レース日: {bad} / {tot}")
    # ten1f_best が『過去走の最小』と一致するか（as-of の確認）
    m = runners[["race_id", "horse_id", "date", "ten1f_best"]].copy()
    m["best"] = m["ten1f_best"].map(S.parse_ten1f)
    m = m.dropna(subset=["best"]).sample(min(20000, len(m)), random_state=0)
    c = career.dropna(subset=["ten"])[["horse_id", "rdate", "ten"]]
    j = m.merge(c, on="horse_id")
    j = j[j["rdate"] < j["date"].astype(str)]
    asof_min = j.groupby(["race_id", "horse_id"])["ten"].min().rename("asof_min")
    k = m.set_index(["race_id", "horse_id"]).join(asof_min, how="inner")
    print("ten1f_best == 過去走の最小(as-of):", float((np.isclose(k["best"], k["asof_min"])).mean()), "n =", len(k))
    json.dump({"h_past_dates_ge_race_date": [bad, tot], "ten1f_best_eq_asof_min": float(np.isclose(k['best'], k['asof_min']).mean()), "n": len(k)},
              open(OUT / "leak_check.json", "w"))


# ── p_win ───────────────────────────────────────────────────────────────────

def cmd_win():
    from src.pipeline.models.time_split import yearly_expanding_folds

    df, ab = load_frame(with_unverified=True)
    base = base_columns(ab)
    folds = list(yearly_expanding_folds(df))
    res_path = OUT / "win_results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    done = {(r["exp"], r["fold"], r["seed"], r["target"]) for r in rows}
    unv = ["sxu_" + c for c in S.UNVERIFIED]
    specs = [ab.Spec("BASE", base), ab.Spec("+SX_core", base + S.CORE), ab.Spec("+SX_core+rel", base + S.CORE + S.RELATIVE),
             ab.Spec("+SX_all", base + S.CORE + S.RELATIVE + S.RUNNER_RANKS),
             ab.Spec("+SX_all+未検証(ten_has等)", base + S.CORE + S.RELATIVE + S.RUNNER_RANKS + unv)]
    only = os.environ.get("SPECS")
    for sp in specs:
        if only and sp.name not in only.split(","):
            continue
        for fold in folds:
            todo = [s for s in SEEDS if (sp.name, fold.fold_id, s, "win") not in done]
            if not todo:
                continue
            t0 = time.time()
            for r in ab.evaluate(df, [fold], sp, todo, THREADS, ("win",)):
                rows.append(r)
            pd.DataFrame(rows).to_csv(res_path, index=False)
            print(f"[{time.strftime('%H:%M:%S')}] {sp.name} {fold.fold_id} {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    summ = ab.summarize(res)
    summ.to_csv(OUT / "win_summary.csv")
    cmp_ = ab.compare_to_base(summ, "BASE")
    cmp_.to_csv(OUT / "win_compare.csv")
    print(summ.round(5).to_string()); print(cmp_.round(5).to_string())


# ── 前段（序盤の速さ・位置取り）────────────────────────────────────────────────

def _spearman_in_race(race, a, b, min_n=5):
    d = pd.DataFrame({"r": race, "a": a, "b": b})
    out = []
    for _, g in d.groupby("r"):
        if len(g) >= min_n and g["a"].nunique() > 1 and g["b"].nunique() > 1:
            out.append(g["a"].rank().corr(g["b"].rank()))
    return float(np.nanmean(out)) if out else float("nan")


def cmd_front():
    import lightgbm as lgb
    from src.pipeline.models.time_split import yearly_expanding_folds

    df, ab = load_frame()
    base = base_columns(ab)
    folds = list(yearly_expanding_folds(df))
    params = dict(ab.LGB_PARAMS, objective="regression", metric="l2")
    res_path = OUT / "front_results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    done = {(r["exp"], r["fold"], r["seed"], r["target"]) for r in rows}
    sets = {"BASE": base, "+SX_all": base + S.CORE + S.RELATIVE + S.RUNNER_RANKS}
    targets = {"ten_rel": "ten_rel_actual", "c1_rel": "c1_rel_actual"}
    cats = [c for c in base if str(df[c].dtype) == "category"]
    for tname, tcol in targets.items():
        ok = df[tcol].notna().to_numpy()
        for fold in folds:
            tr = fold.train_index[ok[fold.train_index]]
            va = fold.valid_index[ok[fold.valid_index]]
            for ename, feats in sets.items():
                ds = None
                for seed in SEEDS:
                    if (ename, fold.fold_id, seed, tname) in done:
                        continue
                    ds = ds or lgb.Dataset(df.iloc[tr][feats], label=df[tcol].to_numpy()[tr], categorical_feature=cats or "auto", free_raw_data=False)
                    bst = lgb.train(dict(params, seed=seed, num_threads=THREADS), ds, num_boost_round=ab.N_ROUNDS)
                    pred = bst.predict(df.iloc[va][feats])
                    y = df[tcol].to_numpy()[va]
                    rows.append({"exp": ename, "fold": fold.fold_id, "seed": seed, "target": tname, "n_train": len(tr), "n_valid": len(va),
                                 "mae": float(np.mean(np.abs(pred - y))), "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                                 "spearman_in_race": _spearman_in_race(df["race_id"].to_numpy()[va], pred, y),
                                 "mae_const": float(np.mean(np.abs(y - np.median(df[tcol].to_numpy()[tr]))))})
                    pd.DataFrame(rows).to_csv(res_path, index=False)
                print(f"[{time.strftime('%H:%M:%S')}] {tname} {fold.fold_id} {ename}", flush=True)
    r = pd.DataFrame(rows)
    per = r.groupby(["target", "exp", "fold"])[["mae", "rmse", "spearman_in_race", "mae_const"]].mean().round(5)
    print(per.to_string())
    per.to_csv(OUT / "front_summary.csv")


# ── 前段の予測を OOF で p_win に足す ──────────────────────────────────────────────

def cmd_oof():
    import lightgbm as lgb
    from src.pipeline.models.time_split import yearly_expanding_folds

    df, ab = load_frame()
    base = base_columns(ab)
    sx = base + S.CORE + S.RELATIVE + S.RUNNER_RANKS
    cats = [c for c in base if str(df[c].dtype) == "category"]
    params = dict(ab.LGB_PARAMS, objective="regression", metric="l2")
    years = df["year"].to_numpy()
    cache = OUT / "oof_front.parquet"
    if cache.exists():
        o = pd.read_parquet(cache)
        for c in o.columns:
            df[c] = o[c].to_numpy()
    else:
        cols = {}
        for var, feats in (("A", base), ("B", sx)):
            for tname, tcol in (("ten", "ten_rel_actual"), ("c1", "c1_rel_actual")):
                col = np.full(len(df), np.nan)
                y_all = df[tcol].to_numpy()
                for Y in (2021, 2022, 2023, 2024):
                    tr = np.where((years < Y) & ~np.isnan(y_all))[0]
                    va = np.where(years == Y)[0]
                    bst = lgb.train(dict(params, seed=0, num_threads=THREADS), lgb.Dataset(df.iloc[tr][feats], label=y_all[tr], categorical_feature=cats or "auto"),
                                    num_boost_round=ab.N_ROUNDS)
                    col[va] = bst.predict(df.iloc[va][feats])
                    print(f"[{time.strftime('%H:%M:%S')}] oof {var} {tname} {Y}", flush=True)
                cols[f"oof{var}_{tname}"] = col
        for c, v in cols.items():
            df[c] = v
        pd.DataFrame(cols).to_parquet(cache, index=False)
    folds = list(yearly_expanding_folds(df))
    specs = [ab.Spec("BASE", base), ab.Spec("BASE+OOF_A(ベースの特徴量で作った序盤予測)", base + ["oofA_ten", "oofA_c1"]),
             ab.Spec("BASE+OOF_B(SmartRC を足して作った序盤予測)", base + ["oofB_ten", "oofB_c1"]),
             ab.Spec("BASE+SX_all+OOF_B", sx + ["oofB_ten", "oofB_c1"])]
    res_path = OUT / "oof_win_results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    done = {(r["exp"], r["fold"], r["seed"], r["target"]) for r in rows}
    for sp in specs:
        for fold in folds:
            todo = [s for s in SEEDS if (sp.name, fold.fold_id, s, "win") not in done]
            if todo:
                rows += ab.evaluate(df, [fold], sp, todo, THREADS, ("win",))
                pd.DataFrame(rows).to_csv(res_path, index=False)
                print(f"[{time.strftime('%H:%M:%S')}] {sp.name} {fold.fold_id}", flush=True)
    res = pd.DataFrame(rows)
    summ = ab.summarize(res)
    summ.to_csv(OUT / "oof_win_summary.csv")
    print(summ.round(5).to_string()); print(ab.compare_to_base(summ, "BASE").round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "coverage", "win", "front", "oof"])
    a = ap.parse_args()
    {"extract": cmd_extract, "coverage": cmd_coverage, "win": cmd_win, "front": cmd_front, "oof": cmd_oof}[a.cmd]()
