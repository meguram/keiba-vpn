"""ファクトリー G2（関係者の特徴量）の生成と群ごとの評価。

python -m src.scripts.data.factory_conn_eval build    # 履歴から全群の特徴量を作り、学習表の行に結合して factory/conn/groups/*.parquet に保存
python -m src.scripts.data.factory_conn_eval stage1   # 第1段: valid 2022+2023、seed 0。Δwin<−0.0001 かつ 2 年とも悪化しない群を通す
python -m src.scripts.data.factory_conn_eval stage2   # 第2段: 通過群を valid 2024 + 新 seed(2,3) で確認（採用 = CI 上限<0・効果≥0.0002・連対/複勝の悪化 0.0002 以内）
python -m src.scripts.data.factory_conn_eval combo    # 第2段の採用群（なければ通過群）を同時に追加して 3 年で評価
基準 V0 = 修正後の学習表の tp-03 採用 175 列（LightGBM 固定）。test 2025・2026 年は使わない。結果は factory/conn/results.json（再開可）。
環境変数: FAC_THREADS（既定 3）、FAC_GROUPS（カンマ区切りで群を絞る）
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
OUT = WORK / "factory/conn"
GROUPS = OUT / "groups"
KEY = ["race_id", "horse_id"]


def build() -> None:
    import polars as pl

    from src.pipeline.features import factory_conn as fc
    from src.pipeline.models import ablation as ab

    t0 = time.time()
    GROUPS.mkdir(parents=True, exist_ok=True)
    h = fc.prepare(fc.load_history())
    print(f"[{time.time() - t0:.0f}s] 履歴 {len(h)} 行 / 馬主あり {h['owner'].notna().mean() if 'owner' in h.columns else 0:.3f}", flush=True)
    groups = fc.build_all(h)
    dff = ab.load_table(FIXED / "features/train_table.parquet")[KEY]
    dff["race_id"], dff["horse_id"] = dff["race_id"].astype(str), dff["horse_id"].astype(str)
    meta = {}
    for g, cols in groups.items():
        F = pd.DataFrame({c: np.asarray(a, dtype="float32") for c, a in cols.items()})
        F[KEY[0]], F[KEY[1]] = h["race_id"].to_numpy(), h["horse_id"].to_numpy()
        F = F.drop_duplicates(KEY)
        m = dff.merge(F, on=KEY, how="left")
        assert len(m) == len(dff)
        m.drop(columns=KEY).to_parquet(GROUPS / f"{g}.parquet", index=False)
        meta[g] = {"n_cols": len(cols), "cols": list(cols), "nan_share": float(m.drop(columns=KEY).isna().mean().mean())}
        print(f"[{time.time() - t0:.0f}s] {g}: {len(cols)} 列 欠損率 {meta[g]['nan_share']:.3f}", flush=True)
    (OUT / "groups_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("合計", sum(v["n_cols"] for v in meta.values()), "列 /", len(meta), "群", flush=True)


def _setup():
    from src.pipeline.models import ablation as ab, zoo
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    threads = int(os.environ.get("FAC_THREADS", "3"))
    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((WORK / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    return df, base, lp, meta, folds, threads, meta.set_index(KEY)["finish_position"]


def _group_frame(df: pd.DataFrame, g: str) -> pd.DataFrame:
    """build 時の並び（学習表の保存順）から、評価用の並び（date, race_id, horse_number）へ合わせる。"""
    from src.pipeline.models import ablation as ab

    raw = ab.load_table(FIXED / "features/train_table.parquet")[KEY].copy()
    raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
    G = pd.read_parquet(GROUPS / f"{g}.parquet")
    G[KEY[0]], G[KEY[1]] = raw["race_id"].to_numpy(), raw["horse_id"].to_numpy()
    m = df[KEY].merge(G, on=KEY, how="left")
    m.index = df.index
    return m.drop(columns=KEY)


def _load_results() -> dict:
    p = OUT / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_results(r: dict) -> None:
    (OUT / "results.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _run(name, X, meta, folds, seeds, threads, lp):
    from src.pipeline.models import zoo_runner

    return zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, seeds, threads, params=dict(lp))


def _stage(stage: str) -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base, lp, meta, folds, threads, fin_map = _setup()
    meta_g = json.loads((OUT / "groups_meta.json").read_text(encoding="utf-8"))
    names = [g for g in meta_g if not os.environ.get("FAC_GROUPS") or g in os.environ["FAC_GROUPS"].split(",")]
    res = _load_results()
    st = res.setdefault(stage, {})
    if stage == "stage1":
        fsel, seeds = folds[:2], [0]       # 第1段は粗い選別（1 seed）。通過群は第2段で新 seed と別の年で確認する
    else:
        fsel, seeds = folds[2:], [2, 3]
        names = [g for g in names if res.get("stage1", {}).get(g, {}).get("pass")]
    t0 = time.time()
    if "_base" not in st:
        r0, o0 = _run(f"V0_{stage}", df[base], meta, fsel, seeds, threads, lp)
        st["_base"] = {m: float(r0[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")} | {"by_fold": r0[["fold", "logloss_win"]].round(5).to_dict("records")}
        o0.to_parquet(OUT / f"oof_base_{stage}.parquet", index=False)
        _save_results(res)
        print(f"[{time.time() - t0:.0f}s] V0 {stage}: {st['_base']}", flush=True)
    o0 = pd.read_parquet(OUT / f"oof_base_{stage}.parquet")
    base_by = {d["fold"]: d["logloss_win"] for d in st["_base"]["by_fold"]}
    for g in names:
        if g in st:
            continue
        X = pd.concat([df[base], _group_frame(df, g)], axis=1)
        r, o = _run(f"{g}_{stage}", X, meta, fsel, seeds, threads, lp)
        c = fb.compare_oof(o0, o, fin_map, g)
        by = {d["fold"]: d["logloss_win"] for d in r[["fold", "logloss_win"]].to_dict("records")}
        worse = [k for k, v in by.items() if v > base_by.get(k, 9) + 1e-9]
        c = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}
        c["n_cols"] = int(meta_g[g]["n_cols"])
        c["worse_folds"] = list(worse)
        if stage == "stage1":
            c["pass"] = bool(c["Δwin"] < -0.0001 and not worse)
        st[g] = c
        _save_results(res)
        print(f"[{time.time() - t0:.0f}s] {stage} {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} {'通過/採用' if c.get('pass') or c.get('採用') else ''}", flush=True)
    res["n_groups_tested"] = len([k for k in res.get("stage1", {}) if not k.startswith("_")])
    _save_results(res)


def combo() -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base, lp, meta, folds, threads, fin_map = _setup()
    res = _load_results()
    adopt = [g for g, v in res.get("stage2", {}).items() if not g.startswith("_") and v.get("採用")]
    pick = adopt or [g for g, v in res.get("stage1", {}).items() if not g.startswith("_") and v.get("pass")]
    print("同時追加する群:", pick, flush=True)
    if not pick:
        return
    seeds = [0, 1]
    r0, o0 = _run("V0_combo", df[base], meta, folds, seeds, threads, lp)
    X = pd.concat([df[base]] + [_group_frame(df, g) for g in pick], axis=1)
    r1, o1 = _run("V0+picked", X, meta, folds, seeds, threads, lp)
    c = fb.compare_oof(o0, o1, fin_map, "+".join(pick))
    res["combo"] = {"groups": pick, "n_cols": int(X.shape[1] - len(base)), "base": {m: float(r0[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")},
                    "new": {m: float(r1[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")},
                    "compare": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}}
    _save_results(res)
    print(json.dumps(res["combo"], ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    {"build": build, "stage1": lambda: _stage("stage1"), "stage2": lambda: _stage("stage2"), "combo": combo}[cmd]()
