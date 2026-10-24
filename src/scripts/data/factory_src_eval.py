"""特徴量ファクトリー G4: 候補群の生成・先読み検査・2 段階スクリーニング。

python -m src.scripts.data.factory_src_eval build  GROUP [GROUP ...]   # 候補を data/local/modeling/training_pipeline/factory/src/cand_<GROUP>.parquet に保存
python -m src.scripts.data.factory_src_eval leak   GROUP [GROUP ...]   # 後半の着順を入れ替えても前半の特徴量が変わらないこと
python -m src.scripts.data.factory_src_eval screen GROUP [GROUP ...]   # 第1段 valid 2022+2023（seed 0,1）→ 通過群を第2段 valid 2024（seed 2,3）
基準: 修正後の学習表の tp-03 採用 175 列（V0）。LightGBM 固定。test 2025・2026 年は使わない。
第1段の通過: Δwin < −0.0001 かつ 2 年とも悪化しない。採用: 第2段で CI 上限 < 0・効果 ≥ 0.0002・連対/複勝の悪化 0.0002 以内。
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
OUT = OLD / "factory/src"
REPORT = Path(os.environ.get("FAC_REPORT", "/tmp/fac_src_report.json"))
KEY = ["race_id", "horse_id"]


def load_base() -> pd.DataFrame:
    from src.pipeline.models import ablation as ab

    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    return df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)


def base_cols() -> list[str]:
    a = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    return list(a["b0_features"]) + list(a["features"])


def cmd_build(groups: list[str]) -> None:
    from src.pipeline.features import factory_src as fs

    OUT.mkdir(parents=True, exist_ok=True)
    B = load_base()
    S = None
    for g in groups:
        t0 = time.time()
        if g in fs.GROUPS_SRC and S is None:
            S = fs.load_src(B)
        f = fs.build_group(g, B, S)
        f = f.loc[:, f.notna().any()]
        f.assign(race_id=B["race_id"].to_numpy(), horse_id=B["horse_id"].to_numpy()).to_parquet(OUT / f"cand_{g}.parquet", index=False)
        res = fs.res_score(B)
        rho = {}
        for c in f.columns:
            x = f[c].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(res)
            if ok.sum() > 1000 and np.nanstd(x[ok]) > 0:
                rho[c] = float(pd.Series(x[ok]).rank().corr(pd.Series(res[ok]).rank()))
        sus = {c: round(v, 3) for c, v in rho.items() if abs(v) > 0.25}
        (OUT / f"cand_{g}.rho.json").write_text(json.dumps(rho, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{g}: {f.shape[1]} 列, 欠損率 {f.isna().mean().mean():.3f}, {time.time() - t0:.0f}s / 結果との順位相関が大きい列(>0.25): {sus or 'なし'}", flush=True)


def cmd_leak(groups: list[str]) -> None:
    """後半（2023-01-01 以降）の着順をレース内で入れ替えて作り直し、前半の行の特徴量が変わらないこと。"""
    from src.pipeline.features import factory_src as fs

    B = load_base()
    S = fs.load_src(B)
    B2 = B.copy()
    late = pd.to_datetime(B2["date"]) >= pd.Timestamp("2023-01-01")
    rng = np.random.default_rng(0)
    fp = B2["finish_position"].to_numpy(dtype=float).copy()
    idx = np.flatnonzero(late.to_numpy())
    fp[idx] = rng.permutation(fp[idx])
    B2["finish_position"] = fp
    early = (~late).to_numpy()
    res = {}
    for g in groups:
        a = fs.build_group(g, B, S)
        b = fs.build_group(g, B2, S)
        d = (a[early].to_numpy(dtype=float) - b[early].to_numpy(dtype=float))
        bad = int((np.abs(np.nan_to_num(d, nan=0.0)) > 1e-6).sum() + (np.isnan(a[early].to_numpy(dtype=float)) ^ np.isnan(b[early].to_numpy(dtype=float))).sum())
        res[g] = {"changed_cells_in_early_rows": bad}
        print(g, res[g], flush=True)
    (OUT / "leak_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_srccheck() -> None:
    """SmartRC の件数系の列が as-of（当該レースを含まない / 未来を含まない）かを、学習表の horse_prior_* と比べて確かめる。"""
    from src.pipeline.features import factory_src as fs

    B = load_base()
    S = fs.load_src(B)
    out = {}
    for c, ref in (("cnt_all", "horse_prior_races"), ("cnt_r1", "horse_prior_wins")):
        d = (pd.to_numeric(S[c], errors="coerce") - pd.to_numeric(B[ref], errors="coerce")).dropna()
        out[c] = {"diff_value_counts": d.value_counts(normalize=True).head(5).round(4).to_dict(), "n": int(len(d))}
    # 馬体重の SmartRC 値が結果の値と一致するか（発走後の値が混ざっていないか）
    out["weight_vs_card"] = float((pd.to_numeric(S["weight"], errors="coerce") == pd.to_numeric(B["weight"], errors="coerce")).mean())
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (OUT / "srccheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def _setup():
    from src.pipeline.models import zoo
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    df = load_base()
    bc = base_cols()
    key = KEY
    meta = df[key + ["finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    threads = int(os.environ.get("FAC_THREADS", min(3, detect_compute(reserve_cores=6).threads)))
    return df, bc, meta, folds, lp, threads, meta.set_index(key)["finish_position"]


def _wait_load(limit: float = 16.5) -> None:
    """5 分平均の load が limit 未満になるまで待つ（ホストの負荷を 18 未満に保つ）。"""
    while os.getloadavg()[1] >= limit:
        time.sleep(30)


def cmd_screen(groups: list[str]) -> None:
    from src.pipeline.models import feature_bakeoff as fb, zoo_runner

    df, bc, meta, folds, lp, threads, fin_map = _setup()
    X0 = df[bc]
    s1_seeds = [int(x) for x in os.environ.get("FAC_S1_SEEDS", "0,1").split(",")]
    stage = {"s1": (folds[:2], s1_seeds), "s2": (folds[2:3], [2, 3])}
    rep = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"groups": {}, "tested": []}
    base_oof = {}
    for st, (fl, seeds) in stage.items():
        p = OUT / (f"base_oof_{st}.parquet" if (st != "s1" or seeds == [0, 1]) else f"base_oof_s1_seed{'_'.join(map(str, seeds))}.parquet")
        if p.exists():
            base_oof[st] = pd.read_parquet(p)
        else:
            _wait_load()
            res, oof = zoo_runner.run_oof(f"fac_base_{st}", "lgb_binary", X0, meta, fl, seeds, threads, params=dict(lp))
            oof.to_parquet(p, index=False)
            base_oof[st] = oof
            rep.setdefault("base", {})[st] = {k: float(res[k].mean()) for k in ("logloss_win", "logloss_top2", "logloss_top3")}
            print(f"基準 {st}: win {res['logloss_win'].mean():.5f}", flush=True)
    for g in groups:
        mk = OUT / f"done_{g}.json"
        if mk.exists():                                     # 別のジョブが済ませた群は飛ばす（結果だけ取り込む）
            rep["groups"][g] = json.loads(mk.read_text(encoding="utf-8"))
            if g not in rep["tested"]:
                rep["tested"].append(g)
            continue
        c = pd.read_parquet(OUT / f"cand_{g}.parquet")
        M = df[KEY].merge(c, on=KEY, how="left")
        X = pd.concat([X0.reset_index(drop=True), M.drop(columns=KEY).add_prefix("fac_")], axis=1)
        out = {"n_cols": int(M.shape[1] - 2)}
        passed = True
        for st, (fl, seeds) in stage.items():
            if st == "s2" and not passed:
                break
            t0 = time.time()
            _wait_load()
            res, oof = zoo_runner.run_oof(f"fac_{g}_{st}", "lgb_binary", X, meta, fl, seeds, threads, params=dict(lp))
            cmp_ = fb.compare_oof(base_oof[st], oof, fin_map, g)
            per_year = {}
            if st == "s1":
                m = oof.merge(base_oof[st], on=KEY, suffixes=("_v", "_b")).merge(meta[KEY + ["year", "finish_position"]], on=KEY)
                y = (m["finish_position"] == 1).astype(float).to_numpy()
                for yr, gg in m.groupby("year"):
                    yy = (gg["finish_position"] == 1).astype(float).to_numpy()
                    from src.pipeline.models import win_models as wm
                    per_year[int(yr)] = float(wm.binary_logloss(gg["p_win_v"].to_numpy(), yy) - wm.binary_logloss(gg["p_win_b"].to_numpy(), yy))
                passed = bool(cmp_["Δwin"] < -1e-4 and all(v <= 0 for v in per_year.values()))
            out[st] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cmp_.items()} | ({"per_year": per_year, "passed": passed} if st == "s1" else {})
            print(f"{g} {st}: Δwin {cmp_['Δwin']:+.5f} (上限 {cmp_['win_hi']:+.5f}) top2 {cmp_['Δtop2']:+.5f} top3 {cmp_['Δtop3']:+.5f} {per_year} ({time.time() - t0:.0f}s)", flush=True)
        out["adopted"] = bool(out.get("s2", {}).get("採用", False))
        rep["groups"][g] = out
        mk.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
        if g not in rep["tested"]:
            rep["tested"].append(g)
        REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def cmd_confirm(groups: list[str]) -> None:
    """v3 に近い基準（V0 + 追い切り評価/厩舎コメントの 7 列 + 追い切り時計 TL_A 50 列）に対する、各群の増分。valid 2022〜2024 の 3 fold、seed 0。
    調教系の群が、v3 に既にある情報の言い換えかどうかを確かめる。"""
    from src.pipeline.models import feature_bakeoff as fb, tp05_data, zoo_runner

    df, bc, meta, folds, lp, threads, fin_map = _setup()
    ex = pd.read_parquet(OLD / "features/prerace_extras.parquet").drop(columns=["comment_result_words"], errors="ignore")
    ex["race_id"], ex["horse_id"] = ex["race_id"].astype(str), ex["horse_id"].astype(str)
    E = df[KEY].merge(ex, on=KEY, how="left")
    E["has_comment"] = E["has_comment"].fillna(0.0)
    E["oikiri_eval_rank"] = E.groupby("race_id")["oikiri_eval"].rank(ascending=False, method="average")
    E["oikiri_eval_gap"] = E["oikiri_eval"] - E.groupby("race_id")["oikiri_eval"].transform("max")
    TL = pd.read_parquet(OLD / "training_laps_b/cand_TL_A.parquet")
    TL["race_id"], TL["horse_id"] = TL["race_id"].astype(str), TL["horse_id"].astype(str)
    T = df[KEY].merge(TL, on=KEY, how="left").drop(columns=KEY).add_prefix("tl_")
    X0 = pd.concat([df[bc].reset_index(drop=True), E.drop(columns=KEY).reset_index(drop=True), T.reset_index(drop=True)], axis=1)
    rep = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"groups": {}, "tested": []}
    p = OUT / "base_oof_v3like.parquet"
    if p.exists():
        base = pd.read_parquet(p)
    else:
        _wait_load()
        res, base = zoo_runner.run_oof("fac_base_v3like", "lgb_binary", X0, meta, folds, [0], threads, params=dict(lp))
        base.to_parquet(p, index=False)
        rep.setdefault("base", {})["v3like"] = {k: float(res[k].mean()) for k in ("logloss_win", "logloss_top2", "logloss_top3")}
        print(f"基準(v3 近似) win {res['logloss_win'].mean():.5f}", flush=True)
    for g in groups:
        parts = [pd.read_parquet(OUT / f"cand_{x}.parquet").set_index(KEY).add_prefix(f"{x}_") for x in g.split("+")]   # 「A+B」で複数群をまとめて足す
        M = df[KEY].merge(pd.concat(parts, axis=1).reset_index(), on=KEY, how="left")
        X = pd.concat([X0, M.drop(columns=KEY).reset_index(drop=True).add_prefix("fac_")], axis=1)
        _wait_load()
        t0 = time.time()
        res, oof = zoo_runner.run_oof(f"fac_{g}_v3like", "lgb_binary", X, meta, folds, [0], threads, params=dict(lp))
        c_ = fb.compare_oof(base, oof, fin_map, g)
        rep["groups"].setdefault(g, {})["v3like"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c_.items()}
        print(f"{g} v3like: Δwin {c_['Δwin']:+.5f} (上限 {c_['win_hi']:+.5f}) top2 {c_['Δtop2']:+.5f} top3 {c_['Δtop3']:+.5f} 採用={c_['採用']} ({time.time() - t0:.0f}s)", flush=True)
        REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"build": cmd_build, "leak": cmd_leak, "screen": cmd_screen, "confirm": cmd_confirm}.get(cmd, lambda a: cmd_srccheck())(args)
