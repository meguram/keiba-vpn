"""特徴量ファクトリー G3（レース・メンバー構成・コース/環境）の候補を作って、V0（tp-03 採用 175 列）への増分を測る。

python -m src.scripts.data.factory_field_eval build    # 候補を作って data/local/modeling/training_pipeline/factory/field/<群>.parquet に保存
python -m src.scripts.data.factory_field_eval stage1   # valid 2022+2023、seed 0,1 で群ごとにスクリーニング
python -m src.scripts.data.factory_field_eval stage2   # 通過群を valid 2024、seed 2,3 で確認 + 通過群の同時追加
python -m src.scripts.data.factory_field_eval report   # /tmp/fac_field_report.{json,md}
採否: 勝率 LogLoss の差の 95% 上限 < 0、効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002（stage2 の結果で判定）。test 2025・2026 は使わない。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
OUT = OLD / "factory/field"
KEY = ["race_id", "horse_id"]
# 群を細かく分けて調べる（群の総数 = ここの数）
SUBGROUPS = {
    "mem_strength": ("mem_strength", r".*"),
    "mem_strength_horse": ("mem_strength", r"^fm_(f3_r|sr_exp|sr_perf_ewm2|c_res_ewm2|b_aff_all|a_l3fz_ewm2|c_cls_ewm2|horse_prior_win_rate)_"),
    "mem_demo": ("mem_demo", r".*"),
    "mem_demo_body": ("mem_demo", r"^fd_(age|jw|wt|wchg|jwratio)_|^fd_(wchg_abs|wchg_rel|n_wchg_big)|^fd_own_a[0-3]|^fd_n_a[0-3]|^fd_own_apprentice"),
    "mem_conn": ("mem_conn", r".*"),
    "mem_style": ("mem_style", r".*"),
    "course_geo": ("course_geo", r".*"),                  # 旧: 内外の候補が複数あるときはマスタの先頭の行
    "course_geo_avg": ("course_geo_avg", r".*"),          # 内外の数値を平均（従来の扱いに近い）
    "course_geo_var": ("course_geo_var", r".*"),          # 内回り/外回りを区別（決まらないレースは平均＋フラグ）
    "course_draw_var": ("course_draw_var", r".*"),        # コース別の枠・脚質の有利不利を内外別に
    "course_draw": ("course_draw", r".*"),
    "course_draw_post": ("course_draw", r"^fdr_(course|br|hn|third|cond_third|fs_third)_"),
    "course_draw_style": ("course_draw", r"^fdr_(style|style_third)_"),
    "recent_bias": ("recent_bias", r".*"),
    "recent_bias_same": ("recent_bias", r"^frb_same"),
    "calendar_env": ("calendar_env", r".*"),
    "going_time": ("calendar_env", r"^fe_(start|round|going|weather|prev_day)"),
    "race_type": ("race_type", r".*"),
    "race_same_name": ("race_type", r"^frn_"),
    "race_pace_baseline": ("race_type", r"^frp_"),
    "cross_misc": ("cross_misc", r".*"),
}


def _load_threads() -> int:
    return int(os.environ.get("FAC_THREADS", "3"))


def _wait_load(max_wait: int = 4 * 3600) -> None:
    limit = float(os.environ.get("FAC_LOAD_LIMIT", "15"))
    t0 = time.time()
    while os.getloadavg()[0] >= limit and time.time() - t0 < max_wait:
        time.sleep(30)


def _prep():
    from src.pipeline.models import ablation as ab

    adopted = json.loads((OLD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base_cols = list(adopted["b0_features"]) + list(adopted["features"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    return df, base_cols


def build() -> None:
    from src.pipeline.features import factory_field as ff

    df, _ = _prep()
    OUT.mkdir(parents=True, exist_ok=True)
    pace = ff.load_pace_table()
    info = {}
    t0 = time.time()
    only = [x for x in os.environ.get("FAC_GROUPS", "").split(",") if x]
    for g in (only or ff.GROUPS):
        f = ff.build_group(g, df, **({"pace_table": pace} if g == "race_type" else {}))
        f = f.replace([np.inf, -np.inf], np.nan)
        keep = [c for c in f.columns if f[c].notna().mean() > 0.01 and f[c].nunique(dropna=True) > 1]
        f = f[keep]
        pd.concat([df[KEY], f], axis=1).to_parquet(OUT / f"{g}.parquet", index=False)
        info[g] = {"columns": len(keep), "dropped_constant_or_empty": int(len(f.columns) - len(keep)) if False else None}
        print(f"[{time.time() - t0:.0f}s] {g}: {len(keep)} 列", flush=True)
    bp = OUT / "build_info.json"
    prev = json.loads(bp.read_text(encoding="utf-8"))["groups"] if bp.exists() else {}
    prev.update({g: v["columns"] for g, v in info.items()})
    bp.write_text(json.dumps({"groups": prev, "rows": len(df)}, ensure_ascii=False, indent=1), encoding="utf-8")


def _load_candidates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cands = {}
    full = {}
    for g in {v[0] for v in SUBGROUPS.values()}:
        t = pd.read_parquet(OUT / f"{g}.parquet")
        t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
        assert (t[KEY].to_numpy() == df[KEY].to_numpy()).all()
        full[g] = t.drop(columns=KEY)
    for name, (g, pat) in SUBGROUPS.items():
        cols = [c for c in full[g].columns if re.search(pat, c)]
        if cols:
            cands[name] = full[g][cols]
    return cands


def _runner():
    from src.pipeline.models import zoo, zoo_runner
    from src.pipeline.models.time_split import yearly_expanding_folds

    best = json.loads((OLD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    return zoo, zoo_runner, yearly_expanding_folds, zoo.tuned_params("lgb", best["lgb"]["params"])


def _run(name, X, meta, folds, seeds, threads, zoo_runner, lp):
    _wait_load()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, seeds, threads, params=dict(lp))
    return res, oof


def _cmp(fb, oof_b, oof_n, fin_map, name):
    c = fb.compare_oof(oof_b, oof_n, fin_map, name)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)) for k, v in c.items()}


def stage1() -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base_cols = _prep()
    cands = _load_candidates(df)
    zoo, zoo_runner, yef, lp = _runner()
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    fin_map = meta.set_index(KEY)["finish_position"]
    folds = list(yef(df))[:2]
    seeds, threads = [int(x) for x in os.environ.get('FAC_S1_SEEDS', '0').split(',')], _load_threads()
    tag = os.environ.get("FAC_TAG", "")
    path = OUT / f"stage1{tag}.json"
    only = [x for x in os.environ.get("FAC_ONLY", "").split(",") if x]
    if only:
        cands = {k: v for k, v in cands.items() if k in only}
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    t0 = time.time()
    res_b, oof_b = _run("base", df[base_cols], meta, folds, seeds, threads, zoo_runner, lp)
    done["_base"] = {"win": res_b["logloss_win"].tolist(), "top2": float(res_b["logloss_top2"].mean()), "top3": float(res_b["logloss_top3"].mean())}
    print(f"[{time.time() - t0:.0f}s] base win {res_b['logloss_win'].mean():.5f}", flush=True)
    for name, X in cands.items():
        if name in done:
            continue
        res, oof = _run(name, pd.concat([df[base_cols], X], axis=1), meta, folds, seeds, threads, zoo_runner, lp)
        c = _cmp(fb, oof_b, oof, fin_map, name)
        per_fold = (res["logloss_win"].to_numpy() - res_b["logloss_win"].to_numpy()).tolist()
        c.update({"n_cols": int(X.shape[1]), "per_fold_dwin": per_fold, "pass": bool(c["Δwin"] < -1e-4 and max(per_fold) <= 0.0)})
        done[name] = c
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{time.time() - t0:.0f}s] {name} ({X.shape[1]}列) Δwin {c['Δwin']:+.5f} hi {c['win_hi']:+.5f} folds {[round(x, 5) for x in per_fold]} pass={c['pass']}", flush=True)


def _read_stage1() -> dict:
    out: dict = {}
    for f in sorted(OUT.glob("stage1*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d.pop("_base", None)
        out.update(d)
    return out


def stage2() -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base_cols = _prep()
    cands = _load_candidates(df)
    s1 = _read_stage1()
    passed = [n for n, v in s1.items() if not n.startswith("_") and v.get("pass")]
    print("stage1 通過群:", passed, flush=True)
    zoo, zoo_runner, yef, lp = _runner()
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    fin_map = meta.set_index(KEY)["finish_position"]
    folds = list(yef(df))[2:3]
    seeds, threads = [2, 3], _load_threads()
    only2 = [x for x in os.environ.get("FAC_ONLY", "").split(",") if x]
    if only2:
        passed = [n for n in passed if n in only2]
    path = OUT / f"stage2{os.environ.get('FAC_TAG', '')}.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    t0 = time.time()
    res_b, oof_b = _run("base", df[base_cols], meta, folds, seeds, threads, zoo_runner, lp)
    done["_base"] = {"win": res_b["logloss_win"].tolist()}
    sets = {n: cands[n] for n in passed}
    if len(passed) >= 2:
        sets["ALL_PASSED"] = pd.concat([cands[n] for n in passed], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    for name, X in sets.items():
        if name in done:
            continue
        res, oof = _run(name, pd.concat([df[base_cols], X], axis=1), meta, folds, seeds, threads, zoo_runner, lp)
        c = _cmp(fb, oof_b, oof, fin_map, name)
        c["n_cols"] = int(X.shape[1])
        done[name] = c
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{time.time() - t0:.0f}s] {name} ({X.shape[1]}列) Δwin {c['Δwin']:+.5f} hi {c['win_hi']:+.5f} Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}", flush=True)


def report() -> None:
    s1 = _read_stage1()
    s2 = {}
    for f in sorted(OUT.glob("stage2*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        s2.update({k: v for k, v in d.items() if not (k == "_base" and k in s2)})
    info = json.loads((OUT / "build_info.json").read_text(encoding="utf-8"))
    groups = [k for k in s1 if not k.startswith("_")]
    rep = {"candidates_total": int(sum(info["groups"].values())), "groups_built": info["groups"], "subgroups_tested": len(groups),
           "stage1_passed": [g for g in groups if s1[g].get("pass")], "stage1": s1, "stage2": s2,
           "adopted": [g for g, v in s2.items() if not g.startswith("_") and v.get("採用")]}
    Path("/tmp/fac_field_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# G3 レース・メンバー構成・コース/環境 ファクトリー結果", "", f"候補列 {rep['candidates_total']}、試した群 {len(groups)}（stage1: valid 2022+2023）", "",
             "| 群 | 列数 | Δwin(22+23) | 上限 | 通過 | Δwin(2024) | 上限 | Δtop2 | Δtop3 | 採用 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        a = s1[g]
        b = s2.get(g, {})
        lines.append(f"| {g} | {a['n_cols']} | {a['Δwin']:+.5f} | {a['win_hi']:+.5f} | {'○' if a['pass'] else '×'} | "
                     + (f"{b['Δwin']:+.5f} | {b['win_hi']:+.5f} | {b['Δtop2']:+.5f} | {b['Δtop3']:+.5f} | {'採用' if b['採用'] else '不採用'} |" if b else "- | - | - | - | - |"))
    if "ALL_PASSED" in s2:
        b = s2["ALL_PASSED"]
        lines.append(f"| ALL_PASSED(同時) | {b['n_cols']} | - | - | - | {b['Δwin']:+.5f} | {b['win_hi']:+.5f} | {b['Δtop2']:+.5f} | {b['Δtop3']:+.5f} | {'採用' if b['採用'] else '不採用'} |")
    Path("/tmp/fac_field_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"build": build, "stage1": stage1, "stage2": stage2, "report": report}[sys.argv[1]]()
