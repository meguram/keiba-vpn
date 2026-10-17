"""特徴量ファクトリー G1（馬自身の履歴・状態）の生成と評価。

python -m src.scripts.data.factory_self_eval attrs     # 馬の固定属性（生年月日・産地・毛色・馬主・生産者）を horse_result から集める
python -m src.scripts.data.factory_self_eval build     # 全群を作って data/local/modeling/training_pipeline/factory/self/<群>.parquet に保存
python -m src.scripts.data.factory_self_eval stage1    # V0 + 各群。valid 2022+2023、seed 0（粗いスクリーニング）
python -m src.scripts.data.factory_self_eval stage2    # 通過した群を valid 2022〜2024、新しい seed 2・3 で確認 + 採否
python -m src.scripts.data.factory_self_eval combo     # 採用群（なければ通過群）の同時追加
python -m src.scripts.data.factory_self_eval report    # /tmp/fac_self_report.{json,md}

基準 V0: 修正後の学習表の tp-03 採用 175 列、LightGBM 固定、valid 2022・2023・2024 expanding。test 2025・2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
OUT = WORK / "factory/self"
KEY = ["race_id", "horse_id"]
TABLES = ROOT / "data/page_reference/tables"
STAGE1_MARGIN = -0.0001


def attrs() -> None:
    from src.scraper.storage import HybridStorage

    OUT.mkdir(parents=True, exist_ok=True)
    ids = set()
    for y in range(2020, 2026):
        import polars as pl

        ids |= set(pl.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=["horse_id"])["horse_id"].to_list())
    ids = sorted(i for i in ids if i)
    st = HybridStorage()

    def one(h):
        try:
            d = st.load("horse_result", h) or {}
        except Exception:
            d = {}
        return {"horse_id": h, "birthday": d.get("birthday"), "birthplace": d.get("birthplace"), "color": d.get("color"), "owner": d.get("owner"), "breeder": d.get("breeder")}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, ids))
    df = pd.DataFrame(rows)
    ymd = df["birthday"].astype(str).str.extract(r"(\d{4})年(\d{1,2})月(\d{1,2})日").astype(float)
    ymd.columns = ["year", "month", "day"]
    df["birthday"] = pd.to_datetime(ymd, errors="coerce")
    df.to_parquet(OUT / "horse_attrs.parquet", index=False)
    print(f"馬 {len(df)} 頭 / 生年月日あり {df['birthday'].notna().mean():.3f} / 産地あり {df['birthplace'].notna().mean():.3f} / 毛色あり {df['color'].notna().mean():.3f} ({time.time() - t0:.0f}s)", flush=True)


def _train_keys() -> pd.DataFrame:
    import polars as pl

    return pl.read_parquet(FIXED / "features/train_table.parquet", columns=KEY).to_pandas().astype(str)


def build() -> None:
    from src.pipeline.features import factory_self as fs

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runs = fs.load_runs(TABLES, list(range(2020, 2026)))
    print(f"[{time.time() - t0:.0f}s] runs {len(runs)}", flush=True)
    at = pd.read_parquet(OUT / "horse_attrs.parquet").set_index("horse_id") if (OUT / "horse_attrs.parquet").exists() else None
    bday = at["birthday"] if at is not None else None
    keys = _train_keys()
    meta = {}
    for name in fs.GROUPS:
        if os.environ.get("FAC_SKIP_EXISTING") == "1" and (OUT / f"{name}.parquet").exists():
            continue
        t1 = time.time()
        g = fs.build_groups(runs, birthday=bday, attrs=at, only=[name]).get(name)
        if g is None:
            continue
        g = keys.merge(g, on=KEY, how="left")
        feat = [c for c in g.columns if c not in KEY]
        nn = g[feat].notna().mean()
        keep = [c for c in feat if nn[c] > 0.02 and g[c].nunique(dropna=True) > 1]
        g[KEY + keep].to_parquet(OUT / f"{name}.parquet", index=False)
        meta[name] = {"cols": len(keep), "dropped_const_or_empty": len(feat) - len(keep), "mean_coverage": float(nn[keep].mean()) if keep else 0.0}
        print(f"[{time.time() - t0:.0f}s] {name}: {len(keep)} 列 (除外 {len(feat) - len(keep)}) coverage {meta[name]['mean_coverage']:.2f} ({time.time() - t1:.0f}s)", flush=True)
    (OUT / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────── 評価
def _setup():
    from src.pipeline.models import ablation as ab, zoo
    from src.pipeline.models.compute import detect_compute
    from src.pipeline.models.time_split import yearly_expanding_folds

    threads = int(os.environ.get("FAC_THREADS", min(3, detect_compute(reserve_cores=6).threads)))
    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((WORK / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    meta = df[["race_id", "horse_id", "finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    return df, base, meta, folds, lp, threads


def _group_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    g = pd.read_parquet(OUT / f"{name}.parquet")
    m = df[KEY].merge(g, on=KEY, how="left")
    assert len(m) == len(df)
    m = m.drop(columns=KEY)
    m.index = df.index
    return m


def _run(name, X, meta, folds, seeds, threads, lp):
    from src.pipeline.models import zoo_runner

    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, seeds, threads, params=dict(lp))
    return res, oof


def _by_year(res: pd.DataFrame) -> dict:
    return {str(r["fold"]): {k: float(r[k]) for k in ("logloss_win", "logloss_top2", "logloss_top3")} for _, r in res.iterrows()}


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_json(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _wait_for_load(limit: float | None = None, max_wait: int = 1800) -> None:
    limit = limit or float(os.environ.get("FAC_LOAD_LIMIT", 17.5))
    t0 = time.time()
    while os.getloadavg()[0] >= limit and time.time() - t0 < max_wait:
        time.sleep(30)


def stage(which: str) -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base, meta, folds, lp, threads = _setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    groups = sorted(p.stem for p in OUT.glob("*.parquet") if p.stem != "horse_attrs")
    s1p, s2p = OUT / "stage1.json", OUT / "stage2.json"
    s1, s2 = _load_json(s1p), _load_json(s2p)
    if which == "stage1":
        use_folds, seeds, store, oofp = folds[:2], [0], s1, OUT / "oof_stage1"
    else:
        use_folds, seeds, store, oofp = folds, [2, 3], s2, OUT / "oof_stage2"
        groups = [g for g in groups if s1.get(g, {}).get("pass")]
    oofp.mkdir(exist_ok=True)
    X0 = df[base]
    t0 = time.time()
    if (oofp / "V0.parquet").exists():
        oof0 = pd.read_parquet(oofp / "V0.parquet")
        res0 = _load_json(oofp / "V0.json")
    else:
        _wait_for_load()
        r0, oof0 = _run("V0", X0, meta, use_folds, seeds, threads, lp)
        oof0.to_parquet(oofp / "V0.parquet", index=False)
        res0 = _by_year(r0)
        _save_json(oofp / "V0.json", res0)
    print(f"V0 {which}: " + " ".join(f"{k} {v['logloss_win']:.5f}" for k, v in res0.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    for g in groups:
        if g in store and "win_diff" in store[g]:
            continue
        _wait_for_load()
        G = _group_frame(df, g)
        X = pd.concat([X0, G], axis=1)
        r, oof = _run(f"V0+{g}", X, meta, use_folds, seeds, threads, lp)
        ry = _by_year(r)
        c = fb.compare_oof(oof0, oof, fin_map, g)
        per = {k: ry[k]["logloss_win"] - res0[k]["logloss_win"] for k in ry if k in res0}
        rec = {"cols": int(G.shape[1]), "win_diff": c["Δwin"], "win_hi": c["win_hi"], "top2_diff": c["Δtop2"], "top3_diff": c["Δtop3"], "per_year_win_diff": per,
               "levels": ry}
        if which == "stage1":
            rec["pass"] = bool(c["Δwin"] < STAGE1_MARGIN and all(v <= 0 for v in per.values()))
        else:
            rec["adopt"] = bool(c["採用"])
            rec["adopt_3years_not_worse"] = bool(all(v <= 0.00005 for v in per.values()))
        store[g] = rec
        _save_json(s1p if which == "stage1" else s2p, store)
        oof.to_parquet(oofp / f"{g}.parquet", index=False)
        print(f"{g:14s} cols {G.shape[1]:3d} Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} per-year {({k: round(v, 5) for k, v in per.items()})} "
              f"{'PASS' if rec.get('pass') or rec.get('adopt') else '-'} ({time.time() - t0:.0f}s)", flush=True)


def combo() -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base, meta, folds, lp, threads = _setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    s1, s2 = _load_json(OUT / "stage1.json"), _load_json(OUT / "stage2.json")
    adopted = [g for g, r in s2.items() if r.get("adopt")]
    passed = [g for g, r in s2.items() if r.get("adopt_3years_not_worse") and r.get("win_diff", 0) < -0.0001]
    sets = {"combo_adopted": adopted, "combo_stage2_promising": passed, "combo_stage1_pass": [g for g, r in s1.items() if r.get("pass") and not g.startswith("EXCLUDED")]}
    oof0 = pd.read_parquet(OUT / "oof_stage2/V0.parquet")
    res = _load_json(OUT / "combo.json")
    for nm, gs in sets.items():
        if not gs or nm in res or (nm != "combo_adopted" and sets[nm] == sets["combo_adopted"]):
            continue
        _wait_for_load()
        X = pd.concat([df[base]] + [_group_frame(df, g) for g in gs], axis=1)
        r, oof = _run(nm, X, meta, folds, [2, 3], threads, lp)
        c = fb.compare_oof(oof0, oof, fin_map, nm)
        res[nm] = {"groups": gs, "cols": int(X.shape[1] - len(base)), "win_diff": c["Δwin"], "win_hi": c["win_hi"], "top2_diff": c["Δtop2"], "top3_diff": c["Δtop3"], "adopt": bool(c["採用"]), "levels": _by_year(r)}
        _save_json(OUT / "combo.json", res)
        print(nm, len(gs), "groups", f"Δwin {c['Δwin']:+.5f} hi {c['win_hi']:+.5f} Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}", "ADOPT" if c["採用"] else "-", flush=True)


def confirm() -> None:
    """combo_stage2_promising を、選択に使っていない seed（4・5）で V0 と比べ直す（データの fold は同じ。test 2025 は使わない）。"""
    from src.pipeline.models import feature_bakeoff as fb

    df, base, meta, folds, lp, threads = _setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    cb = _load_json(OUT / "combo.json")
    cname = sys.argv[2] if len(sys.argv) > 2 else "combo_stage1_pass"
    gs = cb[cname]["groups"]
    res = _load_json(OUT / "confirm.json")
    _wait_for_load()
    r0, oof0 = _run("V0_s45", df[base], meta, folds, [4, 5], threads, lp)
    X = pd.concat([df[base]] + [_group_frame(df, g) for g in gs], axis=1)
    _wait_for_load()
    r, oof = _run("combo_s45", X, meta, folds, [4, 5], threads, lp)
    c = fb.compare_oof(oof0, oof, fin_map, "combo_s45")
    res[f"{cname}_seeds45"] = {"groups": gs, "win_diff": c["Δwin"], "win_hi": c["win_hi"], "top2_diff": c["Δtop2"], "top3_diff": c["Δtop3"], "adopt": bool(c["採用"]),
                                              "levels_v0": _by_year(r0), "levels_new": _by_year(r)}
    _save_json(OUT / "confirm.json", res)
    print("confirm seeds 4,5:", gs, f"Δwin {c['Δwin']:+.5f} hi {c['win_hi']:+.5f} Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}", "ADOPT" if c["採用"] else "-", flush=True)


def report() -> None:
    s1, s2, cb = _load_json(OUT / "stage1.json"), _load_json(OUT / "stage2.json"), _load_json(OUT / "combo.json")
    bm = _load_json(OUT / "build_meta.json")
    cf = _load_json(OUT / "confirm.json")
    n_g = len([g for g in s1 if not g.startswith("EXCLUDED")])
    rep = {"tested_groups": n_g, "tested_groups_incl_excluded_variant": len(s1), "tested_columns": int(sum(v.get("cols", 0) for g, v in s1.items() if not g.startswith("EXCLUDED"))),
           "stage1": s1, "stage2": s2, "combo": cb, "confirm": cf, "build": bm}
    Path("/tmp/fac_self_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    L = [f"# 特徴量ファクトリー G1 結果\n", f"試した群 {n_g}（+ 毛色入りの産地群 1 件は除外）/ 列 {rep['tested_columns']}（stage1 の対象。stage2 は通過群のみ）。同時追加 {len(cb)} 組、新 seed 確認 {len(cf)} 組\n",
         "| 群 | 列 | stage1 Δwin | 2022 | 2023 | 通過 | stage2 Δwin (CI上限) | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, r in sorted([kv for kv in s1.items() if not kv[0].startswith("EXCLUDED")], key=lambda kv: kv[1].get("win_diff", 0)):
        py = r.get("per_year_win_diff", {})
        r2 = s2.get(g)
        L.append(f"| {g} | {r['cols']} | {r['win_diff']:+.5f} | {py.get('2022', float('nan')):+.5f} | {py.get('2023', float('nan')):+.5f} | {'○' if r.get('pass') else '-'} | "
                 + (f"{r2['win_diff']:+.5f} ({r2['win_hi']:+.5f}) | {r2['top2_diff']:+.5f} | {r2['top3_diff']:+.5f} | {'採用' if r2.get('adopt') else '不採用'} |" if r2 else "- | - | - | - |"))
    if cb:
        L += ["", "## 同時追加", "| 組 | 群数 | 列 | Δwin (CI上限) | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|---|"]
        for k, v in cb.items():
            L.append(f"| {k} | {len(v['groups'])} | {v['cols']} | {v['win_diff']:+.5f} ({v['win_hi']:+.5f}) | {v['top2_diff']:+.5f} | {v['top3_diff']:+.5f} | {'採用' if v['adopt'] else '不採用'} |")
    if cf:
        L += ["", "## 新しい seed（4・5）での確認（fold は同じ）", "| 組 | 群 | Δwin (CI上限) | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|"]
        for k, v in cf.items():
            L.append(f"| {k} | {', '.join(v['groups'])} | {v['win_diff']:+.5f} ({v['win_hi']:+.5f}) | {v['top2_diff']:+.5f} | {v['top3_diff']:+.5f} | {'採用' if v['adopt'] else '不採用'} |")
    ex = s1.get("EXCLUDED_attrs_with_color")
    if ex:
        L += ["", f"除外: 産地+毛色コードの群は stage1 で Δwin {ex['win_diff']:+.5f} と大きく見えたが、毛色は生まれ年によって取れている割合が違う（2023 年生まれ 46%、2019 年生まれ 4%）ため時間依存の欠損が先読み代わりに効いていた。毛色を外すと Δwin {s1['attrs']['win_diff']:+.5f}。"]
    Path("/tmp/fac_self_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"attrs": attrs, "build": build, "stage1": lambda: stage("stage1"), "stage2": lambda: stage("stage2"), "combo": combo, "confirm": confirm, "report": report}[cmd]()
