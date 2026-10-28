"""tp-28: 追い切りパターン別の成績表（勝率・複勝率・着順残差・回収率）の特徴量を作り、現行 v3 相当の基準に対する増分を測る。

python -m src.scripts.data.tp28_eval build     # 群ごとの特徴量を data/local/modeling/training_pipeline/tp28/<群>.parquet に保存（+ 払戻の解釈率）
python -m src.scripts.data.tp28_eval stage1    # B3 + 各群。valid 2022+2023、seed 0（粗いスクリーニング）
python -m src.scripts.data.tp28_eval stage2    # 通過群を valid 2022〜2024、seed 2・3 で確認 + 採否
python -m src.scripts.data.tp28_eval combo     # 通過群/採用群の同時追加（seed 2・3、さらに新しい seed 4・5 で確認）
python -m src.scripts.data.tp28_eval corr      # ROI 系の特徴量が市場情報（確定オッズの水準）と相関していないかの確認
python -m src.scripts.data.tp28_eval report    # /tmp/tp28_report.{json,md}

基準 B3 = 修正後の学習表の tp-03 採用 175 列 + 現行前段 3 列 + TL_A（tl_*）。LightGBM 固定、valid 2022・2023・2024 expanding。
test 2025・2026 年は使わない。市場オッズ・人気は特徴量に使わない（ROI 系は過去の確定払戻の統計）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.factory_self_eval import _by_year, _load_json, _run, _save_json, _setup, _wait_for_load

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
OUT = WORK / "tp28"
TABLES = ROOT / "data/page_reference/tables"
KEY = ["race_id", "horse_id"]
STAGE1_MARGIN = -0.0001
S3 = ["s_time_pred", "s_time_gap", "s_rank_pred"]
GROUPS = ["wp_f", "wp_w", "wp_fw", "roi", "horse_pat", "trainer_pat", "diff"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_returns(df: pd.DataFrame) -> pd.DataFrame:
    """学習表の各行に、100 円あたりの単勝/複勝の払戻（/100）を付ける。解釈できないレースは NaN。"""
    import polars as pl

    from src.pipeline.features import tp28_workout_stats as T

    years = sorted({int(y) for y in df["year"].unique()})
    frames = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            frames.append(pl.read_parquet(f, columns=["race_id", "horse_number", "finish_position", "payoff"]).to_pandas())
    r = pd.concat(frames, ignore_index=True)
    r["race_id"] = r["race_id"].astype(str)
    r["horse_number"] = pd.to_numeric(r["horse_number"], errors="coerce")
    r["finish_position"] = pd.to_numeric(r["finish_position"], errors="coerce")
    payoff_by_race = r.dropna(subset=["payoff"]).drop_duplicates("race_id").set_index("race_id")["payoff"].to_dict()
    rr = T.runner_returns(r, payoff_by_race)
    r = pd.concat([r[["race_id", "horse_number"]], rr], axis=1).drop_duplicates(["race_id", "horse_number"])
    m = df[["race_id", "horse_number"]].astype({"race_id": str}).merge(r, on=["race_id", "horse_number"], how="left")
    m.index = df.index
    return m[["win_ret", "place_ret"]]


def cmd_build() -> None:
    from src.pipeline.features import tp28_workout_stats as T

    OUT.mkdir(parents=True, exist_ok=True)
    df, base, meta, folds, lp, threads = _setup()
    laps = pd.read_parquet(WORK / "features/training_laps.parquet")
    ret = _load_returns(df)
    meta_rep = {"rows": int(len(df)), "win_ret_parsed": float(ret["win_ret"].notna().mean()), "place_ret_parsed": float(ret["place_ret"].notna().mean()),
                "win_roi_mean": float(ret["win_ret"].mean()), "place_roi_mean": float(ret["place_ret"].mean())}
    _log(f"払戻の解釈: {meta_rep}")
    runs = df[["race_id", "horse_id", "date", "trainer_id", "finish_position", "field_size"]].copy()
    feats = T.build_features(runs, laps, ret)
    cover = {}
    for g, f in feats.items():
        assert (f["race_id"].to_numpy() == df["race_id"].to_numpy()).all() and (f["horse_id"].to_numpy() == df["horse_id"].to_numpy()).all()
        f.to_parquet(OUT / f"{g}.parquet", index=False)
        cover[g] = {"cols": int(f.shape[1] - 2), "non_null_share": float(f.drop(columns=KEY).notna().mean().mean())}
        _log(f"{g}: {cover[g]}")
    meta_rep["groups"] = cover
    (OUT / "build_meta.json").write_text(json.dumps(meta_rep, ensure_ascii=False, indent=1), encoding="utf-8")


def _base_frame(df: pd.DataFrame, base: list[str]) -> pd.DataFrame:
    s3 = pd.read_parquet(WORK / "oof/stage_features_expanding.parquet")[["race_id", "horse_id"] + S3]
    s3 = df[KEY].merge(s3.assign(race_id=s3["race_id"].astype(str), horse_id=s3["horse_id"].astype(str)), on=KEY, how="left")
    s3.index = df.index
    tla = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    tla = df[KEY].merge(tla.assign(race_id=tla["race_id"].astype(str), horse_id=tla["horse_id"].astype(str)), on=KEY, how="left")
    tla.index = df.index
    tcols = [c for c in tla.columns if c not in KEY]
    TL = tla[tcols].add_prefix("tl_")
    return pd.concat([df[base], s3[S3], TL], axis=1)


def _group_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    g = pd.read_parquet(OUT / f"{name}.parquet")
    m = df[KEY].merge(g, on=KEY, how="left")
    assert len(m) == len(df)
    m = m.drop(columns=KEY)
    m.index = df.index
    return m


def stage(which: str) -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base, meta, folds, lp, threads = _setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = _base_frame(df, base)
    s1p, s2p = OUT / "stage1.json", OUT / "stage2.json"
    s1, s2 = _load_json(s1p), _load_json(s2p)
    if which == "stage1":
        use_folds, seeds, store, oofp, groups = folds[:2], [0], s1, OUT / "oof_stage1", list(GROUPS)
    else:
        use_folds, seeds, store, oofp = folds, [2, 3], s2, OUT / "oof_stage2"
        groups = [g for g in GROUPS if s1.get(g, {}).get("pass")]
    oofp.mkdir(exist_ok=True)
    t0 = time.time()
    if (oofp / "B3.parquet").exists():
        oof0, res0 = pd.read_parquet(oofp / "B3.parquet"), _load_json(oofp / "B3.json")
    else:
        _wait_for_load()
        r0, oof0 = _run("B3", X0, meta, use_folds, seeds, threads, lp)
        oof0.to_parquet(oofp / "B3.parquet", index=False)
        res0 = _by_year(r0)
        _save_json(oofp / "B3.json", res0)
    print(f"B3 {which}: " + " ".join(f"{k} {v['logloss_win']:.5f}" for k, v in res0.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    for g in groups:
        if g in store and "win_diff" in store[g]:
            continue
        _wait_for_load()
        G = _group_frame(df, g)
        r, oof = _run(f"B3+{g}", pd.concat([X0, G], axis=1), meta, use_folds, seeds, threads, lp)
        ry = _by_year(r)
        c = fb.compare_oof(oof0, oof, fin_map, g)
        per = {k: ry[k]["logloss_win"] - res0[k]["logloss_win"] for k in ry if k in res0}
        rec = {"cols": int(G.shape[1]), "win_diff": c["Δwin"], "win_hi": c["win_hi"], "top2_diff": c["Δtop2"], "top3_diff": c["Δtop3"], "per_year_win_diff": per, "levels": ry}
        if which == "stage1":
            rec["pass"] = bool(c["Δwin"] < STAGE1_MARGIN and all(v <= 0 for v in per.values()))
        else:
            rec["adopt"] = bool(c["採用"])
        store[g] = rec
        _save_json(s1p if which == "stage1" else s2p, store)
        oof.to_parquet(oofp / f"{g}.parquet", index=False)
        print(f"{g:12s} cols {G.shape[1]:3d} Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} per-year {({k: round(v, 5) for k, v in per.items()})} "
              f"{'PASS' if rec.get('pass') or rec.get('adopt') else '-'} ({time.time() - t0:.0f}s)", flush=True)


def combo() -> None:
    from src.pipeline.models import feature_bakeoff as fb

    df, base, meta, folds, lp, threads = _setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    s1, s2 = _load_json(OUT / "stage1.json"), _load_json(OUT / "stage2.json")
    sets = {"combo_all": list(GROUPS), "combo_stage1_pass": [g for g, r in s1.items() if r.get("pass")], "combo_adopted": [g for g, r in s2.items() if r.get("adopt")],
            "combo_no_roi": [g for g in GROUPS if g != "roi"]}
    res = _load_json(OUT / "combo.json")
    X0 = _base_frame(df, base)
    for seeds_name, seeds in (("s23", [2, 3]), ("s45", [4, 5])):
        _wait_for_load()
        r0, oof0 = _run(f"B3_{seeds_name}", X0, meta, folds, seeds, threads, lp)
        done = set()
        for nm, gs in sets.items():
            key = tuple(gs)
            if not gs or key in done or f"{nm}_{seeds_name}" in res:
                continue
            done.add(key)
            _wait_for_load()
            X = pd.concat([X0] + [_group_frame(df, g) for g in gs], axis=1)
            r, oof = _run(f"{nm}_{seeds_name}", X, meta, folds, seeds, threads, lp)
            c = fb.compare_oof(oof0, oof, fin_map, nm)
            res[f"{nm}_{seeds_name}"] = {"groups": gs, "cols": int(X.shape[1] - X0.shape[1]), "win_diff": c["Δwin"], "win_hi": c["win_hi"], "top2_diff": c["Δtop2"], "top3_diff": c["Δtop3"],
                                        "adopt": bool(c["採用"]), "levels_base": _by_year(r0), "levels_new": _by_year(r)}
            _save_json(OUT / "combo.json", res)
            print(f"{nm} {seeds_name} {len(gs)}groups Δwin {c['Δwin']:+.5f} hi {c['win_hi']:+.5f} Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}", "ADOPT" if c["採用"] else "-", flush=True)


def corr() -> None:
    """ROI 系の特徴量が、確定オッズの水準（対数）とどれくらい相関するか。レース内の順位相関と全体の相関。市場情報を持ち込んでいないかの確認（オッズは評価にだけ使う）。"""
    import polars as pl

    df, *_ = _setup()
    years = sorted({int(y) for y in df["year"].unique()})
    odds = pd.concat([pl.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "odds"]).to_pandas() for y in years if (TABLES / str(y) / "race_result_flat.parquet").exists()])
    odds["race_id"] = odds["race_id"].astype(str)
    odds["horse_number"] = pd.to_numeric(odds["horse_number"], errors="coerce")
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    from src.scraper.odds_fix import apply_pandas as _fix_odds

    odds = _fix_odds(odds)   # 2025 年は odds が指数列だった（scraper.odds_fix）
    m = df[["race_id", "horse_number", "year"]].astype({"race_id": str}).merge(odds.drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left")
    lo = np.log(m["odds"].clip(lower=1.0))
    rows = {}
    for g in ("roi", "wp_f", "diff"):
        G = _group_frame(df, g)
        for c in G.columns:
            x = G[c].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(lo.to_numpy())
            if ok.sum() < 1000:
                continue
            rho = float(np.corrcoef(pd.Series(x[ok]).rank().to_numpy(), pd.Series(lo.to_numpy()[ok]).rank().to_numpy())[0, 1])
            within = pd.DataFrame({"r": m["race_id"].to_numpy()[ok], "x": x[ok], "o": lo.to_numpy()[ok]})
            wx = within.groupby("r")["x"].rank(pct=True)
            wo = within.groupby("r")["o"].rank(pct=True)
            rows[f"{g}.{c}"] = {"spearman_all": rho, "spearman_within_race": float(np.corrcoef(wx, wo)[0, 1])}
    _save_json(OUT / "odds_corr.json", rows)
    top = sorted(rows.items(), key=lambda kv: -abs(kv[1]["spearman_within_race"]))[:8]
    for k, v in top:
        print(f"{k:28s} 全体 {v['spearman_all']:+.3f} レース内 {v['spearman_within_race']:+.3f}")


def report() -> None:
    s1, s2, cb = _load_json(OUT / "stage1.json"), _load_json(OUT / "stage2.json"), _load_json(OUT / "combo.json")
    bm, oc = _load_json(OUT / "build_meta.json"), _load_json(OUT / "odds_corr.json")
    (Path("/tmp") / "tp28_report.json").write_text(json.dumps({"build": bm, "stage1": s1, "stage2": s2, "combo": cb, "odds_corr": oc}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = ["# tp-28 追い切りパターン別の成績表", "", f"払戻の解釈率: 単勝 {bm.get('win_ret_parsed', float('nan')):.3f} / 複勝 {bm.get('place_ret_parsed', float('nan')):.3f}（解釈できないレースは ROI 系の集計から外す）", "",
             "| 群 | 列数 | stage1 Δwin | 通過 | stage2 Δwin | 上限 | Δtop2 | Δtop3 | 採用 |", "|---|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        a, b = s1.get(g, {}), s2.get(g, {})
        lines.append(f"| {g} | {a.get('cols', '')} | {a.get('win_diff', float('nan')):+.5f} | {'○' if a.get('pass') else '×'} | {b.get('win_diff', float('nan')):+.5f} | {b.get('win_hi', float('nan')):+.5f} | "
                     f"{b.get('top2_diff', float('nan')):+.5f} | {b.get('top3_diff', float('nan')):+.5f} | {'採用' if b.get('adopt') else ('不採用' if b else '-')} |")
    lines += ["", "## 同時追加", "", "| 構成 | seed | 列数 | Δwin | 上限 | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|---|---|"]
    for k, v in cb.items():
        lines.append(f"| {k.rsplit('_', 1)[0]} | {k.rsplit('_', 1)[1]} | {v['cols']} | {v['win_diff']:+.5f} | {v['win_hi']:+.5f} | {v['top2_diff']:+.5f} | {v['top3_diff']:+.5f} | {'採用' if v['adopt'] else '不採用'} |")
    top = sorted(oc.items(), key=lambda kv: -abs(kv[1]["spearman_within_race"]))[:6]
    lines += ["", "## 市場情報の混入度（確定オッズ水準との順位相関。上位）", "", "| 特徴量 | 全体 | レース内 |", "|---|---|---|"] + [f"| {k} | {v['spearman_all']:+.3f} | {v['spearman_within_race']:+.3f} |" for k, v in top]
    (Path("/tmp") / "tp28_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"build": cmd_build, "stage1": lambda: stage("stage1"), "stage2": lambda: stage("stage2"), "combo": combo, "corr": corr, "report": report}[sys.argv[1]]()
