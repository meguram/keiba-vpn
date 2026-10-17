"""tp-27: v4 特徴量セットの統合（群の累積評価）と、v3 のアンサンブルでの確認。

python -m src.scripts.data.tp27_eval stages --order A      # V0+S3 から群を 1 つずつ足す（馬主→補助列→G1→G3→G4）。各段の増分と採否
python -m src.scripts.data.tp27_eval stages --order B      # 順序の感度（G1→G3→馬主→補助列）
python -m src.scripts.data.tp27_eval stagesC               # TL_A を含む土台（v3 相当）で、G4 を『静的で検証可能な群』と『未検証の SmartRC 由来』に分けて足す
python -m src.scripts.data.tp27_eval prune                 # V4 の列の絞り込み（gain 上位・相関整理）との比較
python -m src.scripts.data.tp27_eval v4table               # V4 の特徴量表を tp27/ に保存
python -m src.scripts.data.tp27_eval member NAME           # v3 のメンバー + 補助列 + V4 の列で OOF（valid 2022-24）
python -m src.scripts.data.tp27_eval mtl BASE              # 補助損失つき NN + V4 の列で OOF
python -m src.scripts.data.tp27_eval ens --name v4         # LOYO アンサンブルを v3 / e3 と比べる

基準: 修正後の学習表の tp-03 採用 175 列 + 現行の前段 3 列(V0+S3)。LightGBM 固定、valid 2022・2023・2024 expanding、2 seed。
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数: TP27_THREADS（既定 5）, TP27_LOAD_LIMIT（既定 17.5）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
FIXED = Path(os.environ.get("TP_BASE_DIR") or ROOT / "data/local/modeling/training_pipeline_fixed")      # v4 の土台で回すときは TP_BASE_DIR=training_pipeline_v4
WORK_V = Path(os.environ.get("TP_WORK") or WORK)                                                          # 前段(S3)・補助列・tp05_data の読み先（v4: TP_WORK=training_pipeline_v4）
FAC = WORK / "factory"
FAC_FIELD = Path(os.environ.get("TP27_FAC_FIELD") or FAC / "field")
AUX_DIR = Path(os.environ.get("TP25E_OUT") or WORK_V / "tp25e")
GROUPDEF = WORK / "tp27"                                                                                  # 群の定義（採用された群）は旧 tp-27 の結果を読む
OUT = Path(os.environ.get("TP27_OUT") or WORK / "tp27")
KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP27_THREADS", 5))
SEEDS = [0, 1]
NN_TOP = int(os.environ.get("TP27_NN_TOP", 100))
STAGE_COLS = ["s_time_pred", "s_time_gap", "s_rank_pred"]

GROUPS = {
    "owner": {"src": "conn", "names": ["ent_o", "win_o", "streak_o", "ent_ob"]},
    "aux": {"src": "aux", "names": ["aux_B"]},
    "self8": {"src": "self", "names": ["change", "experience", "lags_margin", "pair", "rest", "seq_stats", "timedev", "attrs"]},
    "cross_misc": {"src": "field", "names": ["cross_misc"]},
    "field7": {"src": "field", "names": ["mem_strength", "mem_strength_horse", "mem_demo", "mem_conn", "course_draw_post", "calendar_env", "race_type"]},
}
ORDERS = {"A": ["owner", "aux", "self8", "cross_misc", "field7", "src"], "B": ["self8", "cross_misc", "field7", "owner", "aux"]}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wait_for_load(limit: float | None = None, max_wait: int = 3600) -> None:
    limit = limit or float(os.environ.get("TP27_LOAD_LIMIT", 17.5))
    t0 = time.time()
    while os.getloadavg()[0] >= limit and time.time() - t0 < max_wait:
        time.sleep(30)


def _jload(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _jsave(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


# ───────────────────────── 群の読み込み
def adopted_src_groups() -> list[str]:
    """G4 の第 2 段で採用された群。複数のレポートに同じ群があるときは、更新時刻が新しいレポートを優先する（修正版の再評価が後から出るため）。"""
    latest: dict[str, bool] = {}
    for p in sorted(Path("/tmp").glob("fac_src_report*.json"), key=lambda q: q.stat().st_mtime):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for g, v in (d.get("groups") or {}).items():
            s2 = v.get("s2") or {}
            latest[g] = bool(s2.get("採用") or (v.get("adopted") and not s2))
    return sorted(g for g, ok in latest.items() if ok)


def _raw_keys() -> pd.DataFrame:
    from src.pipeline.models import ablation as ab

    raw = ab.load_table(FIXED / "features/train_table.parquet")[KEY].copy()
    raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
    return raw


def load_one(src: str, name: str, df: pd.DataFrame, raw: pd.DataFrame | None) -> pd.DataFrame:
    if src == "conn":
        G = pd.read_parquet(FAC / "conn/groups" / f"{name}.parquet")
        G[KEY[0]], G[KEY[1]] = raw["race_id"].to_numpy(), raw["horse_id"].to_numpy()
    elif src == "aux":
        G = pd.read_parquet(AUX_DIR / f"{name}.parquet")
    elif src == "src":
        G = pd.read_parquet(FAC / "src" / f"cand_{name}.parquet")
    elif src == "field":
        import re

        from src.scripts.data.factory_field_eval import SUBGROUPS

        g, pat = SUBGROUPS.get(name, (name, None))
        G = pd.read_parquet(FAC_FIELD / f"{g}.parquet")
        if pat:
            G = G[KEY + [c for c in G.columns if c not in KEY and re.search(pat, c)]]
    else:
        G = pd.read_parquet(FAC / src / f"{name}.parquet")
    G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
    m = df[KEY].merge(G.drop_duplicates(KEY), on=KEY, how="left")
    assert len(m) == len(df), (src, name, len(m), len(df))
    m = m.drop(columns=KEY)
    m.index = df.index
    return m.add_prefix("") if src != "aux" else m


def load_group(group: str, df: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    if group == "src" or group.startswith("src:"):
        names = adopted_src_groups() if group == "src" else [group.split(":", 1)[1]]
        parts = [load_one("src", n, df, raw) for n in names]
    else:
        spec = GROUPS[group]
        if spec["src"] == "conn" and raw is None:
            raw = _raw_keys()
        parts = [load_one(spec["src"], n, df, raw) for n in spec["names"]]
    if not parts:
        return pd.DataFrame(index=df.index)
    G = pd.concat(parts, axis=1)
    return G.loc[:, ~G.columns.duplicated()]


# ───────────────────────── 土台
def setup():
    from src.pipeline.models import ablation as ab, zoo
    from src.pipeline.models.time_split import yearly_expanding_folds

    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    best = json.loads((WORK / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    lp = zoo.tuned_params("lgb", best["lgb"]["params"])
    df = ab.load_table(FIXED / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    stage = pd.read_parquet(WORK_V / "oof/stage_features_expanding.parquet")[KEY + STAGE_COLS]
    stage["race_id"], stage["horse_id"] = stage["race_id"].astype(str), stage["horse_id"].astype(str)
    df = df.merge(stage, on=KEY, how="left")
    df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    meta = df[KEY + ["finish_position", "year", "date"]].copy()
    folds = list(yearly_expanding_folds(df))
    return df, base, meta, folds, lp


def _run(name, X, meta, folds, lp):
    from src.pipeline.models import zoo_runner

    _wait_for_load()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, SEEDS, THREADS, params=dict(lp))
    lv = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
    by = {str(r["fold"]): float(r["logloss_win"]) for _, r in res.iterrows()}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")
    return lv, by, oof


def _cmp(prev_oof, oof, fin_map, label):
    from src.pipeline.models import feature_bakeoff as fb

    c = fb.compare_oof(prev_oof, oof, fin_map, label)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()}


# ───────────────────────── 段階評価
def cmd_stages(a) -> None:
    df, base, meta, folds, lp = setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    raw = _raw_keys()
    out = OUT / f"stages_{a.order}"
    out.mkdir(parents=True, exist_ok=True)
    resp = out / "results.json"
    res = _jload(resp)
    X = df[base + STAGE_COLS]
    cache0 = out / "V0S3.parquet"
    if cache0.exists() and "V0S3" in res:
        oof_prev = pd.read_parquet(cache0)
    else:
        lv, by, oof_prev = _run("V0+S3", X, meta, folds, lp)
        oof_prev.to_parquet(cache0, index=False)
        res["V0S3"] = {"levels": lv, "by_fold": by, "cols": X.shape[1]}
        _jsave(resp, res)
    cur_lv = res["V0S3"]["levels"]
    kept: list[str] = []
    steps = res.setdefault("steps", [])
    done = {s["group"] for s in steps}
    order = [x for g in ORDERS[a.order] for x in ([f"src:{n}" for n in adopted_src_groups()] if g == "src" else [g])]
    for g in order:
        if g in done:
            s = next(s for s in steps if s["group"] == g)
            if s["adopted"]:
                kept.append(g)
                X = pd.concat([X, load_group(g, df, raw)], axis=1)
                oof_prev = pd.read_parquet(out / f"after_{g}.parquet")
                cur_lv = s["levels"]
            continue
        G = load_group(g, df, raw)
        if G.shape[1] == 0:
            steps.append({"group": g, "cols": 0, "adopted": False, "note": "列なし（採用された群なし）"})
            _jsave(resp, res)
            continue
        Xn = pd.concat([X, G], axis=1)
        lv, by, oof = _run(f"{'+'.join(kept + [g])}", Xn, meta, folds, lp)
        c = _cmp(oof_prev, oof, fin_map, g)
        adopt = bool(c["採用"])
        steps.append({"group": g, "cols": int(G.shape[1]), "levels": lv, "by_fold": by, "vs_prev": c, "adopted": adopt, "prev_levels": cur_lv})
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} → {'採用' if adopt else '不採用'}")
        if adopt:
            kept.append(g)
            X, oof_prev, cur_lv = Xn, oof, lv
            oof.to_parquet(out / f"after_{g}.parquet", index=False)
    res["V4_groups"] = kept
    res["V4_cols"] = int(X.shape[1])
    res["V4_levels"] = cur_lv
    res["tested_groups"] = len(order)
    _jsave(resp, res)
    _log(f"order {a.order}: V4 groups={kept} cols={X.shape[1]} win={cur_lv['logloss_win']:.5f}")


# ───────────────────────── 段階評価 C（TL_A を土台に入れて G4 を分けて評価）
STATIC_G4 = ["PED_sire_cond", "TV_birth", "SRC_hist_misc"]            # 静的・検証可能（種牡馬 ID、生年月日、tp-21 で検証済みの SmartRC 過去5走）
TRAIN_G4 = ["TRN_misc", "OK_struct"]                                  # 調教・追い切り（TL_A と重なるので TL_A を入れた土台で再検証）
SMARTRC_G4 = ["SRC_rota", "SRC_equip", "SRC_owner", "SRC_travel", "SRC_ten_agari"]   # 取得日(2026)時点の値で as-of を検証できない SmartRC 由来


def tl_frame(df: pd.DataFrame) -> pd.DataFrame:
    TL = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    TL["race_id"], TL["horse_id"] = TL["race_id"].astype(str), TL["horse_id"].astype(str)
    m = df[KEY].merge(TL, on=KEY, how="left").drop(columns=KEY)
    m.index = df.index
    return m.add_prefix("tl_")


def cmd_stages_c(a) -> None:
    df, base, meta, folds, lp = setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    raw = _raw_keys()
    out = OUT / "stages_C"
    out.mkdir(parents=True, exist_ok=True)
    resp = out / "results.json"
    res = _jload(resp)
    resA = _jload(OUT / "stages_A/results.json")
    core_prev = [g for g in resA.get("V4_groups", []) if not g.startswith("src:")] or ["owner", "aux", "self8", "cross_misc", "field7"]
    res["core_from_A"] = core_prev
    X = pd.concat([df[base + STAGE_COLS], tl_frame(df)] + [load_group(g, df, raw) for g in core_prev], axis=1)
    # 土台 C0: V0+S3+TL_A+（A で採用された G1〜G3・馬主・補助列）
    c0 = out / "C0.parquet"
    if c0.exists() and "C0" in res:
        oof_prev = pd.read_parquet(c0)
    else:
        lv, by, oof_prev = _run("C0(V0+S3+TL_A+core)", X, meta, folds, lp)
        oof_prev.to_parquet(c0, index=False)
        res["C0"] = {"levels": lv, "by_fold": by, "cols": X.shape[1]}
        _jsave(resp, res)
    cur_lv = res["C0"]["levels"]
    steps = res.setdefault("steps", [])
    done = {s["group"]: s for s in steps}
    kept: list[str] = []
    track = {}                                                          # 群 → 'static' / 'train' / 'smartrc'
    plan = [(g, "static") for g in STATIC_G4] + [(g, "train") for g in TRAIN_G4]
    X_core = X
    for g, kind in plan + [(g, "smartrc") for g in SMARTRC_G4]:
        if g == SMARTRC_G4[0]:
            X_core, oof_core, lv_core = X, oof_prev, cur_lv             # ここまでが V4 本体（SmartRC 由来を含まない）
            res["V4_core_groups"], res["V4_core_levels"], res["V4_core_cols"] = list(kept), cur_lv, int(X.shape[1])
            _jsave(resp, res)
        key = g
        if key in done:
            s = done[key]
            if s["adopted"]:
                kept.append(g)
                X = pd.concat([X, load_group(f"src:{g}", df, raw)], axis=1)
                oof_prev = pd.read_parquet(out / f"after_{g}.parquet")
                cur_lv = s["levels"]
            continue
        G = load_group(f"src:{g}", df, raw)
        Xn = pd.concat([X, G], axis=1)
        lv, by, oof = _run("C:" + "+".join(kept + [g]), Xn, meta, folds, lp)
        c = _cmp(oof_prev, oof, fin_map, g)
        adopt = bool(c["採用"])
        steps.append({"group": g, "kind": kind, "cols": int(G.shape[1]), "levels": lv, "by_fold": by, "vs_prev": c, "adopted": adopt, "prev_levels": cur_lv})
        _jsave(resp, res)
        _log(f"  {g} [{kind}]: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} → {'採用' if adopt else '不採用'}")
        if adopt:
            kept.append(g)
            X, oof_prev, cur_lv = Xn, oof, lv
            oof.to_parquet(out / f"after_{g}.parquet", index=False)
    res["V4_plus_smartrc_groups"] = list(kept)
    res["V4_plus_smartrc_levels"], res["V4_plus_smartrc_cols"] = cur_lv, int(X.shape[1])
    # 強制版: 静的な 3 群を（採否によらず）C0 に足した場合
    if "forced_static" not in res:
        Xf = pd.concat([X_core] + [load_group(f"src:{g}", df, raw) for g in STATIC_G4 if g not in res.get("V4_core_groups", [])], axis=1)
        lvf, byf, oof_f = _run("C:forced_static3", Xf, meta, folds, lp)
        res["forced_static"] = {"levels": lvf, "by_fold": byf, "cols": int(Xf.shape[1]), "vs_core": _cmp(oof_core, oof_f, fin_map, "forced_static3")}
    _jsave(resp, res)
    _log(f"C: core={res.get('V4_core_groups')} +smartrc={kept}")


def v4_groups(variant: str = "core") -> list[str]:
    """V4 の群。core = A で採用された G1〜G3・馬主・補助列 + 静的/検証可能な G4（TL_A を入れた土台で採用されたもの）。
    plus = core + 未検証の SmartRC 由来のうち採用されたもの。"""
    r = _jload(GROUPDEF / "stages_C/results.json")
    core = list(r.get("core_from_A", [])) + [f"src:{g}" for g in r.get("V4_core_groups", [])]
    if variant == "core":
        return core
    return core + [f"src:{g}" for g in r.get("V4_plus_smartrc_groups", []) if f"src:{g}" not in core]


def _v4_frame(df, base, raw, groups):
    X = pd.concat([df[base + STAGE_COLS], tl_frame(df)], axis=1)
    for g in groups:
        X = pd.concat([X, load_group(g, df, raw)], axis=1)
    return X


def _importance(df, X, added, lp):
    """追加列の gain 重要度。2020〜2021 年の行（最初の fold の学習行）だけで測る（検証年を見ない）。"""
    import lightgbm as lgb

    tr = np.flatnonzero(df["year"].to_numpy() <= 2021)
    y = (df["finish_position"].to_numpy()[tr] == 1).astype(int)
    p = {"objective": "binary", "learning_rate": 0.05, "num_threads": THREADS, "verbose": -1, **lp["lgb"]}
    Xt = X.iloc[tr]
    bst = lgb.train(p, lgb.Dataset(Xt, y), num_boost_round=250)
    imp = pd.Series(bst.feature_importance("gain"), index=X.columns)
    return imp[added].sort_values(ascending=False), Xt


def cmd_importance(a) -> None:
    df, base, meta, folds, lp = setup()
    raw = _raw_keys()
    X = _v4_frame(df, base, raw, v4_groups("core"))
    added = [c for c in X.columns if c not in base + STAGE_COLS and not c.startswith("tl_") and not c.startswith("mtl_")]
    imp, _ = _importance(df, X, added, lp)
    _jsave(OUT / "importance_added.json", {k: float(v) for k, v in imp.items()})
    _log(f"importance_added.json: {len(imp)} 列 / 上位 {list(imp.index[:5])}")


def cmd_prune(a) -> None:
    df, base, meta, folds, lp = setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    raw = _raw_keys()
    groups = v4_groups("core")
    X = _v4_frame(df, base, raw, groups)
    added = [c for c in X.columns if c not in base + STAGE_COLS and not c.startswith("tl_")]
    out = OUT / "prune"
    out.mkdir(parents=True, exist_ok=True)
    rp = out / "results.json"
    r = _jload(rp)
    if "full" not in r:
        lv, by, oof = _run("V4_full", X, meta, folds, lp)
        oof.to_parquet(out / "full.parquet", index=False)
        r["full"] = {"levels": lv, "by_fold": by, "cols": X.shape[1]}
        _jsave(rp, r)
    oof_full = pd.read_parquet(out / "full.parquet")
    imp_added, Xt = _importance(df, X, added, lp)
    corr = Xt[added].select_dtypes("number").sample(min(len(Xt), 40000), random_state=0).corr().abs()
    drop_corr, seen = set(), []
    for c in imp_added.index:
        if c not in corr.columns:
            continue
        if any(corr.loc[c, s] > 0.98 for s in seen if s in corr.columns):
            drop_corr.add(c)
        else:
            seen.append(c)
    variants = {"top150": list(imp_added.index[:150]), "top300": list(imp_added.index[:300]), "corr098": [c for c in added if c not in drop_corr]}
    for name, cols in variants.items():
        if name in r:
            continue
        Xv = X[[c for c in X.columns if c not in added] + cols]
        lv, by, oof = _run(f"V4_{name}", Xv, meta, folds, lp)
        c = _cmp(oof_full, oof, fin_map, name)
        r[name] = {"levels": lv, "by_fold": by, "cols": int(Xv.shape[1]), "vs_full": c, "n_added": len(cols)}
        _jsave(rp, r)
    r["importance_top30"] = {k: float(v) for k, v in imp_added.head(30).items()}
    _jsave(rp, r)
    _jsave(OUT / "importance_added.json", {k: float(v) for k, v in imp_added.items()})


def cmd_v4table(a) -> None:
    df, base, meta, folds, lp = setup()
    raw = _raw_keys()
    cols: dict[str, list[str]] = {}
    parts = []
    for g in v4_groups(a.v4):
        G = load_group(g, df, raw)
        cols[g] = list(G.columns)
        parts.append(G)
    T = pd.concat([df[KEY]] + parts, axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_parquet(OUT / "v4_features.parquet", index=False)
    _jsave(OUT / "v4_columns.json", {"groups": cols, "n_cols": int(T.shape[1] - 2), "base": "tp-03 175 列 + 前段3列（別）"})
    _log(f"v4_features.parquet {T.shape}")


# ───────────────────────── アンサンブル
def _extra_for_ensemble(df: pd.DataFrame, variant: str = "core", top: int | None = None) -> pd.DataFrame:
    """top を渡すと、追加列を gain 重要度の上位 top 列に絞る（NN は 1000 列規模の追加で大きく悪化するため）。"""
    raw = _raw_keys()
    gs = [g for g in v4_groups(variant) if g != "aux"]
    parts = [load_group(g, df, raw) for g in gs]
    E = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    if top:
        imp = _jload(OUT / "importance_added.json")
        keep = [c for c in imp if c in E.columns][:top]
        E = E[keep]
    return E


def cmd_member(a) -> None:
    from src.pipeline.models import zoo_runner
    from src.scripts.data import tp25e_eval as e

    df, meta, folds, members, TLX = e.load_v3()
    recipe, params, X = members[a.name]
    X = pd.concat([X, TLX], axis=1)
    aux = pd.read_parquet(AUX_DIR / "aux_B.parquet")
    aux["race_id"], aux["horse_id"] = aux["race_id"].astype(str), aux["horse_id"].astype(str)
    m = df[KEY].merge(aux, on=KEY, how="left")
    A = m[[c for c in m.columns if c not in KEY]]
    A.index = df.index
    E = _extra_for_ensemble(df, a.v4, NN_TOP if recipe == "nn" else None)
    X = pd.concat([X, A, E], axis=1)
    name = f"{a.name}__v4{'' if a.v4 == 'core' else a.v4}"
    p = OUT / "oof" / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not a.force:
        _log(f"{name} 既存")
        return
    _wait_for_load()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, recipe, X, meta, folds, SEEDS + [2], THREADS, params=dict(params), out_dir=None)
    oof.to_parquet(p, index=False)
    _log(f"{name}: win {res['logloss_win'].mean():.5f} top2 {res['logloss_top2'].mean():.5f} top3 {res['logloss_top3'].mean():.5f} ({time.time() - t0:.0f}s) cols={X.shape[1]}")


def cmd_mtl(a) -> None:
    import torch

    from src.pipeline.models import win_models as wm, zoo
    from src.pipeline.models.tp25c_mtl import mtl_fit_predict
    from src.scripts.data import tp25e_eval as e

    torch.set_num_threads(THREADS)
    p_out = OUT / "oof" / f"mtl_{a.base}__v4{'' if a.v4 == 'core' else a.v4}.parquet"
    if p_out.exists():
        _log(f"{p_out.name} 既存")
        return
    df, meta, folds, members, TLX = e.load_v3()
    _r, params, X = members[a.base]
    X = pd.concat([X, TLX, _extra_for_ensemble(df, a.v4, NN_TOP)], axis=1)
    ex = pd.read_parquet(e.C_OUT / "frame_extra.parquet")
    A = df[KEY].merge(ex[KEY + e.AUX_NAMES], on=KEY, how="left")[e.AUX_NAMES].to_numpy(dtype="float64")
    race, fin = meta["race_id"].to_numpy(), meta["finish_position"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    kw = {k: params[k] for k in ("hidden", "epochs", "lr", "wd", "dropout") if k in params}
    sums = {k: np.zeros(len(meta)) for k in ("win", "top2", "top3")}
    covered = np.zeros(len(meta), bool)
    t0 = time.time()
    seeds = SEEDS + [2]
    for f in folds:
        tr, va = f.train_index, f.valid_index
        if len(tr) < 1000:
            continue
        for seed in seeds:
            o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[va], race[va], seed=seed, aux_w=0.5, **kw)
            fp = zoo.finalize(race[va], {"win": wm.softmax_by_race(race[va], o["win_score"]), "top2": zoo.sigmoid(o["top2_logit"]), "top3": zoo.sigmoid(o["top3_logit"])})
            for k in sums:
                sums[k][va] += fp[k]
        covered[va] = True
        _log(f"mtl {a.base} fold {f.fold_id} done")
    idx = np.flatnonzero(covered)
    oof = pd.DataFrame({"race_id": race[idx], "horse_id": meta["horse_id"].to_numpy()[idx], **{f"p_{k}": sums[k][idx] / len(seeds) for k in sums}})
    p = OUT / "oof" / f"mtl_{a.base}__v4{'' if a.v4 == 'core' else a.v4}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(p, index=False)
    yy = {t: (fin[idx] <= k).astype(float) for t, k in (("win", 1), ("top2", 2), ("top3", 3))}
    _log(f"mtl_{a.base}__v4{a.v4}: " + " ".join(f"{t} {wm.binary_logloss(oof[f'p_{t}'].to_numpy(), yy[t]):.5f}" for t in yy) + f" ({time.time() - t0:.0f}s) cols={X.shape[1]}")


def _read_oof(spec: str) -> tuple[str, pd.DataFrame]:
    kind, name = spec.split(":", 1)
    if kind in ("v4", "v4plus"):
        suf = "__v4" + ("" if kind == "v4" else "plus")
        return name + suf, pd.read_parquet(OUT / "oof" / f"{name}{suf}.parquet")
    if kind == "e3":                         # tp-25e の e3 構成の OOF（補助列のみ）
        from src.scripts.data import tp25e_eval as e

        return e._load_oof(name)
    raise ValueError(spec)


def cmd_ens(a) -> None:
    from src.pipeline.models import ensemble as en
    from src.pipeline.models import feature_bakeoff as fb
    from src.pipeline.models import tp05_data

    D = tp05_data.load_dataset(WORK_V)
    meta = D["meta"]
    base = meta[KEY + ["finish_position", "year"]].copy()
    names = []
    for s in a.members:
        n, o = _read_oof(s)
        names.append(n)
        o = o[KEY + ["p_win", "p_top2", "p_top3"]].copy()
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        base = base.merge(o.rename(columns={f"p_{t}": f"{n}__{t}" for t in ("win", "top2", "top3")}), on=KEY, how="inner")
    race, years, fin = base["race_id"].to_numpy(), base["year"].to_numpy(), base["finish_position"].to_numpy()
    TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}
    Y = {"win": (fin == 1).astype(float), "top2": (fin <= 2).astype(float), "top3": (fin <= 3).astype(float)}
    METHOD = {"win": "weighted_logit", "top2": "greedy", "top3": "greedy"}
    ENS = base[KEY + ["year"]].copy()
    SPEC = {}
    for t in TOT:
        P = np.stack([base[f"{n}__{t}"].to_numpy() for n in names], axis=1)
        llv, pred = en.loyo_eval(METHOD[t], P, years, race, Y[t], TOT[t])
        ENS[f"p_{t}"] = pred
        st = en.fit_method(METHOD[t], P, en.logit(P), Y[t])
        SPEC[t] = {"method": METHOD[t], "weights": dict(zip(names, [round(float(x), 4) for x in st]))}
        _log(f"{a.name} {t} LOYO {llv:.5f} 重み {SPEC[t]['weights']}")
    (OUT / "ens").mkdir(parents=True, exist_ok=True)
    ENS.to_parquet(OUT / "ens" / f"{a.name}.parquet", index=False)
    fin_map = meta.set_index(KEY)["finish_position"]
    out = {"spec": SPEC}
    refs = {"v3": WORK / "oof/model_oof/ensemble_v3.parquet"}
    e3 = WORK / "tp25e/ens/e3.parquet"
    if e3.exists():
        refs["e3"] = e3
    for rn, rp in refs.items():
        ref = pd.read_parquet(rp)[KEY + ["p_win", "p_top2", "p_top3"]]
        ref["race_id"], ref["horse_id"] = ref["race_id"].astype(str), ref["horse_id"].astype(str)
        ref = ref.merge(ENS[KEY], on=KEY, how="inner")
        out[f"vs_{rn}"] = _cmp(ref, ENS[KEY + ["p_win", "p_top2", "p_top3"]], fin_map, f"{a.name} vs {rn}")
        _log(f"{a.name} vs {rn}: " + str({k: round(v, 5) for k, v in out[f'vs_{rn}'].items() if isinstance(v, float)}))
    _jsave(OUT / "ens" / f"{a.name}.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stages")
    s.add_argument("--order", default="A", choices=["A", "B"])
    sp.add_parser("stagesC")
    sp.add_parser("importance")
    sp.add_parser("prune")
    v = sp.add_parser("v4table")
    v.add_argument("--v4", default="core", choices=["core", "plus"])
    m = sp.add_parser("member")
    m.add_argument("name")
    m.add_argument("--force", action="store_true")
    m.add_argument("--v4", default="core", choices=["core", "plus"])
    mt = sp.add_parser("mtl")
    mt.add_argument("base", choices=["nn_extras_ped", "nn_s"])
    mt.add_argument("--v4", default="core", choices=["core", "plus"])
    e = sp.add_parser("ens")
    e.add_argument("--name", required=True)
    e.add_argument("--members", nargs="+", required=True)
    a = ap.parse_args()
    {"stages": cmd_stages, "stagesC": cmd_stages_c, "importance": cmd_importance, "prune": cmd_prune, "v4table": cmd_v4table, "member": cmd_member, "mtl": cmd_mtl, "ens": cmd_ens}[a.cmd](a)


if __name__ == "__main__":
    main()
