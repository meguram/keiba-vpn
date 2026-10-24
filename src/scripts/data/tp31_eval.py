"""tp-31: コース特注の特徴量の検討と評価。

python -m src.scripts.data.tp31_eval audit      # V4 アンサンブル(v4_e2)の OOF 残差が、コース×属性で系統的にずれているか（検定+年をまたぐ検証）
python -m src.scripts.data.tp31_eval groups     # 候補群（T1〜T6）を作って tp31/groups/*.parquet に保存
python -m src.scripts.data.tp31_eval frame      # V4 の LightGBM 用の特徴量表を tp31/v4_frame.parquet に保存（以降の評価で使い回す）
python -m src.scripts.data.tp31_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp31_eval stage2     # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認
test 2025・2026 は使わない。市場オッズ・人気は使わない。環境変数: TP31_THREADS（既定 6）, TP31_LOAD_LIMIT（既定 16）。
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
V = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = V / "tp31"
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field", "TP27_OUT": V / "tp27"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("TP27_THREADS", os.environ.get("TP31_THREADS", "6"))

from src.pipeline.features import tp31_course_specific as cs  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402

KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP31_THREADS", 6))
LOAD_LIMIT = float(os.environ.get("TP31_LOAD_LIMIT", 16))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wait() -> None:
    t0 = time.time()
    while os.getloadavg()[0] >= LOAD_LIMIT and time.time() - t0 < 3600:
        time.sleep(30)


def _jsave(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ───────────────────────── 監査
def _bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[o] <= thr
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    sig = np.zeros(n, bool)
    sig[o[:k]] = True
    return sig


def cmd_audit(a) -> None:
    from scipy import stats

    df, base, meta, folds, lp = t.setup()
    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    m = df[KEY + ["year", "finish_position", "venue_code", "surface", "distance", "course_type", "sire", "dam_sire", "jockey_id", "trainer_id", "a_style", "field_size",
                  "horse_number", "sex_age", "jockey_weight", "weight", "d_rest", "c_cls_step", "date"]].merge(ens[KEY + ["p_win"]], on=KEY, how="inner")
    m = m.reset_index(drop=True)
    y = (pd.to_numeric(m["finish_position"]) == 1).astype(float).to_numpy()
    p = m["p_win"].clip(1e-6, 1 - 1e-6).to_numpy()
    r = y - p
    var = p * (1 - p)
    ck = cs.course_keys(m)
    fv = cs.feature_values(m)
    years = m["year"].to_numpy()
    _log(f"監査対象 {len(m)} 行（valid 2022〜24）、厳密コース {ck['c0'].nunique()}")

    rid = m["race_id"].to_numpy()

    # コース別の較正: p はレース内で合計 1 に正規化されているので、残差の合計は 0（平均の較正は恒等的に成り立つ）。
    # 代わりに『コース別の温度 b』（p^b をレース内で再正規化）で LogLoss が下がるかを、年をまたいで見る。
    def _temp(pv: np.ndarray, yv: np.ndarray, rv: np.ndarray, b: float) -> float:
        q = pv ** b
        den = pd.Series(q).groupby(rv).transform("sum").to_numpy()
        q = np.clip(q / den, 1e-9, 1 - 1e-9)
        return float(-np.sum(yv * np.log(q)))

    grid = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    rows = []
    for c0, idx in ck.groupby("c0").indices.items():
        if len(idx) < 3000:
            continue
        per = {}
        for ty in (2023, 2024):
            tr = idx[years[idx] < ty]
            te = idx[years[idx] == ty]
            if len(tr) < 1500 or len(te) < 500:
                continue
            best = min(grid, key=lambda b: _temp(p[tr], y[tr], rid[tr], b))
            per[ty] = {"b": best, "gain": (_temp(p[te], y[te], rid[te], best) - _temp(p[te], y[te], rid[te], 1.0)) / len(te)}
        if per:
            rows.append({"c0": c0, "n": int(len(idx)), **{f"b_{ty}": v["b"] for ty, v in per.items()}, **{f"gain_{ty}": v["gain"] for ty, v in per.items()}})
    cal = pd.DataFrame(rows)
    cal.to_csv(OUT / "audit_temperature_by_course.csv", index=False)

    # コース×属性のセル。残差 r はレース内で合計 0 なので、独立な二項分布の分散は過大になる。レースを単位にしたクラスター頑健な z = Σ s / sqrt(Σ s²) を使う。
    cells = []
    for lvl in ("c0", "c1"):
        for fname, f in fv.items():
            key = (ck[lvl] + "#" + f).to_numpy()
            tt = pd.DataFrame({"k": key, "rid": rid, "r": r, "y": years})
            g = tt.groupby(["k", "rid", "y"], sort=False)["r"].agg(["sum", "size"]).reset_index()
            g["s2"] = g["sum"] ** 2
            tot = g.groupby("k").agg(sr=("sum", "sum"), s2=("s2", "sum"), n=("size", "sum"))
            tot = tot[tot["n"] >= 80]
            if tot.empty:
                continue
            gy = g.groupby(["k", "y"]).agg(sr=("sum", "sum"), s2=("s2", "sum")).reset_index()
            py = gy.pivot(index="k", columns="y", values="sr").reindex(tot.index)
            pv = gy.pivot(index="k", columns="y", values="s2").reindex(tot.index)
            tot["z"] = tot["sr"] / np.sqrt(tot["s2"])
            tot["pval"] = 2 * stats.norm.sf(np.abs(tot["z"]))
            first = (py[2022].fillna(0) + py[2023].fillna(0)) / np.sqrt(pv[2022].fillna(0) + pv[2023].fillna(0) + 1e-9)
            last = (py[2024] / np.sqrt(pv[2024])).fillna(0)
            tot["rep"] = (np.sign(first) == np.sign(last)) & (np.abs(first) > 1.0) & (last * np.sign(first) > 1.0)
            tot["level"], tot["feature"] = lvl, fname
            cells.append(tot.reset_index())
    C = pd.concat(cells, ignore_index=True)
    C["sig_bh"] = _bh(C["pval"].to_numpy())
    C["sig_rep"] = C["sig_bh"] & C["rep"]
    C["course"] = C["k"].str.split("#").str[0]
    C["value"] = C["k"].str.split("#").str[1]
    C.sort_values("pval").to_csv(OUT / "audit_cells.csv", index=False)
    summ = {"cells_tested": int(len(C)), "z_sd": float(C["z"].std()), "frac_abs_z_gt2": float((C["z"].abs() > 2).mean()), "sig_bh": int(C["sig_bh"].sum()), "sig_bh_and_replicated": int(C["sig_rep"].sum()),
            "by_feature": C.groupby("feature").agg(tested=("k", "size"), sig=("sig_bh", "sum"), sig_rep=("sig_rep", "sum")).reset_index().to_dict("records"),
            "temperature_courses": int(len(cal)),
            "temperature_gain_mean_2023": float(cal["gain_2023"].mean()) if "gain_2023" in cal else None,
            "temperature_gain_mean_2024": float(cal["gain_2024"].mean()) if "gain_2024" in cal else None}
    # 年をまたぐ検証: 前の年で推定したセル効果で logit を補正して、次の年の LogLoss が下がるか（属性ごと、c0 レベル）
    lg = np.log(p / (1 - p))
    cross = []
    for fname, f in fv.items():
        key = ck["c0"] + "#" + f
        tot_gain = {}
        for ty in (2023, 2024):
            tr = years < ty
            te = years == ty
            g = pd.DataFrame({"k": key[tr], "r": r[tr], "v": var[tr]}).groupby("k").agg(sr=("r", "sum"), sv=("v", "sum"))
            for kappa in (50.0, 200.0):
                d = (g["sr"] / (g["sv"] + kappa)).rename("d")
                dd = key[te].map(d).fillna(0.0).to_numpy()
                p2 = 1 / (1 + np.exp(-(lg[te] + dd)))
                ll0 = -np.mean(y[te] * np.log(p[te]) + (1 - y[te]) * np.log(1 - p[te]))
                ll1 = -np.mean(y[te] * np.log(p2) + (1 - y[te]) * np.log(1 - p2))
                tot_gain[(ty, kappa)] = float(ll1 - ll0)
        cross.append({"feature": fname, **{f"d_logloss_{ty}_k{int(k)}": v for (ty, k), v in tot_gain.items()}})
    summ["cross_year"] = cross
    top = C[C["sig_rep"]].sort_values("pval").head(40)
    summ["examples_replicated"] = top[["level", "feature", "course", "value", "n", "z"]].to_dict("records")
    summ["examples_top_z"] = C.sort_values("pval").head(12)[["level", "feature", "course", "value", "n", "z", "pval"]].to_dict("records")
    _jsave(OUT / "audit_summary.json", summ)
    _log(f"監査: セル {len(C)}（z の sd {C['z'].std():.3f}、|z|>2 の割合 {(C['z'].abs() > 2).mean():.4f}）、BH 有意 {int(C['sig_bh'].sum())}、再現あり {int(C['sig_rep'].sum())}")
    _log(f"  コース別の温度: 対象 {len(cal)} コース、2023 の平均利得 {summ['temperature_gain_mean_2023']}、2024 の平均利得 {summ['temperature_gain_mean_2024']}")
    for c in cross:
        _log("  年またぎ " + c["feature"] + " " + " ".join(f"{k}={v:+.5f}" for k, v in c.items() if k != "feature"))


# ───────────────────────── 候補群
def cmd_groups(a) -> None:
    df, base, meta, folds, lp = t.setup()
    parts = cs.build_all(df)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    for g, G in parts.items():
        T = pd.concat([df[KEY].reset_index(drop=True), G.reset_index(drop=True)], axis=1)
        T.to_parquet(gdir / f"{g}.parquet", index=False)
        _log(f"{g}: {G.shape[1]} 列、欠損率 {float(G.isna().mean().mean()):.3f}")


def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


def cmd_frame(a) -> None:
    df, base, meta, folds, lp = t.setup()
    raw = t._raw_keys()
    X = t._v4_frame(df, base, raw, t.v4_groups("core"))
    X = X.loc[:, ~X.columns.duplicated()]
    p = OUT / "v4_frame.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "string")]
    num = X.drop(columns=cat)
    out = pd.concat([num.astype("float32"), X[cat]], axis=1) if cat else num.astype("float32")
    out.to_parquet(p, index=False)
    _log(f"v4_frame {out.shape}（カテゴリ {len(cat)}）")


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    p = OUT / "v4_frame.parquet"
    if not p.exists():
        raise FileNotFoundError("先に frame を実行")
    X = pd.read_parquet(p)
    X.index = df.index
    # カテゴリ列は学習表と同じ扱い（category）にそろえる
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    return X


def _run_subset(name, X, meta, folds, lp, years: list[int], seeds: list[int]):
    from src.pipeline.models import zoo_runner

    fs = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    _wait()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, fs, seeds, THREADS, params=dict(lp))
    lv = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")
    return lv, oof


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str], a) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = _frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
    if bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = _run_subset("V4", X0, meta, folds, lp, years, seeds)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        _jsave(resp, res)
    steps = res.setdefault("groups", {})
    for g in groups:
        if g in steps:
            continue
        G = pd.concat([_load_group(x, df) for x in groups if x != "ALL_PASSED"], axis=1) if g == "ALL_PASSED" else _load_group(g, df)
        Xn = pd.concat([X0, G], axis=1)
        lv, oof = _run_subset(f"V4+{g}", Xn, meta, folds, lp, years, seeds)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


GROUP_NAMES = ["T1_sire", "T2_conn", "T3_style_post", "T4_body", "T5_rest_cls", "T6_agg"]


def cmd_stage1(a) -> None:
    _stage("stage1", [2022, 2023], [0], GROUP_NAMES, a)


def cmd_stage2(a) -> None:
    r = _jload(OUT / "stage1/results.json")
    passed = [g for g, s in r.get("groups", {}).items() if s["vs_V4"]["Δwin"] < -0.00012 and s["vs_V4"]["win_hi"] < 0.0001]       # 第 1 段は seed 1 本・2 年の粗い選別なので緩めに通す
    _log(f"第 1 段を通過: {passed}")
    if not passed:
        return
    _stage("stage2", [2024], [2, 3], passed + (["ALL_PASSED"] if len(passed) > 1 else []), a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "groups", "frame", "stage1", "stage2"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"audit": cmd_audit, "groups": cmd_groups, "frame": cmd_frame, "stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()
