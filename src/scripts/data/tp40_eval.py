"""tp-40: 調教（追い切り）の時計の出方から『調子・成長』を読む特徴量の構築・記述分析・V4 への増分評価。

python -m src.scripts.data.tp40_eval build      # 群ごとの特徴量を tp40/groups/*.parquet に保存（追い切り表は tp35/training_laps_fresh.parquet）
python -m src.scripts.data.tp40_eval audit      # 実データで『未来を切り捨てても同じ値になるか』（2023-06-30 で切った版と全期間版を完全一致で比較）
python -m src.scripts.data.tp40_eval repro      # 馬ごとの傾きの再現性（前半/後半の相関、3 分類の一致）
python -m src.scripts.data.tp40_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。基準は V4（tl_ は再取得版）
python -m src.scripts.data.tp40_eval stage2     # 第 2 段: 通過群を valid 2024、新 seed 2 本で確認 + 同時追加
python -m src.scripts.data.tp40_eval describe   # 分位別の V4 との差（レース単位）。第 1 段の基準 OOF を使う
python -m src.scripts.data.tp40_eval report     # /tmp/tp40_report.{json,md}
test 2025・2026 は使わない。市場オッズ・人気は特徴量に使わない。環境変数: TP40_THREADS（既定 6）, TP40_LOAD_LIMIT（既定 14）
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
OUT = V / "tp40"
WORK = ROOT / "data/local/modeling/training_pipeline"
LAPS = Path(os.environ.get("TP40_LAPS") or V / "tp35/training_laps_fresh.parquet")
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field", "TP27_OUT": V / "tp27"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("TP27_THREADS", os.environ.get("TP40_THREADS", "6"))

from src.pipeline.features import tp40_condition as C  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402

KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP40_THREADS", 6))
LOAD_LIMIT = float(os.environ.get("TP40_LOAD_LIMIT", 14))
GROUPS = ["CY", "CYt", "SF", "TJ", "LY", "PO", "IN"]
RUN_COLS = ["race_id", "horse_id", "date", "trainer_id", "finish_position", "field_size", "sex_age", "d_rest", "race_class"]
CUT_AUDIT = "2023-06-30"
LAST_USED = "2024-12-31"                      # 記述分析・再現性は 2024 年末まで（test 2025 は見ない）


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


def _runs(df: pd.DataFrame) -> pd.DataFrame:
    r = df[RUN_COLS].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    return r


def _class_group(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str)
    return pd.Series(np.select([x.str.contains("新馬"), x.str.contains("未勝利"), x.str.contains("勝クラス")], ["新馬", "未勝利", "条件"], "OP以上他"), index=s.index)


def _load_laps() -> pd.DataFrame:
    return pd.read_parquet(LAPS)


def _years_cov(g: pd.DataFrame, df: pd.DataFrame) -> dict:
    m = df[KEY + ["date"]].merge(g, on=KEY, how="left")
    yr = pd.to_datetime(m["date"]).dt.year
    cols = [c for c in g.columns if c not in KEY]
    return {str(y): float(m.loc[yr == y, cols].notna().any(axis=1).mean()) for y in sorted(yr.unique())}


# ───────────────────────── build
def cmd_build(a) -> None:
    df, base, meta, folds, lp = t.setup()
    laps = _load_laps()
    _log(f"追い切り表 {len(laps):,} 本、レース {len(df):,} 行")
    t0 = time.time()
    feats = C.build_features(_runs(df), laps)
    _log(f"構築 {time.time() - t0:.0f}s")
    d = OUT / "groups"
    d.mkdir(parents=True, exist_ok=True)
    rep: dict = {"laps_rows": int(len(laps)), "groups": {}}
    for g, f in feats.items():
        f = f.drop_duplicates(KEY)
        f.to_parquet(d / f"{g}.parquet", index=False)
        cols = [c for c in f.columns if c not in KEY]
        rep["groups"][g] = {"cols": len(cols), "coverage_by_year": _years_cov(f, df), "columns": cols}
        _log(f"  {g}: {len(cols)} 列")
    pr = df[KEY + ["date", "race_class"]].copy()
    pr["prior_runs"] = df.groupby("horse_id").cumcount().to_numpy()
    pr["cls"] = _class_group(pr["race_class"])
    pr[KEY + ["prior_runs", "cls"]].to_parquet(OUT / "run_meta.parquet", index=False)
    cy = feats["SF"].merge(df[KEY + ["date"]], on=KEY)
    cy["year"] = pd.to_datetime(cy["date"]).dt.year
    rep["hist_n_by_year"] = {str(y): {"median": float(g["wc_sf_n"].median()), "zero_share": float((g["wc_sf_n"] == 0).mean())} for y, g in cy.groupby("year")}
    _jsave(OUT / "build_meta.json", rep)


# ───────────────────────── audit
def _audit_one(df: pd.DataFrame, laps: pd.DataFrame, mode: str) -> dict:
    """mode=asof: その日までに存在したデータ（追い切りは調教日 <= 切る日、レースは日付 <= 切る日）だけで作った特徴量が、全期間版と完全一致するか。
    mode=race: 追い切りを『紐づくレースの日付 <= 切る日』で切る（切る日より後のレースに紐づく、それ以前の追い切りも落とす）。標準化の基準が少しずれるかを見る参考。"""
    runs = _runs(df)
    cut = pd.Timestamp(CUT_AUDIT)
    keep = runs[pd.to_datetime(runs["date"]) <= cut]
    if mode == "asof":
        lp = laps[pd.to_datetime(laps["train_date"], errors="coerce") <= cut]
    else:
        lp = laps[pd.to_datetime(laps["race_date"], errors="coerce") <= cut]
    _log(f"[{mode}] 切り捨て版: レース {len(keep):,}、追い切り {len(lp):,} 本")
    part = C.build_features(keep, lp)
    ids = set(keep["race_id"])
    res: dict = {"cut": CUT_AUDIT, "races": len(ids), "mode": mode, "groups": {}}
    n_bad = 0
    for g, p in part.items():
        full = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
        x = full[full["race_id"].isin(ids)].sort_values(KEY).reset_index(drop=True)
        y = p[p["race_id"].isin(ids)].drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)
        assert len(x) == len(y), (g, len(x), len(y))
        cols = {}
        for c in [c for c in x.columns if c not in KEY]:
            xv, yv = x[c].to_numpy(dtype=float), y[c].to_numpy(dtype=float)
            nan_diff = int((np.isnan(xv) != np.isnan(yv)).sum())
            both = ~np.isnan(xv) & ~np.isnan(yv)
            diff = np.abs(xv[both] - yv[both])
            cols[c] = {"nan_diff_rows": nan_diff, "value_diff_rows": int((diff > 0).sum()), "max_abs_diff": float(diff.max()) if diff.size else 0.0}
            n_bad += nan_diff + int((diff > 0).sum())
        res["groups"][g] = cols
        bad = {c: v for c, v in cols.items() if v["nan_diff_rows"] or v["value_diff_rows"]}
        _log(f"  [{mode}] {g}: 不一致の列 {len(bad)} / {len(cols)} {list(bad)[:4]}")
    res["total_mismatch_cells"] = n_bad
    _log(f"[{mode}] 完全一致検査: 不一致セル {n_bad}")
    return res


def cmd_audit(a) -> None:
    df, *_ = t.setup()
    laps = _load_laps()
    out = {m: _audit_one(df, laps, m) for m in ("asof", "race")}
    _jsave(OUT / "audit.json", out)


# ───────────────────────── repro
def _half_slopes(g: pd.DataFrame, xc: str) -> dict:
    out = {}
    for h, part in (("a", g.iloc[: len(g) // 2]), ("b", g.iloc[len(g) // 2:])):
        x, y = part[xc].to_numpy(dtype=float), part["y"].to_numpy(dtype=float)
        ok = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[ok], y[ok]
        n = len(x)
        if n < 4 or x.std() < 1e-9:
            out[h] = (np.nan, np.nan, n)
            continue
        xc_ = x - x.mean()
        b = (xc_ * (y - y.mean())).sum() / (xc_ ** 2).sum()
        resid = y - y.mean() - b * xc_
        se = np.sqrt((resid ** 2).sum() / max(n - 2, 1) / (xc_ ** 2).sum())
        out[h] = (b, b / se if se > 0 else np.nan, n)
    return out


def cmd_repro(a) -> None:
    df, *_ = t.setup()
    cy = pd.read_parquet(OUT / "groups/CY.parquet")[KEY + ["wc_cc_last", "wc_cc_dcyc"]]
    R = df[KEY + ["date", "finish_position", "field_size"]].merge(cy, on=KEY, how="left")
    R = R[pd.to_datetime(R["date"]) <= pd.Timestamp(LAST_USED)].sort_values(["horse_id", "date", "race_id"], kind="stable")
    fs = R["field_size"].to_numpy(dtype=float)
    R["y"] = np.where(fs > 1, 1.0 - (R["finish_position"].to_numpy(dtype=float) - 1.0) / np.maximum(fs - 1.0, 1.0), np.nan)
    rng = np.random.default_rng(0)
    rep: dict = {"through": LAST_USED}
    for xc, tag in (("wc_cc_last", "最終の良さ x1"), ("wc_cc_dcyc", "サイクルの上昇 x2")):
        rows, rows_null = [], []
        for hid, g in R.groupby("horse_id", sort=False):
            if len(g) < 8:
                continue
            s = _half_slopes(g, xc)
            rows.append((hid, *s["a"], *s["b"]))
            gn = g.copy()
            gn["y"] = rng.permutation(gn["y"].to_numpy())             # 馬の中で着順パーセンタイルを並べ替えた帰無
            sn = _half_slopes(gn, xc)
            rows_null.append((hid, *sn["a"], *sn["b"]))
        cols = ["horse_id", "b_a", "t_a", "n_a", "b_b", "t_b", "n_b"]
        d = pd.DataFrame(rows, columns=cols).dropna(subset=["b_a", "b_b"])
        dn = pd.DataFrame(rows_null, columns=cols).dropna(subset=["b_a", "b_b"])

        def cls(tv: float) -> str:
            return "増えると上がる" if tv > 1.5 else ("減ると上がる" if tv < -1.5 else "無関係")

        def agree(dd: pd.DataFrame) -> dict:
            ca, cb = dd["t_a"].map(cls), dd["t_b"].map(cls)
            tab = pd.crosstab(ca, cb)
            return {"n": int(len(dd)), "agree_share": float((ca == cb).mean()), "share_a": {k: float(v) for k, v in ca.value_counts(normalize=True).items()}, "table": {str(i): {str(j): int(tab.loc[i, j]) for j in tab.columns} for i in tab.index}}
        boots = []
        for _ in range(500):
            ix = rng.integers(0, len(d), len(d))
            boots.append(np.corrcoef(d["b_a"].to_numpy()[ix], d["b_b"].to_numpy()[ix])[0, 1])
        rep[tag] = {"horses": int(len(d)), "corr_slope": float(np.corrcoef(d["b_a"], d["b_b"])[0, 1]), "corr_slope_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "spearman": float(d["b_a"].corr(d["b_b"], method="spearman")), "null_corr": float(np.corrcoef(dn["b_a"], dn["b_b"])[0, 1]) if len(dn) > 2 else float("nan"),
                    "classes": agree(d), "classes_null": agree(dn)}
        _log(f"  {tag}: 馬 {len(d)} 頭 前半×後半の傾きの相関 {rep[tag]['corr_slope']:+.3f} {rep[tag]['corr_slope_ci']}（帰無 {rep[tag]['null_corr']:+.3f}）、3 分類の一致 {rep[tag]['classes']['agree_share']:.3f}（帰無 {rep[tag]['classes_null']['agree_share']:.3f}）")
    # 追い切りの『上がり方』そのものの馬ごとの安定性（体質）
    for xc, tag in (("wc_cc_dcyc", "サイクルの上昇の馬ごとの平均"), ("wc_cc_last", "最終の良さの馬ごとの平均")):
        rows = []
        for hid, g in R.groupby("horse_id", sort=False):
            x = g[xc].dropna().to_numpy()
            if len(x) < 8:
                continue
            rows.append((x[: len(x) // 2].mean(), x[len(x) // 2:].mean()))
        d = np.array(rows)
        rep[tag] = {"horses": int(len(d)), "corr": float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])}
        _log(f"  {tag}: 馬 {len(d)} 頭 前半×後半の相関 {rep[tag]['corr']:+.3f}")
    _jsave(OUT / "repro.json", rep)


# ───────────────────────── stage1 / stage2
def _frame(df: pd.DataFrame) -> pd.DataFrame:
    """基準 = V4 の 1,062 列のうち tl_ を再取得版の TL_A（cand_TL_A_v2）に差し替えたもの（再取得による被覆の差を、この調査の効果に混ぜないため）。"""
    X = pd.read_parquet(V / "tp31" / "v4_frame.parquet")
    X.index = df.index
    if os.environ.get("TP40_BASE", "fresh") == "fresh":
        new = pd.read_parquet(V / "tp35/cand_TL_A_v2.parquet")
        new["race_id"], new["horse_id"] = new["race_id"].astype(str), new["horse_id"].astype(str)
        m = df[KEY].merge(new, on=KEY, how="left")
        m.index = df.index
        for c in [c for c in X.columns if c.startswith("tl_")]:
            if f"{c[3:]}" in m.columns:
                X[c] = m[c[3:]].to_numpy()
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    return X


SPLITS = {"IN_cp": ("IN", ["wc_in1_cp", "wc_in1_beta", "wc_in2_cp", "wc_in2_beta"]), "IN_meta": ("IN", ["wc_in1_n", "wc_in1_rel", "wc_in2_n", "wc_in2_rel"])}


def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    """IN_cp = 馬ごとの傾きと補正ポイント（主張している仕組み）、IN_meta = 件数と信頼度（その馬の過去の追い切りの出来のばらつき）だけ。IN の効果の出どころを切り分ける。"""
    if g in SPLITS:
        src, cols = SPLITS[g]
        T = pd.read_parquet(OUT / "groups" / f"{src}.parquet")[KEY + cols]
    else:
        T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


def _run_subset(name, X, meta, folds, lp, years: list[int], seeds: list[int]):
    from src.pipeline.models import zoo_runner

    fs = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    _wait()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, fs, seeds, THREADS, params=dict(lp))
    lv = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")
    return lv, oof


def _subset_cmp(oof0, oof, fin_map, rm: pd.DataFrame, label: str) -> dict:
    out = {}
    rmi = rm.set_index(KEY)
    k0 = pd.MultiIndex.from_frame(oof0[KEY].astype(str))
    pr, cl = rmi["prior_runs"].reindex(k0).to_numpy(), rmi["cls"].reindex(k0).to_numpy()
    masks = {"履歴0走(初出走)": pr == 0, "履歴0〜2走": pr <= 2, "履歴3走以上": pr >= 3, "新馬戦": cl == "新馬", "未勝利": cl == "未勝利", "条件戦": cl == "条件", "OP以上他": cl == "OP以上他"}
    for nm, mk in masks.items():
        if mk.sum() < 2000:
            continue
        a, b = oof0[mk], oof[mk]
        r = t._cmp(a, b, fin_map, f"{label}/{nm}")
        out[nm] = {"rows": int(mk.sum()), "Δwin": r["Δwin"], "win_hi": r["win_hi"], "Δtop2": r["Δtop2"], "Δtop3": r["Δtop3"]}
    return out


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str]) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    rm = pd.read_parquet(OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    X0 = _frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
    if bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = _run_subset("V4(再取得tl_)", X0, meta, folds, lp, years, seeds)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        _jsave(resp, res)
    steps = res.setdefault("groups", {})
    for g in groups:
        if g in steps:
            continue
        if g.startswith("ALL"):
            names = [x for x in groups if not x.startswith("ALL") and x not in SPLITS]
            G = pd.concat([_load_group(x, df) for x in names], axis=1)
        else:
            G = _load_group(g, df)
        lv, oof = _run_subset(f"V4+{g}", pd.concat([X0, G], axis=1), meta, folds, lp, years, seeds)
        oof.to_parquet(d / f"oof_{g}.parquet", index=False)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c, "subsets": _subset_cmp(oof0, oof, fin_map, rm, g)}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
        for nm, s in steps[g]["subsets"].items():
            _log(f"      {nm}（{s['rows']:,} 行）: Δwin {s['Δwin']:+.5f} (hi {s['win_hi']:+.5f})")


def cmd_stage1(a) -> None:
    _stage("stage1", [2022, 2023], [0], GROUPS + ["ALL"])


def cmd_stage2(a) -> None:
    r = _jload(OUT / "stage1/results.json")
    passed = [g for g, s in r.get("groups", {}).items() if not g.startswith("ALL") and s["vs_V4"]["Δwin"] < -0.00012 and s["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1/passed.json", passed)
    if not passed:
        return
    extra = ["IN_cp", "IN_meta"] if "IN" in passed else []
    _stage("stage2", [2024], [2, 3], passed + extra + (["ALL_PASSED"] if len(passed) > 1 else []))


# ───────────────────────── describe
FEATS = {"wc_cc_last": "最終追い切りの良さ", "wc_cc_d1": "最終−1 本前（上昇度）", "wc_cc_dcyc": "最終−サイクル最初（上昇）", "wc_cc_slope": "サイクル内の傾き",
         "wc_sf_last_vs_mean": "最終−自分の過去の平均", "wc_sf_last_vs_best": "最終−自己ベスト", "wc_tj_slope4": "サイクルごとの最終の推移（傾き）", "wc_in1_cp": "馬ごとの傾き×今回（補正ポイント）",
         "wc_po_cell": "母集団曲線（年齢×性別×帯）"}


def cmd_describe(a) -> None:
    df, *_ = t.setup()
    bp = OUT / "stage1/base.parquet"
    if not bp.exists():                                   # 第 1 段の基準がまだ無ければ、同じ valid 2022+2023 の V4 の OOF（tp-35 第 1 段の基準）で記述する
        bp = V / "tp35/stage1/base.parquet"
    _log(f"基準 OOF: {bp}")
    bo = pd.read_parquet(bp)
    bo["race_id"], bo["horse_id"] = bo["race_id"].astype(str), bo["horse_id"].astype(str)
    rm = pd.read_parquet(OUT / "run_meta.parquet")
    feats = pd.concat([pd.read_parquet(OUT / "groups" / f"{g}.parquet").set_index(KEY) for g in ("CY", "SF", "TJ", "IN", "PO")], axis=1).reset_index()
    for f in (feats, rm):
        f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    d = bo.merge(df[KEY + ["date", "finish_position"]].assign(race_id=lambda x: x["race_id"].astype(str), horse_id=lambda x: x["horse_id"].astype(str)), on=KEY).merge(feats, on=KEY, how="left").merge(rm[KEY + ["prior_runs", "cls"]], on=KEY, how="left")
    T = C.prepare_workouts(_load_laps(), _runs(df))
    S = C._order(T)
    kl = S[S["k_end"] == 0][C.KEYC + ["kind"]].rename(columns={"kind": "kind_last"})
    kr = _runs(df)[KEY + ["date"]].assign(race_date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")).merge(kl, on=C.KEYC, how="left")[KEY + ["kind_last"]]
    d = d.merge(kr, on=KEY, how="left")
    d["win"] = (d["finish_position"] == 1).astype(float)
    d["top3"] = (d["finish_position"] <= 3).astype(float)
    train = df.merge(feats, on=KEY, how="left")
    train = train[pd.to_datetime(train["date"]) < pd.Timestamp("2022-01-01")]
    out: dict = {"rows": int(len(d)), "years": "valid 2022〜2023（基準 OOF）", "features": {}}
    for f, nm in FEATS.items():
        if f not in d.columns:
            continue
        edges = np.unique(np.nanquantile(train[f].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8]))
        if len(edges) < 2:
            continue
        rows = []
        subs = [("全体", np.ones(len(d), bool)), ("履歴0〜2走", (d["prior_runs"] <= 2).to_numpy())] + [(f"最終={k}", (d["kind_last"] == k).to_numpy()) for k in ("slope", "wood", "poly")]
        for subset_name, mk in subs:
            sub = d[mk & d[f].notna().to_numpy()].copy()
            sub["bin"] = np.digitize(sub[f].to_numpy(dtype=float), edges)
            for b, g in sub.groupby("bin"):
                race = g.groupby("race_id").agg(win_any=("win", "max"), exp_win=("p_win", "sum"), n=("win", "size"))
                rows.append({"subset": subset_name, "bin": int(b), "runs": int(len(g)), "races": int(len(race)), "horse_win": float(g["win"].mean()), "horse_pred": float(g["p_win"].mean()),
                             "horse_top3": float(g["top3"].mean()), "horse_top3_pred": float(g["p_top3"].mean()), "race_win": float(race["win_any"].mean()), "race_win_pred": float(race["exp_win"].clip(upper=1).mean()),
                             "mean_horses_per_race": float(race["n"].mean())})
        out["features"][f] = {"label": nm, "edges": [float(e) for e in edges], "rows": rows}
        base = [r for r in rows if r["subset"] == "全体"]
        _log(f"  {nm}: 帯 {len(base)}、レース勝率 実測−V4 予測 = " + " ".join(f"{r['race_win'] - r['race_win_pred']:+.3f}" for r in base))
    _jsave(OUT / "describe.json", out)


# ───────────────────────── report
def cmd_report(a) -> None:
    rep = {k: _jload(OUT / f"{k}.json") for k in ("build_meta", "audit", "repro", "describe")}
    rep["stage1"], rep["stage2"] = _jload(OUT / "stage1/results.json"), _jload(OUT / "stage2/results.json")
    Path("/tmp/tp40_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lines = ["# tp-40 調教の時計の出方（調子・成長）", ""]
    for st in ("stage1", "stage2"):
        gs = rep[st].get("groups", {})
        if not gs:
            continue
        lines += [f"## {st}", "", "| 群 | 列数 | Δwin(上限) | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|"]
        for g, s in gs.items():
            c = s["vs_V4"]
            lines.append(f"| {g} | {s['cols']} | {c['Δwin']:+.5f}({c['win_hi']:+.5f}) | {c['Δtop2']:+.5f} | {c['Δtop3']:+.5f} | {'採用' if c['採用'] else '不採用'} |")
        lines.append("")
    Path("/tmp/tp40_report.md").write_text("\n".join(lines), encoding="utf-8")
    _log("report: /tmp/tp40_report.{json,md}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "audit", "repro", "stage1", "stage2", "describe", "report"])
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
