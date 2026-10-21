"""tp-63: アンサンブルの『幅』— ランダムフォレスト・決定木の袋詰め・別の目的関数・学習データの見せ方違いのメンバーを OOF で作る。

python -m ... run NAME [--years 2022,2023,2024] [--seeds 0] [--threads 3]    # NAME は CONFIGS のキー
python -m ... list

入力は V5 と同じ 1,142 列（v4_frame 1,062 列 + tl_ 再取得版 + CY + ALL_BEST）。学習は年の前進（検証年より前の全年）、検証は 2022〜2024 年（V5 の 7 メンバーの OOF と同じ行）。
test 2025・2026 年は使わない。市場オッズ・人気は使わない。重い処理は /tmp/heavy_run.sh 経由で起動する。
環境変数: TP63_OUT（既定 /home/jovyan/work/tp63_out）, TP63_THREADS（既定 3）。
ホストのリポジトリは jovyan から書けないので、このファイルは /home/jovyan/work/tp63_code に置き PYTHONPATH で読む。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TP40_BASE", "old")
os.environ.setdefault("V5_GROUPS", "CY,BEST")
from src.scripts.data import tp27_eval as t          # noqa: E402
from src.scripts.data import tp40_eval as e40        # noqa: E402
from src.scripts.data import v5_common               # noqa: E402

V = Path(e40.V)
KEY = ["race_id", "horse_id"]
OUT = Path(os.environ.get("TP63_OUT") or "/home/jovyan/work/tp63_out")
THREADS = int(os.environ.get("TP63_THREADS", 3))
SIZES: list[float] = []
SENTINEL = -999.0


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────── データ
def load_frame():
    df, base, meta, folds, lp = t.setup()
    X = pd.read_parquet(V / "tp31" / "v4_frame.parquet")
    X.index = df.index
    tl = [c for c in X.columns if c.startswith("tl_")]
    X[tl] = v5_common.fresh_tlx(df, X[tl])
    G = v5_common.group_frame(df)
    X = pd.concat([X, G.loc[:, ~G.columns.isin(X.columns)]], axis=1)
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    _log(f"V5 の入力 {X.shape}（期待 1,142 列）")
    return df, X, meta, folds, lp


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """木（sklearn）用: カテゴリは符号、欠損は -999、無限は -999。"""
    cols = {}
    for c in X.columns:
        s = X[c]
        if str(s.dtype) == "category":
            cols[c] = s.cat.codes.astype("float32")
        else:
            v = pd.to_numeric(s, errors="coerce").astype("float32").to_numpy()
            v = np.where(np.isfinite(v), v, np.float32(SENTINEL))
            cols[c] = v
    return pd.DataFrame(cols, index=X.index)


def condition_labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """行ごとの条件ラベル（レース条件はレース前に分かる量だけ）。"""
    dist = df["distance"].astype(float).to_numpy()
    out = {
        "surface": df["surface"].astype(str).to_numpy(),
        "distance": np.select([dist <= 1400, dist <= 1800, dist <= 2200], ["短(〜1400)", "マイル(〜1800)", "中(〜2200)"], "長(2201〜)"),
        "going": np.where(df["track_condition"].astype(str).isin(["良"]), "良", "稍重〜不良"),
        "field": np.select([df["field_size"].astype(float).to_numpy() <= 10, df["field_size"].astype(float).to_numpy() <= 14], ["〜10頭", "11〜14頭"], "15頭〜"),
    }
    cl = df["race_class"].astype(str)
    out["class"] = np.select([cl.str.contains("新馬|未勝利"), cl.str.contains("１勝"), cl.str.contains("２勝|３勝|オープン")], ["新馬・未勝利", "1勝", "2勝以上・OP"], "障害他")
    return out


LABELS: dict[str, pd.Series] = {}          # 条件名 -> ラベル（df の index）
QUALITY: dict = {}                          # race_id -> 実測のペース段階 / 事前の確率
ORACLE: list = []                           # レース質別の専門モデルで、実測の段階を使った（オラクル）予測


def load_quality(df: pd.DataFrame) -> None:
    """共有表 shared_cache/race_quality の実測ペース段階（学習の分け方・オラクル）と、事前の確率 p_joint_*（拡張窓で学習済み、実運用の混合比）。"""
    q = pd.read_parquet(V.parent / "shared_cache" / "race_quality" / "race_quality.parquet")
    q["race_id"] = q["race_id"].astype(str)
    q = q.drop_duplicates("race_id").set_index("race_id")
    pace = ["slow", "mid", "high"]
    pj = pd.DataFrame({c: q[[f"p_joint_{c}|{sh}" for sh in ("compact", "normal", "stretched")]].sum(axis=1) for c in pace})
    pj = pj.div(pj.sum(axis=1), axis=0)
    pj[q[[f"p_joint_{c}|normal" for c in pace]].isna().any(axis=1)] = np.nan
    QUALITY["pace"] = q["pace_cls"]
    QUALITY["pi"] = pj
    LABELS["pace"] = pd.Series(df["race_id"].astype(str).map(q["pace_cls"]).fillna("unk").to_numpy(), index=df.index)


# ───────────────────────── 追加のレシピ（zoo.fit_predict に差し込む）
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _platt(p_oob: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    ok = np.isfinite(p_oob)
    m = LogisticRegression(C=1e4, max_iter=500).fit(_logit(p_oob[ok])[:, None], y[ok])
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def fit_sk_trees(kind: str, Xtr, ytr, Xva, seed: int, threads: int, prm: dict) -> dict:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    out = {}
    Xt, Xv = np.asarray(Xtr, dtype=np.float32), np.asarray(Xva, dtype=np.float32)
    cal = prm.get("cal")
    for tgt in ("win", "top2", "top3"):
        y = ytr[tgt]
        if kind == "tree":
            m = DecisionTreeClassifier(max_depth=prm.get("max_depth", 5), min_samples_leaf=prm.get("min_samples_leaf", 200), random_state=seed)
            m.fit(Xt, y)
            p = m.predict_proba(Xv)[:, 1]
        else:
            cls = ExtraTreesClassifier if kind == "et" else RandomForestClassifier
            kw = dict(n_estimators=prm.get("n_estimators", 120), max_depth=prm.get("max_depth"), max_features=prm.get("max_features", "sqrt"),
                      min_samples_leaf=prm.get("min_samples_leaf", 30), n_jobs=threads, random_state=seed, bootstrap=True,
                      max_samples=prm.get("max_samples", 0.35), oob_score=bool(cal), class_weight=prm.get("class_weight"))
            m = cls(**kw)
            m.fit(Xt, y)
            p = m.predict_proba(Xv)[:, 1]
            if cal == "platt":
                a, b = _platt(m.oob_decision_function_[:, 1], y)
                p = 1 / (1 + np.exp(-(a * _logit(p) + b)))
        out[tgt] = p
        if kind != "tree" or tgt == "win":
            SIZES.append(len(pickle.dumps(m)) / 1e6)
    return out


def lgb_rank(Xtr, ytr, race_tr, Xva, race_va, threads, seed, params, kind):
    from src.pipeline.models import win_models as wm

    o = wm.fit_predict(kind, Xtr, ytr, race_tr, Xva, race_va, threads, seed=seed, params=params.get("lgb"), n_rounds=params.get("n_rounds", wm.N_ROUNDS))
    return {"win": wm.softmax_by_race(race_va, o["win_score"]), "win_score": o["win_score"]}


def lgb_surface(Xtr, ytr, race_tr, Xva, race_va, threads, seed, params, orig):
    """芝・ダートそれぞれの専門モデル。各馬は自分の面のモデルで予測する（障害などその他の面は全体モデル）。"""
    col = "surface"
    out = {k: np.full(len(Xva), np.nan) for k in ("win", "top2", "top3")}
    s_tr, s_va = Xtr[col].astype(str).to_numpy(), Xva[col].astype(str).to_numpy()
    full = None
    for v in sorted(set(s_va)):
        mt, mv = s_tr == v, s_va == v
        if mt.sum() < 5000:
            if full is None:
                full = orig("lgb_binary", Xtr, ytr, race_tr, Xva, race_va, threads, seed=seed, params=params)
            for k in out:
                out[k][mv] = full[k][mv]
            continue
        sub = orig("lgb_binary", Xtr[mt], {k: np.asarray(a)[mt] for k, a in ytr.items()}, race_tr[mt], Xva[mv], race_va[mv], threads, seed=seed, params=params)
        for k in out:
            out[k][mv] = sub[k]
    return out


def lgb_split(Xtr, ytr, race_tr, Xva, race_va, threads, seed, params, orig):
    """条件（params['by']）ごとの専門モデル。学習行の少ない水準は全体モデルで補う。"""
    by = params["by"]
    sub_params = {k: v for k, v in params.items() if k != "by"}
    l_tr, l_va = LABELS[by].loc[Xtr.index].to_numpy(), LABELS[by].loc[Xva.index].to_numpy()
    out = {k: np.full(len(Xva), np.nan) for k in ("win", "top2", "top3")}
    full = None
    for v in sorted(set(l_va)):
        mt, mv = l_tr == v, l_va == v
        if mt.sum() < 8000:
            if full is None:
                full = orig("lgb_binary", Xtr, ytr, race_tr, Xva, race_va, threads, seed=seed, params=sub_params)
            for k in out:
                out[k][mv] = full[k][mv]
            continue
        sub = orig("lgb_binary", Xtr[mt], {k: np.asarray(a)[mt] for k, a in ytr.items()}, race_tr[mt], Xva[mv], race_va[mv], threads, seed=seed, params=sub_params)
        for k in out:
            out[k][mv] = sub[k]
    return out


def lgb_quality(Xtr, ytr, race_tr, Xva, race_va, threads, seed, params, orig):
    """レース質（実測のペース段階: スロー/ミドル/ハイ）別の専門モデルを作り、(a) 実測の段階のモデルで予測（オラクル、上限）、(b) 事前の確率 π で混合（実運用）を出す。
    返すのは (b)。(a) は ORACLE に貯める。事前の確率が無い行（2020 年）は全体の頻度で混ぜる。"""
    sub_params = {k: v for k, v in params.items() if k != "by"}
    cls_tr = LABELS["pace"].loc[Xtr.index].to_numpy()
    preds = {}
    for c in ("slow", "mid", "high"):
        mt = cls_tr == c
        preds[c] = orig("lgb_binary", Xtr[mt], {k: np.asarray(a)[mt] for k, a in ytr.items()}, race_tr[mt], Xva, race_va, threads, seed=seed, params=sub_params)
    pi = QUALITY["pi"].reindex(pd.Index(race_va)).to_numpy()
    freq = pd.Series(cls_tr).value_counts(normalize=True).reindex(["slow", "mid", "high"]).fillna(0).to_numpy()
    pi = np.where(np.isnan(pi), freq[None, :], pi)
    mix = {k: sum(pi[:, i] * preds[c][k] for i, c in enumerate(("slow", "mid", "high"))) for k in ("win", "top2", "top3")}
    real = pd.Series(race_va).map(QUALITY["pace"]).to_numpy()
    orc = {k: np.where(real == "slow", preds["slow"][k], np.where(real == "high", preds["high"][k], preds["mid"][k])) for k in ("win", "top2", "top3")}
    ORACLE.append((Xva.index.to_numpy(), np.asarray(race_va), orc))
    return mix


def lgb_dart(Xtr, ytr, race_tr, Xva, race_va, threads, seed, params, orig):
    p = json.loads(json.dumps(params))
    p.setdefault("lgb", {})["boosting"] = "dart"
    p["lgb"]["drop_rate"] = 0.1
    p["n_rounds"] = int(p.get("n_rounds", 300))
    return orig("lgb_binary", Xtr, ytr, race_tr, Xva, race_va, threads, seed=seed, params=p)


def logreg(Xtr, ytr, Xva, seed, prm):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xt, Xv = np.asarray(Xtr, dtype=np.float32), np.asarray(Xva, dtype=np.float32)
    Xt, Xv = np.where(Xt == SENTINEL, np.nan, Xt), np.where(Xv == SENTINEL, np.nan, Xv)
    med = np.nanmedian(Xt, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xt, Xv = np.where(np.isnan(Xt), med, Xt), np.where(np.isnan(Xv), med, Xv)
    sc = StandardScaler().fit(Xt)
    lo, hi = -5.0, 5.0
    Xt, Xv = np.clip(sc.transform(Xt), lo, hi), np.clip(sc.transform(Xv), lo, hi)
    out = {}
    for tgt in ("win", "top2", "top3"):
        m = LogisticRegression(C=prm.get("C", 0.01), penalty="l2", max_iter=300, solver="lbfgs")
        m.fit(Xt, ytr[tgt])
        out[tgt] = m.predict_proba(Xv)[:, 1]
    SIZES.append(Xt.shape[1] * 3 * 8 / 1e6)
    return out


def knn(Xtr, ytr, Xva, seed, prm):
    """標準化した上位列（prm['cols']）の k 近傍。近傍の着順ラベルの平均（距離で重み付け）。"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    cols = prm["cols"]
    A = Xtr[cols].astype("float32")
    B = Xva[cols].astype("float32")
    med = A.median()
    A, B = A.fillna(med), B.fillna(med)
    sc = StandardScaler().fit(A)
    A, B = np.clip(sc.transform(A), -5, 5), np.clip(sc.transform(B), -5, 5)
    out = {}
    for tgt in ("win", "top2", "top3"):
        m = KNeighborsClassifier(n_neighbors=prm.get("k", 200), weights="distance", n_jobs=THREADS, algorithm="brute")
        m.fit(A, ytr[tgt])
        out[tgt] = np.clip(m.predict_proba(B)[:, 1], 1e-4, 1 - 1e-4)
    SIZES.append(A.nbytes / 1e6)
    return out


CUSTOM = {"lgb_split", "lgb_quality", "sk", "lgb_xendcg", "lgb_lambdarank", "lgb_surface", "lgb_dart", "logreg", "knn"}


def patch_zoo(cfg_params: dict) -> None:
    from src.pipeline.models import zoo

    orig = zoo.fit_predict

    def fp(recipe, X_tr, y_tr, race_tr, X_va, race_va, threads, seed=0, params=None, model_out=None):
        params = params or {}
        if recipe == "sk":
            return fit_sk_trees(params["kind"], X_tr, y_tr, X_va, seed, threads, params)
        if recipe == "lgb_xendcg":
            return lgb_rank(X_tr, y_tr, race_tr, X_va, race_va, threads, seed, params, "xendcg")
        if recipe == "lgb_lambdarank":
            return lgb_rank(X_tr, y_tr, race_tr, X_va, race_va, threads, seed, params, "lambdarank")
        if recipe == "lgb_surface":
            return lgb_surface(X_tr, y_tr, race_tr, X_va, race_va, threads, seed, params, orig)
        if recipe == "lgb_split":
            return lgb_split(X_tr, y_tr, race_tr, X_va, race_va, threads, seed, params, orig)
        if recipe == "lgb_quality":
            return lgb_quality(X_tr, y_tr, race_tr, X_va, race_va, threads, seed, params, orig)
        if recipe == "lgb_dart":
            return lgb_dart(X_tr, y_tr, race_tr, X_va, race_va, threads, seed, params, orig)
        if recipe == "logreg":
            return logreg(X_tr, y_tr, X_va, seed, params)
        if recipe == "knn":
            return knn(X_tr, y_tr, X_va, seed, params)
        return orig(recipe, X_tr, y_tr, race_tr, X_va, race_va, threads, seed=seed, params=params, model_out=model_out)

    zoo.fit_predict = fp


# 名前 -> (recipe, 入力 numeric か, params, 学習設定)
CONFIGS: dict[str, dict] = {
    # ランダムフォレスト系（sklearn。欠損は -999、カテゴリは符号）
    "rf_deep": dict(recipe="sk", numeric=True, params=dict(kind="rf", n_estimators=120, max_depth=14, max_features=0.08, min_samples_leaf=30, max_samples=0.35)),
    "rf_deep_cal": dict(recipe="sk", numeric=True, params=dict(kind="rf", n_estimators=120, max_depth=14, max_features=0.08, min_samples_leaf=30, max_samples=0.35, cal="platt")),
    "rf_shallow": dict(recipe="sk", numeric=True, params=dict(kind="rf", n_estimators=200, max_depth=8, max_features=0.25, min_samples_leaf=100, max_samples=0.35)),
    "rf_shallow_cal": dict(recipe="sk", numeric=True, params=dict(kind="rf", n_estimators=200, max_depth=8, max_features=0.25, min_samples_leaf=100, max_samples=0.35, cal="platt")),
    "rf_full": dict(recipe="sk", numeric=True, params=dict(kind="rf", n_estimators=100, max_depth=None, max_features=0.1, min_samples_leaf=20, max_samples=0.25)),
    "et_deep": dict(recipe="sk", numeric=True, params=dict(kind="et", n_estimators=150, max_depth=16, max_features=0.1, min_samples_leaf=30, max_samples=0.35, cal="platt")),
    "bagtree4": dict(recipe="sk", numeric=True, params=dict(kind="rf", n_estimators=150, max_depth=4, max_features=0.2, min_samples_leaf=200, max_samples=0.3, cal="platt")),
    "bagtree6": dict(recipe="sk", numeric=True, params=dict(kind="rf", n_estimators=150, max_depth=6, max_features=0.2, min_samples_leaf=150, max_samples=0.3, cal="platt")),
    "tree5": dict(recipe="sk", numeric=True, params=dict(kind="tree", max_depth=5, min_samples_leaf=200)),
    # 別の目的関数（LightGBM）
    "lgb_xendcg": dict(recipe="lgb_xendcg", numeric=False, params="lp"),
    "lgb_lambdarank": dict(recipe="lgb_lambdarank", numeric=False, params="lp"),
    # 学習データの見せ方違い
    "lgb_surface": dict(recipe="lgb_surface", numeric=False, params="lp"),
    "lgb_recent3": dict(recipe="lgb_binary", numeric=False, params="lp", extra=dict(min_train_year=2021)),
    "lgb_half2": dict(recipe="lgb_binary", numeric=False, params="lp", extra=dict(halflife_years=2.0)),
    "lgb_dart": dict(recipe="lgb_dart", numeric=False, params="lp"),
    "lgb_split_dist": dict(recipe="lgb_split", numeric=False, params="lp", extra=dict(by="distance")),
    "lgb_split_going": dict(recipe="lgb_split", numeric=False, params="lp", extra=dict(by="going")),
    "lgb_split_class": dict(recipe="lgb_split", numeric=False, params="lp", extra=dict(by="class")),
    # レース質の事前の確率（共有表の p_joint_*・p_lap_*・p_l3f_*。拡張窓で学習済みで as-of）を入力に足した全体モデル
    # 感度の掃引で決めた設定（tune/best_params.json: {"lgb": {...上書きするキー}, "n_rounds": 木数}）を、V5 と同じ入力・同じ学習率で使う
    # 対照: V5 の LightGBM の設定のまま、1 seed・同じ入力（調整後メンバーとの比較用）
    "lgb_ctrl": dict(recipe="lgb_binary", numeric=False, params="lp"),
    "lgb_retuned": dict(recipe="lgb_binary", numeric=False, params="tuned"),
    "lgb_qfeat": dict(recipe="lgb_binary", numeric=False, params="lp", xcols="quality"),
    "lgb_q_pace": dict(recipe="lgb_quality", numeric=False, params="lp", oracle="lgb_q_pace_oracle"),
    # 線形・距離
    "logreg_l2": dict(recipe="logreg", numeric=True, params=dict(C=0.01)),
}


def cmd_run(a) -> None:
    from src.pipeline.models import zoo_runner

    cfg = CONFIGS[a.name]
    if (OUT / f"skip_{a.name}").exists():
        _log(f"{a.name}: skip（時間の都合で見送り）")
        return
    out_p = OUT / "ev" / "members" / f"{a.name}.parquet"
    if out_p.exists() and not a.force:
        _log(f"{a.name}: 済み")
        return
    df, X, meta, folds, lp = load_frame()
    years = [int(y) for y in a.years.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    fs = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    if cfg["params"] == "tuned":
        ov = json.loads((OUT / "tune" / (os.environ.get("TP63_TUNED", "best_params") + ".json")).read_text(encoding="utf-8"))
        prm = json.loads(json.dumps(lp))
        prm["lgb"].update(ov.get("lgb", {}))
        if "n_rounds" in ov:
            prm["n_rounds"] = int(ov["n_rounds"])
        _log(f"調整後の設定: {prm}")
    else:
        prm = dict(lp) if cfg["params"] == "lp" else dict(cfg["params"])
    prm.update(cfg.get("extra", {}))
    if a.fast and cfg["params"] in ("lp", "tuned"):
        prm = json.loads(json.dumps(prm))
        prm["lgb"]["learning_rate"] = float(prm["lgb"]["learning_rate"]) * 3
        prm["n_rounds"] = max(60, int(prm["n_rounds"] // 3))
    if cfg.get("xcols") == "quality":
        q = pd.read_parquet(V.parent / "shared_cache" / "race_quality" / "race_quality.parquet")
        q["race_id"] = q["race_id"].astype(str)
        qc = [c for c in q.columns if c.startswith(("p_joint_", "p_lap_", "p_l3f_"))]
        qq = q.drop_duplicates("race_id").set_index("race_id")[qc]
        add = qq.reindex(df["race_id"].astype(str)).astype("float32")
        add.index = X.index
        add.columns = [f"rq_{c}" for c in qc]
        X = pd.concat([X, add], axis=1)
        _log(f"レース質の事前確率 {len(qc)} 列を追加 → {X.shape}")
    Xin = to_numeric(X) if cfg["numeric"] else X
    if cfg["recipe"] == "knn":
        prm["cols"] = json.loads((OUT / "knn_cols.json").read_text(encoding="utf-8"))
    global THREADS
    THREADS = a.threads
    LABELS.update({k: pd.Series(v, index=df.index) for k, v in condition_labels(df).items()})
    if cfg["recipe"] == "lgb_quality":
        load_quality(df)
    patch_zoo(prm)
    t0 = time.time()
    res, oof = zoo_runner.run_oof(a.name, cfg["recipe"], Xin, meta, fs, seeds, a.threads, params=prm)
    (OUT / "ev" / "members").mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_p, index=False)
    if cfg.get("oracle") and ORACLE:
        from src.pipeline.models import zoo

        parts = []
        for idx, race_va, orc in ORACLE:
            fp_ = zoo.finalize(race_va, orc)
            o = meta.loc[idx, ["race_id", "horse_id", "year"]].copy().reset_index(drop=True)
            for k in fp_:
                o[f"p_{k}"] = fp_[k]
            parts.append(o)
        pd.concat(parts, ignore_index=True).to_parquet(OUT / "ev" / "members" / f"{cfg['oracle']}.parquet", index=False)
    info = {"name": a.name, "seconds": time.time() - t0, "logloss_win": float(res["logloss_win"].mean()), "logloss_top2": float(res["logloss_top2"].mean()),
            "logloss_top3": float(res["logloss_top3"].mean()), "model_mb_per_fold": SIZES[-3:] if SIZES else None, "cols": int(Xin.shape[1]), "params": {k: v for k, v in prm.items() if k != "cols"}}
    (OUT / "ev" / "members").joinpath(f"{a.name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"{a.name}: win {info['logloss_win']:.5f} top2 {info['logloss_top2']:.5f} top3 {info['logloss_top3']:.5f} ({info['seconds']:.0f}s) size={info['model_mb_per_fold']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("name")
    r.add_argument("--years", default="2022,2023,2024")
    r.add_argument("--seeds", default="0")
    r.add_argument("--threads", type=int, default=THREADS)
    r.add_argument("--fast", action="store_true")
    r.add_argument("--force", action="store_true")
    sp.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "list":
        print("\n".join(CONFIGS))
        return
    cmd_run(a)


if __name__ == "__main__":
    main()
