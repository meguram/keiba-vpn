"""tp-54 (3/3): 追走難度 v2 の学習・walk-forward 評価・特徴量エクスポート。

python -m src.scripts.data.tp54_model eval      # 年別 walk-forward（学習は評価年より前）。現行 ease_pct との比較、前走比、結果を results.json に
python -m src.scripts.data.tp54_model oof       # 年別 leave-one-year-out の予測 → $TP54_OUT/groups/EASE2.parquet（V5 評価の特徴量候補）
python -m src.scripts.data.tp54_model final     # 2020〜2025 全体で当てはめて config/tracking_difficulty_v2/{model.txt,meta.json} を書く

学習表は tp54_dataset.build の出力（$TP54_OUT/horse_ease.parquet）。KEIBA_ROOT / TP54_OUT で場所を指定。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])
OUT = Path(os.environ.get("TP54_OUT") or ROOT / "data/local/modeling/tp54")
sys.path.insert(0, str(ROOT))

try:
    from src.pipeline.models import tracking_difficulty_v2 as td2  # noqa: E402
except ImportError:      # ホストは repo が root 所有で新規ファイルを置けないので、同じディレクトリのコピーを使う
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tracking_difficulty_v2 as td2  # noqa: E402

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, verbose=-1, num_threads=int(os.environ.get("TP54_THREADS", "3")), seed=0)
N_ROUNDS = int(os.environ.get("TP54_ROUNDS", "500"))
# 『追走』に関わる量だけ（要求＋位置取り・前で運んだときの癖）。年齢・馬体重・休養・距離の変化・前走の上がりなど、一般の調子の変化は含めない
CORE = ["pf_early", "pf_rank_norm", "gate_norm", "field_size", "z_hat", "pf_front", "pf_stalker", "pf_pressure", "pf_known_frac", "turf", "distance_km", "heavy", "venue_code",
        "demand_idx", "demand_idx_z", "w_adj", "w_dev", "w_dev_front", "w_dev_rear", "w_early", "w_n", "w_nfront", "w_nrear", "prev_early", "early_shift", "same_style_run"]
DEMAND_ONLY = ["pf_early", "pf_rank_norm", "gate_norm", "field_size", "z_hat", "pf_front", "pf_stalker", "pf_pressure", "pf_known_frac", "turf", "distance_km", "heavy",
               "venue_code", "demand_idx", "demand_idx_z"]


def load() -> pd.DataFrame:
    h = pd.read_parquet(OUT / "horse_ease.parquet")
    h = h.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = h.groupby("horse_id", sort=False)
    for c in ("z_hat", "pf_front", "pf_stalker", "pf_pressure", "pf_known", "venue", "track_condition", "horse_number", "Y", "pf_early", "pf_rank"):
        h[f"prevc_{c}"] = g[c].shift(1)
    return h


def _baseline_design(h: pd.DataFrame) -> np.ndarray:
    """その馬の過去の上がりの力だけで今回の上がり偏差を説明するベースライン（平均への回帰を含む）。"""
    wa, wd = h["w_adj"].astype(float), h["w_dev"].astype(float)
    return np.column_stack([np.ones(len(h)), wa.fillna(0), wa.isna().astype(float), wd.fillna(0), h["w_n"].astype(float).fillna(0)])


def baseline_fit(h: pd.DataFrame, train: pd.Series) -> np.ndarray:
    t = h["act_l3f_dev"].clip(-td2.Y_CLIP, td2.Y_CLIP)
    m = train & t.notna()
    coef, *_ = np.linalg.lstsq(_baseline_design(h[m]), t[m].to_numpy(), rcond=None)
    return coef


def make_target(h: pd.DataFrame, coef: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """返り値: (T=実測の上がり偏差を±4 秒に丸めたもの, Y2=T − ベースライン)。Y2 は『過去の力では説明できない分』＝要求（位置取り・ペース・コース）と文脈の効果＋雑音。
    ベースラインを引くのは、過去の上がりの力が『特徴量でもあり、目的変数の一部でもある』ことによる見かけの相関（平均への回帰）を除くため。"""
    t = h["act_l3f_dev"].clip(-td2.Y_CLIP, td2.Y_CLIP)
    b = pd.Series(_baseline_design(h) @ coef, index=h.index)
    return t, (t - b).clip(-td2.Y_CLIP, td2.Y_CLIP)


def fit(X: pd.DataFrame, y: np.ndarray, rounds: int = N_ROUNDS, valid: tuple | None = None):
    import lightgbm as lgb

    ds = lgb.Dataset(X, y)
    cb = []
    vs = []
    if valid is not None:
        vs = [lgb.Dataset(valid[0], valid[1], reference=ds)]
        cb = [lgb.early_stopping(50, verbose=False)]
    return lgb.train(PARAMS, ds, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


def corr(a, b) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    return float(d["a"].corr(d["b"])) if len(d) > 100 else float("nan")


def scorr(a, b) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    return float(d["a"].corr(d["b"], method="spearman")) if len(d) > 100 else float("nan")


def decile_table(score: pd.Series, y: pd.Series, n: int = 10) -> list[dict]:
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    d["q"] = pd.qcut(d["s"].rank(method="first"), n, labels=False)
    t = d.groupby("q")["y"].agg(["mean", "size"]).round(4)
    return [{"q": int(i), "mean_Y": float(r["mean"]), "n": int(r["size"])} for i, r in t.iterrows()]


def prev_swap(h: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """前走比: x_a=『今回の条件・前走と同じ位置』, x_b=『前走の条件・前走と同じ位置』。能力側の特徴は今回のまま。"""
    pe = h["prev_act_early_norm"].astype(float)
    fs_prev = h["prev_field_size"].astype(float)
    xa = X.copy()
    xa["pf_early"] = pe
    xa["demand_idx"] = (1.0 - pe) * (-h["z_hat"].astype(float))
    xa["demand_idx_z"] = (1.0 - pe) * (-h["z_hat"].astype(float).clip(-2, 2)) * (xa["distance_km"] - 0.6)
    xb = xa.copy()
    xb["z_hat"] = h["prevc_z_hat"].astype(float)
    xb["pf_front"], xb["pf_stalker"], xb["pf_pressure"] = (h[f"prevc_{c}"].astype(float) for c in ("pf_front", "pf_stalker", "pf_pressure"))
    xb["pf_known_frac"] = (h["prevc_pf_known"].astype(float) / fs_prev.clip(lower=1)).clip(upper=1.0)
    xb["turf"] = (h["prev_surface"] == "芝").astype(float)
    xb["distance_km"] = h["prev_distance"].astype(float) / 1000.0
    xb["heavy"] = (h["prevc_track_condition"].astype(str) != "良").astype(float)
    xb["venue_code"] = h["prevc_venue"].map({v: i for i, v in enumerate(td2.VENUES)}).astype(float)
    xb["field_size"] = fs_prev
    xb["gate_norm"] = h["prevc_horse_number"].astype(float) / fs_prev.clip(lower=1)
    xb["pf_rank_norm"] = (h["prevc_pf_rank"].astype(float) - 1.0) / (fs_prev - 1.0).clip(lower=1)
    xb["demand_idx"] = (1.0 - pe) * (-xb["z_hat"])
    xb["demand_idx_z"] = (1.0 - pe) * (-xb["z_hat"].clip(-2, 2)) * (xb["distance_km"] - 0.6)
    return xa, xb


def run_eval() -> dict:
    h = load()
    X = td2.make_features(h)
    ok = h["pf_early"].notna()
    res: dict = {"n_rows": int(len(h)), "n_with_T": int(h["act_l3f_dev"].notna().sum()), "by_year": {}}
    pairs_all = []
    for yr in (2023, 2024, 2025):
        coef = baseline_fit(h, ok & (h["year"] < yr))
        T, y = make_target(h, coef)
        h["y2"] = y
        h["prev_y2"] = h.groupby("horse_id", sort=False)["y2"].shift(1)
        tr = ok & y.notna() & (h["year"] < yr)
        te = ok & y.notna() & (h["year"] == yr)
        mB = fit(X[tr], y[tr].to_numpy())
        mA = fit(X.loc[tr, DEMAND_ONLY], y[tr].to_numpy())
        mC = fit(X.loc[tr, CORE], y[tr].to_numpy())
        pB, pA, pC = mB.predict(X[te]), mA.predict(X.loc[te, DEMAND_ONLY]), mC.predict(X.loc[te, CORE])
        yt = y[te]                 # ベースラインを引いた実測 Y2（秒）
        yc = y[te]
        bte = (T - y)[te]          # ベースライン（過去の力による予測）
        tte = T[te]
        row = {
            "n_test": int(te.sum()),
            "v2_full": {"corr": corr(pB, yc), "corr_on_T": corr(pB + bte, tte), "corr_baseline_on_T": corr(bte, tte), "spearman": scorr(pB, yc), "r2": 1 - float(np.mean((yc - pB) ** 2) / np.var(yc))},
            "v2_core": {"corr": corr(pC, yc), "spearman": scorr(pC, yc), "r2": 1 - float(np.mean((yc - pC) ** 2) / np.var(yc)), "corr_on_T": corr(pC + bte, tte), "calibration_slope": float(np.polyfit(pC, yc, 1)[0]),
                        "r2_on_T": 1 - float(np.mean((tte - bte - pC) ** 2) / np.var(tte))},
            "v2_demand_only": {"corr": corr(pA, yc), "spearman": scorr(pA, yc), "r2": 1 - float(np.mean((yc - pA) ** 2) / np.var(yc))},
            "baseline_current_ease": {"corr_neg_ease": corr(-h.loc[te, "td_ease"], yc), "spearman": scorr(-h.loc[te, "td_ease"], yc), "corr_neg_ease_on_T": corr(-h.loc[te, "td_ease"], tte)},
            "baseline_current_stamina": {"corr_stamina": corr(-h.loc[te, "td_stamina"], yc)},
            "baseline_position": {"corr_pf_early": corr(h.loc[te, "pf_early"], yc)},
            "calibration_slope": float(np.polyfit(pB, yc, 1)[0]),
            "r2_on_T": {"baseline_ability_only": 1 - float(np.mean((tte - bte) ** 2) / np.var(tte)), "baseline_plus_v2": 1 - float(np.mean((tte - bte - pB) ** 2) / np.var(tte)),
                        "baseline_plus_demand_only": 1 - float(np.mean((tte - bte - pA) ** 2) / np.var(tte))},
            "deciles_by_pred": decile_table(pd.Series(pB, index=yc.index), yt),
            "deciles_by_current_ease": decile_table(-h.loc[te, "td_ease"], yt),
            "deciles_by_baseline_only": decile_table(bte, yt),
            "deciles_by_core": decile_table(pd.Series(pC, index=yc.index), yt),
        }
        # 現行の ease_pct（絶対値）の目盛りの偏り
        row["current_ease_share_ge60"] = float((h.loc[te, "td_ease"] >= 60).mean())
        row["current_ease_share_lt30"] = float((h.loc[te, "td_ease"] < 30).mean())
        # 位置取り三分位 × ペース三分位での一致（予測と実測の平均 Y）
        d = pd.DataFrame({"pred": pC, "Y": yt.to_numpy(), "pos": pd.qcut(h.loc[te, "pf_early"].rank(method="first"), 3, labels=["前", "中", "後"]).to_numpy(),
                          "pace": pd.cut(h.loc[te, "z_hat"], [-9, -0.3, 0.3, 9], labels=["速", "並", "遅"]).to_numpy()})
        grid = d.groupby(["pos", "pace"], observed=True).agg(pred=("pred", "mean"), actual=("Y", "mean"), n=("Y", "size")).round(3).reset_index()
        row["position_pace_grid"] = grid.to_dict("records")
        # (c) 前走比
        pe_ok = te & h["prev_act_early_norm"].notna() & h["prevc_z_hat"].notna() & h["prev_y2"].notna()
        xa, xb = prev_swap(h[pe_ok], X[pe_ok])
        delta = mC.predict(xb[CORE]) - mC.predict(xa[CORE])          # 正＝前走の条件のほうが重かった＝今回は楽（追走に関わる量だけ）
        delta_full = mB.predict(xb) - mB.predict(xa)
        sub = h[pe_ok]
        same = (np.abs(sub["pf_early"] - sub["prev_act_early_norm"]) < 0.15) & ((sub["act_early_norm"] - sub["prev_act_early_norm"]).abs() < 0.15)
        improve = (sub["prev_y2"] - sub["y2"]).to_numpy()  # 正＝今回のほうが終いに余裕
        row["vs_prev"] = {"n": int(pe_ok.sum()), "n_same_pos": int(same.sum()), "corr_all": corr(delta, improve), "corr_same_pos": corr(delta[same.to_numpy()], improve[same.to_numpy()]), "corr_same_pos_full": corr(delta_full[same.to_numpy()], improve[same.to_numpy()]),
                          "spearman_same_pos": scorr(delta[same.to_numpy()], improve[same.to_numpy()]),
                          "deciles_same_pos": decile_table(pd.Series(delta[same.to_numpy()]), pd.Series(improve[same.to_numpy()]), 5)}
        res["by_year"][str(yr)] = row
        pairs_all.append(pd.DataFrame({"year": yr, "pred": pB}))
        if yr == 2025:
            imp = pd.Series(mB.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
            res["feature_gain_top"] = {k: round(float(v / imp.sum()), 4) for k, v in imp.head(15).items()}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res


def run_oof() -> None:
    h = load()
    X = td2.make_features(h)
    ok = h["pf_early"].notna()
    out = pd.DataFrame(index=h.index, columns=["drain", "full", "demand", "delta_prev"], dtype=float)
    for yr in sorted(h["year"].unique()):
        coef = baseline_fit(h, ok & (h["year"] != yr))
        T, y = make_target(h, coef)
        tr = ok & y.notna() & (h["year"] != yr)
        te = ok & (h["year"] == yr)
        mB = fit(X[tr], y[tr].to_numpy())
        mA = fit(X.loc[tr, DEMAND_ONLY], y[tr].to_numpy())
        mC = fit(X.loc[tr, CORE], y[tr].to_numpy())
        out.loc[te, "drain"] = mC.predict(X.loc[te, CORE])
        out.loc[te, "full"] = mB.predict(X[te])
        out.loc[te, "demand"] = mA.predict(X.loc[te, DEMAND_ONLY])
        pe_ok = te & h["prev_act_early_norm"].notna() & h["prevc_z_hat"].notna()
        xa, xb = prev_swap(h[pe_ok], X[pe_ok])
        out.loc[pe_ok, "delta_prev"] = mC.predict(xb[CORE]) - mC.predict(xa[CORE])
        print("oof", yr, int(te.sum()), flush=True)
    q = np.nanquantile(out["drain"], np.linspace(0, 1, 101)).tolist()
    g = pd.DataFrame({"race_id": h["race_id"], "horse_id": h["horse_id"], "EASE2_drain": out["drain"], "EASE2_demand": out["demand"], "EASE2_full": out["full"],
                      "EASE2_capacity": out["drain"] - out["demand"], "EASE2_pct": td2.to_ease_pct(out["drain"].fillna(np.nan).to_numpy(), q), "EASE2_vs_prev": out["delta_prev"]})
    g.loc[out["drain"].isna(), "EASE2_pct"] = np.nan
    (OUT / "groups").mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT / "groups" / "EASE2.parquet", index=False)
    (OUT / "oof_quantiles.json").write_text(json.dumps(q), encoding="utf-8")
    print("wrote", OUT / "groups" / "EASE2.parquet", len(g))


def run_final() -> None:
    h = load()
    X = td2.make_features(h)
    base_ok = h["pf_early"].notna()
    coef = baseline_fit(h, base_ok)
    T, y = make_target(h, coef)
    ok = base_ok & y.notna()
    m = fit(X.loc[ok, CORE], y[ok].to_numpy())
    mD = fit(X.loc[ok, DEMAND_ONLY], y[ok].to_numpy())
    pred = m.predict(X.loc[h["pf_early"].notna(), CORE])
    q = np.quantile(pred, np.linspace(0, 1, 101)).tolist()
    d = Path(os.environ.get("TP54_CONFIG_DIR") or td2.CONFIG_DIR)
    d.mkdir(parents=True, exist_ok=True)
    m.save_model(str(d / "model.txt"))
    mD.save_model(str(d / "model_demand.txt"))
    meta = {"version": "tp54-1", "features": CORE, "demand_features": DEMAND_ONLY, "quantiles": q, "y_clip": td2.Y_CLIP, "trained_years": [int(h["year"].min()), int(h["year"].max())],
            "n_train": int(ok.sum()), "params": {k: v for k, v in PARAMS.items() if k != "num_threads"}, "rounds": N_ROUNDS,
            "baseline_coef": [float(c) for c in coef],
            "note": "Y2 = 実測の上がり偏差（レース内の中央値との差、±4秒）− 過去の上がりの力によるベースライン。予測が大きい＝過去の力から見て終いが落ちる（大変）"}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    pe = OUT / "position_effect.json"
    if pe.exists():
        (d / "position_effect.json").write_text(pe.read_text(encoding="utf-8"), encoding="utf-8")
    print("saved", d)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"eval": lambda: print(json.dumps(run_eval(), ensure_ascii=False, indent=1, default=float)[:6000]), "oof": run_oof, "final": run_final}[cmd]()
