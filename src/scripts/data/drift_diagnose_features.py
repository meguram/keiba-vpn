"""ドリフト診断（特徴量側）: drift_dump_features で抜き取った『推論経路の特徴量』を、2025 年と 2026 年で比べる。

  python -m src.scripts.data.drift_diagnose_features --out /home/jovyan/work/drift_out/features.json

比較（すべて同じ推論経路なので、学習表との構成差は含まない）:
  y25_vs_y26        2025 年(基準) vs 2026 年の特徴量ごとの PSI / KS / 欠損率 / 平均のずれ
  y26_jan_vs_jun    2026 年 1〜5 月 vs 6〜10 月（6 月に何かが変わったか）
  y25_jan_vs_jun    2025 年 1〜5 月 vs 6〜10 月（季節の対照。ここでも出るずれは季節差）
  noise             同じ分布から引いた PSI の中央値（標本が小さいと PSI が上振れする）
さらに、列の系統(接頭辞)ごとの集計、月ごとの欠損率、敵対的検証(AUC と上位列)、2025 年の推論経路 vs 学習側バッチ予測の差（推論経路の誤差）を出す。
指標は報告のみ。モデルや閾値の選択には使わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M
from src.scripts.data.drift_monitor import load_dumps

META = {"race_id", "horse_id", "day", "p_win", "p_top2", "p_top3"}


def family(col: str) -> str:
    for pre in ("mtl_", "wc_", "tl_", "ped_", "pd_", "ob_", "exp_", "seq_", "fdr_", "fd_", "fm_", "fc_", "fr_", "fe_", "frn_", "s_", "o_", "px_"):
        if col.startswith(pre):
            return pre.rstrip("_")
    return col.split("_")[0] if "_" in col else "base"


def shift_table(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    t = M.feature_shift(ref, cur, cols)
    t["family"] = t["feature"].map(family)
    return t


def family_summary(t: pd.DataFrame) -> list[dict]:
    g = t.groupby("family")
    out = []
    for fam, x in g:
        out.append({"family": fam, "n": int(len(x)), "psi_ge_025": int((x["psi"] >= 0.25).sum()), "psi_ge_010": int((x["psi"] >= 0.10).sum()),
                    "psi_median": float(x["psi"].median()), "miss_jump_ge_010": int(((x["miss_cur"] - x["miss_ref"]) >= 0.10).sum()),
                    "miss_ref_mean": float(x["miss_ref"].mean()), "miss_cur_mean": float(x["miss_cur"].mean())})
    return sorted(out, key=lambda r: -r["psi_ge_025"])


def month_of_day(d: pd.DataFrame) -> pd.Series:
    return d["day"].astype(str).str[:6]


def monthly_missing(d: pd.DataFrame, cols: list[str]) -> dict:
    d = d.assign(ym=month_of_day(d))
    fam = pd.Series({c: family(c) for c in cols})
    out = {}
    for ym, g in d.groupby("ym"):
        miss = g[cols].isna().mean()
        out[ym] = {f: float(miss[fam[fam == f].index].mean()) for f in sorted(fam.unique())} | {"_rows": int(len(g))}
    return out


def adversarial(a: pd.DataFrame, b: pd.DataFrame, cols: list[str], seed: int = 0) -> dict:
    """a と b を見分けるモデルの AUC（レース単位 4 分割）と、見分けに効いた列の上位。AUC 0.5 付近なら同じ分布。"""
    import lightgbm as lgb

    x = pd.concat([a[cols + ["race_id"]], b[cols + ["race_id"]]], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    for c in cols:
        if x[c].dtype == object:
            x[c] = x[c].astype("category")
    races = x["race_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(races, rng.integers(0, 4, len(races))))
    fold = x["race_id"].astype(str).map(fold_of).to_numpy()
    X = x[cols]
    aucs, imp = [], pd.Series(0.0, index=cols)
    for k in range(4):
        tr, te = fold != k, fold == k
        if y[te].min() == y[te].max():
            continue
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, importance_type="gain", verbose=-1, random_state=seed)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        from sklearn.metrics import roc_auc_score

        aucs.append(float(roc_auc_score(y[te], p)))
        imp += pd.Series(m.feature_importances_, index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    return {"auc_mean": float(np.mean(aucs)) if aucs else float("nan"), "auc_folds": aucs, "rows_a": int(len(a)), "rows_b": int(len(b)),
            "top_features": [{"feature": k, "gain_share": float(v), "family": family(k)} for k, v in imp.head(25).items()]}


def path_error(root: Path, d25: pd.DataFrame, pred_file: str) -> dict:
    """2025 年の抜き取り行で、推論経路の予測と学習側バッチの予測(bundle_*_2024 の test 2025)を比べる。"""
    from src.scripts.data.drift_data import load_results_year, market_probs

    p = root / pred_file
    if not p.exists() or "p_win" not in d25.columns:
        return {"status": "no_data"}
    tr = pd.read_parquet(p)[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]]
    for c in ("race_id", "horse_id"):
        tr[c] = tr[c].astype(str)
    s = d25[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]].copy()
    s["race_id"], s["horse_id"] = s["race_id"].astype(str), s["horse_id"].astype(str)
    m = s.merge(tr, on=["race_id", "horse_id"], suffixes=("_srv", "_trn"))
    tables = root / "data/page_reference/tables"
    res = load_results_year(tables, 2025)
    mk, _ = market_probs(tables, 2025)
    m = m.merge(res[["race_id", "horse_id", "horse_number", "finish_position"]], on=["race_id", "horse_id"]).merge(mk, on=["race_id", "horse_number"])
    out = {"status": "ok", "rows": int(len(m)), "races": int(m["race_id"].nunique())}
    y = (m["finish_position"] == 1).astype(float).to_numpy()
    for tag in ("srv", "trn"):
        out[f"ll_win_{tag}"] = float(M.bin_ll(y, m[f"p_win_{tag}"]).mean())
        lg = np.log(np.clip(m[f"p_win_{tag}"], 1e-6, 1 - 1e-6) / (1 - np.clip(m[f"p_win_{tag}"], 1e-6, 1 - 1e-6)))
        out[f"logit_sd_{tag}"] = float(lg.std())
    out["ll_market"] = float(M.bin_ll(y, m["p_mkt"]).mean())
    out["gap_srv"], out["gap_trn"] = out["ll_win_srv"] - out["ll_market"], out["ll_win_trn"] - out["ll_market"]
    out["corr_win"] = float(np.corrcoef(m["p_win_srv"], m["p_win_trn"])[0, 1])
    out["mean_abs_diff_win"] = float((m["p_win_srv"] - m["p_win_trn"]).abs().mean())
    out["ll_delta_srv_minus_trn"] = out["ll_win_srv"] - out["ll_win_trn"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--path-error-pred", default="data/local/modeling/training_pipeline/final_eval_v5/test2025_predictions.parquet")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    dump = root / "data/local/inference/drift_features"
    d25, d26 = load_dumps(dump, [2025]), load_dumps(dump, [2026])
    out: dict = {"rows_2025": int(len(d25)), "rows_2026": int(len(d26))}
    if len(d26) == 0:
        raise SystemExit("2026 年の抜き取りが無い")
    cols = [c for c in d26.columns if c not in META and (len(d25) == 0 or c in d25.columns)]
    m25 = month_of_day(d25).str[4:6].astype(int) if len(d25) else None
    m26 = month_of_day(d26).str[4:6].astype(int)
    cmp = {}
    if len(d25):
        cmp["y25_vs_y26"] = shift_table(d25, d26, cols)
        cmp["y25_jan_vs_jun"] = shift_table(d25[m25 <= 5], d25[m25 >= 6], cols)
        cmp["y25_jun_vs_y26_jun"] = shift_table(d25[m25 >= 6], d26[m26 >= 6], cols)
    cmp["y26_jan_vs_jun"] = shift_table(d26[m26 <= 5], d26[m26 >= 6], cols)
    n_a, n_b = int((m26 <= 5).sum()), int((m26 >= 6).sum())
    noise = [M.noise_psi(d26[c].to_numpy(dtype=float, na_value=np.nan), n_a, n_b) for c in cols[:60] if pd.api.types.is_numeric_dtype(d26[c])]
    out["noise_psi_median"] = float(np.nanmedian(noise)) if noise else float("nan")
    for k, t in cmp.items():
        top = t.sort_values("psi", ascending=False)
        out[k] = {"n_features": int(len(t)), "psi_ge_025": int((t["psi"] >= 0.25).sum()), "psi_ge_010": int((t["psi"] >= 0.10).sum()),
                  "miss_jump_ge_010": int(((t["miss_cur"] - t["miss_ref"]) >= 0.10).sum()),
                  "families": family_summary(t), "top_psi": top.head(40).round(4).to_dict("records"),
                  "missing_jump": t[(t["miss_cur"] - t["miss_ref"]).abs() >= 0.05].assign(d=lambda x: (x["miss_cur"] - x["miss_ref"]).abs())
                  .sort_values("d", ascending=False).head(40).round(4).to_dict("records")}
    out["monthly_missing"] = {"2025": monthly_missing(d25, cols) if len(d25) else {}, "2026": monthly_missing(d26, cols)}
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(d26[c])]
    out["adversarial"] = {"y26_jan_vs_jun": adversarial(d26[m26 <= 5], d26[m26 >= 6], num_cols)}
    if len(d25):
        out["adversarial"]["y25_vs_y26"] = adversarial(d25, d26, num_cols)
        out["adversarial"]["y25_jan_vs_jun"] = adversarial(d25[m25 <= 5], d25[m25 >= 6], num_cols)
        out["path_error_2025"] = path_error(root, d25, a.path_error_pred)
    # 予測の尖り（対数オッズ SD）と欠損の関係: 重要列の欠損が多い馬は平らか
    if "p_win" in d26.columns:
        lg = np.log(np.clip(d26["p_win"], 1e-6, 1 - 1e-6) / (1 - np.clip(d26["p_win"], 1e-6, 1 - 1e-6)))
        out["logit_sd_by_month_2026"] = {k: float(lg[month_of_day(d26) == k].std()) for k in sorted(month_of_day(d26).unique())}
        if len(d25) and "p_win" in d25.columns:
            l25 = np.log(np.clip(d25["p_win"], 1e-6, 1 - 1e-6) / (1 - np.clip(d25["p_win"], 1e-6, 1 - 1e-6)))
            out["logit_sd_by_month_2025"] = {k: float(l25[month_of_day(d25) == k].std()) for k in sorted(month_of_day(d25).unique())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k.startswith(("rows", "noise"))} | {"psi_ge_025": {k: out[k]["psi_ge_025"] for k in cmp},
                                                                                       "auc": {k: v["auc_mean"] for k, v in out["adversarial"].items()}}, default=float))


if __name__ == "__main__":
    main()
