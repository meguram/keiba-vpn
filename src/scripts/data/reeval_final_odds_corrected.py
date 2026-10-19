"""最終オッズ予測の bundle 比較を、2025 年の単勝オッズ（指数列の誤読を補正）で評価し直す。学習はしない・bundle は変更しない。

2025-05〜12 の評価ラベル target_win_odds は race_result の odds（指数列）だった。scraper.odds_fix の補正表で正しい確定オッズに直し、
補正前後で現行 bundle（candidate2）と旧 bundle（candidate）の優劣が変わるかを見る。
特徴量は odds を使わない（過去走の人気は horse_result 由来で無傷）ので、キャッシュ済みの評価行のラベルだけを差し替えて評価する。

python -m src.scripts.data.reeval_final_odds_corrected --new models/final_odds_bundle.json --old models/final_odds_bundle.candidate.json --out /tmp/odds2025_fix/reeval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scraper import odds_fix
from src.scripts.data.eval_final_odds_bundles import HEADS, metrics, predict_bundle

CACHE = Path("data/local/modeling/final_odds_eval_cache/eval_overlay_2025_2026.parquet")
CACHE_FIXED = Path("data/local/modeling/final_odds_eval_cache/eval_overlay_2025_2026_oddsfix.parquet")


def correct_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """補正表で単勝ラベルを直し、直せない壊れたレースは落とす。複勝ラベルが旧単勝からの推定値だった行は推定し直す。"""
    out = df.copy()
    lk = odds_fix.load_lookup()
    old = out["target_win_odds"].to_numpy(dtype=float)
    key = list(zip(out["race_id"].astype(str), out["horse_number"].astype(int)))
    new = np.array([lk[k][0] if k in lk else np.nan for k in key])
    hit = ~np.isnan(new)
    # 複勝ラベルが「旧単勝×係数」で埋められていた行（race_odds に複勝が無かった行）は新しい単勝で埋め直す
    pm_old, px_old = out["target_place_min"].to_numpy(float), out["target_place_max"].to_numpy(float)
    pm_est = np.isclose(pm_old, np.maximum(old * 0.22, 1.0)) & np.isclose(px_old, np.maximum(old * 0.52, pm_old + 0.1))
    out.loc[hit, "target_win_odds"] = new[hit]
    out.loc[hit, "target_log_win_odds"] = np.log(new[hit])
    redo = hit & pm_est
    if redo.any():
        pm = np.maximum(new[redo] * 0.22, 1.0)
        out.loc[redo, "target_place_min"] = pm
        out.loc[redo, "target_place_max"] = np.maximum(new[redo] * 0.52, pm + 0.1)
        out.loc[redo, "target_log_place_min"] = np.log(out.loc[redo, "target_place_min"])
        out.loc[redo, "target_log_place_max"] = np.log(out.loc[redo, "target_place_max"])
    ok = out.groupby("race_id")["target_win_odds"].apply(lambda x: odds_fix.is_plausible_win_odds(x.tolist()))
    drop = set(ok.index[~ok])
    out = out[~out["race_id"].isin(drop)].reset_index(drop=True)
    return out, {"rows": int(len(df)), "rows_corrected": int(hit.sum()), "place_label_reestimated": int(redo.sum()), "races_dropped": len(drop)}


def boot_diff(df: pd.DataFrame, p_new: np.ndarray, p_old: np.ndarray, col: str, n: int = 400, seed: int = 0) -> dict:
    """log 誤差(絶対値)の平均を new − old でレースごとにブートストラップ。負なら new の方が良い。"""
    a = df[col].astype(float).to_numpy()
    la = np.log(np.maximum(a, 1.0))
    d = np.abs(np.log(p_new) - la) - np.abs(np.log(p_old) - la)
    g = pd.DataFrame({"r": df["race_id"].astype(str).to_numpy(), "d": d}).groupby("r")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": round(float(s.sum() / c.sum()), 5), "lo": round(float(np.quantile(bs, 0.025)), 5), "hi": round(float(np.quantile(bs, 0.975)), 5)}


def evaluate(df: pd.DataFrame, bundles: dict[str, Path]) -> dict:
    preds = {k: predict_bundle(p, df) for k, p in bundles.items()}
    yr = df["race_id"].astype(str).str[:4]
    res: dict = {}
    for name, mask in (("2025", yr == "2025"), ("2026", yr == "2026"), ("2025+2026", yr.isin(["2025", "2026"]))):
        sub = df[mask.to_numpy()]
        res[name] = {"rows": int(len(sub)), "races": int(sub["race_id"].nunique()), "bundles": {}, "new_minus_old_log_mae": {}}
        for k, p in preds.items():
            res[name]["bundles"][k] = metrics(sub, {h: v[mask.to_numpy()] if h != "_n_missing_cols" else v for h, v in p.items()})
        if "new" in preds and "old" in preds:
            for h, col in (("win", "target_log_win_odds"), ("place_min", "target_log_place_min"), ("place_max", "target_log_place_max")):
                tcol = {"win": "target_win_odds", "place_min": "target_place_min", "place_max": "target_place_max"}[h]
                res[name]["new_minus_old_log_mae"][h] = boot_diff(sub, preds["new"][h][mask.to_numpy()], preds["old"][h][mask.to_numpy()], tcol)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="models/final_odds_bundle.json")
    ap.add_argument("--old", default="models/final_odds_bundle.candidate.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = pd.read_parquet(CACHE)
    fixed, info = correct_labels(raw)
    CACHE_FIXED.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_parquet(CACHE_FIXED, index=False)
    bundles = {"new": Path(a.new), "old": Path(a.old)}
    rep = {"label_correction": info, "before_correction": evaluate(raw, bundles), "after_correction": evaluate(fixed, bundles)}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for tag in ("before_correction", "after_correction"):
        print("=====", tag)
        for per, r in rep[tag].items():
            line = {h: {b: r["bundles"][b][h]["mape_pct"] for b in r["bundles"]} for h in ("win", "place_min", "place_max")}
            sp = {h: {b: r["bundles"][b][h]["spearman_within_race"] for b in r["bundles"]} for h in ("win",)}
            print(per, r["rows"], "MAPE%", line, "spearman(win)", sp, "Δlogmae(new-old)", r["new_minus_old_log_mae"])
    print(info)


if __name__ == "__main__":
    main()
