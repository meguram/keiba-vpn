"""ドリフト診断: 入力の一部が欠けると予測がどれだけ平らになり、市場との差がどれだけ開くかを、同じ行の上で測る（マスク実験）。

  python -m src.scripts.data.drift_mask_experiment --year 2025 --bundle data/local/modeling/training_pipeline/final_v5/bundle_v5_2024 \
      --out /home/jovyan/work/drift_out/mask_2025.json [--months 1-12]

drift_dump_features で抜き取った『推論経路の特徴量』の行に対して、列の群を欠損にして学習済みバンドルで予測し直し、
LogLoss・市場との差・対数オッズの SD（尖り）が、欠損なしの予測に対してどう変わるかを出す。同じ行・同じバンドルの比較なので、レースの難しさの影響は入らない。
群: 調教(tl_・wc_＝horse_training 由来)、レース追い切り評価・厩舎コメント(oikiri_*・has_comment・comment_*＝race_oikiri/コメント由来)、両方。
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


def groups_of(cols: list[str]) -> dict[str, list[str]]:
    train = [c for c in cols if c.startswith(("tl_", "wc_"))]
    extras = [c for c in cols if c.startswith(("oikiri_", "comment_")) or c == "has_comment"]
    return {"training_tl_wc": train, "oikiri_comment": extras, "both": sorted(set(train) | set(extras))}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), M.EPS, 1 - M.EPS)
    return np.log(p / (1 - p))


def evaluate(d: pd.DataFrame, p: np.ndarray, y: np.ndarray, pm: np.ndarray) -> dict:
    lm = _logit(p)
    return {"ll_win": float(M.bin_ll(y, p).mean()), "gap": float((M.bin_ll(y, p) - M.bin_ll(y, pm)).mean()), "logit_sd": float(lm.std()),
            "logit_sd_ratio_vs_market": float(lm.std() / _logit(pm).std())}


def main() -> None:
    from src.pipeline.models.ensemble_bundle import EnsembleBundle
    from src.scripts.data.drift_data import load_results_year, market_probs

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--months", default="1-12")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    lo, hi = (int(x) for x in a.months.split("-")) if "-" in a.months else (int(a.months), int(a.months))
    d = load_dumps(root / "data/local/inference/drift_features", [a.year])
    d = d[d["day"].astype(str).str[4:6].astype(int).between(lo, hi)].reset_index(drop=True)
    b = EnsembleBundle.load(Path(a.bundle))
    cols = list(dict.fromkeys(c for m in b.members for c in m.feature_cols))
    tables = root / "data/page_reference/tables"
    res = load_results_year(tables, a.year)
    mk, _ = market_probs(tables, a.year)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    meta = d[["race_id", "horse_id"]].merge(res[["race_id", "horse_id", "horse_number", "finish_position"]], on=["race_id", "horse_id"], how="left").merge(
        mk, on=["race_id", "horse_number"], how="left")
    keep = meta["finish_position"].notna() & meta["p_mkt"].notna() & (meta["finish_position"] >= 1)
    d, meta = d[keep.to_numpy()].reset_index(drop=True), meta[keep.to_numpy()].reset_index(drop=True)
    y, pm = (meta["finish_position"] == 1).to_numpy(float), meta["p_mkt"].to_numpy(float)
    race = d["race_id"].to_numpy()
    out: dict = {"year": a.year, "rows": int(len(d)), "races": int(d["race_id"].nunique()), "bundle": Path(a.bundle).name,
                 "missing_rate_actual": {g: float(d[c].isna().to_numpy().mean()) for g, c in groups_of(cols).items() if c}}
    p0 = b.predict(d[cols], race)["p_win"].to_numpy()
    out["full"] = evaluate(d, p0, y, pm)
    out["market_ll"] = float(M.bin_ll(y, pm).mean())
    out["stored_vs_recomputed_corr"] = float(np.corrcoef(p0, d["p_win"])[0, 1]) if "p_win" in d else None
    out["masked"] = {}
    for g, cs in groups_of(cols).items():
        if not cs:
            continue
        X = d[cols].copy()
        for c in cs:
            X[c] = np.nan
        p = b.predict(X, race)["p_win"].to_numpy()
        e = evaluate(d, p, y, pm)
        dd = M.bin_ll(y, p) - M.bin_ll(y, p0)
        gr = pd.DataFrame({"race_id": race, "d": dd}).groupby("race_id")["d"].agg(["sum", "size"])
        rng = np.random.default_rng(0)
        s, n = gr["sum"].to_numpy(), gr["size"].to_numpy()
        idx = rng.integers(0, len(gr), (500, len(gr)))
        est = s[idx].sum(1) / n[idx].sum(1)
        e.update({"n_cols_masked": len(cs), "delta_ll_vs_full": float(dd.mean()), "delta_lo": float(np.quantile(est, 0.025)), "delta_hi": float(np.quantile(est, 0.975)),
                  "corr_with_full": float(np.corrcoef(p, p0)[0, 1])})
        out["masked"][g] = e
    if "p_win_d" in d.columns and d["p_win_d"].notna().all():                       # --drop-training で抜き取った『調教記録を一部落とした』実際に近い再現
        pd_ = d["p_win_d"].to_numpy(float)
        e = evaluate(d, pd_, y, pm)
        dd = M.bin_ll(y, pd_) - M.bin_ll(y, p0)
        gr = pd.DataFrame({"race_id": race, "d": dd}).groupby("race_id")["d"].agg(["sum", "size"])
        rng = np.random.default_rng(1)
        s_, n_ = gr["sum"].to_numpy(), gr["size"].to_numpy()
        idx = rng.integers(0, len(gr), (500, len(gr)))
        est = s_[idx].sum(1) / n_[idx].sum(1)
        e.update({"delta_ll_vs_full": float(dd.mean()), "delta_lo": float(np.quantile(est, 0.025)), "delta_hi": float(np.quantile(est, 0.975)), "corr_with_full": float(np.corrcoef(pd_, p0)[0, 1])})
        out["partial_drop_95"] = e
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"rows": out["rows"], "full": out["full"], "masked": {k: {x: round(v[x], 5) for x in ("gap", "logit_sd_ratio_vs_market", "delta_ll_vs_full")} for k, v in out["masked"].items()},
                      "partial_drop_95": out.get("partial_drop_95")}, default=float))


if __name__ == "__main__":
    main()
