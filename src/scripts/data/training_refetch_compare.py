"""調教ページを取り直す前後で、同じ日・同じ行の V5 予測がどう変わるかを比べる（欠損あり vs 取り直し後）。

  python -m src.scripts.data.training_refetch_compare --before /home/jovyan/work/tr_check/before --after /home/jovyan/work/tr_check/after \
      --year 2026 --out /home/jovyan/work/tr_check/compare.json

drift_dump_features（--days で同じ開催日）で作った 2 つの抜き取り（取り直しの前・後）を、(race_id, horse_id) で突き合わせる。
指標: 勝率 LogLoss、市場との差（モデル−市場、正が悪化）、対数オッズの SD の比（モデル/市場。尖り）、調教系の列の欠損率。
レース単位ブートストラップで前後の LogLoss 差の区間も出す。結果の確認だけで、モデルの選択や閾値には使わない（2026 年はテストセット②）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M
from src.scripts.data.drift_monitor import load_dumps


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), M.EPS, 1 - M.EPS)
    return np.log(p / (1 - p))


def training_missing_rate(d: pd.DataFrame) -> float:
    cols = [c for c in d.columns if c.startswith(("tl_", "wc_"))]
    return float(d[cols].isna().to_numpy().mean()) if cols else float("nan")


def metrics(y: np.ndarray, p: np.ndarray, pm: np.ndarray) -> dict:
    ll, lm = M.bin_ll(y, p), M.bin_ll(y, pm)
    return {"ll_win": float(ll.mean()), "gap_vs_market": float((ll - lm).mean()), "logit_sd": float(_logit(p).std()),
            "logit_sd_ratio_vs_market": float(_logit(p).std() / _logit(pm).std())}


def paired_ll_delta(race: np.ndarray, y: np.ndarray, p_before: np.ndarray, p_after: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """after − before の勝率 LogLoss 差（負＝取り直し後が良い）。レース単位ブートストラップ。"""
    dd = M.bin_ll(y, p_after) - M.bin_ll(y, p_before)
    g = pd.DataFrame({"race": race, "d": dd}).groupby("race")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return {"delta_ll": float(dd.mean()), "lo": float(np.quantile(est, 0.025)), "hi": float(np.quantile(est, 0.975)), "races": int(len(g))}


def compare(before: pd.DataFrame, after: pd.DataFrame, res: pd.DataFrame, mk: pd.DataFrame) -> dict:
    keys = ["race_id", "horse_id"]
    for d in (before, after):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    both = before[keys + ["p_win"]].merge(after[keys + ["p_win"]], on=keys, suffixes=("_before", "_after"))
    meta = both[keys].merge(res[["race_id", "horse_id", "horse_number", "finish_position"]], on=keys, how="left").merge(mk, on=["race_id", "horse_number"], how="left")
    keep = (meta["finish_position"].notna() & meta["p_mkt"].notna() & (meta["finish_position"] >= 1)).to_numpy()
    both, meta = both[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
    y, pm = (meta["finish_position"] == 1).to_numpy(float), meta["p_mkt"].to_numpy(float)
    pb, pa = both["p_win_before"].to_numpy(float), both["p_win_after"].to_numpy(float)
    race = both["race_id"].to_numpy()
    bm = before.merge(both[keys], on=keys)
    am = after.merge(both[keys], on=keys)
    return {"rows": int(len(both)), "races": int(both["race_id"].nunique()), "market_ll": float(M.bin_ll(y, pm).mean()),
            "before": {**metrics(y, pb, pm), "training_cols_missing_rate": training_missing_rate(bm)},
            "after": {**metrics(y, pa, pm), "training_cols_missing_rate": training_missing_rate(am)},
            "paired": paired_ll_delta(race, y, pb, pa), "corr_before_after": float(np.corrcoef(pb, pa)[0, 1])}


def main() -> None:
    from src.scripts.data.drift_data import load_results_year, market_probs

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    tables = root / "data/page_reference/tables"
    out = compare(load_dumps(Path(a.before), [a.year]), load_dumps(Path(a.after), [a.year]), load_results_year(tables, a.year), market_probs(tables, a.year)[0])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
