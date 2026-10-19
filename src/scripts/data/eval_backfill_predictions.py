"""バックフィル（または shadow）の予測を、確定している着順と突き合わせて LogLoss と較正の目安を出す。

  python -m src.scripts.data.eval_backfill_predictions --kind backfill --label bundle_v4_2025 --out /tmp/eval_backfill.json
  着順は race_result_on_time（速報）→ race_result の順に引く。v4 の test 2025（勝率 LogLoss 0.20740）と大きく外れていないかを見る目安。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(preds: pd.DataFrame, fin: pd.DataFrame, bins: int = 5) -> dict:
    """preds: race_id・horse_id・p_win・p_top2・p_top3、fin: race_id・horse_id・finish_position。"""
    m = preds.merge(fin, on=["race_id", "horse_id"], how="inner")
    m = m[m["finish_position"] > 0]
    out: dict = {"races": int(m["race_id"].nunique()), "rows": int(len(m))}
    if m.empty:
        return out
    for key, col, k in (("win", "p_win", 1), ("top2", "p_top2", 2), ("top3", "p_top3", 3)):
        y = (m["finish_position"] <= k).to_numpy(float)
        p = m[col].to_numpy(float)
        base = float(y.mean())
        qs = np.quantile(p, np.linspace(0, 1, bins + 1))
        qs[-1] += 1e-9
        idx = np.clip(np.digitize(p, qs[1:-1]), 0, bins - 1)
        out[key] = {
            "logloss": logloss(p, y), "uniform_logloss": logloss(np.full_like(p, base), y), "mean_pred": float(p.mean()), "rate": base,
            "calibration": [{"mean_pred": float(p[idx == b].mean()), "rate": float(y[idx == b].mean()), "n": int((idx == b).sum())} for b in range(bins) if (idx == b).any()],
        }
    top = m.sort_values(["race_id", "p_win"], ascending=[True, False]).groupby("race_id").head(1)
    out["top1_hit"] = float((top["finish_position"] == 1).mean())
    return out


def load_fin(storage, race_ids: list[str]) -> pd.DataFrame:
    rows = []
    for rid in race_ids:
        for cat in ("race_result_on_time", "race_result"):
            d = storage.load(cat, rid)
            es = (d or {}).get("entries") or (d or {}).get("results") or []
            es = [e for e in es if isinstance(e, dict) and (e.get("finish_position") or 0) > 0]
            if es:
                rows += [{"race_id": rid, "horse_id": str(e.get("horse_id")), "finish_position": int(e["finish_position"])} for e in es]
                break
    return pd.DataFrame(rows, columns=["race_id", "horse_id", "finish_position"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["backfill", "shadow"], default="backfill")
    ap.add_argument("--label", default="bundle_v4_2025")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.api.local_predictions import inference_dir
    from src.scraper.storage import HybridStorage

    inf = inference_dir()
    files = sorted((inf / "predictions_backfill" / a.label).glob("*/*.parquet")) if a.kind == "backfill" else sorted((inf / "predictions_v3").glob("*/*.parquet"))
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"])
    preds["race_id"], preds["horse_id"] = preds["race_id"].astype(str), preds["horse_id"].astype(str)
    fin = load_fin(HybridStorage(), sorted(preds["race_id"].unique()))
    res = evaluate(preds, fin)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
