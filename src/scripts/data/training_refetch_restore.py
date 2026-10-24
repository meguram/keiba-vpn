"""調教ページを取り直したあとの記録で、抜き取り済みの行の調教系の列（tl_・wc_）だけを作り直して V5 を再予測し、取り直し前の予測と比べる。

  python -m src.scripts.data.training_refetch_restore --days 20260830 20260906 --bundle <bundle_v5_2025> --out compare.json

drift_dump_features で抜き取った特徴量（取り直し前＝調教が欠けた状態）に対して、保存済みの horse_training から tl_・wc_ を作り直し（`drop_training_predict` の frac=0 と同じ経路）、
同じバンドルで予測する。特徴量全体は作り直さない（軽い）。比較は training_refetch_compare.compare（勝率 LogLoss・市場との差・尖り・欠損率・レース単位ブートストラップ）。
2026 年はテストセット②。結果の確認だけで、選択や閾値には使わない。読み取りのみ（GCS/DB へ書かない）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def restored_frame(v3, X: pd.DataFrame, storage) -> pd.DataFrame:
    """X の tl_・wc_ を、現在保存されている horse_training から作り直した特徴量にする（予測の列は p_win を付け直す）。"""
    from src.pipeline.features import serving_stage_rel as ssr
    from src.pipeline.features import serving_v5 as sv5

    keys = ["race_id", "horse_id"]
    recs = v3._training_records(X["horse_id"].astype(str).unique().tolist(), storage, 4)
    base = X[keys + ["day"]].copy()
    base["date"] = pd.to_datetime(base["day"].astype(str), format="%Y%m%d")
    tl_cols = [c for c in v3.feature_cols if c.startswith(ssr.TL_PREFIX)]
    wc_cols = [c for c in v3.feature_cols if c.startswith("wc_")]
    X2 = X.copy()
    tl = ssr.tl_columns_for_races(base[keys + ["date"]], recs, v3.tl_ref)
    X2[tl_cols] = base[keys].merge(tl, on=keys, how="left").drop(columns=keys).reindex(columns=tl_cols).to_numpy()
    if wc_cols:
        wc = sv5.cy_columns_for_races(base[keys + ["date"]], recs, v3.tl_ref)
        X2[wc_cols] = base[keys].merge(wc, on=keys, how="left").drop(columns=keys).reindex(columns=wc_cols).to_numpy()
    return X2


def main() -> None:
    from src.pipeline.inference import v3_serving as vs
    from src.scraper.storage import HybridStorage
    from src.scripts.data import training_refetch_compare as T
    from src.scripts.data.drift_data import load_results_year, market_probs
    from src.scripts.data.drift_monitor import load_dumps
    from src.utils.project_env import load_project_dotenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dump-dir", default="data/local/inference/drift_features")
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--bundle", default="", help="省略時は v5 の既定（KEIBA_V5_BUNDLE）")
    ap.add_argument("--stage", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    load_project_dotenv()
    root = Path(a.root).resolve()
    d = load_dumps(root / a.dump_dir, [a.year])
    d = d[d["day"].astype(str).isin(a.days)].reset_index(drop=True)
    pth = vs.v5_paths()
    v3 = vs.V3Serving(Path(a.bundle) if a.bundle else pth["bundle"], Path(a.stage) if a.stage else pth["stage"], tl_reference=pth["tl_reference"])
    v3.warm()
    before = d.copy()
    before["p_win"] = v3.predict(d)["p_win"].to_numpy()                          # 取り直し前の特徴量を、同じバンドルで予測し直す（保存済みの p_win との差は推論の再現誤差）
    after = restored_frame(v3, d, HybridStorage())
    after["p_win"] = v3.predict(after)["p_win"].to_numpy()
    tables = root / "data/page_reference/tables"
    out = T.compare(before, after, load_results_year(tables, a.year), market_probs(tables, a.year)[0])
    out["days"] = a.days
    out["stored_vs_recomputed_before_corr"] = float(pd.Series(before["p_win"]).corr(pd.Series(d["p_win"]))) if "p_win" in d else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
