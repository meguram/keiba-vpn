"""v3・v4 の最終バンドル（2020〜2024 年で学習）で、学習に使った行を再予測して train の精度（in-sample）を出す。

python -m src.scripts.data.accuracy_summary --version v3
python -m src.scripts.data.accuracy_summary --version v4
出力: data/local/modeling/training_pipeline/accuracy_summary/train_<version>.json（勝率・連対・複勝の LogLoss と AUC、行数）
バンドルと学習表は読み取りのみ。test 2025 の予測（final_eval_*/test2025_predictions.parquet）の AUC も同じ式で出す。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "data/local/modeling/training_pipeline"
OUT = OLD / "accuracy_summary"
KEY = ["race_id", "horse_id"]
TARGETS = (("win", 1), ("top2", 2), ("top3", 3))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(finish: np.ndarray, pred: dict[str, np.ndarray]) -> dict:
    out = {}
    for t, k in TARGETS:
        y = (finish <= k).astype(float)
        p = np.clip(np.asarray(pred[t], dtype="float64"), 1e-9, 1 - 1e-9)
        out[t] = {"logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "auc": float(roc_auc_score(y, p))}
    return out


def _train_index(meta: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(meta["date"])
    return np.flatnonzero(dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10))


def matrices_v3():
    from src.pipeline.models import tp05_data
    from src.scripts.data.fit_final_v3 import WORK

    D = tp05_data.load_dataset(WORK)
    df, meta = D["df"], D["meta"]
    EP, _EXTRAS, _PED = tp05_data.load_extras_ped(df, WORK)
    TL = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    TLm = df[KEY].merge(TL, on=KEY, how="left")
    TLm.index = df.index
    TLX = TLm[[c for c in TLm.columns if c not in KEY]].add_prefix("tl_")
    X_all = pd.concat([df[D["FEATS"]["base+S+C1"]], EP, TLX], axis=1)
    return X_all.loc[:, ~X_all.columns.duplicated()], meta, OLD / "final_v3/bundle_2024", OLD / "final_eval_v3/test2025_predictions.parquet"


def matrices_v4():
    from src.scripts.data import fit_final_v4 as f4

    _df, meta, mats, _A, _A_feat = f4.build_matrices()
    names = [m[0] for m in f4.MEMBERS]
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    return X_all.loc[:, ~X_all.columns.duplicated()], meta, OLD / "final_v4/bundle_v4_2024", OLD / "final_eval_v4/test2025_predictions.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v3", "v4"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    from src.pipeline.models import ensemble_bundle as eb

    t0 = time.time()
    X_all, meta, bundle_dir, test_pred = (matrices_v3 if a.version == "v3" else matrices_v4)()
    _log(f"特徴量行列 {X_all.shape}（{time.time() - t0:.0f}s）")
    bundle = eb.EnsembleBundle.load(bundle_dir)
    fin = meta["finish_position"].to_numpy().astype(float)
    race = meta["race_id"].to_numpy()
    tr = _train_index(meta)
    tr = tr[np.isfinite(fin[tr])]
    _log(f"train 再予測 {len(tr):,} 行 / レース {len(np.unique(race[tr])):,}")
    pred = bundle.predict(X_all.iloc[tr], race[tr])
    train = metrics(fin[tr], {t: pred[f"p_{t}"].to_numpy() for t, _ in TARGETS})
    P = pd.read_parquet(test_pred)
    if "finish_position" not in P.columns:
        P = P.merge(meta[KEY + ["finish_position"]], on=KEY, how="left")
    test = metrics(P["finish_position"].to_numpy().astype(float), {t: P[f"p_{t}"].to_numpy() for t, _ in TARGETS})
    rep = {"version": a.version, "bundle": str(bundle_dir.relative_to(ROOT)), "train_rows": int(len(tr)), "train_races": int(len(np.unique(race[tr]))),
           "train": train, "test_from_saved_predictions": test, "test_rows": int(len(P)),
           "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"train_{a.version}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("train: " + json.dumps(train, ensure_ascii=False))
    _log("test(保存済み予測): " + json.dumps(test, ensure_ascii=False))


if __name__ == "__main__":
    main()
