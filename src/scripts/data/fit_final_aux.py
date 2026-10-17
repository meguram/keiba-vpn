"""補助タスクつき NN（tp-25c の mtl_C）の推論用の最終モデルを作る（tp-25e）。

python -m src.scripts.data.fit_final_aux            # 2020〜2025 の全行で学習 → data/local/modeling/training_pipeline/final_aux/model_s{seed}.pkl
  - 2025 年の行の補助予測（2024 年までで学習したモデルの OOF）も final_aux/aux_oof_2025.parquet に出す（最終アンサンブルの学習行に使う）
  - 学習時の OOF（2021〜2024）と最終モデルの予測の相関を final_aux/parity.json に出す（最終モデルには 2021〜2024 も入っているので、
    相関は『学習行での再現』の確認であり、汎化の確認ではない）
環境変数: SEEDS=0,1,2,3,4  THREADS=3
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import serving_aux as sa
from src.scripts.data import tp25c_eval as c

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline/final_aux"
SEEDS = [int(s) for s in (os.environ.get("SEEDS") or "0,1,2,3,4").split(",")]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    import torch

    from src.pipeline.models.tp25c_mtl import mtl_fit_predict

    torch.set_num_threads(int(os.environ.get("THREADS") or 3))
    OUT.mkdir(parents=True, exist_ok=True)
    df, base, _sx = c.load_frame()
    assert [x for x in c.AUX_C] == sa.AUX_NAMES
    X = df[base]
    A = df[c.AUX_C].to_numpy(dtype="float64")
    race, fin, dates, years = df["race_id"].to_numpy(), df["finish_position"].to_numpy(), df["date"], df["year"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    names = ["mtl_" + x for x in sa.AUX_NAMES]

    # (1) 2025 年の行の OOF（2024 年末までで学習）
    gap = pd.Timedelta(days=10)
    tr = np.flatnonzero((years >= 2020) & (dates < pd.Timestamp("2025-01-01") - gap))
    va = np.flatnonzero(years == 2025)
    acc = np.zeros((len(va), len(names)))
    for s in SEEDS[:3]:
        o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[va], race[va], seed=s, aux_w=0.5)
        acc += o["aux"] / len(SEEDS[:3])
        _log(f"OOF 2025 seed{s} done")
    t = pd.DataFrame(acc, columns=names, index=df.index[va]).astype("float32")
    t = pd.concat([t, sa.relative(t, race[va])], axis=1)
    t.insert(0, "horse_id", df["horse_id"].to_numpy()[va])
    t.insert(0, "race_id", race[va])
    t.to_parquet(OUT / "aux_oof_2025.parquet", index=False)

    # (2) 最終モデル（2020〜2025 の全行）
    allr = np.flatnonzero(np.isfinite(fin.astype(float)) & (years <= 2025))
    for s in SEEDS:
        p = OUT / f"model_s{s}.pkl"
        if p.exists():
            continue
        o = mtl_fit_predict(X.iloc[allr], {q: x[allr] for q, x in Y.items()}, A[allr], race[allr], X.iloc[va], race[va], seed=s, aux_w=0.5, want_model=True)
        p.write_bytes(pickle.dumps(o["model"]))
        _log(f"final model seed{s} saved ({len(allr)} rows)")
    (OUT / "meta.pkl").write_bytes(pickle.dumps({"base_cols": base, "aux": sa.AUX_NAMES, "seeds": SEEDS, "train_rows": int(len(allr)), "trained_through": "2025-12-31"}))

    # (3) 学習時の OOF と、読み込んだ最終モデル（serving_aux）の予測の相関
    am = sa.AuxModels.load(OUT)
    oof = np.load(c.C_OUT / "oof_aux_mtl_C.npy") if hasattr(c, "C_OUT") else np.load(c.OUT / "oof_aux_mtl_C.npy")
    rep: dict = {"final_models": len(am.models), "by_year": {}}
    for Y_ in (2022, 2023, 2024):
        m = np.flatnonzero(years == Y_)
        pr = am.predict(X.iloc[m], race[m])[names].to_numpy()
        rep["by_year"][str(Y_)] = {n: float(np.corrcoef(pr[:, j], oof[m, j])[0, 1]) for j, n in enumerate(names)}
    m = va
    pr = am.predict(X.iloc[m], race[m])[names].to_numpy()
    rep["by_year"]["2025"] = {n: float(np.corrcoef(pr[:, j], acc[:, j])[0, 1]) for j, n in enumerate(names)}
    rep["note"] = "最終モデルは 2021〜2025 の行も学習に含む（in-sample）。相関は再現の確認で、汎化の確認ではない。"
    (OUT / "parity.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("parity: " + json.dumps({k: round(float(np.mean(list(v.values()))), 4) for k, v in rep["by_year"].items()}))


if __name__ == "__main__":
    main()
