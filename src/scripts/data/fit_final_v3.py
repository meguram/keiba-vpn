"""アンサンブル v3 の最終モデルを学習して、推論用に保存する（tp-07 の一部）。

python -m src.scripts.data.fit_final_v3 --fit-end 2024     # 検証用: 2020〜2024 で学習 → 2025 を予測して、tp-06c で評価済みの予測と比べる
python -m src.scripts.data.fit_final_v3 --fit-end 2025     # 本番用: 2020〜2025 の全行で学習（2026 年は使わない）
出力: data/local/modeling/training_pipeline/final_v3/bundle_{fit_end}/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.models import ensemble_bundle as eb
from src.pipeline.models import tp05_data, zoo
from src.pipeline.models.compute import detect_compute

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"


def build_members(D, EP, EXTRAS, TLX, best):
    base_cols = list(D["FEATS"]["base+S+C1"])
    tl = list(TLX.columns)
    lgbp, xgbp, nnp = zoo.tuned_params("lgb", best["lgb"]["params"]), zoo.tuned_params("xgb", best["xgb"]["params"]), zoo.tuned_params("nn", best["nn"]["params"])
    ep_cols, ex_cols = base_cols + list(EP.columns) + tl, base_cols + list(EXTRAS) + tl
    plain = base_cols + tl
    return [
        eb.MemberSpec("lgb_extras_ped", "lgb_binary", lgbp, ep_cols),
        eb.MemberSpec("xgb_extras_ped", "xgb_binary", xgbp, ep_cols),
        eb.MemberSpec("nn_extras_ped", "nn", nnp, ep_cols),
        eb.MemberSpec("lgb_extras", "lgb_binary", lgbp, ex_cols),
        eb.MemberSpec("lgb_softmax", "lgb_softmax", {}, plain),
        eb.MemberSpec("linear", "linear", {"epochs": 14}, plain),
        eb.MemberSpec("nn_s", "nn", {"hidden": (128, 64), "epochs": 14}, plain),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    D = tp05_data.load_dataset(WORK)
    df, meta = D["df"], D["meta"]
    EP, EXTRAS, _PED = tp05_data.load_extras_ped(df, WORK)
    TL = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    TLm = df[["race_id", "horse_id"]].merge(TL, on=["race_id", "horse_id"], how="left")
    TLm.index = df.index
    TLX = TLm[[c for c in TLm.columns if c not in ("race_id", "horse_id")]].add_prefix("tl_")
    X_all = pd.concat([df[D["FEATS"]["base+S+C1"]], EP, TLX], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    best = json.loads((WORK / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    members = build_members(D, EP, EXTRAS, TLX, best)
    spec = json.loads((WORK / "ensemble_v3/ensemble_spec.json").read_text(encoding="utf-8"))
    fin = meta["finish_position"].to_numpy()
    dates = pd.to_datetime(meta["date"])
    if a.fit_end == 2024:
        tr = np.flatnonzero(dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10))
    else:
        tr = np.flatnonzero((meta["year"].to_numpy() <= 2025) & np.isfinite(fin.astype(float)))
    race = meta["race_id"].to_numpy()
    y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    ytr = {k: v[tr] for k, v in y.items()}
    # lgb_softmax の top2/top3 は、OOF で当てはめた γ の割引 Harville で導く
    o = pd.read_parquet(WORK / "oof/model_oof/lgb_softmax_tl3.parquet").merge(meta[["race_id", "horse_id", "finish_position"]], on=["race_id", "horse_id"])
    gam = (zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 2), zoo.fit_gamma(o["race_id"].to_numpy(), o["p_win"].to_numpy(), o["finish_position"].to_numpy(), 3))
    bundle = eb.EnsembleBundle(members, spec, {"lgb_softmax": gam})
    threads = detect_compute(reserve_cores=6).threads
    print(f"学習 {len(tr):,} 行（{dates.iloc[tr].min():%Y-%m-%d}〜{dates.iloc[tr].max():%Y-%m-%d}） / 列 {X_all.shape[1]} / seed {a.seeds} / threads {threads}", flush=True)
    bundle.fit(X_all.iloc[tr], ytr, race[tr], threads, list(range(a.seeds)))
    print(f"[{time.time() - t0:.0f}s] 学習完了", flush=True)
    out = WORK / f"final_v3/bundle_{a.fit_end}"
    bundle.save(out)
    print("保存:", out, flush=True)
    if a.fit_end == 2024:
        va = np.flatnonzero(meta["year"].to_numpy() == 2025)
        pred = bundle.predict(X_all.iloc[va], race[va])
        ref = pd.read_parquet(WORK / "final_eval_v3/test2025_predictions.parquet")[["race_id", "horse_id", "p_win", "p_top2", "p_top3"]]
        m = pd.DataFrame({"race_id": race[va], "horse_id": meta["horse_id"].to_numpy()[va]}).join(pred.reset_index(drop=True)).merge(ref, on=["race_id", "horse_id"], suffixes=("", "_ref"))
        rep = {"rows": len(m)}
        for t in ("win", "top2", "top3"):
            d = np.abs(m[f"p_{t}"] - m[f"p_{t}_ref"])
            rep[t] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "corr": float(np.corrcoef(m[f"p_{t}"], m[f"p_{t}_ref"])[0, 1])}
        (out / "parity_vs_tp06c.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("tp-06c の予測との一致:", json.dumps(rep, ensure_ascii=False), flush=True)
    (out / "fit_info.json").write_text(json.dumps({"fit_end": a.fit_end, "train_rows": int(len(tr)), "seeds": a.seeds, "n_features": int(X_all.shape[1]), "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
