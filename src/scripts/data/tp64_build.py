"""tp-64: 過去走の展開利と、次走とのずれの特徴量表を作る（ホストで heavy_run 経由）。

python -m src.scripts.data.tp64_build            # data/local/modeling/tp64/{pb_features,pcell}.parquet と validity.json
入力: data/local/modeling/tp61/{horse,race_cls,pre}.parquet（tp-61 の実測・事前の表）
- 今走の展開セル（ペース3×隊列3）の事前確率 pcell は、pre.parquet のレース前の量から年ごと leave-one-year-out の
  9 クラス LightGBM で作る（tp-55 のラップ予想と同じ作法。評価年のラベルは評価年の予測に入らない）。
- 展開利の係数表は『その走の年より前』の結果だけ（tp64_pace_benefit.benefit_table）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp64_pace_benefit as pb

ROOT = Path(__file__).resolve().parents[3]
T61 = Path(os.environ.get("TP64_T61") or ROOT / "data/local/modeling/tp61")
OUT = Path(os.environ.get("TP64_OUT") or ROOT / "data/local/modeling/tp64")
THREADS = int(os.environ.get("TP64_THREADS", "3"))
YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
DROP = {"race_id", "date", "year", "venue", "surface", "going_group", "grade", "race_class", "band", "pace_cls", "shape_cls",
        "bal", "t3f_dev", "race_l3f_dev", "c4_gap_per_horse", "c4_lead_group_frac", "lap_type", "joint"}


def _runs() -> pd.DataFrame:
    h = pd.read_parquet(T61 / "horse.parquet")
    r = pd.read_parquet(T61 / "race_cls.parquet")[["race_id", "pace_cls", "shape_cls"]]
    h = h.drop_duplicates(["race_id", "horse_id"]).merge(r, on="race_id", how="left")
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    h["date"] = pd.to_datetime(h["date"])
    h["cell_idx"] = pb.cell_id(h["pace_cls"], h["shape_cls"])
    h["style_idx"] = pb.style_id(h["early_norm"].where(h["early_norm"] <= 1.0))
    return h[["race_id", "horse_id", "date", "year", "field_size", "early_norm", "fin_pct", "cell_idx", "style_idx", "pace_cls", "shape_cls"]]


def _pcell() -> tuple[pd.DataFrame, dict]:
    import lightgbm as lgb

    p = pd.read_parquet(T61 / "pre.parquet")
    p["race_id"] = p["race_id"].astype(str)
    p["cell"] = pb.cell_id(p["pace_cls"], p["shape_cls"])
    feats = [c for c in p.columns if c not in DROP and c != "cell" and pd.api.types.is_numeric_dtype(p[c])]
    X = p[feats].astype("float32")
    params = dict(objective="multiclass", num_class=9, learning_rate=0.05, num_leaves=15, min_data_in_leaf=60, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=THREADS, verbose=-1, seed=0)
    P = np.full((len(p), 9), np.nan)
    ll = {}
    for y in YEARS:
        te = (p["year"] == y).to_numpy()
        tr = (~te) & (p["cell"] >= 0).to_numpy() & ((p["year"] <= 2025).to_numpy())
        m = lgb.train(params, lgb.Dataset(X[tr], p.loc[tr, "cell"]), num_boost_round=250)
        P[te] = m.predict(X[te])
        ok = te & (p["cell"] >= 0).to_numpy()
        if ok.sum() and y <= 2025:
            pr = P[ok][np.arange(ok.sum()), p.loc[ok, "cell"].to_numpy()]
            prior = np.bincount(p.loc[tr, "cell"].to_numpy(), minlength=9) / tr.sum()
            ll[str(y)] = {"logloss": float(-np.log(np.clip(pr, 1e-9, 1)).mean()),
                          "prior_logloss": float(-np.log(prior[p.loc[ok, "cell"].to_numpy()]).mean())}
    out = pd.DataFrame(P, columns=[f"p{i}" for i in range(9)])
    out.insert(0, "race_id", p["race_id"].to_numpy())
    return out, {"features": feats, "logloss": ll}


def _validity(runs: pd.DataFrame) -> dict:
    """係数表（その年より前）を対象年の走に当てはめた展開利が、対象年の実際の着順の良さと結びつくか（標本外の再現）。"""
    res = {}
    for y in range(2021, 2026):
        tab = pb.benefit_table(runs, y)
        d = runs[(runs["year"] == y) & (runs["field_size"] >= pb.MIN_FIELD) & runs["fin_pct"].notna()].copy()
        d["ben"] = pb.lookup(tab, d["cell_idx"].to_numpy(), d["style_idx"].to_numpy())
        d = d.dropna(subset=["ben"])
        d["dev"] = -(d["fin_pct"] - d.groupby("style_idx")["fin_pct"].transform("mean"))
        slope = float(np.polyfit(d["ben"], d["dev"], 1)[0])
        res[str(y)] = {"n": int(len(d)), "corr": float(np.corrcoef(d["ben"], d["dev"])[0, 1]), "slope": slope,
                       "ben_sd": float(d["ben"].std()), "share_abs_ge_003": float((d["ben"].abs() >= pb.LUCKY).mean())}
    tab = pb.benefit_table(runs, 2026)
    res["table_2026_asof"] = {pb.CELLS[i]: [round(float(x), 4) for x in tab[i]] for i in range(9)}
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = _runs()
    print("runs", len(runs), "cell valid", float((runs["cell_idx"] >= 0).mean()), "style valid", float((runs["style_idx"] >= 0).mean()), flush=True)
    pc, meta = _pcell()
    pc.to_parquet(OUT / "pcell.parquet", index=False)
    print("pcell", json.dumps(meta["logloss"], ensure_ascii=False), flush=True)
    feats = pb.build_features(runs[["race_id", "horse_id", "date", "year", "field_size", "early_norm", "fin_pct", "cell_idx", "style_idx"]], pc, YEARS)
    feats.to_parquet(OUT / "pb_features.parquet", index=False)
    print("features", feats.shape, feats[pb.FEATURE_COLS].notna().mean().round(3).to_dict(), flush=True)
    val = _validity(runs)
    (OUT / "validity.json").write_text(json.dumps({"validity": val, "pcell": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(val, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
