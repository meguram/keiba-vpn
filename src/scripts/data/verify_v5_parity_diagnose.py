"""通しの推論経路（V3Serving + v5 バンドル）の確率が学習側とずれる出どころを、群ごとに切り分ける。

KEIBA_ENV=stg V5_GROUPS=CY,BEST python -m src.scripts.data.verify_v5_parity_diagnose --days 20250614 --max-races 8 --out /tmp/v5_diag.json
手順: 推論経路で特徴量 X を作り、同じ行の学習側の特徴量（fit_final_v5 の行列）と比べる。(1) 列ごとの不一致の割合を群別に集計、(2) 群ごとに『その群の列だけを学習側の値に差し替えて』予測し直し、
学習側の予測（final_eval_v5/test2025_predictions.parquet）との相関・LogLoss 差がどれだけ縮まるか（= その群が主なずれの出どころか）を見る。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import serving_v4 as sv
from src.pipeline.features import serving_v5 as sv5
from src.pipeline.inference import weight_triggered_predict as wtp
from src.pipeline.inference.v3_serving import V3Serving, WORK

KEY = ["race_id", "horse_id"]
V4 = WORK.parent / "training_pipeline_v4"


def _ll(y, p) -> float:
    return float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--max-races", type=int, default=8)
    ap.add_argument("--bundle", default=str(WORK / "final_v5/bundle_v5_2024"))
    ap.add_argument("--stage", default=str(WORK / "final_stage/v4_verify_2024"))
    ap.add_argument("--tl-ref", default=str(V4 / "tp35/tl_reference_v2.parquet"))
    ap.add_argument("--ref", default=str(WORK / "final_eval_v5/test2025_predictions.parquet"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage
    from src.scripts.data import fit_final_v4 as f4
    from src.scripts.data import v5_common

    storage = HybridStorage()
    ctx = wtp.InferenceContext()
    ctx.warm()
    raw_full = ctx.raw_hist
    v3 = V3Serving(a.bundle, a.stage, tl_reference=a.tl_ref)
    v3.warm()
    v5_common.patch()
    df, meta, mats, _A, _Af = f4.build_matrices()
    Xt = pd.concat([mats[n][2] for n in mats], axis=1)
    Xt = Xt.loc[:, ~Xt.columns.duplicated()]
    Xt = pd.concat([df[KEY].astype(str).reset_index(drop=True), Xt.reset_index(drop=True)], axis=1)
    ref = pd.read_parquet(a.ref)
    ref["race_id"], ref["horse_id"] = ref["race_id"].astype(str), ref["horse_id"].astype(str)
    cols_v5 = sv5.v5_columns()
    v4g = {g: set(cs) for g, cs in sv.group_columns().items()}
    groups = {"tl_（追い切り TL_A）": lambda c: c.startswith("tl_"), "CY（wc_）": lambda c: c in set(cols_v5["CY"]), "PX（ALL_BEST）": lambda c: c in set(cols_v5["PX"])}
    for g, cs in v4g.items():
        groups[f"v4:{g}"] = (lambda cs: (lambda c: c in cs))(cs)
    hist_full = v3.history.copy()
    rep: dict = {}
    for day in a.days:
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        flat = pl.read_parquet(sv.TABLES / day[:4] / "race_shutuba_flat.parquet", columns=["race_id", "date"]).filter(pl.col("date").cast(pl.Utf8) == d)
        ids = sorted(flat["race_id"].unique().to_list())[: a.max_races]
        card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
        cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
        ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
        F, _ = ctx.build(card)
        h = hist_full[hist_full["date"].astype(str) < d]
        X = v3.features(card, F, storage, history=h)
        X["race_id"], X["horse_id"] = X["race_id"].astype(str), X["horse_id"].astype(str)
        T = X[KEY].merge(Xt, on=KEY, how="left", suffixes=("", "__t"))
        T.index = X.index
        fc = [c for c in v3.feature_cols if c in T.columns]
        known = {c for g in groups.values() for c in fc if g(c)}
        groups_here = dict(groups) | {"V0（LayerA・血統・前段・相対化など）": (lambda c: c not in known)}
        P0 = v3.predict(X).merge(ref, on=KEY, suffixes=("", "__r"))
        yy = {t: (P0["finish_position"].to_numpy() <= k).astype(float) for t, k in (("win", 1), ("top2", 2), ("top3", 3))}

        def score(Xm):
            P = v3.predict(Xm).merge(ref, on=KEY, suffixes=("", "__r"))
            return {t: {"corr": float(np.corrcoef(P[f"p_{t}"], P[f"p_{t}__r"])[0, 1]), "diff": _ll(yy[t], P[f"p_{t}"].to_numpy()) - _ll(yy[t], P[f"p_{t}__r"].to_numpy())} for t in ("win", "top2", "top3")}

        res = {"rows": int(len(X)), "base": score(X), "groups": {}}
        for g, fn in groups_here.items():
            gc = [c for c in fc if fn(c)]
            if not gc:
                continue
            badcol = np.zeros(len(gc))
            for i, c in enumerate(gc):
                xs, ys = X[c], T[c]
                if pd.api.types.is_numeric_dtype(xs) and pd.api.types.is_numeric_dtype(ys):
                    x, y = xs.to_numpy(dtype=float), ys.to_numpy(dtype=float)
                    ok = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-3, atol=1e-4)
                else:
                    ok = (xs.astype(str).to_numpy() == ys.astype(str).to_numpy()) | (xs.isna().to_numpy() & ys.isna().to_numpy())
                badcol[i] = float((~ok).mean())
            Xm = X.copy()
            for c in gc:
                Xm[c] = T[c].to_numpy()
            res["groups"][g] = {"cols": len(gc), "cols_with_diff": int((badcol > 0).sum()), "mean_bad_row_rate": float(badcol.mean()),
                                "worst": {gc[i]: float(badcol[i]) for i in np.argsort(-badcol)[:4] if badcol[i] > 0}, "after_replace": score(Xm)}
        res["all_training"] = score(X.assign(**{c: T[c].to_numpy() for c in fc}))
        rep[day] = res
        print(day, json.dumps({"base": res["base"]["win"], "all_training": res["all_training"]["win"], "groups": {g: (v["cols_with_diff"], round(v["after_replace"]["win"]["corr"], 4)) for g, v in res["groups"].items()}}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
