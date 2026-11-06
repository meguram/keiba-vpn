"""tp-84 段 2: 『前半の時計・RPCI の予想』を、展開タイプ（ペース3×隊列3）の分類器の入力に足すと当たりが良くなるか。tp-61 の評価（学習 2020〜2023 → 検証 2024・2025）に揃える。ホストで実行。

python -m src.scripts.data.tp84_pace_to_type [--out /tmp/tp84_out]
段 1: レース単位の時計のずれ（F3・F5・HYB・L3・RPCI）を回帰。検証年 y の予想は y より前の年だけで学習。学習年（2020〜2023）の予想は、その年を除いた 2020〜2023 の年で学習（先読みなし・自分のレースは入らない）。
段 2: tp-61 の分類器（LightGBM、RACE_FEATS + HANA）に、段 1 の予想（ずれ）を足したものと足さないものを、同じ行・同じ設定で比べる。2026 年は使わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp61_pre as pre
from src.scripts.data import tp84_pace_eval as ev

TGT = ["F3", "F5", "HYB", "L3", "RPCI"]
PRED = ["pp_F3", "pp_F5", "pp_HYB", "pp_L3", "pp_RPCI", "pp_bal"]
D = pre.D


def stage1(d: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = []
    for y in range(2020, 2026):
        te = d[d.year == y]
        tr = d[d.year < y] if y >= 2024 else d[(d.year <= 2023) & (d.year != y)]
        if len(tr) < 1500:
            continue
        r = pd.DataFrame({"race_id": te["race_id"].to_numpy()})
        for t in TGT:
            p_te, pred, _ = ev.fit_predict(tr, te, feats, t)
            r["pp_" + t] = pred - p_te
        r["pp_bal"] = r["pp_L3"] - r["pp_F3"]
        out.append(r)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tp84_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(ev.FEAT)
    d["year"] = d["date"].str[:4].astype(int)
    d = d[d["surface"].isin(["芝", "ダート"]) & (d["year"] <= 2025)].copy()
    d = ev.race_table(d).reset_index(drop=True)
    cols0 = list(pd.read_parquet(ev.FEAT).columns)
    for k, v in (("cush", ev.cushion_features(d)), ("sd2", ev.sameday2_features(d)), ("mem", ev.member_features(d, list(range(2019, 2026))))):
        d = d.merge(v, on="race_id", how="left")
        cols0 += [c for c in v.columns if c != "race_id"]
    for c in ["venue", "surface", "direction", "grade", "race_class", "weather", "track_condition", "course_key"]:
        d[c] = d[c].astype("category")
    drop = {"race_id", "date", "laps_json", "year", "F3", "F5", "L3", "RPCI", "HYB"}
    feats = list(dict.fromkeys(c for c in cols0 if c not in drop and c in d.columns))
    ev._log(f"stage1 features {len(feats)}")
    s1 = stage1(d, feats)
    s1.to_parquet(out / "stage1_pred.parquet", index=False)

    p = pd.read_parquet(D / "pre.parquet")
    p = p[p["pace_cls"].isin(pre.PACE) & p["shape_cls"].isin(pre.SHAPE)].merge(s1, on="race_id", how="inner")
    base = [c for c in pre.RACE_FEATS + pre.HANA if c in p.columns]
    tr, va = p[p.year.isin([2020, 2021, 2022, 2023])], p[p.year.isin([2024, 2025])]
    ev._log(f"classifier rows train {len(tr):,} / valid {len(va):,}")
    variants = {"base(+hana)": base, "+pred": base + PRED, "+pred_bal_only": base + ["pp_bal"], "+pred_time": base + ["pp_F3", "pp_F5", "pp_HYB", "pp_L3"]}
    res = {}
    for name, classes, col in (("pace", pre.PACE, "pace_cls"), ("shape", pre.SHAPE, "shape_cls"), ("joint", pre.JOINT, "joint")):
        ytr = tr[col].map({c: i for i, c in enumerate(classes)}).to_numpy()
        yva = va[col].map({c: i for i, c in enumerate(classes)}).to_numpy()
        res[f"{name}/prior"] = pre._ll(yva, np.tile(np.bincount(ytr, minlength=len(classes)) / len(ytr), (len(yva), 1)))
        for vn, fs in variants.items():
            ps = []
            for seed in (0, 1, 2):
                import lightgbm as lgb

                m = lgb.LGBMClassifier(objective="multiclass", num_class=len(classes), n_estimators=300, **{**pre.LGB_PARAMS, "random_state": seed}).fit(tr[fs], ytr)
                ps.append(m.predict_proba(va[fs]))
            pm = np.mean(ps, axis=0)
            res[f"{name}/{vn}"] = pre._ll(yva, pm)
            res[f"{name}/{vn}/acc"] = float((pm.argmax(1) == yva).mean())
            for yr in (2024, 2025):
                k = (va["year"] == yr).to_numpy()
                res[f"{name}/{vn}/{yr}"] = pre._ll(yva[k], pm[k])
    for k, v in res.items():
        print(f"{k:40s} {v:.5f}")
    (out / "to_type.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
