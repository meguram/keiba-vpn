"""子モデル: 欠損値を他の特徴量から予測して埋める（モデルベースの欠損補完）。

python -m src.scripts.data.impute_child build   # 候補列の選定 → 列ごとに予測 → 精度確認 → 合格した列だけ imp_rows.parquet に保存
方針:
- 構造的な欠損（初出走・履歴なしで当然に無い）は補完しない（欠損フラグのまま）。『履歴が3走以上ある馬の欠損』が欠損行の半分以上の列を候補にする。
- 補完モデルは 2020〜2021 年の観測行だけで学習する（検証の前の年。結果の目的変数は使わない＝特徴量の関係だけを学ぶ）。
- 精度の確認は、学習に使わない 2022 年の観測行を隠した予測で、中央値で埋めたときと比べる（MAE と R²）。R² ≥ 0.2 かつ MAE が中央値より 15% 以上小さい列だけ『許容』とする。
- 出力は imp_<列名>（欠損を予測値で埋めた値。観測があればそのまま）と imp_miss_<列名>（欠損フラグ）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp27_eval as t
from src.scripts.data import v6_eval as ev

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
KEY = ["race_id", "horse_id"]
MIN_MISS, STRUCT_SHARE = 0.03, 0.5
R2_MIN, MAE_GAIN = 0.2, 0.15


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    import lightgbm as lgb

    df, base, meta, folds, lp = t.setup()
    X = ev.build_frame(ev.config_of("z:RB@e"), df)
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    prior = pd.to_numeric(X.get("horse_prior_races"), errors="coerce").to_numpy()
    num = X.select_dtypes(include=[np.number])
    cand = []
    for c in num.columns:
        m = num[c].isna().to_numpy()
        if m.mean() < MIN_MISS or m.mean() > 0.9:
            continue
        data_missing = float(((prior >= 3) & m).sum() / max(m.sum(), 1))
        if data_missing >= STRUCT_SHARE:
            cand.append((c, float(m.mean()), data_missing))
    _log(f"候補 {len(cand)} 列（欠損3〜90%、履歴3走以上の欠損が半分以上）")
    tr_mask = np.isin(year, [2020, 2021]); te_mask = year == 2022
    Xn = num.astype("float32")
    res, fill = {}, {}
    for i, (c, mr, ds) in enumerate(cand):
        others = [x for x in Xn.columns if x != c and not x.startswith(c.split("_")[0] + "_" + (c.split("_")[1] if "_" in c else ""))]
        obs = ~Xn[c].isna().to_numpy()
        tr = tr_mask & obs; te = te_mask & obs
        if tr.sum() < 3000 or te.sum() < 1000:
            continue
        rng = np.random.default_rng(i)
        cols = list(rng.choice(others, size=min(400, len(others)), replace=False))
        m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=31, colsample_bytree=0.5, subsample=0.7, subsample_freq=1, min_child_samples=40, n_jobs=4, verbose=-1)
        m.fit(Xn.loc[tr, cols], Xn.loc[tr, c])
        pred = m.predict(Xn.loc[te, cols]); y = Xn.loc[te, c].to_numpy()
        med = float(np.nanmedian(Xn.loc[tr, c]))
        mae_m, mae_p = float(np.abs(y - med).mean()), float(np.abs(y - pred).mean())
        r2 = 1 - float(((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        ok = bool(r2 >= R2_MIN and mae_p <= mae_m * (1 - MAE_GAIN))
        res[c] = {"miss_rate": round(mr, 3), "data_missing_share": round(ds, 3), "r2_2022": round(r2, 3), "mae_median": round(mae_m, 4), "mae_model": round(mae_p, 4), "accepted": ok}
        if ok:
            miss = ~obs
            v = Xn[c].to_numpy().copy()
            if miss.any():
                v[miss] = m.predict(Xn.loc[miss, cols])
            fill[f"imp_{c}"] = v.astype("float32"); fill[f"imp_miss_{c}"] = miss.astype("float32")
        if (i + 1) % 10 == 0:
            _log(f"{i + 1}/{len(cand)} 列、合格 {sum(r['accepted'] for r in res.values())}")
    out = pd.concat([df[KEY].astype(str).reset_index(drop=True), pd.DataFrame(fill)], axis=1)
    out.to_parquet(OUT / "imp_rows.parquet", index=False)
    (OUT / "imp_report.json").write_text(json.dumps({"n_candidates": len(cand), "n_tested": len(res), "n_accepted": sum(r["accepted"] for r in res.values()), "columns": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"完了: 候補 {len(cand)}、検証 {len(res)}、合格 {sum(r['accepted'] for r in res.values())}、保存 {out.shape}")


if __name__ == "__main__":
    main()
