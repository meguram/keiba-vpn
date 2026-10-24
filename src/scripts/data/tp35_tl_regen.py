"""tp-35 付録: 調教の再取得（ログイン切れで空保存だった 5,261 頭を取り直し）後の追い切り表で TL_A を再生成し、被覆と V4 への効果を測る。

python -m src.scripts.data.tp35_tl_regen build   # cand_TL_A_v2.parquet（既存の cand_TL_A.parquet は上書きしない）と tl_reference_v2.parquet、年別の被覆の表
python -m src.scripts.data.tp35_tl_regen eval    # V4 の tl_（50 列）を再生成版に差し替えた場合の増分（valid 2022・2023・2024、2 seed。基準は旧 tl_ の V4）
追い切り表は TP35_LAPS（既定: tp35/training_laps_fresh.parquet）。test 2025・2026 は使わない。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = V / "tp35"
WORK = ROOT / "data/local/modeling/training_pipeline"
os.environ.setdefault("TP35_LAPS", str(OUT / "training_laps_fresh.parquet"))

from src.pipeline.features import training_patterns as tp  # noqa: E402
from src.pipeline.features import training_serving as ts  # noqa: E402
from src.scripts.data import tp35_eval as E  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402

KEY = ["race_id", "horse_id"]


def _years_cov(M: pd.DataFrame, df: pd.DataFrame, cols: list[str]) -> dict:
    y = pd.to_datetime(df["date"]).dt.year.to_numpy()
    out = {}
    for c in cols:
        out[c] = {int(k): round(float(M[c].notna().groupby(y).mean().get(k, np.nan)), 4) for k in sorted(set(y))}
    return out


def cmd_build() -> None:
    df, base, meta, folds, lp = t.setup()
    laps = pd.read_parquet(Path(os.environ["TP35_LAPS"]))
    E._log(f"追い切り表 {len(laps):,} 本（旧 1,190,429 本）")
    tt = tp.add_reference_z(tp.prepare(laps))
    feat = tp.add_self_relative(tp.race_features(tt))
    RD = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    M = pd.DataFrame({"horse_id": df["horse_id"].astype(str).to_numpy(), "race_date": RD.to_numpy()}, index=df.index).merge(feat, on=["horse_id", "race_date"], how="left")
    M.index = df.index
    cols = ts.tl_a_columns(M.columns)
    new = pd.concat([df[KEY].reset_index(drop=True), M[cols].reset_index(drop=True)], axis=1)
    new.to_parquet(OUT / "cand_TL_A_v2.parquet", index=False)
    ref = ts.build_reference_table(laps)
    ref.to_parquet(OUT / "tl_reference_v2.parquet", index=False)
    old = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    old["race_id"], old["horse_id"] = old["race_id"].astype(str), old["horse_id"].astype(str)
    oldm = df[KEY].merge(old, on=KEY, how="left")
    oldm.index = df.index
    rep = {"laps_rows_new": int(len(laps)), "n_cols": len(cols),
           "coverage_old": _years_cov(oldm, df, ["has_training", "f_t1", "w1_exists"] if "f_t1" in oldm.columns else ["has_training"]),
           "coverage_new": _years_cov(M, df, ["has_training", "f_t1", "w1_exists"]), "reference_rows": int(len(ref))}
    # 旧版と値が変わった割合（列ごと）
    chg = {}
    for c in cols[:60]:
        if c in oldm.columns:
            a, b = oldm[c].to_numpy(dtype=float), M[c].to_numpy(dtype=float)
            chg[c] = float(np.mean(~((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, equal_nan=False))))
    rep["changed_share"] = chg
    E._jsave(OUT / "tl_regen_build.json", rep)
    E._log("被覆（has_training）年別 旧→新: " + " ".join(f"{y}:{rep['coverage_old']['has_training'].get(y, float('nan')):.3f}→{rep['coverage_new']['has_training'].get(y, float('nan')):.3f}" for y in range(2020, 2026)))


def cmd_eval() -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = E._frame(df)
    tl_old = [c for c in X0.columns if c.startswith("tl_")]
    new = pd.read_parquet(OUT / "cand_TL_A_v2.parquet")
    new["race_id"], new["horse_id"] = new["race_id"].astype(str), new["horse_id"].astype(str)
    m = df[KEY].merge(new, on=KEY, how="left")
    m.index = df.index
    renamed = {c: f"tl_{c}" for c in m.columns if c not in KEY}
    G = m.drop(columns=KEY).rename(columns=renamed)
    miss = [c for c in tl_old if c not in G.columns]
    E._log(f"V4 の tl_ 列 {len(tl_old)}、再生成版 {G.shape[1]}、再生成版に無い列 {miss[:5]}")
    X1 = X0.copy()
    for c in tl_old:
        if c in G.columns:
            X1[c] = G[c].to_numpy()
    years, seeds = [2022, 2023, 2024], [2, 3]
    resp = OUT / "tl_regen_eval.json"
    res = E._jload(resp)
    lv0, oof0 = E._run_subset("V4(旧tl_)", X0, meta, folds, lp, years, seeds)
    lv1, oof1 = E._run_subset("V4(新tl_)", X1, meta, folds, lp, years, seeds)
    c = t._cmp(oof0, oof1, fin_map, "TL_A再生成")
    # 年別・被覆が上がった行（旧版で tl_has_training が欠損だった行）での差
    yr = meta.set_index(KEY)["year"]
    miss_old = X0["tl_has_training"].isna().to_numpy() if "tl_has_training" in X0.columns else np.zeros(len(df), bool)
    mrow = pd.Series(miss_old, index=pd.MultiIndex.from_frame(df[KEY]))

    def ll_by(o: pd.DataFrame) -> dict:
        k = list(zip(o["race_id"].astype(str), o["horse_id"].astype(str)))
        y = (fin_map.loc[k].to_numpy() == 1).astype(float)
        p = np.clip(o["p_win"].to_numpy(), 1e-6, 1 - 1e-6)
        l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        yy = yr.loc[k].to_numpy()
        mm = mrow.reindex(pd.MultiIndex.from_tuples(k)).to_numpy()
        out = {str(int(v)): float(l[yy == v].mean()) for v in sorted(set(yy))}
        out["旧版で調教なし(欠損)の行"] = float(l[mm].mean()) if mm.any() else float("nan")
        out["旧版で調教ありの行"] = float(l[~mm].mean())
        out["n_missing_rows"] = int(mm.sum())
        return out

    res["by_year_old"], res["by_year_new"] = ll_by(oof0), ll_by(oof1)
    res.update({"levels_old": lv0, "levels_new": lv1, "vs_old": c, "cols_replaced": len([c_ for c_ in tl_old if c_ in G.columns])})
    E._jsave(resp, res)
    E._log(f"TL_A再生成: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


if __name__ == "__main__":
    {"build": cmd_build, "eval": cmd_eval}[sys.argv[1]]()
