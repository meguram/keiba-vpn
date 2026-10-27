"""tp-82: 初角位置（act_pos）の事前予測を、前段 ML (ts_c1_pred) の上に積んで改善する。

  python -m src.scripts.data.tp82_c1 run   # 年別 LOYO の予測 → 指標（前段 ML との比較）→ ptrain/c1x_rows.parquet

追加情報（race_result_flat / SmartRC の過去走と、騎手・調教師の過去の傾向のみ。先読みなし。市場オッズ・人気は使わない）:
 (a) 出遅れ・出脚: 過去走の ten_rel（相対テン）と c1_rel（初角相対位置）の新しい順の値・指数平滑・ばらつき・出遅れ率・前に行く率
 (b) 同型馬の並び: 前段 ML の想定位置から、レース内の想定先行頭数、自分より内/外の想定先行馬の数と最寄りの差
 (c) 騎手・調教師の作戦傾向: 過去(前の開催日まで)の (実際の初角位置 − 前段 ML 予測) の縮約平均
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.pipeline.features import tp80_userpos as U
from src.scripts.data import tp80_eval as T

ROOT = T.ROOT
OUT = ROOT / "data/local/modeling/ptrain"
REP = ROOT / "data/local/modeling/tp82"
KEY = ["race_id", "horse_id"]
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def _ewm(M: np.ndarray, decay: float = 0.6) -> np.ndarray:
    w = decay ** np.arange(M.shape[1])
    ok = ~np.isnan(M)
    num = np.where(ok, M, 0.0) @ w
    den = ok @ w
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def _tendency(D: pd.DataFrame, idcol: str, res: np.ndarray, k: float = 20.0) -> np.ndarray:
    """ID 別に『それより前の開催日までの』残差の縮約平均。同日は含めない。"""
    d = pd.DataFrame({"id": D[idcol].astype(str).to_numpy(), "date": D["date"].to_numpy(), "res": res})
    ok = d["res"].notna()
    g = d[ok].groupby(["id", "date"])["res"].agg(["sum", "count"]).reset_index().sort_values(["id", "date"])
    g["cs"] = g.groupby("id")["sum"].cumsum() - g["sum"]
    g["cn"] = g.groupby("id")["count"].cumsum() - g["count"]
    m = d[["id", "date"]].merge(g[["id", "date", "cs", "cn"]], on=["id", "date"], how="left")
    cs, cn = m["cs"].fillna(0).to_numpy(), m["cn"].fillna(0).to_numpy()
    return cs / (cn + k)


def build_features(D: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    H = T.histories(D, career)
    trel, c1 = H["trel"], H["c1"]
    F = pd.DataFrame(index=D.index)
    F["ts_c1"], F["ts_ten"] = D["ts_c1_pred"], D["ts_ten_pred"]
    for c in ("gate", "horse_number", "field_size", "distance"):
        F[c] = D[c].astype("float64")
    F["turf"] = (D["surface"].astype(str).str.contains("芝")).astype("float64")
    # (a)
    for i in range(3):
        F[f"trel_{i}"], F[f"c1_{i}"] = trel[:, i], c1[:, i]
    F["trel_ewm"], F["c1_ewm"] = _ewm(trel), _ewm(c1)
    with np.errstate(all="ignore"):
        F["trel_sd"], F["c1_sd"] = np.nanstd(trel[:, :6], axis=1), np.nanstd(c1[:, :6], axis=1)
        F["slow_start_rate"] = np.nanmean(np.where(np.isnan(trel[:, :6]), np.nan, (trel[:, :6] > 0.5).astype(float)), axis=1)
        F["front_rate"] = np.nanmean(np.where(np.isnan(c1[:, :6]), np.nan, (c1[:, :6] < 0.2).astype(float)), axis=1)
        F["back_rate"] = np.nanmean(np.where(np.isnan(c1[:, :6]), np.nan, (c1[:, :6] > 0.7).astype(float)), axis=1)
    F["gain"] = F["c1_ewm"] - F["trel_ewm"]
    # (b)
    rk = U.order_in_race(D["race_id"].to_numpy(), D["ts_c1_pred"].to_numpy(dtype="float64"), D["horse_number"].to_numpy())
    n = D["field_size"].to_numpy(dtype="float64")
    ph = np.where(n > 1, (rk - 1) / np.maximum(n - 1, 1), 0.5)
    F["ph"] = ph
    r = pd.DataFrame({"race": D["race_id"].to_numpy(), "g": D["horse_number"].to_numpy(dtype="float64"), "ph": ph, "front": (ph < 0.25).astype(float)})
    F["n_front"] = r.groupby("race")["front"].transform("sum")
    r["fg"] = np.where(r["front"] > 0, r["g"], np.nan)
    # 自分より内（馬番が小さい）/ 外にいる想定先行馬の数と、最寄りの先行馬との馬番差
    inside, outside, near_in, near_out = [], [], [], []
    for _, grp in r.groupby("race", sort=False):
        g, fr = grp["g"].to_numpy(), grp["front"].to_numpy() > 0
        fg = g[fr]
        for x in g:
            a, b = fg[fg < x], fg[fg > x]
            inside.append(len(a)), outside.append(len(b))
            near_in.append(x - a.max() if len(a) else np.nan), near_out.append(b.min() - x if len(b) else np.nan)
    order = r.groupby("race", sort=False).ngroup()  # 並びは groupby(sort=False) の出現順と一致させるため再配置
    idx = np.concatenate([grp.index.to_numpy() for _, grp in r.groupby("race", sort=False)])
    for nm, v in (("n_front_in", inside), ("n_front_out", outside), ("near_front_in", near_in), ("near_front_out", near_out)):
        s = pd.Series(v, index=idx).sort_index()
        F[nm] = s.to_numpy()
    F["ph_minus_mean_front"] = ph - r.groupby("race")["ph"].transform(lambda s: s[s < 0.25].mean() if (s < 0.25).any() else np.nan)
    # (c)
    res = (D["act_pos"] - D["ts_c1_pred"]).to_numpy()
    ids = pd.read_parquet(ROOT / "data/local/modeling/training_pipeline_v4/features/train_table.parquet", columns=KEY + ["jockey_id", "trainer_id"])
    ids["race_id"], ids["horse_id"] = ids["race_id"].astype(str), ids["horse_id"].astype(str)
    D2 = D[KEY + ["date"]].merge(ids, on=KEY, how="left")
    F["jockey_tend"] = _tendency(D2, "jockey_id", res)
    F["trainer_tend"] = _tendency(D2, "trainer_id", res)
    return F


def run() -> None:
    import lightgbm as lgb

    REP.mkdir(parents=True, exist_ok=True)
    D, career = T.load()
    D["horse_id"] = D["horse_id"].astype(str)
    F = build_features(D, career)
    y = D["act_pos"].to_numpy(dtype="float64")
    yr = D["year"].to_numpy()
    pred = np.full(len(D), np.nan)
    use = np.isin(yr, YEARS)
    cols = list(F.columns)
    for Y in YEARS:
        tr = use & (yr != Y) & ~np.isnan(y)
        te = use & (yr == Y)
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0, n_jobs=4, verbose=-1)
        m.fit(F.loc[tr, cols], y[tr])
        pred[te] = m.predict(F.loc[te, cols])
        T._log(f"{Y}: 学習 {int(tr.sum())} 行")
    hn = D["horse_number"].to_numpy()
    base_rank = U.order_in_race(D["race_id"].to_numpy(), D["ts_c1_pred"].to_numpy(dtype="float64"), hn)
    new_rank = U.order_in_race(D["race_id"].to_numpy(), np.where(np.isnan(pred), D["ts_c1_pred"].to_numpy(dtype="float64"), pred), hn)
    rep = {"base": T.by_year(D, base_rank), "c1x": T.by_year(D, new_rank)}
    (REP / "c1x_metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for k in ("sp_c1", "sp_ten", "class_acc", "leader_hit"):
        print(k, {yy: (round(rep["base"][yy][k], 4), round(rep["c1x"][yy][k], 4)) for yy in ["2022", "2023", "2024", "2025", "2022-24"]})
    n = D["field_size"].to_numpy(dtype="float64")
    out = D[KEY].copy()
    out["c1x_pos"] = pred
    out["c1x_ph"] = np.where(n > 1, (new_rank - 1) / np.maximum(n - 1, 1), 0.5)
    out["c1x_delta"] = out["c1x_ph"] - np.where(n > 1, (base_rank - 1) / np.maximum(n - 1, 1), 0.5)
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    print("top importance", imp.head(8).to_dict())
    out.to_parquet(OUT / "c1x_rows.parquet", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
