"""オッズ予測 試行5（複勝）: 複勝オッズを「レースの単勝占有率の並び」から写す 2 段目。
学習（各年より前の年）: 実際の市場の占有率から作った量（自分の log 占有率・レース内の順位・上位の占有率・頭数・エントロピー）→ log 複勝（中央）。
予測: 1 段目で予測した占有率（share_raw をレース内で正規化）から同じ量を作って写す。直接の予測（place_raw）とは、前の年までの OOF で重みを決めて混ぜる。

    python docs/operations/retrain_trials/odds_place2.py <占有率の出力元> <書き出し先>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
os.environ["ODDS_PRED_OUT"] = str(DST)
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.scripts.data import odds_pred_eval as oe  # noqa: E402

oe.OUT = DST
DST.mkdir(parents=True, exist_ok=True)
raw = pd.read_parquet(SRC / "wp_raw_oof.parquet")
wp = oe.load_wp()
d = raw.merge(wp[oe.KEY + ["p_mkt", "place_mid"]], on=oe.KEY, how="inner").sort_values(["race_id", "horse_id"]).reset_index(drop=True)
rid = d["race_id"].to_numpy()


def feats(logp: np.ndarray) -> pd.DataFrame:
    s = pd.Series(logp)
    g = s.groupby(rid)
    p = np.exp(s - g.transform("max"))
    p = p / p.groupby(rid).transform("sum")
    gp = p.groupby(rid)
    f = pd.DataFrame({"lp": np.log(p), "rk": gp.rank(ascending=False), "n": gp.transform("size"),
                      "ent": (-(p * np.log(p))).groupby(rid).transform("sum"),
                      "p1": gp.transform("max"), "p2": gp.transform(lambda x: np.sort(x.to_numpy())[-2] if len(x) > 1 else 0),
                      "p3": gp.transform(lambda x: np.sort(x.to_numpy())[-3] if len(x) > 2 else 0)})
    f["cum_above"] = p.groupby(rid).transform(lambda x: x.sort_values(ascending=False).cumsum().reindex(x.index)) - p
    f["rk_n"] = f["rk"] / f["n"]
    return f


Ft = feats(np.log(d["p_mkt"].to_numpy()))
Fp = feats(d["share_raw"].fillna(np.log(1 / 14)).to_numpy())
y = np.log(d["place_mid"].to_numpy())
yr = d["year"].to_numpy()
P = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, verbose=-1, num_threads=4)
struct = np.full(len(d), np.nan)
# 構造の写像は 2020〜（実際の市場の量なので、学習は 2020 年以降のすべての前の年）
full = pd.read_parquet(SRC / "wp_raw_oof.parquet")[oe.KEY + ["year"]]
allwp = wp.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
rid_all = allwp["race_id"].to_numpy()
rid_bak = rid
rid = rid_all
FA = feats(np.log(allwp["p_mkt"].to_numpy()))
rid = rid_bak
yA = np.log(allwp["place_mid"].to_numpy())
yrA = allwp["year"].to_numpy()
for Y in oe.YEARS:
    tr = (yrA < Y) & (yrA >= 2020)
    m = lgb.train(P, lgb.Dataset(FA[tr], yA[tr]), 800)
    te = yr == Y
    struct[te] = m.predict(Fp[te])
    print(Y, "構造の写像 学習", int(tr.sum()), "真の占有率での RMSE（参考）", round(float(np.sqrt(((m.predict(Ft[te]) - y[te]) ** 2).mean())), 4), flush=True)
d["place_struct"] = struct
# 混ぜる重み: 前の年までの OOF（2022 は 0.5）
w = {}
for Y in oe.YEARS:
    prev = (yr >= oe.YEARS[0]) & (yr < Y)
    if prev.sum() < 1000:
        w[Y] = 0.5
        continue
    a, b_ = d.loc[prev, "place_raw"].to_numpy(), d.loc[prev, "place_struct"].to_numpy()
    grid = np.linspace(0, 1, 21)
    w[Y] = float(grid[np.argmin([np.mean((t * b_ + (1 - t) * a - y[prev]) ** 2) for t in grid])])
mix = d["place_raw"].to_numpy().copy()
for Y, t in w.items():
    te = yr == Y
    mix[te] = t * struct[te] + (1 - t) * d.loc[te, "place_raw"].to_numpy()
print("混ぜる重み（構造の側）", w)
out = raw.merge(d[oe.KEY].assign(place_new=mix), on=oe.KEY, how="left")
out["place_raw"] = out["place_new"].fillna(out["place_raw"])
out.drop(columns=["place_new"]).to_parquet(DST / "wp_raw_oof.parquet", index=False)
for f in ("wp_fit_info.json",):
    if (SRC / f).exists():
        shutil.copy(SRC / f, DST / f)
(DST / "place2_weights.json").write_text(json.dumps(w))
oe.evaluate_phase1()
