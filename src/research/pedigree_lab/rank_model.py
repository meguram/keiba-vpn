"""探す側の記述: モデルの種類の比べ（LightGBM の lambdarank と二値。第 4 版 pure の列・2 代母の単位の 5 分割・開発の行。未確認）。

lambdarank の関連度は 1 着 3・2 着 2・3 着 1・それ以外 0。得点はレースの中の softmax（温度 1 つを開発の OOF で当てる）で勝ちの確率に直す。
折はレースを割らない「レースの 2 代母の多数決」ではなく、kfold と同じ 2 代母の単位で、学習ではレースの一部の馬だけになる。
lambdarank はレースの中の相対で学ぶので、学習の行をレース単位にまとめ、その折に入る馬だけで群を作る。
    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.rank_model
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data, kfold, model, v5


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "rank_model", "探す側の記述: lambdarank と二値の比べ（尤度だけ）"))
    X = v5.base_frame("v4")
    X = X.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    dev = X["split"].isin([0, 1]).to_numpy()
    cols = ablate.col_sets(X)["pure"]
    f = kfold.folds(X, dev, "dd")
    rel = np.clip(4 - X["finish_position"].to_numpy(), 0, 3)
    rid = X["race_id"].to_numpy()
    score = np.full(len(X), np.nan)
    params = dict(model.PARAMS, objective="lambdarank", metric="ndcg", eval_at=[3], lambdarank_truncation_level=5)
    params.pop("seed", None)
    iters = []
    for k in range(kfold.K):
        te, va = f == k, f == (k + 1) % kfold.K
        tr = dev & ~te & ~va
        def ds(m):
            g = pd.Series(rid[m]).groupby(rid[m], sort=False).size().to_numpy()
            return lgb.Dataset(X.loc[m, cols], rel[m], group=g)
        m = lgb.train(params, ds(tr), 2000, valid_sets=[ds(va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration)
        score[te] = m.predict(X.loc[te, cols], num_iteration=m.best_iteration)
    s = np.where(dev, score, -np.inf)
    w = X["win"].to_numpy() == 1
    best = None
    for t in np.arange(0.2, 3.01, 0.1):
        eta = s * t
        mx = pd.Series(eta).groupby(rid).transform("max").to_numpy()
        e = np.where(dev, np.exp(eta - mx), 0.0)
        p = e / np.where((ss := pd.Series(e).groupby(rid).transform("sum").to_numpy()) > 0, ss, 1)
        ll = -np.mean(np.log(np.clip(p[w & dev], 1e-9, 1)))
        if best is None or ll < best[0]:
            best = (ll, t, p)
    _, t, p = best
    ref = v5.v4_ref(X)["dd"]["p_win"].to_numpy()
    res = {"iters": iters, "temp": round(float(t), 2), "rank_dd_win_gain": v5.gain(X, p, dev), "binary_dd_win_gain": v5.gain(X, ref, dev),
           "vs_binary": v5.paired_diff(X, p, ref, dev)}
    print(json.dumps(res, ensure_ascii=False))
    (data.OUT / "rank_model.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(data.OUT / "rank_model.json"))


if __name__ == "__main__":
    main()
