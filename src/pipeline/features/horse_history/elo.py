"""順位から逐次更新する多頭数 Elo。1 日分のレースは前日までのレーティングで評価し、日の終わりにまとめて更新する。

delta > 0 のとき、期待勝率に斤量差のハンデ項を入れる: P(i が j に勝つ) = σ((R_i − R_j − delta·(w_i − w_j)) / s)。
このとき R は「同じ斤量だった場合」の能力になり、r_eff が今回の斤量を込みにした実効値になる。"""
from __future__ import annotations
import numpy as np
import polars as pl

SCALE = 400.0 / np.log(10.0)


def race_block(r: np.ndarray, wr: np.ndarray, fin: np.ndarray, delta: float) -> dict:
    """1 レース（2 頭以上）の事前統計。r は各馬の事前レーティング、wr は斤量、fin は着順（未確定なら仮の順位）。"""
    N = len(r)
    diff = r[:, None] - r[None, :] - delta * (wr[:, None] - wr[None, :])
    ep = 1.0 / (1.0 + np.exp(-diff / SCALE))
    E = (ep.sum(1) - 0.5) / (N - 1)
    S = (N - fin) / (N - 1)
    tot = r.sum()
    fs_top3 = np.empty(N)
    fs_std = np.empty(N)
    for a in range(N):
        others = np.delete(r, a)
        fs_top3[a] = np.sort(others)[::-1][:3].mean()
        fs_std[a] = others.std()
    return {"r_eff": r - delta * (wr - wr.mean()), "S": S, "E": E, "fs_mean": (tot - r) / (N - 1), "fs_top3": fs_top3, "fs_std": fs_std}


def elo_prerace(rating: dict, cnt: dict, runs: pl.DataFrame, delta: float, init: float = 1500.0) -> pl.DataFrame:
    """結果が無いレースの Elo 事前統計（run_elo の出力と同じ列）。レーティングは更新しない。runs は 1 日分以内。
    着順は馬番順の仮の値（今日の行の特徴量には影響しない）。"""
    cols = ("r_pre", "n_pre", "S", "E", "fs_mean", "fs_top3", "fs_std", "r_eff")
    parts = []
    for rid, g in runs.sort(["race_id", "horse_number"]).group_by("race_id", maintain_order=True):
        hs = g["horse_id"].to_numpy()
        N = len(hs)
        r = np.array([rating.get(h, init) for h in hs])
        c = np.array([cnt.get(h, 0) for h in hs], dtype=float)
        o = {k: np.full(N, np.nan) for k in cols}
        o["r_pre"], o["n_pre"] = r, c
        if N >= 2:
            wr = g["jockey_weight"].fill_null(56.0).to_numpy().astype(float) if delta else np.full(N, 56.0)
            o.update(race_block(r, wr, np.arange(1, N + 1, dtype=float), delta))
        parts.append(pl.DataFrame({"race_id": g["race_id"].to_numpy(), "horse_id": hs, **o}))
    res = pl.concat(parts) if parts else pl.DataFrame()
    return res.with_columns((pl.col("S") - pl.col("E")).alias("res"))


def run_elo(runs: pl.DataFrame, k: float = 300.0, init: float = 1500.0, delta: float = 0.0, return_state: bool = False):
    df = runs.select("date", "race_id", "horse_id", "finish_position").with_row_index("i")
    wgt = runs["jockey_weight"].fill_null(56.0).to_numpy().astype(float) if delta else np.full(runs.height, 56.0)
    n = df.height
    horse = df["horse_id"].to_numpy()
    fin = df["finish_position"].to_numpy().astype(float)
    # (date, race) 単位の行インデックス
    order = df.sort(["date", "race_id"])["i"].to_numpy()
    dates = df["date"].to_numpy()[order]
    races = df["race_id"].to_numpy()[order]
    out = {c: np.full(n, np.nan) for c in ("r_pre", "n_pre", "S", "E", "fs_mean", "fs_top3", "fs_std", "r_eff")}
    rating, cnt = {}, {}
    pos = 0
    while pos < n:
        d = dates[pos]
        end = pos
        while end < n and dates[end] == d:
            end += 1
        pending = []
        s = pos
        while s < end:
            e = s
            while e < end and races[e] == races[s]:
                e += 1
            idx = order[s:e]
            N = len(idx)
            hs = horse[idx]
            r = np.array([rating.get(h, init) for h in hs])
            c = np.array([cnt.get(h, 0) for h in hs], dtype=float)
            out["r_pre"][idx] = r
            out["n_pre"][idx] = c
            if N >= 2:
                st = race_block(r, wgt[idx], fin[idx], delta)
                for key in ("r_eff", "S", "E", "fs_mean", "fs_top3", "fs_std"):
                    out[key][idx] = st[key]
                kk = k * np.where(c < 3, 1.5, 1.0)
                pending.append((hs, kk * (st["S"] - st["E"])))
            s = e
        for hs, dl in pending:
            for h, v in zip(hs, dl):
                rating[h] = rating.get(h, init) + v
                cnt[h] = cnt.get(h, 0) + 1
        pos = end
    res = pl.DataFrame({"i": np.arange(n), **{c: v for c, v in out.items()}})
    res = res.with_columns((pl.col("S") - pl.col("E")).alias("res"))
    out_df = df.select("race_id", "horse_id", "i").join(res, on="i").drop("i")
    return (out_df, rating, cnt) if return_state else out_df


def select_delta(runs: pl.DataFrame, fit_end, grid=(0.0, 5.0, 10.0, 20.0, 30.0)) -> tuple[float, dict]:
    """斤量差のハンデ項 δ（Elo ポイント/kg）を、fit 期間（fit_end 以前）の勝ち馬の対数尤度で選ぶ。0 が最良なら 0。"""
    sub = runs.filter(pl.col("date") <= fit_end)
    lls = {}
    for dl in grid:
        el = run_elo(sub, delta=dl)
        d = (sub.select("race_id", "horse_id", "finish_position", "jockey_weight")
             .join(el.select("race_id", "horse_id", "r_pre", "n_pre"), on=["race_id", "horse_id"])
             .filter(pl.col("n_pre") >= 2))
        d = d.with_columns(
            ((pl.col("r_pre") - dl * (pl.col("jockey_weight").fill_null(56.0) - pl.col("jockey_weight").fill_null(56.0).mean().over("race_id"))) / SCALE).alias("x"))
        d = d.with_columns((pl.col("x") - pl.col("x").max().over("race_id")).exp().sum().over("race_id").log().alias("_lse"), pl.col("x").max().over("race_id").alias("_mx"))
        ll = d.filter(pl.col("finish_position") == 1).select((pl.col("x") - pl.col("_lse") - pl.col("_mx")).sum()).item()
        lls[float(dl)] = float(ll)
    best = max(lls, key=lambda k: (lls[k], -k))
    return best, lls
