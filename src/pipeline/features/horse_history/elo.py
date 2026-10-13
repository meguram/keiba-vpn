"""順位から逐次更新する多頭数 Elo。1 日分のレースは前日までのレーティングで評価し、日の終わりにまとめて更新する。"""
from __future__ import annotations
import numpy as np
import polars as pl

SCALE = 400.0 / np.log(10.0)


def run_elo(runs: pl.DataFrame, k: float = 300.0, init: float = 1500.0) -> pl.DataFrame:
    df = runs.select("date", "race_id", "horse_id", "finish_position").with_row_index("i")
    n = df.height
    horse = df["horse_id"].to_numpy()
    fin = df["finish_position"].to_numpy().astype(float)
    # (date, race) 単位の行インデックス
    order = df.sort(["date", "race_id"])["i"].to_numpy()
    dates = df["date"].to_numpy()[order]
    races = df["race_id"].to_numpy()[order]
    out = {c: np.full(n, np.nan) for c in ("r_pre", "n_pre", "S", "E", "fs_mean", "fs_top3", "fs_std")}
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
                diff = r[:, None] - r[None, :]
                ep = 1.0 / (1.0 + np.exp(-diff / SCALE))
                E = (ep.sum(1) - 0.5) / (N - 1)
                S = (N - fin[idx]) / (N - 1)
                out["S"][idx], out["E"][idx] = S, E
                tot = r.sum()
                out["fs_mean"][idx] = (tot - r) / (N - 1)
                srt = np.sort(r)[::-1]
                for a, ii in enumerate(idx):
                    others = np.delete(r, a)
                    out["fs_top3"][ii] = np.sort(others)[::-1][:3].mean()
                    out["fs_std"][ii] = others.std()
                kk = k * np.where(c < 3, 1.5, 1.0)
                pending.append((hs, kk * (S - E)))
            s = e
        for hs, dl in pending:
            for h, v in zip(hs, dl):
                rating[h] = rating.get(h, init) + v
                cnt[h] = cnt.get(h, 0) + 1
        pos = end
    res = pl.DataFrame({"i": np.arange(n), **{c: v for c, v in out.items()}})
    res = res.with_columns((pl.col("S") - pl.col("E")).alias("res"))
    return df.select("race_id", "horse_id", "i").join(res, on="i").drop("i")
