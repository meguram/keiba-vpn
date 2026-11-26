"""z の効きの年ごとの移り（log.md「調べ R」。記述）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.z_years

- 行と式は z_bands.py と同じ（帯 = 年 2019〜24・帯ごとの市場の曲線を入れる）。
- 主の値は、年ごとの z の傾きの、年に対する直線の傾き（1 年あたり）。区間は開催日の単位（500 回）。
"""
from __future__ import annotations

import json

import numpy as np

from src.research.pedigree_lab import data
from src.research.pedigree_lab.sire_market import irls
from src.research.pedigree_lab.z_bands import load


def main() -> None:
    # 台帳の範囲は z_bands.py の load（feat_v4 の 2016 年からの行を読む）と同じ。使うのは 2019〜24 年だけ
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "z_years", "z の効きの年ごとの移り（記述）"))
    X = load()
    X = X[X["y"].between(2019, 2024)].reset_index(drop=True)
    band = (X["y"] - 2019).to_numpy()
    lq, z = X["lq"].to_numpy(), X["z"].to_numpy()
    cols = [np.ones(len(X)), lq, lq ** 2, z]
    for b in range(1, 6):
        ind = (band == b).astype(float)
        cols += [ind, z * ind, lq * ind, lq ** 2 * ind]
    M = np.column_stack(cols); y = X["win"].to_numpy(float)
    yrs = np.arange(6) - 2.5

    def slopes(c):
        return np.array([c[3]] + [c[3] + c[5 + 4 * (b - 1)] for b in range(1, 6)])
    s0 = slopes(irls(M, y))
    du, di = np.unique(X["date"].dt.strftime("%Y%m%d"), return_inverse=True)
    idx = [np.flatnonzero(di == j) for j in range(len(du))]
    rng = np.random.default_rng(7)
    bs = []
    for _ in range(500):
        ii = np.concatenate([idx[j] for j in rng.integers(0, len(du), len(du))])
        bs.append(slopes(irls(M[ii], y[ii])))
    bs = np.array(bs)
    tr = (bs * yrs).sum(axis=1) / (yrs ** 2).sum()
    res = {"走り": int(len(X)),
           "年ごとの z の傾き": {str(2019 + b): [round(float(s0[b]), 4), np.percentile(bs[:, b], [2.5, 97.5]).round(4).tolist()] for b in range(6)},
           "1 年あたりの移り": [round(float((s0 * yrs).sum() / (yrs ** 2).sum()), 4), np.percentile(tr, [2.5, 97.5]).round(4).tolist()]}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    path = data.OUT / "z_years.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
