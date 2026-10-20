"""tp-54: 追走難度 v2（EASE2）が v5 の勝率予測に足せる情報を持つかの一次選別（条件付きロジット）。

python -m src.scripts.data.tp54_stack      # ホストで実行。v5 構成 d（V4 + tl_ 再取得 + CY + ALL_BEST）の OOF p_win の log に、γ×(標準化した EASE2 の列) を足した softmax。
- γ は学習年で最尤、検証年のレース単位の負の対数尤度の差（負＝改善）をレース単位ブートストラップで区間にする。
- 分け方: 2022〜23 年で当てはめ → 2024 年、2022 年で当てはめ → 2023 年。全体と『履歴 0〜2 走』の部分集合。EASE2 は年別 leave-one-year-out の予測（対象年の結果は学習に不使用）。
- 全列を LightGBM に入れる本評価（tp54_v5_eval）とは別の軽い選別。市場オッズ・人気は使わない。test 2025・2026 年は使わない。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

V = Path(os.environ.get("V4_DIR") or "/home/jovyan/work/keiba-vpn/data/local/modeling/training_pipeline_v4")
TP54_OUT = Path(os.environ.get("TP54_OUT") or "/home/jovyan/work/tp54_out")
KEY = ["race_id", "horse_id"]
COLS = ["EASE2_drain", "EASE2_demand", "EASE2_capacity", "EASE2_full", "EASE2_pct", "EASE2_vs_prev"]


def race_nll(codes: np.ndarray, starts: np.ndarray, a: np.ndarray, win: np.ndarray) -> np.ndarray:
    mx = np.maximum.reduceat(a, starts)
    lse = np.log(np.add.reduceat(np.exp(a - mx[codes]), starts)) + mx
    num = np.add.reduceat(np.where(win, a, 0.0), starts)
    return lse - num


def fit_gamma(fc, fst, s, win, Z: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    k = Z.shape[1]
    f = lambda g: float(race_nll(fc, fst, s + Z @ g, win).mean()) + 1e-4 * float(g @ g)
    r = minimize(f, np.zeros(k), method="L-BFGS-B", bounds=[(-1.5, 1.5)] * k)
    return r.x


def main() -> None:
    base = pd.read_parquet(V / "v5_select/oof_d_CY_BEST.parquet")
    for c in KEY:
        base[c] = base[c].astype(str)
    h = pd.read_parquet(TP54_OUT / "horse_ease.parquet", columns=KEY + ["finish_position", "w_n", "year"])
    g = pd.read_parquet(TP54_OUT / "groups/EASE2.parquet")
    for t in (h, g):
        for c in KEY:
            t[c] = t[c].astype(str)
    m = base.merge(h[KEY + ["finish_position", "w_n"]], on=KEY, how="left").merge(g, on=KEY, how="left").sort_values(["race_id", "horse_id"], kind="stable").reset_index(drop=True)
    codes_all = pd.factorize(m["race_id"])[0]
    year = m["year"].to_numpy()
    win = m["finish_position"].to_numpy() == 1
    ok_race = pd.Series(win).groupby(codes_all).transform("sum").to_numpy() == 1
    s = np.log(np.clip(m["p_win"].to_numpy(float), 1e-9, 1))
    nprior = m["w_n"].to_numpy(float)
    rng = np.random.default_rng(0)

    def sub(mask):
        idx = np.flatnonzero(mask & ok_race)
        multi = pd.Series(1, index=idx).groupby(codes_all[idx]).transform("size").to_numpy() >= 2
        idx = idx[multi]
        c = pd.factorize(codes_all[idx])[0]
        st = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
        return idx, c, st

    res: dict = {"note": "v5 構成 d の log p_win に γ×標準化した EASE2 の列を足した条件付きロジット（負＝改善）", "results": {}}
    for split, (fit_y, test_y) in {"fit22-23→test24": ((2022, 2023), (2024,)), "fit22→test23": ((2022,), (2023,))}.items():
        for sname, smask in (("全体", np.ones(len(m), bool)), ("履歴0〜2走", nprior <= 2)):
            fi, fc, fst = sub(np.isin(year, fit_y) & smask)
            ti, tc, tst = sub(np.isin(year, test_y) & smask)
            if len(fi) < 5000 or len(ti) < 3000:
                continue
            base_t = race_nll(tc, tst, s[ti], win[ti])
            rows = []
            sets = {c: [c] for c in COLS}
            sets["core3(drain,demand,vs_prev)"] = ["EASE2_drain", "EASE2_demand", "EASE2_vs_prev"]
            sets["all"] = COLS
            for name, cols in sets.items():
                Z = np.column_stack([m[c].to_numpy(float) for c in cols])
                mu, sd = np.nanmean(Z[fi], axis=0), np.nanstd(Z[fi], axis=0)
                Zs = np.clip(np.nan_to_num((Z - mu) / np.where(sd > 1e-9, sd, 1), nan=0.0), -4, 4)
                gam = fit_gamma(fc, fst, s[fi], win[fi], Zs[fi])
                dt = race_nll(tc, tst, s[ti] + Zs[ti] @ gam, win[ti]) - base_t
                bs = np.array([dt[rng.integers(0, len(dt), len(dt))].mean() for _ in range(300)])
                rows.append({"features": name, "gamma": [round(float(x), 4) for x in gam], "d_nll": round(float(dt.mean()), 6), "hi95": round(float(np.quantile(bs, 0.975)), 6), "lo95": round(float(np.quantile(bs, 0.025)), 6),
                             "n_races": int(len(dt)), "t": round(float(dt.mean() / (dt.std() / np.sqrt(len(dt)))), 2)})
            res["results"][f"{split}|{sname}"] = rows
            print(split, sname, len(ti), json.dumps(rows, ensure_ascii=False)[:600], flush=True)
    (TP54_OUT / "stack.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
