"""T-60 の再現オッズ（bet_cycles.load_drift・megu_cycles.simulate_pre と同じ作り。置き場は本体の手元のファイル）。

2026-10 以降の本物の T-60 と確定の比 d = log(確定 / T-60) を、レースごとに確定の人気の順に並べて丸ごと借り、
過去のレースの確定オッズに exp(−d) を当てる（1/オッズの和は確定に合わせる）。読むのはオッズだけ（着順は読まない）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path("/home/jovyan/work/keiba-vpn")
PRE = MAIN / "data/local/research/persona1/pre_odds.parquet"
TABLES = MAIN / "data/page_reference/tables"


def load_drift(phase: str = "t60") -> list[np.ndarray]:
    pre = pd.read_parquet(PRE)
    pre["race_id"] = pre["race_id"].astype(str)
    pre = pre[(pre["phase"] == phase) & (pd.to_numeric(pre["win_odds"], errors="coerce") > 0)][["race_id", "horse_number", "win_odds"]]
    fl = pd.read_parquet(TABLES / "2026" / "race_odds_flat.parquet", columns=["race_id", "horse_number", "win_odds", "popularity"])
    fl = fl.rename(columns={"win_odds": "odds"})
    fl["race_id"] = fl["race_id"].astype(str)
    pre["horse_number"] = pd.to_numeric(pre["horse_number"], errors="coerce")
    fl["horse_number"] = pd.to_numeric(fl["horse_number"], errors="coerce")
    m = pre.merge(fl, on=["race_id", "horse_number"])
    m = m[(m["odds"] > 0) & (m["popularity"] > 0)].copy()
    m["d"] = np.log(m["odds"].to_numpy(float) / pd.to_numeric(m["win_odds"]).to_numpy(float))
    return [g.sort_values("popularity")["d"].to_numpy() for _, g in m.groupby("race_id") if len(g) >= 5]


def simulate(race_id: np.ndarray, odds: np.ndarray, pop: np.ndarray, pool: list[np.ndarray], n: int = 5) -> np.ndarray:
    """行はレースごとに連続している前提。戻り値 [n, 行]。"""
    pop = np.nan_to_num(pop.astype(float), nan=99.0)
    out = np.empty((n, len(odds)))
    starts = np.flatnonzero(np.r_[True, race_id[1:] != race_id[:-1]])
    ends = np.r_[starts[1:], len(odds)]
    for k in range(n):
        rng = np.random.default_rng(1000 + k)
        for s, e in zip(starts, ends):
            src = pool[int(rng.integers(len(pool)))]
            rk = np.clip(np.where(pop[s:e] < 99, pop[s:e], len(src)).astype(int) - 1, 0, len(src) - 1)
            o = odds[s:e] * np.exp(-src[rk])
            out[k, s:e] = o * (np.sum(1.0 / o) / np.sum(1.0 / odds[s:e]))
    return out
