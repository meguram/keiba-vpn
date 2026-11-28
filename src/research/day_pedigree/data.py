"""行・日の順・市場の見込み（段 1-a・1-b）。手元の parquet だけ（GCS を読まない・費用 0 円）。

- 行: 血統の研究の runs.parquet（平地の芝・ダートだけ。障害は入っていない）を 2024-12-31 まで読み、pedigree_lab.data._prep で取消・除外を落とす。
- 日 = 日付 × 場 × 芝ダ。k = その日のその芝ダの中の R 番号の順（1 始まり）。
- 8 頭以下のレースは使わない（5 着内の情報がほぼ無い）。2016 年は燃やしの年（事前と登場率の溜め）。
- 分け: 開催（race_id の先頭 8 桁）の無作為 70/30。aptitude-group の common.split と同じ作り・同じ seed（0 = 探す・1 = 確かめ）。
- π5H: 確定の単勝の q（1/odds を正規化）から、Harville（Plackett-Luce）の 5 着内の確率をモンテカルロで出す。着順を使わない。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data as pl

OUT = Path(os.environ.get("DAYPED_OUT", "/home/jovyan/work/day_pedigree_out"))
DEV_END = pd.Timestamp("2024-12-31")
SEED = 20261127
MIN_FIELD = 9           # 8 頭以下は使わない
TOP = 5
TOKENS: list = []


def open_token(purpose: str, version: str, reason: str, period=("2016-01-01", "2024-12-31")) -> None:
    from src.research.day_pedigree import day_ledger
    TOKENS.append(day_ledger.open_token(period, purpose, version, reason))


def finish_all(result_path: str | None = None) -> None:
    while TOKENS:
        try:
            TOKENS.pop().finish(result_path=result_path)
        except Exception as exc:  # 記録の失敗で研究を止めない
            print(f"[台帳] finish に失敗: {exc}")


def split(meetings: pd.Series) -> np.ndarray:
    """開催の単位の無作為の分け。0 = 探す（70%）、1 = 確かめ（30%）。aptitude_group.common.split と同じ。"""
    u = np.array(sorted(pd.unique(meetings.astype(str))))
    perm = np.random.default_rng(SEED).permutation(len(u))
    lab = np.zeros(len(u), dtype=np.int8)
    lab[perm[int(len(u) * 0.70):]] = 1
    return meetings.astype(str).map(dict(zip(u, lab))).to_numpy()


def load_rows() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "surface", "distance", "track_condition", "race_class", "grade", "course_type",
            "finish_position", "horse_number", "horse_id", "odds", "popularity", "meeting"]
    r = pd.read_parquet(pl.OUT / "runs.parquet", columns=cols, filters=[("date", "<=", DEV_END)])
    assert r["date"].max() <= DEV_END
    r = pl._prep(r)
    r = r[r["n_run"] >= MIN_FIELD].copy()
    r["surf"] = np.where(r["surface"].eq("芝"), "T", "D")
    r["rno"] = r["race_id"].str[-2:].astype(int)
    r["day"] = r["date"].dt.strftime("%Y%m%d") + "_" + r["venue"].astype(str) + "_" + r["surf"]
    rk = r.drop_duplicates("race_id")[["race_id", "day", "rno"]].sort_values(["day", "rno"])
    rk["k"] = rk.groupby("day").cumcount() + 1
    rk["k_last"] = rk.groupby("day")["k"].transform("max")
    r = r.merge(rk[["race_id", "k", "k_last"]], on="race_id", how="left")
    r["part"] = split(r["meeting"])
    r["year"] = r["date"].dt.year
    o = pd.to_numeric(r["odds"], errors="coerce")
    inv = 1.0 / o.where(o > 0)
    inv = inv.fillna(inv.groupby(r["race_id"]).transform("min") * 0.5)       # オッズの欠けは、そのレースの最小の半分（大穴の扱い）
    r["q"] = inv / inv.groupby(r["race_id"]).transform("sum")
    r = r.sort_values(["day", "k", "horse_number"]).reset_index(drop=True)
    return r


def harville_top(r: pd.DataFrame, top: int = TOP, n_sim: int = 2000, seed: int = 7) -> np.ndarray:
    """Harville（Plackett-Luce）の上位 top 頭に入る確率。指数の競争 T_j = E_j / q_j の小さい順で引く。着順を使わない。"""
    out = np.zeros(len(r))
    rng = np.random.default_rng(seed)
    idx = r.groupby("race_id", sort=False).indices
    by_n: dict[int, list[np.ndarray]] = {}
    for ii in idx.values():
        by_n.setdefault(len(ii), []).append(np.sort(ii))
    q = r["q"].to_numpy(float)
    for n, lst in by_n.items():
        m = min(top, n)
        for c0 in range(0, len(lst), 200):
            chunk = np.stack(lst[c0:c0 + 200])                     # [R, n]
            Q = q[chunk][:, None, :]                                # [R, 1, n]
            T = rng.exponential(size=(chunk.shape[0], n_sim, n)) / Q
            thr = np.partition(T, m - 1, axis=2)[:, :, m - 1:m]     # m 番目の小ささ
            out[chunk] = (T <= thr).mean(axis=1)
    return out
