"""tp-70 確率の較正（予測と実績の差）— 共通部品。

入力は tp-36 の frame_all（V4 の OOF 予測 p_win/p_top2/p_top3、確定の単勝オッズ、複勝オッズ・払戻）と rank_tbl の確定着順。
市場の確率は単勝オッズのみから作る（複勝オッズは 2026 年に無いため使わない）: q = 正規化した 1/オッズ → Harville で連対・複勝の市場確率。
市場オッズ・人気は『予測モデルの特徴量』にはしない。較正と馬券判断にだけ使う（Benter 型ブレンドと同じ位置づけ）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

EPS = 1e-6
TARGETS = ("win", "top2", "top3")
HIT = {"win": "h1", "top2": "h2", "top3": "h3"}
PRED = {"win": "p_win", "top2": "p_top2", "top3": "p_top3"}
MKT = {"win": "m_win", "top2": "m_top2", "top3": "m_top3"}
KSUM = {"win": 1.0, "top2": 2.0, "top3": 3.0}
POP_CLASSES = [(0, 1, "1人気"), (1, 2, "2人気"), (2, 3, "3人気"), (3, 5, "4-5人気"), (5, 9, "6-9人気"), (9, 99, "10人気以下")]
MIN_FIELD = 8


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def load_base(frame: str | Path, rank_dir: str | Path) -> pd.DataFrame:
    """frame_all + 確定着順 → 1 馬 1 行の評価表。着順が取れない馬のいるレースと 8 頭未満のレースは除く。"""
    d = pd.read_parquet(frame)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    rk = pd.concat([pd.read_parquet(Path(rank_dir) / f"rank{y}.parquet") for y in (2022, 2023, 2024, 2025, 2026)])
    rk = rk.assign(race_id=rk["race_id"].astype(str), horse_id=rk["horse_id"].astype(str))[["race_id", "horse_id", "rank"]].drop_duplicates(["race_id", "horse_id"])
    m = d.merge(rk, on=["race_id", "horse_id"], how="left")
    bad = m.groupby("race_id")["rank"].apply(lambda s: bool(s.isna().any()))
    m = m[~m["race_id"].map(bad)].copy()
    m["h1"] = (m["rank"] == 1).astype(int)
    m["h2"] = ((m["rank"] > 0) & (m["rank"] <= 2)).astype(int)
    m["h3"] = ((m["rank"] > 0) & (m["rank"] <= 3)).astype(int)
    m["n"] = m.groupby("race_id")["horse_id"].transform("size")
    m["pop"] = m["popularity"].where(m["popularity"].notna(), m.groupby("race_id")["win_odds"].rank(method="first"))
    m = m[(m["n"] >= MIN_FIELD) & m["win_odds"].notna() & (m["win_odds"] > 0) & m["p_top3"].notna()].copy()
    m = m.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    return add_market(m)


def add_market(m: pd.DataFrame) -> pd.DataFrame:
    """単勝オッズから市場の勝率・連対率・複勝率（Harville）と、モデルとの食い違いを付ける。"""
    m = m.copy()
    inv = 1.0 / m["win_odds"].astype(float)
    m["m_win"] = inv / inv.groupby(m["race_id"]).transform("sum")
    t2 = np.zeros(len(m))
    t3 = np.zeros(len(m))
    for _, idx in m.groupby("race_id").indices.items():
        w = m["m_win"].to_numpy()[idx]
        t2[idx] = harville_top2_prob(w)
        t3[idx] = harville_top3_prob(w)
    m["m_top2"] = t2
    m["m_top3"] = t3
    m["lag_gap3"] = logit(m["p_top3"]) - logit(m["m_top3"])
    m["month"] = pd.to_datetime(m["date"]).dt.month
    m["venue"] = m["race_id"].str[4:6]
    return m


def scale_to_sum(p: np.ndarray, k: float) -> np.ndarray:
    """レース内の確率の合計を k に合わせる（logit に共通のオフセットを足して二分法）。n<=k のときは上限 1 で打ち切り。"""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k = min(k, len(p) - 1e-3)
    z = logit(p)
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).sum() > k:
            hi = mid
        else:
            lo = mid
    return sigmoid(z + (lo + hi) / 2)


def normalize_races(df: pd.DataFrame, col: str, target: str) -> np.ndarray:
    out = np.empty(len(df))
    for _, idx in df.groupby("race_id").indices.items():
        out[idx] = scale_to_sum(df[col].to_numpy()[idx], KSUM[target])
    return out


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece(y, p, bins: int = 10) -> float:
    """予測確率の分位ビン（等件数）での |予測−実績| の件数加重平均。"""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    if np.ptp(p) == 0:
        return float(abs(y.mean() - p.mean()))
    q = pd.qcut(pd.Series(p), bins, duplicates="drop", labels=False)
    g = pd.DataFrame({"y": y, "p": p, "q": q.to_numpy()}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


def pop_class(pop: pd.Series) -> pd.Series:
    bins = [0] + [b for _, b, _ in POP_CLASSES]
    return pd.cut(pop, bins, labels=[c for _, _, c in POP_CLASSES])


def race_boot_ci(df: pd.DataFrame, fn, reps: int = 300, seed: int = 0) -> tuple[float, float, float]:
    """レース単位ブートストラップ。fn(df_subset) -> float。"""
    rng = np.random.default_rng(seed)
    ids = df["race_id"].unique()
    groups = {k: v for k, v in df.groupby("race_id").indices.items()}
    vals = []
    for _ in range(reps):
        pick = rng.choice(ids, len(ids))
        idx = np.concatenate([groups[k] for k in pick])
        vals.append(fn(df.iloc[idx]))
    return float(fn(df)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
