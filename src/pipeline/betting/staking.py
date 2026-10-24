"""レースごとの購入額の段階づけ（見送り / 500 円 / 1000 円 / 2000 円）。

方策（policy）が決めた「レース内の買い目の配分」を前提に、レース単位の指標から 1 レースの購入額を 4 段階のどれにするか決める。
- 期待回収率 EV: 配分で重みづけした p × オッズ（生）と、等張回帰で較正した値（cal）。確定オッズで選ぶ楽観を実績に合わせて縮める。
- 的中確率 hit: 買い目のどれか 1 つでも当たる確率（上位 3 着の並びの全列挙で厳密に計算）。
- ppos: 払戻が賭け金以上になる確率（元返し以上）。
- 段階の決め方は ``TierRule``（EV と hit のしきい値）。オッズが下がる場合の許容は、EV × (1 − δ) ≥ 1 となる最大の δ（許容低下率）で表す。
金額は 100 円単位に丸める（``round_units``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.pipeline.inference.exotic_probabilities import RankDistribution

STAKES = (0, 500, 1000, 2000)


def round_units(w: np.ndarray, total_yen: int, unit: int = 100) -> np.ndarray:
    """配分 w（合計 1 に正規化）を total_yen に按分して unit 円単位の整数にする（最大剰余法）。合計は必ず total_yen。"""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if total_yen <= 0 or w.sum() <= 0:
        return np.zeros(len(w), dtype=int)
    u = int(total_yen // unit)
    x = w / w.sum() * u
    base = np.floor(x).astype(int)
    rem = u - int(base.sum())
    if rem > 0:
        frac = x - base
        base[np.argsort(-frac, kind="stable")[:rem]] += 1
    return base


def hit_masks(hn: np.ndarray, tickets: list[tuple[str, str]]) -> np.ndarray:
    """(買い目数, n, n, n) の的中判定。順列 (i, j, k) = 1・2・3 着の位置。"""
    n = len(hn)
    pos = {int(h): i for i, h in enumerate(hn)}
    I, J, K = np.indices((n, n, n))
    out = np.zeros((len(tickets), n, n, n), bool)
    for t, (ty, combo) in enumerate(tickets):
        if ty == "tansho":
            out[t] = I == pos[int(combo)]
        elif ty == "fukusho":
            a = pos[int(combo)]
            out[t] = (I == a) | (J == a) | (K == a)
        elif ty == "umaren":
            a, b = (pos[int(x)] for x in combo.split("-"))
            out[t] = ((I == a) & (J == b)) | ((I == b) & (J == a))
        elif ty == "wide":
            a, b = (pos[int(x)] for x in combo.split("-"))
            out[t] = ((I == a) | (J == a) | (K == a)) & ((I == b) | (J == b) | (K == b))
        elif ty == "umatan":
            a, b = (pos[int(x)] for x in combo.split(">"))
            out[t] = (I == a) & (J == b)
    return out


def portfolio_outcomes(hn: np.ndarray, win: np.ndarray, alpha2: float, alpha3: float, tickets: list[tuple[str, str]], w: np.ndarray, odds: np.ndarray) -> dict:
    """配分 w（合計 1）の買い目について、的中確率 hit（どれか 1 つ以上当たる）、元返し以上の確率 ppos、期待払戻倍率 ev を、上位 3 着の並びの全列挙で計算する。"""
    d = RankDistribution(np.asarray(win, dtype=float), alpha2, alpha3)
    p3 = d._p3
    m = hit_masks(hn, tickets)
    w = np.asarray(w, dtype=float)
    ret = np.tensordot(w * np.asarray(odds, dtype=float), m.astype(float), axes=(0, 0))     # (n, n, n) 当たったときの払戻倍率（予算比）
    any_hit = m.any(axis=0)
    return {"hit": float((p3 * any_hit).sum()), "ppos": float((p3 * (ret >= 1.0 - 1e-9)).sum()), "ev": float((p3 * ret).sum())}


def race_features(tickets: pd.DataFrame, dists: dict[str, tuple], w_col: str = "w", cal_col: str | None = "cal_ev") -> pd.DataFrame:
    """買い目の表（race_id・type・combo・p_comb・odds・w）から、レース単位の指標を作る。w が 0 のレースは除く。
    dists: race_id → (馬番の配列, 勝率, α2, α3)。"""
    rows = []
    t = tickets[tickets[w_col] > 1e-9]
    for rid, g in t.groupby("race_id", sort=False):
        if rid not in dists:
            continue
        hn, win, a2, a3 = dists[rid]
        w = g[w_col].to_numpy(dtype=float)
        w = w / w.sum()
        odds = g["odds"].to_numpy(dtype=float)
        o = portfolio_outcomes(hn, win, a2, a3, list(zip(g["type"], g["combo"].astype(str))), w, odds)
        ev_raw = float((w * g["p_comb"].to_numpy() * odds).sum())
        ev_cal = float((w * g[cal_col].to_numpy()).sum()) if cal_col and cal_col in g.columns else ev_raw
        rows.append({"race_id": rid, "ev_raw": ev_raw, "ev_cal": ev_cal, "ev_mix": ev_cal + 1e-3 * ev_raw, "hit": o["hit"], "ppos": o["ppos"], "n_tickets": len(g), "max_w": float(w.max())})
    return pd.DataFrame(rows)


@dataclass
class TierRule:
    """EV（ev_col）と hit のしきい値で 0 / 500 / 1000 / 2000 円を決める。
    見送り: ev < e1 または hit < h0（異常に当たらない）/ 500: e1 ≤ ev < e2 / 1000: e2 ≤ ev < e3 かつ hit ≥ h1 / 2000: ev ≥ e3 かつ hit ≥ h2。
    条件を満たさない上位は1つ下の段階に落とす。"""
    e1: float
    e2: float
    e3: float
    h0: float = 0.15
    h1: float = 0.25
    h2: float = 0.35
    ev_col: str = "ev_cal"
    stakes: tuple = field(default=STAKES)

    def assign(self, f: pd.DataFrame) -> np.ndarray:
        ev, hit = f[self.ev_col].to_numpy(), f["hit"].to_numpy()
        s = np.zeros(len(f), dtype=int)
        s[(ev >= self.e1) & (hit >= self.h0)] = self.stakes[1]
        s[(ev >= self.e2) & (hit >= self.h1)] = self.stakes[2]
        s[(ev >= self.e3) & (hit >= self.h2)] = self.stakes[3]
        return s

    def describe(self) -> dict:
        return {k: getattr(self, k) for k in ("e1", "e2", "e3", "h0", "h1", "h2", "ev_col")}


def quantile_rule(f: pd.DataFrame, shares=(0.4, 0.3, 0.2, 0.1), ev_col: str = "ev_mix", **kw) -> TierRule:
    """参加の割合を固定した規則: shares = (見送り, 500, 1000, 2000) の割合になるよう EV のしきい値を学習側の分布から決める。
    等張回帰で較正した EV は同じ値が多い（階段状）ので、同順位を ev_raw で崩した ev_mix を既定にする（そうしないと分位点のしきい値が重なり、段階が潰れる）。"""
    q = np.cumsum(shares)[:3]
    e1, e2, e3 = (float(np.quantile(f[ev_col], x)) for x in q)
    return TierRule(e1, e2, e3, ev_col=ev_col, **kw)


def settle_tiered(tickets: pd.DataFrame, f: pd.DataFrame, stakes: np.ndarray, w_col: str = "w") -> pd.DataFrame:
    """レースごとの購入額（円）に応じて 100 円単位に按分し、確定の払戻（mult）で精算する。戻り値: race_id・stake・ret・profit（円）。"""
    s = dict(zip(f["race_id"], stakes))
    out = []
    t = tickets[tickets["race_id"].isin([r for r, v in s.items() if v > 0])]
    for rid, g in t.groupby("race_id", sort=False):
        w = g[w_col].to_numpy(dtype=float)
        u = round_units(w, int(s[rid]))
        ret = float((u * g["mult"].to_numpy()).sum()) * 100.0
        out.append({"race_id": rid, "stake": float(u.sum() * 100), "ret": ret})
    d = pd.DataFrame(out, columns=["race_id", "stake", "ret"])
    d["profit"] = d["ret"] - d["stake"]
    return d


def summarize_tiers(f: pd.DataFrame, stakes: np.ndarray, settled: pd.DataFrame) -> pd.DataFrame:
    """段階ごとのレース数・賭け金・払戻・回収率・的中率（払戻 > 0）。"""
    m = f.assign(tier=stakes).merge(settled, on="race_id", how="left").fillna({"stake": 0.0, "ret": 0.0, "profit": 0.0})
    g = m.groupby("tier").agg(races=("race_id", "size"), stake=("stake", "sum"), ret=("ret", "sum"), hit=("ret", lambda s: float((s > 0).mean())), ev_cal=("ev_cal", "mean"), hit_est=("hit", "mean"))
    g["roi"] = g["ret"] / g["stake"].replace(0, np.nan)
    g["profit"] = g["ret"] - g["stake"]
    return g
