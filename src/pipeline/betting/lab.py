"""馬券戦略の探索（strategy lab）: 戦略を登録して同じ手順で評価し、多数を試した事実を補正した検定で『本当に 100% を超える戦略があるか』を判定する。

- 規則ベースの戦略（箱・流し・フォーメーション・1 点）は、買い目をモデルの確率の順位だけで決める。オッズを使わないので、確定の払戻表で精算しても
  確定オッズの先読みにならず、バックテストが偏らない（払戻率の下限の目安になる）。
- 比較の基準: 同じ構造でランダムに馬を選んだ場合の回収率（モデルの寄与 = 差）。
- 検定: ``reality_check``（White の reality check をレース単位の bootstrap で実装）。『試した戦略の中の最良が 100% を超えて見えるのは、偶然を超えるか』を p 値で出す。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable

import numpy as np
import pandas as pd

UNIT = 100.0


@dataclass(frozen=True)
class Rule:
    """kind: single（1 点）/ box（上位 k 頭の箱）/ nagashi（軸 1 頭 + 相手 m 頭）/ formation（sanrentan: 1 着 a 頭 × 2 着 b 頭 × 3 着 c 頭）。
    order: 順位付けに使う確率（win = 勝率順、top3 = 複勝率順）。"""
    bet: str            # tansho fukusho umaren wide umatan sanrenpuku sanrentan
    kind: str
    k: int = 0          # box の頭数 / nagashi の相手数 / formation の (a, b, c) は a=k
    b: int = 0
    c: int = 0
    order: str = "win"

    def name(self) -> str:
        if self.kind == "single":
            return f"{self.bet}:1点({self.order})"
        if self.kind == "box":
            return f"{self.bet}:箱{self.k}({self.order})"
        if self.kind == "nagashi":
            return f"{self.bet}:軸1-相手{self.k}({self.order})"
        return f"{self.bet}:F{self.k}-{self.b}-{self.c}({self.order})"


def combos(rule: Rule, top: list[int]) -> list[str]:
    """順位付けした馬番のリスト top（先頭が最有力）から、買い目の文字列を作る。足りなければ空。"""
    t = rule.bet
    if rule.kind == "single":
        return [str(top[0])] if top else []
    if rule.kind == "box":
        h = top[: rule.k]
        if len(h) < rule.k:
            return []
        if t in ("tansho", "fukusho"):
            return [str(x) for x in h]
        if t in ("umaren", "wide"):
            return ["-".join(map(str, sorted(p))) for p in combinations(h, 2)]
        if t == "umatan":
            return [f"{a}>{b}" for a, b in permutations(h, 2)]
        if t == "sanrenpuku":
            return ["-".join(map(str, sorted(p))) for p in combinations(h, 3)]
        if t == "sanrentan":
            return [">".join(map(str, p)) for p in permutations(h, 3)]
    if rule.kind == "nagashi":
        if len(top) < rule.k + 1:
            return []
        ax, ps = top[0], top[1: rule.k + 1]
        if t in ("umaren", "wide"):
            return ["-".join(map(str, sorted((ax, p)))) for p in ps]
        if t == "umatan":
            return [f"{ax}>{p}" for p in ps]
        if t == "sanrenpuku":
            return ["-".join(map(str, sorted((ax,) + q))) for q in combinations(ps, 2)]
        if t == "sanrentan":
            return [f"{ax}>{a}>{b}" for a, b in permutations(ps, 2)]
    if rule.kind == "formation" and t == "sanrentan":
        first, second, third = top[: rule.k], top[: rule.k + rule.b], top[: rule.k + rule.b + rule.c]
        if len(third) < rule.k + rule.b + rule.c:
            return []
        out = set()
        for a in first:
            for b in second:
                for c in third:
                    if len({a, b, c}) == 3:
                        out.add(f"{a}>{b}>{c}")
        return sorted(out)
    return []


def random_top(hn: list[int], rng: np.random.Generator) -> list[int]:
    return list(rng.permutation(hn))


def build_orders(pred: pd.DataFrame, hn_map: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    """race_id → {'win': 勝率順の馬番, 'top3': 複勝率順の馬番, 'p1': 最有力の勝率, 'field': 頭数}。pred は race_id・horse_id・p_win・p_top3、hn_map は race_id・horse_id・horse_number。"""
    d = pred.merge(hn_map, on=["race_id", "horse_id"], how="inner")
    out = {}
    for rid, g in d.groupby("race_id", sort=False):
        out[rid] = {"win": g.sort_values("p_win", ascending=False)["horse_number"].astype(int).tolist(), "top3": g.sort_values("p_top3", ascending=False)["horse_number"].astype(int).tolist(),
                    "p1": float(g["p_win"].max()), "field": int(len(g))}
    return out


def payoff_lookup(payoff: pd.DataFrame) -> dict[str, dict[tuple, float]]:
    """race_id → {(bet_type, combo): 払戻金（100 円あたり）}。解析できなかった行・曖昧な行は除く。"""
    d = payoff[payoff["combo"].notna() & payoff["payout"].notna() & payoff["parsed"].astype(bool) & ~payoff["ambiguous"].astype(bool)]
    out: dict[str, dict] = {}
    for rid, bt, cb, pv in zip(d["race_id"], d["bet_type"], d["combo"], d["payout"]):
        out.setdefault(rid, {})[(bt, str(cb))] = float(pv)
    return out


def evaluate_rule(rule: Rule, orders: dict, pay: dict, race_ids: Iterable[str], random_seed: int | None = None, min_p1: float = 0.0) -> pd.DataFrame:
    """レースごとの賭け金・払戻（円）。random_seed を渡すと、順位をランダムにした同じ構造の基準を返す。min_p1 > 0 なら最有力の勝率がそれ以上のレースだけ買う。"""
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    rows = []
    for rid in race_ids:
        o = orders.get(rid)
        if o is None or o["p1"] < min_p1:
            continue
        top = random_top(o["win"], rng) if rng is not None else o[rule.order]
        cs = combos(rule, top)
        if not cs:
            continue
        pr = pay.get(rid, {})
        ret = sum(pr.get((rule.bet, c), 0.0) for c in cs)
        rows.append({"race_id": rid, "stake": UNIT * len(cs), "ret": ret})
    return pd.DataFrame(rows, columns=["race_id", "stake", "ret"])


def roi_ci(d: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    st, rt = d["stake"].to_numpy(), d["ret"].to_numpy()
    n = len(d)
    roi = rt.sum() / max(st.sum(), 1e-12)
    rng = np.random.default_rng(seed)
    b = np.array([(lambda ii: rt[ii].sum() / max(st[ii].sum(), 1e-12))(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {"roi": float(roi), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)), "races": n, "stake_per_race": float(st.mean())}


def reality_check(profit: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """White の reality check。profit: (レース数, 戦略数) の『1 レース当たりの損益（賭け金に対する比 = 回収率 − 1）』。
    帰無仮説: どの戦略の期待損益も 0 以下。統計量 = 戦略ごとの平均損益の標準化の最大値。bootstrap は、平均を引いた損益をレース単位で再標本化して作る。
    戻り値: 最良の戦略の番号・平均・p 値（小さいほど、『試した中の最良が偶然ではない』）。"""
    n, m = profit.shape
    mu = profit.mean(0)
    sd = profit.std(0, ddof=1)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    t_obs = np.sqrt(n) * mu / sd
    centered = (profit - mu) / sd
    rng = np.random.default_rng(seed)
    mx = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, n, n)
        mx[b] = (np.sqrt(n) * centered[ii].mean(0)).max()
    best = int(np.argmax(t_obs))
    return {"best": best, "best_mean": float(mu[best]), "t_obs": float(t_obs[best]), "p_value": float((mx >= t_obs[best]).mean()), "n_strategies": int(m)}
