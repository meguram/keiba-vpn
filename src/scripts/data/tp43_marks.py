"""tp-43 レースの性質（堅い/荒れやすい）で印（◎○▲）の付け方と買い方を変える方式の評価。

python -m src.scripts.data.tp43_marks build   # 1 レース×(印方式×買い方)の回収率の表を作る（軽い）
python -m src.scripts.data.tp43_marks analyze # 荒れ度の定義×区分ごとの回収率・交互作用・reality check・破産確率
python -m src.scripts.data.tp43_marks final2026 --rules <analyze が出した json>  # 2026 年を一度だけ

入力: tp-36 の frame_all.parquet（2022〜2026 年の V4 予測 + 確定の単勝・複勝オッズ + 混合確率 p_blend。2022 年の p_blend は無い）と payoff.parquet（確定払戻）。
市場オッズは馬券の判断（EV）と精算にだけ使い、特徴量にはしない。1 レースの賭け金は一定（買い目で均等割り）。母数はレース単位。
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
TP = BET / "tp36"
OUT = BET / "tp43"

DESIGN_YEARS = (2022, 2023)
CONFIRM_YEAR = 2024
REF_YEAR = 2025
MIN_HORSES = 6
TAUS = (0.25, 0.35, 0.45)

MARKS = ["M1_pwin", "M2_ptop3", "M3m_evwin_model", "M3b_evwin_blend", "M4_evplace",
         "M5b_t25", "M5b_t35", "M5b_t45", "M5m_t35", "M7_ev_x_top3"]
MARK_JP = {
    "M1_pwin": "勝率順（p_win）",
    "M2_ptop3": "複勝確率順（p_top3）",
    "M3m_evwin_model": "単勝期待値順（モデル単体 p_win×オッズ）",
    "M3b_evwin_blend": "単勝期待値順（混合 p_blend×オッズ）",
    "M4_evplace": "複勝期待値順（p_top3×複勝オッズ中央）",
    "M5b_t25": "期待値（混合）順、ただし複勝確率 25% 以上を優先",
    "M5b_t35": "期待値（混合）順、ただし複勝確率 35% 以上を優先",
    "M5b_t45": "期待値（混合）順、ただし複勝確率 45% 以上を優先",
    "M5m_t35": "期待値（モデル単体）順、ただし複勝確率 35% 以上を優先",
    "M7_ev_x_top3": "混合の期待値 × 複勝確率の積",
}
BLEND_MARKS = {"M3b_evwin_blend", "M5b_t25", "M5b_t35", "M5b_t45", "M7_ev_x_top3"}

# 荒れ度の定義（値が大きいほど荒れやすい向きにそろえる）
DEFS = ["ent", "max", "fav", "mkt", "div", "n"]
DEF_JP = {
    "ent": "モデルの勝率の分散度（エントロピー）",
    "max": "モデルの最大勝率の低さ",
    "fav": "上位3人気のモデル勝率合計の低さ",
    "mkt": "市場の勝率の分散度（エントロピー）",
    "div": "モデルと市場の乖離（TV 距離）",
    "n": "頭数",
}


def _nag(o, k):
    return [frozenset((o[0], x)) for x in o[1:k]]


def _box(o, k, r):
    return [frozenset(c) for c in combinations(o[:k], r)]


def _axis3(o, k):
    return [frozenset((o[0], a, b)) for a, b in combinations(o[1:k], 2)]


SCHEMES: dict[str, tuple[str, callable]] = {
    "tan1": ("tansho", lambda o: [o[0]]),
    "fuku1": ("fukusho", lambda o: [o[0]]),
    "umaren_n3": ("umaren", lambda o: _nag(o, 3)),
    "umaren_n4": ("umaren", lambda o: _nag(o, 4)),
    "umaren_n5": ("umaren", lambda o: _nag(o, 5)),
    "umaren_b3": ("umaren", lambda o: _box(o, 3, 2)),
    "umaren_b4": ("umaren", lambda o: _box(o, 4, 2)),
    "wide_n3": ("wide", lambda o: _nag(o, 3)),
    "wide_n4": ("wide", lambda o: _nag(o, 4)),
    "wide_n5": ("wide", lambda o: _nag(o, 5)),
    "wide_b3": ("wide", lambda o: _box(o, 3, 2)),
    "wide_b4": ("wide", lambda o: _box(o, 4, 2)),
    "umatan_n3": ("umatan", lambda o: [(o[0], x) for x in o[1:3]]),
    "umatan_n4": ("umatan", lambda o: [(o[0], x) for x in o[1:4]]),
    "umatan_n5": ("umatan", lambda o: [(o[0], x) for x in o[1:5]]),
    "sanpuku_b3": ("sanrenpuku", lambda o: _box(o, 3, 3)),
    "sanpuku_a4": ("sanrenpuku", lambda o: _axis3(o, 4)),
    "sanpuku_b4": ("sanrenpuku", lambda o: _box(o, 4, 3)),
    "sanpuku_a5": ("sanrenpuku", lambda o: _axis3(o, 5)),
    "sanpuku_b5": ("sanrenpuku", lambda o: _box(o, 5, 3)),
}
SCHEME_NAMES = list(SCHEMES)
BET_CLASS = {s: v[0] for s, v in SCHEMES.items()}
COMBOS = [(m, s) for m in MARKS for s in SCHEME_NAMES]
COMBO_IDX = {c: i for i, c in enumerate(COMBOS)}
CLASS_JP = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連", "wide": "ワイド", "umatan": "馬単", "sanrenpuku": "3連複"}
SCHEME_JP = {
    "tan1": "単勝 ◎1点", "fuku1": "複勝 ◎1点",
    "umaren_n3": "馬連 ◎流し(○▲)", "umaren_n4": "馬連 ◎流し(4頭)", "umaren_n5": "馬連 ◎流し(5頭)", "umaren_b3": "馬連 3頭ボックス", "umaren_b4": "馬連 4頭ボックス",
    "wide_n3": "ワイド ◎流し(○▲)", "wide_n4": "ワイド ◎流し(4頭)", "wide_n5": "ワイド ◎流し(5頭)", "wide_b3": "ワイド 3頭ボックス", "wide_b4": "ワイド 4頭ボックス",
    "umatan_n3": "馬単 ◎→○▲", "umatan_n4": "馬単 ◎→3頭", "umatan_n5": "馬単 ◎→4頭",
    "sanpuku_b3": "3連複 3頭ボックス", "sanpuku_a4": "3連複 ◎軸+4頭から2頭", "sanpuku_b4": "3連複 4頭ボックス", "sanpuku_a5": "3連複 ◎軸+5頭から2頭", "sanpuku_b5": "3連複 5頭ボックス",
}


# ------------------------------------------------------------------ データ
def load_frame(years: tuple[int, ...]) -> pd.DataFrame:
    d = pd.read_parquet(TP / "frame_all.parquet")
    d = d[d["year"].isin(years)].copy()
    d["race_id"] = d["race_id"].astype(str)
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce").astype(int)
    return d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)


def load_payoffs(race_ids: set[str]) -> tuple[dict, set[str]]:
    """race_id → {bet_type: {組合せ: 払戻(100 円あたり)}}、除外するレース（曖昧・解析不能）。"""
    pay = pd.read_parquet(BET / "payoff.parquet")
    pay = pay[pay["race_id"].isin(race_ids)]
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    pay = pay[~pay["race_id"].isin(bad)]
    out: dict = {}
    for r in pay.itertuples(index=False):
        bt, c = r.bet_type, str(r.combo)
        if bt in ("tansho", "fukusho"):
            key = int(c)
        elif bt == "umatan":
            key = tuple(int(x) for x in c.split(">"))
        elif bt in ("umaren", "wide", "sanrenpuku"):
            key = frozenset(int(x) for x in c.split("-"))
        else:
            continue
        out.setdefault(r.race_id, {}).setdefault(bt, {})[key] = float(r.payout)
    return out, bad


def payoffs_from_json(hits: dict) -> tuple[dict, bool]:
    """tp36_final2026.parse_payoff_json の結果（組合せはタプル）を、このモジュールのキーに直す。壊れた券種があれば True。"""
    out: dict = {}
    for bt, m in hits.items():
        if bt == "_broken" or bt not in ("tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"):
            continue
        for combo, p in m.items():
            if bt in ("tansho", "fukusho"):
                key = int(combo[0])
            elif bt == "umatan":
                key = tuple(combo)
            else:
                key = frozenset(combo)
            out.setdefault(bt, {})[key] = float(p)
    broken = bool(hits.get("_broken", set()) & {"tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"})
    return out, broken


# ------------------------------------------------------------------ 荒れ度・印
def _entropy(p: np.ndarray) -> float:
    n = len(p)
    return float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(n))


def race_features(g: pd.DataFrame) -> dict:
    p, q = g["p_win"].to_numpy(float), g["q_win"].to_numpy(float)
    pop = np.argsort(g["win_odds"].to_numpy(float), kind="stable")[:3]
    return {
        "n": len(g),
        "ent": _entropy(p / p.sum()),
        "max": -float(p.max() / p.sum()),
        "fav": -float(p[pop].sum() / p.sum()),
        "mkt": _entropy(q / q.sum()),
        "div": float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum()),
    }


def mark_scores(g: pd.DataFrame) -> dict[str, np.ndarray | None]:
    """印方式ごとの点数（大きいほど上位）。必要な列が無ければ None。"""
    pw, pt3 = g["p_win"].to_numpy(float), g["p_top3"].to_numpy(float)
    odds = g["win_odds"].to_numpy(float)
    pm = g["place_mid"].to_numpy(float) if "place_mid" in g else np.full(len(g), np.nan)
    pb = g["p_blend"].to_numpy(float) if "p_blend" in g else np.full(len(g), np.nan)
    has_b, has_p = bool(np.isfinite(pb).all()), bool(np.isfinite(pm).all())
    ev_m = pw * odds
    out: dict[str, np.ndarray | None] = {"M1_pwin": pw, "M2_ptop3": pt3, "M3m_evwin_model": ev_m,
                                         "M4_evplace": pt3 * pm if has_p else None}
    ev_b = pb * odds if has_b else None
    out["M3b_evwin_blend"] = ev_b
    for tau in TAUS:
        out[f"M5b_t{int(tau * 100)}"] = (1e4 * (pt3 >= tau) + ev_b) if has_b else None
    out["M5m_t35"] = 1e4 * (pt3 >= 0.35) + ev_m
    out["M7_ev_x_top3"] = ev_b * pt3 if has_b else None
    return out


def order_by(score: np.ndarray, nums: np.ndarray) -> list[int]:
    idx = np.argsort(-score, kind="stable")
    return [int(nums[i]) for i in idx]


def race_returns(order: list[int], pay: dict) -> np.ndarray:
    """1 レースの全買い方の払戻倍率（賭け金 1 に対する戻り）。買い目の払戻/100 の平均。"""
    r = np.zeros(len(SCHEME_NAMES), dtype=np.float32)
    for i, s in enumerate(SCHEME_NAMES):
        bt, f = SCHEMES[s]
        keys = f(order)
        m = pay.get(bt, {})
        r[i] = sum(m.get(k, 0.0) for k in keys) / 100.0 / len(keys)
    return r


def build(frame: pd.DataFrame, payoffs: dict, broken: set[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    broken = broken or set()
    feats, rows, rids = [], [], []
    for rid, g in frame.groupby("race_id", sort=False):
        if rid in broken or rid not in payoffs or len(g) < MIN_HORSES:
            continue
        nums = g["horse_number"].to_numpy()
        sc = mark_scores(g)
        R = np.full(len(COMBOS), np.nan, dtype=np.float32)
        for m in MARKS:
            if sc[m] is None:
                continue
            ret = race_returns(order_by(sc[m], nums), payoffs[rid])
            R[COMBO_IDX[(m, SCHEME_NAMES[0])]: COMBO_IDX[(m, SCHEME_NAMES[0])] + len(SCHEME_NAMES)] = ret
        f = race_features(g)
        f.update({"race_id": rid, "year": int(g["year"].iloc[0])})
        feats.append(f)
        rows.append(R)
    return pd.DataFrame(feats), np.vstack(rows) if rows else np.zeros((0, len(COMBOS)), dtype=np.float32)


# ------------------------------------------------------------------ 分析
def cutoffs(rf: pd.DataFrame, years=DESIGN_YEARS) -> dict[str, tuple[float, float]]:
    d = rf[rf["year"].isin(years)]
    return {k: (float(d[k].quantile(1 / 3)), float(d[k].quantile(2 / 3))) for k in DEFS}


def buckets(rf: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """荒れ度の区分: 0=堅い（下位 3 分の 1）、1=中、2=荒れやすい（上位 3 分の 1）。n は同値が多いので >= / <= で切る。"""
    out = pd.DataFrame(index=rf.index)
    for k in DEFS:
        lo, hi = cuts[k]
        v = rf[k].to_numpy()
        out[k] = np.where(v <= lo, 0, np.where(v >= hi, 2, 1))
    return out


def boot_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summarize_cell(R: np.ndarray, mask: np.ndarray, col: int) -> dict:
    x = R[mask, col]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    roi, lo, hi = boot_mean(x, 600)
    return {"n": int(len(x)), "roi": roi, "lo": lo, "hi": hi, "hit": float((x > 0).mean())}


def derived_m6(R: np.ndarray, bk: np.ndarray, alt_mark: str, base_mark: str = "M1_pwin") -> np.ndarray:
    """荒れ（区分 2）のレースだけ alt_mark の印、それ以外は base_mark の印。戻り値: レース × 買い方。"""
    a = R[:, COMBO_IDX[(base_mark, SCHEME_NAMES[0])]: COMBO_IDX[(base_mark, SCHEME_NAMES[0])] + len(SCHEME_NAMES)]
    b = R[:, COMBO_IDX[(alt_mark, SCHEME_NAMES[0])]: COMBO_IDX[(alt_mark, SCHEME_NAMES[0])] + len(SCHEME_NAMES)]
    return np.where((bk == 2)[:, None], b, a)


ALT_MARKS = ["M5b_t35", "M5m_t35", "M7_ev_x_top3", "M4_evplace"]


def strategy_matrix(R: np.ndarray, bk: pd.DataFrame, test: np.ndarray) -> tuple[list[tuple], np.ndarray]:
    """reality check 用: 戦略 × レースの超過損益（戻り−1。賭けないレースは 0）。test はレースの真偽マスク。"""
    names, cols = [], []
    for i, c in enumerate(COMBOS):
        x = R[:, i]
        names.append(("base", c[0], c[1]))
        cols.append(np.where(np.isfinite(x), x - 1.0, 0.0))
    for d in DEFS:
        b = bk[d].to_numpy()
        for side, v in (("堅", 0), ("荒", 2)):
            for i, c in enumerate(COMBOS):
                x = R[:, i]
                names.append((f"cond_{d}_{side}", c[0], c[1]))
                cols.append(np.where(np.isfinite(x) & (b == v), x - 1.0, 0.0))
        for alt in ALT_MARKS:
            M6 = derived_m6(R, b, alt)
            for j, s in enumerate(SCHEME_NAMES):
                names.append((f"m6_{d}_{alt}", "M1/" + alt, s))
                cols.append(np.where(np.isfinite(M6[:, j]), M6[:, j] - 1.0, 0.0))
    X = np.stack(cols, axis=1)[test].astype(np.float32)
    return names, X


def reality_check(X: np.ndarray, n_boot: int = 500, seed: int = 0, min_bets: int = 150) -> dict:
    """White の reality check（スチューデント化）。H0: どの戦略の平均超過損益（1 レースあたり）も 0 以下。"""
    n = X.shape[0]
    nz = (X != 0).sum(0)
    keep = nz >= min_bets
    Xk = X[:, keep]
    mean = Xk.mean(0)
    sd = Xk.std(0, ddof=1) + 1e-9
    t_obs = mean / (sd / np.sqrt(n))
    rng = np.random.default_rng(seed)
    best_boot = np.empty(n_boot)
    Xk2 = Xk ** 2
    for b in range(n_boot):
        cnt = rng.multinomial(n, np.ones(n) / n).astype(np.float32)
        mb = cnt @ Xk / n
        sdb = np.sqrt(np.maximum(((cnt @ Xk2) / n - mb ** 2) * n / (n - 1), 1e-18))
        best_boot[b] = ((mb - mean) / (sdb / np.sqrt(n))).max()
    k = int(np.argmax(t_obs))
    return {"n_races": int(n), "n_strategies": int(keep.sum()), "best_idx_in_kept": k, "t_max": float(t_obs[k]),
            "p": float((best_boot >= t_obs[k]).mean()), "kept": np.where(keep)[0].tolist(), "mean": mean.tolist(), "t": t_obs.tolist()}


def ruin_prob(x: np.ndarray, capital: float = 50.0, horizon: int = 500, n_sim: int = 4000, seed: int = 0) -> float:
    """1 レース 1 単位を賭け続け、元手 capital 単位が尽きる確率。x は 1 レースの戻り倍率（賭け金 1 あたり）。賭けないレースは除く。"""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    pnl = rng.choice(x - 1.0, size=(n_sim, horizon))
    cum = capital + np.cumsum(pnl, axis=1)
    return float((cum.min(axis=1) <= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "analyze", "final2026"])
    ap.add_argument("--rules", default=str(OUT / "analysis.json"))
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        d = load_frame((2022, 2023, 2024, 2025))
        pay, bad = load_payoffs(set(d["race_id"].unique()))
        rf, R = build(d, pay, bad)
        rf.to_parquet(OUT / "race_features.parquet", index=False)
        np.save(OUT / "returns.npy", R)
        print("races", len(rf), "combos", R.shape[1], "saved", OUT)
    elif a.cmd == "analyze":
        from src.scripts.data import tp43_analysis as an

        an.run(OUT, a.n_boot)
    else:
        from src.scripts.data import tp43_analysis as an

        an.final2026(OUT, Path(a.rules))


if __name__ == "__main__":
    main()
