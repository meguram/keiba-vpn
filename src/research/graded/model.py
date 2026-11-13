"""重賞のルールベース予想家: 縮小推定・型の採否・束ね・点数・印（計画書の第 3.4・6・7 節）。

縮小（階層、期待との比 r = 実際の 3 着内 ÷ 期待の 3 着内）:
  全体   r_g = (O_g + m_g) / (E_g + m_g)                    … 1（頭数どおり）へ縮める
  副     r_s = (O_s + m_s·r_g) / (E_s + m_s)                 … 副の件数が 50 走・20 レース未満なら r_s = r_g
  主     r_m = (λ·O_m + m_m·r_s) / (λ·E_m + m_m)            … 主の件数が 15 走・4 回未満なら r_m = r_s
  主の母集団の 70% 超が当てはまる水準（情報が無い）は点に使わない（x = 0）。
  1 頭の 1 つの型の点 x = log(r_m)。水準が「不明」「前走が海外・地方」なら x = 0。
型の効き目: レースごとに x の偏差と 3 着内の偏差の積の平均を取り、レース間のばらつきで z にする（レースを 1 つの塊とする）。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.research.graded.catalog import PREV_FOREIGN, UNKNOWN

MIN_MAIN_N, MIN_MAIN_ED = 15, 4
MIN_SUB_N, MIN_SUB_RACES = 50, 20
SUB_FALLBACK_N = 300          # 副（OP 以上）の全体がこれ未満なら 3 勝クラス以上まで広げる
MAX_SHARE = 0.7
GOOD_EDHIT = 3                # 好成績パターンとして表示する条件（主で 3 着内が出た回の数）
NO_INFO = {UNKNOWN, PREV_FOREIGN}


def attach_context(T: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """長い表 T に、行の情報（race_id・year・頭数・3 着内）と、主・副の母集団全体の件数（型 ALL）を付ける。"""
    T = T.merge(rows[["row", "race_id", "year", "field", "y3", "y1", "hn"]], on="row", how="left")
    allr = T[T["type"] == "ALL"].set_index("row")
    T = T[T["type"] != "ALL"]
    T = T.join(allr[["main_n", "main_ed", "sub_n", "sub4_n"]].rename(columns=lambda c: "all_" + c), on="row")
    return T


def shrink(T: pd.DataFrame, m_g: float, m_s: float, m_m: float, lam: float = 1.0) -> pd.DataFrame:
    """各行（馬 × 型）の縮小した比と点 x。使った層の印も付ける。"""
    O = lambda p: T[f"{p}_w1"] + T[f"{p}_w2"] + T[f"{p}_w3"]   # noqa: E731
    r_g = (O("glob") + m_g) / (T["glob_e3"] + m_g)
    use4 = T["all_sub_n"] < SUB_FALLBACK_N
    sO = np.where(use4, O("sub4"), O("sub"))
    sE = np.where(use4, T["sub4_e3"], T["sub_e3"])
    sN = np.where(use4, T["sub4_n"], T["sub_n"])
    sR = np.where(use4, T["sub4_ed"], T["sub_ed"])
    use_s = (sN >= MIN_SUB_N) & (sR >= MIN_SUB_RACES)
    r_s = np.where(use_s, (sO + m_s * r_g) / (sE + m_s), r_g)
    use_m = (T["main_n"] >= MIN_MAIN_N) & (T["main_ed"] >= MIN_MAIN_ED)
    r_m = np.where(use_m, (lam * O("main") + m_m * r_s) / (lam * T["main_e3"] + m_m), r_s)
    share = T["main_n"] / T["all_main_n"].replace(0, np.nan)
    x = np.log(np.clip(r_m, 1e-3, None))
    x = np.where(T["level"].isin(NO_INFO) | (use_m & (share > MAX_SHARE)), 0.0, x)
    out = T.copy()
    out["r_g"], out["r_s"], out["r_m"], out["x"] = r_g, r_s, r_m, x
    out["layer"] = np.where(use_m, "主", np.where(use_s, "副", "全体"))
    out["sub_kind"] = np.where(use4, "3勝以上", "OP以上")
    return out


def type_z(S: pd.DataFrame) -> pd.DataFrame:
    """型ごとの効き目の z（レース単位）。S は shrink の出力を、評価する年の行に絞ったもの。"""
    g = S.groupby(["type", "race_id"], sort=False)
    xm = g["x"].transform("mean")
    ym = g["y3"].transform("mean")
    S = S.assign(c=(S["x"] - xm) * (S["y3"] - ym))
    rc = S.groupby(["type", "race_id"], sort=False).agg(c=("c", "mean"), vx=("x", "var")).reset_index()
    rc = rc[rc["vx"] > 1e-12]
    agg = rc.groupby("type")["c"].agg(["mean", "std", "size"])
    agg["z"] = agg["mean"] / (agg["std"] / np.sqrt(agg["size"]))
    return agg.rename(columns={"size": "races"})


def logloss_m(T: pd.DataFrame, grid=(3, 10, 30, 100)) -> pd.DataFrame:
    """縮める強さ (m_g, m_s, m_m) ごとに、p = 3/頭数 × r_m の 3 着内の対数損失（型の平均）。"""
    known = ~T["level"].isin(NO_INFO)
    Tk = T[known]
    rows = []
    for mg, ms, mm in itertools.product(grid, grid, grid):
        S = shrink(Tk, mg, ms, mm)
        p = np.clip(3.0 / S["field"] * S["r_m"], 1e-4, 0.95)
        ll = -(S["y3"] * np.log(p) + (1 - S["y3"]) * np.log(1 - p))
        rows.append({"m_g": mg, "m_s": ms, "m_m": mm, "logloss": float(ll.groupby(S["type"]).mean().mean())})
    return pd.DataFrame(rows).sort_values("logloss")


def cluster_types(S: pd.DataFrame, order: list[str], thr: float = 0.7) -> dict[str, str]:
    """採った型を、点 x の相関が |r| ≥ thr のものどうしで束ねる。order（z の大きい順）で先の型を束の代表にする。返り値: 型 → 代表。"""
    W = S.pivot_table(index="row", columns="type", values="x", aggfunc="first").reindex(columns=order).fillna(0.0)
    C = W.corr().fillna(0.0)
    rep: dict[str, str] = {}
    kept: list[str] = []
    for t in order:
        hit = [k for k in kept if abs(C.loc[t, k]) >= thr]
        if hit:
            rep[t] = hit[0]
        else:
            kept.append(t)
            rep[t] = t
    return rep


def score(S: pd.DataFrame, types: list[str], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """馬ごとの点（採った型の x の和。weights があれば重み付き）と、主の好成績パターンの数（同点の並べ替えに使う）。"""
    s = S[S["type"].isin(types)].copy()
    w = s["type"].map(weights).fillna(1.0) if weights else 1.0
    s["xw"] = s["x"] * w
    good = (s["layer"] == "主") & (s["x"] > 0) & (s["main_edhit"] >= GOOD_EDHIT)
    s["good_main"] = good.astype(int)
    out = s.groupby("row").agg(score=("xw", "sum"), n_good_main=("good_main", "sum"), race_id=("race_id", "first"), hn=("hn", "first"),
                               y3=("y3", "first"), y1=("y1", "first"), year=("year", "first"))
    return out.reset_index()


def race_auc(sc: pd.DataFrame, col: str = "score") -> float:
    """レースごとに、3 着内の馬と 4 着以下の馬の組を正しく並べた割合（同点は 0.5）の平均。"""
    vals = []
    for _, g in sc.groupby("race_id"):
        pos, neg = g.loc[g["y3"] == 1, col].to_numpy(), g.loc[g["y3"] == 0, col].to_numpy()
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            vals.append(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
    return float(np.mean(vals)) if vals else float("nan")


MARKS = ["◎", "○", "▲", "△", "△"]


def marks(sc: pd.DataFrame) -> pd.DataFrame:
    """点の順に ◎○▲△△。同点は主の好成績パターンの数が多い順、さらに馬番の小さい順。"""
    sc = sc.sort_values(["race_id", "score", "n_good_main", "hn"], ascending=[True, False, False, True]).copy()
    sc["rank"] = sc.groupby("race_id").cumcount() + 1
    sc["mark"] = sc["rank"].map(lambda k: MARKS[k - 1] if k <= len(MARKS) else "")
    return sc
