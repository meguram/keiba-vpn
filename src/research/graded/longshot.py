"""重賞のルールベース予想家: うまうまー式の穴馬の分析（docs/ideas/umauma_method_notes.md の第 6 節の設計）。

- 分母: 重賞ごとに、主の層（同じレース名 × 2016〜2024 年で一番多く行われたコース）の 6 番人気以下の全出走（凡走も含む）。人気は分母を選ぶためだけに使い、
  条件の切り口には使わない（人気・オッズ・前走の人気は目録の外）。
- 条件: 切り口（catalog の型）1〜2 個の水準の AND。重賞ごとに探す（主）ほか、全重賞をまとめた分母でも探す（雛形）。
- 指標: 再現率（好走した穴馬のうち条件を満たす割合。うまうまー式）、該当率（分母のうち条件を満たす割合）、該当馬の [1-2-3-着外]・複勝率、
  リフト（該当馬の複勝率 ÷ 分母の複勝率）、単勝・複勝の回収率（物差しだけ。採否に使わない）。
- 採る条件（凍結の前に固定）: 好走 ≥ 3 頭、好走が 2 回以上のレースにまたがる、再現率 ≥ 60%、リフト ≥ 1.5。
- 偶然の扱い: レースの中で 6 番人気以下の馬の結果（着順・払戻）を入れ替えて同じ探索を流した数（ラベルの入れ替えによる帰無）。
  各条件の二項の上側確率（期待 = 分母の複勝率）に、探した条件の数で BH 法（FDR 10%）をかけた印。
計算は手元だけ（費用 0 円）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

POP_MIN = 6
K_MIN, ED_MIN, RECALL_MIN, LIFT_MIN = 3, 2, 0.60, 1.5
FDR = 0.10
N_PERM = 5
POOLED_K_MIN, POOLED_TOP = 20, 60   # 全重賞の条件: 好走 ≥ 20 頭で BH に残るもののうち、二項の確率の小さい順に 60 組（同じ馬の集まりはまとめる）
EVAL_END = "2026-11-14"           # 評価に入れる重賞の最終日（凍結の前日。流した日で分母が変わらないように）
CODE_BITS = 10
COLS = ["n", "k", "w1", "w2", "w3", "np", "rt", "rf"]


def cells(X: np.ndarray, Y: np.ndarray, race: np.ndarray, max_depth: int = 2) -> pd.DataFrame:
    """母集団（X: 行 × 型の水準の番号、Y: COLS の列）の、切り口 1〜max_depth 個の全部の組の件数。
    返す列: ts（型の番号の組）、cs（水準の番号の組）、COLS、edk（好走のいたレースの数）。"""
    n, T = X.shape
    out = []
    racei = pd.factorize(race)[0].astype(np.int64)
    top = Y[:, 1] > 0
    def agg(keys, rows, decode):
        u, inv = np.unique(keys, return_inverse=True)
        S = np.stack([np.bincount(inv, weights=Y[rows, j], minlength=len(u)) for j in range(Y.shape[1])], axis=1)
        # 好走のいたレースの数
        kk = top[rows]
        pair = np.unique(np.stack([inv[kk], racei[rows][kk]], 1), axis=0) if kk.any() else np.zeros((0, 2), np.int64)
        edk = np.bincount(pair[:, 0], minlength=len(u)) if len(pair) else np.zeros(len(u), np.int64)
        ts, cs = decode(u)
        df = pd.DataFrame(S, columns=COLS)
        df["edk"] = edk
        df["ts"], df["cs"] = ts, cs
        out.append(df)
    for t in range(T):
        v = np.flatnonzero(X[:, t] >= 0)
        if len(v):
            agg(X[v, t].astype(np.int64), v, lambda u, t=t: ([(t,)] * len(u), [(int(c),) for c in u]))
    if max_depth >= 2:
        m = (1 << CODE_BITS) - 1
        for t1 in range(T - 1):
            t2s = np.arange(t1 + 1, T)
            K = (t2s[None, :].astype(np.int64) << (2 * CODE_BITS)) | (X[:, [t1]].astype(np.int64) << CODE_BITS) | X[:, t2s].astype(np.int64)
            V = (X[:, [t1]] >= 0) & (X[:, t2s] >= 0)
            rows = np.broadcast_to(np.arange(n)[:, None], K.shape)[V]
            if rows.size == 0:
                continue
            agg(K[V], rows, lambda u, t1=t1: ([(t1, int(k >> (2 * CODE_BITS))) for k in u], [(int((k >> CODE_BITS) & m), int(k & m)) for k in u]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLS + ["edk", "ts", "cs"])


def score(C: pd.DataFrame, N: int, K: int) -> pd.DataFrame:
    """再現率・該当率・複勝率・リフト・二項の確率と、採る条件の判定（BH は呼び出し側でまとめてかける）。"""
    C = C.copy()
    base = K / N if N else np.nan
    C["recall"] = C["k"] / K if K else np.nan
    C["coverage"] = C["n"] / N
    C["rate"] = C["k"] / C["n"]
    C["lift"] = C["rate"] / base
    C["pbin"] = binom.sf(C["k"] - 1, C["n"], base) if K else 1.0
    C["pass"] = (C["k"] >= K_MIN) & (C["edk"] >= ED_MIN) & (C["recall"] >= RECALL_MIN) & (C["lift"] >= LIFT_MIN)
    C["N"], C["K"] = N, K
    return C


def bh(p: np.ndarray, m: int, q: float = FDR) -> np.ndarray:
    """BH 法（m = 検定の総数。渡した p は m 個のうちの一部でもよい）で残る印。"""
    if len(p) == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    thr = q * np.arange(1, len(p) + 1) / m
    ok = p[o] <= thr
    kmax = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    res = np.zeros(len(p), bool)
    res[o[:kmax]] = True
    return res


def permute(y: pd.DataFrame, race: np.ndarray, seed: int, cols=("w1", "w2", "w3", "top3", "np", "rt", "rf")) -> pd.DataFrame:
    """レースの中で、結果の束を馬と一緒に入れ替える（分母の 6 番人気以下の馬の中だけ）。"""
    rng = np.random.default_rng(seed)
    key = rng.random(len(y))
    perm = pd.DataFrame({"r": race, "k": key, "i": np.arange(len(y))}).sort_values(["r", "k"])["i"].to_numpy()
    base = pd.DataFrame({"r": race, "i": np.arange(len(y))}).sort_values(["r", "i"])["i"].to_numpy()
    src = np.empty(len(y), np.int64)
    src[base] = perm
    out = y.copy()
    for c in cols:
        out[c] = y[c].to_numpy()[src]
    return out


def dedupe(C: pd.DataFrame, X: np.ndarray, rows_id: np.ndarray) -> pd.DataFrame:
    """同じ馬の集まりになる条件は 1 つにまとめる（型の数が少ない → 型 → 水準の順で最初のもの）。"""
    keep, seen = [], set()
    C = C.assign(depth=C["ts"].map(len)).sort_values(["depth", "ts", "cs"])
    for i, r in zip(C.index, C.itertuples()):
        m = np.ones(len(X), bool)
        for t, c in zip(r.ts, r.cs):
            m &= X[:, t] == c
        key = tuple(rows_id[m])
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return C.loc[keep]


def matches(X: np.ndarray, ts, cs) -> np.ndarray:
    m = np.ones(len(X), bool)
    for t, c in zip(ts, cs):
        m &= X[:, t] == c
    return m


def rec(y: pd.DataFrame) -> dict:
    """着順・払戻の列（fin・pay_ok・ret_tan・ret_fuku）から [1-2-3-着外]・複勝率・単複の回収率。"""
    n = len(y)
    w = [int((y["fin"] == k).sum()) for k in (1, 2, 3)]
    p = y[y["pay_ok"].fillna(False).astype(bool)]
    return {"n": n, "rec": w + [n - sum(w)], "rate": (sum(w) / n) if n else None,
            "roi_t": float(p["ret_tan"].sum() / len(p)) if len(p) else None, "roi_f": float(p["ret_fuku"].sum() / len(p)) if len(p) else None}


# ------------------------------------------------------------------ 記事の例の再現（docs/ideas/umauma_method_notes.md 第 5 節）
def _g(d, c):
    return pd.to_numeric(d[c], errors="coerce")


NOTE_EXAMPLES = [
    # (重賞, 記事の分母の人気の線, 条件の名前, 記事の数字, 条件)
    ("宝塚記念", 4, "前走の上がり 3F の順位が 6 位以内（前走の上がり順位が無い馬 = 海外など は分母から除く）", "18/20（海外の前走 2 頭を除く）", lambda d: _g(d, "p_l3rank") <= 6),
    ("宝塚記念", 4, "前走 G1 は着順不問／G2 は 3 着内／G3 は 1 着か勝ち馬と 0.1 秒差", "（割合の記載なし。項目の 1 つ）",
     lambda d: (d["p_grade"] == "G1") | ((d["p_grade"] == "G2") & (_g(d, "p_fin") <= 3)) | ((d["p_grade"] == "G3") & ((_g(d, "p_fin") == 1) | (_g(d, "p_tdiff") <= 0.1)))),
    ("新潟大賞典", 6, "馬体重 470kg 以上（記事は前走の体重。手元は今回の体重で代用）", "18/19", lambda d: _g(d, "wt") >= 470),
    ("新潟大賞典", 6, "前走 4 角 1〜4 番手なら上がり 4 位内（後ろの馬は問わない）", "（割合の記載なし。項目の 1 つ）",
     lambda d: _g(d, "p_pos4").notna() & _g(d, "p_l3rank").notna() & ~((_g(d, "p_pos4") <= 4) & (_g(d, "p_l3rank") > 4))),
    ("根岸S", 6, "6 歳以上", "10/13", lambda d: _g(d, "age") >= 6),
    ("根岸S", 6, "前走 4 角 6 番手以下", "9/13", lambda d: _g(d, "p_pos4") >= 6),
    ("根岸S", 99, "中 4 週未満（間隔 35 日未満。人気の絞りなしの全出走。間隔の無い馬は分母から除く）", "(0-0-3-54)。中 4 週以上は (13-13-10-112)", lambda d: _g(d, "interval") < 35),
    ("アルゼンチン共和国杯", 6, "前走 東京なら 4 着内、東京以外なら 9 着内", "（11 頭中 10 頭が全項目クリア）",
     lambda d: ((d["p_vc"] == "05") & (_g(d, "p_fin") <= 4)) | ((d["p_vc"] != "05") & (_g(d, "p_fin") <= 9))),
]


EXCLUDE = {   # 記事に合わせて分母から除く馬（条件の名前 → 残す行）
    "前走の上がり 3F の順位が 6 位以内（前走の上がり順位が無い馬 = 海外など は分母から除く）": lambda d: _g(d, "p_l3rank").notna(),
    "中 4 週未満（間隔 35 日未満。人気の絞りなしの全出走。間隔の無い馬は分母から除く）": lambda d: _g(d, "interval").notna(),
}


def example_stats(R: pd.DataFrame, years: tuple[int, int]) -> list[dict]:
    """記事の例を、手元のデータ（主の層、years の回）で当てはめた値。R は着順・人気・払戻と元の列を持つ行。"""
    out = []
    for rk, popmin, name, claim, f in NOTE_EXAMPLES:
        x = R[(R["rkey"] == rk) & R["main"] & (R["year"] >= years[0]) & (R["year"] <= years[1])]
        den = x[x["pop"] >= popmin] if popmin < 99 else x
        den = den[EXCLUDE[name](den)] if name in EXCLUDE else den
        m = f(den).fillna(False).to_numpy(bool)
        K = int(den["top3"].sum())
        k = int(den["top3"].to_numpy()[m].sum())
        base = K / len(den) if len(den) else None
        hit = rec(den[m])
        out.append({"rkey": rk, "pop_min": popmin if popmin < 99 else None, "condition": name, "note_claim": claim, "years": list(years),
                    "races": int(den["race_id"].nunique()), "N": int(len(den)), "K": K, "k": k, "recall": k / K if K else None,
                    "coverage": float(m.mean()) if len(den) else None, "lift": (hit["rate"] / base) if base and hit["rate"] is not None else None,
                    "hit": hit, "rest": rec(den[~m])})
    return out
