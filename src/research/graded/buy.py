"""重賞のルールベース予想家: 「買い」の組み合わせ探し（3 着内が 7 割以上で、単勝か複勝の回収率が 100% を超える切り口の AND）。

- 母集団: 主 = 同じレース名 × 最新の回と同じコースの過去の回（OP 以上、2016〜2024 年）、副 = 同じコースの OP 以上の別レース。
- 条件: 切り口（型）1〜3 個の水準の AND（主は 3 個まで、副は 2 個まで）。オッズ・人気は使わない（catalog の型だけ）。
- 「買い」の組: 走数 ≥ 2、3 着内率 ≥ 0.70、単勝か複勝の回収率（払戻の和 ÷ 払戻の解けた走数）> 1。
  版で、走数・異なる回・異なる馬・払戻の解けた走数の下限、型の数の上限、券種を絞る。
- 同じ馬の集まり（過去の走の集合）になる条件は 1 つにまとめ、型の数が少ない → 型の ID の順 → 水準の順 で最初のものを残す。
- 刈り込み: 3 着内が 2 走未満の 2 個の組は 3 個に伸ばさない（伸ばしても 3 着内は増えないので、3 着内率 7 割・走数 2 以上に届かない）。
- 偶然の見込み: レース（race_id）の中で、結果の束（着順・払戻）を馬と一緒に入れ替えたデータで同じ探索を流し、見つかる組の数を数える。
計算は手元だけ（費用 0 円）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.research.graded import model as M

RATE_MIN = 0.70
N_FLOOR = 2                         # 探索で拾う走数の下限（極小の版）。版で 3・5・8 に上げる
CODE_BITS = 10                      # 水準の番号のビット数（1 型あたり 1024 水準まで）
OUT = ["n", "k", "w1", "w2", "w3", "np", "rt", "rf", "e3"]   # 組ごとに足す列（k = 3 着内）
PERM_COLS = ["w1", "w2", "w3", "top3", "np", "rt", "rf"]      # 並べ替えで馬と一緒に動かす結果の束（e3 は頭数なので馬に残す）


def encode(L: pd.DataFrame, types: list[str]) -> tuple[np.ndarray, dict[str, list[str]]]:
    """水準を型ごとの番号（0 から）にする。NO_INFO と欠けは -1（その型では組に入れない）。番号 → 水準の辞書も返す。"""
    X = np.full((len(L), len(types)), -1, dtype=np.int32)
    names = {}
    for j, t in enumerate(types):
        s = L[t].astype("string")
        s = s.where(~s.isin(list(M.NO_INFO)))
        cat = pd.Categorical(s)
        if len(cat.categories) >= 1 << CODE_BITS:
            raise ValueError(f"{t}: 水準が多すぎる（{len(cat.categories)}）")
        X[:, j] = cat.codes
        names[t] = [str(c) for c in cat.categories]
    return X, names


def _agg(keys: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """keys ごとの Y の列の和。(一意な key, 和の行列, 行 → 一意の番号)。"""
    u, inv = np.unique(keys, return_inverse=True)
    S = np.stack([np.bincount(inv, weights=Y[:, j], minlength=len(u)) for j in range(Y.shape[1])], axis=1)
    return u, S, inv


def _qual(S: np.ndarray) -> np.ndarray:
    n, k, np_, rt, rf = S[:, 0], S[:, 1], S[:, 5], S[:, 6], S[:, 7]
    with np.errstate(divide="ignore", invalid="ignore"):
        roi_ok = (np_ > 0) & ((rt / np_ > 1) | (rf / np_ > 1))
    return (n >= N_FLOOR) & (k >= RATE_MIN * n - 1e-9) & roi_ok


def search(X: np.ndarray, Y: np.ndarray, max_depth: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """母集団（X: 行 × 型の水準の番号、Y: OUT の列）で条件を満たす組 [(型の番号の組, 水準の番号の組)]。"""
    n, T = X.shape
    found = []
    m = (1 << CODE_BITS) - 1
    # 1 個
    for t in range(T):
        v = X[:, t] >= 0
        if not v.any():
            continue
        u, S, _ = _agg(X[v, t].astype(np.int64), Y[v])
        for c in u[_qual(S)]:
            found.append(((t,), (int(c),)))
    if max_depth < 2:
        return found
    # 2 個（t1 を固定し、t2 > t1 をまとめて数える）
    for t1 in range(T - 1):
        v1 = X[:, t1] >= 0
        if not v1.any():
            continue
        t2s = np.arange(t1 + 1, T)
        X1 = X[v1]
        K = (t2s[None, :].astype(np.int64) << (2 * CODE_BITS)) | (X1[:, [t1]].astype(np.int64) << CODE_BITS) | X1[:, t2s].astype(np.int64)
        V = X1[:, t2s] >= 0
        rows = np.broadcast_to(np.arange(len(X1))[:, None], K.shape)[V]
        keys = K[V]
        if keys.size == 0:
            continue
        u, S, inv = _agg(keys, Y[v1][rows])
        for key in u[_qual(S)]:
            found.append(((t1, int(key >> (2 * CODE_BITS))), (int((key >> CODE_BITS) & m), int(key & m))))
        if max_depth < 3:
            continue
        # 3 個: 3 着内が 2 走以上ある 2 個の組だけを伸ばす
        grow = S[:, 1] >= 2
        if not grow.any():
            continue
        X1i = np.flatnonzero(v1)
        sel = grow[inv]
        r_sel, key_sel = rows[sel], keys[sel]
        t2_of = (key_sel >> (2 * CODE_BITS)).astype(np.int64)
        for t2 in np.unique(t2_of):
            if t2 >= T - 1:
                continue
            pick = t2_of == t2
            rr = X1i[r_sel[pick]]                       # 母集団の行番号
            c12 = (key_sel[pick] & ((1 << (2 * CODE_BITS)) - 1)).astype(np.int64)
            t3s = np.arange(t2 + 1, T)
            X3 = X[rr][:, t3s]
            V3 = X3 >= 0
            K3 = (t3s[None, :].astype(np.int64) << (3 * CODE_BITS)) | (c12[:, None] << CODE_BITS) | X3.astype(np.int64)
            r3 = np.broadcast_to(np.arange(len(rr))[:, None], K3.shape)[V3]
            keys3 = K3[V3]
            if keys3.size == 0:
                continue
            u3, S3, _ = _agg(keys3, Y[rr][r3])
            for key in u3[_qual(S3)]:
                c12k = int((key >> CODE_BITS) & ((1 << (2 * CODE_BITS)) - 1))
                found.append(((t1, int(t2), int(key >> (3 * CODE_BITS))), (c12k >> CODE_BITS, c12k & m, int(key & m))))
    return found


def describe(X: np.ndarray, Y: np.ndarray, race: np.ndarray, horse: np.ndarray, rows_id: np.ndarray, found) -> list[dict]:
    """見つかった組の件数・回の数・馬の数・二項の確率。同じ過去の走の集合になる組は 1 つにまとめる（型の数が少ない → 型 → 水準の順で残す）。"""
    best: dict[tuple, dict] = {}
    for ts, cs in sorted(found, key=lambda f: (len(f[0]), f[0], f[1])):
        m = np.ones(len(X), bool)
        for t, c in zip(ts, cs):
            m &= X[:, t] == c
        idx = np.flatnonzero(m)
        key = tuple(rows_id[idx])
        if key in best:
            best[key]["n_equiv"] += 1
            continue
        s = Y[idx].sum(0)
        n, k = int(s[0]), int(s[1])
        p0 = float(s[8] / n)
        best[key] = {"ts": ts, "cs": cs, "n": n, "k": k, "rec": [int(s[2]), int(s[3]), int(s[4]), n - int(s[2] + s[3] + s[4])],
                     "np": int(s[5]), "roi_t": float(s[6] / s[5]) if s[5] else None, "roi_f": float(s[7] / s[5]) if s[5] else None,
                     "ed": int(len(set(race[idx]))), "nh": int(len(set(horse[idx]))), "p0": p0,
                     "pbin": float(binom.sf(k - 1, n, min(p0, 1.0))), "n_equiv": 1}
    return list(best.values())


def permute_within_race(b: pd.DataFrame, seed: int) -> pd.DataFrame:
    """レースの中で、結果の束（PERM_COLS）を馬と一緒に入れ替える（水準・頭数・期待は馬に残す）。"""
    rng = np.random.default_rng(seed)
    key = rng.random(len(b))
    perm = pd.DataFrame({"race_id": b["race_id"].to_numpy(), "k": key, "i": np.arange(len(b))}).sort_values(["race_id", "k"])["i"].to_numpy()
    base = pd.DataFrame({"race_id": b["race_id"].to_numpy(), "i": np.arange(len(b))}).sort_values(["race_id", "i"])["i"].to_numpy()
    out = b.copy()
    src = np.empty(len(b), dtype=np.int64)
    src[base] = perm                   # 同じレースの中の、並べ替えた先の行
    for c in PERM_COLS:
        out[c] = b[c].to_numpy()[src]
    return out


# ------------------------------------------------------------------ 期間ごとの探索と当てはめ
LAYER_DEPTH = {"主": 3, "副": 2}
NMINS = (2, 3, 5, 8)                 # 2 は「極小」
BETS = ("どちらか", "単勝", "複勝")
HITS = (1, 3, 10)                    # 馬が当たる組の数の下限
LAYER_SETS = {"主・3つまで": ("主", 3), "主・2つまで": ("主", 2), "主・1つ": ("主", 1), "副・2つまで": ("副", 2), "副・1つ": ("副", 1),
              "主＋副": ("主＋副", 3)}
ED_MIN = NH_MIN = 2
SELECT_MIN_BUY = 50                  # 主に見る版の選び方: 確認期間で買いの馬が 50 頭以上の版のうち、3 着内率が最大の版
N_PERM = 5


def all_versions() -> list[dict]:
    import itertools
    return [{"layers": ls, "nmin": n, "bet": bt, "hits": h, "family": "極小（2 走）" if n == 2 else "通常"}
            for ls, n, bt, h in itertools.product(LAYER_SETS, NMINS, BETS, HITS)]


def version_name(v: dict) -> str:
    return f"買い版[{v['layers']}・{v['nmin']}走以上{'（極小）' if v['nmin'] == 2 else ''}・{v['bet']}・{v['hits']}つ以上]"


def cell_ok(c, v: dict) -> bool:
    lay, depth = LAYER_SETS[v["layers"]]
    if lay != "主＋副" and c["layer"] != lay:
        return False
    if c["depth"] > depth or c["n"] < v["nmin"] or c["np"] < v["nmin"] or c["ed"] < ED_MIN or c["nh"] < NH_MIN:
        return False
    rt, rf = (c["roi_t"] or 0) > 1, (c["roi_f"] or 0) > 1
    return {"単勝": rt, "複勝": rf, "どちらか": rt or rf}[v["bet"]]


def cells_ok(cells: pd.DataFrame, v: dict) -> np.ndarray:
    """cell_ok を組の表全体にまとめて当てる。"""
    lay, depth = LAYER_SETS[v["layers"]]
    ok = (cells["depth"].to_numpy() <= depth) & (cells["n"].to_numpy() >= v["nmin"]) & (cells["np"].to_numpy() >= v["nmin"])
    ok &= (cells["ed"].to_numpy() >= ED_MIN) & (cells["nh"].to_numpy() >= NH_MIN)
    if lay != "主＋副":
        ok &= cells["layer"].to_numpy() == lay
    rt = cells["roi_t"].fillna(0).to_numpy() > 1
    rf = cells["roi_f"].fillna(0).to_numpy() > 1
    return ok & {"単勝": rt, "複勝": rf, "どちらか": rt | rf}[v["bet"]]


_G: dict = {}


def _work(job):
    rk, course, layer = job
    b, X = _G["b"], _G["X"]
    m = (b["course"].to_numpy() == course) & b["op"].to_numpy() & ((b["rkey"].to_numpy() == rk) if layer == "主" else (b["rkey"].to_numpy() != rk))
    idx = np.flatnonzero(m)
    if len(idx) < N_FLOOR:
        return []
    Y = np.column_stack([np.ones(len(idx)), b["top3"].to_numpy()[idx]] + [b[k].to_numpy()[idx] for k in ["w1", "w2", "w3", "np", "rt", "rf", "e3"]])
    f = search(X[idx], Y, LAYER_DEPTH[layer])
    cells = describe(X[idx], Y, b["race_id"].to_numpy()[idx], b["horse_id"].to_numpy()[idx], b["row"].to_numpy()[idx], f)
    for c in cells:
        c.update(rkey=rk, course=course, layer=layer, depth=len(c["ts"]))
    return cells


def search_period(b: pd.DataFrame, X: np.ndarray, courses: dict[str, str], workers: int = 6) -> pd.DataFrame:
    """重賞（rkey → コース）ごとに主・副を探し、組の表を返す。b と X は同じ行の並び。"""
    from multiprocessing import get_context
    _G["b"], _G["X"] = b, X
    jobs = [(rk, c, ly) for rk, c in courses.items() for ly in LAYER_DEPTH]
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(_work, jobs, chunksize=2)
    rows = [c for r in res for c in r]
    out = pd.DataFrame(rows)
    if len(out):
        out["ts"] = out["ts"].map(tuple)
        out["cs"] = out["cs"].map(tuple)
    return out


def hits_for(cells: pd.DataFrame, rows: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """対象のレースの馬（rows: row・rkey・course、X は同じ並びの水準の番号）が当たる組の一覧 (i: rows の位置, cell: cells の位置)。
    組の重賞・コースが馬のレースと同じときだけ当てる。"""
    out = []
    by = rows.groupby(["rkey", "course"]).indices
    for ci, c in enumerate(cells.itertuples()):
        ii = by.get((c.rkey, c.course))
        if ii is None:
            continue
        m = np.ones(len(ii), bool)
        for t, v in zip(c.ts, c.cs):
            m &= X[ii, t] == v
        for i in ii[m]:
            out.append((i, ci))
    return pd.DataFrame(out, columns=["i", "cell"])


def buy_metrics(rows: pd.DataFrame, buy: np.ndarray, ref_score: np.ndarray) -> dict:
    """買いの馬（buy: rows の位置の真偽）の [1-2-3-着外]・3 着内率・単複の回収率（全頭 100 円）と、◎ を買いの馬に限ったとき
    （買いの馬のうち前日版の点が最大の馬）の成績。比べる相手は同じレースの前日版の ◎（ref_score の最大）と、そのレースの全頭。"""
    r = rows.assign(buy=buy, ref=ref_score, pay_ok=rows["pay_ok"].fillna(False).astype(bool))
    def rec(x):
        n = len(x)
        w = [int((x["fin"] == k).sum()) for k in (1, 2, 3)]
        p = x[x["pay_ok"]]
        return {"n": n, "rec": w + [n - sum(w)], "top3_rate": float(sum(w) / n) if n else None,
                "roi_t": float(p["ret_tan"].sum() / len(p)) if len(p) else None, "roi_f": float(p["ret_fuku"].sum() / len(p)) if len(p) else None}
    B = r[r["buy"]]
    races = B["race_id"].unique()
    R = r[r["race_id"].isin(races)]
    hon = R[R["buy"]].sort_values(["race_id", "ref", "hn"], ascending=[True, False, True]).groupby("race_id").head(1)
    base = R.sort_values(["race_id", "ref", "hn"], ascending=[True, False, True]).groupby("race_id").head(1)
    return {"runners": int(len(r)), "races_all": int(r["race_id"].nunique()), "buy_share": float(r["buy"].mean()) if len(r) else None,
            "races_with_buy": int(len(races)), "buy": rec(B), "honmei_buy": rec(hon), "honmei_preday_same_races": rec(base), "all_same_races": rec(R)}


RULES = {
    "cell": f"切り口（型）1〜3 個の水準の AND。主（同じレース名 × 同じコースの過去の回）は 3 個まで、副（同じコースの OP 以上の別レース）は 2 個まで。"
            f"走数 ≥ 2、3 着内率 ≥ {RATE_MIN:.0%}、単勝か複勝の回収率（払戻の和 ÷ 払戻の解けた走数）> 100%",
    "version": f"版で 走数と払戻の解けた走数 ≥ n の下限（2 は極小）、異なる回 ≥ {ED_MIN}、異なる馬 ≥ {NH_MIN}、型の数の上限、券種（単勝・複勝・どちらか）、"
               "馬が当たる組の数の下限（1・3・10）を絞る",
    "dedupe": "同じ過去の走の集合になる条件は 1 つにまとめ、型の数が少ない → 型の並び → 水準の番号の順で最初のものを残す（同じ集合の条件の数は n_equiv）",
    "prune": "3 着内が 2 走未満の 2 個の組は 3 個に伸ばさない（3 着内率 70%・走数 2 以上に届かないため）。このため 3 個の水準の組は全部は数えていない",
    "chance": "偶然の組: レースの中で結果の束（着順・払戻）を馬と一緒に入れ替えたデータで同じ探索を流した組。本物の組と同じ規則で当てはめて比べる。"
              "各組の pbin は、その組を単独で見たときの二項の上側確率（期待 = Σ3/頭数 ÷ 走数）で、探した数による水増しは入っていない",
    "buy": "馬単位: 版の条件を満たす組に、当たる組の数の下限以上当たった馬を買い（1 頭が複数の組に当たっても 1 頭）。組の重賞・コースが馬のレースと同じときだけ当てる",
    "honmei": "◎ を買いの馬に限る: 買いの馬のうち前日版の点が最大の馬を ◎。比べる相手は、同じレースの前日版の ◎ と、そのレースの全頭",
    "primary": f"2016〜2021 年で探して 2022〜2024 年の重賞に当てはめ、極小でない版のうち買いの馬が {SELECT_MIN_BUY} 頭以上の版で、"
               "3 着内率が最大の版（同じなら複勝の回収率が高い → n の下限が大きい）",
    "dnf": "着順の無い馬（取消・除外・中止）は、探索の母集団からも評価の対象からも外す",
    "roi_note": "3 着内率 70% 以上なら、複勝の払戻の平均が約 143 円を超えるだけで複勝の回収率は 100% を超えるので、複勝の条件は 3 着内率の条件からほぼ従う。単勝だけの版も並べる",
}


def constants() -> dict:
    return {"RATE_MIN": RATE_MIN, "N_FLOOR": N_FLOOR, "CODE_BITS": CODE_BITS, "LAYER_DEPTH": LAYER_DEPTH, "NMINS": list(NMINS), "BETS": list(BETS),
            "HITS": list(HITS), "LAYER_SETS": {k: list(v) for k, v in LAYER_SETS.items()}, "ED_MIN": ED_MIN, "NH_MIN": NH_MIN,
            "SELECT_MIN_BUY": SELECT_MIN_BUY, "N_PERM": N_PERM, "PERM_COLS": PERM_COLS}


def n_combos(T: int, depth: int) -> int:
    from math import comb
    return int(sum(comb(T, k) for k in range(1, depth + 1)))


def choose_primary(versions: list[dict], dev: dict) -> str:
    def key(v):
        e = dev[version_name(v)]["real"]["buy"]
        return (-(e["top3_rate"] or 0), -(e["roi_f"] or 0), -v["nmin"])
    cand = [v for v in versions if v["nmin"] >= 3 and dev[version_name(v)]["real"]["buy"]["n"] >= SELECT_MIN_BUY]
    if not cand:
        cand = [v for v in versions if v["nmin"] >= 3 and dev[version_name(v)]["real"]["buy"]["n"] >= 10] or [v for v in versions if v["nmin"] >= 3]
    return version_name(sorted(cand, key=key)[0])


def race_ci(rows: pd.DataFrame, buy: np.ndarray, ref: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """主に見る版の、買いの馬の 3 着内率・単複の回収率と、◎（買いの馬に限る）と前日版の ◎ の差の、レース単位のブートストラップ 95% 区間。"""
    r = rows.assign(buy=buy, ref=ref, y3=(rows["fin"] <= 3).astype(float), pay=rows["pay_ok"].fillna(False).astype(bool))
    r["rt"] = np.where(r["pay"], r["ret_tan"].fillna(0), 0.0)
    r["rf"] = np.where(r["pay"], r["ret_fuku"].fillna(0), 0.0)
    B = r[r["buy"]]
    if B.empty:
        return {"races": 0}
    pr = B.groupby("race_id").agg(n=("y3", "size"), y3=("y3", "sum"), np=("pay", "sum"), rt=("rt", "sum"), rf=("rf", "sum"))
    R = r[r["race_id"].isin(pr.index)].sort_values(["race_id", "ref", "hn"], ascending=[True, False, True])
    hb = R[R["buy"]].groupby("race_id").head(1).set_index("race_id")
    hp = R.groupby("race_id").head(1).set_index("race_id")
    A = np.column_stack([pr["n"], pr["y3"], pr["np"], pr["rt"], pr["rf"], hb.loc[pr.index, "y3"], hp.loc[pr.index, "y3"],
                         hb.loc[pr.index, "rf"], hp.loc[pr.index, "rf"], hb.loc[pr.index, "rt"], hp.loc[pr.index, "rt"]]).astype(float)
    W = np.random.default_rng(seed).poisson(1.0, size=(n_boot, len(A))).astype(float)
    S, nb = W @ A, W.sum(1)
    def ci(est, b):
        b = b[np.isfinite(b)]
        return {"est": float(est), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975))}
    a = A.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"races": int(len(A)),
                "top3_rate": ci(a[1] / a[0], S[:, 1] / S[:, 0]), "roi_t": ci(a[3] / a[2], S[:, 3] / S[:, 2]), "roi_f": ci(a[4] / a[2], S[:, 4] / S[:, 2]),
                "honmei_place_diff": ci((a[5] - a[6]) / len(A), (S[:, 5] - S[:, 6]) / nb),
                "honmei_fuku_diff": ci((a[7] - a[8]) / len(A), (S[:, 7] - S[:, 8]) / nb),
                "honmei_tan_diff": ci((a[9] - a[10]) / len(A), (S[:, 9] - S[:, 10]) / nb)}
