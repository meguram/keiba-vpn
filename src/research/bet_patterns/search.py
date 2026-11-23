"""◎ を固定した馬券の組み方を、訓練（2019〜22）だけで選ぶ。

段 A: 組み方（券種 × 形）ごとに、相手の選び方の格子（順位＋閾値: 予測の確率・1 位との比・偏差値・想定単勝オッズの範囲・組の期待値）を均等買いで全部試す。
段 B: 組み方ごとに段 A の最良に、資金配分（均等・払戻をそろえる・Kelly の 1/4 の上限付き）と見送り（点数の上限・合成オッズの下限・期待値の下限）を足す。
段 C: 段 B の上位の組み方の 2 つを同じレースで両方買う組み合わせ。
物差しは bet_cycles と同じ min(上位 5 件を除いた回収率, 年ごとの回収率の最小)。訓練の各年 100 レース以上の参戦が条件。
割引き: 段 A〜C の手順ごとの訓練の 1 年抜きの交差検証、段 A の全規則に対する White の reality check（ブートストラップ）。
費用 0 円（手元の配列だけ）。
"""

from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from math import comb

import numpy as np

from src.research.bet_patterns import data as D
from src.research.bet_patterns import odds_map as OM

NH, NF = D.NH, D.NF
MIN_BETS_PER_YEAR = 100
N_BOOT = 1000
UNIT = 100

# ---------------------------------------------------------------- 組み方
_N7 = tuple(range(1, 8))
_FORM = tuple((a, b) for a in (1, 2, 3) for b in (3, 4, 5, 6, 7, 8, 10) if b > a)
STRUCTS = [
    {"id": "tan", "name": "単勝 ◎", "kind": "tan", "shapes": ((0, 0),)},
    {"id": "fuku", "name": "複勝 ◎", "kind": "fuku", "shapes": ((0, 0),)},
    {"id": "waku", "name": "枠連 ◎ の枠から流し", "kind": "waku", "score": "p2", "shapes": tuple((n, n) for n in _N7)},
    {"id": "um_n", "name": "馬連 ◎ 流し", "kind": "um", "score": "p2", "shapes": tuple((n, n) for n in _N7)},
    {"id": "um_box", "name": "馬連 ボックス（◎＋相手）", "kind": "um_box", "score": "p2", "shapes": tuple((n, n) for n in (1, 2, 3, 4)), "ev": False},
    {"id": "wide_n", "name": "ワイド ◎ 流し", "kind": "wide", "score": "p3", "shapes": tuple((n, n) for n in _N7)},
    {"id": "wide_box", "name": "ワイド ボックス（◎＋相手）", "kind": "wide_box", "score": "p3", "shapes": tuple((n, n) for n in (1, 2, 3, 4)), "ev": False},
    {"id": "ut_1", "name": "馬単 ◎ 1 着流し", "kind": "uhj", "score": "p2", "shapes": tuple((n, n) for n in _N7)},
    {"id": "ut_2", "name": "馬単 ◎ 2 着流し", "kind": "ujh", "score": "p2", "shapes": tuple((n, n) for n in _N7)},
    {"id": "ut_m", "name": "馬単 ◎ 流しマルチ", "kind": "ut_multi", "score": "p2", "shapes": tuple((n, n) for n in _N7)},
    {"id": "trio_1", "name": "3連複 ◎ 1 頭軸流し", "kind": "trio_form", "score": "p3", "shapes": tuple((n, n) for n in range(2, 8))},
    {"id": "trio_2", "name": "3連複 ◎＋1 頭の 2 頭軸流し", "kind": "trio_form", "score": "p3", "shapes": tuple((1, b) for b in (3, 4, 5, 6, 7, 8, 10))},
    {"id": "trio_box", "name": "3連複 ボックス（◎＋相手）", "kind": "trio_box", "score": "p3", "shapes": tuple((n, n) for n in (2, 3, 4)), "ev": False},
    {"id": "trio_f", "name": "3連複 フォーメーション ◎-2 列目-3 列目", "kind": "trio_form", "score": "p3", "shapes": _FORM},
    {"id": "tri_1", "name": "3連単 ◎ 1 着固定流し", "kind": "t1_form", "score": "p3", "shapes": tuple((n, n) for n in range(2, 7))},
    {"id": "tri_2", "name": "3連単 ◎ 2 着付け", "kind": "t2", "score": "p3", "shapes": tuple((n, n) for n in range(2, 7))},
    {"id": "tri_3", "name": "3連単 ◎ 3 着付け", "kind": "t3", "score": "p3", "shapes": tuple((n, n) for n in range(2, 7))},
    {"id": "tri_m", "name": "3連単 ◎ 1 頭軸マルチ", "kind": "t_multi", "score": "p3", "shapes": tuple((n, n) for n in (2, 3, 4))},
    {"id": "tri_f", "name": "3連単 フォーメーション ◎→2 列目→3 列目", "kind": "t1_form", "score": "p3", "shapes": _FORM},
]
SBYID = {s["id"]: s for s in STRUCTS}
BOX = ("um_box", "wide_box", "trio_box")

# 相手の閾値の格子（同じ形を全部の組み方に）
P_MIN = {"p2": (0.0, 0.1, 0.2), "p3": (0.0, 0.2, 0.35)}
RATIO_MIN = (0.0, 0.4)           # 予測の 1 位（◎ を含むレースの全馬）との比
DEV_MIN = (None, 50.0, 55.0)     # レース内の偏差値（予測の確率）
ODDS_MIN = (1.0, 3.0)            # 相手の想定単勝オッズ（T-60 の再現）の範囲
ODDS_MAX = (None, 30.0)
EV_MIN = (None, 1.0, 1.3)        # 組の期待値（モデルの組の確率 × 想定オッズ）
# 段 B
STAKES = ("equal", "payout", "kelly")
CAPS = (None, 10, 20)            # 点数の上限（超えるレースは見送り）
SYNTH_MIN = (None, 1.5, 3.0)     # 合成オッズ 1/Σ(1/想定オッズ) の下限
REV_MIN = (None, 1.0, 1.2)       # レースの期待値（Σ口数×期待値 / Σ口数）の下限
KELLY_BANK, KELLY_FRAC, KELLY_CAP = 10000, 0.25, 5
MANDATED_PAIRS = (("tan", "wide_n"), ("um_n", "trio_1"))
TOP_PAIRS = 6


def filters(score: str):
    return list(itertools.product(P_MIN[score], RATIO_MIN, DEV_MIN, ODDS_MIN, ODDS_MAX))


def stage_a_rules(st: dict) -> list[dict]:
    if st["kind"] in ("tan", "fuku"):
        return [{"s": st["id"], "f": None, "sh": (0, 0), "ev": None}]
    evs = EV_MIN if st.get("ev", True) else (None,)
    return [{"s": st["id"], "f": f, "sh": sh, "ev": e} for f in filters(st["score"]) for sh in st["shapes"] for e in evs]


# ---------------------------------------------------------------- 配列の準備
class Prep:
    """1 期間 × 1 設定の行に、想定オッズ・期待値・レースの番号を足したもの。"""

    def __init__(self, rows: D.Rows, coefs: dict):
        a = rows.arr
        self.rows, self.a = rows, a
        self.N = len(rows.meta)
        self.k = int(rows.counts["k"])
        m = rows.meta
        self.race_ids, self.ridx = np.unique(m["race_id"].to_numpy(), return_inverse=True)
        self.nr = len(self.race_ids)
        yrs = m.groupby("race_id")["year"].first()
        self.ryear = yrs.reindex(self.race_ids).to_numpy()
        self.years = sorted(set(self.ryear.tolist()))
        H = m["H"].to_numpy()
        self.H = H
        self.E, self.P, self.Y = {}, {}, {}
        est = lambda bt, x: OM.est_odds(coefs, bt, x)
        self.E["tan"] = a["odds"][np.arange(self.N), H].astype(np.float32)[:, None]
        self.P["tan"] = a["o_win"][:, None]
        self.Y["tan"] = a["y_tan"][:, None]
        self.E["fuku"] = est("fukusho", a["m_fuku"])[:, None]
        self.P["fuku"] = a["o_fuku"][:, None]
        self.Y["fuku"] = a["y_fuku"][:, None]
        for key, bt in (("um", "umaren"), ("wide", "wide"), ("uhj", "umatan"), ("ujh", "umatan"), ("waku", "wakuren"),
                        ("trio", "sanrenpuku"), ("t1", "sanrentan"), ("t2", "sanrentan"), ("t3", "sanrentan")):
            self.E[key] = est(bt, a["m_" + key])
            self.P[key] = a["o_" + key]
            self.Y[key] = a["y_" + key]
        self.valid = a["valid"]
        n = m["n"].to_numpy()
        self.field = np.arange(NH)[None, :] < n[:, None]
        self.frame1h = np.zeros((self.N, NH, NF), bool)
        fr = a["frame"].astype(int)
        ok = fr > 0
        r_, c_ = np.where(ok)
        self.frame1h[r_, c_, fr[ok] - 1] = True
        self.isH = np.zeros((self.N, NH), bool)
        self.isH[np.arange(self.N), H] = True
        self.offdiag = ~np.eye(NH, dtype=bool)[None]
        self.upper = np.triu(np.ones((NH, NH), bool), 1)[None]
        self._R: dict = {}

    def rank(self, score: str, f) -> np.ndarray:
        """相手の候補の順位（0 始まり。候補でない馬は 99）。"""
        key = (score, f)
        if key in self._R:
            return self._R[key]
        a = self.a
        s = a[score].astype(np.float64)
        pmin, rmin, dmin, omin, omax = f
        fs = np.where(self.field, s, np.nan)
        mx = np.nanmax(fs, axis=1, keepdims=True)
        mu = np.nanmean(fs, axis=1, keepdims=True)
        sd = np.nanstd(fs, axis=1, keepdims=True)
        dev = 50 + 10 * (s - mu) / np.where(sd > 0, sd, 1.0)
        od = a["odds"]
        el = self.valid & (s >= pmin) & (s / mx >= rmin) & np.isfinite(od) & (od >= omin)
        if dmin is not None:
            el &= dev >= dmin
        if omax is not None:
            el &= od <= omax
        key_s = np.where(el, -s, np.inf)
        order = np.argsort(key_s, axis=1, kind="stable")
        rk = np.empty_like(order)
        np.put_along_axis(rk, order, np.arange(NH)[None, :].repeat(self.N, 0), axis=1)
        R = np.where(el, rk, 99).astype(np.int16)
        if len(self._R) > 300:
            self._R.clear()
        self._R[key] = R
        return R


def _comps(pp: Prep, st: dict, R: np.ndarray | None, sh) -> list[tuple[str, np.ndarray]]:
    """買い目の部品（券種の鍵, 買う印 [N, m]）。"""
    kind = st["kind"]
    N = pp.N
    if kind == "tan":
        return [("tan", np.ones((N, 1), bool))]
    if kind == "fuku":
        return [("fuku", (pp.a["fuku_sold"] > 0)[:, None])]
    n2, n3 = sh
    C = R < n3
    if kind == "waku":
        Mf = (C[:, :, None] & pp.frame1h).any(axis=1) & (pp.a["waku_sold"] > 0)[:, None]
        return [("waku", Mf)]
    if kind in ("um", "wide", "uhj", "ujh"):
        return [(kind, C)]
    if kind == "ut_multi":
        return [("uhj", C), ("ujh", C)]
    Ra, Rb = R[:, :, None], R[:, None, :]
    if kind == "trio_form":
        lo, hi = np.minimum(Ra, Rb), np.maximum(Ra, Rb)
        M = (lo < n2) & (hi < n3) & pp.upper
        return [("trio", M.reshape(N, -1))]
    if kind == "t1_form":
        M = (Ra < n2) & (Rb < n3) & pp.offdiag
        return [("t1", M.reshape(N, -1))]
    if kind in ("t2", "t3"):
        M = (Ra < n3) & (Rb < n3) & pp.offdiag
        return [(kind, M.reshape(N, -1))]
    if kind == "t_multi":
        M = ((Ra < n3) & (Rb < n3) & pp.offdiag).reshape(N, -1)
        return [("t1", M), ("t2", M), ("t3", M)]
    raise ValueError(kind)


def _flat(pp: Prep, key: str, name: str) -> np.ndarray:
    x = getattr(pp, name)[key]
    return x.reshape(pp.N, -1)


def play_rows(pp: Prep, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """規則 spec を全行で実行 → 行ごとの (購入額, 払戻額, 点数)。"""
    st = SBYID[spec["s"]]
    R = pp.rank(st["score"], tuple(spec["f"])) if spec.get("f") is not None else None
    sk = spec.get("skip") or {}
    if st["kind"] in BOX:
        n = spec["sh"][1]
        S = (R < n) | pp.isH
        m = S.sum(axis=1)
        if st["kind"] == "trio_box":
            pts = np.array([comb(int(x), 3) for x in m])
            w = pp.a["w_trio"]
            ok = (w[:, :, 0] >= 0)
            hit = ok & np.take_along_axis(S, np.clip(w[:, :, 0], 0, NH - 1), 1) & np.take_along_axis(S, np.clip(w[:, :, 1], 0, NH - 1), 1) \
                & np.take_along_axis(S, np.clip(w[:, :, 2], 0, NH - 1), 1)
            ret = (hit * np.maximum(w[:, :, 3], 0)).sum(axis=1)
        else:
            pts = np.array([comb(int(x), 2) for x in m])
            w = pp.a["w_um" if st["kind"] == "um_box" else "w_wide"]
            ok = (w[:, :, 0] >= 0)
            hit = ok & np.take_along_axis(S, np.clip(w[:, :, 0], 0, NH - 1), 1) & np.take_along_axis(S, np.clip(w[:, :, 1], 0, NH - 1), 1)
            ret = (hit * np.maximum(w[:, :, 2], 0)).sum(axis=1)
        if sk.get("cap") is not None:
            pts = np.where(pts > sk["cap"], 0, pts)
        ret = np.where(pts > 0, ret, 0)
        return (pts * UNIT).astype(np.float64), ret.astype(np.float64), pts.astype(np.float64)
    comps = _comps(pp, st, R, spec["sh"])
    Ms, Es, Ps, Ys = [], [], [], []
    for key, M in comps:
        Ms.append(M)
        Es.append(_flat(pp, key, "E"))
        Ps.append(_flat(pp, key, "P"))
        Ys.append(_flat(pp, key, "Y"))
    M = np.concatenate(Ms, axis=1) if len(Ms) > 1 else Ms[0]
    E = np.concatenate(Es, axis=1) if len(Es) > 1 else Es[0]
    P = np.concatenate(Ps, axis=1) if len(Ps) > 1 else Ps[0]
    Y = np.concatenate(Ys, axis=1) if len(Ys) > 1 else Ys[0]
    ev = np.where(np.isfinite(E), P * np.where(np.isfinite(E), E, 0.0), 0.0)
    if spec.get("ev") is not None:
        M = M & (ev >= spec["ev"])
    stake = spec.get("stake", "equal")
    if stake == "equal":
        u = M.astype(np.float32)
    elif stake == "payout":
        w = np.where(M, 1.0 / np.maximum(E, 1.0), 0.0)
        pts0 = M.sum(axis=1, keepdims=True)
        u = np.where(M, np.maximum(1.0, np.rint(pts0 * w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12))), 0.0)
    elif stake == "kelly":
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(M & (E > 1.0), np.maximum(ev - 1.0, 0.0) / (E - 1.0), 0.0)
        u = np.clip(np.rint(KELLY_BANK * KELLY_FRAC * f / UNIT), 0, KELLY_CAP)
    else:
        raise ValueError(stake)
    pts = (u > 0).sum(axis=1)
    keep = pts > 0
    if sk.get("cap") is not None:
        keep &= pts <= sk["cap"]
    if sk.get("synth") is not None:
        with np.errstate(divide="ignore"):
            syn = 1.0 / np.where(u > 0, 1.0 / E, 0.0).sum(axis=1)
        keep &= syn >= sk["synth"]
    if sk.get("rev") is not None:
        tot = u.sum(axis=1)
        rev = (u * ev).sum(axis=1) / np.maximum(tot, 1e-12)
        keep &= rev >= sk["rev"]
    u = u * keep[:, None]
    cost = u.sum(axis=1) * UNIT
    ret = (u * Y).sum(axis=1)
    return cost.astype(np.float64), ret.astype(np.float64), (u > 0).sum(axis=1).astype(np.float64)


def to_races(pp: Prep, x: np.ndarray) -> np.ndarray:
    """行（レース × 版）→ レース（版の平均）。"""
    return np.bincount(pp.ridx, weights=x, minlength=pp.nr) / pp.k


# ---------------------------------------------------------------- 物差し
def year_stats(pp: Prep, cost_r: np.ndarray, ret_r: np.ndarray, bet_r: np.ndarray) -> dict:
    out = {}
    for y in pp.years:
        m = pp.ryear == y
        r = ret_r[m]
        top = np.sort(r)[-5:][::-1] if len(r) else np.zeros(0)
        out[int(y)] = {"cost": float(cost_r[m].sum()), "ret": float(r.sum()), "nb": float(bet_r[m].sum()),
                       "top5": [float(t) for t in top]}
    return out


def robust(ys: dict, years) -> tuple[float, float]:
    """(物差し, 回収率)。物差し = min(上位 5 件を除いた回収率, 年ごとの回収率の最小)。どれかの年で 100 レース未満なら -inf。"""
    c = sum(ys[y]["cost"] for y in years)
    r = sum(ys[y]["ret"] for y in years)
    if c <= 0 or any(ys[y]["nb"] < MIN_BETS_PER_YEAR for y in years):
        return -np.inf, (r / c if c > 0 else np.nan)
    top = sorted((t for y in years for t in ys[y]["top5"]), reverse=True)[:5]
    ex = (r - sum(top)) / c
    ymin = min(ys[y]["ret"] / ys[y]["cost"] for y in years)
    return min(ex, ymin), r / c


def eval_spec(pp: Prep, spec: dict) -> dict:
    c, r, pts = play_rows(pp, spec)
    cr, rr = to_races(pp, c), to_races(pp, r)
    br = to_races(pp, (c > 0).astype(float))
    return {"ys": year_stats(pp, cr, rr, br), "cost_r": cr, "ret_r": rr, "bet_r": br, "rows": (c, r, pts)}


# ---------------------------------------------------------------- 段 A（並列）
_G: dict = {}


def _boot_W(nr: int, seed: int = 20261123) -> np.ndarray:
    return np.random.default_rng(seed).poisson(1.0, size=(N_BOOT, nr)).astype(np.float32)


def _stage_a_worker(sid: str) -> dict:
    pp = _G["pp"]
    W = _G["W"]
    st = SBYID[sid]
    rules = stage_a_rules(st)
    stats, vmax, vstar = [], -np.inf, np.full(N_BOOT, -np.inf)
    for spec in rules:
        e = eval_spec(pp, spec)
        stats.append(e["ys"])
        ok = all(e["ys"][y]["nb"] >= MIN_BETS_PER_YEAR for y in pp.years)
        if ok and e["cost_r"].sum() > 0:
            roi = e["ret_r"].sum() / e["cost_r"].sum()
            bc_ = W @ e["cost_r"].astype(np.float32)
            br_ = W @ e["ret_r"].astype(np.float32)
            vstar = np.maximum(vstar, br_ / np.maximum(bc_, 1e-9) - roi)
            vmax = max(vmax, roi - 1.0)
    return {"sid": sid, "rules": rules, "stats": stats, "vmax": vmax, "vstar": vstar}


def _ev_worker(spec: dict) -> dict:
    e = eval_spec(_G["pp"], spec)
    e.pop("rows")
    return e


def stage_a(pp: Prep, workers: int = 8) -> dict:
    _G["pp"], _G["W"] = pp, _boot_W(pp.nr)
    ids = [s["id"] for s in STRUCTS]
    if workers <= 1:
        res = [_stage_a_worker(i) for i in ids]
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_stage_a_worker, ids))
    return {r["sid"]: r for r in res}


def reality_check(A: dict) -> dict:
    """White の reality check: 「どれかの規則の回収率が 100% を超える」の p 値（再中心化のブートストラップ、レース単位）。"""
    out = {}
    allv, alls = -np.inf, np.full(N_BOOT, -np.inf)
    for sid, r in A.items():
        if np.isfinite(r["vmax"]):
            out[sid] = {"best_roi_minus_1": float(r["vmax"]), "p_value": float(np.mean(r["vstar"] >= r["vmax"]))}
            allv, alls = max(allv, r["vmax"]), np.maximum(alls, r["vstar"])
    out["_all"] = {"best_roi_minus_1": float(allv), "p_value": float(np.mean(alls >= allv)) if np.isfinite(allv) else None}
    return out


# ---------------------------------------------------------------- 段 B・C と手順全体
def stage_b_variants(spec: dict) -> list[dict]:
    st = SBYID[spec["s"]]
    if st["kind"] in BOX:
        return [dict(spec, stake="equal", skip={"cap": c}) for c in CAPS]
    out = []
    for stake, cap, syn, rev in itertools.product(STAKES, CAPS, SYNTH_MIN, REV_MIN):
        out.append(dict(spec, stake=stake, skip={"cap": cap, "synth": syn, "rev": rev}))
    return out


def spec_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, ensure_ascii=False, default=list)


class Searcher:
    def __init__(self, pp: Prep, A: dict, workers: int = 1):
        self.pp, self.A, self.workers = pp, A, workers
        self.cache: dict[str, dict] = {}

    def prefetch(self, specs: list[dict]) -> None:
        """まだ計算していない規則を並列で計算しておく（結果は ev と同じ）。"""
        todo = {spec_key(s): s for s in specs if spec_key(s) not in self.cache}
        if not todo:
            return
        if self.workers <= 1:
            for s in todo.values():
                self.ev(s)
            return
        import multiprocessing as mp

        _G["pp"] = self.pp
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("fork")) as ex:
            for k, e in zip(todo, ex.map(_ev_worker, list(todo.values()), chunksize=4)):
                self.cache[k] = e

    def ev(self, spec: dict) -> dict:
        k = spec_key(spec)
        if k not in self.cache:
            e = eval_spec(self.pp, spec)
            e.pop("rows")
            self.cache[k] = e
        return self.cache[k]

    def pair(self, s1: dict, s2: dict) -> dict:
        e1, e2 = self.ev(s1), self.ev(s2)
        cr, rr = e1["cost_r"] + e2["cost_r"], e1["ret_r"] + e2["ret_r"]
        # 参戦（版の平均）: どちらかを買った行。行の単位が要るので、両方の行から作り直す
        c1, _, _ = play_rows(self.pp, s1)
        c2, _, _ = play_rows(self.pp, s2)
        br = to_races(self.pp, ((c1 + c2) > 0).astype(float))
        return {"ys": year_stats(self.pp, cr, rr, br), "cost_r": cr, "ret_r": rr, "bet_r": br}

    def run(self, years) -> dict:
        """段 A〜C を years だけで行う。返り値: 組み方ごとの最良、組み合わせの最良、全体の最良、試した数。"""
        years = list(years)
        per, n_b = {}, 0
        bests = {}
        for sid, r in self.A.items():
            sc = [robust(ys, years) for ys in r["stats"]]
            j = int(np.argmax([s[0] for s in sc]))
            bests[sid] = (sc, j)
        self.prefetch([v for sid, (sc, j) in bests.items() if np.isfinite(sc[j][0]) for v in stage_b_variants(self.A[sid]["rules"][j])])
        for sid, r in self.A.items():
            sc, j = bests[sid]
            if not np.isfinite(sc[j][0]):
                per[sid] = None
                continue
            best_a = dict(r["rules"][j], stake="equal", skip={})
            cand = [(sc[j][0], sc[j][1], best_a)]
            for v in stage_b_variants(r["rules"][j]):
                n_b += 1
                s, roi = robust(self.ev(v)["ys"], years)
                cand.append((s, roi, v))
            cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
            per[sid] = {"stage_a": best_a, "stage_a_score": sc[j][0], "final": cand[0][2], "score": cand[0][0], "roi": cand[0][1]}
        ok = sorted((sid for sid in per if per[sid]), key=lambda s: per[s]["score"], reverse=True)
        pairs = set(itertools.combinations(ok[:TOP_PAIRS], 2))
        pairs |= {p for p in MANDATED_PAIRS if per.get(p[0]) and per.get(p[1])}
        pres = []
        for a_, b_ in sorted(pairs):
            e = self.pair(per[a_]["final"], per[b_]["final"])
            s, roi = robust(e["ys"], years)
            pres.append({"pair": [a_, b_], "score": s, "roi": roi})
        pres.sort(key=lambda x: (x["score"], x["roi"]), reverse=True)
        best = max([{"kind": "struct", "sid": s, "score": per[s]["score"]} for s in ok] +
                   [{"kind": "pair", "pair": p["pair"], "score": p["score"]} for p in pres[:1]], key=lambda x: x["score"]) if ok else None
        n_a = sum(len(r["rules"]) for r in self.A.values())
        return {"per": per, "pairs": pres, "best": best, "n_tried": {"A": n_a, "B": n_b, "C": len(pres)}}


def spec_of(sel: dict, item: dict) -> list[dict]:
    """全体の最良（組み方 1 つ、または組み合わせ）を規則の並びにする。"""
    if item["kind"] == "struct":
        return [sel["per"][item["sid"]]["final"]]
    return [sel["per"][s]["final"] for s in item["pair"]]


def cv(se: Searcher, years) -> dict:
    """手順ごとの 1 年抜きの交差検証: 3 年で段 A〜C を選び、抜いた 1 年の回収率を見る。"""
    out = {"per": {}, "best": {"cost": 0.0, "ret": 0.0, "folds": []}}
    for y in years:
        fit = [x for x in years if x != y]
        sel = se.run(fit)
        for sid, v in sel["per"].items():
            d = out["per"].setdefault(sid, {"cost": 0.0, "ret": 0.0, "folds": []})
            if v is None:
                d["folds"].append({"year": y, "spec": None})
                continue
            ys = se.ev(v["final"])["ys"][y]
            d["cost"] += ys["cost"]; d["ret"] += ys["ret"]
            d["folds"].append({"year": y, "roi": ys["ret"] / ys["cost"] if ys["cost"] else None, "nb": ys["nb"], "spec": v["final"]})
        if sel["best"]:
            specs = spec_of(sel, sel["best"])
            c = sum(se.ev(s)["ys"][y]["cost"] for s in specs)
            r = sum(se.ev(s)["ys"][y]["ret"] for s in specs)
            out["best"]["cost"] += c; out["best"]["ret"] += r
            out["best"]["folds"].append({"year": y, "roi": r / c if c else None, "best": sel["best"]})
    for d in list(out["per"].values()) + [out["best"]]:
        d["roi_cv"] = d["ret"] / d["cost"] if d["cost"] else None
        d["stable"] = len({json.dumps(f.get("spec") or f.get("best"), sort_keys=True, default=list) for f in d["folds"]})
    return out


# ---------------------------------------------------------------- 期間の成績
def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def period_metrics(pp: Prep, specs: list[dict], seed: int = 20261123, base: list[dict] | None = None) -> dict:
    """規則（1 つ、または同じレースで両方買う 2 つ）の期間の成績。区間はレース単位のブートストラップ（同じレースの版に同じ重み）。
    base（基準の規則）があれば、同じレース・同じ重みで回収率の差（規則 − 基準）と 95% 区間も出す。"""
    c = np.zeros(pp.N); r = np.zeros(pp.N); pts = np.zeros(pp.N)
    for s in specs:
        c_, r_, p_ = play_rows(pp, s)
        c += c_; r += r_; pts += p_
    b = c > 0
    k = pp.k
    cr, rr = to_races(pp, c), to_races(pp, r)
    W = _boot_W(pp.nr, seed)
    bs = (W @ rr.astype(np.float32)) / np.maximum(W @ cr.astype(np.float32), 1e-9)
    hit = (r > 0) & b
    nb = b.sum() / k
    fin = pp.rows.meta["H_fin"].to_numpy()
    rec = [float(((fin == i) & b).sum() / k) for i in (1, 2, 3)]
    rec.append(float(nb - sum(rec)))
    hl, hh = wilson(hit.sum() / k, nb)
    top = np.sort(rr)[-5:]
    diff = {}
    if base is not None and cr.sum():
        c0 = np.zeros(pp.N); r0 = np.zeros(pp.N)
        for s0 in base:
            a_, b_, _ = play_rows(pp, s0)
            c0 += a_; r0 += b_
        cr0, rr0 = to_races(pp, c0), to_races(pp, r0)
        if not cr0.sum():
            return {**period_metrics(pp, specs, seed), "diff_roi": None}
        bs0 = (W @ rr0.astype(np.float32)) / np.maximum(W @ cr0.astype(np.float32), 1e-9)
        d = bs - bs0
        diff = {"diff_roi": float(rr.sum() / cr.sum() - rr0.sum() / cr0.sum()), "diff_lo": float(np.quantile(d, 0.025)),
                "diff_hi": float(np.quantile(d, 0.975)), "base_roi": float(rr0.sum() / cr0.sum())}
    return {**diff, "races": int(pp.nr), "bet_races": float(nb), "points_avg": float(pts[b].mean()) if b.any() else None,
            "cost": float(cr.sum()), "ret": float(rr.sum()),
            "hit_rate": float(hit.sum() / b.sum()) if b.any() else None, "hit_lo": hl, "hit_hi": hh,
            "roi": float(rr.sum() / cr.sum()) if cr.sum() else None,
            "roi_lo": float(np.quantile(bs, 0.025)) if cr.sum() else None, "roi_hi": float(np.quantile(bs, 0.975)) if cr.sum() else None,
            "roi_ex_top5": float((rr.sum() - top.sum()) / cr.sum()) if cr.sum() else None,
            "honmei": rec, "versions": k}


def cpu_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 3))
