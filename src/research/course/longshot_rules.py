"""穴馬の母集団で、コースの中のめぐ分析の条件を探す道具（事前登録 config/longshot_rules/prereg.json）。

- 母集団: 2勝クラス以上（2勝・3勝・OP・L・重賞）、芝ダ、馬場はすべて、T-60 の再現オッズ（simulate_pre の種 1000 の 1 本）で 10〜50 倍。
- 単位: 会場 × 芝ダ × 距離。条件は catalog の水準（L・M・start_year 2019 以上・I09 を除く）の深さ 1〜2。
- 分け方: 2016〜24 の開催（会場 × 年 × 回）を無作為に 5 つ（交差検証）。
- 縮め方: 条件ごとの勝ち数を市場の見込み（確定の q）の比にし、経験ベイズ（積率法。レースの中の排他を入れた分散）でコース・全体へ縮める。
- 数え方: 水準の one-hot の行列 O（走 × 水準）で、G = Oᵀ diag(y) O を fold・列ごとに作る（対角が深さ 1、型の違う組が深さ 2）。
- 2026-11-24 の結末: 合成の検出力の模擬（longshot_synth.py）で、本物の探索（段 1・段 2）は流さずに閉じた（1 号の決定 D1 改訂 R3-3・R3-4）。
  そのため段 2 の命令（本物の探索・帰無 200 回・合否の判定）は持たない。run_split は合成の模擬から使う。

使い方（重いので nice -n 19。2025・2026 年の着順は読まない）:
  python -m src.research.course.longshot_rules frame
  python -m src.research.course.longshot_rules tau --seeds 10001-10020 --sets gate   # 本物のデータの並べ替えの帰無の τ²（較正の記録）
  python -m src.research.course.longshot_rules describe                              # 母集団の記述と合成の結果を JSON に
出力の置き場: <COURSE_OUT>/longshot_rules、結果は docs/operations/retrain_trials/longshot_rules.json。費用 0 円（手元の表だけ。GCS を読まない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from src.research.course import run as CR

PREREG = CR.REPO / "config/longshot_rules/prereg.json"
OUTW = CR.OUT / "longshot_rules"
OUT_REPO = CR.REPO / "docs/operations/retrain_trials/longshot_rules.json"
FRAME = CR.OUT / "v3" / "frame.parquet"
LEVELS = CR.OUT / "v3" / "levels.parquet"
YEAR_END = 2024
NFOLD = 5
E3_EDGES = np.exp(np.linspace(np.log(10.0), np.log(50.0), 9))   # 確定オッズの 8 帯（外は端の帯）
NB = len(E3_EDGES) - 1
FIELD_EDGES = [8, 13]                                            # 頭数の群: 7 以下・8〜12・13 以上
NG = NB * (len(FIELD_EDGES) + 1)                                 # 複勝の物差しの群（帯 × 頭数）
ERAS = [(2016, 2018), (2019, 2021), (2022, 2024)]
COLS = ["race_id", "horse_id", "date", "year", "venue", "surf", "dist", "cls", "hn", "fin", "field", "ret_tan", "ret_fuku", "pay_ok"]


def prereg() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def cut_types() -> list[str]:
    """使う切り口の型（事前登録の規則: L・M の族、start_year 2019 以上、個別に外す型を除く。市場の列は catalog が禁止済み）。"""
    from src.research.graded import catalog as CAT
    ex = set(prereg()["cuts"]["exclude_types"])
    return [t.id for t in CAT.TYPES if t.family not in ("L", "M") and t.start_year < 2019 and t.id not in ex]


# ---------------------------------------------------------------- 母集団の表
def build_population(band_by: str = "pre") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """2016〜2024 年の 2勝クラス以上の全走に確定オッズ・q・再現オッズを付け、穴馬の行と水準を返す。
    band_by = pre（再現オッズで 10〜50 倍。主）か final（確定オッズ。参考）。"""
    from src.research.course.v2 import attach_final_odds
    from src.research.megu_cycles.mc_data import simulate_pre
    d = pd.read_parquet(FRAME, columns=COLS, filters=[("year", "<=", YEAR_END)])
    yr_all = pd.to_numeric(pd.read_parquet(FRAME, columns=["year"])["year"]).to_numpy()
    keep_rows = np.flatnonzero(yr_all <= YEAR_END)
    assert len(keep_rows) == len(d) and int(pd.to_numeric(d["year"]).max()) <= YEAR_END, "2025 年以降の行が入っている"
    types = cut_types()
    L = pd.read_parquet(LEVELS, columns=types + ["__race_id", "__horse_id"]).iloc[keep_rows].reset_index(drop=True)
    if not ((d["race_id"].astype(str).to_numpy() == L["__race_id"].to_numpy()) & (d["horse_id"].astype(str).to_numpy() == L["__horse_id"].to_numpy())).all():
        raise ValueError("frame と levels の行がずれている")
    cls = pd.to_numeric(d["cls"], errors="coerce")
    m = ((cls >= 3) & (cls <= 8) & d["surf"].astype(str).isin(["芝", "ダ"])).to_numpy()
    d, L = d[m].reset_index(drop=True), L[m].reset_index(drop=True)
    d = attach_final_odds(d, range(2016, YEAR_END + 1))
    ok = (d["odds_final"] > 0).to_numpy()
    d, L = d[ok].reset_index(drop=True), L[ok].reset_index(drop=True)
    d["race_id"] = d["race_id"].astype(str)
    order = np.lexsort((pd.to_numeric(d["hn"], errors="coerce").to_numpy(), d["race_id"].to_numpy()))
    d, L = d.iloc[order].reset_index(drop=True), L.iloc[order].reset_index(drop=True)
    inv = 1.0 / d["odds_final"].to_numpy(float)
    d["q"] = inv / pd.Series(inv).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    d["pre"] = simulate_pre(d, n=1, phase="t60")[0]
    lo, hi = prereg()["population"]["odds_band"]
    fin_ok = d["odds_final"].between(lo, hi).to_numpy()
    pre_ok = d["pre"].between(lo, hi).to_numpy()
    info = {"band_by": band_by, "races": int(d["race_id"].nunique()), "runners": int(len(d)), "longshot_pre": int(pre_ok.sum()),
            "longshot_final": int(fin_ok.sum()), "final_in_pre_out": int((fin_ok & ~pre_ok).sum()), "final_out_pre_in": int((~fin_ok & pre_ok).sum()),
            "pre_sha256": hashlib.sha256(np.round(d["pre"].to_numpy(float), 6).tobytes()).hexdigest()}
    keep = pre_ok if band_by == "pre" else fin_ok
    # 合成（Harville の引き直し）用に、レースの全頭の q・確定オッズを残す（着順は入れない）
    full = pd.DataFrame({"race_id": d["race_id"].to_numpy(), "q": d["q"].to_numpy(), "odds_final": d["odds_final"].to_numpy(),
                         "field": pd.to_numeric(d["field"], errors="coerce").to_numpy(), "pop_row": np.where(keep, np.cumsum(keep) - 1, -1)})
    info["full_races"] = full
    d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True).drop(columns=["__race_id", "__horse_id"])
    d["unit"] = d["venue"].astype(str) + d["surf"].astype(str) + d["dist"].astype(int).astype(str)
    d["kaisai"] = d["race_id"].str[:8]
    d["cgrp"] = np.clip(pd.to_numeric(d["cls"]).to_numpy(int) - 3, 0, 2)       # 2勝・3勝・OP 以上
    yr = pd.to_numeric(d["year"]).to_numpy(int)
    d["era"] = np.select([yr <= e[1] for e in ERAS], range(len(ERAS)), len(ERAS) - 1)
    d["obin"] = np.clip(np.searchsorted(E3_EDGES, d["odds_final"].to_numpy(float), side="right") - 1, 0, NB - 1)
    fg = np.searchsorted(FIELD_EDGES, pd.to_numeric(d["field"], errors="coerce").fillna(12).to_numpy(), side="right")
    d["e3g"] = d["obin"].to_numpy() * (len(FIELD_EDGES) + 1) + fg
    fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy()
    pay = d["pay_ok"].fillna(False).astype(bool).to_numpy()
    d["rt"] = np.where(pay, pd.to_numeric(d["ret_tan"], errors="coerce").fillna(0).to_numpy(), 0.0)
    d["rf"] = np.where(pay, pd.to_numeric(d["ret_fuku"], errors="coerce").fillna(0).to_numpy(), 0.0)
    d["win"] = ((fin == 1) & pay).astype(float)
    d["top3"] = ((fin <= 3) & (d["rf"].to_numpy() > 0)).astype(float)       # 複勝の払戻のある 3 着内
    d["qo"] = d["q"] * d["odds_final"]
    return d, L, info


def assign_folds(d: pd.DataFrame, seed: int) -> tuple[np.ndarray, str]:
    """開催（race_id の先頭 8 桁）を無作為に 5 つに分ける。割り当ての sha256 も返す。"""
    keys = np.array(sorted(d["kaisai"].unique()))
    perm = np.random.default_rng(seed).permutation(len(keys))
    fold_of = {k: i % NFOLD for i, k in enumerate(keys[perm])}
    sha = hashlib.sha256(json.dumps(sorted(fold_of.items()), ensure_ascii=False).encode()).hexdigest()
    return d["kaisai"].map(fold_of).to_numpy(int), sha


def cmd_frame(band_by: str = "pre") -> None:
    OUTW.mkdir(parents=True, exist_ok=True)
    P = prereg()
    d, L, info = build_population(band_by)
    d["fold"], sha = assign_folds(d, P["split"]["seed"])
    sfx = "" if band_by == "pre" else "_final"
    info.pop("full_races").to_parquet(OUTW / f"race_all{sfx}.parquet", index=False)
    d.to_parquet(OUTW / f"pop{sfx}.parquet", index=False)
    L.to_parquet(OUTW / f"pop_levels{sfx}.parquet", index=False)
    info |= {"fold_sha256": sha, "types": list(L.columns), "units": int(d["unit"].nunique()), "kaisai": int(d["kaisai"].nunique()),
             "kaisai_days": int((d["date"].astype(str) + d["venue"].astype(str)).nunique())}
    (OUTW / f"pop_info{sfx}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("frame", {k: v for k, v in info.items() if k != "types"}, flush=True)


def load_pop(sfx: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(OUTW / f"pop{sfx}.parquet")
    L = pd.read_parquet(OUTW / f"pop_levels{sfx}.parquet")
    assert int(d["year"].max()) <= YEAR_END and len(d) == len(L)
    return d, L


# ---------------------------------------------------------------- 探す・確かめる
def onehot(L: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """水準の one-hot（走 × 水準）。欠け（不明・前走が海外・地方）は列にしない。列の型の番号と、(型, 水準) の名前も返す。"""
    from src.research.graded import model as M
    cols, tid, names = [], [], []
    for j, t in enumerate(L.columns):
        s = L[t].astype("string")
        s = s.where(~s.isin(list(M.NO_INFO)))
        for v in sorted(s.dropna().unique()):
            cols.append((s == v).fillna(False).to_numpy())
            tid.append(j)
            names.append((t, str(v)))
    return np.column_stack(cols).astype(np.float32), np.array(tid), names


# Y の列: 0 走数・1 勝ち・2 複勝の当たり・3 単の払戻・4 複の払戻・5 q・6 q×オッズ・7〜 複勝の物差しの群の数
def ymat(d: pd.DataFrame) -> np.ndarray:
    g = d["e3g"].to_numpy()
    return np.column_stack([np.ones(len(d)), d["win"], d["top3"], d["rt"], d["rf"], d["q"], d["qo"]]
                           + [(g == j).astype(float) for j in range(NG)]).astype(np.float64)


def group_rates(d: pd.DataFrame, search: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """探す側の全穴馬の、帯 × 頭数の群ごとの複勝の当たりの率と、当たりのときの複勝の払戻の平均（空の群は帯の値）。"""
    s = d[search]
    rate = s.groupby("e3g")["top3"].mean().reindex(range(NG))
    pay = s["rf"].where(s["top3"] > 0).groupby(s["e3g"]).mean().reindex(range(NG))
    rb = s.groupby("obin")["top3"].mean().reindex(range(NB)).fillna(0).to_numpy()
    pb = s["rf"].where(s["top3"] > 0).groupby(s["obin"]).mean().reindex(range(NB)).fillna(0).to_numpy()
    nf = len(FIELD_EDGES) + 1
    rate = rate.fillna(pd.Series(np.repeat(rb, nf))).to_numpy()
    pay = pay.fillna(pd.Series(np.repeat(pb, nf))).to_numpy()
    return rate, pay


def _tau(W: np.ndarray, E: np.ndarray, V: np.ndarray, prior: np.ndarray) -> float:
    """積率法: 比の真のばらつき τ² = (Σ (W − pE)²/E − Σ V/E) / Σ E。V は事前の比 p のもとでの W の分散（レースの中の排他を入れた値）。"""
    m = E > 0
    if not m.any():
        return 0.0
    W, E, V, p = W[m], E[m], V[m], prior[m]
    return float((np.sum((W - p * E) ** 2 / E) - np.sum(V / E)) / np.sum(E))


def race_noise(ridx: np.ndarray, o: np.ndarray, q: np.ndarray, e: np.ndarray, T: np.ndarray, N: np.ndarray,
               pw: np.ndarray, pp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """条件（o の列）ごとの、事前の比のもとでの W・K の分散（レースの中の排他を入れる）。
    単勝: Σ_r P_r(1 − P_r)、P_r = p_w × そのレースの条件の q の和（1 レースの勝ちは 1 頭なので正確）。
    複勝: Σ_r p_p·E3_r·(1 − T_r/N_r)(N_r − m_r)/(N_r − 1)（払戻の枠 T・頭数 N・条件の頭数 m の、等確率の非復元抽出の近似）。
    ridx は行のレースの番号（0 から）、T・N はレースごとの値。"""
    nr = int(ridx.max()) + 1 if len(ridx) else 0
    Z = np.zeros((nr, o.shape[1]))
    np.add.at(Z, ridx, o * q[:, None])
    P = Z * pw[None, :]
    Vw = (P * (1 - P)).sum(0)
    E3 = np.zeros((nr, o.shape[1]))
    np.add.at(E3, ridx, o * e[:, None])
    M = np.zeros((nr, o.shape[1]))
    np.add.at(M, ridx, o)
    f = ((1 - T / N) / np.maximum(N - 1, 1))[:, None]
    Vp = (pp[None, :] * E3 * f * (N[:, None] - M)).sum(0)
    return Vw, Vp


def _shrink(W, E, prior, tau2):
    if tau2 <= 0:
        return np.broadcast_to(prior, np.shape(W)).astype(float)
    a = prior / tau2
    return (W + a * prior) / (E + a)


def estimate_shrink(d: pd.DataFrame, O: np.ndarray, Y: np.ndarray, ulist: list, c: dict, emin: tuple[float, float],
                    noise: bool = True) -> dict:
    """探す側（c["s"]）だけで、全体の比・コースの比のばらつき τc²・条件の比のばらつき τ²（深さ 1、条件を抜いたコースの比を事前に）を推す。
    emin = (単勝の E の下限, 複勝の E3 の下限)。単勝は E、複勝は E3 が下限以上の深さ 1 の条件で推す。"""
    rid_all = d["race_id"].to_numpy()
    q_all, e_all = Y[:, 5], c["rate"][d["e3g"].to_numpy()]
    field = pd.to_numeric(d["field"], errors="coerce").fillna(12).to_numpy(float)
    per = []
    for u, r, live in ulist:
        sr = r[c["s"][r]]
        if not len(sr):
            continue
        ridx, _ = pd.factorize(rid_all[sr])
        N = pd.Series(field[sr]).groupby(ridx).first().to_numpy()
        T = np.where(N <= 7, 2.0, 3.0)
        y = Y[sr]
        o = O[sr][:, live].astype(np.float64)
        cE, cE3 = o.T @ q_all[sr], o.T @ e_all[sr]
        ow, op = o[:, cE >= emin[0]], o[:, cE3 >= emin[1]]
        per.append({"u": u, "sr": sr, "ridx": ridx, "N": N, "T": T, "ow": ow, "op": op,
                    "W": y[:, 1].sum(), "E": q_all[sr].sum(), "K": y[:, 2].sum(), "E3": e_all[sr].sum(),
                    "cW": ow.T @ y[:, 1], "cE": ow.T @ q_all[sr], "cK": op.T @ y[:, 2], "cE3": op.T @ e_all[sr]})
    CW, CE, CK, CE3 = (np.array([x[k] for x in per]) for k in ("W", "E", "K", "E3"))
    rw_all, rp_all = CW.sum() / CE.sum(), CK.sum() / CE3.sum()
    VW, VP = [], []
    for x in per:          # コース全体を 1 つの条件とみた雑音
        one = np.ones((len(x["sr"]), 1))
        vw, vp = race_noise(x["ridx"], one, q_all[x["sr"]], e_all[x["sr"]], x["T"], x["N"], np.array([rw_all]), np.array([rp_all]))
        VW.append(vw[0] * noise); VP.append(vp[0] * noise)   # noqa: E702
    tcw = _tau(CW, CE, np.array(VW), np.full(len(CW), rw_all))
    tcp = _tau(CK, CE3, np.array(VP), np.full(len(CK), rp_all))
    cw, ce, vw_, pw_, ck, ce3, vp_, pp_ = [], [], [], [], [], [], [], []
    for x in per:
        pw = np.broadcast_to(_shrink(x["W"] - x["cW"], x["E"] - x["cE"], rw_all, tcw), x["cW"].shape).astype(float)
        pp = np.broadcast_to(_shrink(x["K"] - x["cK"], x["E3"] - x["cE3"], rp_all, tcp), x["cK"].shape).astype(float)
        q_, e_ = q_all[x["sr"]], e_all[x["sr"]]
        vw, _ = race_noise(x["ridx"], x["ow"], q_, e_, x["T"], x["N"], pw, np.ones(x["ow"].shape[1]))
        _, vp = race_noise(x["ridx"], x["op"], q_, e_, x["T"], x["N"], np.ones(x["op"].shape[1]), pp)
        cw.append(x["cW"]); ce.append(x["cE"]); vw_.append(vw * noise); pw_.append(pw)   # noqa: E702
        ck.append(x["cK"]); ce3.append(x["cE3"]); vp_.append(vp * noise); pp_.append(pp)   # noqa: E702
    cat = np.concatenate
    tw = _tau(cat(cw), cat(ce), cat(vw_), cat(pw_))
    tp = _tau(cat(ck), cat(ce3), cat(vp_), cat(pp_))
    return {"r_all_w": float(rw_all), "r_all_p": float(rp_all), "tau2_course_w": tcw, "tau2_course_p": tcp,
            "tau2_cell_w": tw, "tau2_cell_p": tp, "n_cells_w": int(sum(len(x) for x in cw)), "n_cells_p": int(sum(len(x) for x in ck)),
            "n_courses_w": int(sum(len(x) > 0 for x in cw)), "n_courses_p": int(sum(len(x) > 0 for x in ck)),
            "course": {x["u"]: {"W": float(x["W"]), "E": float(x["E"]), "K": float(x["K"]), "E3": float(x["E3"])} for x in per}}


def make_ulist(d: pd.DataFrame, O: np.ndarray, emin: tuple[float, float], rows: np.ndarray | None = None) -> list:
    """コースごとに、行の全体（探す側 ⊂ 全体）で E か E3 が下限に届きうる水準だけを残す（探す前の間引き。着順は使わない）。"""
    units = d["unit"].to_numpy()
    rows = np.arange(len(d)) if rows is None else rows
    q = d["q"].to_numpy(float)
    out = []
    for u in np.unique(units[rows]):
        r = rows[units[rows] == u]
        o = O[r]
        cE = o.T @ q[r]
        n = o.sum(0)
        # 複勝は走数で間引く近似（3 着内の率を 0.30 とみる）。7 頭以下・10 倍前後の群は 0.30 を超えるので、ごく一部の水準を落としうる（記録の結果はこの作りで出した）
        live = np.flatnonzero((cE >= emin[0]) | (n >= emin[1] / 0.30))
        if len(live):
            out.append((u, r, live))
    return out


class Split:
    """探す側と確かめの側の組（交差検証の 5 つ、または前向きの 1 つ）。part は行ごとの塊の番号、hold は確かめに使う塊の一覧。"""

    def __init__(self, part: np.ndarray, holds: list[int], npart: int):
        self.part, self.holds, self.npart = part, holds, npart


def cv_split(d: pd.DataFrame) -> Split:
    return Split(d["fold"].to_numpy(), list(range(NFOLD)), NFOLD)


def run_split(d: pd.DataFrame, L: pd.DataFrame, P: dict, sp: Split, keep_cells: bool = False, calib: dict | None = None,
              emin: tuple[float, float] | None = None) -> dict:
    """探して採り、確かめの側で買う。買った行の印（単・複）と、試した数・採った数・縮小の値を返す。
    calib[k] = {"cell_w", "cell_p", "course_p"}: 帰無の平均（引く値）。縮めに使う τ² = (推定 − calib) ÷ k（k は単 k_win・複 k_place）。
    τ² ≤ 0 の券種は条件を採らない。"""
    S = P["search"]
    emin = emin or (S["e_min_win"], S["e3_min_place"])
    kw, kp = S["shrink"]["k_win"], S["shrink"]["k_place"]
    O, tid, names = onehot(L)
    Y = ymat(d)
    part = sp.part
    horse = pd.factorize(d["horse_id"].astype(str))[0]
    ctx = {}
    for k in sp.holds:
        s = part != k
        rate, pay = group_rates(d, s)
        ctx[k] = {"rate": rate, "pay": pay, "rf_pay": rate * pay, "s": s}
    ulist = make_ulist(d, O, emin)
    shrink = {}
    for k in sp.holds:
        sh = estimate_shrink(d, O, Y, ulist, ctx[k], emin)
        cb = (calib or {}).get(k, {"cell_w": 0.0, "cell_p": 0.0, "course_p": 0.0})
        sh["tau2_course_p_used"] = sh["tau2_course_p"] - cb["course_p"]
        sh["tau2_cell_w_used"] = (sh["tau2_cell_w"] - cb["cell_w"]) / kw
        sh["tau2_cell_p_used"] = (sh["tau2_cell_p"] - cb["cell_p"]) / kp
        shrink[k] = sh
    bet = {"t": np.zeros(len(d), bool), "f": np.zeros(len(d), bool)}
    tested = {k: {"t": 0, "f": 0} for k in sp.holds}
    nsel = {k: {"t": 0, "f": 0} for k in sp.holds}
    guard_drop = {k: {"t": 0, "f": 0} for k in sp.holds}
    cells_out = []
    for u, r, live in ulist:
        Ou = O[r][:, live].astype(np.float64)
        tu = tid[live]
        pu, yu = part[r], Y[r]
        G = np.zeros((sp.npart, 7 + NG, len(live), len(live)))
        for f in range(sp.npart):
            mf = pu == f
            Of, Yf = Ou[mf], yu[mf]
            for cc in range(Yf.shape[1]):
                G[f, cc] = (Of * Yf[:, [cc]]).T @ Of
        iu = np.triu_indices(len(live))
        valid = (tu[iu[0]] != tu[iu[1]]) | (iu[0] == iu[1])
        I0, J0 = iu[0][valid], iu[1][valid]
        Gt = G.sum(0)
        for k in sp.holds:
            c, sh = ctx[k], shrink[k]
            Gs = Gt - G[k]
            V = Gs[:, I0, J0]
            n, W, K, q, qo, cnt = V[0], V[1], V[2], V[5], V[6], V[7:]
            E3, E3F = c["rate"] @ cnt, c["rf_pay"] @ cnt
            cu = sh["course"].get(u)
            if cu is None:
                continue
            ok = {"t": q >= emin[0], "f": E3 >= emin[1]}
            est = {}
            if sh["tau2_cell_w_used"] > 0:
                pw_ = _shrink(cu["W"] - W, cu["E"] - q, sh["r_all_w"], sh["tau2_course_w"])      # その条件を抜いたコースの比
                est["t"] = _shrink(W, q, pw_, sh["tau2_cell_w_used"]) * qo / np.maximum(n, 1)
            if sh["tau2_cell_p_used"] > 0:
                pp_ = _shrink(cu["K"] - K, cu["E3"] - E3, sh["r_all_p"], sh["tau2_course_p_used"])
                est["f"] = _shrink(K, E3, pp_, sh["tau2_cell_p_used"]) * E3F / np.maximum(n, 1)
            hold = pu == k
            srch = ~hold
            for kind in ("t", "f"):
                tested[k][kind] += int(ok[kind].sum())
                if kind not in est:
                    continue
                for x in np.flatnonzero(ok[kind] & (est[kind] >= 1.0)):
                    i, j = I0[x], J0[x]
                    m = (Ou[:, i] > 0) & (Ou[:, j] > 0)
                    hs = horse[r[m & srch]]
                    if len(np.unique(hs)) < 30 or np.bincount(hs).max() > 0.10 * len(hs):
                        guard_drop[k][kind] += 1
                        continue
                    nsel[k][kind] += 1
                    bet[kind][r[m & hold]] = True
                    if keep_cells:
                        hm = r[m & hold]
                        cells_out.append({"unit": u, "hold": k, "kind": kind, "a": "=".join(names[live[i]]),
                                          "b": "=".join(names[live[j]]) if i != j else "", "n_search": int(n[x]),
                                          "e_search": float(q[x] if kind == "t" else E3[x]), "est": float(est[kind][x]), "n_hold": int(len(hm)),
                                          "w_hold": float(d["win"].to_numpy()[hm].sum()), "q_hold": float(d["q"].to_numpy()[hm].sum()),
                                          "rt_hold": float(d["rt"].to_numpy()[hm].sum()), "k_hold": float(d["top3"].to_numpy()[hm].sum()),
                                          "e3_hold": float(c["rate"][d["e3g"].to_numpy()[hm]].sum()), "rf_hold": float(d["rf"].to_numpy()[hm].sum())})
    out = {"bet_t": bet["t"], "bet_f": bet["f"], "tested": tested, "sel": nsel, "guard_drop": guard_drop, "ctx": ctx,
           "shrink": {k: {kk: vv for kk, vv in v.items() if kk != "course"} for k, v in shrink.items()}}
    if keep_cells:
        out["cells"] = cells_out
    return out


def permute(d: pd.DataFrame, L: pd.DataFrame, P: dict, seed: int) -> pd.DataFrame:
    """馬の型は コース × クラス群 × 時代 × 確定オッズ 8 帯 の中で行ごと、レース単位の型は コース × クラス群 × 時代 の中でレースの束ごとに入れ替える。"""
    rng = np.random.default_rng(seed)
    race_types = [t for t in P["cuts"]["race_level_types"] if t in L.columns]
    horse_types = [t for t in L.columns if t not in race_types]
    key = d["unit"].astype(str) + "|" + d["cgrp"].astype(str) + "|" + d["era"].astype(str) + "|" + d["obin"].astype(str)
    perm = np.arange(len(L))
    for _, idx in key.groupby(key).indices.items():
        if len(idx) > 1:
            perm[idx] = idx[rng.permutation(len(idx))]
    out = L[horse_types].iloc[perm].reset_index(drop=True)
    if race_types:
        rid = d["race_id"].to_numpy()
        first = pd.Series(np.arange(len(d))).groupby(rid).first()           # レースの代表の行
        rk = (d["unit"].astype(str) + "|" + d["cgrp"].astype(str) + "|" + d["era"].astype(str)).to_numpy()
        races = pd.DataFrame({"race": first.index, "row": first.to_numpy(), "key": rk[first.to_numpy()]})
        src = races["row"].to_numpy().copy()
        for _, idx in races.groupby("key").indices.items():
            if len(idx) > 1:
                src[idx] = src[idx[rng.permutation(len(idx))]]
        mp = dict(zip(races["race"], src))
        rows = np.array([mp[x] for x in rid])
        rl = L[race_types].iloc[rows].reset_index(drop=True)
        out = pd.concat([out, rl], axis=1)
    return out[list(L.columns)]


def tau_sets(d: pd.DataFrame, L: pd.DataFrame, emin: tuple[float, float], sets: str) -> dict:
    """τ² の推定（較正・k なし）を集合ごとに返す。sets = gate（2016〜21 の全部。段 1 の関門と前向きの探す側）か cv（交差検証の 5 つの探す側）。"""
    O, _, _ = onehot(L)
    Y = ymat(d)
    ul = make_ulist(d, O, emin)
    yr = pd.to_numeric(d["year"]).to_numpy()
    masks = {"gate": yr <= 2021} if sets == "gate" else {str(k): d["fold"].to_numpy() != k for k in range(NFOLD)}
    out = {}
    for nm, s_ in masks.items():
        rate, _ = group_rates(d, s_)
        sh = estimate_shrink(d, O, Y, ul, {"rate": rate, "s": s_}, emin)
        out[nm] = {k: v for k, v in sh.items() if k.startswith("tau2") or k.startswith("n_")}
    return out


def _tau_one(args) -> dict:
    seed, emin, sets = args
    P = prereg()
    d, L = load_pop()
    Lp = L if seed == 0 else permute(d, L, P, P["null"]["seed_base"] + seed)
    return {"seed": seed, "emin": list(emin), "sets": sets, "tau": tau_sets(d, Lp, emin, sets)}


def cmd_tau(seeds: list[int], sets: str, workers: int) -> None:
    """帰無（seed ≥ 1）か本物（seed 0）の τ² を集合ごとに書く。較正は 10001〜10020、段 1 の帰無は 21〜220（1 号の決定）。
    本物（seed 0）の gate は段 1 の関門の値なので、prereg の commit の後にだけ流す。"""
    from concurrent.futures import ProcessPoolExecutor
    P = prereg()
    emin = (P["search"]["e_min_win"], P["search"]["e3_min_place"])
    if None in emin:
        sys.exit("prereg の下限が未定")
    tdir = OUTW / "tau"
    tdir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if not (tdir / f"{sets}_{s}.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_tau_one, [(s, emin, sets) for s in todo]):
            (tdir / f"{sets}_{res['seed']}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            print("tau", sets, res["seed"], json.dumps(res["tau"], ensure_ascii=False)[:300], flush=True)
            if _stop_window():
                ex.shutdown(cancel_futures=True)
                sys.exit("開催日の 07:30〜17:00 になったので止める（続きは同じ命令で流す）")


def _stop_window() -> bool:
    """開催日（11/28・29）の 07:30〜17:00 JST は止める（統括役の決まり）。"""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=9)))
    return now.strftime("%m-%d") in ("11-28", "11-29") and (7, 30) <= (now.hour, now.minute) < (17, 0)


def cmd_describe() -> None:
    """母集団の大きさ（年・コース・クラス・帯）と基準線（同じ層の穴馬を全部買ったときの単・複の回収率・市場比）、
    合成の確かめ（段 0・模擬 A・模擬 B・段 1 の検出力）と較正を、docs/operations/retrain_trials/longshot_rules.json に書く。
    条件の探索はしない（全部の穴馬の記述だけ）。2016〜24 だけを読む。"""
    P = prereg()
    d, _ = load_pop()
    info = json.loads((OUTW / "pop_info.json").read_text(encoding="utf-8"))
    rate, _ = group_rates(d, np.ones(len(d), bool))
    d = d.assign(e3=rate[d["e3g"].to_numpy()])

    def boot(x: pd.DataFrame, n_boot: int = 1000, seed: int = 20261124) -> dict:
        """開催で束ねたブートストラップの 95% 区間（単・複の回収率と、単の市場比 W/Σq・複の K/ΣE3）。"""
        g = x.groupby("kaisai").agg(n=("win", "size"), rt=("rt", "sum"), rf=("rf", "sum"), w=("win", "sum"), q=("q", "sum"),
                                    k=("top3", "sum"), e3=("e3", "sum"))
        M = g.to_numpy(float)
        rng = np.random.default_rng(seed)
        W = np.stack([np.bincount(rng.integers(len(g), size=len(g)), minlength=len(g)) for _ in range(n_boot)]).astype(float)
        T = W @ M
        stats = {"roi_t": T[:, 1] / T[:, 0], "roi_f": T[:, 2] / T[:, 0], "mr_t": T[:, 3] / T[:, 4], "mr_f": T[:, 5] / T[:, 6]}
        return {f"{k}_ci": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in stats.items()}

    def agg(g):
        a = g.agg(n=("win", "size"), races=("race_id", "nunique"), wins=("win", "sum"), q=("q", "sum"), top3=("top3", "sum"),
                  e3=("e3", "sum"), roi_t=("rt", "mean"), roi_f=("rf", "mean")).reset_index()
        cis = [boot(x) for _, x in g]
        for k in cis[0]:
            a[k] = [c[k] for c in cis]
        a["mr_t"] = a["wins"] / a["q"]
        a["mr_f"] = a["top3"] / a["e3"]
        return a

    out = {"prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "population": {k: v for k, v in info.items() if k != "types"},
           "types": info.get("types"), "all": agg(d.assign(k="all").groupby("k")).to_dict("records"),
           "by_year": agg(d.groupby("year")).to_dict("records"), "by_unit": agg(d.groupby("unit")).sort_values("n", ascending=False).to_dict("records"),
           "by_cls": agg(d.groupby("cgrp")).to_dict("records"), "by_obin": agg(d.groupby("obin")).to_dict("records"),
           "by_surf": agg(d.groupby("surf")).to_dict("records"), "obin_edges": E3_EDGES.tolist()}
    syn = OUTW / "synth"
    for nm in ("simA", "simB", "power1", "power2"):
        f = syn / f"{nm}.json"
        if f.exists():
            out[nm] = json.loads(f.read_text(encoding="utf-8"))
    out["stage0"] = [x for x in (json.loads(f.read_text(encoding="utf-8")) for f in sorted(syn.glob("stage0_*.json"))) if "emin" in x]
    out["calib"] = P["search"]["shrink"].get("calib")
    out["outcome"] = P.get("outcome")
    OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPO.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    print("describe", OUT_REPO, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frame", "tau", "describe"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--final-odds", action="store_true", help="参考: 確定オッズで母集団を決めた版（合否には使わない）")
    ap.add_argument("--seeds", default="", help="tau: 例 10001-10020")
    ap.add_argument("--sets", default="gate", choices=["gate", "cv"])
    a = ap.parse_args()
    if _stop_window():
        sys.exit("開催日の 07:30〜17:00 は重い計算を止める")
    if a.cmd == "frame":
        cmd_frame("final" if a.final_odds else "pre")
    elif a.cmd == "describe":
        cmd_describe()
    else:
        lo, _, hi = a.seeds.partition("-")
        cmd_tau(list(range(int(lo), int(hi or lo) + 1)), a.sets, a.workers)


if __name__ == "__main__":
    main()
