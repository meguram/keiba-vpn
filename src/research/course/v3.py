"""全レース版の第 3 版（2025 年を見ながら試行錯誤する版。ユーザーの依頼・統括役の作法、2026-11-17）。

- 訓練: 2016〜2024 年（過学習してよい。ユーザーの指示）。条件は訓練の走だけで探す（第 1 版と同じ find_cells）。
- 試行錯誤の評価: 2025 年。サイクルごとに 2025 年の値（本番の再現）を見て、次の仮説を決めてよい。目標は単勝か複勝の回収率 120% 超。
- 封印: 2026 年。サイクルの間は一切読まない（folklore・features・払戻・オッズの表をすべて 2025 年で切る。テストで確かめる）。
  最後に凍結・commit してから 1 回だけ見る。2026 年の本物の T-60 の 156R も、そこに入る。
- 2025・2026 年は、第 1 版・第 2 版の評価で 2 回見ている（記録に書く）。
- 本番の再現: オッズの判定は発走前（T-60）のオッズ、払戻は確定。過去の年の発走前のオッズは、第 2 版で作ったずれの分布
  （<COURSE_OUT>/v2/shift.json。2026 年の T-60 と確定のオッズの組だけで作り、着順・払戻は見ていない）を、確定オッズに逆に当てて 20 通り作る。

使い方（出力は <COURSE_OUT>/v3）:
  python -m src.research.course.v3 frame                                   # master・frame（2016〜2025 年）
  python -m src.research.course.v3 cycle --hyp "仮説" --delta '{"top_k": 30}' # 1 サイクル（今の版に delta を当てる）
  python -m src.research.course.v3 review --n 3 --text "回顧" --decision 採用  # 回顧と判断（採用なら今の版を更新）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import search as SE
from src.research.course import v2 as V2

OUT3 = CR.OUT / "v3"
YEAR_END = 2025            # 読む最後の年（2026 年は封印）
TRAIN_END = 2024
EVAL_YEAR = 2025
PREREG3 = Path(__file__).with_name("prereg_course_v3.json")
EVAL_MARK3 = Path(__file__).with_name("EVALUATED_COURSE_V3.json")
NOTE_LOOKS = ("2025・2026 年は、第 1 版・第 2 版の評価で 2 回見ている。第 3 版は 2025 年を見ながら試行錯誤するので、2025 年の値は信用できない（選んだ期間）。"
              "偏りのない値は、凍結の後に 1 回だけ見る 2026 年だけ")
GOAL_ROI = 1.20
# ---------------------------------------------------------------- 根拠の検査（計画書 §10.2）
HYP = {}
for _t, _h in [("A", "能力"), ("B", "能力"), ("F", "能力"), ("C", "適性"), ("D", "状態"), ("L", "状態"), ("E", "展開"), ("I", "人"), ("J", "人")]:
    HYP[_t] = _h
HYP_BY_ID = {"G01": "状態", "G03": "状態", "G09": "状態", "G02": "能力", "G04": "状態", "G05": "人", "G08": "状態",
             "H01": "展開", "H02": "展開", "H06": "展開", "H03": "能力", "H04": "能力",
             "K01": "適性", "K03": "適性", "K07": "適性", "K08": "適性",
             "X01": "展開", "X02": "状態", "X04": "適性", "X05": "展開", "X07": "能力", "X08": "人", "X09": "状態", "X10": "適性", "X11": "能力",
             "X12": "展開", "X13": "適性", "X14": "状態", "X15": "展開"}
NO_HYP = {"H08", "H09", "X06", "K02", "K04", "K05", "K06", "I02", "B04", "B05"}     # 意味を取りにくい切り口（採らない）
PAIR_OK = {frozenset(["適性", "状態"]), frozenset(["展開"]), frozenset(["能力"]), frozenset(["適性"]), frozenset(["状態"]), frozenset(["人"])}
BLOCKS = [(2016, 2018), (2019, 2021), (2022, 2024)]
DIST_BANDS = [0, 1400, 1800, 2200, 9999]


def hyp_of(tid: str) -> str | None:
    if tid in NO_HYP:
        return None
    return HYP_BY_ID.get(tid) or HYP.get(tid[0])


def hyp_ok(types: list[str]) -> tuple[bool, str]:
    """仮説の型: 型を付けない切り口を含まないこと。2 つの AND は、同じ型どうし・能力 × 他・適性 × 状態だけ。"""
    hs = [hyp_of(t) for t in types]
    if any(h is None for h in hs):
        return False, "型なし"
    if len(hs) == 1:
        return True, hs[0]
    a = frozenset(hs)
    if "能力" in a or a in PAIR_OK:
        return True, "×".join(hs)
    return False, "組の意味なし（" + "×".join(hs) + "）"


def level_orders(d: pd.DataFrame, L: pd.DataFrame) -> dict:
    """順序のある切り口（数値の列から作った区分）の水準の並び（元の数値の中央値の順）。順序の無い切り口は入れない。"""
    from src.research.graded import catalog as CAT
    out = {}
    for t in CAT.TYPES:
        if t.id not in L.columns or not t.cols or t.cols[0] not in d.columns or t.id[0] in ("X",) or len(t.cols) > 1 and t.id not in ("A02",):
            continue
        v = pd.to_numeric(d[t.cols[0]], errors="coerce")
        if v.notna().mean() < 0.5:
            continue
        lv = L[t.id].astype(str)
        med = v.groupby(lv).median().drop(labels=[CAT.UNKNOWN], errors="ignore").dropna()
        if len(med) >= 3:
            out[t.id] = list(med.sort_values().index)
    return out


def direction(lv_unit: pd.Series, y3: np.ndarray, order: list[str], level: str) -> str:
    """選んだ水準が並びの端で、隣へ向かって複勝率が下がる（隣 2 つまで）なら ok。間の水準・飛び抜けは ng。"""
    rates = []
    for x in order:
        m = (lv_unit == x).to_numpy()
        rates.append((x, y3[m].mean() if m.sum() >= 5 else None))
    rates = [(x, r) for x, r in rates if r is not None]
    names = [x for x, _ in rates]
    if level not in names or len(rates) < 3:
        return "不足"
    i = names.index(level)
    if i not in (0, len(rates) - 1):
        return "ng"
    seq = [r for _, r in (rates if i == 0 else rates[::-1])]
    return "ok" if seq[0] >= seq[1] >= seq[2] else "ng"


def binom_p(k: int, n: int, p0: float) -> float:
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, min(max(p0, 1e-9), 1 - 1e-9))) if n else 1.0
MIN_BET_2025 = 300


@dataclass(frozen=True)
class C3:
    # 発見（訓練 2016〜2024 の走で探す）
    depth: int = 2
    n_min: int = 2
    kill_n: int = 20
    kill_depth: int = 1
    min_races: int = 10
    groups: int = 4            # 1 = コースだけ（クラスをまとめる）、3・4 = 第 1 版と同じ
    # 条件の絞り込み（訓練の値で）
    min_n: int = 2             # 買いの条件の訓練の走数の下限
    min_roi_t: float = 0.0     # 買いの条件の訓練の単回収の下限（0 = 見ない）
    top_k: int = 0             # 単位ごとに残す買いの条件の数（0 = すべて）。並べ方は rank_by
    rank_by: str = "n"         # n = 走数、rate = 複勝率、roi = 単回収、lb = 複勝率の信頼区間の下限
    # 根拠の検査（計画書 §10.2。既定はすべて入れる）
    req_hyp: bool = True
    req_dir: bool = True
    req_stab: bool = True
    req_breadth: bool = True
    p_max: float = 0.05
    hyp_ex: str = ""           # 使わない仮説の型（「能力,人」のように。市場に織り込まれやすい型を外す版）
    # 買うレースの側（市場が見落としやすいレースに絞る版。出馬表の時点で分かる値だけ）
    edge_min: float = 0.0      # 条件の訓練の勝ち数 ÷ 市場の勝率の和（確定オッズから）の下限。市場より強い条件だけ（0 = 見ない）
    edge_pmax: float = 0.05
    edge_split: bool = False   # True = 訓練の中を時間で分け、2016〜21 のずれ（edge_min 以上）が 2022〜24 でも 1 を超える条件だけ    # そのずれの偶然の見込み（市場の勝率どおりに勝つとしたときに、その勝ち数以上が出る確率。ポアソンの近似）の上限
    classes: str = ""          # 買う単位の群（「新馬,新馬・未勝利」のように。空 = すべて）
    field_min: int = 0         # 出走表の頭数（field_decl）の下限
    # ◎ の作り
    weight: str = "rate"       # count・rate・roi・lb
    min_score: float = 0.0     # ◎ の点の下限（届かないレースは見送り）
    kill_first: bool = False   # True = 消しに当たる馬を ◎ の候補から外す（消した残りの中から ◎）
    use_kill: bool = True
    # 買う／買わない（発走前のオッズで判定）
    odds_min: float = 0.0
    odds_max: float = 0.0      # 0 = 上限なし
    pop_min: int = 1
    pop_max: int = 0           # 0 = 上限なし
    ev_min: float = 0.0        # 期待値（◎ の訓練の勝率の見込み × 発走前のオッズ）の下限

    def params(self) -> SE.Params:
        return SE.Params(depth=self.depth, n_min=self.n_min, kill_n=self.kill_n, kill_depth=self.kill_depth, min_races=self.min_races,
                         weight="rate", groups=4 if self.groups == 4 else 3)

    def fit_key(self) -> tuple:
        return (self.depth, self.n_min, self.kill_n, self.kill_depth, self.min_races, self.groups)

    def name(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v != getattr(C3(), k)}
        return "基準" if not d else "・".join(f"{k}={v}" for k, v in d.items())


# ---------------------------------------------------------------- 表（2016〜2025 年。2026 年は読まない）
def env3() -> dict:
    e = CR.env(YEAR_END)
    return e | {"FOLKLORE_OUT": str(OUT3 / "master"), "GRADED_OUT": str(OUT3)}


def cmd_frame() -> None:
    OUT3.mkdir(parents=True, exist_ok=True)
    e = env3()
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "src.scripts.data.folklore_build"], env=e, cwd=CR.REPO, check=True)
    subprocess.run([sys.executable, "-m", "src.research.course.v3", "_frame_inner"], env=e, cwd=CR.REPO, check=True)
    print("frame", round(time.time() - t0), "秒", flush=True)


def _frame_inner() -> None:
    from src.research.graded import features as F
    from src.research.graded.run import with_returns
    if max(F.YEARS) > YEAR_END:
        sys.exit("GRADED_YEAR_END が 2025 になっていない（2026 年の表を読んでしまう）")
    d = with_returns(F.build(out=None))
    if (pd.to_numeric(d["year"]) > YEAR_END).any():
        sys.exit("2026 年の行が入っている")
    d, L = CR.frame_and_levels(d)
    d.to_parquet(OUT3 / "frame.parquet", index=False)
    L.to_parquet(OUT3 / "levels.parquet", index=False)
    print("frame", d.shape, "years", int(d["year"].min()), int(d["year"].max()), flush=True)


def load3():
    """単位の表（良・芝ダ・重賞以外）に確定オッズを付け、訓練（〜2024）と 2025 年に分ける。2026 年の行があれば止める。"""
    d = pd.read_parquet(OUT3 / "frame.parquet")
    L = pd.read_parquet(OUT3 / "levels.parquet")
    yr = pd.to_numeric(d["year"]).to_numpy()
    if (yr > YEAR_END).any():
        raise ValueError("2026 年の行が入っている")
    chk = (d["race_id"].astype(str).to_numpy() == L["__race_id"].to_numpy()) & (d["horse_id"].astype(str).to_numpy() == L["__horse_id"].to_numpy())
    if not chk.all():
        raise ValueError("frame と levels の行がずれている")
    keep = ((d["track"].astype(str) == "良") & d["cls"].map(CR.group_of).notna()).to_numpy()
    d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True).drop(columns=["__race_id", "__horse_id"])
    d["group"] = d["cls"].map(CR.group_of)
    d = V2.attach_final_odds(d, range(2016, YEAR_END + 1))
    yr = pd.to_numeric(d["year"]).to_numpy()
    return d, L, yr <= TRAIN_END, yr == EVAL_YEAR


def units_of(d: pd.DataFrame, c: C3) -> np.ndarray:
    if c.groups == 1:
        return (d["course"].astype(str) + "|全クラス").to_numpy()
    return CR.regroup(d, c.groups).to_numpy()


# ---------------------------------------------------------------- 発見・絞り込み・当てはめ
def fit_cells(d: pd.DataFrame, L: pd.DataFrame, c: C3) -> dict:
    """訓練の走で条件を探し（run_config）、条件ごとの訓練の値（走数・勝ち・複勝・単複の回収率）を名前つきで返す。"""
    dd = d.copy()
    if c.groups == 1:
        dd["group"] = "全クラス"
    fit = CR.run_config(dd, L, c.params(), keep_cells=True)
    cells = fit["cells"]
    X, types, code = V2._codes(L, cells)
    Y = SE.y_matrix(dd)
    unit = units_of(dd, c)
    course = dd["course"].astype(str).to_numpy()
    yr = pd.to_numeric(dd["year"]).to_numpy()
    fam = (dd["surf"].astype(str) + "|" + pd.cut(pd.to_numeric(dd["dist"], errors="coerce"), DIST_BANDS, labels=False).astype(str) + "|"
           + dd["group"].astype(str)).to_numpy()
    orders = level_orders(dd, L)
    Ls = {t: L[t].astype(str) for t in orders}
    for u, cu in cells.items():
        idx = np.flatnonzero(course == u.split("|", 1)[0]) if cu.get("layer") == "副" else np.flatnonzero(unit == u)
        Xu, Yu = X[idx], Y[idx]
        p0 = float(Yu[:, 1].mean()) if len(Yu) else 0.0
        fidx = np.flatnonzero((fam == fam[idx[0]]) & (course != course[idx[0]])) if len(idx) else np.array([], int)
        Xf, Yf = X[fidx], Y[fidx]
        pf0 = float(Yf[:, 1].mean()) if len(Yf) else 0.0
        for b in cu["buy"]:
            m = V2._mask_names(Xu, types, code, b)
            s = Yu[m].sum(0)
            b |= {"n_t": int(s[0]), "k_t": int(s[1]), "w_t": int(s[2]), "rt_t": float(s[6] / s[5]) if s[5] else 0.0,
                  "rf_t": float(s[7] / s[5]) if s[5] else 0.0}
            ok, h = hyp_ok(b["types"])
            ev = {"hyp": h, "hyp_ok": ok}
            if ok:
                # 方向（順序のある切り口ごと）
                dirs = [direction(Ls[t].iloc[idx].reset_index(drop=True), Yu[:, 1], orders[t], lv) if t in orders else "順序なし"
                        for t, lv in zip(b["types"], b["levels"])]
                ev["dir"] = "ng" if "ng" in dirs else ("不足" if "不足" in dirs else ("ok" if "ok" in dirs else "順序なし"))
                # 年ごとの安定性（3 年ごとに、単位の平均より複勝率が高い）
                st, nb = [], 0
                for lo, hi in BLOCKS:
                    bm = (yr[idx] >= lo) & (yr[idx] <= hi)
                    mb = m & bm
                    if mb.sum() >= 3 and bm.sum():
                        nb += 1
                        st.append(Yu[mb, 1].mean() > Yu[bm, 1].mean())
                ev["stab"] = "不足" if nb < 2 else ("ok" if all(st) else "ng")
                # 広さ（同じ芝ダ・距離帯・群のほかのコース）
                if len(fidx):
                    mf = V2._mask_names(Xf, types, code, b)
                    ev["breadth_n"] = int(mf.sum())
                    ev["breadth"] = "不足" if mf.sum() < 10 else ("ok" if Yf[mf, 1].mean() > pf0 else "ng")
                else:
                    ev["breadth"] = "不足"
                ev["p_null"] = binom_p(b["k_t"], b["n_t"], p0)
            b["ev"] = ev
        for k in cu["kill"]:
            ok, h = hyp_ok(k["types"])
            k["ev"] = {"hyp": h, "hyp_ok": ok}
    return cells


def ev_pass(b: dict, c: C3) -> str | None:
    """根拠の検査で落ちた理由（通れば None）。"""
    e = b.get("ev") or {}
    if c.req_hyp and not e.get("hyp_ok"):
        return "仮説の型"
    if c.req_dir and e.get("dir") == "ng":
        return "方向"
    if c.req_stab and e.get("stab") != "ok":
        return "年ごとの安定性"
    if c.req_breadth and e.get("breadth") != "ok":
        return "広さ"
    if c.p_max and e.get("p_null", 1.0) > c.p_max:
        return "偶然の見込み"
    return None


def edge_p(b: dict) -> float:
    """市場の勝率どおりに勝つとしたとき、訓練の勝ち数以上が偶然に出る確率（ポアソンの上側）。"""
    from scipy.stats import poisson
    return float(poisson.sf(b["w_t"] - 1, max(b.get("q_t") or 0.0, 1e-9)))


def market_q(d: pd.DataFrame) -> np.ndarray:
    """確定オッズから作る市場の勝率（レースの中で 1/オッズ を合計 1 に直す）。条件の切り口には使わず、条件の強さを市場と比べる物差しにだけ使う。"""
    inv = 1.0 / pd.to_numeric(d["odds_final"], errors="coerce").where(lambda x: x > 0)
    tot = inv.groupby(d["race_id"].astype(str).to_numpy()).transform("sum")
    return (inv / tot).fillna(0.0).to_numpy()


def add_market(cells: dict, d: pd.DataFrame, L: pd.DataFrame, c: C3) -> dict:
    """条件ごとに、訓練の走の市場の勝率の和 q_t を足す（勝ち数 w_t ÷ q_t が 1 を超えれば、市場より勝っている）。"""
    dd = d.copy()
    if c.groups == 1:
        dd["group"] = "全クラス"
    X, types, code = V2._codes(L, cells)
    q = market_q(dd)
    yr = pd.to_numeric(dd["year"]).to_numpy()
    Yw = SE.y_matrix(dd)[:, 2]
    unit = units_of(dd, c)
    course = dd["course"].astype(str).to_numpy()
    for u, cu in cells.items():
        idx = np.flatnonzero(course == u.split("|", 1)[0]) if cu.get("layer") == "副" else np.flatnonzero(unit == u)
        Xu, qu, yu, wu = X[idx], q[idx], yr[idx], Yw[idx]
        early = yu <= 2021
        for b in cu["buy"]:
            m = V2._mask_names(Xu, types, code, b)
            b["q_t"] = float(qu[m].sum())
            b |= {"q_t1": float(qu[m & early].sum()), "w_t1": int(wu[m & early].sum()), "q_t2": float(qu[m & ~early].sum()), "w_t2": int(wu[m & ~early].sum())}
    return cells


def select_cells(cells: dict, c: C3) -> dict:
    """訓練の値で買いの条件を絞り、◎ の点（rate）を版の重みで付ける。"""
    out = {}
    for u, cu in cells.items():
        ex = {x for x in c.hyp_ex.split(",") if x}
        buy = [b for b in cu["buy"] if b["n_t"] >= c.min_n and (not c.min_roi_t or b["rt_t"] >= c.min_roi_t) and ev_pass(b, c) is None
               and not (ex & set(str((b.get("ev") or {}).get("hyp", "")).split("×")))]
        key = {"n": lambda b: b["n_t"], "rate": lambda b: b["k_t"] / max(b["n_t"], 1), "roi": lambda b: b["rt_t"],
               "lb": lambda b: V2.wilson_lb(b["k_t"], b["n_t"])}[c.rank_by]
        if c.top_k:
            buy = sorted(buy, key=lambda b: (-key(b), len(b["types"])))[:c.top_k]
        if c.edge_min and c.edge_split:
            buy = [b for b in buy if b.get("q_t1") and b.get("q_t2") and b["w_t1"] / b["q_t1"] >= c.edge_min and b["w_t2"] / b["q_t2"] > 1.0
                   and edge_p({"w_t": b["w_t1"], "q_t": b["q_t1"]}) <= c.edge_pmax]
        elif c.edge_min:
            buy = [b for b in buy if b.get("q_t") and b["w_t"] / b["q_t"] >= c.edge_min and edge_p(b) <= c.edge_pmax]
        w = {"count": lambda b: 1.0, "rate": lambda b: b["k_t"] / max(b["n_t"], 1), "roi": lambda b: b["rt_t"],
             "lb": lambda b: V2.wilson_lb(b["k_t"], b["n_t"]), "edge": lambda b: b["w_t"] / b["q_t"] if b.get("q_t") else 0.0}[c.weight]
        out[u] = {"layer": cu.get("layer"), "buy": [b | {"rate": float(w(b)), "win_c": b["w_t"] / max(b["n_t"], 1)} for b in buy],
                  "kill": [k for k in cu["kill"] if not c.req_hyp or (k.get("ev") or {}).get("hyp_ok")] if c.use_kill else []}
    return out


def pick(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: C3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """◎ の行・◎ の勝率の見込み・消す行。kill_first なら、消しに当たる馬を ◎ の候補から外す。min_score に届かないレースは見送る。"""
    X, types, code = V2._codes(L, kept)
    race = d["race_id"].astype(str).to_numpy()
    hn = pd.to_numeric(d["hn"], errors="coerce").fillna(99).to_numpy()
    unit = units_of(d, c)
    cls_ok = set(x for x in c.classes.split(",") if x)
    fd = pd.to_numeric(d["field_decl"] if "field_decl" in d.columns else d["field"], errors="coerce").fillna(0).to_numpy()
    H, Pw, K = [], [], np.zeros(len(d), bool)
    for u in sorted(set(unit)):
        cu = kept.get(u)
        if not cu:
            continue
        idx = np.flatnonzero(unit == u)
        Xi = X[idx]
        sc, pw = np.zeros(len(idx)), np.zeros(len(idx))
        for b in cu["buy"]:
            m = V2._mask_names(Xi, types, code, b)
            sc += m * b["rate"]
            pw = np.where(m, np.maximum(pw, b["win_c"]), pw)
        kh = np.zeros(len(idx), bool)
        for k in cu["kill"]:
            kh |= V2._mask_names(Xi, types, code, k)
        ri = race[idx]
        for r in np.unique(ri):
            j = np.flatnonzero(ri == r)
            s = sc[j].copy()
            if c.kill_first:
                s[kh[j]] = 0.0
            hm = -1
            race_ok = (not cls_ok or u.split("|", 1)[1] in cls_ok) and fd[idx[j[0]]] >= c.field_min
            if race_ok and s.max() > max(c.min_score, 0.0):
                hm = int(j[np.lexsort((hn[idx][j], -s))][0])
                H.append(idx[hm])
                Pw.append(pw[hm])
            kk = j[kh[j]]
            kk = kk[kk != hm]
            if len(kk) >= 3:
                K[idx[kk]] = True
    return np.array(H, int), np.array(Pw, float), K


def keep_odds(c: C3, odds: np.ndarray, pop: np.ndarray, pw: np.ndarray) -> np.ndarray:
    o, p = np.nan_to_num(odds, nan=0.0), np.nan_to_num(pop, nan=0.0)
    k = np.ones(len(o), bool)
    if c.odds_min:
        k &= o >= c.odds_min
    if c.odds_max:
        k &= (o > 0) & (o <= c.odds_max)
    if c.pop_min > 1:
        k &= p >= c.pop_min
    if c.pop_max:
        k &= (p > 0) & (p <= c.pop_max)
    if c.ev_min:
        k &= pw * o >= c.ev_min
    return k


def has_odds(c: C3) -> bool:
    return bool(c.odds_min or c.odds_max or c.pop_min > 1 or c.pop_max or c.ev_min)


def trimmed(Y: np.ndarray, rows: np.ndarray, col: int, k: int = 5) -> float | None:
    """払戻の上位 k 件を除いた回収率（大きい当たり数件に頼っていないかを見る）。"""
    if not len(rows):
        return None
    h = Y[rows]
    npay = h[:, 5].sum()
    if not npay:
        return None
    pays = np.sort(h[:, col])[::-1]
    return float((pays[k:].sum()) / npay)


def measure(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: C3, shift: dict, real_pre: dict | None = None) -> dict:
    """ある期間の値: 絞り込みなし・楽観（確定オッズ）・本番の再現（20 通りの平均・5〜95%）。上位 5 件を除いた回収率も。"""
    Y = SE.y_matrix(d)
    race = d["race_id"].astype(str).to_numpy()
    nr = len(set(race))
    H, Pw, K = pick(d, L, kept, c)
    kr = np.flatnonzero(K)

    def m(rows):
        x = SE.metrics(Y, race, rows, kr, nr)
        return x | {"roi_t_trim5": trimmed(Y, rows, 6), "roi_f_trim5": trimmed(Y, rows, 7)}
    out = {"no_filter": m(H)}
    if real_pre:
        # 本物の T-60（発走前のオッズがあり、この表にもあるレースだけ）。オッズの無い ◎ は、絞り込みのある版では買わない
        rr = {k[0] for k in real_pre} & set(race)
        if rr:
            hn_all = pd.to_numeric(d["hn"], errors="coerce").fillna(-1).astype(int).to_numpy()
            mm = np.isin(race, list(rr))
            Hs, pws = H[mm[H]], Pw[mm[H]]
            ro = np.array([real_pre.get((race[h], hn_all[h]), (np.nan, np.nan))[0] for h in Hs])
            rp = np.array([real_pre.get((race[h], hn_all[h]), (np.nan, np.nan))[1] for h in Hs])
            ks = kr[mm[kr]]
            sel = Hs[keep_odds(c, ro, rp, pws)] if has_odds(c) else Hs
            x = SE.metrics(Y, race, sel, ks, len(rr))
            out["real_t60"] = x | {"roi_t_trim5": trimmed(Y, sel, 6), "roi_f_trim5": trimmed(Y, sel, 7)}
            of, pf = d["odds_final"].to_numpy(float), d["pop_final"].to_numpy(float)
            so = Hs[keep_odds(c, of[Hs], pf[Hs], pws)] if has_odds(c) else Hs
            out["real_t60_optimistic"] = SE.metrics(Y, race, so, ks, len(rr))
            out["real_t60_missing_odds"] = int((np.isnan(ro) | (np.nan_to_num(ro) <= 0)).sum())   # T-60 のオッズが無い ◎（絞り込みでは買わない）
    if not has_odds(c):
        out["optimistic"] = out["reproduction"] = out["no_filter"]
        return out
    of, pf = d["odds_final"].to_numpy(float), d["pop_final"].to_numpy(float)
    out["optimistic"] = m(H[keep_odds(c, of[H], pf[H], Pw)])
    draws = []
    for k in range(V2.N_DRAWS):
        pre, pop = V2.simulate_pre(d, shift, k)
        draws.append(m(H[keep_odds(c, pre[H], pop[H], Pw)]))
    rep = {}
    for key in ("bet_races", "coverage", "win", "rentai", "place", "roi_t", "roi_f", "roi_t_trim5", "roi_f_trim5"):
        v = np.array([x[key] for x in draws if x[key] is not None], float)
        rep[key] = float(v.mean()) if len(v) else None
        rep[key + "_p05"] = float(np.percentile(v, 5)) if len(v) else None
        rep[key + "_p95"] = float(np.percentile(v, 95)) if len(v) else None
    out["reproduction"] = rep | {k: out["no_filter"][k] for k in ("races", "killed", "kill_races", "kill_coverage", "kill_per_race", "kill_precision")}
    return out


def ev_summary(cells: dict, c: C3) -> dict:
    """根拠の検査を通った条件の数（単位ごと・型ごと）と、落ちた理由の内訳。"""
    from collections import Counter
    why, by_h, per_unit = Counter(), Counter(), {}
    for u, cu in cells.items():
        k = 0
        for b in cu["buy"]:
            r = ev_pass(b, c) if b["n_t"] >= c.min_n else "走数"
            why[r or "通った"] += 1
            if r is None:
                k += 1
                by_h[(b.get("ev") or {}).get("hyp")] += 1
        per_unit[u] = k
    vals = np.array(list(per_unit.values()))
    return {"reasons": dict(why), "by_hyp": dict(by_h.most_common()), "units": len(vals), "units_with_any": int((vals > 0).sum()),
            "per_unit_median": float(np.median(vals)) if len(vals) else 0, "per_unit_max": int(vals.max()) if len(vals) else 0}


def goal_met(r: dict) -> bool:
    a = r["eval2025"]["reproduction"]
    return (a.get("bet_races") or 0) >= MIN_BET_2025 and max(a.get("roi_t") or 0, a.get("roi_f") or 0) > GOAL_ROI


# ---------------------------------------------------------------- サイクル（1 本ずつ流し、回顧と判断を書く）
def _log_path() -> Path:
    return OUT3 / "course_v3_cycles.json"


def _load_log() -> dict:
    p = _log_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"note": "全レース版の第 3 版（訓練 2016〜2024・試行錯誤の評価 2025・封印 2026）", "note_looks": NOTE_LOOKS,
            "goal": f"2025 年の本番の再現で、単勝か複勝の回収率 {GOAL_ROI:.0%} 超（{MIN_BET_2025} レース以上、払戻の上位 5 件を除いた値も見る）",
            "shift_note": "本番の再現のずれの分布は第 2 版の shift.json（2026 年の T-60 と確定のオッズの組だけ。着順・払戻は見ていない）",
            "started_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "current": asdict(C3()), "cycles": []}


def _save_log(log: dict) -> None:
    V2._write_json(_log_path(), log)


_CACHE: dict = {}


def run_cycle(c: C3, d, L, tr, ev, shift) -> tuple[dict, dict]:
    t0 = time.time()
    key = c.fit_key()
    if key not in _CACHE:
        cache_p = OUT3 / f"fit_{hashlib.sha1(repr(key).encode()).hexdigest()[:12]}.json"
        if cache_p.exists():
            _CACHE[key] = json.loads(cache_p.read_text(encoding="utf-8"))
        else:
            _CACHE[key] = fit_cells(d[tr].reset_index(drop=True), L[tr].reset_index(drop=True), c)
            cache_p.write_text(json.dumps(_CACHE[key], ensure_ascii=False), encoding="utf-8")
    if c.edge_min or c.weight == "edge":
        first = next((b for cu in _CACHE[key].values() for b in cu["buy"]), None)
        if first is not None and "q_t2" not in first:
            add_market(_CACHE[key], d[tr].reset_index(drop=True), L[tr].reset_index(drop=True), c)
            (OUT3 / f"fit_{hashlib.sha1(repr(key).encode()).hexdigest()[:12]}.json").write_text(json.dumps(_CACHE[key], ensure_ascii=False), encoding="utf-8")
    kept = select_cells(_CACHE[key], c)
    dt, Lt = d[tr].reset_index(drop=True), L[tr].reset_index(drop=True)
    de, Le = d[ev].reset_index(drop=True), L[ev].reset_index(drop=True)
    r = {"params": asdict(c) | {"name": c.name()}, "train": measure(dt, Lt, kept, c, shift), "eval2025": measure(de, Le, kept, c, shift),
         "n_buy": sum(len(v["buy"]) for v in kept.values()), "ev_summary": ev_summary(_CACHE[key], c), "seconds": round(time.time() - t0, 1), "mem": CR.mem_now()}
    return r, kept


def cmd_cycle(hyp: str, delta: dict, base: int | None) -> None:
    if PREREG3.exists() or EVAL_MARK3.exists():
        sys.exit("凍結・評価の後にサイクルを足さない")
    log = _load_log()
    if len(log["cycles"]) >= 20:
        sys.exit("20 サイクルまで")
    if log["cycles"] and not log["cycles"][-1].get("decision"):
        sys.exit("前のサイクルの回顧・判断がまだ（review で書く）")
    cur = log["current"] if base is None else next(x for x in log["cycles"] if x["cycle"] == base)["params"]
    c = replace(C3(**{k: v for k, v in cur.items() if k != "name"}), **delta)
    CR.wait_resources()
    shift = json.loads((CR.OUT / "v2" / "shift.json").read_text(encoding="utf-8"))
    d, L, tr, ev = load3()
    r, kept = run_cycle(c, d, L, tr, ev, shift)
    n = len(log["cycles"]) + 1
    log["cycles"].append({"cycle": n, "hyp": hyp, "delta": delta, "base": base, "goal_met": goal_met(r)} | r)
    (OUT3 / f"kept_{n}.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    _save_log(log)
    a, t = r["eval2025"]["reproduction"], r["train"]["reproduction"]
    print(json.dumps({"cycle": n, "name": c.name(), "2025": {k: a.get(k) for k in ("bet_races", "coverage", "win", "rentai", "place", "roi_t", "roi_f",
                                                                                 "roi_t_trim5", "roi_f_trim5", "roi_t_p05", "roi_t_p95", "kill_precision", "kill_coverage")},
                      "2025_opt": {k: r["eval2025"]["optimistic"].get(k) for k in ("bet_races", "roi_t", "roi_f")},
                      "train": {k: t.get(k) for k in ("bet_races", "win", "place", "roi_t", "roi_f")}, "goal_met": goal_met(r), "seconds": r["seconds"],
                      "mem": r["mem"]}, ensure_ascii=False), flush=True)


def cmd_evidence() -> None:
    """サイクルの前に: 基準の版で、根拠の検査を通った条件の数（単位ごと・型ごと）と、落ちた理由の内訳を出す（2025 年は見ない）。"""
    CR.wait_resources()
    d, L, tr, ev = load3()
    c = C3()
    key = c.fit_key()
    cache_p = OUT3 / f"fit_{hashlib.sha1(repr(key).encode()).hexdigest()[:12]}.json"
    if cache_p.exists():
        cells = json.loads(cache_p.read_text(encoding="utf-8"))
    else:
        cells = fit_cells(d[tr].reset_index(drop=True), L[tr].reset_index(drop=True), c)
        cache_p.write_text(json.dumps(cells, ensure_ascii=False), encoding="utf-8")
    out = {"base": ev_summary(cells, c)}
    for name, cc in [("偶然の見込み 10%", replace(c, p_max=0.10)), ("広さを外す", replace(c, req_breadth=False)),
                     ("安定性を外す", replace(c, req_stab=False)), ("広さ・安定性を外す", replace(c, req_breadth=False, req_stab=False))]:
        out[name] = ev_summary(cells, cc)
    (OUT3 / "evidence_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


def cmd_freeze(n: int) -> None:
    """凍結: 選んだサイクル n の版と条件（kept_<n>.json）・ずれの分布の sha を prereg_course_v3.json に書く（commit する）。"""
    if EVAL_MARK3.exists() or PREREG3.exists():
        sys.exit("凍結・評価は済んでいる")
    log = _load_log()
    c = next((x for x in log["cycles"] if x["cycle"] == n), None)
    if c is None:
        sys.exit(f"サイクル {n} が無い（--n で選ぶ）")
    if c.get("decision") != "採用" or log.get("adopted_cycle") != n:
        sys.exit(f"サイクル {n} は今の採用の版ではない（adopted_cycle = {log.get('adopted_cycle')}）")
    kp = OUT3 / f"kept_{n}.json"
    body = {"frozen_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cycle": n, "params": c["params"], "note_looks": NOTE_LOOKS,
            "note": "訓練 2016〜2024・試行錯誤 2025 年で選んだ。2026 年の着順・払戻はこの版では見ていない（第 1・2 版の評価で 2 回見た。この評価は 3 回目）。"
                    "ただし本番の再現のずれの分布（shift.json）は、2026 年の T-60 と確定オッズの組から作っている（着順・払戻は使っていない）",
            "n_draws": V2.N_DRAWS, "seed": V2.SEED,
            "eval2025": c["eval2025"], "train": c["train"], "cells_file": kp.name, "sha256_cells": CR._sha_file(kp),
            "shift_file": "v2/shift.json", "sha256_shift": CR._sha_file(CR.OUT / "v2" / "shift.json"),
            "cycles": len(log["cycles"]), "goal_met_2025": c.get("goal_met")}
    PREREG3.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("frozen", PREREG3)


def cmd_evaluate() -> None:
    """凍結の後に 1 回だけ: 2026 年（封印していた期間。2025・2026 年は 3 回目）。凍結が commit されていて sha が一致するときだけ。印は計算の前に立てる。"""
    import secrets
    if EVAL_MARK3.exists():
        sys.exit("評価は済んでいる（1 回だけ）")
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"] or CR._sha_file(CR.OUT / P["shift_file"]) != P["sha256_shift"]:
        sys.exit("条件の表・ずれの分布が凍結の sha と違う")
    r = subprocess.run(["git", "-C", str(CR.REPO), "ls-files", "--error-unmatch", str(PREREG3)], capture_output=True)
    if r.returncode != 0 or subprocess.run(["git", "-C", str(CR.REPO), "diff", "--quiet", "HEAD", "--", str(PREREG3)]).returncode != 0:
        sys.exit("prereg_course_v3.json が commit されていない")
    master = CR.OUT / "eval" / "master" / "master.parquet"
    need = [master, V2.PRE_ODDS, CR.REPO / "data/page_reference/tables/2026/race_result_flat.parquet",
            CR.REPO / "data/page_reference/tables/2026/race_result_race.parquet"]
    miss = [str(x) for x in need if not x.exists()]
    if miss:
        sys.exit(f"評価に要る表が無い（印を立てる前に止める。データのつながった作業場所で流す）: {miss}")
    tok = secrets.token_hex(16)
    EVAL_MARK3.write_text(json.dumps({"started_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "prereg_sha_cells": P["sha256_cells"],
                                      "token_sha": hashlib.sha256(tok.encode()).hexdigest()}), encoding="utf-8")
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval"), "COURSE_EVAL_TOKEN": tok}
    subprocess.run([sys.executable, "-m", "src.research.course.v3", "_eval_inner"], env=e, cwd=CR.REPO, check=True)


def _frame_2026() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2026 年の単位の表（良・芝ダ・重賞以外）と水準。CR.env(2026) の環境で流す子の処理の中だけで呼ぶ。"""
    from src.research.graded import features as F
    from src.research.graded.run import with_returns
    d, L = CR.frame_and_levels(with_returns(F.build(out=None)))
    keep = ((pd.to_numeric(d["year"]) == 2026) & (d["track"].astype(str) == "良") & d["cls"].map(CR.group_of).notna()).to_numpy()
    d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True).drop(columns=["__race_id", "__horse_id"])
    d["group"] = d["cls"].map(CR.group_of)
    return V2.attach_final_odds(d, (2026,)), L


def _eval_inner() -> None:
    tok = os.environ.get("COURSE_EVAL_TOKEN", "")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if not tok or mark.get("token_sha") != hashlib.sha256(tok.encode()).hexdigest() or "evaluated_at" in mark:
        sys.exit("評価は cmd_evaluate から 1 回だけ流す")
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if (CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"] or mark.get("prereg_sha_cells") != P["sha256_cells"]
            or CR._sha_file(CR.OUT / P["shift_file"]) != P["sha256_shift"]):
        sys.exit("条件の表・ずれの分布が凍結の sha と違う")
    d, L = _frame_2026()
    c = C3(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    shift = json.loads((CR.OUT / P["shift_file"]).read_text(encoding="utf-8"))
    pre = pd.read_parquet(V2.PRE_ODDS)
    pre = pre[pre["phase"] == "t60"]
    real = {(str(r), int(h)): (float(o), float(pp)) for r, h, o, pp in pre[["race_id", "horse_number", "win_odds", "popularity"]].itertuples(index=False)}
    res = measure(d, L, kept, c, shift, real_pre=real)
    _, _, code = V2._codes(L, kept)
    res["unmatched_cells"] = sum(any(lv not in code[t] for t, lv in zip(b["types"], b["levels"])) for cu in kept.values()
                                 for kind in ("buy", "kill") for b in cu[kind])
    out = {"evaluated_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "round": 3, "year": 2026,
           "note": "2026 年は第 3 版では封印していた（サイクルの間は読んでいない）。2025・2026 年を見るのは 3 回目",
           "all": res, "params": P["params"], "eval2025": P["eval2025"], "races": int(d["race_id"].nunique())}
    (OUT3 / "eval_course_v3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    EVAL_MARK3.write_text(json.dumps(mark | {"evaluated_at": out["evaluated_at"]}), encoding="utf-8")
    a = res["reproduction"]
    print(f"評価 2026（3 回目）: 再現 単回収 {a.get('roi_t')}（{a.get('roi_t_p05')}〜{a.get('roi_t_p95')}） 複 {a.get('roi_f')} 買う {a.get('bet_races')}"
          f" 楽観 {res['optimistic'].get('roi_t')} 本物の T-60 {(res.get('real_t60') or {}).get('roi_t')}", flush=True)


# ---------------------------------------------------------------- 単位ごとの値（ルールの一覧の HTML 用。評価ではない）
PER_UNIT = OUT3 / "per_unit_v3.json"
REP_KEYS = ("bet_races", "coverage", "win", "rentai", "place", "roi_t", "roi_f", "roi_t_trim5", "roi_f_trim5")


def per_unit(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: C3, shift: dict) -> dict:
    """凍結した規則を期間の表 d に 1 回だけ当てはめ（pick）、◎・消しの行を単位ごとに分けて数える。本番の再現のずれの 20 通りは
    d 全体で作ってから単位ごとに切る（単位ごとに作り直すと乱数の並びが変わり、単位の値の合計が全体の値と合わなくなる）。
    単位の本番の再現は 20 通りを足し合わせた比（pooled_draws）、全体は measure と同じ回ごとの平均（記録との照合用）。
    戻り値: {"units": {単位: {no_filter, optimistic, reproduction}}, "total": 同じ形（measure と一致するはず）}。"""
    Y = SE.y_matrix(d)
    race = d["race_id"].astype(str).to_numpy()
    unit = units_of(d, c)
    H, Pw, K = pick(d, L, kept, c)
    kr = np.flatnonzero(K)
    uh, uk = unit[H], unit[kr]
    nr = pd.Series(race).groupby(unit).nunique().to_dict()
    groups = {u: (uh == u, uk == u, int(nr[u])) for u in sorted(nr)}
    groups["*"] = (np.ones(len(H), bool), np.ones(len(kr), bool), len(set(race)))

    def pooled_draws(keeps: list[np.ndarray]) -> dict:
        """単位ごとの本番の再現: 20 通りを足し合わせた比（勝ち数の和 ÷ ◎ の数の和、払戻の和 ÷ 払戻の解けた ◎ の和）。
        買わなかった回も分母に入る（回ごとの率の平均だと、まれにしか買わない単位で率が上振れする）。p05・p95 は回ごとの値の幅。"""
        out = {}
        for u, (hm, km, n) in groups.items():
            rows = [H[hm & kp] for kp in keeps]
            allr = np.concatenate(rows) if rows else np.array([], int)
            h = Y[allr]
            nb, npay = len(allr), h[:, 5].sum()
            r = {"bet_races": nb / len(keeps), "coverage": nb / len(keeps) / n if n else None,
                 "win": float(h[:, 2].mean()) if nb else None, "rentai": float((h[:, 2] + h[:, 3]).mean()) if nb else None,
                 "place": float(h[:, 1].mean()) if nb else None,
                 "roi_t": float(h[:, 6].sum() / npay) if npay else None, "roi_f": float(h[:, 7].sum() / npay) if npay else None,
                 "roi_t_trim5": trimmed(Y, allr, 6, 5 * len(keeps)), "roi_f_trim5": trimmed(Y, allr, 7, 5 * len(keeps))}
            for key in ("roi_t", "roi_f"):
                v = np.array([x for x in (SE.metrics(Y, race, rr, kr[km], n)[key] for rr in rows) if x is not None], float)
                r[key + "_p05"] = float(np.percentile(v, 5)) if len(v) else None
                r[key + "_p95"] = float(np.percentile(v, 95)) if len(v) else None
            out[u] = r | {k: nf[u][k] for k in ("races", "killed", "kill_races", "kill_coverage", "kill_per_race", "kill_precision")}
        return out

    def m(keep: np.ndarray) -> dict:
        out = {}
        for u, (hm, km, n) in groups.items():
            rows = H[hm & keep]
            out[u] = SE.metrics(Y, race, rows, kr[km], n) | {"roi_t_trim5": trimmed(Y, rows, 6), "roi_f_trim5": trimmed(Y, rows, 7)}
        return out
    allk = np.ones(len(H), bool)
    nf = m(allk)
    if has_odds(c):
        of, pf = d["odds_final"].to_numpy(float), d["pop_final"].to_numpy(float)
        op = m(keep_odds(c, of[H], pf[H], Pw))
        draws, keeps = [], []
        for k in range(V2.N_DRAWS):
            pre, pop = V2.simulate_pre(d, shift, k)
            keeps.append(keep_odds(c, pre[H], pop[H], Pw))
            draws.append(m(keeps[-1]))
        rp = pooled_draws(keeps)
        for u in ("*",):              # 全体は measure と同じ定義（回ごとの値の平均）にして、記録の値と照合する
            r = {}
            for key in REP_KEYS:
                v = np.array([x[u][key] for x in draws if x[u][key] is not None], float)
                r[key] = float(v.mean()) if len(v) else None
                r[key + "_p05"] = float(np.percentile(v, 5)) if len(v) else None
                r[key + "_p95"] = float(np.percentile(v, 95)) if len(v) else None
            rp[u] = r | {k: nf[u][k] for k in ("races", "killed", "kill_races", "kill_coverage", "kill_per_race", "kill_precision")}
    else:
        op = rp = nf
    res = {u: {"no_filter": nf[u], "optimistic": op[u], "reproduction": rp[u]} for u in groups}
    return {"units": {u: v for u, v in res.items() if u != "*"}, "total": res["*"]}


def check_total(got: dict, want: dict, keys=("bet_races", "roi_t", "roi_f", "win", "kill_precision")) -> list[str]:
    """数え直した全体の値と、記録に残る値（サイクルの記録・凍結・評価）の食い違い。空なら一致。"""
    bad = []
    for kind in ("no_filter", "optimistic", "reproduction"):
        if kind not in want:
            continue
        for k in keys:
            a, b = got[kind].get(k), want[kind].get(k)
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                bad.append(f"{kind}.{k}: {a} != {b}")
    return bad


def cmd_per_unit() -> None:
    """ルールの一覧の HTML 用に、凍結した版（prereg_course_v3.json）を訓練 2016〜2024・2025 年・2026 年に当て直し、単位ごとの値を
    <COURSE_OUT>/v3/per_unit_v3.json に書く。規則は変えない（sha を照合）。2026 年は 1 回だけの評価が済んだ後にだけ読む（封印を破らない）。
    評価の印・評価の結果のファイルには触らない。全体の値が記録の値と違えば止める（当て直しが凍結の規則と違う）。"""
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"] or CR._sha_file(CR.OUT / P["shift_file"]) != P["sha256_shift"]:
        sys.exit("条件の表・ずれの分布が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark or not (OUT3 / "eval_course_v3.json").exists():
        sys.exit("2026 年の 1 回だけの評価が済んでいない（封印中の 2026 年は読まない）")
    CR.wait_resources()
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval")}
    subprocess.run([sys.executable, "-m", "src.research.course.v3", "_per_unit_inner"], env=e, cwd=CR.REPO, check=True)


def _per_unit_inner() -> None:
    from src.research.graded import catalog as CAT
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"] or CR._sha_file(CR.OUT / P["shift_file"]) != P["sha256_shift"]:
        sys.exit("条件の表・ずれの分布が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark or not (OUT3 / "eval_course_v3.json").exists():
        sys.exit("2026 年の 1 回だけの評価が済んでいない（封印中の 2026 年は読まない）")   # 子を直接流しても封印を破らない
    c = C3(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    shift = json.loads((CR.OUT / P["shift_file"]).read_text(encoding="utf-8"))
    log = _load_log()
    cyc = next(x for x in log["cycles"] if x["cycle"] == P["cycle"])
    E = json.loads((OUT3 / "eval_course_v3.json").read_text(encoding="utf-8"))
    out, bad = {}, []
    d, L, tr, ev = load3()
    for key, sel, want in (("train", tr, cyc["train"]), ("2025", ev, P["eval2025"])):
        t0 = time.time()
        out[key] = per_unit(d[sel].reset_index(drop=True), L[sel].reset_index(drop=True), kept, c, shift)
        bad += [f"{key} {x}" for x in check_total(out[key]["total"], want)]
        print(key, f"{time.time() - t0:.0f}s", CR.mem_now(), flush=True)
    del d, L
    d6, L6 = _frame_2026()
    out["2026"] = per_unit(d6, L6, kept, c, shift)
    bad += [f"2026 {x}" for x in check_total(out["2026"]["total"], E["all"])]
    if bad:
        sys.exit("当て直した全体の値が記録と違う: " + "; ".join(bad[:10]))
    names = {t.id: t.name for t in CAT.TYPES}
    res = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cycle": P["cycle"], "params": P["params"],
           "sha256_cells": P["sha256_cells"], "periods": {"train": [2016, TRAIN_END], "2025": [EVAL_YEAR, EVAL_YEAR], "2026": [2026, 2026]},
           "note": "凍結した第 3 版の規則を当て直して単位ごとに数えた値（評価ではない）。全体の値はサイクルの記録・凍結・2026 年の評価と一致を確かめた",
           "type_names": names, "values": out}
    PER_UNIT.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    print("per_unit", PER_UNIT, {k: len(v["units"]) for k, v in out.items()}, flush=True)


# ---------------------------------------------------------------- 条件ごとの成績（めぐ分析の表示用。評価ではない）
RULE_STATS = OUT3 / "rule_stats_v3.json"
RULE_STATS_REPO = CR.REPO / "data/meta/course/rule_stats_v3.json"   # めぐ分析（src/research/race_analysis/course_rules.py）が読む写し
RULE_TOP = 5            # 単位ごとに出す買い・消しの条件の数（買いは市場より強い度合いの高い順、消しは訓練の走数の多い順）
POP_CAP = 13            # 人気の基準線の区切り（13 番人気以下はまとめる）


def _rstat(Y: np.ndarray) -> dict:
    """y_matrix の行の集まりから [1-2-3-着外]・単複の回収率（払戻の解けた走で割る）・件数。年を合わせられるよう、払戻の和（npay・tan・fuku）も持つ。"""
    s = Y.sum(0) if len(Y) else np.zeros(Y.shape[1] if Y.ndim == 2 else 10)
    n, k = int(s[0]), int(s[1])
    return {"n": n, "rec": [int(s[2]), int(s[3]), int(s[4]), n - k], "place": float(k / n) if n else None,
            "roi_t": float(s[6] / s[5]) if s[5] else None, "roi_f": float(s[7] / s[5]) if s[5] else None,
            "npay": int(s[5]), "tan": float(s[6]), "fuku": float(s[7])}


def _pop_tab(Y: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """人気ごとの [走, 3 着内, 払戻の解けた走, 単の払戻, 複の払戻]（人気 × 5）。"""
    return np.stack([np.bincount(pop, weights=Y[:, j], minlength=POP_CAP + 1) for j in (0, 1, 5, 6, 7)], axis=1)


def _pop_base(m: np.ndarray, pu: np.ndarray, Yu: np.ndarray, pool: np.ndarray) -> dict | None:
    """当たった馬（m）と同じ人気の馬の値。人気ごとの率は、同じ芝ダ × クラスの群の全単位を合わせた表（pool）から、当たった馬を除いて取り、
    当たった馬の人気の構成で重みづけする（単位 × 年の小さな表ではぶれすぎるため。別のエージェントのレビュー）。"""
    if not m.any():
        return None
    t = pool - _pop_tab(Yu[m], pu[m])
    n_p, k_p, np_p, rt_p, rf_p = (t[:, j] for j in range(5))
    w = np.bincount(pu[m], minlength=POP_CAP + 1).astype(float)
    ok, okp = n_p > 0, np_p > 0
    pl = (w[ok] * k_p[ok] / n_p[ok]).sum() / w[ok].sum() if w[ok].sum() else None
    rt = (w[okp] * rt_p[okp] / np_p[okp]).sum() / w[okp].sum() if w[okp].sum() else None
    rf = (w[okp] * rf_p[okp] / np_p[okp]).sum() / w[okp].sum() if w[okp].sum() else None
    f = lambda v: None if v is None else float(v)
    return {"place": f(pl), "roi_t": f(rt), "roi_f": f(rf)}


def ordered_conds(cu: dict) -> tuple[list[dict], list[dict]]:
    """表示する条件の並び（rule_stats_period と rule_list で同じにする）。買いは市場より強い度合いの高い順、消しは訓練の走数の多い順。"""
    buy = sorted(cu["buy"], key=lambda b: -(b.get("rate") or 0))
    kill = sorted(cu["kill"], key=lambda k: -(k.get("n") or 0))
    return buy, kill


def rule_stats_period(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: C3) -> dict:
    """期間の表 d に、凍結した条件（kept）を単位の走ごとに当てて数える（全条件。表示の絞り込みは訓練の値で後から行う）。
    基準線（base_same_pop）は _pop_base を見る。"""
    X, types, code = V2._codes(L, kept)
    Y = SE.y_matrix(d)
    unit = units_of(d, c)
    pop = np.clip(np.nan_to_num(pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float), nan=0), 0, POP_CAP).astype(int)
    fam = (d["surf"].astype(str) + "|" + pd.Series(unit).str.split("|", n=1).str[1].fillna("")).to_numpy()
    pools = {f: _pop_tab(Y[fam == f], pop[fam == f]) for f in pd.unique(fam)}
    out = {}
    for u, cu in kept.items():
        idx = np.flatnonzero(unit == u)
        if not len(idx):
            continue
        Xu, Yu, pu = X[idx], Y[idx], pop[idx]
        pool = pools[fam[idx[0]]]
        buy, kill = ordered_conds(cu)
        res = {}
        for kind, conds in (("buy", buy), ("kill", kill)):
            res[kind] = []
            for cond in conds:
                m = V2._mask_names(Xu, types, code, cond)
                res[kind].append(_rstat(Yu[m]) | {"base_same_pop": _pop_base(m, pu, Yu, pool)})
        out[u] = {"unit": _rstat(Yu)} | res
    return out


def pick_shown(train: dict, kept: dict, top: int = RULE_TOP) -> dict:
    """単位ごとに表示する条件の番号（ordered_conds の並びの中の位置）。訓練の期間にこの単位で 1 件も当たらない条件（コース全体＝副の層を写した条件が
    新馬の単位に当たらない、など）は出さない。上位 top 個まで。"""
    out = {}
    for u in kept:
        t = train.get(u) or {}
        hit = {k: [i for i, x in enumerate(t.get(k) or []) if x["n"] > 0] for k in ("buy", "kill")}
        out[u] = {k: v[:top] for k, v in hit.items()} | {"n_hit": {k: len(v) for k, v in hit.items()}}
    return out


def rule_list(kept: dict, names: dict, shown: dict) -> dict:
    """単位ごとの表示する条件の文（shown の番号）と、根拠の検査の結果。"""
    def txt(b):
        return " かつ ".join(f"「{names.get(t, t)}」が {lv}" for t, lv in zip(b["types"], b["levels"]))
    out = {}
    for u, cu in kept.items():
        buy, kill = ordered_conds(cu)
        sh = shown.get(u) or {"buy": [], "kill": []}
        out[u] = {"layer": cu.get("layer"), "n_buy": len(buy), "n_kill": len(kill),
                  "n_buy_hit": (sh.get("n_hit") or {}).get("buy"), "n_kill_hit": (sh.get("n_hit") or {}).get("kill"),
                  "buy": [{"i": i, "text": txt(buy[i]), "hyp": (buy[i].get("ev") or {}).get("hyp"), "edge": buy[i].get("rate"),
                           "p_null": (buy[i].get("ev") or {}).get("p_null")} for i in sh["buy"]],
                  "kill": [{"i": i, "text": txt(kill[i]), "hyp": (kill[i].get("ev") or {}).get("hyp")} for i in sh["kill"]]}
    return out


def _pick_values(v: dict, shown: dict) -> dict:
    """期間ごとの値を、表示する条件だけに絞る（並びは rule_list と同じ）。"""
    out = {}
    for u, x in v.items():
        sh = shown.get(u) or {"buy": [], "kill": []}
        out[u] = {"unit": x["unit"]} | {k: [x[k][i] for i in sh[k]] for k in ("buy", "kill")}
    return out


def cmd_rule_stats() -> None:
    """凍結した第 3 版の条件ごとに、訓練 2016〜2024・2025・2026 年の成績を数えて <COURSE_OUT>/v3/rule_stats_v3.json に書く（めぐ分析の表示用）。
    規則は変えない（sha を照合）。2026 年は 1 回だけの評価が済んだ後にだけ読む。見た回数の台帳には eval_only（表示用）として記録する。"""
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark:
        sys.exit("2026 年の 1 回だけの評価が済んでいない（封印中の 2026 年は読まない）")
    CR.wait_resources()
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval")}
    subprocess.run([sys.executable, "-m", "src.research.course.v3", "_rule_stats_inner"], env=e, cwd=CR.REPO, check=True)


def _rule_stats_inner() -> None:
    from src.ops import data_ledger as DL
    from src.research.graded import catalog as CAT
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark:
        sys.exit("2026 年の 1 回だけの評価が済んでいない（封印中の 2026 年は読まない）")
    c = C3(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    names = {t.id: t.name for t in CAT.TYPES}
    with DL.guard(tool="src/research/course/v3.py:rule_stats", family="course", version=P["sha256_cells"][:12],
                  period=("2025-01-01", "2026-11-20"), purpose="eval_only",
                  reason="めぐ分析の表示用に、凍結した第 3 版の条件ごとの成績を数える（条件は選び直さない・採否に使わない）",
                  frozen_paths=[PREREG3, OUT3 / P["cells_file"]]) as tok:
        out = {}
        d, L, tr, ev = load3()
        out["train"] = rule_stats_period(d[tr].reset_index(drop=True), L[tr].reset_index(drop=True), kept, c)
        out["2025"] = rule_stats_period(d[ev].reset_index(drop=True), L[ev].reset_index(drop=True), kept, c)
        tok.check_dates(d.loc[ev, "date"]) if "date" in d.columns else None
        del d, L
        d6, L6 = _frame_2026()
        tok.check_dates(d6["date"]) if "date" in d6.columns else None
        out["2026"] = rule_stats_period(d6, L6, kept, c)
        data_until = str(pd.to_datetime(d6["date"]).max())[:10] if "date" in d6.columns and len(d6) else None
        shown = pick_shown(out["train"], kept)
        res = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cycle": P["cycle"], "sha256_cells": P["sha256_cells"],
               "periods": {"train": [2016, TRAIN_END], "2025": [EVAL_YEAR, EVAL_YEAR], "2026": [2026, 2026]}, "data_until": data_until,
               "rule_top": RULE_TOP, "rules": rule_list(kept, names, shown), "values": {k: _pick_values(v, shown) for k, v in out.items()},
               "ledger": tok.summary(),
               "note": "凍結した第 3 版の条件を当て直して数えた値（評価ではない・表示用）。2025・2026 年は何度か見た期間"}
        RULE_STATS.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
        RULE_STATS_REPO.parent.mkdir(parents=True, exist_ok=True)
        RULE_STATS_REPO.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")   # めぐ分析が読む写し（commit する）
        tok.finish(result_path=RULE_STATS)
    print("rule_stats", RULE_STATS, {k: len(v) for k, v in out.items()}, flush=True)


def cmd_review(n: int, text: str, decision: str, event: str = "", basis: str = "") -> None:
    """回顧と判断。事象（何が起きているか）と根拠（なぜそうなるか）も書く（統括役の方針、2026-11-17）。"""
    log = _load_log()
    c = next(x for x in log["cycles"] if x["cycle"] == n)
    c["review"], c["decision"] = text, decision
    if event:
        c["event"] = event
    if basis:
        c["basis"] = basis
    if decision == "採用":
        log["current"] = {k: v for k, v in c["params"].items() if k != "name"}
        log["adopted_cycle"] = n
    _save_log(log)
    print("review", n, decision)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frame", "_frame_inner", "evidence", "cycle", "review", "freeze", "evaluate", "_eval_inner",
                                       "per_unit", "_per_unit_inner", "rule_stats", "_rule_stats_inner"])
    ap.add_argument("--hyp", default="")
    ap.add_argument("--delta", default="{}")
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--text", default="")
    ap.add_argument("--decision", default="")
    ap.add_argument("--event", default="")
    ap.add_argument("--basis", default="")
    a = ap.parse_args()
    if a.cmd == "frame":
        cmd_frame()
    elif a.cmd == "_frame_inner":
        _frame_inner()
    elif a.cmd == "freeze":
        cmd_freeze(a.n)
    elif a.cmd == "evaluate":
        cmd_evaluate()
    elif a.cmd == "_eval_inner":
        _eval_inner()
    elif a.cmd == "evidence":
        cmd_evidence()
    elif a.cmd == "per_unit":
        cmd_per_unit()
    elif a.cmd == "rule_stats":
        cmd_rule_stats()
    elif a.cmd == "_rule_stats_inner":
        _rule_stats_inner()
    elif a.cmd == "_per_unit_inner":
        _per_unit_inner()
    elif a.cmd == "cycle":
        cmd_cycle(a.hyp, json.loads(a.delta), a.base)
    else:
        cmd_review(a.n, a.text, a.decision, a.event, a.basis)


if __name__ == "__main__":
    main()
