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


def measure(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: C3, shift: dict) -> dict:
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
    ap.add_argument("cmd", choices=["frame", "_frame_inner", "evidence", "cycle", "review"])
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
    elif a.cmd == "evidence":
        cmd_evidence()
    elif a.cmd == "cycle":
        cmd_cycle(a.hyp, json.loads(a.delta), a.base)
    else:
        cmd_review(a.n, a.text, a.decision, a.event, a.basis)


if __name__ == "__main__":
    main()
