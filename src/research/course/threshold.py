"""閾値のルールの版（重賞・通常レース）。計画書: docs/ideas/course_class_handicapper.md §12。

粗い閾値（前走の上がり 3 位以内・着差 0.5 秒以内など）を 1〜3 個 AND でつなぎ、全部を満たす馬だけを残す（点の足し合わせはしない）。
- 期間: 探す 2016〜2021 年、採否と確かめ 2022〜2024 年（最近の持続の検査に使う）。2025 年以降は凍結の後に 1 回だけ（search では読まない）。
- 買う時点の層（LAYERS: 全馬・1 番人気・2 番人気以内・単勝 2.5 倍以下・2.0 倍以下）ごとに探す。人気は探す切り口には使わず、買う時点の最後の絞り込みの層として使う。
  探す・採否は確定の人気で層を切る。確かめは、発走前のオッズ 20 通り（第 2 版の shift.json）で層を切り直した本番の再現も並べる。
- 単位: 通常 = コース × クラスの群 × 良（件数の足りない条件はコースだけ → 芝ダ × 距離帯 × 群に寄せる）。
  重賞 = レース名 × コース（同じコースの 3 勝以上・全馬場の走で探す。足りなければ同じ芝ダ × 距離帯の 3 勝以上）。
- 根拠の検査（すべて満たしたものだけを採る）:
  1. オッズから見込む 3 着内（確定の単勝オッズの帯 × 頭数の帯 × 層の鍵の率。通常 = 芝ダ・距離帯・群、重賞 = 芝ダ・距離帯）を超える。正規の近似の片側 p を、違う仮説（探した層 × 鍵 × ルール）の数で
     BH 補正する（補正は面 × 人気の層ごと。3 つの層・2 つの面を合わせた数では補正していない）。あわせて、層のほかの馬（ルールに当たらない馬）を
     上回る（上乗せの 95% 区間の下限 > 0）。
  2. 閾値の 1 段ずらし: 厳しい側の隣の段（無ければ緩い側）で、オッズから見込む値を超えた分が元の半分以上。
  3. 年ごとの安定: 2016〜18・19〜21 の両方で、複勝率の区間の下限がオッズから見込む値を超える。
  4. 似た単位: 探した層の外（同じ芝ダ × 群のほかの距離帯）で、オッズから見込む値を超える。
  5. 最近の持続: 2022〜24 で、オッズから見込む値を超えた分が探す期間の半分以上（向きが同じ）。2024 年だけの逆向きは印にする。
- 線（75・60・50%）を超えると言えるかは、二項の p（帰無 = 複勝率 ≤ 線）の BH で並べる。届いた = 探した層の 2022〜24 の
  走数が MIN_CONF 以上で、Wilson の下限が線を超える。層の全馬だけで線を超えるときは「人気だけで届く」とし、届いた数に数えない。
  主の指標は、層のほかの馬との差（上乗せ）。2022〜24 は ⑤ で選ぶのにも使うので「採否に使った期間」と呼び、⑤ の前の見込み（pooled_pre5）も並べる。
  単位の「良」・人気・オッズは当日（発走前）の値なので、前日版ではなく当日のルール。
使い方: python -m src.research.course.threshold search   # <COURSE_OUT>/threshold/search.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import v2 as V2

OUTT = CR.OUT / "threshold"
FRAME = CR.OUT / "v3" / "frame.parquet"        # 2016〜2025 年の全クラスの表（第 3 版の frame。2025 年の行は search では捨てる）
DISC, CONF = (2016, 2021), (2022, 2024)
BLOCKS = ((2016, 2018), (2019, 2021))
LINES = (0.75, 0.60, 0.50)
MIN_DISC = 24        # 探す期間の走数の下限（複勝率 80% のとき 95% 区間の下限が 60% に届く走数）
MIN_CONF = 20        # 確かめの走数の下限（これ未満は「件数が足りない」）
Q_MAX = 0.05         # BH の上限
TOP = 3              # 単位ごとに出すルールの数
UPPER_CLS = 4        # 重賞の寄せ先: 3 勝クラス以上（folklore の CLS: 4 = 3 勝、5 = OP・L、6〜8 = G3〜G1）
DBANDS = [0, 1400, 1800, 2200, 9999]
# 買う時点の層（最後の絞り込み）。人気・オッズは探す切り口には使わない。("pop", k) = k 番人気以内、("odds", x) = 単勝 x 倍以下
LAYERS = {"全馬": None, "1 番人気": ("pop", 1), "2 番人気以内": ("pop", 2), "単勝 2.5 倍以下": ("odds", 2.5), "単勝 2.0 倍以下": ("odds", 2.0)}
ODDS_EDGES = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0]   # 見込みの物差しのオッズの帯


def in_layer(spec, pop: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """層 spec に入る馬（人気・オッズの無い馬は入らない）。"""
    if spec is None:
        return np.ones(len(pop), bool)
    kind, v = spec
    x = pop if kind == "pop" else odds
    return np.nan_to_num(x, nan=np.inf) <= v
HALF_LIFE = 3.0      # 重み付きの複勝率の半減期（年）。並べるだけで採否には使わない
Z = 1.96

# 種類（fam）・閾値の段（順序のある種類は段の並びが緩い ← → 厳しい）・ラベル・仮説の型・文の型
FAMS = {
    "着順": ("能力", "前走で上位に来た"), "着差": ("能力", "前走で勝ち馬に僅差だった"), "上がり": ("能力", "前走で速い上がりを使えた"),
    "4角": ("展開", "前走で前に付けられた"), "間隔": ("状態", None), "距離": ("適性", None), "斤量": ("状態", None),
    "コース実績": ("適性", "このコースで好走したことがある"), "通算": ("能力", "これまで安定して好走している"),
    "騎手": ("人", "成績の良い騎手が乗る"), "厩舎": ("人", "成績の良い厩舎の馬"),
}
SENT = {("間隔", "中 2〜8 週"): "使い詰めでも休み明けでもない間隔", ("間隔", "中 9 週以上"): "休み明け",
        ("距離", "同じ距離"): "前走と同じ距離", ("距離", "延長"): "前走から距離を延ばす", ("距離", "短縮"): "前走から距離を縮める",
        ("斤量", "斤量減"): "斤量が前走より軽い", ("斤量", "斤量増"): "斤量が前走より重い（評価が上がった）"}


def conditions(d: pd.DataFrame) -> list[dict]:
    """閾値の条件の一覧。順序のある種類は step（0 = いちばん緩い）を持つ。値の無い馬は当てはまらない。4 角の 0 は不明として外す。"""
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    pf, pm, l3, p4 = num("p_fin"), num("p_margin"), num("p_l3rank"), num("p_pos4r").where(num("p_pos4") > 0)
    iv, dd, jd = num("interval"), num("dist") - num("p_dist"), num("jw") - num("p_jw")
    st = num("starts_prev")
    rate = num("top3_prev") / st.where(st > 0)
    C = []

    def add(fam, label, m, step=None):
        C.append({"fam": fam, "label": label, "step": step, "m": m.fillna(False).to_numpy(bool)})
    for i, k in enumerate((5, 3, 2, 1)):
        add("着順", f"前走 {k} 着以内", pf <= k, i)
    for i, k in enumerate((1.0, 0.5, 0.2)):
        add("着差", f"前走の着差 {k} 秒以内", pm <= k, i)
    for i, k in enumerate((5, 3, 2, 1)):
        add("上がり", f"前走の上がり {k} 位以内", l3 <= k, i)
    for i, k in enumerate((0.5, 0.25)):
        add("4角", f"前走の 4 角が前から {int(k * 100)}% 以内", p4 <= k, i)
    add("間隔", "中 2〜8 週", (iv >= 21) & (iv <= 63))
    add("間隔", "中 9 週以上", iv >= 64)
    add("距離", "同じ距離", dd.abs() < 100)
    add("距離", "延長", dd >= 100)
    add("距離", "短縮", dd <= -100)
    add("斤量", "斤量減", jd <= -1)
    add("斤量", "斤量増", jd >= 1)
    add("コース実績", "同じコースで 3 着内あり", num("course_top3_prev") >= 1)
    for i, k in enumerate((0.3, 0.5)):
        add("通算", f"通算の複勝率 {int(k * 100)}% 以上", rate >= k, i)
    for i, k in enumerate((0.2, 0.3)):
        add("騎手", f"騎手の複勝率 {int(k * 100)}% 以上（前日まで 1 年）", num("j_rate") >= k, i)
    for i, k in enumerate((0.2, 0.3)):
        add("厩舎", f"厩舎の複勝率 {int(k * 100)}% 以上（前日まで 1 年）", num("t_rate") >= k, i)
    return C


def rules_of(C: list[dict]) -> list[tuple[int, ...]]:
    """1〜3 個の AND（同じ種類は 1 個まで）。"""
    idx = range(len(C))
    out = [(i,) for i in idx]
    out += [r for r in itertools.combinations(idx, 2) if C[r[0]]["fam"] != C[r[1]]["fam"]]
    out += [r for r in itertools.combinations(idx, 3) if len({C[i]["fam"] for i in r}) == 3]
    return out


def neighbors(C: list[dict], r: tuple[int, ...], tight: bool | None = None) -> list[tuple[int, ...]]:
    """閾値を 1 段ずらしたルール（順序のある種類の各成分を、隣の段に替える）。tight=True なら厳しい側だけ、False なら緩い側だけ。"""
    out = []
    for j, i in enumerate(r):
        if C[i]["step"] is None:
            continue
        for k, c in enumerate(C):
            if (c["fam"] == C[i]["fam"] and c["step"] is not None and abs(c["step"] - C[i]["step"]) == 1
                    and (tight is None or (c["step"] > C[i]["step"]) == tight)):
                out.append(tuple(sorted(r[:j] + (k,) + r[j + 1:])))
    return out


def text_of(C: list[dict], r: tuple[int, ...]) -> tuple[str, str, str]:
    """(ルールの文, 仮説の型, なぜ効くかの 1 文)。"""
    labels = " かつ ".join(C[i]["label"] for i in r)
    hyp = "×".join(dict.fromkeys(FAMS[C[i]["fam"]][0] for i in r))
    why = "・".join(SENT.get((C[i]["fam"], C[i]["label"])) or FAMS[C[i]["fam"]][1] for i in r)
    return labels, hyp, f"{why}馬は、力・状態・展開の裏付けが重なり、3 着以内に来やすい（仮説）"


def wilson(k: float, n: float) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg の q 値。"""
    n = len(p)
    if not n:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


# ---------------------------------------------------------------- 表
def load(year_max: int = CONF[1]) -> pd.DataFrame:
    """第 3 版の frame（全クラス）から year_max 年までを読み、確定の人気・オッズを付ける（人気の層と、オッズから見込む複勝率の物差しだけに使う）。
    year_max を超える行は捨て、残っていないことを確かめる。"""
    d = pd.read_parquet(FRAME, filters=[("year", "<=", year_max)])           # 読み込みの時点で、封印の年を外す
    yr = pd.to_numeric(d["year"]).to_numpy()
    d = d[yr <= year_max].reset_index(drop=True)
    if (pd.to_numeric(d["year"]) > year_max).any():
        raise ValueError("読まない年の行が残っている")
    d = d[d["surf"].isin(["芝", "ダ"])].reset_index(drop=True)
    d["yr"] = pd.to_numeric(d["year"]).astype(int)
    d["clsn"] = pd.to_numeric(d["cls"], errors="coerce")
    d["dband"] = pd.cut(pd.to_numeric(d["dist"], errors="coerce"), DBANDS, labels=["〜1400", "1401〜1800", "1801〜2200", "2201〜"]).astype(str)
    d["group"] = d["cls"].map(CR.group_of)
    return V2.attach_final_odds(d, range(2016, year_max + 1))


def expected(d: pd.DataFrame, fit: np.ndarray, strata: np.ndarray | None = None) -> np.ndarray:
    """オッズから見込む 3 着内の確率: 確定の単勝オッズの帯（ODDS_EDGES）× 頭数の帯（〜10・11〜14・15〜）× strata（通常 = 芝ダ × 距離帯 × 群、
    重賞 = 芝ダ × 距離帯）の 3 着内率を、fit の行（探す期間の母集団）で数える。30 走未満の組は strata を外した率、それも無ければ平均。
    オッズの無い馬（取消・欠け）は母集団の平均。人気の順位ではなくオッズの帯にするのは、1 番人気の中の強さの差まで差し引くため（2 回目のレビュー）。"""
    ov = pd.to_numeric(d["odds_final"], errors="coerce").to_numpy(float)
    has = ~np.isnan(ov) & (ov > 0)
    pop = np.digitize(np.nan_to_num(ov, nan=999), ODDS_EDGES).astype(int)
    fs = pd.to_numeric(d["field"], errors="coerce").fillna(12).to_numpy()
    fb = np.digitize(fs, [10.5, 14.5])
    top3 = (pd.to_numeric(d["fin"], errors="coerce") <= 3).to_numpy().astype(float)
    mean = float(top3[fit].mean()) if fit.any() else 0.0
    cell = pop * 3 + fb
    fit = fit & has                                                           # オッズの無い馬は、帯の率の当てはめに入れない
    mean = float(top3[fit].mean()) if fit.any() else mean
    n0 = np.bincount(cell[fit], minlength=(len(ODDS_EDGES) + 1) * 3 + 3)
    k0 = np.bincount(cell[fit], weights=top3[fit], minlength=(len(ODDS_EDGES) + 1) * 3 + 3)
    r0 = np.where(n0 >= 30, k0 / np.maximum(n0, 1), mean)[cell]
    if strata is not None:
        sc = pd.factorize(pd.Series(strata).astype(str) + "#" + pd.Series(cell).astype(str))[0]
        n1 = np.bincount(sc[fit], minlength=sc.max() + 1)
        k1 = np.bincount(sc[fit], weights=top3[fit], minlength=sc.max() + 1)
        r0 = np.where(n1[sc] >= 30, k1[sc] / np.maximum(n1[sc], 1), r0)
    return np.where(has, r0, mean)


def ymat(d: pd.DataFrame, e: np.ndarray) -> np.ndarray:
    """列: 0 走・1 1 着・2 2 着・3 3 着・4 払戻の解けた走・5 単勝の払戻・6 複勝の払戻（1 円あたり）・7 オッズから見込む 3 着内・8 その分散。"""
    fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy()
    pay = d["pay_ok"].fillna(False).astype(bool).to_numpy()
    rt = np.where(pay, pd.to_numeric(d["ret_tan"], errors="coerce").fillna(0).to_numpy(), 0.0)
    rf = np.where(pay, pd.to_numeric(d["ret_fuku"], errors="coerce").fillna(0).to_numpy(), 0.0)
    return np.column_stack([np.ones(len(d)), fin == 1, fin == 2, fin == 3, pay, rt, rf, e, e * (1 - e)]).astype(float)


def pools(d: pd.DataFrame, side: str) -> dict:
    """side = "course" / "graded"。単位（target）・寄せる層（levels: 細かい順）・似た単位を見る鍵（探した層の外）を、行ごとに作る。対象外の行は ""。"""
    good = (d["track"].astype(str) == "良").to_numpy()
    if side == "course":
        ok = good & d["group"].notna().to_numpy()
        unit = np.where(ok, CR.regroup(d.assign(group=d["group"].fillna("")), 4).to_numpy(), "")
        course = np.where(ok, d["course"].astype(str).to_numpy(), "")
        grp = pd.Series(unit).str.split("|", n=1).str[1].fillna("").to_numpy()        # 新馬を未勝利から分けた群（単位と同じ分け方）
        fam = np.where(ok, (d["surf"].astype(str) + "|" + d["dband"] + "|").to_numpy() + grp, "")
        outer = np.where(ok, (d["surf"].astype(str) + "|").to_numpy() + grp, "")   # 似た単位: 同じ芝ダ × 群の、ほかの距離帯
        return {"target": unit, "levels": [("単位", unit), ("コース", course), ("芝ダ × 距離帯 × 群", fam)], "fam": fam, "outer": outer,
                "pop_rows": ok, "outer_name": "同じ芝ダ × 群のほかの距離帯"}
    up = (d["clsn"] >= UPPER_CLS).to_numpy()
    gr = (d["clsn"] >= 6).to_numpy() & d["rkey"].notna().to_numpy()
    unit = np.where(gr, (d["rkey"].astype(str) + "|" + d["course"].astype(str)).to_numpy(), "")
    course = np.where(up, d["course"].astype(str).to_numpy(), "")
    fam = np.where(up, (d["surf"].astype(str) + "|" + d["dband"]).to_numpy(), "")
    outer = np.where(up, d["surf"].astype(str).to_numpy(), "")
    return {"target": unit, "levels": [("同じコースの 3 勝以上", course), ("同じ芝ダ × 距離帯の 3 勝以上", fam)], "fam": fam, "outer": outer,
            "pop_rows": up, "outer_name": "同じ芝ダの 3 勝以上のほかの距離帯"}


def _sums(codes: np.ndarray, Y: np.ndarray, m: np.ndarray, K: int) -> np.ndarray:
    """codes ごとに Y の列を足す（m の行だけ）。"""
    c = codes[m]
    return np.stack([np.bincount(c, weights=Y[m, j], minlength=K) for j in range(Y.shape[1])], axis=1)


def _ex(s) -> float | None:
    """オッズから見込む値を超えた分（1 走あたり）。"""
    return float((s[1] + s[2] + s[3] - s[7]) / s[0]) if s[0] else None


def search(d: pd.DataFrame, side: str, layer: str = "全馬", shift: dict | None = None, log=print) -> dict:
    """人気の層 layer の中で、探す（2016〜21）→ 検査 → 単位ごとに TOP 個 → 確かめ（2022〜24）。2025 年以降の行があれば止める。
    層は確定の人気で切る（探す・採否）。確かめの本番の再現は、発走前のオッズ 20 通りで層を切り直した平均（shift があれば）。"""
    if (d["yr"] > CONF[1]).any():
        raise ValueError("2025 年以降の行がある（search では読まない）")
    t0 = time.time()
    yr = d["yr"].to_numpy()
    A, A1, A2, B = ((yr >= lo) & (yr <= hi) for lo, hi in (DISC, BLOCKS[0], BLOCKS[1], CONF))
    P = pools(d, side)
    Y = ymat(d, expected(d, P["pop_rows"] & A, P["fam"]))
    popf = pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float)
    spec = LAYERS[layer]
    pre = [V2.simulate_pre(d, shift, kd) for kd in range(V2.N_DRAWS)] if (shift is not None and spec is not None) else None
    pre_in = [in_layer(spec, pp_, po_) for po_, pp_ in pre] if pre else None   # 発走前のオッズ 20 通りで切り直した層
    lay = in_layer(spec, popf, pd.to_numeric(d["odds_final"], errors="coerce").to_numpy(float))
    C = conditions(d)
    R = rules_of(C)
    rid = {r: i for i, r in enumerate(R)}
    lv = []
    for name, key in P["levels"]:
        codes, names = pd.factorize(key)
        lv.append({"name": name, "codes": codes, "names": list(names), "K": len(names), "valid": (key != "") & lay})
    fc, fn = pd.factorize(P["fam"])
    oc, on = pd.factorize(P["outer"])
    fvalid, ovalid = (P["fam"] != "") & lay, (P["outer"] != "") & lay
    tcodes, tnames = pd.factorize(P["target"])
    Kt = len(tnames)
    grade_of = {}
    if side == "graded" and "grade" in d.columns:
        gtab = pd.DataFrame({"u": P["target"], "g": d["grade"].astype(str).to_numpy(), "y": yr})
        gtab = gtab[gtab["u"] != ""].sort_values("y")
        grade_of = gtab.groupby("u")["g"].last().to_dict()   # いちばん新しい回の格
    for L in lv:   # 層の中の全馬（基準線）
        L["baseA"] = _sums(L["codes"], Y, A & L["valid"], L["K"])
        L["baseB"] = _sums(L["codes"], Y, B & L["valid"], L["K"])
    S = {k: [] for k in ("A", "A1", "A2", "B", "Bt", "F", "O")}
    for r in R:
        m = C[r[0]]["m"].copy()
        for i in r[1:]:
            m &= C[i]["m"]
        S["A"].append([_sums(L["codes"], Y, m & A & L["valid"], L["K"]) for L in lv])
        S["A1"].append([_sums(L["codes"], Y, m & A1 & L["valid"], L["K"]) for L in lv])
        S["A2"].append([_sums(L["codes"], Y, m & A2 & L["valid"], L["K"]) for L in lv])
        S["B"].append([_sums(L["codes"], Y, m & B & L["valid"], L["K"]) for L in lv])
        S["Bt"].append(_sums(tcodes, Y, m & B & (P["target"] != "") & lay, Kt))
        S["F"].append(_sums(fc, Y, m & A & fvalid, len(fn)))
        S["O"].append(_sums(oc, Y, m & A & ovalid, len(on)))
    log(f"{side}/{layer}: {len(R)} ルール × 層 {[L['K'] for L in lv]} の集計 {time.time() - t0:.0f}s", CR.mem_now())
    tgt_rows = {}
    for u in tnames:
        if not u:
            continue
        j = np.flatnonzero(P["target"] == u)[0]
        keys = [L["codes"][j] if P["levels"][li][1][j] != "" else -1 for li, L in enumerate(lv)]
        fo = (int(fc[j]), int(oc[j])) if P["fam"][j] != "" and P["outer"][j] != "" else None
        tgt_rows[u] = (keys, fo, int(np.flatnonzero(tnames == u)[0]))
    cand = []
    for u, (keys, fo, ti) in tgt_rows.items():
        for ri in range(len(R)):
            for li in range(len(lv)):
                if keys[li] < 0:
                    continue
                s = S["A"][ri][li][keys[li]]
                if s[0] >= MIN_DISC:
                    cand.append((u, ri, li, keys[li], fo, ti, s))
                    break
    log(f"{side}/{layer}: 候補（単位 × ルール） {len(cand):,}", f"{time.time() - t0:.0f}s")
    from scipy.stats import binom, norm
    n = np.array([c[6][0] for c in cand], float)
    k = np.array([c[6][1] + c[6][2] + c[6][3] for c in cand], float)
    E = np.array([c[6][7] for c in cand], float)
    V = np.array([c[6][8] for c in cand], float)
    hk = pd.factorize(pd.Series([f"{c[2]}|{c[3]}|{c[1]}" for c in cand]))[0] if cand else np.array([], int)
    first = np.unique(hk, return_index=True)[1] if cand else np.array([], int)
    n_hyp = len(first)

    def q_of(p):
        return bh(p[first])[hk] if cand else p
    q = {L: q_of(binom.sf(k - 1, n, L)) for L in LINES}                       # 線を超えるか（言えるかどうかを並べる）
    p_pop = norm.sf((k - E) / np.sqrt(np.maximum(V, 1e-9))) if cand else np.array([])
    q_pop = q_of(p_pop)                                                       # オッズから見込む値を超えるか（採る条件）
    ref = {}
    for ci, c in enumerate(cand):
        if c[0] not in ref or p_pop[ci] < p_pop[ref[c[0]]]:
            ref[c[0]] = ci
    why = {k_: 0 for k_ in ("オッズから見込む値を超えると言えない（BH）", "1 段ずらし", "1 段ずらしを確かめられない", "層のほかの馬を上回ると言えない", "年ごとの安定",
                            "似た単位（探した層の外）", "似た単位の走が足りない", "最近（2022〜24）で効きが半分未満・逆向き", "最近の走が足りない", "通った")}
    out_units, pass4 = {}, []
    for ci, (u, ri, li, key, fo, ti, s) in enumerate(cand):
        if q_pop[ci] > Q_MAX or k[ci] <= E[ci]:
            why["オッズから見込む値を超えると言えない（BH）"] += 1
            continue
        ex = (k[ci] - E[ci]) / n[ci]
        ordered = any(C[i]["step"] is not None for i in R[ri])
        # 順序のある成分ごとに、厳しい側の隣（10 走以上）を使い、無ければ緩い側の隣を使う。どちらも無い成分があれば確かめられない
        nb, unver = [], False
        for j, i in enumerate(R[ri]):
            if C[i]["step"] is None:
                continue
            got = None
            for tight in (True, False):
                for c2, cc in enumerate(C):
                    if cc["fam"] == C[i]["fam"] and cc["step"] is not None and cc["step"] - C[i]["step"] == (1 if tight else -1):
                        r2 = tuple(sorted(R[ri][:j] + (c2,) + R[ri][j + 1:]))
                        if r2 in rid and S["A"][rid[r2]][li][key][0] >= 10:
                            got = _ex(S["A"][rid[r2]][li][key])
                if got is not None:
                    break
            if got is None:
                unver = True
            else:
                nb.append(got)
        if unver:
            why["1 段ずらしを確かめられない"] += 1
            continue
        if nb and not all(x >= 0.5 * ex for x in nb):
            why["1 段ずらし"] += 1
            continue
        upA = _uplift(_stat(s), _stat(lv[li]["baseA"][key] - s))
        if upA is None or upA["lo"] <= 0:
            why["層のほかの馬を上回ると言えない"] += 1
            continue
        blk = [S["A1"][ri][li][key], S["A2"][ri][li][key]]
        wl = [wilson(b[1] + b[2] + b[3], b[0]) for b in blk]
        if not all(b[0] >= 5 and w is not None and w[0] > b[7] / b[0] for b, w in zip(blk, wl)):
            why["年ごとの安定"] += 1
            continue
        if fo is None:
            why["似た単位の走が足りない"] += 1
            continue
        so = S["O"][ri][fo[1]] - S["F"][ri][fo[0]]
        if so[0] < 20:
            why["似た単位の走が足りない"] += 1
            continue
        if _ex(so) <= 0:
            why["似た単位（探した層の外）"] += 1
            continue
        pass4.append((ri, li, int(key)))                    # ①〜④ を通った（⑤ の前）。⑤ で選ぶ前の見込みを数えるため
        sb = S["B"][ri][li][key]
        if sb[0] < 10:
            why["最近の走が足りない"] += 1
            continue
        if _ex(sb) < 0.5 * ex:
            why["最近（2022〜24）で効きが半分未満・逆向き"] += 1
            continue
        why["通った"] += 1
        out_units.setdefault(u, []).append({"rule": ri, "li": li, "kc": int(key), "hk": int(hk[ci]), "level": lv[li]["name"], "key": lv[li]["names"][key],
                                            "disc": _stat(s), "ex_disc": ex, "neighbors": nb, "shift": "あり" if ordered else "対象なし",
                                            "blocks": [_stat(b) for b in blk], "outer": _stat(so), "conf_level": _stat(sb), "conf": _stat(S["Bt"][ri][ti]),
                                            "q_pop": float(q_pop[ci]), "q": {str(Lq): float(q[Lq][ci]) for Lq in LINES},
                                            "pass": {str(Lq): bool(q[Lq][ci] <= Q_MAX) for Lq in LINES},
                                            "baseA": _stat(lv[li]["baseA"][key]), "baseB": _stat(lv[li]["baseB"][key])})
    log(f"{side}/{layer}: 検査で落ちた理由 {why}")
    res = {}
    for u in sorted(tgt_rows):
        xs = out_units.get(u, [])
        xs.sort(key=lambda x: -(x["disc"]["lo"] - x["disc"]["e"]))
        pick = []
        for x in xs:
            if all(len(set(R[x["rule"]]) & set(R[y["rule"]])) < 2 for y in pick):
                pick.append(x)
            if len(pick) >= TOP:
                break
        for x in pick:
            _finish(x, d, Y, C, R, lv, P, lay, B, yr, pre_in)
        rx = None
        if u in ref and not pick:
            ci = ref[u]
            _, ri, li, key, _, ti2, s0 = cand[ci]
            t, h, w = text_of(C, R[ri])
            rx = {"text": t, "hyp": h, "why": w, "level": lv[li]["name"], "passed1": bool(q_pop[ci] <= Q_MAX and k[ci] > E[ci]), "disc": _stat(s0), "conf": _stat(S["Bt"][ri][ti2]),
                  "conf_level": _stat(S["B"][ri][li][key]), "p_pop": float(p_pop[ci]), "q_pop": float(q_pop[ci]),
                  "baseA": _stat(lv[li]["baseA"][key]), "baseB": _stat(lv[li]["baseB"][key])}
        ti = tgt_rows[u][2]
        res[u] = {"rules": pick, "ref": rx, "n_disc": int(np.sum((tcodes == ti) & A & lay)), "n_conf": int(np.sum((tcodes == ti) & B & lay)),
                  "grade": grade_of.get(u, "")}
    U = np.zeros(len(d), bool)
    for v in res.values():
        for x in v["rules"]:
            U |= x.pop("_mask")
    pooled = {"disc": _stat(Y[U & A].sum(0)), "conf_level": _stat(Y[U & B].sum(0))} if U.any() else {}
    # ⑤（最近の持続）で選ぶ前の見込み: ①〜④ を通った組すべての走（OR）の 2022〜24。⑤ は 2022〜24 で選ぶので、選んだ後の値は楽観になる
    U4 = np.zeros(len(d), bool)
    for ri, li, key in set(pass4):
        m = lv[li]["valid"] & (lv[li]["codes"] == key)
        for i in R[ri]:
            m = m & C[i]["m"]
        U4 |= m
    pooled_pre5 = {"disc": _stat(Y[U4 & A].sum(0)), "conf_level": _stat(Y[U4 & B].sum(0)), "pairs": len(set(pass4))} if U4.any() else {}
    # 層の全体（対象の単位の走）の基準線
    tg = (P["target"] != "") & lay
    layer_base = {"disc": _stat(Y[tg & A].sum(0)), "conf_level": _stat(Y[tg & B].sum(0))}
    reached = {}
    for Lq in LINES:
        lk = str(Lq)
        reached[lk] = {"units_level": sum(1 for v in res.values() if any(x["reached"][lk] and not x["pop_alone"][lk] for x in v["rules"])),
                       "hyp": len({x["hk"] for v in res.values() for x in v["rules"] if x["reached"][lk] and not x["pop_alone"][lk]}),
                       "units_own": sum(1 for v in res.values() if any(x["reached_unit"][lk] and not x["pop_alone"][lk] for x in v["rules"])),
                       "pop_only": len({x["hk"] for v in res.values() for x in v["rules"] if x["reached"][lk] and x["pop_alone"][lk]})}
    log(f"{side}/{layer}: 単位 {len(res)}・ルールのある単位 {sum(1 for v in res.values() if v['rules'])}・{time.time() - t0:.0f}s")
    return {"side": side, "layer": layer, "rules_n": len(R), "tests": n_hyp, "pairs": len(cand), "why": why, "units": res, "pooled_selected": pooled,
            "pooled_pre5": pooled_pre5, "layer_base": layer_base,
            "reached": reached, "levels": [L["name"] for L in lv], "outer_name": P["outer_name"]}


def _finish(x, d, Y, C, R, lv, P, lay, B, yr, pre_in) -> None:
    """選んだルールに、文・3 期間と年ごとの値・重み付きの複勝率・効きの流れ・上乗せ・届いた線・本番の再現の確かめを足す。"""
    L = lv[x["li"]]
    m_rule = np.ones(len(d), bool)
    for i in R[x["rule"]]:
        m_rule &= C[i]["m"]
    lvl = (L["codes"] == x["kc"]) & (P["levels"][x["li"]][1] != "")
    m = m_rule & lvl & lay
    x["_mask"] = m.copy()
    x["text"], x["hyp"], x["why"] = text_of(C, R[x["rule"]])
    x["conds"] = [C[i]["label"] for i in R[x["rule"]]]
    per = {}
    for lo, hi in ((2016, 2018), (2019, 2021), (2022, 2024)):
        mm = m & (yr >= lo) & (yr <= hi)
        bb = lvl & lay & (yr >= lo) & (yr <= hi)
        per[f"{lo}〜{hi}"] = {"rule": _stat(Y[mm].sum(0)), "base": _stat(Y[bb].sum(0))}
    x["periods"] = per
    exs = [p["rule"]["ex"] for p in per.values()]
    x["trend"] = "弱まり続け" if all(v is not None for v in exs) and exs[0] > exs[1] > exs[2] else (
        "強まり続け" if all(v is not None for v in exs) and exs[0] < exs[1] < exs[2] else "一定でない")
    w = 0.5 ** ((CONF[1] - yr) / HALF_LIFE)
    sw = (Y[m][:, [0, 1, 2, 3, 7]] * w[m][:, None]).sum(0)
    x["weighted"] = {"place": float((sw[1] + sw[2] + sw[3]) / sw[0]) if sw[0] else None, "e": float(sw[4] / sw[0]) if sw[0] else None,
                     "half_life": HALF_LIFE}
    s24 = Y[m & (yr == CONF[1])].sum(0)
    x["y2024"] = _stat(s24)
    w24 = wilson(s24[1] + s24[2] + s24[3], s24[0])
    x["y2024_reversed"] = bool(s24[0] >= 5 and w24 is not None and w24[1] < s24[7] / s24[0])
    cl, cu, bB = x["conf_level"], x["conf"], x["baseB"]
    x["uplift"] = _uplift(cl, _stat(Y[lvl & lay & B & ~m_rule].sum(0)))    # ルール対、層のほかの馬（ルールは層の全馬の部分集合なので、残りと比べる）
    x["reached"] = {str(Lq): bool(cl["n"] >= MIN_CONF and cl["lo"] is not None and cl["lo"] > Lq) for Lq in LINES}
    x["reached_unit"] = {str(Lq): bool(cu["n"] >= MIN_CONF and cu["lo"] is not None and cu["lo"] > Lq) for Lq in LINES}
    x["pop_alone"] = {str(Lq): bool(bB["place"] is not None and bB["place"] > Lq) for Lq in LINES}   # 層の全馬だけで線を超える（人気だけで届く）
    if pre_in is not None:
        x["conf_repro"] = _repro(Y, m_rule & lvl & B, lvl & B, pre_in)


def _uplift(r: dict, b: dict) -> dict | None:
    """閾値による上乗せ（ルールの複勝率 − b の複勝率）と、その 95% 区間（正規の近似）。b はルールと重ならない残り（層のほかの馬）を渡す。"""
    if not r["n"] or not b["n"] or r["place"] is None or b["place"] is None:
        return None
    dlt = r["place"] - b["place"]
    se = np.sqrt(r["place"] * (1 - r["place"]) / r["n"] + b["place"] * (1 - b["place"]) / b["n"])
    return {"pt": float(dlt), "lo": float(dlt - Z * se), "hi": float(dlt + Z * se)}


def _repro(Y, m_rule_b, m_base_b, pre_in) -> dict:
    """本番の再現: 発走前のオッズ 20 通り（第 2 版のずれの分布）で層を切り直した、2022〜24 の値の平均。上乗せは、ルール対、層のほかの馬。"""
    rs, bs, os_ = [], [], []
    for inl in pre_in:
        rs.append(Y[m_rule_b & inl].sum(0))
        bs.append(Y[m_base_b & inl].sum(0))
        os_.append(Y[m_base_b & inl & ~m_rule_b].sum(0))
    r, b, o = _stat(np.mean(rs, axis=0)), _stat(np.mean(bs, axis=0)), _stat(np.mean(os_, axis=0))
    return {"rule": r, "base": b, "uplift": _uplift(r, o)}


def _stat(s) -> dict:
    """[走, 1 着, 2 着, 3 着, 払戻の解けた走, 単勝の払戻, 複勝の払戻, オッズから見込む 3 着内, 分散] から成績。"""
    s = np.asarray(s, float)
    n, k = s[0], s[1] + s[2] + s[3]
    w = wilson(k, n)
    e = float(s[7] / n) if n and len(s) > 7 else None
    return {"n": float(n), "w": [float(s[1]), float(s[2]), float(s[3])], "out": float(n - k), "place": float(k / n) if n else None,
            "win": float(s[1] / n) if n else None, "lo": w[0] if w else None, "hi": w[1] if w else None, "npay": float(s[4]),
            "rt": float(s[5]), "rf": float(s[6]), "e": e, "ex": float(k / n - e) if n and e is not None else None,
            "roi_t": float(s[5] / s[4]) if s[4] else None, "roi_f": float(s[6] / s[4]) if s[4] else None}   # 払戻は 1 円あたり（ret_tan）


def cmd_search() -> None:
    CR.wait_resources()
    OUTT.mkdir(parents=True, exist_ok=True)
    d = load()
    shift = json.loads((CR.OUT / "v2" / "shift.json").read_text(encoding="utf-8"))
    out = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "disc": DISC, "conf": CONF, "lines": LINES, "min_disc": MIN_DISC,
           "min_conf": MIN_CONF, "q_max": Q_MAX, "layers": list(LAYERS), "half_life": HALF_LIFE,
           "note": "2025 年以降は読んでいない。探す 2016〜21・採否と確かめ 2022〜24（最近の持続の検査に使うので、独立した確かめは凍結の後の 2025〜26 だけ）。"
                   "層（人気・オッズ）は確定の値で切る。全馬以外の層の 2022〜24 は、発走前のオッズ 20 通り（第 2 版の shift.json）で切り直した値も並べる",
           "shift_sha": CR._sha_file(CR.OUT / "v2" / "shift.json")}
    for side in ("course", "graded"):
        out[side] = {}
        for layer in LAYERS:
            out[side][layer] = search(d, side, layer, shift, log=lambda *a: print(*a, flush=True))
    (OUTT / "search.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("search", OUTT / "search.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search"])
    a = ap.parse_args()
    if a.cmd == "search":
        cmd_search()


if __name__ == "__main__":
    sys.exit(main())
