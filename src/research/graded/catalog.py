"""重賞のルールベース予想家: 切り口の目録（docs/ideas/graded_handicapper_cuts.md の確定版をコードにしたもの）。

1 つの切り口（型）は、各出走に「水準」の文字列を 1 つ付ける関数。値が無い・対象外の走は UNKNOWN（点にも件数の表示にも使わない）。
- cols: その型が使う列。全型の cols の和を ALLOWED_COLUMNS とし、市場（オッズ・人気）とその走自身の結果の列が入っていないことを
  tests/research/test_graded_catalog.py で検査する（ホワイトリスト方式）。
- day_of: 当日に決まる値（馬体重・馬場など）を使う型。前日版では使わない。
- start_year: この年より前の走は、この型の集計に入れない（前走系は 2017〜、経歴系は 2018〜。履歴が 2016 年からしか無いため）。
- optional: 比較の版だけで使う型（めぐ指数の順位。採否の対象にしない）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

UNKNOWN = "不明"
PREV_FOREIGN = "前走が海外・地方"


@dataclass(frozen=True)
class CutType:
    id: str
    name: str
    family: str
    cols: tuple[str, ...]
    fn: Callable[[pd.DataFrame], pd.Series]
    day_of: bool = False
    start_year: int = 2016
    prev: bool = False          # 前走の列を使う型（前走が海外・地方のときは PREV_FOREIGN にする）
    optional: bool = False
    note: str = ""


FAMILIES = {
    "A": "前走の成績", "B": "前走のクラス・レース名", "C": "前走からの条件の変化", "D": "間隔・休み明け", "E": "前走の脚質・位置・上がり",
    "F": "馬の経歴", "G": "馬の属性", "H": "今回の条件", "I": "騎手", "J": "調教師・コンビ", "K": "血統", "L": "調教", "X": "組み合わせ",
    "M": "指数（比較の版だけ）",
}

# 市場の情報・その走自身の結果（catalog の cols に入ってはいけない列）
FORBIDDEN_EXACT = {
    "odds", "popularity", "pop", "p_pop", "p_odds", "q", "p_q", "fav1_prev", "tr_pop_rank", "own_pop_rank", "jday_minpop_rank",
    "fin", "y1", "y3", "pos1", "pos4", "pos1r", "pos4r", "l3f", "l3rank", "tdiff", "tsec", "pace_cat", "lap_cat", "pdiff", "pace2", "megu",
    "pd_rank", "bias_pos", "bias_hn", "race_fast", "day_fast", "p_race_fast", "finish_position", "time_sec", "margin", "last_3f",
    "passing_order", "payoff",
    # 当日の結果で決まる列（発走したか・発走した頭数・その走が数えられるか）。頭数の切り口は出走表の頭数 field_decl を使う
    "started", "field", "ok",
}
FORBIDDEN_PATTERN = r"odds|pop|ninki|tansho|fukusho|payoff|market|ret_"


# ------------------------------------------------------------------ 水準を作る道具
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def cut(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    """右閉じの区間で水準を付ける（edges は昇順の境目。len(labels) == len(edges) + 1）。値が無ければ UNKNOWN。"""
    v = _num(x).to_numpy(float)
    idx = np.searchsorted(np.asarray(edges, float), v, side="left")
    out = np.asarray(labels, dtype=object)[np.clip(idx, 0, len(labels) - 1)]
    out[np.isnan(v)] = UNKNOWN
    return pd.Series(out, index=x.index)


def fin_bucket(x: pd.Series, kind: str = "5") -> pd.Series:
    if kind == "5":
        return cut(x, [1, 3, 5, 9], ["1着", "2-3着", "4-5着", "6-9着", "10着以下"])
    if kind == "4":
        return cut(x, [1, 3, 5], ["1着", "2-3着", "4-5着", "6着以下"])
    if kind == "3":
        return cut(x, [3, 5], ["3着以内", "4-5着", "6着以下"])
    if kind == "3b":
        return cut(x, [3, 9], ["3着以内", "4-9着", "10着以下"])
    if kind == "2":
        return cut(x, [3], ["3着以内", "4着以下"])
    if kind == "2b":
        return cut(x, [5], ["5着以内", "6着以下"])
    raise ValueError(kind)


def combo(*parts: pd.Series) -> pd.Series:
    """水準の掛け合わせ。どれかが UNKNOWN なら UNKNOWN。"""
    out = parts[0].astype(str)
    bad = parts[0].astype(str) == UNKNOWN
    for p in parts[1:]:
        out = out + "×" + p.astype(str)
        bad |= p.astype(str) == UNKNOWN
    return out.where(~bad, UNKNOWN)


def truthy(s: pd.Series) -> pd.Series:
    """真偽の列（bool・"True"/"False" の文字列・0/1 のどれでも）を bool にする。値が無ければ False。"""
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def when(cond: pd.Series, s: pd.Series, other: str = UNKNOWN) -> pd.Series:
    return s.where(cond.fillna(False).astype(bool), other)


RATE_EDGES = (0.15, 0.28)     # 累計の 3 着内率の帯の境目（平均は頭数から見て約 21%。データから決めないので年による偏りが出ない）


def rate_bucket(n: pd.Series, k: pd.Series, min_n: int, edges=RATE_EDGES,
                labels=("3着内率 15%未満", "15-28%", "28%以上")) -> pd.Series:
    """as-of の累計の 3 着内率の帯（固定の境目）。件数が min_n 未満は「件数が少ない」。"""
    n = _num(n)
    r = _num(k) / n.where(n >= min_n)
    out = cut(r, list(edges), list(labels))
    return out.where(n >= min_n, "件数が少ない").where(n.notna(), UNKNOWN)


# ------------------------------------------------------------------ 各族の水準（d は features.build の表）
def _ref(d):   # 率の帯の境目を決める基準の行（集計の発見期間より前の 2017 年の重賞）
    return d["grade"].isin(["G1", "G2", "G3"]) & (d["year"] == 2017)


def _p_cls(d):
    c = _num(d["p_cls"])
    return cut(c, [4, 5, 6, 7], ["3勝以下", "OP・L", "G3", "G2", "G1"])


def _style(d):
    r = _num(d["p_pos1r"])
    out = cut(r, [0.3, 0.7], ["先行", "中団", "後方"])
    return out.where(_num(d["p_pos1"]) != 1, "逃げ")


def _style2(d):
    s = _style(d)
    return s.replace({"逃げ": "逃げ・先行", "先行": "逃げ・先行", "中団": "中団・後方", "後方": "中団・後方"})


def _l3(d):
    return cut(d["p_l3rank"], [1, 3, 5], ["上がり1位", "上がり2-3位", "上がり4-5位", "上がり6位以下"])


def _dist_chg(d):
    return cut(_num(d["dist"]) - _num(d["p_dist"]), [-400, -100, 99, 399], ["400m以上短縮", "100-300m短縮", "同距離", "100-300m延長", "400m以上延長"])


def _interval(d):
    return cut(d["interval"], [14, 35, 63, 182], ["連闘・中1週", "中2-4週", "中5-8週", "中9-25週", "半年以上"])


def _age(d):
    return cut(d["age"], [2, 3, 4, 5, 6], ["2歳", "3歳", "4歳", "5歳", "6歳", "7歳以上"])


def _track(s):
    return s.astype(object).map({"良": "良", "稍": "稍重", "重": "重・不良", "不": "重・不良"}).fillna(UNKNOWN)


def _cnt3(n, k, labels=("3着内あり", "走ったが3着内なし", "未経験")):
    n, k = _num(n), _num(k)
    out = pd.Series(np.where(n > 0, np.where(k > 0, labels[0], labels[1]), labels[2]), index=n.index, dtype=object)
    return out.where(n.notna(), UNKNOWN)


def _fl(d):
    return d["f_large"].astype(object).fillna(UNKNOWN)


def _season(d):
    m = _num(d["month"])
    return pd.Series(np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])], ["冬", "春", "夏"], "秋"), index=d.index)


def _rotation(d):
    vc, pv = d["vc"].astype(str), d["p_vc"].astype(str)
    same_dir = d["direction"].astype(str) == d["p_direction"].astype(str)
    out = pd.Series(np.where(vc == pv, "同じ会場", np.where(same_dir, "違う会場・同じ回り", "違う会場・違う回り")), index=d.index, dtype=object)
    return out.where(d["p_vc"].notna(), UNKNOWN)


def _surf_chg(d):
    return combo(d["p_surf"].astype(object).fillna(UNKNOWN).astype(str), d["surf"].astype(str)).str.replace("×", "→")


def _blinker(d):
    b, pb, ever = _num(d["blinker"]), _num(d["p_blinker"]), _num(d["blk_ever_prev"])
    out = np.select([(b == 1) & (ever.fillna(0) == 0), b == 1, (b == 0) & (pb == 1), b == 0], ["初着用", "着用（2回目以上）", "外した", "なし"], UNKNOWN)
    return pd.Series(out, index=d.index, dtype=object)


def _away(d):
    v = d["venue"].astype(str)
    out = np.select([v.isin(["札幌", "函館"]), v == "小倉", _num(d["away"]) == 1, _num(d["away"]) == 0], ["北海道", "小倉", "遠征", "地元"], UNKNOWN)
    out = pd.Series(out, index=d.index, dtype=object)
    tz = d["tozai"].astype(object).fillna("")
    return out.where(out != "遠征", np.where(tz == "美浦", "遠征（関東馬が関西へ）", "遠征（関西馬が関東へ）"))


def _jockey_cont(d):
    c = _num(d["j_cont"])
    out = cut(c, [0, 1], ["乗り替わり", "継続2戦目", "継続3戦以上"])
    return out.where(_num(d["starts_prev"]) > 0, UNKNOWN)


def _j_switch(d):
    diff = _num(d["j_rate"]) - _num(d["pj_rate"])
    out = cut(diff, [-0.05, 0.05], ["格下げの乗り替わり", "同等の乗り替わり", "格上げの乗り替わり"])
    out = out.where(_num(d["j_cont"]) == 0, "継続騎乗")
    return out.where(_num(d["starts_prev"]) > 0, UNKNOWN)


def _nige(d):
    me = _num(d["nige_cand"]) == 1
    n = _num(d["n_nige_cand"])
    out = np.select([me & (n <= 1), me, ~me & (n <= 1), ~me], ["逃げ候補（自分だけ）", "逃げ候補（ほかにも候補）", "逃げ候補でない（候補0-1頭）", "逃げ候補でない（候補2頭以上）"], UNKNOWN)
    return pd.Series(out, index=d.index, dtype=object).where(n.notna(), UNKNOWN)


def _b04(d):
    pc = _num(d["p_cls"])
    out = pd.Series(np.where(pc >= 6, "前走 " + d["p_rkey"].astype(str), np.where(pc == 5, "前走 OP・L", "前走 3勝以下")), index=d.index, dtype=object)
    return out.where(pc.notna(), UNKNOWN)


def _meet(d):
    return cut(d["meet_day"], [4, 6], ["開催1-4日目", "開催5-6日目", "開催7日目以降"])


def _jw_rank(d):
    r = _num(d["jw_rank"])
    out = cut(r, [1 / 3, 2 / 3], ["斤量 上位1/3", "斤量 中", "斤量 下位1/3"])
    return out.where(~truthy(d["jw_max"]), "最重量")


def _eval(d):
    e = d["ok_eval"].astype(object)
    return e.map({"A": "A", "B": "B", "C": "C・D", "D": "C・D"}).fillna("評価なし")


def _io_chg(d):
    io, pio = d["io"].astype(object), d["p_io"].astype(object)
    ok = io.isin(["内", "外"]) & pio.isin(["内", "外"])
    out = pd.Series(np.where(io == pio, "内外が同じ", pio.astype(str) + "回り→" + io.astype(str) + "回り"), index=d.index, dtype=object)
    return out.where(ok, UNKNOWN)


def _best_range(d):
    lo, hi, x = _num(d["best_dist_min"]), _num(d["best_dist_max"]), _num(d["dist"])
    out = np.select([lo.isna(), x > hi, x < lo], ["3着内なし", "好走範囲より長い", "好走範囲より短い"], "好走範囲の中")
    return pd.Series(out, index=d.index, dtype=object).where(_num(d["starts_prev"]) > 0, UNKNOWN)


def _dist_exp(d):
    mx, mn, x = _num(d["dist_max_prev"]), _num(d["dist_min_prev"]), _num(d["dist"])
    out = np.select([x >= mx + 200, x < mn], ["初めての距離（200m以上延長）", "経験した最短より短い"], "経験の範囲")
    return pd.Series(out, index=d.index, dtype=object).where(mx.notna(), UNKNOWN)


def _rest_k(d):
    return cut(d["since_rest"], [1, 2, 3], ["休み明け初戦", "休み明け2戦目", "休み明け3戦目", "休み明け4戦目以上"])


def _types() -> list[CutType]:
    T = CutType
    P = dict(prev=True, start_year=2017)
    H = dict(start_year=2018)
    return [
        # A 前走の成績
        T("A01", "前走の着順", "A", ("p_fin",), lambda d: fin_bucket(d["p_fin"]), **P),
        T("A02", "前走の着差", "A", ("p_margin", "p_fin"), lambda d: cut(d["p_margin"], [-0.3], ["勝ち（0.3秒以上離した）", "勝ち（0.2秒差以内）"]).where(
            _num(d["p_fin"]) == 1, cut(d["p_margin"], [0.2, 0.5, 0.9], ["負け 0.0-0.2秒", "負け 0.3-0.5秒", "負け 0.6-0.9秒", "負け 1.0秒以上"])), **P),
        T("A03", "前走の着順 × 着差", "A", ("p_fin", "p_margin"), lambda d: pd.Series(np.select(
            [_num(d["p_fin"]) <= 5, _num(d["p_margin"]) <= 0.5, _num(d["p_margin"]) > 0.5], ["5着以内", "6着以下・0.5秒以内", "6着以下・0.6秒以上"], UNKNOWN),
            index=d.index, dtype=object), **P),
        T("A04", "2走前の着順", "A", ("p2_fin",), lambda d: fin_bucket(d["p2_fin"]), **P),
        T("A05", "3走前の着順", "A", ("p3_fin",), lambda d: fin_bucket(d["p3_fin"]), **P),
        T("A06", "近3走の最高着順", "A", ("p_fin", "p2_fin", "p3_fin"),
          lambda d: fin_bucket(d[["p_fin", "p2_fin", "p3_fin"]].apply(_num).min(axis=1), "4"), **P),
        T("A07", "近5走の3着内の回数", "A", ("r5_top3",), lambda d: cut(d["r5_top3"], [0, 1, 2], ["0回", "1回", "2回", "3回以上"]), **P),
        T("A08", "連勝中", "A", ("win_streak",), lambda d: cut(d["win_streak"], [0, 1], ["前走負け", "前走勝ち（1連勝）", "2連勝以上"]), **P),
        T("A09", "前走で着順を上げたか", "A", ("p_fin", "p2_fin"), lambda d: cut(_num(d["p2_fin"]) - _num(d["p_fin"]), [-3, 2],
          ["着順を下げた（3つ以上）", "横ばい", "着順を上げた（3つ以上）"]), **P),
        T("A10", "前走のレースレベル", "A", ("lvl_mates_won",), lambda d: cut(d["lvl_mates_won"], [0, 2], ["同走馬のその後の勝ち 0頭", "1-2頭", "3頭以上"]), **P,
          note="前走で一緒に走った馬のうち、前走の翌日から今回の前日までに勝った頭数"),
        # B 前走のクラス・レース名
        T("B01", "前走のクラス", "B", ("p_cls",), _p_cls, **P),
        T("B02", "前走のクラス × 着順", "B", ("p_cls", "p_fin"), lambda d: combo(_p_cls(d), fin_bucket(d["p_fin"], "4")), **P),
        T("B03", "格の上げ下げ", "B", ("cls", "p_cls"), lambda d: cut(_num(d["cls"]) - _num(d["p_cls"]), [-1, 0], ["格下げ", "同格", "格上げ"]), **P),
        T("B04", "前走のレース名（ステップの組）", "B", ("p_cls", "p_rkey"), _b04, **P),
        T("B05", "前走のレース名 × 着順", "B", ("p_cls", "p_rkey", "p_fin"), lambda d: combo(_b04(d), fin_bucket(d["p_fin"], "3")), **P),
        T("B06", "前走が同じ重賞の前年の回", "B", ("p_cls", "p_rkey", "rkey"), lambda d: when(_num(d["p_cls"]).notna(), pd.Series(
            np.where((d["p_rkey"].astype(str) == d["rkey"].astype(str)) & (_num(d["p_cls"]) >= 6), "前走が前年の同じ重賞", "それ以外"), index=d.index, dtype=object)), **P),
        T("B07", "前走が海外・地方", "B", ("prev_region",), lambda d: d["prev_region"].astype(object).fillna(UNKNOWN), start_year=2019,
          note="出馬表の過去 5 走（2019 年〜）"),
        T("B08", "前走が古馬との混合戦（3歳）", "B", ("p_mixed", "age"), lambda d: when(_num(d["age"]) == 3, pd.Series(
            np.where(truthy(d["p_mixed"]), "前走 古馬との混合", "前走 3歳限定"), index=d.index, dtype=object).where(d["p_mixed"].notna() & (d["p_mixed"].astype(str) != "<NA>"), UNKNOWN)), **P),
        T("B09", "前走が牝馬限定（牝馬）", "B", ("p_fillies", "sex"), lambda d: when(d["sex"].astype(str) == "牝", pd.Series(
            np.where(truthy(d["p_fillies"]), "前走 牝馬限定", "前走 混合"), index=d.index, dtype=object).where(d["p_fillies"].notna() & (d["p_fillies"].astype(str) != "<NA>"), UNKNOWN)), **P),
        # C 前走からの条件の変化
        T("C01", "距離の変化", "C", ("dist", "p_dist"), _dist_chg, **P),
        T("C02", "芝ダの替わり", "C", ("surf", "p_surf"), _surf_chg, **P),
        T("C03", "会場の替わり", "C", ("vc", "p_vc", "direction", "p_direction"), _rotation, **P),
        T("C04", "内外回りの替わり", "C", ("io", "p_io"), _io_chg, **P),
        T("C05", "前走の馬場", "C", ("p_track",), lambda d: _track(d["p_track"]), **P),
        T("C06", "斤量の変化", "C", ("jw", "p_jw"), lambda d: cut(_num(d["jw"]) - _num(d["p_jw"]), [-2, 1.5], ["斤量 2kg以上減", "斤量 ±1kg", "斤量 2kg以上増"]), **P),
        T("C07", "経験した距離との差", "C", ("dist", "dist_max_prev", "dist_min_prev"), _dist_exp, **P),
        T("C08", "好走した距離の範囲", "C", ("dist", "best_dist_min", "best_dist_max", "starts_prev"), _best_range, **P),
        T("C09", "前走のペース", "C", ("p_pace2",), lambda d: d["p_pace2"].astype(object).map({"H": "前走ハイペース", "M": "前走ミドル", "S": "前走スロー"}).fillna(UNKNOWN), **P),
        T("C10", "前走の頭数", "C", ("p_field",), lambda d: cut(d["p_field"], [10, 14], ["前走 10頭以下", "前走 11-14頭", "前走 15頭以上"]), **P),
        # D 間隔・休み明け
        T("D01", "前走からの間隔", "D", ("interval",), _interval, **P),
        T("D02", "休み明け何戦目", "D", ("since_rest",), _rest_k, **P),
        T("D03", "休み明け初戦の過去の成績", "D", ("fb_n_prev", "fb_top3_prev"), lambda d: _cnt3(d["fb_n_prev"], d["fb_top3_prev"],
          ("休み明け初戦で3着内あり", "休み明け初戦で3着内なし", "休み明けの経験なし")), **P),
        T("D04", "叩き2戦目", "D", ("since_rest",), lambda d: when(_num(d["since_rest"]).notna(), pd.Series(
            np.where(_num(d["since_rest"]) == 2, "叩き2戦目", "それ以外"), index=d.index, dtype=object)), **P),
        T("D05", "年明け初戦", "D", ("p_dt", "dt"), lambda d: when(d["p_dt"].notna(), pd.Series(
            np.where(d["p_dt"].dt.year < d["dt"].dt.year, "年明け初戦", "それ以外"), index=d.index, dtype=object)), **P),
        # E 前走の脚質・位置・上がり
        T("E01", "前走の脚質", "E", ("p_pos1", "p_pos1r"), _style, **P),
        T("E02", "前走の4角の位置", "E", ("p_pos4r",), lambda d: cut(d["p_pos4r"], [0.2, 0.5, 0.8], ["4角 前20%", "4角 前半", "4角 後半", "4角 後方20%"]), **P),
        T("E03", "前走の位置の上げ下げ", "E", ("p_pos1", "p_pos4"), lambda d: cut(_num(d["p_pos1"]) - _num(d["p_pos4"]), [-3, 2],
          ["位置を下げた（3つ以上）", "位置は横ばい", "位置を上げた（3つ以上）"]), **P),
        T("E04", "前走の上がり順位", "E", ("p_l3rank",), _l3, **P),
        T("E05", "前走の上がり順位 × 着順", "E", ("p_l3rank", "p_fin"), lambda d: combo(cut(d["p_l3rank"], [1, 3], ["上がり1位", "上がり2-3位", "上がり4位以下"]),
          fin_bucket(d["p_fin"], "2")), **P),
        T("E06", "前走の脚質 × 着順", "E", ("p_pos1", "p_pos1r", "p_fin"), lambda d: combo(_style(d), fin_bucket(d["p_fin"], "3b")), **P),
        T("E07", "通算の逃げ率", "E", ("nige_rate",), lambda d: cut(d["nige_rate"], [0, 0.2, 0.5], ["逃げたことなし", "逃げ率 20%以下", "逃げ率 20-50%", "逃げ率 50%超"]), **P),
        T("E08", "通算の先行率", "E", ("senko_rate",), lambda d: cut(d["senko_rate"], [0.2, 0.5], ["先行率 20%以下", "先行率 20-50%", "先行率 50%超"]), **P),
        T("E09", "上がり最速の回数", "E", ("agari1_prev",), lambda d: cut(d["agari1_prev"], [0, 1], ["上がり最速 0回", "上がり最速 1回", "上がり最速 2回以上"]), **P),
        T("E10", "前走のペース × 位置 × 着順", "E", ("p_pace2", "p_pos1", "p_pos1r", "p_fin"), lambda d: combo(
            d["p_pace2"].astype(object).map({"H": "H", "M": "M", "S": "S"}).fillna(UNKNOWN), _style2(d), fin_bucket(d["p_fin"], "2b")), **P),
        T("E11", "今回の逃げ候補", "E", ("nige_cand", "n_nige_cand"), _nige, **P),
        # F 馬の経歴
        T("F01", "同じコースの実績", "F", ("course_n_prev", "course_top3_prev"), lambda d: _cnt3(d["course_n_prev"], d["course_top3_prev"]), **H),
        T("F02", "同じ会場・芝ダの実績", "F", ("vs_n_prev", "vs_top3_prev"), lambda d: _cnt3(d["vs_n_prev"], d["vs_top3_prev"]), **H),
        T("F03", "同じ距離帯・芝ダの実績", "F", ("dbs_n_prev", "dbs_top3_prev"), lambda d: _cnt3(d["dbs_n_prev"], d["dbs_top3_prev"]), **H),
        T("F04", "同じ回りの実績", "F", ("dir_n_prev", "dir_top3_prev"), lambda d: _cnt3(d["dir_n_prev"], d["dir_top3_prev"]), **H),
        T("F05", "道悪の実績", "F", ("bad_n_prev", "bad_top3_prev"), lambda d: _cnt3(d["bad_n_prev"], d["bad_top3_prev"]), **H),
        T("F06", "重賞の実績", "F", ("g_n_prev", "g_top3_prev", "g_win_prev"), lambda d: pd.Series(np.select(
            [_num(d["g_win_prev"]) > 0, _num(d["g_top3_prev"]) > 0, _num(d["g_n_prev"]) > 0], ["重賞勝ちあり", "重賞3着内あり", "重賞は出走のみ"], "重賞は未経験"),
            index=d.index, dtype=object), **H),
        T("F07", "G1の実績", "F", ("g1_n_prev", "g1_top3_prev"), lambda d: _cnt3(d["g1_n_prev"], d["g1_top3_prev"], ("G1 3着内あり", "G1 出走のみ", "G1 未経験")), **H),
        T("F08", "同じ重賞の実績（リピーター）", "F", ("rep_n_prev", "rep_top3_prev"), lambda d: _cnt3(d["rep_n_prev"], d["rep_top3_prev"],
          ("同じ重賞で3着内あり", "同じ重賞は出走のみ", "同じ重賞は初出走")), **H),
        T("F09", "キャリア（出走数）", "F", ("starts_prev",), lambda d: cut(d["starts_prev"], [5, 10, 20], ["キャリア 5戦以下", "6-10戦", "11-20戦", "21戦以上"]), **H),
        T("F10", "通算の複勝率", "F", ("top3_prev", "starts_prev"), lambda d: when(_num(d["starts_prev"]) >= 3, cut(
            _num(d["top3_prev"]) / _num(d["starts_prev"]), [0.2, 0.4, 0.6], ["通算複勝率 20%以下", "20-40%", "40-60%", "60%超"])), **H),
        T("F11", "勝ち数", "F", ("wins_prev",), lambda d: cut(d["wins_prev"], [1, 3, 5], ["0-1勝", "2-3勝", "4-5勝", "6勝以上"]), **H),
        T("F12", "古馬との対戦経験（3歳）", "F", ("vs_old_prev", "age"), lambda d: when(_num(d["age"]) == 3, pd.Series(
            np.where(_num(d["vs_old_prev"]) > 0, "古馬との対戦あり", "古馬との対戦なし"), index=d.index, dtype=object)), **H),
        T("F13", "経験した最多頭数", "F", ("max_field_prev",), lambda d: cut(d["max_field_prev"], [12, 15], ["最多 12頭以下", "最多 13-15頭", "最多 16頭以上"]), **H),
        T("F14", "同じ会場の経験", "F", ("venue_n_prev",), lambda d: pd.Series(np.where(_num(d["venue_n_prev"]) > 0, "同じ会場の経験あり", "同じ会場は初めて"),
          index=d.index, dtype=object), **H),
        T("F15", "重賞・OPの経験数", "F", ("op_n_prev",), lambda d: cut(d["op_n_prev"], [0, 2, 5], ["OP以上 0戦", "1-2戦", "3-5戦", "6戦以上"]), **H),
        # G 馬の属性
        T("G01", "年齢", "G", ("age",), _age),
        T("G02", "性別", "G", ("sex",), lambda d: d["sex"].astype(object).fillna(UNKNOWN)),
        T("G03", "年齢 × 性別", "G", ("age", "sex"), lambda d: combo(_age(d), d["sex"].astype(object).fillna(UNKNOWN))),
        T("G04", "セン馬になってからの戦数", "G", ("since_geld",), lambda d: cut(d["since_geld"], [1, 3], ["去勢後1戦目", "去勢後2-3戦目", "去勢後4戦目以上"]).replace(UNKNOWN, "去勢なし"), start_year=2018),
        T("G05", "所属と遠征", "G", ("tozai", "away", "venue"), _away),
        T("G06", "馬体重", "G", ("wt",), lambda d: cut(d["wt"], [439, 479, 519], ["439kg以下", "440-479kg", "480-519kg", "520kg以上"]), day_of=True),
        T("G07", "馬体重の増減", "G", ("wtc",), lambda d: cut(d["wtc"], [-10, -4, 3, 9], ["-10kg以下", "-9〜-4kg", "-3〜+3kg", "+4〜+9kg", "+10kg以上"]), day_of=True),
        T("G08", "ブリンカー", "G", ("blinker", "p_blinker", "blk_ever_prev"), _blinker, start_year=2020, note="SmartRC（2020 年〜）"),
        T("G09", "転厩明けの戦数", "G", ("since_transfer",), lambda d: cut(d["since_transfer"], [1, 3], ["転厩後1戦目", "転厩後2-3戦目", "転厩後4戦目以上"]).replace(UNKNOWN, "転厩なし"), start_year=2018),
        # H 今回の条件
        T("H01", "枠", "H", ("bracket_number",), lambda d: cut(d["bracket_number"], [2, 4, 6], ["1-2枠", "3-4枠", "5-6枠", "7-8枠"])),
        T("H02", "馬番の位置", "H", ("hn", "field_decl"), lambda d: cut(_num(d["hn"]) / _num(d["field_decl"]), [1 / 3, 2 / 3], ["内（馬番 1/3まで）", "中", "外（馬番 2/3より外）"])),
        T("H03", "斤量", "H", ("jw",), lambda d: cut(d["jw"], [54.5, 56.5, 58.5], ["54kg以下", "55-56kg", "57-58kg", "59kg以上"])),
        T("H04", "斤量のレース内の順位", "H", ("jw_rank", "jw_max"), _jw_rank),
        T("H06", "頭数", "H", ("field_decl",), lambda d: cut(d["field_decl"], [12, 15], ["12頭以下", "13-15頭", "16頭以上"])),
        T("H07", "馬場状態", "H", ("track",), lambda d: _track(d["track"]), day_of=True),
        T("H08", "開催の何日目", "H", ("meet_day",), _meet),
        T("H09", "季節", "H", ("month",), _season),
        T("H10", "クッション値", "H", ("cush", "surf"), lambda d: when(d["surf"].astype(str) == "芝", cut(d["cush"], [8.5, 9.9], ["クッション 8.5以下", "8.6-9.9", "10以上"])),
          day_of=True, start_year=2018),
        # I 騎手
        T("I01", "騎手の格（直近1年の複勝率）", "I", ("j_rate",), lambda d: when(_num(d["j_rate"]).notna(), cut(
            d["j_rate"], [float(_num(d.loc[_ref(d), "j_rate"]).quantile(q)) for q in (0.4, 0.7, 0.9)], ["騎手 下位", "騎手 中位", "騎手 上位", "騎手 トップ10%"]),
            "騎乗が少ない"), **H),
        T("I02", "騎手（個人）", "I", ("jockey_id", "jockey_name"), lambda d: d["jockey_name"].astype(object).fillna(UNKNOWN)),
        T("I03", "継続騎乗・乗り替わり", "I", ("j_cont", "starts_prev"), _jockey_cont),
        T("I04", "乗り替わりの格の上げ下げ", "I", ("j_rate", "pj_rate", "j_cont", "starts_prev"), _j_switch),
        T("I05", "外国人騎手・短期免許", "I", ("j_foreign", "j_short"), lambda d: pd.Series(np.select(
            [truthy(d["j_short"]), truthy(d["j_foreign"])], ["短期免許の外国人", "通年の外国人"], "日本人"), index=d.index, dtype=object)),
        T("I06", "騎手のコース実績", "I", ("jc_n", "jc_y3"), lambda d: rate_bucket(d["jc_n"], d["jc_y3"], 20)),
        T("I07", "騎手の重賞実績", "I", ("jgr_n", "jgr_y3"), lambda d: rate_bucket(d["jgr_n"], d["jgr_y3"], 20)),
        T("I08", "騎手の逃げの多さ", "I", ("j_nige_rate",), lambda d: when(_num(d["j_nige_rate"]).notna(), cut(
            d["j_nige_rate"], [float(_num(d.loc[_ref(d), "j_nige_rate"]).quantile(q)) for q in (0.2, 0.8)], ["逃げの少ない騎手", "逃げ 中", "逃げの多い騎手"]),
            "騎乗が少ない")),
        T("I09", "騎手のその日の勝ち数", "I", ("jday_w_before",), lambda d: cut(d["jday_w_before"], [0, 1], ["その日 0勝", "その日 1勝", "その日 2勝以上"]), day_of=True),
        # J 調教師・コンビ
        T("J01", "厩舎の格（直近1年の複勝率）", "J", ("t_rate",), lambda d: when(_num(d["t_rate"]).notna(), cut(
            d["t_rate"], [float(_num(d.loc[_ref(d), "t_rate"]).quantile(q)) for q in (0.4, 0.7, 0.9)], ["厩舎 下位", "厩舎 中位", "厩舎 上位", "厩舎 トップ10%"]),
            "出走が少ない"), **H),
        T("J02", "厩舎の勢い（直近30日）", "J", ("t30_n", "t30_y3"), lambda d: rate_bucket(d["t30_n"], d["t30_y3"], 10)),
        T("J03", "厩舎のコース実績", "J", ("tc_n", "tc_y3"), lambda d: rate_bucket(d["tc_n"], d["tc_y3"], 20)),
        T("J04", "厩舎の重賞実績", "J", ("tgr_n", "tgr_y3"), lambda d: rate_bucket(d["tgr_n"], d["tgr_y3"], 10)),
        T("J05", "厩舎の休み明けの成績（休み明けの馬）", "J", ("tfb_n", "tfb_y3", "since_rest"), lambda d: when(_num(d["since_rest"]) == 1,
          rate_bucket(d["tfb_n"], d["tfb_y3"], 10))),
        T("J06", "騎手 × 調教師のコンビ", "J", ("jt_n", "jt_y3"), lambda d: rate_bucket(d["jt_n"], d["jt_y3"], 10)),
        T("J07", "同じ厩舎からの複数出し", "J", ("n_same_tr",), lambda d: cut(d["n_same_tr"], [1], ["単騎", "同じ厩舎から2頭以上"])),
        # K 血統
        T("K01", "父の大系統", "K", ("f_large",), _fl),
        T("K02", "父の小系統", "K", ("f_small",), lambda d: d["f_small"].astype(object).fillna(UNKNOWN)),
        T("K03", "母父の大系統", "K", ("mf_large",), lambda d: d["mf_large"].astype(object).fillna(UNKNOWN)),
        T("K04", "母父の小系統", "K", ("mf_small",), lambda d: d["mf_small"].astype(object).fillna(UNKNOWN)),
        T("K05", "父 × 母父の大系統", "K", ("f_large", "mf_large"), lambda d: combo(_fl(d), d["mf_large"].astype(object).fillna(UNKNOWN))),
        T("K06", "父（種牡馬）", "K", ("f_name",), lambda d: d["f_name"].astype(object).fillna(UNKNOWN)),
        T("K07", "父の産駒の同じコースの成績", "K", ("sc_n", "sc_y3"), lambda d: rate_bucket(d["sc_n"], d["sc_y3"], 30)),
        T("K08", "父の産駒の同じ芝ダ・距離帯の成績", "K", ("sd_n", "sd_y3"), lambda d: rate_bucket(d["sd_n"], d["sd_y3"], 30)),
        # L 調教
        T("L01", "最終追い切りの評価", "L", ("ok_eval",), _eval),
        # X 組み合わせ（先に固定した 15 組）
        T("X01", "前走の脚質 × 枠", "X", ("p_pos1", "p_pos1r", "bracket_number"), lambda d: combo(_style2(d), cut(d["bracket_number"], [4], ["1-4枠", "5-8枠"])), **P),
        T("X02", "間隔 × 年齢", "X", ("interval", "age"), lambda d: combo(_interval(d), cut(d["age"], [3, 5], ["2-3歳", "4-5歳", "6歳以上"])), **P),
        T("X03", "父の大系統 × 馬場", "X", ("f_large", "track"), lambda d: combo(_fl(d), _track(d["track"]).replace({"稍重": "稍重以下", "重・不良": "稍重以下"})), day_of=True),
        T("X04", "距離の変化 × 前走の上がり順位", "X", ("dist", "p_dist", "p_l3rank"), lambda d: combo(
            cut(_num(d["dist"]) - _num(d["p_dist"]), [-100, 99], ["短縮", "同距離", "延長"]), cut(d["p_l3rank"], [3], ["上がり3位以内", "上がり4位以下"])), **P),
        T("X05", "前走の脚質 × 今回の逃げ候補の数", "X", ("p_pos1", "p_pos1r", "n_nige_cand"), lambda d: combo(_style2(d), cut(d["n_nige_cand"], [1], ["候補0-1頭", "候補2頭以上"])), **P),
        T("X06", "性別 × 季節", "X", ("sex", "month"), lambda d: combo(d["sex"].astype(object).fillna(UNKNOWN), _season(d))),
        T("X07", "年齢 × 前走のクラス", "X", ("age", "p_cls"), lambda d: combo(_age(d), _p_cls(d)), **P),
        T("X08", "乗り替わり × 騎手の格", "X", ("j_cont", "starts_prev", "j_rate"), lambda d: combo(
            _jockey_cont(d).replace({"継続2戦目": "継続", "継続3戦以上": "継続"}), when(_num(d["j_rate"]).notna(), cut(
                d["j_rate"], [float(_num(d.loc[_ref(d), "j_rate"]).quantile(0.7))], ["騎手 上位30%以外", "騎手 上位30%"]), "騎乗が少ない")), **H),
        T("X09", "休み明け × 厩舎の休み明けの成績", "X", ("since_rest", "tfb_n", "tfb_y3"), lambda d: combo(
            _rest_k(d).replace({"休み明け3戦目": "3戦目以上", "休み明け4戦目以上": "3戦目以上"}),
            rate_bucket(d["tfb_n"], d["tfb_y3"], 10, edges=(0.22,), labels=("休み明けの厩舎 3着内率 22%未満", "22%以上"))), **P),
        T("X10", "芝ダの替わり × 父の大系統", "X", ("surf", "p_surf", "f_large"), lambda d: combo(_surf_chg(d), _fl(d)), **P),
        T("X11", "年齢 × 重賞の実績", "X", ("age", "g_n_prev", "g_top3_prev"), lambda d: combo(cut(d["age"], [3, 5], ["2-3歳", "4-5歳", "6歳以上"]),
          _cnt3(d["g_n_prev"], d["g_top3_prev"], ("重賞3着内あり", "重賞3着内なし", "重賞未経験"))), **H),
        T("X12", "馬番の位置 × 頭数", "X", ("hn", "field_decl"), lambda d: combo(cut(_num(d["hn"]) / _num(d["field_decl"]), [1 / 3, 2 / 3], ["内", "中", "外"]),
          cut(d["field_decl"], [12, 15], ["12頭以下", "13-15頭", "16頭以上"]))),
        T("X13", "距離の変化 × 父の大系統", "X", ("dist", "p_dist", "f_large"), lambda d: combo(cut(_num(d["dist"]) - _num(d["p_dist"]), [-100, 99], ["短縮", "同距離", "延長"]), _fl(d)), **P),
        T("X14", "前走のクラス × 間隔", "X", ("p_cls", "interval"), lambda d: combo(_p_cls(d), cut(d["interval"], [35, 63], ["中4週以内", "中5-8週", "中9週以上"])), **P),
        T("X15", "前走の上がり順位 × 脚質", "X", ("p_l3rank", "p_pos1", "p_pos1r"), lambda d: combo(cut(d["p_l3rank"], [3], ["上がり3位以内", "上がり4位以下"]), _style2(d)), **P),
        # M 比較の版だけ（めぐ指数の順位）
        T("M01", "前走のめぐ指数のレース内順位", "M", ("mg_rank_p1",), lambda d: cut(d["mg_rank_p1"], [1, 3, 6], ["1位", "2-3位", "4-6位", "7位以下"]),
          start_year=2020, optional=True),
        T("M02", "近5走の最高めぐ指数のレース内順位", "M", ("mg_rank_max5",), lambda d: cut(d["mg_rank_max5"], [1, 3, 6], ["1位", "2-3位", "4-6位", "7位以下"]),
          start_year=2020, optional=True),
    ]


TYPES: list[CutType] = _types()
BY_ID = {t.id: t for t in TYPES}
ALLOWED_COLUMNS: frozenset[str] = frozenset(c for t in TYPES for c in t.cols)

# 作れない・入れない切り口（HTML の「使わなかった切り口」に載せる）
NOT_BUILT = [
    ("H05", "減量・恩恵", "重賞では全行 0（馬齢・定量戦は決まった斤量どおりで、減量騎手も乗れない）"),
    ("K09", "母母父・母母母父の系統", "2026-11-13 の分析で形質への寄与がほぼ 0"),
    ("L02", "追い切りのコース", "2020 年以降のデータがほぼ空"),
    ("L03", "追い切りの強さ", "2020 年以降 0%"),
    ("L04", "追い切りのラップ・時計", "2016〜2019 だけ"),
    ("—", "前走の人気・今回の人気・オッズの帯", "市場の情報（ユーザーの指定で禁止）"),
    ("—", "前走の不利・出遅れ", "レース結果に備考の列が無い"),
    ("—", "パドックの評価", "発走の直前にしか出ない。2020 年〜だけ"),
    ("—", "兄弟の重賞実績", "母の ID の確認が済むまで保留"),
]


def levels(d: pd.DataFrame, types: list[CutType] | None = None) -> pd.DataFrame:
    """全型の水準の表（行は d と同じ、列は型の id）。start_year より前の走と、前走が海外・地方の走の前走系は UNKNOWN / PREV_FOREIGN。"""
    types = types or TYPES
    out = {}
    year = _num(d["year"])
    foreign = d["prev_region"].astype(object).isin(["海外", "地方"]) if "prev_region" in d else pd.Series(False, index=d.index)
    for t in types:
        s = t.fn(d).astype(object).fillna(UNKNOWN)
        if t.prev:
            s = s.where(~foreign, PREV_FOREIGN)
        s = s.where(year >= t.start_year, UNKNOWN)
        out[t.id] = s.astype(str)
    return pd.DataFrame(out, index=d.index)
