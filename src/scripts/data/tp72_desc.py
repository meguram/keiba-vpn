"""tp-72 記述子: 第 2 段（ペア走査）と第 C 部（穴馬の好走の共通要素）で使う、解釈できる記述子の一覧（事前に固定）。

各記述子 = 水準が 2〜5 個の分類（整数コード 0..k-1、欠測は -1）。市場オッズ・人気は含めない（穴馬の定義と層別には別に使う）。
閾値は慣習値で固定し、結果を見て変えない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLS = ["c0_newcomer", "c1_maiden", "c2_1win", "c3_2win", "c4_3win", "op", "g3", "g2", "g1"]


def _cut(x: pd.Series, edges: list[float]) -> np.ndarray:
    """edges で区切った水準コード（x <= edges[0] → 0, ...）。欠測は -1。"""
    v = pd.to_numeric(x, errors="coerce").to_numpy(float)
    code = np.searchsorted(np.array(edges), v, side="left")
    return np.where(np.isnan(v), -1, code).astype(int)


def _map(x: pd.Series, mapping: dict) -> np.ndarray:
    return x.map(mapping).fillna(-1).astype(int).to_numpy()


def descriptors(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    D: dict[str, np.ndarray] = {}
    L: dict[str, list[str]] = {}

    def put(name: str, codes: np.ndarray, labels: list[str]) -> None:
        D[name] = np.asarray(codes, dtype=int)
        L[name] = labels

    put("sex", _map(d["sex"], {"牡": 0, "牝": 1, "セ": 2}), ["牡", "牝", "セ"])
    put("age", _cut(d["age"], [2, 3, 4]), ["2歳以下", "3歳", "4歳", "5歳以上"])
    put("season", _map(d["season"], {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}), ["春", "夏", "秋", "冬"])
    put("venue", _map(d["venue_code"].astype(str), {f"{i:02d}": i - 1 for i in range(1, 11)}), ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"])
    put("local", d["is_local"].fillna(-1).astype(int).to_numpy(), ["中央場", "ローカル場"])
    put("surface", d["is_turf"].fillna(-1).astype(int).to_numpy(), ["ダート", "芝"])
    put("dist", _map(d["dist_band"], {"d_sprint": 0, "d_mile_s": 1, "d_mile": 1, "d_mid": 2, "d_long": 2}), ["短距離", "マイル", "中長距離"])
    put("field", _cut(d["fs"], [10, 15]), ["10頭以下", "11-15頭", "16頭以上"])
    put("draw", np.where(d["inner"] == 1, 0, np.where(d["outer"] == 1, 2, np.where(d["bracket_number"].notna(), 1, -1))), ["内枠", "中枠", "外枠"])
    put("style4", _map(d["style4"], {"nige": 0, "senko": 1, "sashi": 2, "oikomi": 3}), ["逃げ", "先行", "差し", "追込"])
    put("pace_front", _cut(d["n_front"], [1, 2]), ["先行馬0-1頭", "2頭", "3頭以上"])
    put("cushion", d["cush_bin"].fillna(-1).astype(int).to_numpy(), ["芝クッション低", "中", "高"])
    put("dirt_moist", d["dmoist_bin"].fillna(-1).astype(int).to_numpy(), ["ダート含水低", "中", "高"])
    put("going", _map(d["track_condition"], {"良": 0, "稍重": 1, "重": 2, "不良": 2}), ["良", "稍重", "重・不良"])
    lay = _cut(d["lay_days"], [13, 62, 119])
    put("lay", np.where(d["run_no"] == 0, 0, lay + 1), ["初出走", "連闘〜中1週", "2〜8週", "9〜17週", "18週以上"])
    put("weight_chg", _cut(d["weight_change"], [-10 + 1e-9, -5 + 1e-9, 4, 9]), ["-10以下", "-9〜-5", "±4", "+5〜+9", "+10以上"])
    put("weight_dev", _cut(d["wt_dev"], [-8 + 1e-9, 7]), ["自己平均より軽い", "平均並み", "重い"])
    put("train_eval", _cut(d["ev_score"], [2, 3]), ["追切C以下", "B", "A"])
    put("train_delta", _cut(d["ev_delta"], [-1, 0]), ["前走より下", "同じ", "上"])
    put("comment", np.where(d["comment_pos"] == 1, 1, np.where(d["comment_neg"] == 1, 2, 0)), ["コメントなし/中立", "前向き語", "後ろ向き語"])
    jr = np.where(d["ja_one"] >= 100, d["ja_top3f"] / d["ja_one"], np.nan)
    put("jockey_rate", _cut(pd.Series(jr), [0.18, 0.30]), ["騎手複勝率低", "中", "高"])
    tr = np.where(d["ta_one"] >= 150, d["ta_top3f"] / d["ta_one"], np.nan)
    put("trainer_rate", _cut(pd.Series(tr), [0.16, 0.26]), ["調教師複勝率低", "中", "高"])
    put("chg_jockey", d["chg_jockey"].fillna(-1).astype(int).to_numpy(), ["継続騎乗", "乗り替わり"])
    put("prev_fin", _cut(d["prev_fin"], [1, 3, 5, 9]), ["前走1着", "2-3着", "4-5着", "6-9着", "10着以下"])
    put("prev_pop", _cut(d["prev_pop"], [3, 9]), ["前走1-3人気", "4-9人気", "10人気以下"])
    put("prev_agari", _cut(d["prev_agari_rank"], [2, 5]), ["前走上がり1-2位", "3-5位", "6位以下"])
    put("prev_late", _cut(d["prev_posL_frac"], [0.3, 0.7]), ["前走4角前方", "中団", "後方"])
    put("class", _map(d["cls"], {k: i for i, k in enumerate(CLS)}), ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "G3", "G2", "G1"])
    put("run_no", _cut(d["run_no"], [0, 3, 9]), ["初出走", "2-4走目", "5-10走目", "11走目以降"])
    put("first_surface", d["first_surface"].fillna(-1).astype(int).to_numpy(), ["経験あり", "初芝/初ダート"])
    put("dist_chg", _cut(d["dist_chg"], [-200 + 1e-9, 199]), ["200m以上短縮", "±199m", "200m以上延長"])
    put("jw_chg", _cut(d["jw_chg"], [-2 + 1e-9, 1]), ["斤量2kg以上減", "±1kg", "2kg以上増"])
    put("blinker", _map(d["blk_state"], {"none": 0, "first": 1, "cont": 2, "off": 3, "again": 4}), ["着用なし", "初着用", "着用継続", "外した初戦", "再着用"])
    put("meeting_day", _cut(d["day_no"], [2, 6]), ["開催1-2日目", "3-6日目", "7日目以降"])
    put("seas_aff", np.where(d["seas_aff"].isna(), -1, np.where(d["seas_aff"] >= 0.25, 1, np.where(d["seas_aff"] <= -0.25, 2, 0))), ["同季節の実績並み", "同季節に強い", "同季節に弱い"])
    put("pincer_d1", np.where(d["d1_self_nl"] != 1, -1, np.where(d["d1_pin"] == 1, 2, np.where(d["d1_p1"] == 1, 1, 0))), ["隣に先行馬なし", "片隣が先行", "両隣が先行"])
    t_ = (d["t_local_aff"] >= 0.05).astype(int)
    j_ = (d["j_local_aff"] >= 0.05).astype(int)
    put("local_aff", np.where(d["t_local_aff"].isna() & d["j_local_aff"].isna(), -1, t_ + 2 * j_), ["場実績なし", "調教師", "騎手", "両方"])
    pf = d["pref_hard"]
    put("cush_pref", np.where(pf.isna(), -1, np.where(pf >= 0.15, 1, np.where(pf <= -0.15, 2, 0))), ["馬場好み不明確", "硬い馬場が得意", "柔らかい馬場が得意"])
    put("g1_week", np.where(d["is_g1_day"] == 1, 1, np.where(d["d_before_g1"] == 1, 2, np.where(d["d_after_g1"] == 1, 3, 0))), ["通常週", "G1開催日", "G1の前週", "G1の翌週"])
    return pd.DataFrame(D, index=d.index), L


N_DESCRIPTORS_EXPECTED = 39
