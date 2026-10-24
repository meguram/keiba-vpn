"""特徴量ファクトリー G4: 厩舎コメント・追い切り（評価・脚色・コース・馬場）の文字/テキストの特徴。

厩舎コメント（race_trainer_comment）はレース前に公開される約 40 字の談話。結果を示す語（prerace_extras.RESULT_WORDS）を含む行は欠損にする。
語彙の変換（TF-IDF → SVD）は教師なし（着順を使わない）で、全期間のコメントに当てはめる。
生のコメントは保存しない（特徴量は数値だけ）。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.pipeline.features import factory_src as fs
from src.pipeline.features.prerace_extras import NEG, POS, RESULT_WORDS

POS2 = POS + ("好気配", "いい動き", "動きは良", "気配は良", "上々", "充実", "リラックス", "落ち着", "集中", "スムーズ", "軽快", "鋭", "力強", "feel", "成長", "パワーアップ", "文句", "満点", "万全",
              "太め残り", "体が絞れ", "絞れ", "ひと息入って", "息が入", "反応", "素軽", "活気", "闘争心", "前向き", "気合", "気持ち", "いい状態", "状態は良", "デキは")
NEG2 = NEG + ("硬い", "かたい", "遅い", "重め", "余裕", "疲れ", "ひと息", "気が悪", "ムキ", "いれ込", "イレ込", "カリカリ", "テンション", "怖い", "掛かる", "かかる", "モタ", "ズル", "一息", "足りな", "今ひとつ", "いま一つ", "いまひとつ",
              "少し", "ややも", "渋った", "不向き", "クセ", "弱", "キツ", "厳しい", "ハンデ", "相手", "強敵", "メンバーが", "試金石")
KEYWORDS = {
    "kw_dist_ext": ("延長", "距離延び"), "kw_dist_cut": ("短縮",), "kw_dist_suit": ("距離", "ベスト"), "kw_first_turf": ("芝替わり", "芝に替", "芝は初", "初芝"), "kw_first_dirt": ("ダート替わり", "ダートに替", "ダート初", "初ダート"),
    "kw_return": ("休み明け", "久々", "ひと叩き", "叩き2", "叩き良化", "放牧"), "kw_weight": ("馬体", "体重", "太", "絞", "細"), "kw_ground": ("馬場", "道悪", "渋", "良馬場", "雨", "湿"),
    "kw_gate": ("外枠", "内枠", "枠", "ゲート", "出遅れ"), "kw_jockey": ("鞍上", "騎手", "ジョッキー", "乗り替わり", "手が合", "コンビ", "乗り慣れ"), "kw_course": ("コース", "左回り", "右回り", "小回り", "広い", "直線", "坂", "平坦", "内回り", "外回り"),
    "kw_pace": ("ペース", "流れ", "展開"), "kw_front": ("逃げ", "先行", "好位", "前に"), "kw_close": ("差し", "追い込", "末脚", "切れ", "脚を使"), "kw_class": ("クラス", "格上", "昇級", "相手強", "メンバー", "相手関係", "頭打ち", "降級"),
    "kw_condition": ("状態", "デキ", "調子", "仕上"), "kw_train": ("追い切り", "調教", "動き", "稽古", "時計", "一杯", "併せ"), "kw_growth": ("成長", "良化", "上向", "体つき", "筋肉", "トモ"),
    "kw_mind": ("気性", "テンション", "気合", "リラックス", "落ち着", "ムキ", "折り合い", "掛か", "かか"), "kw_win": ("勝ち", "勝て", "勝負", "チャンス", "ねらい", "狙"), "kw_hope": ("期待", "楽しみ", "ほしい", "ほしかった"),
    "kw_cond_if": ("なら", "れば", "たら", "次第", "条件", "ハマ", "かみ合"), "kw_exclaim": ("！", "!"), "kw_question": ("？", "?"),
}


def clean_comment(s: pd.Series) -> pd.Series:
    """本文だけにする（質問者の名前・文末の〈師〉名・先頭の Ｑ を除く）。結果を示す語を含む行は欠損。"""
    t = s.astype(str).str.strip().replace({"": None, "None": None, "nan": None})
    t = t.str.replace(r"〈[^〉]*〉", "", regex=True).str.replace(r"^Ｑ[　 ]*", "", regex=True)
    has_res = t.fillna("").map(lambda x: any(w in x for w in RESULT_WORDS))
    return t.where(~has_res)


def load_comments() -> pd.DataFrame:
    t = fs.read_flat("race_trainer_comment_flat", ["race_id", "horse_id", "comment", "questioner"])
    t = t[t["horse_id"].notna()].copy()
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    t["has_q"] = t["comment"].astype(str).str.startswith("Ｑ").astype(float)
    t["text"] = clean_comment(t["comment"])
    return t.drop_duplicates(["race_id", "horse_id"], keep="last")


def txt_tc_lex(B: pd.DataFrame, C: pd.DataFrame | None = None) -> pd.DataFrame:
    C = load_comments() if C is None else C
    m = B[["race_id", "horse_id"]].astype(str).merge(C[["race_id", "horse_id", "text", "has_q"]], on=["race_id", "horse_id"], how="left")
    txt = m["text"].fillna("")
    ok = m["text"].notna()
    out = pd.DataFrame(index=B.index)
    out["tc_len"] = txt.str.len().where(ok)
    out["tc_sent"] = txt.str.count("。").where(ok)
    out["tc_pos2"] = txt.map(lambda x: sum(x.count(w) for w in POS2)).where(ok)
    out["tc_neg2"] = txt.map(lambda x: sum(x.count(w) for w in NEG2)).where(ok)
    out["tc_pol"] = (out["tc_pos2"] - out["tc_neg2"])
    out["tc_has_q"] = m["has_q"].to_numpy()
    for k, ws in KEYWORDS.items():
        out[f"tc_{k}"] = txt.map(lambda x, ws=ws: float(any(w in x for w in ws))).where(ok)
    out["tc_chars_kanji"] = txt.str.count(r"[一-龥]").where(ok) / out["tc_len"]
    out["tc_chars_kata"] = txt.str.count(r"[ァ-ヶ]").where(ok) / out["tc_len"]
    out["tc_digits"] = txt.str.count(r"[0-9０-９]").where(ok)
    out.index = B.index
    return out.astype("float32")


def txt_tc_svd(B: pd.DataFrame, C: pd.DataFrame | None = None, dim: int = 8) -> pd.DataFrame:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    C = load_comments() if C is None else C
    m = B[["race_id", "horse_id"]].astype(str).merge(C[["race_id", "horse_id", "text"]], on=["race_id", "horse_id"], how="left")
    ok = m["text"].notna().to_numpy()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=40, max_features=30000, sublinear_tf=True)
    X = vec.fit_transform(m.loc[ok, "text"])
    Z = TruncatedSVD(n_components=dim, random_state=0).fit_transform(X)
    out = np.full((len(B), dim), np.nan, dtype="float32")
    out[ok] = Z
    return pd.DataFrame(out, columns=[f"tc_svd{i}" for i in range(dim)], index=B.index)


def txt_tc_change(B: pd.DataFrame, C: pd.DataFrame | None = None) -> pd.DataFrame:
    """同じ馬の前回のコメントとの類似度・極性の変化・過去の極性の平均（過去走のコメントは前走時点で公開済み）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    C = load_comments() if C is None else C
    m = B[["race_id", "horse_id"]].astype(str).merge(C[["race_id", "horse_id", "text"]], on=["race_id", "horse_id"], how="left")
    ok = m["text"].notna().to_numpy()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=20, max_features=20000, sublinear_tf=True)
    X = vec.fit_transform(m["text"].fillna(""))
    lex = txt_tc_lex(B, C)
    order = fs.horse_order(B)
    hid = B["horse_id"].astype(str).to_numpy()
    n = len(B)
    prev = np.full(n, -1)
    same = hid[order[1:]] == hid[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    # 直前の「コメントがあった」走を探す
    last_ok = np.full(n, -1)
    cur = -1
    for t, r in enumerate(order):
        if t == 0 or hid[order[t - 1]] != hid[r]:
            cur = -1
        last_ok[r] = cur
        if ok[r]:
            cur = r
    cos = np.full(n, np.nan)
    has = (last_ok >= 0) & ok
    idx = np.flatnonzero(has)
    if len(idx):
        A = X[idx]
        Bm = X[last_ok[idx]]
        cos[idx] = np.asarray(A.multiply(Bm).sum(axis=1)).ravel()
    pol_hist = fs.history_aggs(B, pd.DataFrame({"race_id": B["race_id"].astype(str), "horse_id": B["horse_id"].astype(str), "pol": lex["tc_pol"].to_numpy(), "len": lex["tc_len"].to_numpy()}),
                               ["pol"], "tcH_")
    out = pd.DataFrame({"tc_cos_prev": cos}, index=B.index)
    out["tc_pol_vs_hist"] = lex["tc_pol"] - pol_hist["tcH_pol_e50"]
    out = pd.concat([out, pol_hist[["tcH_pol_last", "tcH_pol_e50", "tcH_pol_m3", "tcH_n"]]], axis=1)
    return out.astype("float32")


IMP = {"一杯": 2.0, "強め": 1.0}
COND = {"良": 0.0, "稍": 1.0, "重": 2.0, "不": 3.0}


def ok_struct(B: pd.DataFrame) -> pd.DataFrame:
    """追い切り評価(race_oikiri)の中身: 脚色(一杯/強め)・馬場・コース(栗坂/美坂)・評価 A–D、日数、本数、馬自身の過去の評価との差。"""
    o = fs.read_flat("race_oikiri_flat", ["race_id", "horse_id", "training_date", "course", "condition", "impression", "evaluation"])
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    o["ok_eval"] = o["evaluation"].astype(str).str.strip().map({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    o["ok_imp"] = o["impression"].astype(str).str.strip().map(IMP).fillna(0.0)
    o["ok_cond"] = o["condition"].astype(str).str.strip().map(COND)
    c = o["course"].astype(str).str.strip()
    o["ok_slope"] = c.str.contains("坂").astype(float).where(c != "")
    o["ok_ritto"] = c.str.startswith("栗").astype(float).where(c != "")
    o["ok_miho"] = c.str.startswith("美").astype(float).where(c != "")
    td = pd.to_datetime(o["training_date"].astype(str).str.extract(r"(\d{4}/\d{1,2}/\d{1,2})")[0], format="%Y/%m/%d", errors="coerce")
    o["_td"] = td
    g = o.groupby(["race_id", "horse_id"])
    a = g.agg(ok_eval=("ok_eval", "max"), ok_imp=("ok_imp", "max"), ok_cond=("ok_cond", "max"), ok_slope=("ok_slope", "max"), ok_ritto=("ok_ritto", "max"),
              ok_miho=("ok_miho", "max"), ok_n=("ok_eval", "size"), _td=("_td", "max")).reset_index()
    m = B[["race_id", "horse_id", "date"]].astype(str).merge(a, on=["race_id", "horse_id"], how="left")
    m["ok_days"] = (pd.to_datetime(m["date"]) - pd.to_datetime(m["_td"])).dt.days
    m["ok_days"] = m["ok_days"].where((m["ok_days"] >= 0) & (m["ok_days"] <= 30))
    h = fs.history_aggs(B, m[["race_id", "horse_id", "ok_eval", "ok_imp"]], ["ok_eval", "ok_imp"], "okH_")
    out = m[["ok_eval", "ok_imp", "ok_cond", "ok_slope", "ok_ritto", "ok_miho", "ok_n", "ok_days"]].copy()
    out.index = B.index
    out["ok_eval_vs_hist"] = out["ok_eval"] - h["okH_ok_eval_e50"]
    out["ok_imp_vs_hist"] = out["ok_imp"] - h["okH_ok_imp_e50"]
    out = pd.concat([out, h[["okH_ok_eval_e50", "okH_ok_eval_last", "okH_ok_imp_e50", "okH_ok_imp_m3"]]], axis=1)
    g2 = out.groupby(B["race_id"].to_numpy())
    out["ok_eval_rz"] = (out["ok_eval"] - g2["ok_eval"].transform("mean")) / g2["ok_eval"].transform("std").replace(0, np.nan)
    out["ok_imp_field_share"] = g2["ok_imp"].transform(lambda s: (s > 0).mean())
    return out.astype("float32")


GROUPS_TEXT = {"TXT_tc_lex": txt_tc_lex, "TXT_tc_svd": txt_tc_svd, "TXT_tc_change": txt_tc_change, "OK_struct": ok_struct}
