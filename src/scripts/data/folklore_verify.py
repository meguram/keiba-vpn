"""通説の検証: すべての型 × すべての切り口（コース・馬場・クラス など）で『当てはまる馬は当てはまらない馬より 3 着内に来るか』を検定し、
信頼度で S>A>B>C>無 に分ける。

python -m src.scripts.data.folklore_verify run      # <FOLKLORE_OUT>/tests.parquet・hyp.json
python -m src.scripts.data.folklore_verify rank     # 通説の数で並べた成績 → <FOLKLORE_OUT>/rank.json（report の第 8 節）
python -m src.scripts.data.folklore_verify score    # 格の重みの得点・閾値ごとの 1 位の成績・ROC → <FOLKLORE_OUT>/score.json（第 9 節）
python -m src.scripts.data.folklore_verify report   # docs/html/modeling/chapters/folklore_verification.html

検定の作り（詳しくは章の本文）
- 目的変数: 3 着内。比較: 同じ切り口・同じ『資格』（eligible）の中で、当てはまる馬 − 当てはまらない馬。
- 2 つの差: 素の差（3 着内率の差）と、人気をそろえた差（確定単勝オッズから作った市場の 3 着内確率を引いた残差の差）。
- 標準誤差はレース単位のクラスタ頑健（比の推定量の線形化）。
- 期間: 発見（データのある年の前 7 割、全年あれば 2016〜2022）と確認（残り、全年あれば 2023〜2025）。
- 多重比較: 発見期間の全検定（型 × 切り口）に BH。q<0.10 を確認へ送り、確認は発見の向きの片側検定に BH。
- 信頼度 = 1 − max(q_発見, q_確認)。S ≥ 99.9%、A ≥ 99%、B ≥ 95%、C ≥ 90%、それ未満は無。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("FOLKLORE_OUT", "/home/jovyan/work/folklore_out"))
DOC = ROOT / "docs/html/modeling/chapters/folklore_verification.html"
MIN_N1, MIN_R1, MIN_N0, MIN_CONF_N1 = 300, 200, 300, 100
TIERS = [("S", 0.999), ("A", 0.99), ("B", 0.95), ("C", 0.90)]
SHADAI = ("ノーザンファーム", "社台ファーム", "社台コーポレーション白老", "白老ファーム", "追分ファーム", "社台Ｃ白老", "ノーザンＦ", "社台Ｆ")
CLUBS = ("サンデーレーシング", "キャロットファーム", "シルクレーシング", "社台レースホース", "Ｇ１レーシング", "G1レーシング", "東京ホースレーシング",
         "ラフィアン", "インゼルレーシング", "ノルマンディー", "ユニオンオーナーズ", "ターフ・スポート", "広尾レース", "ロードホースクラブ", "キャロット",
         "ウイン", "サラブレッドクラブ", "グリーンファーム", "ＤＭＭドリーム", "DMMドリーム", "ライオンレースホース", "ワラウカド", "大樹レーシング",
         "ゴドルフィン", "ローレルレーシング", "YGGホースクラブ", "ＹＧＧ", "京都ホースレーシング", "ノースヒルズ")
TURF_START_DIRT = {"東京ダ1600", "中山ダ1200", "阪神ダ1400", "阪神ダ2000", "京都ダ1400", "京都ダ1600", "新潟ダ1200", "中京ダ1400", "福島ダ1150"}


# ---------------------------------------------------------------- 追加の列（as-of）
def asof_stats(d: pd.DataFrame, keys: list[str], mask: pd.Series, name: str) -> None:
    """mask の行だけを keys ごとに日別に数えて累積し、全行に『その日より前』の n・3 着内数を付ける（merge_asof）。"""
    m = d[mask & d[keys].notna().all(axis=1)]
    agg = m.groupby(keys + ["dt"], sort=False).agg(n=("y3", "size"), k=("y3", "sum"), w=("y1", "sum")).reset_index()
    agg = agg.sort_values("dt", kind="stable")
    cs = agg.groupby(keys, sort=False)[["n", "k", "w"]].cumsum()
    agg[["n", "k", "w"]] = cs
    left = d[keys + ["dt"]].copy()
    left["_i"] = np.arange(len(d))
    ok = left[keys].notna().all(axis=1)
    l2 = left[ok].sort_values("dt", kind="stable")
    for k in keys:
        l2[k] = l2[k].astype(str)
        agg[k] = agg[k].astype(str)
    mg = pd.merge_asof(l2, agg, on="dt", by=keys, allow_exact_matches=False, direction="backward")
    out_n = np.full(len(d), np.nan)
    out_k = np.full(len(d), np.nan)
    out_n[mg["_i"].to_numpy()] = mg["n"].to_numpy()
    out_k[mg["_i"].to_numpy()] = mg["k"].to_numpy()
    d[name + "_n"] = out_n
    d[name + "_k"] = out_k
    d[name + "_r"] = out_k / np.where(out_n > 0, out_n, np.nan)


def tercile_2020(d: pd.DataFrame, col: str, mask: pd.Series) -> tuple[float, float]:
    b = d.loc[mask & d["year"].between(2020, 2021), col].dropna()
    return float(b.quantile(1 / 3)), float(b.quantile(2 / 3))


def add_extra(d: pd.DataFrame) -> pd.DataFrame:
    # 市場確率 q は、残った出走（オッズが正）の中で正規化し直す（古い master では取消馬の 0 倍でレース全体の q が 0 になっていた）
    inv = 1.0 / d["odds"].where(d["odds"] > 0)
    d["q"] = inv / inv.groupby(d["race_id"]).transform("sum")
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    d["p_race_id"] = g["race_id"].shift(1)
    for c in ["ok_int", "ok_slope", "ok_l1", "ok_eval", "ok_wood"]:
        d["p_" + c] = g[c].shift(1)
    d["p_jt_n"] = g["jt_n"].shift(1)
    d["best_l3f_prev"] = g["l3f"].transform(lambda s: s.shift(1).cummin())
    hk = (d["vc"].isin(["01", "02"]) & (d["surf"] == "芝")).astype(float)
    d["hk_top3_prev"] = (hk * d["y3"]).groupby(d["horse_id"]).cumsum() - hk * d["y3"]
    d["hk_n_prev"] = hk.groupby(d["horse_id"]).cumsum() - hk
    d["hnr"] = d["hn"] / d["field"]
    d["inner"] = d["hnr"] <= 1 / 3
    d["outer"] = d["hnr"] > 2 / 3
    d["by"] = pd.to_numeric(d["horse_id"].str[:4], errors="coerce")
    d["f_dirt"] = pd.to_numeric(d["f_dirt_share"], errors="coerce")
    d["f_1400"] = pd.to_numeric(d["f_1400_share"], errors="coerce")
    for c in ["blinker", "p_blinker", "allow", "j_rate", "pj_rate", "t_rate", "nige_rate", "senko_rate", "cush", "tmoist", "dmoist", "temp", "tmax", "wind", "rain_cum",
              "pd_rank", "tc_has", "tc_pos", "tc_neg", "tc_next", "mg_p1", "mg_p2", "mg_p3", "mg_max5", "mg_sd5", "mg_mean3", "mg_maxall", "ten1f_best_rank",
              "speed_course", "ok_l1", "p_ok_l1", "ok_acc", "ok_n_f", "ok_days", "bias_pos", "bias_hn", "day_fast", "p_race_fast", "member_op"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["ok_slope", "ok_wood", "p_ok_slope", "slope", "p_slope", "local", "p_local", "fillies", "p_fillies", "bad", "p_bad", "mixed", "p_jump",
              "pj_rides_other", "chosen_by_jockey", "choice_set", "nb_nige", "j_foreign", "j_short", "first_vs_old"]:
        if c in d:
            d[c] = d[c].astype(str).isin(["True", "true", "1", "1.0"])
    # 父の as-of 成績（全体・コース・道悪・北海道芝・ダート・夏・休み明け 2 戦目・5 歳以上・初ダート）・ニックス
    allm = pd.Series(True, index=d.index)
    asof_stats(d, ["f_name"], allm, "s")
    asof_stats(d, ["f_name", "course"], allm, "sc")
    asof_stats(d, ["f_name"], d["bad"], "sbad")
    asof_stats(d, ["f_name"], d["vc"].isin(["01", "02"]) & (d["surf"] == "芝"), "shk")
    asof_stats(d, ["f_name"], d["surf"] == "ダ", "sdirt")
    asof_stats(d, ["f_name"], d["month"].between(6, 9), "ssum")
    asof_stats(d, ["f_name"], d["since_rest"] == 2, "sk2")
    asof_stats(d, ["f_name"], d["age"] >= 5, "sold")
    asof_stats(d, ["f_name", "mf_name"], allm, "nick")
    asof_stats(d, ["owner", "jockey_id"], allm, "oj")
    asof_stats(d, ["trainer_id"], d["grade"] == "新馬", "tnew")
    # 調教師 × 最終追い切りのパターン（場所 × 強さ）: B37
    place = np.where(d["ok_slope"], "坂路", np.where(d["ok_course"].fillna("").str.endswith(("Ｐ",)), "ポリ",
                     np.where(d["ok_course"].fillna("").str.endswith(("芝",)), "芝", np.where(d["ok_wood"], "W", None))))
    d["ok_pat"] = pd.Series(place, index=d.index).where(d["ok_int"].notna()) + "・" + d["ok_int"].fillna("")
    d["ok_pat"] = d["ok_pat"].where(pd.Series(place, index=d.index).notna() & d["ok_int"].notna())
    # 調教師の『一杯』の頻度（as-of）: B19
    ip = (d["ok_int"] == "一杯").astype(float).where(d["ok_int"].notna())
    tmp = d[["trainer_id", "dt"]].copy()
    tmp["ip"] = ip
    tmp = tmp.dropna(subset=["ip"]).groupby(["trainer_id", "dt"]).agg(n=("ip", "size"), k=("ip", "sum")).reset_index().sort_values(["trainer_id", "dt"])
    cs = tmp.groupby("trainer_id")[["n", "k"]].cumsum() - tmp[["n", "k"]]
    tmp["t_ippai_rate"] = (cs["k"] / cs["n"]).where(cs["n"] >= 20)
    d = d.merge(tmp[["trainer_id", "dt", "t_ippai_rate"]], on=["trainer_id", "dt"], how="left")
    # 坂路の終い 1F の速さ（調教場・年ごとの四分位）
    sl = d["ok_slope"] & d["ok_l1"].notna()
    ctr = d["ok_course"].fillna("").str[0]
    qq = d[sl].groupby([ctr[sl], d.loc[sl, "year"]])["ok_l1"].quantile([0.25, 0.5]).unstack()
    key = list(zip(ctr, d["year"]))
    d["ok_l1_q25"] = [qq.loc[k, 0.25] if k in qq.index else np.nan for k in key]
    d["ok_l1_q50"] = [qq.loc[k, 0.5] if k in qq.index else np.nan for k in key]
    # 騎手の年の勝ち数の順位（as-of、その日より前）
    jd = d.groupby(["jockey_id", "year", "dt"])["y1"].sum().reset_index().sort_values(["jockey_id", "year", "dt"])
    jd["ytd"] = jd.groupby(["jockey_id", "year"])["y1"].cumsum() - jd["y1"]
    d = d.merge(jd[["jockey_id", "dt", "ytd"]], on=["jockey_id", "dt"], how="left")
    d["ytd_rank"] = d.groupby("dt")["ytd"].rank(ascending=False, method="min")
    # 騎手の初騎乗年・調教師の最初と最後の日
    d["j_first_year"] = d.groupby("jockey_id")["year"].transform("min")
    d["t_first"] = d.groupby("trainer_id")["dt"].transform("min")
    d["t_last"] = d.groupby("trainer_id")["dt"].transform("max")
    # 騎手の減量が取れた直後（その騎手の 120 日前までの減量 > 0、今は 0）
    ja = d[d["allow"].notna()].groupby(["jockey_id", "dt"])["allow"].median().reset_index().sort_values(["jockey_id", "dt"])
    ja["allow_max120"] = ja.groupby("jockey_id").apply(lambda x: x.set_index("dt")["allow"].rolling("120D", closed="left").max()).reset_index(level=0, drop=True).to_numpy()
    d = d.merge(ja[["jockey_id", "dt", "allow_max120"]], on=["jockey_id", "dt"], how="left")
    # 前走騎手がその日の前後 3 日に騎乗していない（騎乗停止・負傷の代わり）
    rides = d[["jockey_id", "dt"]].drop_duplicates()
    rides["one"] = 1
    s = set(zip(rides["jockey_id"], rides["dt"]))
    def absent(j, t):
        if not isinstance(j, str) or pd.isna(t):
            return False
        return not any((j, t + pd.Timedelta(days=k)) in s for k in range(-3, 4))
    sw = d["jockey_id"] != d["p_jockey_id"]
    d["pj_absent"] = False
    idx = d.index[sw & d["p_jockey_id"].notna()]
    d.loc[idx, "pj_absent"] = [absent(j, t) for j, t in zip(d.loc[idx, "p_jockey_id"], d.loc[idx, "dt"])]
    # 前日（同じ場・同じ面の直前の開催日）の前残りの度合い
    r = d[d["fin"] <= 3].groupby(["vc", "surf", "dt"])["pos4r"].mean().reset_index().sort_values(["vc", "surf", "dt"])
    r["prevday_bias"] = r.groupby(["vc", "surf"])["pos4r"].shift(1)
    r["gap"] = (r["dt"] - r.groupby(["vc", "surf"])["dt"].shift(1)).dt.days
    r.loc[r["gap"] > 8, "prevday_bias"] = np.nan
    d = d.merge(r[["vc", "surf", "dt", "prevday_bias"]], on=["vc", "surf", "dt"], how="left")
    # 母父の産駒（母父として）の勝ち距離の平均（2016〜2019 で固定）
    mf = d[(d["y1"] == 1) & (d["year"] <= 2019)].groupby("mf_name")["dist"].agg(["mean", "size"])
    d["mf_meandist"] = d["mf_name"].map(mf["mean"].where(mf["size"] >= 20))
    # 父の初年度産駒
    d["sire_first_by"] = d.groupby("f_name")["by"].transform("min")
    # レース内の持ち時計（コース別の指数）の順位
    d["si_course_rank"] = d.groupby("race_id")["speed_course"].rank(ascending=False, method="min")
    # 同じ馬主の逃げ候補（人気薄）がいる
    pm = d[(d["nige_cand"] == 1) & (d["pop"] >= 6)][["race_id", "owner", "horse_id"]].rename(columns={"horse_id": "pm_h"})
    x = d[["race_id", "owner", "horse_id"]].merge(pm, on=["race_id", "owner"], how="inner")
    x = x[x["pm_h"] != x["horse_id"]]
    d["has_pacemaker"] = d.set_index(["race_id", "horse_id"]).index.isin(x.set_index(["race_id", "horse_id"]).index)
    # A29: 前走で同じレースに出ていて今回もいる相手（再戦）。全員に先着されていたか
    pr = d[["race_id", "horse_id", "p_race_id", "p_fin"]].dropna(subset=["p_race_id"])
    m = pr.merge(pr, on=["race_id", "p_race_id"], suffixes=("", "_o"))
    m = m[m["horse_id"] != m["horse_id_o"]]
    agg = m.groupby(["race_id", "horse_id"]).agg(n_rival=("horse_id_o", "size"), n_beaten=("p_fin_o", "size"))
    beaten = m[m["p_fin_o"] < m["p_fin"]].groupby(["race_id", "horse_id"]).size().rename("n_lost")
    agg = agg.join(beaten).fillna({"n_lost": 0}).reset_index()
    d = d.merge(agg[["race_id", "horse_id", "n_rival", "n_lost"]], on=["race_id", "horse_id"], how="left")
    # レースの 1・2 番人気の単勝の市場確率の合計・1 番人気の単勝オッズ
    rq = d.groupby("race_id").apply(lambda x: x.nsmallest(2, "pop")["q"].sum()).rename("top2q")
    d = d.merge(rq.reset_index(), on="race_id", how="left")
    d["fav_odds"] = d.groupby("race_id")["odds"].transform("min")
    # クッション値・含水率の三分位（2020〜2021 で固定）
    d["cush_lo"], d["cush_hi"] = tercile_2020(d, "cush", d["surf"] == "芝")
    d["dm_lo"], d["dm_hi"] = tercile_2020(d, "dmoist", d["surf"] == "ダ")
    d["tm_lo"], d["tm_hi"] = tercile_2020(d, "tmoist", d["surf"] == "芝")
    # 騎手の格の上位 20%（その年の j_rate の分布）・調教師の上位 15%
    d["j_top"] = d["j_rate"] >= d.groupby("year")["j_rate"].transform(lambda s: s.quantile(0.8))
    d["t_top"] = d["t_rate"] >= d.groupby("year")["t_rate"].transform(lambda s: s.quantile(0.85))
    d["j_nige_top"] = d["j_nige_rate"] >= d.groupby("year")["j_nige_rate"].transform(lambda s: s.quantile(0.8))
    d["j_nige_70"] = d["j_nige_rate"] >= d.groupby("year")["j_nige_rate"].transform(lambda s: s.quantile(0.7))
    d["mg_sd_q25"] = d["mg_sd5"] <= d.groupby("year")["mg_sd5"].transform(lambda s: s.quantile(0.25))
    # B37: 調教師 × パターンの as-of の『人気をそろえた残差』を縮約（経験ベイズ）
    return d


def add_market(d: pd.DataFrame) -> pd.DataFrame:
    """3 着内の市場の見込み p_mkt = 『人気順（14 番人気以上はまとめる）× 同じ人気の中の単勝オッズの 5 分位 × 頭数帯』の区分ごとの
    3 着内率の実績（2016〜2022 年で当てはめ。区分が 200 走未満なら 人気 × 頭数帯 の平均）。

    以前の『q の 40 分位 × 頭数帯』は人気順ごとに ±1.5pt ずれ（1 番人気は見込みが高すぎ、2・3 番人気は低すぎ）、
    人気になりやすい条件の型が有利・不利に出た（2026-11-09、別セッションの指摘で検証して採用。research/columns/folklore_frame.expected_top3 と同じ区分）。
    """
    d["fband"] = pd.cut(d["field"], [0, 8, 11, 14, 18], labels=["~8", "9-11", "12-14", "15-18"]).astype(str)
    pop = d["pop"].clip(upper=14).fillna(14).astype(int)
    fitm = (d["year"] <= 2022).to_numpy()
    # 同じ人気の中の単勝オッズの 5 分位（境目は 2016〜2022 年で決める）
    oq = np.zeros(len(d), dtype=int)
    for k in np.unique(pop):
        m = (pop == k).to_numpy()
        e = np.unique(np.quantile(d.loc[m & fitm, "odds"], [0.2, 0.4, 0.6, 0.8])) if (m & fitm).any() else np.array([])
        oq[m] = np.searchsorted(e, d.loc[m, "odds"].to_numpy(), side="right")
    key3 = pop.astype(str) + "_" + pd.Series(oq, index=d.index).astype(str) + "_" + d["fband"]
    key2 = pop.astype(str) + "_" + d["fband"]
    m3 = d.loc[fitm].groupby(key3[fitm])["y3"].agg(["mean", "size"])
    m2 = d.loc[fitm].groupby(key2[fitm])["y3"].mean()
    p = key3.map(m3["mean"]).where(key3.map(m3["size"]) >= 200, key2.map(m2))
    d["p_mkt"] = p.astype(float).to_numpy()
    d["r3"] = d["y3"] - d["p_mkt"]
    # 払戻を頭打ちにした単勝回収（50 倍超は 50 倍に）
    d["ret_cap"] = np.where(d["y1"] == 1, np.minimum(d["odds"], 50.0), 0.0)
    d["ret"] = np.where(d["y1"] == 1, d["odds"], 0.0)
    return d


def add_b37(d: pd.DataFrame) -> pd.DataFrame:
    """調教師 × パターンの as-of の残差の合計と件数 → 縮約した上積み（k=50 走ぶん全体へ寄せる）。"""
    m = d[d["ok_pat"].notna()]
    agg = m.groupby(["trainer_id", "ok_pat", "dt"]).agg(n=("r3", "size"), s=("r3", "sum")).reset_index().sort_values("dt")
    cs = agg.groupby(["trainer_id", "ok_pat"])[["n", "s"]].cumsum()
    agg[["n", "s"]] = cs
    left = d.loc[d["ok_pat"].notna(), ["trainer_id", "ok_pat", "dt"]].copy()
    left["_i"] = left.index
    left = left.sort_values("dt")
    mg = pd.merge_asof(left, agg.sort_values("dt"), on="dt", by=["trainer_id", "ok_pat"], allow_exact_matches=False)
    d["tp_n"] = np.nan
    d["tp_shr"] = np.nan
    d.loc[mg["_i"], "tp_n"] = mg["n"].to_numpy()
    d.loc[mg["_i"], "tp_shr"] = (mg["s"] / (mg["n"] + 50.0)).to_numpy()
    return d


# ---------------------------------------------------------------- 型の台帳
@dataclass
class H:
    hid: str
    direction: int  # +1: 当てはまると好走する（買い）、−1: 凡走する（消し）
    elig: Callable | None
    flag: Callable | None
    how: str  # この検証での当てはめ方（代理を含む）
    na: str = ""  # 検証できない理由


def _notna(*cols):
    return lambda d: np.logical_and.reduce([d[c].notna().to_numpy() for c in cols])


def R(d, c):
    return d[c]


def hyps() -> list[H]:
    G = ("G1", "G2", "G3")
    prom = lambda d: (d["p_y1"] == 1) & (d["cls"] > d["p_cls"])  # noqa: E731
    sw = lambda d: d["p_jockey_id"].notna() & (d["jockey_id"] != d["p_jockey_id"])  # noqa: E731
    up = {"馬也": 0, "強め": 1, "一杯": 2}
    L = []
    a = L.append
    # ---- A
    a(H("A01", 1, lambda d: d["p_dist"].notna() & (d["p_surf"] == d["surf"]), lambda d: d["dist"] - d["p_dist"] >= 100, "同じ面の前走より 100m 以上の延長 vs それ以外"))
    a(H("A02", 1, lambda d: d["p_dist"].notna() & (d["p_surf"] == d["surf"]), lambda d: d["p_dist"] - d["dist"] >= 100, "同じ面の前走より 100m 以上の短縮 vs それ以外"))
    a(H("A03", 1, lambda d: d["p_dist"].notna(), lambda d: d["course"] == d["p_course"], "前走と同じコース（場・面・距離）"))
    a(H("A04", 1, lambda d: (d["surf"] == "ダ") & (d["surf_n_prev"] == 0) & (d["starts_prev"] >= 1) & d["f_dirt"].notna(), lambda d: d["f_dirt"] >= 6, "初ダートの馬の中で、父の産駒のダート比率が上位（SmartRC の f_dirt_share 6 以上）"))
    a(H("A05", 1, lambda d: (d["surf"] == "芝") & (d["surf_n_prev"] == 0) & (d["starts_prev"] >= 1) & d["cl_cmb"].notna() & (d["cl_cmb"] != "<NA>"), lambda d: d["cl_cmb"].str.contains("欧", na=False), "初芝の馬の中で、血統の型が欧州型（SmartRC の cl_eval_cmb）"))
    a(H("A06", 1, lambda d: (d["surf"] == "ダ") & d["surf_n_prev"].between(1, 4) & ((d["starts_prev"] - d["surf_n_prev"]) >= 1), lambda d: d["surf_n_prev"] == 1, "芝の経験がありダートが 2〜5 戦目の馬で、ダート 2 戦目"))
    a(H("A07", 1, lambda d: d["io"].isin(["内", "外"]) & d["p_io"].isin(["内", "外"]), lambda d: (d["p_io"] == "内") & (d["io"] == "外"), "内回り → 外回り（距離から内外を決められるコースだけ）"))
    a(H("A08", 1, lambda d: d["dir2"].notna() & d["p_direction"].isin(["右", "左"]) & (d["direction"] != d["p_direction"]) & (d["dir_n_prev"] >= 2) & (d["starts_prev"] >= 4),
        lambda d: d["dir_top3_prev"] / d["dir_n_prev"] >= d["top3_prev"] / d["starts_prev"] + 0.1, "回りが替わった馬で、今回の回りの 3 着内率が通算より 10pt 以上高い"))
    a(H("A09", 1, lambda d: d["p_slope"].notna() & d["slope"].notna() & (d["p_slope"] != d["slope"]), lambda d: d["p_slope"] & ~d["slope"], "坂のある場 → 坂の無い場"))
    a(H("A10", -1, lambda d: d["p_straight"].notna() & d["straight"].notna() & (d["p_l3rank"] <= 3), lambda d: d["p_straight"] - d["straight"] >= 150, "前走上がり 3 位以内の馬で、直線が 150m 以上短くなる"))
    a(H("A11", 1, lambda d: d["course"] == "新潟芝1000", lambda d: d["outer"], "新潟千直の外枠（馬番が外 1/3）"))
    a(H("A12", 1, lambda d: (d["rest_len"] >= 70) & (d["since_rest"] <= 4), lambda d: d["since_rest"] == 2, "中 10 週以上の休み明けから 1〜4 戦目の中で 2 戦目"))
    a(H("A13", 1, lambda d: (d["since_rest"] == 1) & (d["rest_len"] >= 70) & (d["tfb_n"] >= 20) & d["t_rate"].notna(), lambda d: d["tfb_y3"] / d["tfb_n"] - d["t_rate"] >= 0.03, "休み明け初戦で、調教師の休み明けの as-of の 3 着内率が普段より 3pt 以上高い"))
    a(H("A14", 1, lambda d: d["interval"] <= 70, lambda d: d["interval"] <= 14, "連闘・中 1 週（間隔 14 日以内）"))
    a(H("A15", -1, lambda d: d["p_cls"].notna(), prom, "前走勝って昇級"))
    a(H("A16", -1, lambda d: d["grade"].isin(G) & d["p_cls"].notna(), lambda d: (d["p_cls"] <= 4) & (d["p_y1"] == 0), "重賞で、前走が条件戦で勝っていない（格上挑戦の代わり）"))
    a(H("A17", 1, lambda d: (d["p_fin"] >= 2) & d["p_tdiff"].notna(), lambda d: d["p_tdiff"] <= 0.5, "前走で負けた馬のうち勝ち馬から 0.5 秒以内"))
    a(H("A18", 1, lambda d: d["p_fin"] >= 4, lambda d: d["p_l3rank"] == 1, "前走 4 着以下で上がり最速"))
    a(H("A19", 1, lambda d: (d["p_fin"] <= 5) & d["p_pos4"].notna(), lambda d: d["p_pos4"] >= 10, "前走掲示板で 4 角 10 番手以下"))
    a(H("A20", 1, lambda d: (d["p_pos4r"] <= 0.3) & (d["p_fin"] <= 5) & d["p_pace_cat"].notna(), lambda d: d["p_pace_cat"] == "H", "前走 4 角で前 3 割・掲示板の馬で、前走がハイペース"))
    a(H("A21", -1, lambda d: (d["p_fin"] >= 2) & d["p_pace_cat"].notna(), lambda d: (d["p_pace_cat"] == "S") & (d["p_pos4r"] >= 0.6) & (d["p_l3rank"] <= 3), "前走スローで後方から上がり 3 位以内で負け"))
    a(H("A22", 1, lambda d: (d["track"] == "良") & d["p_bad"] & (d["good_n_prev"] >= 1), lambda d: (d["p_fin"] >= 6) & (d["good_top3_prev"] / d["good_n_prev"] >= 0.3), "前走重・不良 → 今回良で、前走 6 着以下かつ良の 3 着内率 30% 以上"))
    a(H("A23", 1, lambda d: d["p_fin"] >= 10, lambda d: (d["dist"] != d["p_dist"]) | (d["surf"] != d["p_surf"]) | sw(d) | (d["bad"] != d["p_bad"]), "前走 10 着以下で、距離・面・騎手・馬場のどれかが替わる"))
    a(H("A24", 1, lambda d: d["p_pos1r"].notna() & d["pos1r_mean_prev"].notna() & (d["starts_prev"] >= 3), lambda d: d["p_pos1r"] - d["pos1r_mean_prev"] >= 0.35, "前走の 1 角の位置がいつもより 35% 以上後ろ（出遅れ・不利の代わり）"))
    a(H("A25", 1, lambda d: d["course_n_prev"] >= 1, lambda d: d["course_top3_prev"] >= 1, "同じコースの経験がある馬で、そこで 3 着内がある"))
    a(H("A26", 1, lambda d: d["grade"].isin(G) & (d["age"] >= 4), lambda d: d["rn_top3_prev"] >= 1, "重賞で、同じレース名で過去に 3 着内"))
    a(H("A27", 1, lambda d: d["grade"].isin(["G2", "G3"]) & d["p_grade"].notna(), lambda d: d["p_grade"] == "G1", "G2・G3 で前走 G1"))
    a(H("A28", 1, lambda d: prom(d) & d["p_race_fast"].notna(), lambda d: d["p_race_fast"] <= -0.5, "昇級初戦で、前走の勝ちタイムがコース・クラスの基準より 0.5 秒以上速い"))
    a(H("A29", -1, lambda d: d["n_rival"] >= 1, lambda d: d["n_lost"] == d["n_rival"], "前走の同走馬と再戦する馬で、その相手全員に前走で先着されていた"))
    a(H("A30", 1, lambda d: d["month"].between(6, 9) & d["mixed"] & d["cls"].between(2, 3), lambda d: d["age"] == 3, "6〜9 月の 3 歳以上 1・2 勝クラスで 3 歳"))
    a(H("A31", -1, lambda d: (d["age"] == 2) & (d["dist"] > d["p_dist"]) & d["f_1400"].notna(), lambda d: d["f_1400"] >= 6, "2 歳の延長で、父の産駒の 1400 以下の比率が上位"))
    a(H("A32", 1, lambda d: d["p_vc"].notna(), lambda d: ~d["p_local"] & d["local"], "中央場 → ローカル"))
    a(H("A33", 1, lambda d: (d["weight_rule"] == "ハンデ") & d["p_jw"].notna(), lambda d: d["jw"] - d["p_jw"] <= -2, "ハンデ戦で斤量が前走より 2kg 以上軽い"))
    a(H("A34", -1, lambda d: d["pp_last_region"].notna(), lambda d: d["pp_last_region"] == "海外", "出馬表の前走が海外（2020〜2025）"))
    for hid, why in (("A35", "遠征先の地域別に分けると 1 地域あたり数十走で、最小標本（300 走）に届かない"), ("A36", "出馬表の海外の走歴に着順が無い"),
                     ("A37", "帰国後何戦目かを決める走歴の並びを表に入れていない（次の版）"), ("A38", "海外帰りが数百走で、調教師別に分けると標本が無い"), ("A39", "帰国後の追い切りの本数（追い切りの履歴）が手元に無い")):
        a(H(hid, 0, None, None, "", why))
    a(H("A40", 1, lambda d: (d["since_rest"] == 1) & (d["fb_n_prev"] >= 1) & (d["k23_n_prev"] >= 1), lambda d: d["fb_top3_prev"] / d["fb_n_prev"] > d["k23_top3_prev"] / d["k23_n_prev"], "休み明け初戦で、過去の休み明け初戦の 3 着内率が叩き 2・3 戦目より高い馬"))
    a(H("A41", -1, lambda d: (d["since_rest"] == 2) & (d["rest_len"] >= 70) & (d["interval"] <= 35), lambda d: d["p_fin"] <= 3, "休み明け 2 戦目（中 4 週以内）で、休み明け初戦が 3 着内"))
    a(H("A42", 1, prom, lambda d: d["win_streak"] >= 2, "昇級初戦で 2 連勝以上"))
    a(H("A43", 1, lambda d: d["r5_top3"].notna(), lambda d: (d["r5_top3"] >= 3) & (d["r5_win"] == 0), "直近 5 走で 3 着内 3 回以上・勝ち無し"))
    a(H("A44", 1, lambda d: d["best_dist_min"].notna() & d["p_dist"].notna(), lambda d: ((d["p_dist"] < d["best_dist_min"] - 100) | (d["p_dist"] > d["best_dist_max"] + 100)) & d["dist"].between(d["best_dist_min"], d["best_dist_max"]), "前走が好走距離帯の外 → 今回は好走距離帯の中"))
    a(H("A45", 1, lambda d: d["p_fin"] >= 6, lambda d: ((d["p_surf"] != d["surf"]) | ((d["dist"] - d["p_dist"]).abs() >= 400)) & (d["dbs_top3_prev"] >= 1), "前走 6 着以下で前走が別の面か 400m 以上違い、今回の面・距離帯で 3 着内がある"))
    a(H("A46", -1, lambda d: (d["p_surf"] == d["surf"]) & (d["dist"] - d["p_dist"]).abs().between(100, 200), lambda d: d["dband"] != d["p_dband"], "100〜200m の距離替わりで距離帯の境目をまたぐ"))
    a(H("A47", 1, lambda d: (d["p_fin"] >= 4) & d["p_vc"].notna(), lambda d: d["p_vc"].isin(["05", "06", "08", "09"]) & d["local"] & (d["cls"] == d["p_cls"]), "前走 4 着以下で、中央場 → ローカルの同クラス"))
    # ---- B（最終追い切り。乗り手・強度・ラップ・短評は 2016〜2019）
    is_j = lambda d: pd.Series([isinstance(r, str) and isinstance(j, str) and len(r) >= 2 and j.startswith(r) for r, j in zip(d["ok_rider"], d["jockey_name"])], index=d.index)  # noqa: E731
    staff = lambda d: d["ok_rider"].isin(["助手", "見習", "攻手"])  # noqa: E731
    a(H("B01", 1, lambda d: d["ok_rider"].notna(), is_j, "最終追い切りの乗り手の名前が本番騎手の名前の先頭と一致"))
    for hid, why in (("B02", "1 週前の追い切り（追い切りの履歴）が手元に無い"), ("B03", "騎手の所属の東西が無く、追い切りの場所も欠けが多い")):
        a(H(hid, 0, None, None, "", why))
    a(H("B04", 1, lambda d: sw(d) & d["ok_rider"].notna(), is_j, "乗り替わりの馬で、新しい騎手が最終追い切りに騎乗"))
    a(H("B05", 1, lambda d: d["ok_slope"] & d["ok_acc"].notna(), lambda d: d["ok_acc"] == 1, "坂路の最終追い切りで終い 1F が加速ラップ"))
    a(H("B06", 1, lambda d: (d["ok_n_f"] >= 3) & (d["dist"] >= 2000), lambda d: (d["ok_n_f"] >= 5) & ~d["ok_slope"], "2000m 以上で、最終追い切りが 6F 以上の時計"))
    a(H("B07", 1, lambda d: d["grade"].isin(G) & (d["ok_slope"] | d["ok_wood"]), lambda d: d["ok_slope"], "重賞で、最終が坂路（ウッドと比べる）"))
    a(H("B08", 1, lambda d: d["ok_slope"] & d["ok_l1"].notna() & d["ok_l1_q25"].notna(), lambda d: d["ok_l1"] <= d["ok_l1_q25"], "坂路の終い 1F が調教場・年の上位 25%"))
    for hid in ("B09", "B10", "B11"):
        a(H(hid, 0, None, None, "", "最終追い切りの表に併せ（先着/同入/遅れ・相手）の項目が無い"))
    a(H("B12", 1, lambda d: (d["since_rest"] == 1) & d["ok_int"].isin(list(up)), lambda d: d["ok_int"] == "一杯", "休み明け初戦で最終が一杯"))
    for hid in ("B13", "B14"):
        a(H(hid, 0, None, None, "", "中間の追い切りの本数・時計（追い切りの履歴）が手元に無い"))
    a(H("B15", 1, lambda d: (d["p_fin"] <= 3) & d["ok_int"].notna() & d["p_ok_int"].notna(), lambda d: (d["ok_int"] == d["p_ok_int"]) & (d["ok_slope"] == d["p_ok_slope"]), "前走 3 着内の馬で、最終の強度・場所が前走時と同じ"))
    a(H("B16", 1, lambda d: d["ok_int"].isin(list(up)) & d["p_ok_int"].isin(list(up)), lambda d: d["ok_int"].map(up) > d["p_ok_int"].map(up), "最終の強度が前走時より上がった"))
    a(H("B17", 1, lambda d: d["ok_days"].between(1, 14), lambda d: d["ok_days"] >= 7, "最終追い切りがレースの 7 日以上前（当週に追っていない）"))
    a(H("B18", 1, lambda d: d["ok_wd"].notna(), lambda d: d["ok_wd"].isin(["土", "日"]), "最終追い切りが土日"))
    a(H("B19", 1, lambda d: d["ok_slope"] & d["ok_int"].notna() & d["t_ippai_rate"].notna(), lambda d: (d["ok_int"] == "一杯") & (d["t_ippai_rate"] < 0.3), "坂路の最終で、普段は一杯が 3 割未満の調教師が一杯"))
    a(H("B20", 1, lambda d: d["ok_eval"].notna(), lambda d: d["ok_eval"] == "A", "最終追い切りの評価 A（2016〜2025）"))
    a(H("B21", -1, lambda d: d["ok_tanpyo"].notna(), lambda d: d["ok_tanpyo"].str.contains("不足|残し|太|重い|今一|一息|物足|平凡|疲|硬", regex=True, na=False), "短評に否定的な語"))
    a(H("B22", 1, lambda d: (d["grade"] == "新馬") & d["ok_slope"] & d["ok_l1"].notna() & d["ok_l1_q25"].notna(), lambda d: d["ok_l1"] <= d["ok_l1_q25"], "新馬で坂路の終い 1F が上位 25%"))
    a(H("B23", 1, lambda d: d["ok_eval"].isin(["C", "D"]) & d["p_ok_eval"].notna(), lambda d: d["p_ok_eval"].isin(["C", "D"]) & (d["p_fin"] <= 3), "今回評価 C・D で、前走も C・D なのに 3 着内だった（調教駆けしない馬）"))
    a(H("B24", -1, lambda d: d["ok_course"].fillna("") != "", lambda d: d["ok_course"].fillna("").str.endswith(("Ｐ", "芝", "プ")), "最終がポリ・芝・プール"))
    a(H("B25", 1, lambda d: d["local"] & (d["ok_course"].fillna("") != ""), lambda d: d["ok_course"].fillna("").str[0] == d["venue"].str[0], "ローカル開催で最終追い切りがその場（滞在）"))
    a(H("B26", 1, lambda d: d["ok_slope"] & d["ok_cond"].notna() & d["ok_l1"].notna() & d["ok_l1_q50"].notna(), lambda d: d["ok_cond"].isin(["重", "不"]) & (d["ok_l1"] <= d["ok_l1_q50"]), "坂路の重い馬場で終い 1F が中央値以下"))
    a(H("B27", 1, lambda d: d["ok_rider"].notna(), lambda d: ~staff(d), "最終の乗り手が助手・見習でない"))
    a(H("B28", 0, None, None, "", "追い切りのコース取り（内外の位置）が最終追い切りの表に無い"))
    a(H("B29", 1, lambda d: d["ok_slope"] & d["ok_l1"].notna(), lambda d: d["ok_l1"] <= 12.4, "坂路の終い 1F が 12.4 秒以下（絶対値の目安）"))
    a(H("B30", 0, None, None, "", "併せの頭数が無い"))
    a(H("B31", 1, lambda d: (d["interval"] <= 14) & d["ok_int"].isin(list(up)), lambda d: d["ok_int"] == "馬也", "間隔 14 日以内で最終が馬なり"))
    for hid in ("B32", "B33"):
        a(H(hid, 0, None, None, "", "1 週前・前週の追い切り（追い切りの履歴）が手元に無い"))
    a(H("B34", -1, lambda d: d["ok_slope"] & d["p_ok_slope"].astype(bool) & d["ok_l1"].notna() & d["p_ok_l1"].notna(), lambda d: d["ok_l1"] - d["p_ok_l1"] >= 0.5, "坂路の終い 1F が前走時より 0.5 秒以上遅い"))
    a(H("B35", 1, lambda d: d["ok_slope"] & d["ok_l1"].notna() & d["ok_l1_q25"].notna() & d["month"].isin([12, 1, 2]), lambda d: d["ok_l1"] <= d["ok_l1_q25"], "冬（12〜2 月）の坂路で終い 1F が上位 25%"))
    a(H("B36", 0, None, None, "", "1 日の追い切りの本数（追い切りの履歴）が手元に無い"))
    a(H("B37", 1, lambda d: d["tp_n"] >= 20, lambda d: d["tp_shr"] >= 0.03, "調教師 × 最終のパターン（場所 × 強さ）の as-of の人気補正後の上積み（50 走ぶん縮約）が +3pt 以上。併せ・1 週前・本数は無し"))
    # ---- C
    a(H("C01", 1, lambda d: d["jt_n"].notna() & d["j_rate"].notna(), lambda d: (d["jt_n"] >= 20) & (d["jt_y3"] / d["jt_n"] - d["j_rate"] >= 0.05), "騎手 × 調教師の as-of の 3 着内率が騎手の普段より 5pt 以上高い（20 騎乗以上）"))
    a(H("C02", 1, lambda d: d["jt_n"].notna() & d["j_rate"].notna(), lambda d: (d["jt_n"] <= 2) & d["j_top"], "その厩舎での騎乗が 2 回以下の上位 20% 騎手"))
    a(H("C03", -1, sw, lambda d: d["pj_rides_other"], "乗り替わりで、前走騎手が同じレースの別の馬に騎乗"))
    a(H("C04", 1, lambda d: d["choice_set"], lambda d: d["chosen_by_jockey"], "前走で同じ騎手が乗った馬が 2 頭以上いて、騎手が選んだ馬"))
    a(H("C05", 1, lambda d: d["p_jockey_id"].notna(), lambda d: d["j_cont"] >= 1, "前走と同じ騎手"))
    a(H("C06", 1, sw, lambda d: d["j_short"], "乗り替わりで短期免許の外国人騎手（カタカナ名・直近 365 日 300 騎乗未満）"))
    a(H("C07", 1, lambda d: d["allow"].notna() & (d["surf"] == "ダ") & (d["dist"] <= 1400), lambda d: d["allow"] >= 1, "ダート 1400 以下の馬齢・定量戦で減量騎手"))
    a(H("C08", -1, lambda d: sw(d) & d["j_rate"].notna() & d["pj_rate"].notna(), lambda d: d["j_rate"] - d["pj_rate"] >= 0.08, "乗り替わりで騎手の 3 着内率が 8pt 以上上がる（過剰人気か）"))
    a(H("C09", 1, lambda d: (d["jc_n"] >= 30) & d["j_rate"].notna(), lambda d: d["jc_y3"] / d["jc_n"] - d["j_rate"] >= 0.05, "騎手のそのコースの as-of 3 着内率が普段より 5pt 以上高い"))
    a(H("C10", 1, lambda d: d["j365_n"] >= 300, lambda d: d["jday_n"] <= 2, "年 300 騎乗以上の騎手がその日 2 鞍以下"))
    a(H("C11", 1, lambda d: d["jday_k"] >= 2, lambda d: d["jday_w_before"] >= 1, "騎手がその日の前のレースで勝っている"))
    a(H("C12", -1, lambda d: d["j_gap"].notna(), lambda d: d["j_gap"] >= 14, "騎手の前の騎乗日から 14 日以上空いた"))
    a(H("C13", 1, lambda d: d["j_nige_rate"].notna() & (d["pop"] >= 6), lambda d: d["j_nige_top"], "6 番人気以下で、逃げ率が上位 20% の騎手"))
    a(H("C14", 1, lambda d: d["j_rate"].notna() & (d["dist"] >= 2200), lambda d: d["j_top"], "2200m 以上で上位 20% 騎手"))
    a(H("C15", 1, lambda d: d["j_rate"].notna() & d["owner"].notna(), lambda d: (d["oj_n"] >= 10) & (d["oj_r"] - d["j_rate"] >= 0.05), "馬主 × 騎手の as-of 3 着内率が騎手の普段より 5pt 以上高い"))
    a(H("C16", 1, lambda d: d["grade"].isin(G) & d["j_rate"].notna(), lambda d: (d["jg_n"] >= 30) & (d["jg_y3"] / d["jg_n"] - d["j_rate"] >= 0.03), "重賞で、そのグレードの騎手の as-of 3 着内率が普段より 3pt 以上高い"))
    a(H("C17", 1, lambda d: d["allow"].notna() & (d["surf"] == "ダ") & (d["hnr"] >= 0.6), lambda d: d["allow"] >= 4, "ダートの外寄りの枠で 4kg 減（★、女性騎手）"))
    a(H("C18", 1, lambda d: sw(d) & d["j_rate"].notna() & d["pj_rate"].notna(), lambda d: (d["p_jt_n"] >= 50) & (d["j_rate"] - d["pj_rate"] >= 0.05), "乗り替わりで、前走騎手がその厩舎で 50 騎乗以上（所属の代わり）・新騎手が 5pt 以上上"))
    a(H("C19", 1, lambda d: d["allow"].notna() & d["allow_max120"].notna(), lambda d: (d["allow"] == 0) & (d["allow_max120"] > 0), "騎手の減量が直近 120 日に取れた"))
    a(H("C20", 1, lambda d: d["month"].isin([11, 12]) & d["ytd"].notna(), lambda d: d["ytd_rank"] <= 3, "11〜12 月に年の勝ち数 3 位以内の騎手"))
    a(H("C21", 1, sw, lambda d: d["pj_absent"], "乗り替わりで、前走騎手が前後 3 日に騎乗していない（騎乗停止・負傷の代わり）"))
    a(H("C22", 1, lambda d: d["j_nige_rate"].notna() & d["senko_rate"].notna(), lambda d: d["j_nige_70"] & (d["senko_rate"] >= 0.5), "逃げ率上位 30% の騎手 × 先行率 5 割以上の馬"))
    a(H("C23", 1, lambda d: d["jday_n"] >= 3, lambda d: d["jday_minpop_rank"] == 1, "騎手のその日の騎乗馬の中で最も人気の鞍（3 鞍以上の日）"))
    a(H("C24", -1, lambda d: d["jday_n"] >= 10, lambda d: d["jday_k"] >= 8, "10 鞍以上乗る日の 8 鞍目以降"))
    a(H("C25", -1, lambda d: d["grade"].isin(G) & (d["year"] >= 2020), lambda d: (d["year"] - d["j_first_year"] <= 3) & ~d["j_foreign"], "重賞で初騎乗から 3 年以内の騎手（2020 年以降）"))
    a(H("C26", 0, None, None, "", "騎手のエージェントの名簿が無い"))
    # ---- D
    a(H("D01", 1, lambda d: (d["since_rest"] == 1) & d["producer"].notna(), lambda d: d["producer"].str.contains("ノーザン", na=False), "休み明け初戦でノーザンファーム生産"))
    a(H("D02", 1, lambda d: (d["tc_n"] >= 20) & d["t_rate"].notna(), lambda d: d["tc_y3"] / d["tc_n"] - d["t_rate"] >= 0.05, "調教師のそのコースの as-of 3 着内率が普段より 5pt 以上高い"))
    a(H("D03", 1, lambda d: (d["t30_n"] >= 10) & d["t_rate"].notna(), lambda d: d["t30_y3"] / d["t30_n"] - d["t_rate"] >= 0.10, "調教師の直近 30 日の 3 着内率が直近 365 日より 10pt 以上高い"))
    a(H("D04", 1, lambda d: d["n_same_tr"] >= 2, lambda d: d["tr_pop_rank"] >= 2, "同じ厩舎の 2 頭以上の出走で人気の低い方"))
    a(H("D05", 1, lambda d: d["n_same_own"] >= 2, lambda d: d["own_pop_rank"] >= 2, "同じ馬主の 2 頭以上の出走で人気の低い方"))
    a(H("D06", 1, lambda d: d["p_trainer_id"].notna(), lambda d: d["since_transfer"].isin([1, 2]), "転厩後 1〜2 戦目"))
    a(H("D07", 1, lambda d: d["t_last"] <= pd.Timestamp("2025-03-15"), lambda d: ((d["t_last"] - d["dt"]).dt.days <= 120) & d["t_last"].dt.month.isin([2, 3]), "2〜3 月に最後の出走がある調教師（定年の代わり）の最後の 120 日"))
    a(H("D08", -1, lambda d: d["year"] >= 2018, lambda d: (d["dt"] - d["t_first"]).dt.days <= 365, "初出走から 1 年以内の調教師（2018 年以降）"))
    a(H("D09", -1, lambda d: d["away"].notna(), lambda d: d["away"] == 1, "所属と逆の地域・北海道への遠征"))
    a(H("D10", -1, lambda d: (d["sex"] == "牝") & d["away"].notna(), lambda d: (d["away"] == 1) & d["vc"].isin(["01", "02", "10"]), "牝馬の北海道・小倉への遠征"))
    a(H("D11", 1, lambda d: d["producer"].notna() & (d["surf"] == "ダ"), lambda d: ~d["producer"].str.contains("|".join(SHADAI), na=False), "ダートで社台グループ以外の生産"))
    a(H("D12", 1, lambda d: (d["sex"] == "牝") & (d["age"] == 5) & d["owner"].notna(), lambda d: d["owner"].str.contains("|".join(CLUBS), na=False) & d["month"].between(1, 5), "5 歳牝馬で、クラブ馬の 1〜5 月"))
    for hid, why in (("D13", "帰厩の日（休養後の最初の追い切り）が追い切りの履歴に無い"), ("D14", "出走登録・除外の記録が無い")):
        a(H(hid, 0, None, None, "", why))
    a(H("D15", 1, lambda d: (d["pop"] >= 6) & d["t_rate"].notna(), lambda d: d["t_top"], "6 番人気以下で、3 着内率が上位 15% の調教師"))
    a(H("D16", 1, lambda d: d["tozai"].notna(), lambda d: d["tozai"] == "栗東", "栗東所属"))
    a(H("D17", 1, lambda d: (d["grade"] == "新馬") & d["t_rate"].notna(), lambda d: (d["tnew_n"] >= 20) & (d["tnew_r"] - d["t_rate"] >= 0.05), "新馬で、調教師の新馬の as-of 3 着内率が普段より 5pt 以上高い"))
    a(H("D18", 1, lambda d: (d["n_same_own"] >= 2) & (d["pop"] <= 3), lambda d: d["has_pacemaker"], "3 番人気以内で、同じ馬主の人気薄の逃げ候補がいる"))
    a(H("D19", 1, lambda d: d["grade"].isin(G) & (d["pop"] >= 6) & d["producer"].notna(), lambda d: ~d["producer"].str.contains("|".join(SHADAI), na=False), "重賞の 6 番人気以下で社台グループ以外の生産"))
    # ---- E
    a(H("E01", 1, lambda d: (d["s_n"] >= 100), lambda d: (d["sc_n"] >= 30) & (d["sc_r"] - d["s_r"] >= 0.05), "父のそのコースの as-of 3 着内率が父の通算より 5pt 以上高い"))
    a(H("E02", 1, lambda d: d["bad"] & (d["s_n"] >= 100), lambda d: (d["sbad_n"] >= 30) & (d["sbad_r"] - d["s_r"] >= 0.03), "重・不良で、父の道悪の as-of 3 着内率が通算より 3pt 以上高い"))
    a(H("E03", 1, lambda d: d["vc"].isin(["01", "02"]) & (d["surf"] == "芝") & (d["s_n"] >= 100), lambda d: (d["shk_n"] >= 30) & (d["shk_r"] - d["s_r"] >= 0.03), "北海道の芝で、父の北海道芝の as-of 3 着内率が通算より 3pt 以上高い"))
    a(H("E04", 1, lambda d: (d["surf"] == "芝") & d["cush"].notna() & (d["cush"] <= d["cush_lo"]) & d["cl_cmb"].notna() & (d["cl_cmb"] != "<NA>"), lambda d: d["cl_cmb"].str.contains("欧", na=False), "クッション値が下 1/3 の芝で、血統の型が欧州型"))
    a(H("E05", 1, lambda d: (d["dist"] - d["p_dist"] >= 200) & d["mf_meandist"].notna(), lambda d: d["mf_meandist"] >= 1900, "200m 以上の延長で、母父の産駒の勝ち距離の平均が 1900m 以上"))
    a(H("E06", 1, lambda d: (d["surf"] == "ダ") & (d["surf_n_prev"] == 0) & (d["starts_prev"] >= 1) & (d["s_n"] >= 100), lambda d: (d["sdirt_n"] >= 50) & (d["sdirt_r"] - d["s_r"] >= 0.03), "初ダートで、父のダートの as-of 3 着内率が通算より 3pt 以上高い"))
    a(H("E07", 1, lambda d: (d["since_rest"] == 2) & (d["s_n"] >= 100), lambda d: (d["sk2_n"] >= 30) & (d["sk2_r"] - d["s_r"] >= 0.03), "休み明け 2 戦目で、父の叩き 2 戦目の as-of 3 着内率が通算より 3pt 以上高い"))
    a(H("E08", 1, lambda d: d["month"].between(6, 9) & (d["s_n"] >= 100), lambda d: (d["ssum_n"] >= 30) & (d["ssum_r"] - d["s_r"] >= 0.03), "6〜9 月で、父の夏の as-of 3 着内率が通算より 3pt 以上高い"))
    for hid, why in (("E09", "クロスの位置の表（pedigree_position_cross）を今回の表に結合していない（既存の章で調査済み）"), ("E10", "母（きょうだい）の名前を今回の表に入れていない（次の版）"),
                     ("E11", "既存の章 dam_age_firstfoal で調査済み"), ("E12", "遺伝子型の被覆が薄い"), ("E14", "牝系の分類が無い")):
        a(H(hid, 0, None, None, "", why))
    a(H("E13", 1, lambda d: (d["age"] == 2) & d["month"].between(6, 9) & d["f_1400"].notna(), lambda d: d["f_1400"] >= 6, "2 歳 6〜9 月で、父の産駒の 1400 以下の比率が上位"))
    a(H("E15", 1, lambda d: d["s_n"] >= 100, lambda d: (d["nick_n"] >= 20) & (d["nick_r"] - d["s_r"] >= 0.05), "父 × 母父の as-of 3 着内率が父の通算より 5pt 以上高い"))
    a(H("E16", 0, None, None, "", "父の現役の戦績が手元に無い"))
    a(H("E17", -1, lambda d: (d["dist"] >= 2000) & d["mf_meandist"].notna(), lambda d: d["mf_meandist"] <= 1500, "2000m 以上で、母父の産駒の勝ち距離の平均が 1500m 以下"))
    a(H("E18a", 1, lambda d: (d["surf"] == "芝") & d["line"].notna() & ((d["shape"] == "小回り") | (d["meet_day"] >= d["meet_len"] - 1)), lambda d: d["line"] == "ロベルト系", "小回りか開催の最後の 2 日の芝でロベルト系"))
    a(H("E18b", 1, lambda d: (d["surf"] == "芝") & d["line"].notna() & (((d["vc"] == "08") & (d["io"] == "外")) | (d["vc"] == "05")), lambda d: d["line"] == "ディープ系", "京都外回り・東京の芝でディープ系"))
    a(H("E18c", 1, lambda d: (d["surf"] == "芝") & d["line"].notna() & ((d["vc"] == "06") | ((d["vc"] == "09") & (d["io"] == "内"))), lambda d: d["line"] == "ステイゴールド系", "中山・阪神内回りの芝でステイゴールド系"))
    a(H("E18d", 1, lambda d: (d["surf"] == "芝") & d["line"].notna() & ((d["vc"] == "06") | d["bad"]), lambda d: d["line"] == "キングマンボ系", "中山か重・不良の芝でキングマンボ系"))
    a(H("E18e", 1, lambda d: (d["surf"] == "ダ") & (d["dist"] <= 1400) & d["cl_cmb"].notna() & (d["cl_cmb"] != "<NA>"), lambda d: d["cl_cmb"].str.contains("米", na=False), "ダート 1400 以下で血統の型が米国型"))
    a(H("E19", 1, lambda d: (d["age"] == 2) & d["sire_first_by"].notna(), lambda d: (d["by"] == d["sire_first_by"]) & (d["sire_first_by"] >= 2015), "2 歳で父の初年度産駒（初年度が 2015 年生まれ以降の父）"))
    a(H("E20", 1, lambda d: (d["age"] >= 5) & (d["s_n"] >= 100), lambda d: (d["sold_n"] >= 30) & (d["sold_r"] - d["s_r"] >= 0.03), "5 歳以上で、父の 5 歳以上の as-of 3 着内率が通算より 3pt 以上高い"))
    a(H("E21", 0, None, None, "", "母の現役の戦績が手元に無い"))
    # ---- F
    a(H("F01", 1, lambda d: (d["surf"] == "芝") & (d["meet_day"] <= 2), lambda d: d["inner"], "開催 1〜2 日目の芝で内枠"))
    a(H("F01b", 1, lambda d: (d["surf"] == "芝") & (d["meet_day"] >= d["meet_len"] - 1) & (d["meet_len"] >= 6), lambda d: d["outer"], "開催の最後の 2 日の芝で外枠"))
    a(H("F02", 0, None, None, "", "仮柵の位置（コース替わり）を今回の表に入れていない"))
    a(H("F03", 1, lambda d: (d["n_before"] >= 3) & d["bias_pos"].notna() & d["senko_rate"].notna() & (d["bias_pos"] <= 0.3), lambda d: d["senko_rate"] >= 0.5, "同じ日・同じ面の前のレースで 3 着内の 4 角位置が前 3 割 → 先行馬"))
    a(H("F03b", 1, lambda d: (d["n_before"] >= 3) & d["bias_hn"].notna() & (d["bias_hn"] <= 0.4), lambda d: d["inner"], "同じ日・同じ面の前のレースの 3 着内が内枠寄り → 内枠"))
    a(H("F04", 1, lambda d: d["prevday_bias"].notna() & (d["prevday_bias"] <= 0.3) & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "前の開催日の同じ場・面で前残り → 先行馬"))
    a(H("F05", 1, lambda d: (d["surf"] == "ダ") & d["track"].isin(["稍", "重", "不"]) & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "ダートの稍重〜不良で先行馬"))
    a(H("F06", 1, lambda d: (d["surf"] == "芝") & d["bad"] & (d["meet_day"] >= 5), lambda d: d["outer"], "芝の重・不良・開催 5 日目以降で外枠"))
    a(H("F07", 1, lambda d: (d["surf"] == "芝") & (d["cush"] >= d["cush_hi"]) & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "クッション値が上 1/3 の芝で先行馬"))
    a(H("F08", 1, lambda d: (d["rain_cum"] >= 1) & (d["bad_n_prev"] >= 1), lambda d: d["bad_top3_prev"] / d["bad_n_prev"] >= 0.3, "当日の発走までの降水 1mm 以上で、道悪の 3 着内率 30% 以上の馬"))
    a(H("F09", -1, lambda d: (d["wind"] >= 6) & d["nige_rate"].notna(), lambda d: d["nige_rate"] >= 0.3, "風速 6m 以上で逃げ馬"))
    a(H("F10", -1, lambda d: d["tmax"] >= 30, lambda d: d["sex"] == "牝", "最高気温 30℃以上の日の牝馬"))
    a(H("F11", 1, lambda d: (d["surf"] == "ダ") & d["month"].isin([12, 1, 2]) & d["wt"].notna(), lambda d: d["wt"] >= 500, "冬のダートで 500kg 以上"))
    a(H("F12", 0, None, None, "", "改修の前後の比較は期間そのものの比較で、発見/確認の分け方と両立しない"))
    a(H("F13", 1, lambda d: d["bad"] & (d["bad_n_prev"] >= 1), lambda d: d["bad_top3_prev"] / d["bad_n_prev"] >= 0.3, "重・不良で、道悪の 3 着内率 30% 以上の馬"))
    a(H("F14", 1, lambda d: (d["surf"] == "芝") & (d["meet_day"] >= 7), lambda d: d["outer"], "開催 7 日目以降の芝で外枠"))
    a(H("F15", 1, lambda d: (d["day_fast"] <= -0.5) & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "同じ日・同じ面の前のレースの勝ち時計が基準より 0.5 秒以上速い → 先行馬"))
    a(H("F16", 1, lambda d: (d["surf"] == "ダ") & (d["track"] == "良") & d["month"].isin([12, 1, 2]) & (d["dmoist"] <= d["dm_lo"]) & d["senko_rate"].notna(), lambda d: d["senko_rate"] <= 0.2, "冬の乾いた良ダートで差し馬"))
    a(H("F17", 1, lambda d: (d["rain_cum"] >= 1) & d["nige_rate"].notna(), lambda d: d["nige_rate"] >= 0.3, "当日の発走までの降水 1mm 以上で逃げ馬"))
    a(H("F18", 1, lambda d: (d["track"] == "良") & (((d["surf"] == "芝") & (d["tmoist"] >= d["tm_hi"])) | ((d["surf"] == "ダ") & (d["dmoist"] >= d["dm_hi"]))) & (d["bad_n_prev"] >= 1),
        lambda d: d["bad_top3_prev"] / d["bad_n_prev"] >= 0.3, "発表は良でも含水率が上 1/3 で、道悪の 3 着内率 30% 以上の馬"))
    a(H("F19", 0, None, None, "", "芝の管理の暦が無い"))
    # ---- G
    a(H("G01", 1, lambda d: d["hnr"].notna(), lambda d: d["inner"], "内枠（馬番が内 1/3）。コース別は切り口の『コース』で見る"))
    a(H("G01b", 1, lambda d: d["hnr"].notna(), lambda d: d["outer"], "外枠（馬番が外 1/3）"))
    a(H("G02", 1, lambda d: d["course"].isin(TURF_START_DIRT), lambda d: d["outer"], "芝スタートのダートで外枠"))
    a(H("G03", 1, lambda d: (d["n_nige_cand"] - d["nige_cand"] == 0) & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "ほかに逃げ候補がいないレースの先行馬"))
    a(H("G04", 1, lambda d: (d["n_nige_cand"] >= 3) & d["senko_rate"].notna(), lambda d: d["senko_rate"] <= 0.2, "逃げ候補 3 頭以上のレースの差し馬"))
    a(H("G05", -1, lambda d: (d["senko_rate"] >= 0.5) & (d["nige_cand"] == 0), lambda d: d["nb_nige"], "先行馬で、隣の馬番に逃げ候補"))
    a(H("G06", 1, lambda d: d["p_pos1"] == 1, lambda d: d["n_nige_cand"] - d["nige_cand"] == 0, "前走 1 角先頭の馬で、ほかに逃げ候補がいない"))
    a(H("G07", 1, lambda d: (d["field"] <= 10) & (d["starts_prev"] >= 3), lambda d: d["agari1_prev"] / d["starts_prev"] >= 0.2, "10 頭以下で、上がり最速の割合が 2 割以上の馬"))
    a(H("G08", -1, lambda d: d["field"] >= 16, lambda d: d["hn"] >= d["field"] - 1, "16 頭以上の大外 2 頭"))
    a(H("G09", -1, lambda d: (d["pop"] <= 3) & (d["starts_prev"] >= 3), lambda d: d["pos1r_mean_prev"] >= 0.75, "3 番人気以内で、1 角の位置の平均が後ろ 25%"))
    for hid, why in (("G10", "好走時のラップ型 × そのコースで起きやすいラップ型の対応は lap_forecast の予想が要る（今回は対象外）"), ("G11", "上がり順位の履歴のばらつきを表に入れていない"),
                     ("G12", "スタートが下り坂のコースの一覧が無い")):
        a(H(hid, 0, None, None, "", why))
    a(H("G13", 1, lambda d: d["hn"].notna(), lambda d: d["hn"] % 2 == 0, "偶数の馬番"))
    a(H("G14", -1, lambda d: (d["field"] >= 14) & (d["senko_rate"] <= 0.15), lambda d: d["inner"], "14 頭以上の追込馬で内枠"))
    a(H("G15", -1, lambda d: d["nige_cand"] == 1, lambda d: d["n_out"] >= 1, "逃げ候補で、外に別の逃げ候補"))
    a(H("G16", -1, lambda d: d["nige_cand"] == 1, lambda d: d["hn"] >= d["field"] - 1, "逃げ候補で大外 2 頭"))
    a(H("G17", 1, lambda d: (d["nige_prev"] >= 2) & (d["starts_prev"] >= 4), lambda d: (d["nige_top3_prev"] / d["nige_prev"] - d["top3_prev"] / d["starts_prev"] >= 0.2) & (d["n_nige_cand"] - d["nige_cand"] == 0), "逃げた時の 3 着内率が通算より 20pt 以上高い馬で、ほかに逃げ候補がいない"))
    for hid, why in (("G18", "馬ごとの枠・頭数別の過去成績を表に入れていない"), ("G19", "砂を被ったかどうかの記録が無い")):
        a(H(hid, 0, None, None, "", why))
    a(H("G20", -1, lambda d: (d["shape"] == "小回り") & d["senko_rate"].notna(), lambda d: d["senko_rate"] <= 0.2, "小回りの差し・追込馬"))
    a(H("G21", 1, lambda d: d["hnr"].notna(), lambda d: d["hnr"] <= 0.5, "内半分の馬番"))
    # ---- H
    a(H("H01a", 1, lambda d: (d["since_rest"] == 1) & d["wtc"].notna() & (d["age"] <= 3), lambda d: d["wtc"] >= 10, "3 歳以下の休み明けで +10kg 以上（成長分）"))
    a(H("H01b", -1, lambda d: (d["since_rest"] == 1) & d["wtc"].notna() & (d["age"] >= 4), lambda d: d["wtc"] >= 10, "4 歳以上の休み明けで +10kg 以上（太め残り）"))
    a(H("H02", -1, lambda d: d["wtc"].notna(), lambda d: d["wtc"] <= -10, "−10kg 以上"))
    a(H("H03", 1, lambda d: (d["bad"] | (d["surf"] == "ダ")) & d["wt"].notna(), lambda d: d["wt"] >= 520, "重・不良かダートで 520kg 以上"))
    a(H("H04", -1, lambda d: d["jw"].notna(), lambda d: d["jw"] - d["jw_mean"] >= 2, "斤量がレースの平均より 2kg 以上重い（2023 年の改定をまたいでも比べられる形）"))
    a(H("H04b", 1, lambda d: d["weight_rule"] == "ハンデ", lambda d: d["jw"] - d["jw_mean"] <= -2, "ハンデ戦で斤量が平均より 2kg 以上軽い"))
    a(H("H05", 1, lambda d: d["blinker"].notna() & (d["starts_prev"] >= 1), lambda d: (d["blinker"] == 1) & (d["blk_ever_prev"] == 0), "ブリンカー初装着（2020〜2025）"))
    a(H("H06", 1, lambda d: d["since_geld"] <= 5, lambda d: d["since_geld"].isin([2, 3]), "去勢後 1〜5 戦目の中で 2〜3 戦目"))
    a(H("H07", -1, lambda d: (d["sex"] == "牝") & (d["age"] >= 4), lambda d: ((d["age"] == 4) & (d["month"] >= 9)) | (d["age"] >= 5), "4 歳以上の牝馬で、4 歳秋以降"))
    a(H("H08", -1, lambda d: d["grade"].isin(G), lambda d: d["age"] >= 7, "重賞で 7 歳以上"))
    a(H("H08b", 1, lambda d: (d["surf"] == "ダ") & ~d["grade"].isin(G) & (d["age"] >= 4), lambda d: d["age"] >= 7, "重賞以外の古馬ダートで 7 歳以上"))
    a(H("H09", -1, lambda d: (d["sex"] == "牝") & d["p_fillies"].notna() & (d["n_prev"] >= 1), lambda d: d["p_fillies"] & ~d["fillies"], "牝馬で、前走牝馬限定 → 今回混合"))
    a(H("H10", 1, lambda d: d["pd_rank"].notna(), lambda d: d["pd_rank"] == 1, "パドックの評価が 1（被覆のある年だけ）"))
    a(H("H11", 1, lambda d: d["tc_has"] == 1, lambda d: d["tc_pos"] == 1, "厩舎コメントに強気の語（2020〜2025）"))
    a(H("H12", -1, lambda d: d["tc_has"] == 1, lambda d: d["tc_next"] == 1, "厩舎コメントに『使いつつ』『叩いて』など"))
    a(H("H13", 1, lambda d: (d["age"] == 3) & (d["month"] >= 9) & (d["since_rest"] == 1) & d["wtc"].notna(), lambda d: d["wtc"] >= 10, "3 歳秋の休み明けで +10kg 以上"))
    a(H("H14", -1, lambda d: d["best_wt"].notna() & (d["age"] >= 4) & d["wt"].notna(), lambda d: (d["wt"] - d["best_wt"]).abs() >= 10, "4 歳以上で、3 着内の時の体重の平均から 10kg 以上離れる"))
    a(H("H15", -1, lambda d: d["p3_wt"].notna() & d["wt"].notna(), lambda d: (d["wt"] < d["p_wt"]) & (d["p_wt"] < d["p2_wt"]) & (d["p2_wt"] < d["p3_wt"]), "3 走続けて体重が減っている"))
    a(H("H16", -1, lambda d: (d["sex"] == "牝") & ((d["field"] >= 16) | d["bad"]) & d["wt"].notna(), lambda d: d["wt"] < 420, "16 頭以上か重・不良の牝馬で 420kg 未満"))
    a(H("H17", 0, None, None, "", "ブリンカー以外の馬具が無い"))
    a(H("H18", 1, lambda d: d["p_blinker"] == 1, lambda d: d["blinker"] == 0, "前走ブリンカー → 今回外した"))
    a(H("H19", -1, lambda d: (d["since_rest"] == 1) & (d["rest_len"] >= 70), lambda d: d["rest_len"] >= 365, "休み明け初戦で 1 年以上の休養"))
    a(H("H20", -1, lambda d: d["month"].between(3, 5) & (d["age"] >= 4) & ~d["fillies"], lambda d: d["sex"] == "牝", "3〜5 月の古馬の混合戦の牝馬"))
    a(H("H21", 1, lambda d: d["age"] >= 4, lambda d: d["sex"] == "セ", "4 歳以上のセン馬"))
    # ---- I
    for hid, why in (("I01", "重賞ごとの標本は年 1 回で 10 年でも 10 レース。A27・A26 などで代表させる"), ("I02", "重賞ごとの標本が小さい（B07 で重賞全体を見る）")):
        a(H(hid, 0, None, None, "", why))
    a(H("I03", 1, lambda d: (d["weight_rule"] == "ハンデ") & d["senko_rate"].notna(), lambda d: (d["jw"] - d["jw_mean"] <= -1) & (d["senko_rate"] <= 0.3), "ハンデ戦で軽ハンデ（平均 −1kg 以下）の差し馬"))
    a(H("I04", 1, lambda d: d["fillies"] & (d["starts_prev"] >= 3), lambda d: d["agari1_prev"] / d["starts_prev"] >= 0.2, "牝馬限定戦で、上がり最速の割合が 2 割以上の馬"))
    a(H("I05", -1, lambda d: d["pop"] <= 3, lambda d: d["round"] == 12, "3 番人気以内で最終レース"))
    a(H("I05b", 1, lambda d: d["pop"] <= 3, lambda d: d["round"] == 1, "3 番人気以内で 1R"))
    a(H("I06", 1, lambda d: (d["age"] == 3) & (d["grade"] == "未勝利") & d["month"].between(7, 9) & d["p_vc"].notna(), lambda d: ~d["p_local"] & d["local"], "3 歳未勝利の 7〜9 月で中央場 → ローカル"))
    a(H("I07", 1, lambda d: (d["grade"] == "新馬") & d["ok_eval"].notna(), lambda d: d["ok_eval"] == "A", "新馬で最終追い切りの評価 A"))
    a(H("I08", 1, lambda d: (d["age"] == 2) & d["starts_prev"].between(1, 2), lambda d: d["starts_prev"] == 1, "2 歳の 2〜3 戦目で 2 戦目"))
    for hid, why in (("I09", "『次走が G1』は出走前に分からない（トライアルの一覧が無い）"), ("I10", "特別戦と平場の好走馬の型の違いは型として定まらない"),
                     ("I12", "収得賞金が無い"), ("I13", "優先出走権の記録が無い"), ("I14", "世代の強さの指標を as-of で作っていない"), ("I17", "3 冠は年 1 回で標本が無い"),
                     ("I20", "F01・F01b と同じ（開催の初日・最終日）")):
        a(H(hid, 0, None, None, "", why))
    a(H("I11", -1, lambda d: d["p_surf"].notna(), lambda d: d["p_jump"], "前走が障害"))
    a(H("I11b", -1, lambda d: d["pp_last_region"].notna(), lambda d: d["pp_last_region"] == "地方", "出馬表の前走が地方（2020〜2025）"))
    a(H("I15", 1, lambda d: (d["cls"] >= 5) & d["member_op"].notna() & (d["member_op"] <= d.groupby("cls")["member_op"].transform(lambda s: s.quantile(1 / 3))), lambda d: d["op_n_prev"] >= 3, "OP 以上でメンバーの OP 経験の合計が下 1/3 のレースで、OP 経験 3 回以上の馬"))
    a(H("I16", 1, lambda d: d["mixed"] & d["cls"].between(2, 4) & d["age"].between(3, 5) & d["month"].between(6, 12), lambda d: d["age"] == 4, "6〜12 月の古馬条件戦で 4 歳（3・5 歳と比べる）"))
    a(H("I18", 1, lambda d: d["grade"].isin(G) & d["month"].between(9, 11) & (d["age"] == 3), lambda d: (d["win_streak"] >= 2) & d["p_month"].between(7, 9), "3 歳秋の重賞で、夏に連勝してきた"))
    a(H("I19", -1, lambda d: (d["pop"] == 1) & d["weight_rule"].isin(["別定", "定量"]), lambda d: d["weight_rule"] == "別定", "1 番人気で別定戦（定量戦と比べる）"))
    # ---- J
    a(H("J01", 1, lambda d: d["mg_rank_p1"].notna(), lambda d: d["mg_rank_p1"] == 1, "前走の指数（めぐ指数）がレース内 1 位"))
    a(H("J02", -1, lambda d: d["mg_rank_max5"] == 1, lambda d: d["mg_rank_mean3"] >= 4, "直近 5 走の最高指数が 1 位で、近 3 走の平均が 4 位以下"))
    a(H("J03", 1, lambda d: d["n_nige_cand"] >= 3, lambda d: d["agari1_prev"] >= 3, "逃げ候補 3 頭以上で、上がり最速 3 回以上"))
    a(H("J04", 1, lambda d: d["si_course_rank"].notna(), lambda d: d["si_course_rank"] == 1, "コース別のスピード指数がレース内 1 位"))
    a(H("J05", 1, lambda d: d["ten1f_best_rank"].notna(), lambda d: d["ten1f_best_rank"] == 1, "テン 1F の持ち時計の順位 1 位（SmartRC）"))
    a(H("J06", 0, None, None, "", "バロメーターの表はレースの結果の後の値（着順 finish_order と同じ行にあり、新馬にも値があり、着順との順位相関 −0.57）。出走前の情報ではないので除外（試しに検定すると +34〜46pt の『差』が出るが、先読み）"))
    a(H("J07", 1, lambda d: d["mg_sd5"].notna() & (d["pop"] <= 3), lambda d: d["mg_sd_q25"], "3 番人気以内で、直近 5 走の指数のばらつきが下 25%"))
    a(H("J08", 1, lambda d: (d["cls"] >= 5) & d["mg_rank_mean3"].notna(), lambda d: (d["mg_rank_mean3"] <= 2) & (d["op_n_prev"] <= 2), "OP 以上で、近 3 走の指数の平均が 2 位以内かつ OP 経験 2 回以下（格より調子）"))
    a(H("J09", 1, lambda d: (d["age"] == 2) & (d["p_y1"] == 1) & d["p_race_fast"].notna(), lambda d: d["p_race_fast"] <= -1.0, "2 歳で、前走の勝ちタイムがコース・クラスの基準より 1 秒以上速い"))
    a(H("J10", 1, lambda d: (d["surf"] == "芝") & ((d["vc"] == "05") | ((d["vc"] == "04") & (d["io"] == "外"))) & d["best_l3f_prev"].notna(), lambda d: d["best_l3f_prev"] < 34.0, "東京・新潟外回りの芝で、上がり 3F の最速が 33 秒台以下"))
    a(H("J11", -1, lambda d: d["mg_maxall"].notna() & d["mg_max5"].notna(), lambda d: d["mg_max5"] < d["mg_maxall"] - 3, "通算の最高指数が直近 5 走の最高より 3 以上高い（ピークが古い）"))
    a(H("J12", 1, lambda d: d["mg_p3"].notna(), lambda d: (d["mg_p1"] > d["mg_p2"]) & (d["mg_p2"] > d["mg_p3"]), "指数が 3 走続けて上昇"))
    # ---- K
    a(H("K01", 1, lambda d: d["p_dist"].notna() & (d["p_surf"] == d["surf"]), lambda d: (d["dist"] - d["p_dist"] >= 100) & (d["p_pos4r"] <= 0.3) & (d["p_fin"] <= 5), "延長 × 前走 4 角前 3 割で掲示板（父ステイヤーは抜き）"))
    a(H("K02", 0, None, None, "", "最終追い切りの乗り手（2016〜2019）と生産者（2020〜2025）の年が重ならない"))
    a(H("K03", 1, lambda d: sw(d) & d["ok_rider"].notna() & d["j_rate"].notna() & d["pj_rate"].notna(), lambda d: (d["j_rate"] - d["pj_rate"] >= 0.08) & is_j(d), "上位騎手への乗り替わり × その騎手が最終追い切りに騎乗"))
    a(H("K04", 1, lambda d: (d["n_before"] >= 3) & (d["bias_pos"] <= 0.3), lambda d: d["inner"] & (d["nige_cand"] == 1), "当日前残り × 内枠 × 逃げ候補"))
    a(H("K05", 1, lambda d: (d["surf"] == "ダ") & (d["surf_n_prev"] == 0) & (d["starts_prev"] >= 1) & d["f_dirt"].notna(), lambda d: (d["f_dirt"] >= 6) & d["outer"], "初ダート × 父ダート × 外枠"))
    a(H("K06", 1, lambda d: d["bad"] & (d["s_n"] >= 100) & d["wt"].notna(), lambda d: (d["sbad_n"] >= 30) & (d["sbad_r"] - d["s_r"] >= 0.03) & (d["wt"] >= 500), "道悪 × 父の道悪 × 500kg 以上"))
    a(H("K07", 1, lambda d: (d["since_rest"] == 2) & (d["rest_len"] >= 70) & d["ok_int"].isin(list(up)) & d["p_ok_int"].isin(list(up)) & d["wtc"].notna(), lambda d: (d["ok_int"].map(up) > d["p_ok_int"].map(up)) & (d["wtc"] < 0), "叩き 2 戦目 × 最終の強度が上がる × 体重減"))
    a(H("K08", 0, None, None, "", "I01・I02 と同じ理由（重賞ごとの標本）"))
    a(H("K09", 1, lambda d: prom(d) & d["p_race_fast"].notna(), lambda d: (d["p_race_fast"] <= -0.5) & (d["j_cont"] >= 1), "昇級初戦 × 前走の時計が速い × 継続騎乗"))
    a(H("K10", 1, lambda d: (d["sex"] == "牝") & d["month"].between(6, 9) & d["local"] & (d["ok_course"].fillna("") != ""), lambda d: d["ok_course"].fillna("").str[0] == d["venue"].str[0], "夏のローカルの牝馬 × 滞在（現地で最終追い切り）"))
    a(H("K11", 1, lambda d: (d["rest_len"] >= 70) & (d["since_rest"] <= 4) & d["p_dist"].notna(), lambda d: (d["since_rest"] == 2) & (d["dist"] - d["p_dist"] >= 100), "叩き 2 戦目 × 延長"))
    a(H("K12", 0, None, None, "", "D13（10 日競馬）が作れない"))
    a(H("K13", 1, lambda d: d["p_pos1r"].notna() & d["pos1r_mean_prev"].notna() & (d["starts_prev"] >= 3), lambda d: (d["p_pos1r"] - d["pos1r_mean_prev"] >= 0.35) & (d["j_cont"] >= 1) & (d["course_n_prev"] >= 1), "前走の不利 × 継続騎乗 × コース経験"))
    a(H("K14", 1, lambda d: (d["pop"] >= 6) & (d["nige_cand"] == 1), lambda d: (d["n_nige_cand"] == 1) & (d["meet_day"] <= 2) & (d["hnr"] <= 0.5), "人気薄の逃げ候補 × 単騎 × 開幕 2 日 × 内半分"))
    a(H("K15", 1, lambda d: d["blinker"].notna() & d["p_dist"].notna() & (d["starts_prev"] >= 1), lambda d: (d["blinker"] == 1) & (d["blk_ever_prev"] == 0) & (d["p_dist"] - d["dist"] >= 100), "初ブリンカー × 短縮"))
    a(H("K16", 1, lambda d: d["allow"].notna() & (d["surf"] == "ダ") & (d["dist"] <= 1300) & d["senko_rate"].notna(), lambda d: (d["allow"] >= 1) & d["outer"] & (d["senko_rate"] >= 0.5), "減量 × ダ 1300 以下 × 外枠 × 先行"))
    a(H("K17", 1, lambda d: d["month"].between(6, 9) & prom(d), lambda d: d["sex"] == "牝", "夏の昇級初戦の牝馬"))
    a(H("K18", 1, lambda d: sw(d) & d["j_rate"].notna() & d["pj_rate"].notna() & d["p_pop"].notna(), lambda d: (d["j_rate"] - d["pj_rate"] >= 0.08) & (d["pop"] - d["p_pop"] >= 3), "鞍上強化 × 前走より人気が 3 以上下がった"))
    # ---- L
    a(H("L01", 1, lambda d: d["month"].between(6, 9) & ~d["fillies"], lambda d: d["sex"] == "牝", "6〜9 月の混合戦の牝馬"))
    a(H("L02", 1, prom, lambda d: d["sex"] == "牝", "昇級初戦の牝馬"))
    a(H("L03", -1, lambda d: d["pop"] <= 2, lambda d: d["top2q"] >= 0.6, "1・2 番人気で、2 頭の単勝の市場確率の合計が 60% 以上（2 強）"))
    a(H("L04", 1, lambda d: d["pop"] == 1, lambda d: d["odds"] >= 3.5, "1 番人気で単勝 3.5 倍以上（混戦）"))
    a(H("L05", 1, lambda d: d["pop"] >= 6, lambda d: d["nige_cand"] == 1, "6 番人気以下の逃げ候補"))
    a(H("L06", 1, lambda d: d["dist"] <= 1400, lambda d: d["inner"], "1400m 以下の内枠（短距離は枠）"))
    a(H("L07", 1, lambda d: d["pop"] >= 10, lambda d: d["round"] == 12, "10 番人気以下で最終レース"))
    a(H("L08", 1, lambda d: (d["rest_len"] >= 70) & (d["since_rest"] <= 4) & (d["pop"] >= 6), lambda d: d["since_rest"] == 3, "人気薄の休み明け 1〜4 戦目で 3 戦目"))
    a(H("L09", 1, lambda d: d["p_fin"] >= 6, lambda d: d["p_tdiff"] <= 0.6, "前走 6 着以下で勝ち馬から 0.6 秒以内"))
    a(H("L10", 1, lambda d: (d["surf"] == "ダ") & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "ダートの先行馬"))
    a(H("L10b", 1, lambda d: (d["surf"] == "芝") & d["senko_rate"].notna(), lambda d: d["senko_rate"] <= 0.2, "芝の差し馬"))
    # ---- M
    a(H("M01", -1, lambda d: d["starts_prev"] >= 3, lambda d: d["venue_n_prev"] == 0, "初めての競馬場"))
    a(H("M02", -1, lambda d: (d["starts_prev"] >= 2) & (d["age"] >= 4) & d["dist_max_prev"].notna(), lambda d: (d["dist"] > d["dist_max_prev"] + 200) | (d["dist"] < d["dist_min_prev"] - 200), "4 歳以上で経験の外の距離（±200m 超）"))
    a(H("M03", -1, lambda d: d["cls"] >= 5, lambda d: d["op_n_prev"] == 0, "OP・重賞が初めて"))
    a(H("M04", -1, lambda d: (d["age"] == 3) & d["mixed"], lambda d: d["vs_old_prev"] == 0, "3 歳の古馬混合の初戦"))
    a(H("M05", -1, lambda d: (d["away"] == 1) & (d["starts_prev"] >= 2), lambda d: d["away_n_prev"] == 0, "初めての遠征"))
    a(H("M06", -1, lambda d: (d["field"] >= 16) & d["max_field_prev"].notna(), lambda d: d["max_field_prev"] < d["field"] - 2, "16 頭以上で、経験した最多の頭数より 3 頭以上多い"))
    a(H("M07", 1, lambda d: d["bad"] & (d["bad_n_prev"] == 0) & (d["starts_prev"] >= 1) & (d["s_n"] >= 100), lambda d: (d["sbad_n"] >= 30) & (d["sbad_r"] - d["s_r"] >= 0.03), "初めての道悪で、父の道悪の成績が良い"))
    a(H("M08", -1, lambda d: d["vc"].isin(["01", "02"]) & (d["surf"] == "芝") & (d["starts_prev"] >= 2), lambda d: d["yoshiba_n_prev"] == 0, "初めての洋芝"))
    a(H("M09", -1, lambda d: (d["pop"] == 1) & (d["starts_prev"] >= 1), lambda d: d["fav1_prev"] == 0, "初めての 1 番人気"))
    # ---- N（上位人気に限る）
    top = lambda d: d["pop"] <= 3  # noqa: E731
    a(H("N01", -1, lambda d: top(d) & d["p_pop"].notna(), lambda d: (d["p_pop"] >= 8) & (d["p_fin"] <= 3), "上位人気で、前走 8 番人気以下で 3 着内"))
    a(H("N02", -1, lambda d: top(d) & d["p_race_fast"].notna(), lambda d: (d["p_race_fast"] <= -1.0) & (d["p_fin"] <= 3), "上位人気で、前走が基準より 1 秒以上速い決着で 3 着内"))
    a(H("N03", -1, lambda d: top(d) & prom(d), lambda d: (d["p_pos1"] == 1) & (d["p_pace_cat"] == "S"), "上位人気の昇級初戦で、前走スローの逃げ切り"))
    a(H("N04", -1, lambda d: top(d) & (d["senko_rate"] <= 0.3), lambda d: d["hn"] >= d["field"] - 1, "上位人気の差し馬で大外 2 頭"))
    a(H("N05", -1, lambda d: top(d) & (d["since_rest"] == 1) & (d["age"] >= 4) & d["wtc"].notna(), lambda d: d["wtc"] >= 12, "上位人気の 4 歳以上の休み明けで +12kg 以上"))
    a(H("N06", -1, lambda d: top(d) & d["p_jw"].notna(), lambda d: d["jw"] - d["p_jw"] >= 2, "上位人気で斤量が前走より 2kg 以上重い"))
    a(H("N07", -1, lambda d: top(d) & sw(d) & d["j_rate"].notna() & d["pj_rate"].notna(), lambda d: d["pj_rate"] - d["j_rate"] >= 0.08, "上位人気で鞍上が 8pt 以上弱くなる"))
    a(H("N08", -1, lambda d: top(d) & d["p_pace_cat"].notna(), lambda d: (((d["p_pace_cat"] == "S") & (d["p_pos4r"] <= 0.2)) | ((d["p_pace_cat"] == "H") & (d["p_pos4r"] >= 0.7))) & (d["p_fin"] <= 3), "上位人気で、前走が展開に恵まれた 3 着内（スローの前・ハイの後ろ）"))
    a(H("N09", -1, lambda d: top(d) & (d["senko_rate"] <= 0.2), lambda d: d["n_nige_cand"] <= 1, "上位人気の追込馬で、逃げ候補 1 頭以下（スロー想定）"))
    a(H("N10", -1, lambda d: top(d) & (d["sex"] == "牝") & d["interval"].notna(), lambda d: d["interval"] <= 14, "上位人気の牝馬で間隔 14 日以内"))
    # ---- P
    a(H("P01", 1, lambda d: d["pop"] == 1, lambda d: d["course"] == "東京芝1800", "1 番人気で東京芝 1800（他のコースと比べる）"))
    a(H("P02", 1, lambda d: (d["vc"] == "06") & (d["vs_n_prev"] >= 2), lambda d: d["vs_top3_prev"] / d["vs_n_prev"] >= 0.4, "中山で、同じ場・面の 3 着内率 40% 以上（中山巧者）"))
    a(H("P03", 1, lambda d: (d["vc"] == "08") & (d["io"] == "外") & d["senko_rate"].notna(), lambda d: d["senko_rate"] <= 0.3, "京都外回りの差し馬"))
    a(H("P04", 1, lambda d: (d["io"] == "内") & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "内回りの先行馬"))
    a(H("P05", 1, lambda d: d["vc"].isin(["03", "10"]) & (d["surf"] == "芝") & (d["meet_day"] <= 4) & d["senko_rate"].notna(), lambda d: d["senko_rate"] >= 0.5, "福島・小倉の芝の開催前半の先行馬"))
    a(H("P05b", 1, lambda d: d["vc"].isin(["03", "10"]) & (d["surf"] == "芝") & (d["meet_day"] >= d["meet_len"] - 3) & (d["meet_len"] >= 6) & d["senko_rate"].notna(), lambda d: d["senko_rate"] <= 0.3, "福島・小倉の芝の開催後半の差し馬"))
    a(H("P06", 1, lambda d: (d["vc"] == "04") & (d["io"] == "外") & d["best_l3f_prev"].notna(), lambda d: d["best_l3f_prev"] < 33.5, "新潟外回りで、上がり 3F の最速が 33.4 秒以下"))
    a(H("P07", 1, lambda d: (d["vc"] == "07") & (d["surf"] == "芝") & d["senko_rate"].notna(), lambda d: d["senko_rate"] <= 0.3, "中京芝の差し馬"))
    a(H("P08", 1, lambda d: d["vc"].isin(["01", "02"]) & (d["surf"] == "芝") & (d["hk_n_prev"] >= 1), lambda d: d["hk_top3_prev"] >= 1, "北海道の芝で、北海道の芝で 3 着内がある"))
    a(H("P09", 0, None, None, "", "中山芝 2500 の重賞は年数レースで標本が無い"))
    a(H("P10", 1, lambda d: (d["course"] == "東京芝2400") & d["grade"].isin(G), lambda d: d["hn"] <= 4, "東京芝 2400 の重賞で馬番 1〜4"))
    # ---- X
    for hid, why in (("X01", "前日・当日の人気の時系列は 2026 年だけ"), ("X02", "締め切り前のオッズの時系列は 2026 年だけ"), ("X03", "複勝の人気が過去分に無い")):
        a(H(hid, 0, None, None, "", why))
    a(H("X04", 1, lambda d: d["pop"].notna(), lambda d: d["pop"] == 1, "1 番人気（市場の見込みより来るか。クラス・コース別は切り口で見る）"))
    a(H("X05", 1, lambda d: d["pop"] >= 6, lambda d: d["fav_odds"] >= 3.5, "6 番人気以下で、1 番人気が 3.5 倍以上の混戦"))
    a(H("X06", 1, lambda d: d["p_pop"].notna(), lambda d: (d["p_pop"] <= 3) & (d["p_fin"] >= 4) & (d["pop"] - d["p_pop"] >= 3), "前走上位人気で 4 着以下、今回人気が 3 以上下がった"))
    a(H("X07", -1, lambda d: d["odds"] < 2.0, lambda d: (d["field"] <= 10) | ((d["n_nige_cand"] - d["nige_cand"]) == 0) | (d["venue_n_prev"] == 0), "単勝 1 倍台で、10 頭以下・ほかに逃げ候補なし・初めての競馬場のどれか"))
    a(H("X08", -1, lambda d: d["p_pop"] == 1, lambda d: d["p_fin"] >= 4, "前走 1 番人気で 4 着以下"))
    # ---- Z: 偽の型（シャドウ）。結果と無関係な乱数の旗。誤検出の数を測る（判定の族に含める）
    def rnd(level: str, p: float, seed: int):
        def f(d):
            if level == "row":
                u = np.random.RandomState(seed).rand(len(d))
            else:
                codes = pd.factorize(d[level] if level != "day" else d["vc"].astype(str) + d["date"].astype(str))[0]
                u = np.random.RandomState(seed).rand(codes.max() + 1)[codes]
            return pd.Series(u < p, index=d.index)
        return f
    allr = lambda d: pd.Series(True, index=d.index)  # noqa: E731
    shadows = [("row", 0.05), ("row", 0.1), ("row", 0.3), ("row", 0.5), ("horse_id", 0.1), ("horse_id", 0.3), ("horse_id", 0.5),
               ("trainer_id", 0.2), ("jockey_id", 0.2), ("f_name", 0.2), ("day", 0.3), ("race_id", 0.3)]
    for i, (lev, p) in enumerate(shadows, 1):
        a(H(f"Z{i:02d}", 1, allr, rnd(lev, p, 1000 + i), f"偽の型: {lev} 単位の乱数で {int(p * 100)}% に旗"))
    return L


# ---------------------------------------------------------------- 検定
DIMS = [
    ("全体", lambda d: pd.Series("全体", index=d.index)),
    ("面", lambda d: d["surf"]),
    ("面×距離帯", lambda d: d["surf"] + d["dband"]),
    ("競馬場×面", lambda d: d["venue"] + d["surf"]),
    ("コース", lambda d: d["course"]),
    ("クラス", lambda d: d["grade"].replace({"L": "OP", "G1": "重賞", "G2": "重賞", "G3": "重賞"})),
    ("面×クラス", lambda d: d["surf"] + d["grade"].replace({"L": "OP", "G1": "重賞", "G2": "重賞", "G3": "重賞"})),
    ("面×馬場", lambda d: d["surf"] + d["track"].replace({"不": "重"}).fillna("?")),
    ("頭数", lambda d: pd.cut(d["field"], [0, 10, 14, 18], labels=["~10頭", "11-14頭", "15頭~"]).astype(str)),
    ("季節", lambda d: d["month"].map({3: "春", 4: "春", 5: "春", 6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋", 12: "冬", 1: "冬", 2: "冬"})),
    ("年齢", lambda d: np.where(d["age"] <= 2, "2歳", np.where(d["age"] == 3, "3歳", "4歳以上"))),
    ("性", lambda d: np.where(d["sex"] == "牝", "牝", "牡・セ")),
    ("人気帯", lambda d: pd.cut(d["pop"], [0, 3, 6, 9, 99], labels=["1-3人気", "4-6人気", "7-9人気", "10人気~"]).astype(str)),
]


def diff_test(key: np.ndarray, race: np.ndarray, f: np.ndarray, v: np.ndarray, nk: int):
    """切り口 key ごとに『当てはまる − 当てはまらない』の v の平均の差と、レース単位のクラスタ頑健 SE。"""
    nr = race.max() + 1
    comb = key.astype(np.int64) * nr + race
    u, inv = np.unique(comb, return_inverse=True)
    f1 = f.astype(float)
    n1 = np.bincount(inv, f1)
    n0 = np.bincount(inv, 1 - f1)
    s1 = np.bincount(inv, f1 * v)
    s0 = np.bincount(inv, (1 - f1) * v)
    gk = (u // nr).astype(np.int64)
    N1 = np.bincount(gk, n1, nk)
    N0 = np.bincount(gk, n0, nk)
    S1 = np.bincount(gk, s1, nk)
    S0 = np.bincount(gk, s0, nk)
    R1 = np.bincount(gk, (n1 > 0).astype(float), nk)
    G = np.bincount(gk, None, nk).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = S1 / N1
        m0 = S0 / N0
        uu = (s1 - m1[gk] * n1) / N1[gk] - (s0 - m0[gk] * n0) / N0[gk]
        var = np.bincount(gk, uu * uu, nk) * G / np.maximum(G - 1, 1)
    return N1, N0, R1, m1, m0, m1 - m0, np.sqrt(var)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def periods(years_with: list[int]) -> tuple[list[int], list[int]]:
    ys = sorted(years_with)
    if set(range(2016, 2026)) <= set(ys):
        return list(range(2016, 2023)), [2023, 2024, 2025]
    if len(ys) <= 1:
        return ys, []
    k = max(1, int(round(len(ys) * 0.7)))
    k = min(k, len(ys) - 1)
    return ys[:k], ys[k:]


def run() -> None:
    d = pd.read_parquet(OUT / "master.parquet") if not (OUT / "_prep.pkl").exists() else pd.DataFrame()
    for c in d.columns:
        if str(d[c].dtype) == "string":
            d[c] = d[c].astype(object).where(d[c].notna(), None)
    prep = OUT / "_prep.pkl"
    if prep.exists() and os.environ.get("FOLKLORE_RESUME", "1") == "1":
        d = pd.read_pickle(prep)
        d = add_market(d)  # 市場の見込みは作り方を変えることがあるので、毎回作り直す（安い）
        d = add_b37(d)
        d.to_pickle(prep)
    else:
        d = d[d["year"].between(2016, 2025)].reset_index(drop=True)
        d = add_extra(d)
        d = add_market(d)
        d = add_b37(d)
        d = d[d["p_mkt"].notna()].reset_index(drop=True)
        d.to_pickle(prep)
    print("prepared", d.shape, flush=True)
    race = pd.factorize(d["race_id"])[0]
    clusters = {c: pd.factorize(d[c].astype(str))[0] for c in ("horse_id", "trainer_id", "jockey_id", "f_name")}
    years = d["year"].to_numpy()
    keys = {}
    for name, fn in DIMS:
        k = pd.Series(np.asarray(fn(d)), index=d.index).astype(str)
        codes, labels = pd.factorize(k)
        keys[name] = (codes, np.asarray(labels))
    rows, meta = [], []
    for h in hyps():
        if h.elig is None:
            meta.append({"hid": h.hid, "dir": 0, "how": h.how, "na": h.na})
            continue
        try:
            e = np.asarray(pd.Series(h.elig(d), index=d.index).fillna(False).astype(bool))
            f = np.asarray(pd.Series(h.flag(d), index=d.index).fillna(False).astype(bool)) & e
        except Exception as ex:  # noqa: BLE001
            meta.append({"hid": h.hid, "dir": h.direction, "how": h.how, "na": f"計算できなかった: {ex}"})
            print("ERR", h.hid, ex)
            continue
        yc = pd.Series(years[e]).value_counts()
        ywith = [int(y) for y, n in yc.items() if n >= 500 and (pd.Series(years[f]) == y).sum() >= 50]
        disc, conf = periods(ywith)
        meta.append({"hid": h.hid, "dir": h.direction, "how": h.how, "na": "", "n_elig": int(e.sum()), "n_flag": int(f.sum()), "disc": disc, "conf": conf,
                     "years": sorted(ywith)})
        if not disc:
            continue
        for per, ys in (("disc", disc), ("conf", conf)):
            if not ys:
                continue
            m = e & np.isin(years, ys)
            if m.sum() == 0:
                continue
            for name, (codes, labels) in keys.items():
                kk = codes[m]
                nk = len(labels)
                out = {}
                for vn in ("r3", "y3", "ret_cap", "ret"):
                    out[vn] = diff_test(kk, race[m], f[m], d[vn].to_numpy()[m], nk)
                # 標準誤差: レース・馬・調教師・騎手・父のそれぞれでまとめた（クラスタ頑健）誤差の最大（保守的）
                se = {"r3": out["r3"][6].copy(), "y3": out["y3"][6].copy()}
                for cn, cc in clusters.items():
                    for vn in ("r3", "y3"):
                        se[vn] = np.fmax(se[vn], diff_test(kk, cc[m], f[m], d[vn].to_numpy()[m], nk)[6])
                N1, N0, R1 = out["r3"][0], out["r3"][1], out["r3"][2]
                present = np.flatnonzero((N1 > 0) & (N0 > 0))
                for i in present:
                    rows.append({"hid": h.hid, "period": per, "dim": name, "key": labels[i], "n1": N1[i], "n0": N0[i], "r1": R1[i],
                                 "adj": out["r3"][5][i], "adj_se": se["r3"][i], "adj_se_race": out["r3"][6][i], "raw": out["y3"][5][i], "raw_se": se["y3"][i],
                                 "raw_se_race": out["y3"][6][i],
                                 "y3_1": out["y3"][3][i], "y3_0": out["y3"][4][i], "p_mkt_1": out["y3"][3][i] - out["r3"][3][i],
                                 "roi_cap_1": out["ret_cap"][3][i], "roi_cap_0": out["ret_cap"][4][i], "roi_1": out["ret"][3][i]})
        print(h.hid, "elig", int(e.sum()), "flag", int(f.sum()), disc, conf, flush=True)
    t = pd.DataFrame(rows)
    t.to_parquet(OUT / "tests_raw.parquet", index=False)
    json.dump(meta, open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1, default=str)
    judge(t, meta)


def judge(t: pd.DataFrame, meta: list[dict]) -> None:
    disc = t[t["period"] == "disc"].copy()
    conf = t[t["period"] == "conf"].copy()
    ok = (disc["n1"] >= MIN_N1) & (disc["r1"] >= MIN_R1) & (disc["n0"] >= MIN_N0)
    disc = disc[ok].copy()
    res = disc[["hid", "dim", "key", "n1", "n0", "r1", "adj", "adj_se", "raw", "raw_se", "y3_1", "y3_0", "p_mkt_1", "roi_cap_1", "roi_cap_0", "roi_1"]].copy()
    cj = conf.set_index(["hid", "dim", "key"])
    for fam in ("adj", "raw"):
        z = res[fam] / res[fam + "_se"]
        res[fam + "_p"] = 2 * norm.sf(np.abs(z))
        res[fam + "_q"] = bh(res[fam + "_p"].fillna(1).to_numpy())
        c = cj.reindex(pd.MultiIndex.from_frame(res[["hid", "dim", "key"]]))
        res[fam + "_c"] = c[fam].to_numpy()
        res[fam + "_c_se"] = c[fam + "_se"].to_numpy()
        res["c_n1"] = c["n1"].to_numpy()
        res["c_roi_cap_1"] = c["roi_cap_1"].to_numpy()
        fwd = (res[fam + "_q"] < 0.10) & (res["c_n1"] >= MIN_CONF_N1) & res[fam + "_c"].notna()
        zc = np.sign(res[fam]) * res[fam + "_c"] / res[fam + "_c_se"]
        pc = norm.sf(zc)
        res[fam + "_cp"] = np.where(fwd, pc, np.nan)
        qc = np.full(len(res), np.nan)
        qc[fwd.to_numpy()] = bh(pc[fwd.to_numpy()])
        res[fam + "_cq"] = qc
        confd = 1 - np.maximum(res[fam + "_q"], np.nan_to_num(res[fam + "_cq"], nan=1.0))
        res[fam + "_conf"] = confd
        tier = np.full(len(res), "無", dtype=object)
        for name, th in TIERS[::-1]:
            tier[confd >= th] = name
        res[fam + "_tier"] = tier
    dirmap = {m["hid"]: m["dir"] for m in meta}
    res["claim"] = res["hid"].map(dirmap)
    res["adj_agree"] = np.sign(res["adj"]) == res["claim"]
    res["raw_agree"] = np.sign(res["raw"]) == res["claim"]
    res.to_parquet(OUT / "tests.parquet", index=False)
    print("tests", len(res), res["adj_tier"].value_counts().to_dict(), res["raw_tier"].value_counts().to_dict())


# ---------------------------------------------------------------- 通説の数で並べたときの成績
def _payouts(years: list[int]) -> pd.DataFrame:
    """単勝・複勝の実際の払戻（100 円あたり）。同着も払戻の表どおり。"""
    rows = []
    for y in years:
        r = pd.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_result_race.parquet", columns=["race_id", "payoff"])
        for rid, pay in zip(r["race_id"], r["payoff"]):
            try:
                j = json.loads(pay) if isinstance(pay, str) else (pay or {})
            except ValueError:
                continue
            tan = j.get("単勝") or []
            for k, items in (("win", tan if isinstance(tan, list) else [tan]), ("plc", j.get("複勝") or [])):
                for it in items if isinstance(items, list) else [items]:
                    for num in str(it.get("numbers", "")).split():
                        try:
                            rows.append((str(rid), int(num), k, float(str(it.get("payout", "0")).replace(",", ""))))
                        except ValueError:
                            pass
    pdf = pd.DataFrame(rows, columns=["race_id", "horse_number", "kind", "pay"])
    return pdf.pivot_table(index=["race_id", "horse_number"], columns="kind", values="pay", aggfunc="max").reset_index()


def rank() -> None:
    """すべての通説（検証できる型）を全出走に当てはめ、『買いの通説の数 − 消しの通説の数』でレース内に並べたときの成績。"""
    d = pd.read_pickle(OUT / "_prep.pkl")
    hs = [h for h in hyps() if h.elig is not None and not h.hid.startswith("Z")]
    base_dir: dict = {}
    buy = np.zeros(len(d), dtype=np.int16)
    kill = np.zeros(len(d), dtype=np.int16)
    ver = np.zeros(len(d), dtype=np.int16)
    # 検証済み（発見期間 2016〜2022 の判定だけで選ぶ。確認期間 2023〜2025 は選択に使わない）
    disc = pd.read_parquet(OUT / "tests_raw.parquet")
    disc = disc[(disc["period"] == "disc") & (disc["dim"] == "全体") & (disc["n1"] >= MIN_N1) & (disc["r1"] >= MIN_R1) & (disc["n0"] >= MIN_N0)].copy()
    disc = disc[~disc["hid"].str.startswith("Z")]
    disc["p"] = 2 * norm.sf(np.abs(disc["adj"] / disc["adj_se"]))
    disc["q"] = bh(disc["p"].fillna(1).to_numpy())
    vsign = {r.hid: int(np.sign(r.adj)) for r in disc.itertuples() if r.q < 0.05}
    flags: dict = {}
    for h in hs:
        try:
            e = np.asarray(pd.Series(h.elig(d), index=d.index).fillna(False).astype(bool))
            f = np.asarray(pd.Series(h.flag(d), index=d.index).fillna(False).astype(bool)) & e
        except Exception:  # noqa: BLE001
            continue
        b = h.hid[:3]
        flags.setdefault(b, np.zeros(len(d), dtype=bool))
        flags[b] |= f
        base_dir[b] = h.direction
        if h.hid in vsign:
            ver += (vsign[h.hid] * f).astype(np.int16)
    for b, f in flags.items():
        if base_dir[b] > 0:
            buy += f
        else:
            kill += f
    d = d[["race_id", "year", "horse_number", "fin", "pop", "odds"]].copy()
    d["buy"], d["kill"], d["net"], d["ver"] = buy, kill, buy.astype(int) - kill, ver
    d["race_id"] = d["race_id"].astype(str)
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce")
    pay = _payouts(sorted(d["year"].unique().tolist()))
    d = d.merge(pay, on=["race_id", "horse_number"], how="left")
    d["win"] = d["win"].fillna(0.0)
    d["plc"] = d["plc"].fillna(0.0)
    rng = np.random.default_rng(0)
    d["tb"] = rng.random(len(d))  # 同点は無作為に並べる（人気で並べると市場の情報が混ざる）
    out = {"n_hyp": len(flags), "n_buy": int(sum(1 for v in base_dir.values() if v > 0)), "n_kill": int(sum(1 for v in base_dir.values() if v < 0)),
           "n_ver": len(vsign), "ver_hids": sorted(vsign), "races": int(d["race_id"].nunique()), "runs": len(d)}

    def summ(g):
        n = len(g)
        return {"n": n, "win": float((g["fin"] == 1).mean()), "top2": float((g["fin"] <= 2).mean()), "top3": float((g["fin"] <= 3).mean()),
                "roi_win": float(g["win"].sum() / (100 * n)) if n else np.nan, "roi_plc": float(g["plc"].sum() / (100 * n)) if n else np.nan,
                "pop": float(g["pop"].mean()), "odds_med": float(g["odds"].median())}

    for score in ("net", "buy", "ver"):
        d = d.sort_values(["race_id", score, "tb"], ascending=[True, False, True])
        d["rk"] = d.groupby("race_id").cumcount() + 1
        for per, ys in (("all", range(2016, 2026)), ("conf", range(2023, 2026))):
            x = d[d["year"].isin(list(ys))]
            out[f"{score}_{per}_rank"] = {int(k): summ(g) for k, g in x[x["rk"] <= 18].groupby("rk")}
            lo, hi = {"net": (-2, 20), "buy": (0, 24), "ver": (-5, 10)}[score]
            sc = x[score].clip(lo, hi)
            out[f"{score}_{per}_score"] = {int(k): summ(g) for k, g in x.groupby(sc)}
        top = d[d["rk"] == 1]
        out[f"{score}_year_top1"] = {int(y): summ(g) for y, g in top.groupby("year")}
    pb = d[d["pop"] == 1]
    out["pop1_all"] = summ(pb)
    out["pop1_conf"] = summ(pb[pb["year"] >= 2023])
    out["pop1_year"] = {int(y): summ(g) for y, g in pb.groupby("year")}
    out["all_runs"] = summ(d)
    out["score_dist"] = {k: {"mean": float(d[k].mean()), "p50": float(d[k].median()), "p90": float(d[k].quantile(0.9)), "max": int(d[k].max())} for k in ("buy", "kill", "net", "ver")}
    json.dump(out, open(OUT / "rank.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("n_hyp", "n_buy", "n_kill", "n_ver", "races", "runs", "score_dist", "pop1_all")}, ensure_ascii=False, indent=1))
    for score in ("net", "ver"):
        for per in ("all", "conf"):
            print(score, per)
            for k in range(1, 7):
                r = out[f"{score}_{per}_rank"][k]
                print(f" {k} n={r['n']} win={r['win']:.3f} top3={r['top3']:.3f} roiW={r['roi_win']:.3f} roiP={r['roi_plc']:.3f} pop={r['pop']:.1f}")



# ---------------------------------------------------------------- 通説の格（S/A/B/C）で重みを付けた得点と、閾値ごとの 1 位の成績
TIER_W = [(0.001, 4), (0.01, 3), (0.05, 2), (0.10, 1)]  # 発見期間の q → 重み（S=4・A=3・B=2・C=1）


def _w_of_q(q: np.ndarray) -> np.ndarray:
    w = np.zeros(len(q), dtype=np.int8)
    for th, v in TIER_W[::-1]:
        w[q < th] = v
    return w


def _roc(score: np.ndarray, y: np.ndarray, npts: int = 120) -> tuple[list, float]:
    """同点を 1 つの閾値として扱う ROC と AUC。"""
    df = pd.DataFrame({"s": score, "y": y}).groupby("s")["y"].agg(["sum", "count"]).sort_index(ascending=False)
    tp = np.concatenate([[0], df["sum"].cumsum().to_numpy()])
    fp = np.concatenate([[0], (df["count"] - df["sum"]).cumsum().to_numpy()])
    tpr, fpr = tp / tp[-1], fp / fp[-1]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    idx = np.unique(np.searchsorted(fpr, np.linspace(0, 1, npts)).clip(0, len(fpr) - 1))
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx], auc


def score() -> None:
    """通説の格（発見期間 2016〜2022 の検定の S/A/B/C）を重みにした得点で各レースの 1 位を選び、
    1 位の得点の閾値を上げていったときの成績（2023〜2025 年、重みの決定に使っていない期間）を出す。"""
    sp = OUT / "scores.parquet"
    if sp.exists() and os.environ.get("FOLKLORE_RESUME", "1") == "1":
        d = pd.read_parquet(sp)
        out = json.load(open(OUT / "score_used.json"))
    else:
        d, out = _score_rows()
        d.to_parquet(sp, index=False)
        json.dump(out, open(OUT / "score_used.json", "w"))
    _score_eval(d, out)


def _score_rows() -> tuple[pd.DataFrame, dict]:
    d = pd.read_pickle(OUT / "_prep.pkl")
    tr = pd.read_parquet(OUT / "tests_raw.parquet")
    disc = tr[(tr["period"] == "disc") & (tr["n1"] >= MIN_N1) & (tr["r1"] >= MIN_R1) & (tr["n0"] >= MIN_N0) & ~tr["hid"].str.startswith("Z")].copy()
    disc["p"] = 2 * norm.sf(np.abs(disc["adj"] / disc["adj_se"]))
    disc["q"] = bh(disc["p"].fillna(1).to_numpy())  # 全切り口の発見期間の検定を 1 つの族として補正（judge と同じ）
    disc["w"] = _w_of_q(disc["q"].to_numpy()) * np.sign(disc["adj"]).astype(int)
    disc = disc[disc["w"] != 0]
    keys = {name: pd.Series(np.asarray(fn(d)), index=d.index).astype(str).to_numpy() for name, fn in DIMS}
    n = len(d)
    s_all = np.zeros(n, dtype=np.int16)  # 全体の格だけ
    s_slc = np.zeros(n, dtype=np.int16)  # その馬の切り口（コース・馬場など）の中で最も格の高い判定
    cnt = np.zeros(n, dtype=np.int16)  # 通説の向きで単純に数えた（第 8 節）
    used = {"all": set(), "slice": set()}
    for h in hyps():
        if h.elig is None or h.hid.startswith("Z"):
            continue
        try:
            e = np.asarray(pd.Series(h.elig(d), index=d.index).fillna(False).astype(bool))
            f = np.asarray(pd.Series(h.flag(d), index=d.index).fillna(False).astype(bool)) & e
        except Exception:  # noqa: BLE001
            continue
        cnt += (h.direction * f).astype(np.int16)
        x = disc[disc["hid"] == h.hid]
        if not len(x) or not f.any():
            continue
        best = np.zeros(n, dtype=np.int8)
        for dim, g in x.groupby("dim"):
            mp = dict(zip(g["key"].astype(str), g["w"].astype(int)))
            v = pd.Series(keys[dim][f]).map(mp).fillna(0).astype(np.int8).to_numpy()
            cur = best[f]
            best[f] = np.where(np.abs(v) > np.abs(cur), v, cur)
            if dim == "全体":
                s_all[f] += v
                used["all"].add(h.hid)
        s_slc += best
        if (best != 0).any():
            used["slice"].add(h.hid)
    d = d[["race_id", "year", "horse_number", "fin", "pop", "odds", "p_mkt", "y3"]].copy()
    d["s_cnt"], d["s_all"], d["s_slc"] = cnt, s_all, s_slc
    d["race_id"] = d["race_id"].astype(str)
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce")
    d = d.merge(_payouts(sorted(d["year"].unique().tolist())), on=["race_id", "horse_number"], how="left")
    d["win"], d["plc"] = d["win"].fillna(0.0), d["plc"].fillna(0.0)
    d["tb"] = np.random.default_rng(0).random(len(d))
    return d, {"n_all": len(used["all"]), "n_slice": len(used["slice"]), "tier_w": TIER_W}


def _score_eval(d: pd.DataFrame, out: dict) -> None:
    te = d[d["year"] >= 2023]
    y = (te["fin"] <= 3).to_numpy()
    yw = (te["fin"] == 1).to_numpy()
    out["roc"] = {}
    for k, col in (("cnt", "s_cnt"), ("all", "s_all"), ("slc", "s_slc"), ("mkt", "p_mkt")):
        pts, auc = _roc(te[col].to_numpy(), y)
        _, aucw = _roc(te[col].to_numpy(), yw, 2)
        out["roc"][k] = {"pts": pts, "auc": auc, "auc_win": aucw}
    covs = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]
    out["thr"] = {}
    for k, col in (("cnt", "s_cnt"), ("all", "s_all"), ("slc", "s_slc")):
        for per, ys in (("conf", (2023, 2025)), ("disc", (2016, 2022))):
            x = d[d["year"].between(*ys)].sort_values(["race_id", col, "tb"], ascending=[True, False, True])
            top = x.groupby("race_id", sort=False).head(1)
            nr = len(top)
            top = top.sort_values([col, "tb"], ascending=[False, True]).reset_index(drop=True)
            rows = []
            grid = sorted(set([c for c in covs] + [i / 50 for i in range(1, 51)]), reverse=True)
            for c in grid:
                m = max(1, int(round(c * nr)))
                th = top[col].iloc[m - 1]
                g = top[top[col] >= th]  # 閾値以上（同点は全部含む）
                rw, rp = g["win"] / 100, g["plc"] / 100
                rows.append({"cov_target": c, "thr": float(th), "n": len(g), "cov": len(g) / nr, "win": float((g["fin"] == 1).mean()),
                             "top2": float((g["fin"] <= 2).mean()), "top3": float((g["fin"] <= 3).mean()),
                             "roi_win": float(rw.mean()), "roi_win_se": float(rw.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan,
                             "roi_plc": float(rp.mean()), "roi_plc_se": float(rp.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan,
                             "pop": float(g["pop"].mean()), "mkt_win": float((1 / g["odds"]).mean() * 0.8)})
            out["thr"][f"{k}_{per}"] = rows
    # 比べるもの: 1 番人気（2023〜2025）
    pb = te[te["pop"] == 1]
    out["pop1"] = {"n": len(pb), "win": float((pb["fin"] == 1).mean()), "top2": float((pb["fin"] <= 2).mean()), "top3": float((pb["fin"] <= 3).mean()),
                   "roi_win": float(pb["win"].mean() / 100), "roi_plc": float(pb["plc"].mean() / 100)}
    json.dump(out, open(OUT / "score.json", "w"), ensure_ascii=False, indent=1)
    print("used", out["n_all"], out["n_slice"], {k: (round(v["auc"], 4), round(v["auc_win"], 4)) for k, v in out["roc"].items()})
    for k in ("cnt", "all", "slc"):
        print(k)
        for r in out["thr"][f"{k}_conf"]:
            if r["cov_target"] in covs:
                print(f"  cov={r['cov']:.3f} thr={r['thr']:.0f} n={r['n']} win={r['win']:.3f} top3={r['top3']:.3f} roiW={r['roi_win']:.3f}±{r['roi_win_se']:.3f} roiP={r['roi_plc']:.3f}±{r['roi_plc_se']:.3f} pop={r['pop']:.1f}")



# 玄人の読み（S・A を持つ型の解釈。数値は表から自動で入る）
INTERP = {
    "L03": "通説と逆。2 強のレースでは 2 頭とも市場の見込みより 3 着内に来る（全体 S、確認期間でも +6pt）。『2 強は堅い』と読み替える",
    "A15": "通説どおり。昇級初戦は人気より来ない（全体 S、−2pt）。特にダート",
    "G13": "偶数の馬番（後入れ）は人気以上に来る（小さいが全体 S）",
    "J04": "コース別の持ち時計（指数）1 位は人気以上に来る。市場が持ち時計を割り引きすぎている",
    "L10": "通説と逆。ダートの先行馬は人気ほど来ない（先行の有利は人気に織り込まれ過ぎ）",
    "A25": "通説と逆。同じコースで 3 着内の経験がある『コース巧者』は人気ほど来ない（巧者は過剰人気）",
    "E18e": "米国型血統のダート短距離は人気以上に来る（未勝利・稍重のダートで強い。確認期間は弱まる）",
    "D11": "社台グループ以外の生産馬のダートは人気以上に来る（4 歳以上で強い）",
    "A03": "前走と同じコースの続戦は、未勝利戦（特にダート）で人気以上に来る",
    "L01": "夏の牝馬は、芝の短距離（〜1300m）でだけ人気以上に来る",
    "A01": "通説と逆。距離延長は未勝利戦で人気より来ない",
    "A42": "通説と逆。連勝中の昇級初戦は人気より来ない（勢いは人気で過大評価。多頭数で強い）",
}

# ---------------------------------------------------------------- 章（HTML）
CAT_NAMES = {"A": "ローテーション・条件替わり", "B": "調教（追い切り）", "C": "騎手", "D": "厩舎・馬主・生産者", "E": "血統", "F": "馬場・天候・バイアス",
             "G": "枠・展開・隊列", "H": "馬体・体調・個体", "I": "番組・レースの性格", "J": "能力指標", "K": "組み合わせ", "L": "格言", "M": "初物",
             "N": "消し（危険な人気馬）", "P": "競馬場・コースの格言", "X": "市場（オッズ）"}


def claims() -> dict:
    import html as _h
    import re as _re
    s = (ROOT / "docs/html/modeling/chapters/folklore_hypotheses.html").read_text(encoding="utf-8")
    out = {}
    for m in _re.finditer(r"<tr><td class='id'>([A-Z]\d\d)</td><td>(.*?)</td>", s):
        out[m.group(1)] = _h.unescape(_re.sub(r"<[^>]+>", "", m.group(2)))
    return out


DIM_ORDER = ["全体", "コース", "競馬場×面", "面×距離帯", "面", "面×馬場", "クラス", "面×クラス", "頭数", "季節", "年齢", "性", "人気帯"]


def _correct_section(w, tested, meta, cl, esc, tier_html, pt, pc, order) -> None:
    """型ごとに、通説の向きで有意（S〜C）になった切り口を折り畳みで並べる。"""
    ok_adj = (tested["adj_tier"] != "無") & tested["adj_agree"]
    ok_raw = (tested["raw_tier"] != "無") & tested["raw_agree"]
    ok = tested[ok_adj | ok_raw].copy()
    ok["ok_adj"] = ok_adj[ok.index]
    ok["oa"] = ok["adj_tier"].where(ok["ok_adj"], "無").map(order)
    ok["or"] = ok["raw_tier"].where(ok_raw[ok.index], "無").map(order)
    n_h_adj = ok[ok["ok_adj"]]["hid"].nunique()
    n_h_raw = ok["hid"].nunique()
    w(f"<h2 id='correct'>3. 型ごと: 通説が正しいと言える切り口（折り畳み）</h2>"
      "<p class='muted'>型ごとに、<b>通説の向きで有意（S〜C）</b>になった切り口（コース・競馬場×面・距離帯・馬場・クラスなど）を並べた。"
      "型を開くと切り口の種類ごとの折り畳みがあり、開くと値ごとの判定が出る。"
      "<b>人気をそろえた差</b>で有意 = 人気以上に来る（または来ない）＝序列づけに使える。"
      "<b>素の差だけ</b>で有意 = 通説は事実だが人気が織り込み済み（行を薄く表示）。"
      f"人気をそろえた差で正しい切り口を持つ型 {n_h_adj}、素の差まで含めると {n_h_raw}。</p>")
    w("<div class='ok-tools'><label><input type='checkbox' id=\"ok-raw\" checked/> 素の差だけで有意な切り口も表示</label>"
      "<label>絞り込み <input type='search' id=\"ok-q\" placeholder='例: 東京芝1600・ダ・重・G1・D04'/></label>"
      "<button type='button' id=\"ok-open\">すべて開く</button><button type='button' id=\"ok-close\">すべて閉じる</button>"
      "<span class='muted' id=\"ok-cnt\"></span></div>")
    cats: dict = {}
    for hid in meta:
        if hid in set(ok["hid"]):
            cats.setdefault(hid[0], []).append(hid)
    for c in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "X"]:
        if c not in cats:
            continue
        w(f"<h3>{c}. {CAT_NAMES[c]}（{len(cats[c])} 型）</h3>")
        for hid in cats[c]:
            m = meta[hid]
            x = ok[ok["hid"] == hid]
            xa = x[x["ok_adj"]]
            ca = xa["adj_tier"].value_counts()
            cr = x[(x["raw_tier"] != "無") & x["raw_agree"]]["raw_tier"].value_counts()
            chips = "".join(f"<span class='t{k}'>{k}</span>{int(ca[k])} " for k in "SABC" if ca.get(k, 0))
            rchips = "・".join(f"{k}{int(cr[k])}" for k in "SABC" if cr.get(k, 0))
            courses = xa[xa["dim"] == "コース"].sort_values(["oa", "adj_conf"], ascending=[True, False])
            dirs = "買い" if m["dir"] > 0 else "消し"
            claim = esc(cl.get(hid[:3], ""))
            w(f"<details class='okh' data-hid='{hid}'><summary><span class='sid'>{hid}</span>{claim}"
              f"<span class='chips'>{('人気そろえ ' + chips) if chips else '<span class=muted>人気そろえ なし</span> '}</span>"
              f"<span class='muted'>／素の差 {rchips or 'なし'}（{dirs}）</span></summary>")
            if len(courses):
                cs = "、".join(f"<b>{esc(str(r['key']))}</b>{tier_html(r['adj_tier'], True)}" for _, r in courses.iterrows())
                w(f"<div class='ok-courses'>人気をそろえても正しいコース: {cs}</div>")
            for dim in DIM_ORDER:
                y = x[x["dim"] == dim].sort_values(["oa", "adj_conf", "or", "raw_conf"], ascending=[True, False, True, False])
                if not len(y):
                    continue
                na = int(y["ok_adj"].sum())
                vals = "、".join(esc(str(k)) for k in y[y["ok_adj"]]["key"].head(6))
                more = "…" if na > 6 else ""
                w(f"<details class='okc'><summary>{esc(dim)}: 人気そろえ {na} ・素の差のみ {len(y) - na}"
                  f"{('<span class=muted>（' + vals + more + '）</span>') if na else ''}</summary><div class='wrap'><table><thead><tr>"
                  "<th>値</th><th>人気そろえ</th><th>Δ 発見/確認(pt)</th><th>素の差</th><th>Δ 発見/確認(pt)</th><th>3 着内率 / 市場 / 他</th><th>当てはまる走数（発見/確認）</th><th>単勝回収率 頭打ち</th></tr></thead><tbody>")
                for _, r in y.iterrows():
                    cls = "fam-adj" if r["ok_adj"] else "fam-raw"
                    at = tier_html(r["adj_tier"], r["adj_agree"]) + (f" <span class='muted'>{r['adj_conf'] * 100:.1f}%</span>" if r["adj_tier"] != "無" else "")
                    rt = tier_html(r["raw_tier"], r["raw_agree"]) + (f" <span class='muted'>{r['raw_conf'] * 100:.1f}%</span>" if r["raw_tier"] != "無" else "")
                    cn = int(r["c_n1"]) if pd.notna(r["c_n1"]) else 0
                    w(f"<tr class='{cls}'><td>{esc(str(r['key']))}</td><td>{at}</td><td class='num'>{pt(r['adj'])} / {pt(r['adj_c'])}</td><td>{rt}</td><td class='num'>{pt(r['raw'])} / {pt(r['raw_c'])}</td>"
                      f"<td class='num'>{pc(r['y3_1'])} / {pc(r['p_mkt_1'])} / {pc(r['y3_0'])}%</td><td class='num'>{int(r['n1']):,} / {cn:,}</td><td class='num'>{r['roi_cap_1'] * 100:.0f}%</td></tr>")
                w("</tbody></table></div></details>")
            w("</details>")
    w("""<script>(function(){
var raw=document.getElementById('ok-raw'),q=document.getElementById('ok-q'),cnt=document.getElementById('ok-cnt');
var hs=[].slice.call(document.querySelectorAll('details.okh'));
function apply(){var s=(q.value||'').trim().toLowerCase(),showRaw=raw.checked,nh=0;
hs.forEach(function(h){var idm=s&&h.dataset.hid.toLowerCase().indexOf(s)===0,anyH=false;
[].slice.call(h.querySelectorAll('details.okc')).forEach(function(c){var any=false;
[].slice.call(c.querySelectorAll('tbody tr')).forEach(function(tr){var vis=(showRaw||tr.classList.contains('fam-adj'))&&(!s||idm||tr.cells[0].textContent.toLowerCase().indexOf(s)>=0);tr.classList.toggle('hide',!vis);any=any||vis;});
c.classList.toggle('hide',!any);if(s&&!idm)c.open=any;anyH=anyH||any;});
var cm=h.querySelector('.ok-courses');if(cm)cm.classList.toggle('hide',!!s&&!idm);
h.classList.toggle('hide',!anyH);if(s)h.open=anyH;if(anyH)nh++;});
cnt.textContent=nh+' 型を表示';}
raw.addEventListener('change',apply);q.addEventListener('input',apply);
document.getElementById('ok-open').onclick=function(){hs.forEach(function(h){if(!h.classList.contains('hide'))h.open=true;});};
document.getElementById('ok-close').onclick=function(){document.querySelectorAll('details.okh,details.okc').forEach(function(d){d.open=false;});};
apply();})();</script>""")



def _rank_section(w) -> None:
    """第 8 節: すべての通説を全レースに当てはめ、当てはまる数で並べたときの成績（rank.json から）。"""
    f = OUT / "rank.json"
    if not f.exists():
        return
    o = json.load(open(f))

    def row(label, r, bold=False):
        b0, b1 = ("<b>", "</b>") if bold else ("", "")
        return (f"<tr><td>{b0}{label}{b1}</td><td class='num'>{r['n']:,}</td><td class='num'>{r['win'] * 100:.1f}%</td><td class='num'>{r['top2'] * 100:.1f}%</td>"
                f"<td class='num'>{r['top3'] * 100:.1f}%</td><td class='num'>{b0}{r['roi_win'] * 100:.1f}%{b1}</td><td class='num'>{b0}{r['roi_plc'] * 100:.1f}%{b1}</td>"
                f"<td class='num'>{r['pop']:.1f}</td><td class='num'>{r['odds_med']:.1f}</td></tr>")

    head = ("<div class='wrap'><table><thead><tr><th>{}</th><th>頭数</th><th>勝率</th><th>連対率</th><th>複勝率</th><th>単勝回収率</th><th>複勝回収率</th>"
            "<th>平均人気</th><th>単勝オッズの中央値</th></tr></thead><tbody>")

    def rank_table(key, title):
        w(head.format(title))
        for k in range(1, 19):
            r = o[key].get(str(k))
            if r:
                w(row(f"{k} 位", r, bold=k <= 3))
        w("</tbody></table></div>")

    sd = o["score_dist"]
    w("<h2 id='rank'>8. すべての通説を全レースに当てはめ、当てはまる数で並べたときの成績</h2>")
    w(f"<p>検証できる通説 {o['n_hyp']} 型（買い {o['n_buy']}・消し {o['n_kill']}。E18a〜e のような下位の型は元の型 1 つとして数える）を、2016〜2025 年の全 {o['races']:,} レース・{o['runs']:,} 走に当てはめた。"
      f"各馬の点数 = <b>当てはまる『買い』の通説の数 − 当てはまる『消し』の通説の数</b>（1 頭あたり買い 平均 {sd['buy']['mean']:.1f}・消し 平均 {sd['kill']['mean']:.1f}、差の中央値 {sd['net']['p50']:.0f}・最大 {sd['net']['max']}）。"
      "レースの中で点数の高い順に 1 位・2 位…と並べ、順位ごとの成績を出した。同点は無作為に並べた（人気で並べると市場の情報が混ざるため）。"
      "回収率は払戻の表の実際の単勝・複勝の払戻（頭打ちなし、同着も表どおり）。</p>")
    w("<div class='note'><b>結論.</b> "
      f"通説の数の 1 位は、勝率 {o['net_all_rank']['1']['win'] * 100:.1f}%・複勝率 {o['net_all_rank']['1']['top3'] * 100:.1f}% で、全出走（勝率 {o['all_runs']['win'] * 100:.1f}%・複勝率 {o['all_runs']['top3'] * 100:.1f}%）より明らかに走り、順位が下がるほど成績も下がる（通説はまとまると能力の順位づけとして効く）。"
      f"ただし回収率は 1 位でも単勝 {o['net_all_rank']['1']['roi_win'] * 100:.1f}%・複勝 {o['net_all_rank']['1']['roi_plc'] * 100:.1f}% で、1 番人気（単勝 {o['pop1_all']['roi_win'] * 100:.1f}%・複勝 {o['pop1_all']['roi_plc'] * 100:.1f}%）と同程度。控除率（単勝・複勝 20%）を超えない。"
      "通説の多くは人気が既に織り込んでおり、数を足しても『人気どおりの馬』を選ぶことに近づくだけ。"
      f"検定で残った通説だけを人気をそろえた差の向きで数えて並べると（下の 8.2）、1 位は平均 {o['ver_all_rank']['1']['pop']:.1f} 番人気と人気薄に寄り、2016〜2025 年の回収率は単勝 {o['ver_all_rank']['1']['roi_win'] * 100:.1f}%・複勝 {o['ver_all_rank']['1']['roi_plc'] * 100:.1f}%。"
      f"ただし型の選択に使っていない 2023〜2025 年では単勝 {o['ver_conf_rank']['1']['roi_win'] * 100:.1f}%・複勝 {o['ver_conf_rank']['1']['roi_plc'] * 100:.1f}% に落ちる（2016〜2022 年の数字は選択の当たりを含む）。</div>")
    w("<h3>8.1 すべての通説（買い − 消し）の順位ごとの成績（2016〜2025 年）</h3>")
    rank_table("net_all_rank", "通説の数の順位")
    w(head.format("比べるもの") + row("1 番人気", o["pop1_all"]) + row("全出走", o["all_runs"]) + "</tbody></table></div>")
    w("<details><summary>2023〜2025 年だけ（検証の確認期間）</summary>")
    rank_table("net_conf_rank", "通説の数の順位")
    w(head.format("比べるもの") + row("1 番人気（2023〜2025）", o["pop1_conf"]) + "</tbody></table></div></details>")
    w("<details><summary>点数（買い − 消し）ごとの成績</summary>")
    w(head.format("点数"))
    for k, r in o["net_all_score"].items():
        w(row(("≤" if int(k) == -2 else "≥" if int(k) == 20 else "") + str(k), r))
    w("</tbody></table></div></details>")
    w("<details><summary>『買い』の通説の数だけで並べた場合</summary>")
    rank_table("buy_all_rank", "買いの通説の数の順位")
    w("</details>")
    w("<details><summary>年ごとの 1 位の成績（データのある年で使える通説が変わる: 追い切りの詳細は 2016〜2019、指数・SmartRC は 2020〜）</summary>")
    w(head.format("年"))
    for y, r in o["net_year_top1"].items():
        w(row(f"{y}（1 番人気 単 {o['pop1_year'][y]['roi_win'] * 100:.0f}%・複 {o['pop1_year'][y]['roi_plc'] * 100:.0f}%）", r))
    w("</tbody></table></div></details>")
    w(f"<h3>8.2 検定で残った通説だけで並べた場合（{o['n_ver']} 型）</h3>"
      "<p class='muted'>発見期間（2016〜2022）の『全体・人気をそろえた差』で BH の q &lt; 0.05 になった型だけを使い、<b>向きはデータの向き</b>（通説と逆だった型は逆の向き）で ±1 を足した。"
      "型の選び方に 2023〜2025 年を使っていないので、2023〜2025 年の成績は選び方に対して手付かずの検証になる。"
      f"使った型: {', '.join(o['ver_hids'])}</p>")
    rank_table("ver_all_rank", "検定済みの通説の順位")
    w("<details><summary>2023〜2025 年だけ（型の選択に使っていない期間）</summary>")
    rank_table("ver_conf_rank", "検定済みの通説の順位")
    w("</details><details><summary>点数ごとの成績</summary>")
    w(head.format("点数"))
    for k, r in o["ver_all_score"].items():
        w(row(("≤" if int(k) == -5 else "≥" if int(k) == 10 else "") + str(k), r))
    w("</tbody></table></div></details>")
    w("<details><summary>年ごとの 1 位の成績</summary>" + head.format("年"))
    for y, r in o["ver_year_top1"].items():
        w(row(y, r))
    w("</tbody></table></div></details>")
    w("<div class='note'><b>注意.</b> (1) 人気・オッズの型（X 分類・L04 など）は確定オッズを使うので、前日に並べるなら前日オッズでの当てはめになる。"
      "(2) 型ごとに使えるデータの年が違うため、年によって当てはまる通説の数の基準が変わる（年をまたいだ点数の比較はしない。順位はレースの中だけ）。"
      "(3) 回収率は 1 頭 100 円の単純な単勝・複勝。1 位だけでも 3.4 万レースあるので、回収率の標準誤差はおよそ単勝 ±2〜3pt・複勝 ±1pt。</div>")


SER_COL = {"cnt": "#3987e5", "all": "#d95926", "slc": "#199e70", "mkt": "#c98500"}
SER_NAME = {"cnt": "通説の数（向きどおり ±1）", "all": "格の重み（全体の判定）", "slc": "格の重み（その馬の切り口の判定）", "mkt": "市場（単勝オッズ）"}


def _svg_lines(series, xr, yr, xlab, ylab, xfmt, yfmt, refs=(), diag=False, w=560, h=330) -> str:
    """折れ線の SVG（1 軸）。series = [(key, [(x, y, tip), ...])]。点に重ねた透明な円で値のツールチップ（title）を出す。"""
    L, R, T, B = 58, 16, 14, 46
    pw, ph = w - L - R, h - T - B

    def X(v):
        return L + (v - xr[0]) / (xr[1] - xr[0]) * pw

    def Y(v):
        return T + ph - (min(max(v, yr[0]), yr[1]) - yr[0]) / (yr[1] - yr[0]) * ph

    o = [f"<svg viewBox='0 0 {w} {h}' role='img' style='width:100%;max-width:{w}px;height:auto;font-size:11px'>"]
    for i in range(6):
        yv = yr[0] + (yr[1] - yr[0]) * i / 5
        o.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(yv):.1f}' y2='{Y(yv):.1f}' stroke='#243358' stroke-width='1'/>"
                 f"<text x='{L - 6}' y='{Y(yv) + 4:.1f}' text-anchor='end' fill='#9aa7c7'>{yfmt(yv)}</text>")
        xv = xr[0] + (xr[1] - xr[0]) * i / 5
        o.append(f"<text x='{X(xv):.1f}' y='{T + ph + 16}' text-anchor='middle' fill='#9aa7c7'>{xfmt(xv)}</text>")
    o.append(f"<text x='{L + pw / 2}' y='{h - 6}' text-anchor='middle' fill='#e6ecff'>{xlab}</text>"
             f"<text x='12' y='{T + ph / 2}' text-anchor='middle' fill='#e6ecff' transform='rotate(-90 12 {T + ph / 2})'>{ylab}</text>")
    if diag:
        o.append(f"<line x1='{X(0)}' y1='{Y(0)}' x2='{X(1)}' y2='{Y(1)}' stroke='#9aa7c7' stroke-dasharray='4 4' stroke-width='1'/>")
    for yv, lab in refs:
        o.append(f"<line x1='{L}' x2='{L + pw}' y1='{Y(yv):.1f}' y2='{Y(yv):.1f}' stroke='#9aa7c7' stroke-dasharray='4 4' stroke-width='1'/>"
                 f"<text x='{L + pw - 4}' y='{Y(yv) - 4:.1f}' text-anchor='end' fill='#9aa7c7'>{lab}</text>")
    for key, pts in series:
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y, _) in enumerate(pts))
        o.append(f"<path d='{path}' fill='none' stroke='{SER_COL[key]}' stroke-width='2' stroke-linejoin='round'/>")
        for x, y, tip in pts:
            if tip:
                o.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='3' fill='{SER_COL[key]}' stroke='#15213c' stroke-width='1'/>"
                         f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='9' fill='transparent'><title>{SER_NAME[key]}: {tip}</title></circle>")
    o.append("</svg>")
    leg = "".join(f"<span style='margin-right:14px;white-space:nowrap'><span style='display:inline-block;width:14px;height:3px;background:{SER_COL[k]};vertical-align:middle;margin-right:5px;border-radius:2px'></span>{SER_NAME[k]}</span>" for k, _ in series)
    return f"<div class='muted' style='margin:4px 0'>{leg}</div>" + "".join(o)


def _score_section(w) -> None:
    """第 9 節: 通説の格で重みを付けた得点と、1 位の得点の閾値ごとの成績・ROC。"""
    f = OUT / "score.json"
    if not f.exists():
        return
    o = json.load(open(f))
    p1 = o["pop1"]
    keys = ("cnt", "all", "slc")

    def pick(k, per="conf"):
        return [r for r in o["thr"][f"{k}_{per}"]]

    a100 = {k: pick(k)[0] for k in keys}
    a10 = {k: min(pick(k), key=lambda r: abs(r["cov"] - 0.10)) for k in keys}
    w("<h2 id='score'>9. 通説の格で得点を付け、閾値を超えたレースの 1 位だけを買ったときの成績</h2>")
    w("<p><b>得点</b> = 当てはまる通説ごとの <b>格の重み（S=4・A=3・B=2・C=1）× 向き</b>の合計。格と向きは<b>発見期間（2016〜2022）の検定だけ</b>で決めた"
      "（人気をそろえた差、全切り口を 1 つの族として BH。向きはデータの向きなので、通説と逆だった型は逆に数える）。成績は<b>2023〜2025 年</b>（重みの決定に使っていない期間）で出した。"
      "各レースで得点が最も高い馬を 1 位とし（同点は無作為）、<b>1 位の得点が閾値以上のレースだけ</b>買う。閾値を上げると買うレースが減る（横軸 = 買うレースの割合）。3 つの得点を比べた:</p><ul>"
      "<li><b>通説の数</b>: 第 8 節と同じ、通説の向きどおりに ±1（格は見ない）。</li>"
      f"<li><b>格の重み（全体の判定）</b>: 型の『全体』の格だけを使う（格の付いた型 {o['n_all']}）。</li>"
      f"<li><b>格の重み（その馬の切り口の判定）</b>: 型ごとに、その馬が入る切り口（コース・競馬場×面・距離帯・馬場・クラス・頭数・季節・年齢・性・人気帯・全体）のうち最も格の高い判定の重みを使う（格の付いた型 {o['n_slice']}）。</li></ul>")
    def bz(k, kind):
        rows = [r for r in pick(k) if r["n"] >= 300]
        return max(rows, key=lambda r: r[kind]), max((r[kind] - 1) / r[kind + "_se"] for r in rows)

    bw_, zw = bz("all", "roi_win")
    bp_, zp = bz("all", "roi_plc")
    w("<div class='note'><b>結論.</b> "
      f"格の重みは『人気をそろえた差』の向きで付けるので、得点は能力の順位ではなく『人気以上に来るか』の順位になる（3 着内の AUC: 格の重み（全体）{o['roc']['all']['auc']:.3f}・切り口 {o['roc']['slc']['auc']:.3f}・通説の数 {o['roc']['cnt']['auc']:.3f}、市場は {o['roc']['mkt']['auc']:.3f}）。"
      f"全レースで 1 位を買うと平均 {a100['all']['pop']:.1f} 番人気・勝率 {a100['all']['win'] * 100:.1f}%・回収率 単勝 {a100['all']['roi_win'] * 100:.0f}%・複勝 {a100['all']['roi_plc'] * 100:.0f}%、"
      f"上位 10% のレースだけなら 平均 {a10['all']['pop']:.1f} 番人気・勝率 {a10['all']['win'] * 100:.1f}%・単勝 {a10['all']['roi_win'] * 100:.0f}%・複勝 {a10['all']['roi_plc'] * 100:.0f}%。"
      f"閾値を上げると回収率は上がる傾向があり、最も高いのは単勝 {bw_['roi_win'] * 100:.0f}%±{bw_['roi_win_se'] * 100:.0f}（{bw_['n']:,} レース）・複勝 {bp_['roi_plc'] * 100:.0f}%±{bp_['roi_plc_se'] * 100:.0f}（{bp_['n']:,} レース）だが、"
      f"どの閾値でも 100% との差は標準誤差の {max(zw, zp):.1f} 倍以内で、100% を超えるとは言えない（同じ期間の 1 番人気は単勝 {p1['roi_win'] * 100:.0f}%・複勝 {p1['roi_plc'] * 100:.0f}%）。</div>")
    # 図
    def thr_series(metric, k, scale=1.0):
        return (k, [(r["cov"], r[metric] * scale, f"買うレース {r['cov'] * 100:.0f}%（閾値 {r['thr']:.0f}・{r['n']:,} レース）→ {r[metric] * 100:.1f}%"
                     + (f"±{r[metric + '_se'] * 100:.1f}" if metric + "_se" in r else "")) for r in sorted(pick(k), key=lambda r: r["cov"])])

    pc_ = lambda v: f"{v * 100:.0f}%"  # noqa: E731
    w("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:18px'>")
    roc = [(k, [(x, y, f"偽陽性率 {x * 100:.0f}% → 真陽性率 {y * 100:.0f}%" if i % 6 == 0 else "") for i, (x, y) in enumerate(o["roc"][k]["pts"])]) for k in ("mkt", "all", "slc", "cnt")]
    w("<figure style='margin:0'><figcaption><b>ROC（3 着内、2023〜2025 の全出走）</b> AUC: " + "・".join(f"{SER_NAME[k]} {o['roc'][k]['auc']:.3f}" for k in ("mkt", "all", "slc", "cnt")) + "</figcaption>")
    w(_svg_lines(roc, (0, 1), (0, 1), "偽陽性率（3 着外を拾った割合）", "真陽性率", pc_, pc_, diag=True) + "</figure>")
    w("<figure style='margin:0'><figcaption><b>1 位の勝率</b>（閾値を上げて買うレースを減らす）</figcaption>")
    w(_svg_lines([thr_series("win", k) for k in keys], (0, 1), (0, 0.5), "買うレースの割合（右ほど閾値が低い）", "勝率", pc_, pc_, refs=[(p1["win"], f"1 番人気 {p1['win'] * 100:.0f}%")]) + "</figure>")
    w("<figure style='margin:0'><figcaption><b>1 位の単勝回収率</b></figcaption>")
    w(_svg_lines([thr_series("roi_win", k) for k in keys], (0, 1), (0.5, 1.5), "買うレースの割合", "単勝回収率", pc_, pc_, refs=[(1.0, "100%"), (p1["roi_win"], f"1 番人気 {p1['roi_win'] * 100:.0f}%")]) + "</figure>")
    w("<figure style='margin:0'><figcaption><b>1 位の複勝回収率</b></figcaption>")
    w(_svg_lines([thr_series("roi_plc", k) for k in keys], (0, 1), (0.6, 1.1), "買うレースの割合", "複勝回収率", pc_, pc_, refs=[(1.0, "100%"), (p1["roi_plc"], f"1 番人気 {p1['roi_plc'] * 100:.0f}%")]) + "</figure>")
    w("</div><p class='muted'>点にカーソルを置くと値が出る。買うレースが少ない左端ほど標本が小さく、回収率の揺れが大きい（表の ± は標準誤差）。</p>")
    covs = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]
    head = ("<div class='wrap'><table><thead><tr><th>買うレース（目安）</th><th>閾値（1 位の得点 ≥）</th><th>レース数</th><th>実際の割合</th><th>勝率</th><th>連対率</th><th>複勝率</th>"
            "<th>単勝回収率 ±SE</th><th>複勝回収率 ±SE</th><th>平均人気</th></tr></thead><tbody>")

    def table(k, per):
        w(head)
        for r in o["thr"][f"{k}_{per}"]:
            if r["cov_target"] not in covs:
                continue
            w(f"<tr><td>上位 {r['cov_target'] * 100:.0f}%</td><td class='num'>{r['thr']:.0f}</td><td class='num'>{r['n']:,}</td><td class='num'>{r['cov'] * 100:.1f}%</td>"
              f"<td class='num'>{r['win'] * 100:.1f}%</td><td class='num'>{r['top2'] * 100:.1f}%</td><td class='num'>{r['top3'] * 100:.1f}%</td>"
              f"<td class='num'>{r['roi_win'] * 100:.1f}% ±{r['roi_win_se'] * 100:.1f}</td><td class='num'>{r['roi_plc'] * 100:.1f}% ±{r['roi_plc_se'] * 100:.1f}</td><td class='num'>{r['pop']:.1f}</td></tr>")
        w("</tbody></table></div>")

    w(f"<p>比べるもの: 2023〜2025 年の 1 番人気 {p1['n']:,} 頭 = 勝率 {p1['win'] * 100:.1f}%・連対率 {p1['top2'] * 100:.1f}%・複勝率 {p1['top3'] * 100:.1f}%・単勝回収率 {p1['roi_win'] * 100:.1f}%・複勝回収率 {p1['roi_plc'] * 100:.1f}%。</p>")
    for k in ("all", "slc", "cnt"):
        w(f"<h3>9.{('all', 'slc', 'cnt').index(k) + 1} {SER_NAME[k]}: 閾値ごとの 1 位の成績（2023〜2025 年）</h3>")
        table(k, "conf")
        w("<details><summary>2016〜2022 年（重みを決めた期間。参考）</summary>")
        table(k, "disc")
        w("</details>")
    w("<div class='note'><b>注意.</b> (1) 人気・オッズの型（1 番人気・混戦の 1 番人気・2 強など）も格が付けば得点に入る。これらは確定オッズで当てはめているので、前日に使うなら前日オッズで測り直しが要る。"
      "(2) 型ごとに使えるデータの年が違い（追い切りの詳細 2016〜2019、指数・SmartRC 2020〜）、2023〜2025 年は追い切りの型が入らない。(3) 閾値は得点の順位から決めた（買うレースの割合で比べるため）。</div>")


def report() -> None:
    import html as _h
    t = pd.read_parquet(OUT / "tests.parquet")
    meta = {m["hid"]: m for m in json.load(open(OUT / "meta.json"))}
    cl = claims()
    esc = _h.escape
    order = {"S": 0, "A": 1, "B": 2, "C": 3, "無": 4}

    def base(h):
        return h[:3]

    def pt(x):
        return "" if pd.isna(x) else f"{x * 100:+.1f}"

    def pc(x):
        return "" if pd.isna(x) else f"{x * 100:.1f}"

    def tier_html(tr, agree):
        if tr == "無" or pd.isna(tr):
            return "<span class='t0'>無</span>"
        rev = "" if agree else "<span class='rev' title='通説と逆の向きで有意'>逆</span>"
        return f"<span class='t{tr}'>{tr}</span>{rev}"

    shadow = t[t["hid"].str.startswith("Z")].copy()
    tested = t[~t["hid"].str.startswith("Z")].copy()
    meta = {k: v for k, v in meta.items() if not k.startswith("Z")}
    n_tests = len(tested)
    cnt_adj = tested["adj_tier"].value_counts().to_dict()
    cnt_raw = tested["raw_tier"].value_counts().to_dict()
    # 型ごとのまとめ
    rows = []
    for hid, m in meta.items():
        x = tested[tested["hid"] == hid]
        allr = x[x["dim"] == "全体"]
        a0 = allr.iloc[0] if len(allr) else None
        best = x.sort_values("adj_conf", ascending=False).head(3) if len(x) else x
        rows.append((hid, m, x, a0, best))
    css = """:root{--bg:#0f172a;--panel:#15213c;--border:#243358;--text:#e6ecff;--muted:#9aa7c7;--accent:#4f8bff;--accent-2:#7aa9ff;--warn:#fbbf24}
*{box-sizing:border-box}body{font-family:"Noto Sans JP",system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.8;font-size:15px;padding:40px 56px 96px;max-width:1280px;margin:0 auto}
@media (max-width:700px){body{padding:24px 16px 64px}}a{color:var(--accent-2)}h1{font-size:26px}h2{font-size:20px;margin:40px 0 12px;border-left:4px solid var(--accent);padding-left:12px}h3{font-size:16px;margin:22px 0 8px}
table{border-collapse:collapse;margin:12px 0 18px;font-size:12.5px;background:var(--panel);width:100%}th,td{border:1px solid var(--border);padding:4px 8px;text-align:left;vertical-align:top}th{background:#1b2a4d}
td.num{text-align:right;font-family:monospace;white-space:nowrap}td.id{font-family:monospace;white-space:nowrap}.muted{color:var(--muted);font-size:13px}.note{background:var(--panel);border:1px solid var(--border);border-left:4px solid var(--warn);border-radius:8px;padding:12px 16px;margin:14px 0}
.wrap{overflow-x:auto}.tS,.tA,.tB,.tC,.t0{display:inline-block;min-width:22px;text-align:center;border-radius:4px;font-weight:700;padding:0 4px}
.tS{background:#dc2626;color:#fff}.tA{background:#ea580c;color:#fff}.tB{background:#ca8a04;color:#fff}.tC{background:#2563eb;color:#fff}.t0{color:var(--muted);font-weight:400}.rev{margin-left:3px;font-size:11px;color:#f87171;border:1px solid #f87171;border-radius:3px;padding:0 2px}
details{margin:6px 0}summary{cursor:pointer;color:var(--accent-2)}
.ok-tools{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin:10px 0}
.ok-tools input[type=search]{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;min-width:220px}
details.okh{border:1px solid var(--border);border-radius:8px;background:var(--panel);padding:4px 12px;margin:6px 0}details.okh>summary{color:var(--text);padding:4px 0}
details.okh>summary .sid{font-family:monospace;color:var(--accent-2);margin-right:6px}details.okh>summary .chips{margin-left:6px}
details.okc{margin:4px 0 4px 8px}details.okc>summary{font-size:13px}.ok-courses{font-size:12.5px;color:var(--muted);margin:2px 0 6px}
.ok-courses b{color:var(--text);font-weight:600}tr.fam-raw td{opacity:.75}.hide{display:none!important}"""
    H = []
    w = H.append
    w(f"<!doctype html><html lang='ja'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>通説の検証結果— keiba-vpn</title><style>{css}</style></head><body>")
    w("<a href='../../index.html' class='muted'>← ドキュメント一覧</a>")
    w("<h1>通説の検証結果（全型 × 全切り口・信頼度 S/A/B/C）</h1>")
    w("<p class='muted'>ユーザーの依頼（2026-11-08）: 『すべてのパターンに関して、すべてのターゲット（コースや馬場など、あり得るすべてのターゲット）で仮説検証してください。X% の信頼度で、高いものから S&gt;A&gt;B&gt;C&gt;無 としてください。境目はあなたが決めて大丈夫です。有意差があるかどうかの検定は、データサイエンスの観点から適切に処理してください』。"
      "対象は <a href='folklore_hypotheses.html'>通説の仮説一覧</a> の 308 型（B37 を含む）。コード: <code>src/scripts/data/folklore_build.py</code>（表）・<code>src/scripts/data/folklore_verify.py</code>（検定・この章）。入力はすべてローカルの表で、GCS は読んでいない（費用は 0 円）。</p>")
    n_h = sum(1 for m in meta.values() if not m.get("na"))
    n_na = sum(1 for m in meta.values() if m.get("na"))
    adj_S = tested[(tested["adj_tier"].isin(["S", "A"])) & tested["adj_agree"]]["hid"].nunique()
    adj_rev = tested[(tested["adj_tier"].isin(["S", "A"])) & ~tested["adj_agree"]]["hid"].nunique()
    raw_S = tested[(tested["raw_tier"].isin(["S", "A"])) & tested["raw_agree"]]["hid"].nunique()
    w("<div class='note'><b>要点.</b> "
      f"検証できる形にした型 {n_h}（下位の型を含む）を、13 の切り口（全体・面・面×距離帯・競馬場×面・コース・クラス・面×クラス・面×馬場・頭数・季節・年齢・性・人気帯）の全ての値で検定した。標本の下限を満たした検定は {n_tests:,} 本。"
      f"<br>・<b>人気をそろえた差</b>（市場が見込む 3 着内率より上か下か。序列づけの本命の基準）: S {cnt_adj.get('S', 0):,}・A {cnt_adj.get('A', 0):,}・B {cnt_adj.get('B', 0):,}・C {cnt_adj.get('C', 0):,} 本。"
      f"通説の向きどおりに S か A の切り口を 1 つ以上持つ型は {adj_S}、通説と逆の向きで S・A になった型は {adj_rev}。"
      f"<br>・<b>素の差</b>（当てはまる馬の 3 着内率そのものが高いか。『通説は事実か』）: S {cnt_raw.get('S', 0):,}・A {cnt_raw.get('A', 0):,}・B {cnt_raw.get('B', 0):,}・C {cnt_raw.get('C', 0):,} 本。通説の向きで S・A を持つ型は {raw_S}。"
      f"<br>・データが無く検証できない型は {n_na}（第 7 節に理由）。"
      "<br>素の差では多くの通説が『事実』として残るが、その多くは人気（市場）が既に織り込んでおり、人気をそろえると消えるか小さくなる。序列づけに使うのは人気をそろえた差で S・A の型（第 2・3・5 節）。<br>・すべての通説を当てはめ、当てはまる数で並べたときの着順・単複回収率は <a href='#rank'>第 8 節</a>、通説の格（S/A/B/C）で得点を付け閾値を超えたレースの 1 位を買う成績と ROC は <a href='#score'>第 9 節</a>、その課題を直したモデル（市場を土台に通説を束ね、期待値で買う）は <a href='#model'>第 10 節</a>、血統（父・系統・母父）× 条件替わり（延長・短縮・初ダート・道悪など）は <a href='#blood'>第 11 節</a>。</div>")
    # 方法
    w("<h2>1. 検定の作り</h2><ul>")
    w("<li><b>対象</b>: 2016〜2025 年の JRA 平地（5 頭立て以上、着順・確定単勝オッズのある出走）。約 46 万走。履歴の列は同じ馬のその走より前だけ、騎手・調教師・父の集計はその日より前だけ（as-of）。</li>")
    w("<li><b>比較</b>: 型ごとに『資格』（eligible、例: 前走のある馬）を決め、同じ資格・同じ切り口の中で <b>当てはまる馬 − 当てはまらない馬</b> の差を取る。目的変数は 3 着内。</li>")
    w("<li><b>2 つの差</b>: (1) 素の差 = 3 着内率の差。(2) 人気をそろえた差 = 各馬の『3 着内 − 市場の見込み』の平均の差。市場の見込みは確定単勝オッズから作った単勝の市場確率を、頭数帯（〜8・9〜11・12〜14・15〜18）ごとに 40 分位に分け、2016〜2022 年の実績の 3 着内率を当てたもの。</li>")
    w("<li><b>標準誤差</b>: クラスタ頑健な分散（比の推定量の線形化）を、レース・馬・調教師・騎手・父の 5 つの単位でそれぞれ出し、<b>最も大きいもの</b>を使う（保守的）。"
      "同じレースの馬は結果が連動し（3 着内は 3 頭）、同じ馬・陣営・父は何度も出てきて実力の差が年をまたいで続くため、レース単位だけでは誤差を小さく見積もる。"
      "実際、レース単位だけで判定した初回は、調教師・騎手・父・馬の単位で無作為に選んだ偽の型が素の差で S 20・A 28 本になった（1.1）。</li>")
    w("<li><b>期間</b>: 発見 = データのある年の前 7 割（全年あれば 2016〜2022）、確認 = 残り（2023〜2025）。最終追い切りの乗り手・強度・ラップ・短評は 2016〜2019 年しか無いので発見 2016〜2018・確認 2019、SmartRC・コメント・指数は 2020〜2025 年なので発見 2020〜2023・確認 2024〜2025、バロメーターは 2024 発見・2025 確認。2026 年は使っていない。</li>")
    w(f"<li><b>標本の下限</b>: 発見期間で当てはまる馬 {MIN_N1} 走・{MIN_R1} レース以上、当てはまらない馬 {MIN_N0} 走以上。確認期間で当てはまる馬 {MIN_CONF_N1} 走以上（満たさない切り口は検定せず、多重比較の数にも入れない）。</li>")
    w("<li><b>多重比較</b>: 発見期間の全検定（型 × 切り口の値）に Benjamini–Hochberg（両側）。q &lt; 0.10 の検定だけを確認期間へ送り、確認は発見と同じ向きの片側検定にもう一度 BH。差の 2 種類（素・人気そろえ）はそれぞれ別の族として補正。</li>")
    w("<li><b>信頼度</b> = 1 − max(q<sub>発見</sub>, q<sub>確認</sub>)（発見と再現の両方を FDR で抑えた値）。<b>S ≥ 99.9%</b>・<b>A ≥ 99%</b>・<b>B ≥ 95%</b>・<b>C ≥ 90%</b>・それ未満は <b>無</b>。"
      "向きが通説と逆で有意なものには <span class='rev'>逆</span> を付けた（通説が逆だったという意味）。</li>")
    w("<li><b>回収率</b>（参考）: 当てはまる馬の単勝回収率を、払戻を 50 倍で頭打ちにした値（B37 の考え方。1 走の大穴で底上げされない）で並べる。判定には使わない。</li></ul>")
    w("<div class='note'><b>読み方の注意.</b> (1) 同じ型の切り口どうしは独立ではない（全体と面×距離帯は重なる）。型の強さは『全体』の判定と、S・A の切り口の数の両方で見る。"
      "(2) 判定条件の数値（例: 延長 100m 以上、先行率 5 割以上）は一覧の『条件の切り方』に沿って事前に決め、結果を見て変えていない。代理（データが無いところを近い列で代えたもの）は各型の『当てはめ』に書いた。"
      "(3) 人気をそろえた差がゼロでも『通説が外れ』ではなく『市場が織り込み済み』の意味。素の差と並べて読む。(4) 差の単位は pt（3 着内率の百分率の差）。</div>")
    # 偽の型
    sh_adj = shadow["adj_tier"].value_counts().to_dict()
    sh_raw = shadow["raw_tier"].value_counts().to_dict()
    w(f"<h3>1.1 偽の型（シャドウ）での誤検出</h3><p>結果と無関係な乱数で旗を立てた偽の型 12 個（1 走ごと・馬ごと・調教師ごと・騎手ごと・父ごと・開催日ごと・レースごと、割合 5〜50%）を、本物の型と同じ族に入れて同じ手順で判定した。"
      f"標本の下限を満たした偽の検定 {len(shadow):,} 本のうち、人気をそろえた差で S {sh_adj.get('S', 0)}・A {sh_adj.get('A', 0)}・B {sh_adj.get('B', 0)}・C {sh_adj.get('C', 0)}、"
      f"素の差で S {sh_raw.get('S', 0)}・A {sh_raw.get('A', 0)}・B {sh_raw.get('B', 0)}・C {sh_raw.get('C', 0)}。"
      "馬・調教師・騎手・父の単位の乱数は、同じ馬や陣営が何度も当てはまる型（リピーター・コース巧者・厩舎の型）と同じ重なりを持つ。"
      "レース単位のクラスタだけで判定した初回は、偽の検定が人気をそろえた差で S 0・A 0・B 2・C 2、素の差で S 20・A 28・B 23・C 40 本になった（無作為に選んだ調教師の組でも、実力の差は本当にあり年をまたいで続くので、素の差は『再現』してしまう）。"
      "そのため標準誤差を 5 単位の最大に改めた（上の数字は改めた後）。</p>")
    # 玄人の読み
    sa_h = tested[tested["adj_tier"].isin(["S", "A"])]
    g = sa_h.groupby("hid").agg(n=("dim", "size"), nS=("adj_tier", lambda s: int((s == "S").sum())), agree=("adj_agree", "mean"), best=("adj_conf", "max")).reset_index()
    a0 = tested[tested["dim"] == "全体"].set_index("hid")
    g = g.sort_values(["nS", "n", "best"], ascending=False)
    w("<h2>2. 読み（人気をそろえた差で S・A の切り口を持つ型）</h2><p class='muted'>『全体』の列は切り口を分けない判定。読みは玄人の目で付けた解釈（数値は表から）。</p>"
      "<div class='wrap'><table><thead><tr><th>ID</th><th>通説</th><th>S・A の切り口（うち S）</th><th>向き</th><th>全体の判定</th><th>Δ 発見/確認(pt)</th><th>当てはまる馬の 3 着内率 / 市場の見込み</th><th>読み</th></tr></thead><tbody>")
    for _, r in g.iterrows():
        z = a0.loc[r["hid"]] if r["hid"] in a0.index else None
        agree = "通説どおり" if r["agree"] >= 0.5 else "<span class='rev'>逆</span>"
        tot = f"{tier_html(z['adj_tier'], z['adj_agree'])}" if z is not None else ""
        dd = f"{pt(z['adj'])} / {pt(z['adj_c'])}" if z is not None else ""
        rr = f"{pc(z['y3_1'])}% / {pc(z['p_mkt_1'])}%" if z is not None else ""
        w(f"<tr><td class='id'>{r['hid']}</td><td>{esc(cl.get(base(r['hid']), ''))[:70]}</td><td class='num'>{int(r['n'])}（{int(r['nS'])}）</td><td>{agree}</td><td>{tot}</td><td class='num'>{dd}</td><td class='num'>{rr}</td><td>{esc(INTERP.get(r['hid'], ''))}</td></tr>")
    w("</tbody></table></div>")
    # 型ごと: 通説が正しいと言える切り口（折り畳み）
    _correct_section(w, tested, meta, cl, esc, tier_html, pt, pc, order)
    # 分類ごとの表
    w("<h2>4. 型ごとの結果</h2><p class='muted'>『全体』は切り口を分けない判定。『上位の切り口』は人気をそろえた差の信頼度が高い順に 3 つ（全体を含む）。S/A/B/C の数は人気をそろえた差で通説の向きのもの（逆の向きは括弧）。Δ は発見期間の差、確認は確認期間の差。</p>")
    cats = {}
    for r in rows:
        cats.setdefault(r[0][0], []).append(r)
    for c in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "X"]:
        if c not in cats:
            continue
        w(f"<h3 id='cat-{c}'>{c}. {CAT_NAMES[c]}</h3><div class='wrap'><table><thead><tr><th>ID</th><th>通説</th><th>向き</th><th>この検証での当てはめ</th><th>期間（発見/確認）</th>"
          "<th>全体: 人気そろえ</th><th>全体: 素の差</th><th>当てはまる馬の 3 着内率（全体）</th><th>単勝回収率（50 倍頭打ち）</th><th>S/A/B/C の切り口の数</th><th>上位の切り口（人気そろえ）</th></tr></thead><tbody>")
        for hid, m, x, a0, best in cats[c]:
            claim = esc(cl.get(base(hid), ""))
            if m.get("na"):
                w(f"<tr><td class='id'>{hid}</td><td>{claim}</td><td></td><td colspan='8' class='muted'>検証できない: {esc(m['na'])}</td></tr>")
                continue
            dirs = "買い" if m["dir"] > 0 else "消し"
            per = f"{min(m['disc'])}–{max(m['disc'])} / " + (f"{min(m['conf'])}–{max(m['conf'])}" if m.get("conf") else "なし") if m.get("disc") else "標本不足"
            if a0 is not None:
                adj = f"{tier_html(a0['adj_tier'], a0['adj_agree'])} Δ{pt(a0['adj'])}（確認 {pt(a0['adj_c'])}）"
                raw = f"{tier_html(a0['raw_tier'], a0['raw_agree'])} Δ{pt(a0['raw'])}（確認 {pt(a0['raw_c'])}）"
                rate = f"{pc(a0['y3_1'])}%（市場 {pc(a0['p_mkt_1'])}%・他 {pc(a0['y3_0'])}%）"
                roi = f"{a0['roi_cap_1'] * 100:.0f}%（他 {a0['roi_cap_0'] * 100:.0f}%）"
            else:
                adj = raw = rate = roi = "<span class='muted'>標本不足</span>"
            ok = x[x["adj_agree"]]["adj_tier"].value_counts()
            ng = x[~x["adj_agree"]]["adj_tier"].value_counts()
            cnt = "・".join(f"{k}{int(ok.get(k, 0))}" + (f"({int(ng.get(k, 0))})" if ng.get(k, 0) else "") for k in "SABC")
            bl = "<br>".join(f"{esc(b['dim'])}={esc(str(b['key']))}: {tier_html(b['adj_tier'], b['adj_agree'])} Δ{pt(b['adj'])}/{pt(b['adj_c'])}" for _, b in best.iterrows()) if len(best) else ""
            w(f"<tr><td class='id'>{hid}</td><td>{claim}</td><td>{dirs}</td><td>{esc(m['how'])}<br><span class='muted'>当てはまる {m.get('n_flag', 0):,} / 資格 {m.get('n_elig', 0):,} 走</span></td><td class='num'>{per}</td>"
              f"<td>{adj}</td><td>{raw}</td><td class='num'>{rate}</td><td class='num'>{roi}</td><td class='num'>{cnt}</td><td>{bl}</td></tr>")
        w("</tbody></table></div>")
    # S・A の一覧
    sa = tested[tested["adj_tier"].isin(["S", "A"])].copy()
    sa["o"] = sa["adj_tier"].map(order)
    sa = sa.sort_values(["o", "adj_conf", "hid"], ascending=[True, False, True])
    w(f"<h2>5. 人気をそろえた差で S・A になった切り口（{len(sa):,} 本）</h2><p class='muted'>序列づけ（展望）に使う候補。通説と逆の向きのものも含む（『逆』）。同じ型の重なる切り口が並ぶので、型の単位で束ねて使う。</p>")
    w("<div class='wrap'><table><thead><tr><th>判定</th><th>信頼度</th><th>ID</th><th>通説</th><th>切り口</th><th>値</th><th>Δ 発見(pt)</th><th>Δ 確認(pt)</th><th>素の差(pt)</th><th>当てはまる馬の 3 着内率</th><th>市場の見込み</th><th>当てはまる走数（発見/確認）</th><th>単勝回収率 頭打ち（発見/確認）</th></tr></thead><tbody>")
    for _, r in sa.iterrows():
        w(f"<tr><td>{tier_html(r['adj_tier'], r['adj_agree'])}</td><td class='num'>{r['adj_conf'] * 100:.2f}%</td><td class='id'>{r['hid']}</td><td>{esc(cl.get(base(r['hid']), ''))[:60]}</td><td>{esc(r['dim'])}</td><td>{esc(str(r['key']))}</td>"
          f"<td class='num'>{pt(r['adj'])}</td><td class='num'>{pt(r['adj_c'])}</td><td class='num'>{pt(r['raw'])}</td><td class='num'>{pc(r['y3_1'])}%</td><td class='num'>{pc(r['p_mkt_1'])}%</td>"
          f"<td class='num'>{int(r['n1']):,} / {int(r['c_n1']) if pd.notna(r['c_n1']) else 0:,}</td><td class='num'>{r['roi_cap_1'] * 100:.0f}% / {'' if pd.isna(r['c_roi_cap_1']) else f'{r.c_roi_cap_1 * 100:.0f}%'}</td></tr>")
    w("</tbody></table></div>")
    # B の切り口（素の差で S・A）
    rs = tested[tested["raw_tier"].isin(["S", "A"]) & (tested["dim"] == "全体")].copy()
    rs = rs.sort_values("raw_conf", ascending=False)
    w(f"<h2>6. 素の差で S・A（全体）: 通説は事実だが人気が織り込んでいるか（{len(rs)} 型）</h2><div class='wrap'><table><thead><tr><th>ID</th><th>通説</th><th>素の差</th><th>Δ 素(pt)</th><th>人気そろえ</th><th>Δ 人気そろえ(pt)</th><th>当てはまる馬の 3 着内率</th><th>市場の見込み</th></tr></thead><tbody>")
    for _, r in rs.iterrows():
        w(f"<tr><td class='id'>{r['hid']}</td><td>{esc(cl.get(base(r['hid']), ''))[:70]}</td><td>{tier_html(r['raw_tier'], r['raw_agree'])}</td><td class='num'>{pt(r['raw'])}</td><td>{tier_html(r['adj_tier'], r['adj_agree'])}</td><td class='num'>{pt(r['adj'])}</td>"
          f"<td class='num'>{pc(r['y3_1'])}%</td><td class='num'>{pc(r['p_mkt_1'])}%</td></tr>")
    w("</tbody></table></div>")
    # 検証できない
    w("<h2>7. 検証できなかった型と注意</h2><ul>")
    for hid, m in meta.items():
        if m.get("na"):
            w(f"<li><b>{hid}</b> {esc(cl.get(base(hid), ''))[:50]} — {esc(m['na'])}</li>")
    w("</ul><div class='note'><b>限界.</b> (1) 最終追い切りは出馬表の 1 本だけで、中間の本数・1 週前・併せは無い（B 分類の多くと B37 の『併せ・1 週前・本数』は代理か対象外）。B37 は『場所 × 強さ』のパターンで、調教師 × パターンの as-of の人気補正後の上積みを 50 走ぶん全体へ縮めた値で選んだ。"
      "(2) 海外・地方の走歴は出馬表の過去 5 走（2020〜2025）から。(3) 外国人・短期免許・減量・女性騎手は名前・斤量からの推定。(4) 内回り/外回りは距離からの規則で、京都 1400/1600・新潟 2000 は不明扱い。"
      "(5) 市場の見込みは確定オッズ（発走時点の情報）。序列を前日に作るなら、人気をそろえた差は前日オッズで測り直す必要がある。(6) 改修（京都）・斤量の改定（2023）は期間をまたぐ型の効果を薄める。</div>")
    _rank_section(w)
    _score_section(w)
    from src.scripts.data.folklore_model import model_section

    model_section(w)
    from src.scripts.data.folklore_sire import sire_section

    sire_section(w)
    w(f"<p class='muted'>生成: {pd.Timestamp.now(tz='Asia/Tokyo'):%Y-%m-%d %H:%M}（python -m src.scripts.data.folklore_verify report）。</p></body></html>")
    DOC.write_text("\n".join(H), encoding="utf-8")
    print("wrote", DOC, len("\n".join(H)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "judge":
        judge(pd.read_parquet(OUT / "tests_raw.parquet"), json.load(open(OUT / "meta.json")))
    elif cmd == "report":
        report()
    elif cmd == "rank":
        rank()
    elif cmd == "score":
        score()
