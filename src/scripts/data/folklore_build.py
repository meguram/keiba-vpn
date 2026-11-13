"""通説の検証（folklore_verification）用の 1 出走 = 1 行の表を作る。

python -m src.scripts.data.folklore_build            # 出力 <FOLKLORE_OUT>/master.parquet

- 入力はすべてローカル（data/page_reference/tables・cushion・knowledge、data/local/features/horse/ped_tbl）。GCS は読まない。
- 履歴の列は同じ馬の『その走より前』の出走だけ（2016 年以降の JRA の出走。障害も履歴には含める）。
- 騎手・調教師・父の集計は『その日より前』だけ（日ごとに集計して 1 日ずらす）。同じ日の前のレースまでの騎手の成績は、発走時刻順に数える。
- 市場（確定の単勝オッズ・人気）は比較の基準（人気をそろえた差）と人気帯の切り口にだけ使う。
"""

from __future__ import annotations

import json
import sys
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("FOLKLORE_OUT", "/home/jovyan/work/folklore_out"))
TABLES = ROOT / "data/page_reference/tables"
# 最後の年（既定 2025）。FOLKLORE_YEAR_END=2026 で 2026 年の行も作る（重賞の予想家の検証用。出力先は FOLKLORE_OUT で分ける）
YEAR_END = int(os.environ.get("FOLKLORE_YEAR_END", "2025"))
# 発走して中止した馬（着順なし・オッズが正）の行も master に残す（既定 0 = 今までどおり着順の付いた馬だけ）。
# 重賞の予想家で、中止の馬を「着外」として集計・評価に入れるため（docs/ideas/graded_rule_based_handicapper.md 第 16 節）
KEEP_DNF = os.environ.get("FOLKLORE_KEEP_DNF", "0") == "1"
YEARS = list(range(2016, YEAR_END + 1))
LOCAL_VC = {"01", "02", "03", "04", "10"}
CENTRAL_VC = {"05", "06", "08", "09"}
CLS = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "L": 5, "G3": 6, "G2": 7, "G1": 8}
OVERSEAS = ("メイダン", "シャティン", "ロンシャン", "パリロンシャン", "サンタアニタ", "チャーチルダウンズ", "アスコット", "ドバイ", "キングアブドゥルアジーズ",
            "リヤド", "デルマー", "ケンタッキー", "ベルモント", "シャンティイ", "ドーヴィル", "ニューマーケット", "エプソム", "グッドウッド", "ヨーク",
            "レパーズタウン", "カラ", "フレミントン", "ムーニーバレー", "ランドウィック", "コーフィールド", "ローズヒル", "ハッピーバレー", "キーンランド",
            "ガルフストリーム", "ジェベルアリ", "アルクオーツ", "サウジ", "ソウル", "ブダペスト", "バーデンバーデン", "ミュンヘン", "ケルン", "ホッペガルテン")
NAR = ("大井", "川崎", "船橋", "浦和", "園田", "姫路", "名古屋", "笠松", "金沢", "高知", "佐賀", "門別", "盛岡", "水沢", "帯広", "ばんえい")
# 内回り/外回り（芝）。両方あるのは中山・京都・阪神・新潟。京都 1400/1600・新潟 2000 は両方あるので不明（None）。
IO_RULE = {
    ("06", 1200): "外", ("06", 1600): "外", ("06", 2200): "外", ("06", 1800): "内", ("06", 2000): "内", ("06", 2500): "内", ("06", 3600): "内",
    ("08", 1200): "内", ("08", 1800): "外", ("08", 2000): "内", ("08", 2200): "外", ("08", 2400): "外", ("08", 3000): "外", ("08", 3200): "外",
    ("09", 1200): "内", ("09", 1400): "内", ("09", 1600): "外", ("09", 1800): "外", ("09", 2000): "内", ("09", 2200): "内", ("09", 2400): "外",
    ("09", 2600): "外", ("09", 3000): "内", ("09", 3200): "外",
    ("04", 1000): "直", ("04", 1200): "内", ("04", 1400): "内", ("04", 1600): "外", ("04", 1800): "外", ("04", 2200): "内", ("04", 2400): "内",
}
ROBERTO = {"Roberto", "ブライアンズタイム", "シンボリクリスエス", "Kris S.", "スクリーンヒーロー", "グラスワンダー", "Silver Hawk", "Red Ransom", "タニノギムレット",
           "エピファネイア", "モーリス", "リアルシャダイ", "マヤノトップガン", "サクラローレル"}
STAYGOLD = {"ステイゴールド", "オルフェーヴル", "ゴールドシップ", "ドリームジャーニー", "ナカヤマフェスタ", "フェノーメノ", "オジュウチョウサン"}
KINGMAMBO = {"Kingmambo", "キングカメハメハ", "ロードカナロア", "ルーラーシップ", "ドゥラメンテ", "エルコンドルパサー", "King's Best", "Lemon Drop Kid",
             "レイデオロ", "ホッコータルマエ", "ベルシャザール", "リオンディーズ", "トゥザグローリー", "ローズキングダム", "アパパネ", "ミッキーアイル"}
DEEP = {"ディープインパクト"}


def _f(x):
    return pd.to_numeric(x, errors="coerce")


def gshift_cum(v: pd.Series, by: pd.Series, how: str) -> pd.Series:
    """馬ごとに 1 つずらしてから累積（cummax/cummin）。lambda の transform より速い。"""
    s1 = v.groupby(by, sort=False).shift(1)
    return getattr(s1.groupby(by, sort=False), how)()


def gshift_roll(v: pd.Series, by: pd.Series, win: int, minp: int, how: str) -> pd.Series:
    s1 = v.groupby(by, sort=False).shift(1)
    r = getattr(s1.groupby(by, sort=False).rolling(win, min_periods=minp), how)()
    return r.reset_index(level=0, drop=True).reindex(v.index)


def read_years(name: str, years, cols: list[str]) -> pd.DataFrame:
    """年ごとの表を、ある列だけ読む（race_id・horse_id の無い年は飛ばす）。"""
    import pyarrow.parquet as pq
    parts = []
    for y in years:
        f = TABLES / str(y) / f"{name}.parquet"
        if not f.exists():
            continue
        have = set(pq.ParquetFile(f).schema_arrow.names)
        if not {"race_id", "horse_id"} <= have:
            print("skip", name, y, "(race_id/horse_id 無し)", flush=True)
            continue
        parts.append(pd.read_parquet(f, columns=[c for c in cols if c in have]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def load_results() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "venue_code", "surface", "distance", "direction", "grade", "race_class", "weather", "track_condition",
            "start_time", "field_size", "race_name", "round", "pace", "lap_times", "finish_position", "bracket_number", "horse_number", "horse_id",
            "sex_age", "jockey_weight", "jockey_name", "jockey_id", "time_sec", "passing_order", "last_3f", "odds", "popularity", "weight",
            "weight_change", "trainer_id"]
    parts = [pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS]
    d = pd.concat(parts, ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    d = d[d["horse_id"].str.len() > 0].drop_duplicates(["race_id", "horse_id"])
    sh = read_years("race_shutuba_flat", YEARS, ["race_id", "horse_id", "weight_rule"])
    sh["race_id"] = sh["race_id"].astype(str)
    sh["horse_id"] = sh["horse_id"].astype(str)
    rule = sh.drop_duplicates("race_id")[["race_id", "weight_rule"]]
    d = d.merge(rule, on="race_id", how="left")
    return d


def course_meta() -> dict:
    cp = json.load(open(ROOT / "data/page_reference/knowledge/course_profiles.json", encoding="utf-8"))["venues"]
    out = {}
    for vc, v in cp.items():
        out[vc] = {
            "turf_straight": v.get("turf_straight_m"), "dirt_straight": v.get("dirt_straight_m"),
            "slope": bool(v.get("has_slope")), "shape": v.get("course_shape"),
            "first_corner": {k: (x or {}).get("first_corner_m") for k, x in (v.get("draw_bias") or {}).items()},
            "corners": {k: (x or {}).get("corners") for k, x in (v.get("draw_bias") or {}).items()},
        }
    return out


def parse_basic(d: pd.DataFrame) -> pd.DataFrame:
    d["dt"] = pd.to_datetime(d["date"])
    d["year"] = d["dt"].dt.year
    d["month"] = d["dt"].dt.month
    d["vc"] = d["venue_code"].astype(str).str.zfill(2)
    d["surf"] = d["surface"].map({"芝": "芝", "ダート": "ダ", "障": "障"}).fillna("他")
    d["dist"] = _f(d["distance"])
    d["field"] = _f(d["field_size"])
    d["fin"] = _f(d["finish_position"])
    d["hn"] = _f(d["horse_number"])
    d["odds"] = _f(d["odds"])
    d["pop"] = _f(d["popularity"])
    d["wt"] = _f(d["weight"])
    d["wtc"] = _f(d["weight_change"])
    d["jw"] = _f(d["jockey_weight"])
    d["l3f"] = _f(d["last_3f"])
    d["tsec"] = _f(d["time_sec"])
    d["round"] = _f(d["round"])
    d["cls"] = d["grade"].map(CLS)
    d["sex"] = d["sex_age"].astype(str).str[0]
    d["age"] = _f(d["sex_age"].astype(str).str[1:])
    d["track"] = d["track_condition"].map({"良": "良", "稍重": "稍", "重": "重", "不良": "不"})
    d["bad"] = d["track"].isin(["重", "不"])
    d["fillies"] = d["race_class"].astype(str).str.contains("牝") | d["race_name"].astype(str).str.contains("牝")
    d["y3"] = (d["fin"] <= 3).astype(float)
    d["y1"] = (d["fin"] == 1).astype(float)
    d["dband"] = pd.cut(d["dist"], [0, 1300, 1500, 1799, 2200, 9999], labels=["~1300", "1400", "1600", "1800-2200", "2300~"]).astype(str)
    d["course"] = d["venue"].astype(str) + d["surf"] + d["dist"].astype("Int64").astype(str)
    d["io"] = [IO_RULE.get((v, int(x))) if s == "芝" and x == x else None for v, x, s in zip(d["vc"], d["dist"], d["surf"])]
    cm = course_meta()
    st, sl, shp, fc, nc = [], [], [], [], []
    for v, s, x in zip(d["vc"], d["surf"], d["dist"]):
        m = cm.get(v, {})
        st.append(m.get("turf_straight") if s == "芝" else m.get("dirt_straight"))
        sl.append(m.get("slope"))
        shp.append(m.get("shape"))
        key = ("turf_" if s == "芝" else "dirt_") + (str(int(x)) if x == x else "")
        fc.append((m.get("first_corner") or {}).get(key))
        nc.append((m.get("corners") or {}).get(key))
    d["straight"] = _f(pd.Series(st, index=d.index))
    d["slope"] = pd.Series(sl, index=d.index).astype("boolean")
    d["shape"] = shp
    d["first_corner"] = _f(pd.Series(fc, index=d.index))
    d["corners"] = _f(pd.Series(nc, index=d.index))
    d["local"] = d["vc"].isin(LOCAL_VC)
    d["direction"] = d["direction"].astype(str)
    # 通過順・時間差
    po = d["passing_order"].fillna("").astype(str).str.split("-")
    d["pos1"] = _f(po.str[0])
    d["pos4"] = _f(po.str[-1])
    d["pos1r"] = d["pos1"] / d["field"]
    d["pos4r"] = d["pos4"] / d["field"]
    d.loc[d["fin"] <= 0, "fin"] = np.nan
    g = d.groupby("race_id")
    d["tdiff"] = d["tsec"] - g["tsec"].transform("min")
    d["l3rank"] = g["l3f"].rank(method="min")
    # 取消などでオッズが 0 の馬がいると 1/0 = inf でレース全体の q が 0 になるので、オッズが正の馬だけで正規化する
    inv = 1.0 / d["odds"].where(d["odds"] > 0)
    d["q"] = inv / inv.groupby(d["race_id"]).transform("sum")
    # ペース（前半 3F − 後半 3F）とラップの型
    def pdiff(s):
        try:
            j = json.loads(s)
            return j.get("first_half_3f", np.nan) - j.get("second_half_3f", np.nan)
        except Exception:
            return np.nan
    rr = d.drop_duplicates("race_id")[["race_id", "pace", "lap_times", "surf", "dband"]].copy()
    rr["pdiff"] = rr["pace"].map(pdiff)
    def kick(s):
        try:
            laps = json.loads(s)
            if len(laps) < 6:
                return np.nan
            mid = np.mean(laps[1:-3])
            return mid - np.mean(laps[-3:])
        except Exception:
            return np.nan
    rr["kick"] = rr["lap_times"].map(kick)
    base = rr[rr["race_id"].str[:4].astype(int) <= 2019]
    for c, lab in (("pdiff", "pace_cat"), ("kick", "lap_cat")):
        q = base.groupby(["surf", "dband"])[c].quantile([1 / 3, 2 / 3]).unstack()
        lo = rr.set_index(["surf", "dband"]).index.map(lambda k: q.loc[k, 1 / 3] if k in q.index else np.nan)
        hi = rr.set_index(["surf", "dband"]).index.map(lambda k: q.loc[k, 2 / 3] if k in q.index else np.nan)
        v = rr[c].to_numpy()
        # pdiff: 前半−後半 が小さい（前半が速い）ほどハイペース。kick: 大きいほど瞬発（終いが速い）
        if c == "pdiff":
            rr[lab] = np.where(np.isnan(v), None, np.where(v < np.asarray(lo, float), "H", np.where(v > np.asarray(hi, float), "S", "M")))
        else:
            rr[lab] = np.where(np.isnan(v), None, np.where(v > np.asarray(hi, float), "瞬発", np.where(v < np.asarray(lo, float), "持続", "中")))
    d = d.merge(rr[["race_id", "pdiff", "pace_cat", "lap_cat"]], on="race_id", how="left")
    # 発走時刻（同じ日の騎手の前のレース）
    d["stime"] = pd.to_datetime(d["date"] + " " + d["start_time"].fillna("00:00").astype(str), errors="coerce")
    return d


def add_history(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    prev = ["dt", "vc", "surf", "dist", "cls", "fin", "field", "tdiff", "pop", "odds", "pos1", "pos4", "pos1r", "pos4r", "l3rank", "jockey_id",
            "trainer_id", "wt", "jw", "track", "direction", "pace_cat", "lap_cat", "race_name", "weight_rule", "io", "straight", "slope", "local",
            "fillies", "y1", "y3", "dband", "course", "sex", "grade", "q", "round", "venue", "bad", "month"]
    for c in prev:
        d["p_" + c] = g[c].shift(1)
    for c in ["dt", "fin", "wt", "cls", "y1", "pos1", "tdiff", "jockey_id"]:
        d["p2_" + c] = g[c].shift(2)
    for c in ["dt", "wt", "fin"]:
        d["p3_" + c] = g[c].shift(3)
    d["p_jump"] = d["p_surf"] == "障"
    d["interval"] = (d["dt"] - d["p_dt"]).dt.days
    d["n_prev"] = g.cumcount()
    # 休み明けからの戦数（70 日以上の間隔で区切る）
    brk = (d["interval"].isna()) | (d["interval"] >= 70)
    seq = brk.astype(int).groupby(d["horse_id"]).cumsum()
    d["since_rest"] = d.groupby([d["horse_id"], seq]).cumcount() + 1
    d["rest_len"] = d["interval"].where(brk).groupby([d["horse_id"], seq]).transform("first")
    d.loc[d["n_prev"] == 0, ["since_rest", "rest_len"]] = np.nan
    # 前の休み明け（過去の休み明け初戦の成績）
    fb = (brk & (d["n_prev"] > 0)).astype(float)
    d["fb_flag"] = fb
    d["fb_n_prev"] = (fb).groupby(d["horse_id"]).cumsum() - fb
    d["fb_top3_prev"] = (fb * d["y3"]).groupby(d["horse_id"]).cumsum() - fb * d["y3"]
    k23 = d["since_rest"].isin([2, 3]).astype(float)
    d["k23_n_prev"] = k23.groupby(d["horse_id"]).cumsum() - k23
    d["k23_top3_prev"] = (k23 * d["y3"]).groupby(d["horse_id"]).cumsum() - k23 * d["y3"]
    # 通算・条件別の累積（その走を含まない）
    d["ok"] = d["fin"].notna() & (d["surf"] != "障")
    okf = d["ok"].astype(float)
    def cum(key_cols, val, name):
        k = [d["horse_id"]] + [d[c] for c in key_cols]
        d[name] = val.groupby(k).cumsum() - val
    y3o = d["y3"] * okf
    y1o = d["y1"] * okf
    cum([], okf, "starts_prev")
    cum([], y3o, "top3_prev")
    cum([], y1o, "wins_prev")
    cum(["course"], okf, "course_n_prev")
    cum(["course"], y3o, "course_top3_prev")
    d["vs"] = d["vc"] + d["surf"]
    cum(["vs"], okf, "vs_n_prev")
    cum(["vs"], y3o, "vs_top3_prev")
    cum(["vc"], okf, "venue_n_prev")
    cum(["surf"], okf, "surf_n_prev")
    cum(["surf"], y3o, "surf_top3_prev")
    cum(["dband"], okf, "dband_n_prev")
    cum(["dband"], y3o, "dband_top3_prev")
    cum(["dband", "surf"], okf, "dbs_n_prev")
    cum(["dband", "surf"], y3o, "dbs_top3_prev")
    badf = d["bad"].astype(float) * okf
    cum([], badf, "bad_n_prev")
    cum([], badf * d["y3"], "bad_top3_prev")
    goodf = (d["track"] == "良").astype(float) * okf
    cum([], goodf, "good_n_prev")
    cum([], goodf * d["y3"], "good_top3_prev")
    d["dir2"] = d["direction"].where(d["direction"].isin(["右", "左"]))
    cum(["dir2"], okf, "dir_n_prev")
    cum(["dir2"], y3o, "dir_top3_prev")
    hk = d["vc"].isin(["01", "02"]) & (d["surf"] == "芝")
    cum([], hk.astype(float) * okf, "yoshiba_n_prev")
    g = d.groupby("horse_id", sort=False)
    d["max_field_prev"] = gshift_cum(d["field"], d["horse_id"], "cummax")
    d["fav1_prev"] = (d["pop"] == 1).astype(float).groupby(d["horse_id"]).cumsum() - (d["pop"] == 1).astype(float)
    nige = ((d["pos1"] == 1) & d["ok"]).astype(float)
    senko = ((d["pos1r"] <= 0.3) & d["ok"]).astype(float)
    d["nige_prev"] = nige.groupby(d["horse_id"]).cumsum() - nige
    d["senko_prev"] = senko.groupby(d["horse_id"]).cumsum() - senko
    d["nige_rate"] = d["nige_prev"] / d["starts_prev"].replace(0, np.nan)
    d["senko_rate"] = d["senko_prev"] / d["starts_prev"].replace(0, np.nan)
    d["pos1r_mean_prev"] = (d["pos1r"].fillna(0) * okf).groupby(d["horse_id"]).cumsum().sub(d["pos1r"].fillna(0) * okf) / d["starts_prev"].replace(0, np.nan)
    # 逃げた時・逃げなかった時の 3 着内（逃げ専）
    cum([], nige * d["y3"], "nige_top3_prev")
    # 上がり最速の回数・上がり順位の平均
    l1 = ((d["l3rank"] == 1) & d["ok"]).astype(float)
    d["agari1_prev"] = l1.groupby(d["horse_id"]).cumsum() - l1
    # 同じレース名での 3 着内（リピーター）
    rn = d["race_name"].astype(str).str.replace(r"\s", "", regex=True)
    d["rn"] = rn
    gr = d["grade"].isin(["G1", "G2", "G3"])
    tmp = (d["y3"] * okf * gr).groupby([d["horse_id"], d["rn"]]).cumsum() - d["y3"] * okf * gr
    d["rn_top3_prev"] = tmp.where(gr)
    # 継続騎乗・連勝
    same_j = (d["jockey_id"] == d["p_jockey_id"]).astype(int)
    blk = (same_j == 0).astype(int).groupby(d["horse_id"]).cumsum()
    d["j_cont"] = same_j.groupby([d["horse_id"], blk]).cumsum()  # 0 = テン乗り/乗り替わり, 1 = 2 戦連続 …
    w = d["y1"].fillna(0).astype(int)
    blk2 = (w == 0).astype(int).groupby(d["horse_id"]).cumsum()
    streak = w.groupby([d["horse_id"], blk2]).cumsum()
    d["win_streak"] = streak.groupby(d["horse_id"]).shift(1)
    # 掲示板続き（直近 5 走）
    for c, v in (("r5_top3", d["y3"]), ("r5_win", d["y1"]), ("r5_top5", (d["fin"] <= 5).astype(float))):
        d[c] = gshift_roll(v, d["horse_id"], 5, 3, "sum")
    # ベスト体重（3 着内の時の体重の平均）
    wt3 = (d["wt"].fillna(0) * d["y3"] * okf)
    n3 = (d["y3"] * okf * d["wt"].notna())
    d["best_wt"] = (wt3.groupby(d["horse_id"]).cumsum() - wt3) / (n3.groupby(d["horse_id"]).cumsum() - n3).replace(0, np.nan)
    # 去勢（牡 → セ）からの戦数・転厩からの戦数
    gel = ((d["sex"] == "セ") & (d["p_sex"] == "牡")).astype(int)
    gblk = gel.groupby(d["horse_id"]).cumsum()
    d["since_geld"] = np.where(gblk > 0, d.groupby([d["horse_id"], gblk]).cumcount() + 1, np.nan)
    tr = ((d["trainer_id"] != d["p_trainer_id"]) & d["p_trainer_id"].notna()).astype(int)
    tblk = tr.groupby(d["horse_id"]).cumsum()
    d["since_transfer"] = np.where(tblk > 0, d.groupby([d["horse_id"], tblk]).cumcount() + 1, np.nan)
    # 好走距離帯（3 着内の距離の範囲）
    dist3 = d["dist"].where((d["y3"] == 1) & d["ok"])
    d["best_dist_min"] = gshift_cum(dist3, d["horse_id"], "cummin")
    d["best_dist_max"] = gshift_cum(dist3, d["horse_id"], "cummax")
    # 経験した最長・最短の距離
    dd = d["dist"].where(d["ok"])
    d["dist_max_prev"] = gshift_cum(dd, d["horse_id"], "cummax")
    d["dist_min_prev"] = gshift_cum(dd, d["horse_id"], "cummin")
    # 重賞・OP の経験、古馬との対戦経験
    gop = (d["cls"] >= 5).astype(float)
    d["op_n_prev"] = gop.groupby(d["horse_id"]).cumsum() - gop
    old = d["race_class"].astype(str).str.contains("以上").astype(float)
    d["vs_old_prev"] = old.groupby(d["horse_id"]).cumsum() - old
    d["mixed"] = old.astype(bool)
    # 遠征（所属と逆の地域・北海道・小倉）の経験は後で東西が付いてから
    return d


def daily_asof(d: pd.DataFrame, keys: list[str], name: str, window: str | None = None, ycols=("y3", "y1")) -> pd.DataFrame:
    """keys ごとの『その日より前』の出走数・3 着内数・勝ち数（window を指定すると直近の期間だけ）。"""
    m = d[d["ok"] & d[keys].notna().all(axis=1)]
    agg = m.groupby(keys + ["dt"], sort=False).agg(n=("y3", "size"), **{c: (c, "sum") for c in ycols}).reset_index()
    agg = agg.sort_values(keys + ["dt"], kind="stable")
    if window is None:
        cs = agg.groupby(keys, sort=False)[["n", *ycols]].cumsum() - agg[["n", *ycols]]
    else:
        parts = []
        for _, gdf in agg.groupby(keys, sort=False):
            r = gdf.set_index("dt")[["n", *ycols]].rolling(window, closed="left").sum()
            parts.append(r.fillna(0).set_index(gdf.index))
        cs = pd.concat(parts).loc[agg.index]
    agg = pd.concat([agg[keys + ["dt"]], cs.add_prefix(name + "_")], axis=1)
    # その日（dt）の値は『その日より前』の累計。d の各行（同じ dt）に付ける
    return d.merge(agg, on=keys + ["dt"], how="left")


def add_people(d: pd.DataFrame) -> pd.DataFrame:
    d = daily_asof(d, ["jockey_id"], "j365", "365D")
    d = daily_asof(d, ["trainer_id"], "t365", "365D")
    d = daily_asof(d, ["trainer_id"], "t30", "30D")
    d = daily_asof(d, ["jockey_id", "trainer_id"], "jt")
    d = daily_asof(d, ["jockey_id", "course"], "jc")
    d = daily_asof(d, ["trainer_id", "course"], "tc")
    d = daily_asof(d, ["jockey_id", "grade"], "jg")
    d["j_rate"] = (d["j365_y3"] / d["j365_n"]).where(d["j365_n"] >= 100)
    d["t_rate"] = (d["t365_y3"] / d["t365_n"]).where(d["t365_n"] >= 50)
    d["t_rank"] = d.groupby("year")["t365_y1"].rank(ascending=False, pct=True)
    # 乗り替わり前の騎手の格（前走騎手の今の j_rate）
    jr = d[["jockey_id", "dt", "j_rate"]].drop_duplicates(["jockey_id", "dt"]).rename(columns={"jockey_id": "p_jockey_id", "j_rate": "pj_rate"})
    d = d.merge(jr, on=["p_jockey_id", "dt"], how="left")
    # 前走騎手のその日の j_rate が無い（その日に乗っていない）ときは前走時の j_rate を使う
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    d["pj_rate_then"] = d.groupby("horse_id")["j_rate"].shift(1)
    d["pj_rate"] = d["pj_rate"].fillna(d["pj_rate_then"])
    # 調教師の休み明けの成績（as-of）
    fb = d[d["fb_flag"] == 1]
    agg = fb[fb["ok"]].groupby(["trainer_id", "dt"]).agg(n=("y3", "size"), y3=("y3", "sum")).reset_index().sort_values(["trainer_id", "dt"])
    cs = agg.groupby("trainer_id")[["n", "y3"]].cumsum() - agg[["n", "y3"]]
    agg["tfb_n"], agg["tfb_y3"] = cs["n"], cs["y3"]
    d = d.merge(agg[["trainer_id", "dt", "tfb_n", "tfb_y3"]], on=["trainer_id", "dt"], how="left")
    # 同じ日の騎手の前のレースまで
    d = d.sort_values(["jockey_id", "stime", "race_id"], kind="stable").reset_index(drop=True)
    gj = d.groupby(["jockey_id", "date"], sort=False)
    d["jday_k"] = gj.cumcount() + 1
    d["jday_n"] = gj["race_id"].transform("size")
    d["jday_w_before"] = gj["y1"].cumsum() - d["y1"]
    d["jday_minpop_rank"] = gj["pop"].rank(method="min")  # 騎手のその日の騎乗馬の中の人気の順位
    jd = d[["jockey_id", "dt"]].drop_duplicates().sort_values(["jockey_id", "dt"])
    jd["j_gap"] = jd.groupby("jockey_id")["dt"].diff().dt.days
    d = d.merge(jd, on=["jockey_id", "dt"], how="left")
    # 騎手の積極性（as-of の逃げ率）
    d["nige_f"] = ((d["pos1"] == 1) & d["ok"]).astype(float)
    m = d[d["ok"]].groupby(["jockey_id", "dt"]).agg(n=("nige_f", "size"), k=("nige_f", "sum")).reset_index().sort_values(["jockey_id", "dt"])
    cs = m.groupby("jockey_id")[["n", "k"]].cumsum() - m[["n", "k"]]
    m["jn_n"], m["jn_k"] = cs["n"], cs["k"]
    d = d.merge(m[["jockey_id", "dt", "jn_n", "jn_k"]], on=["jockey_id", "dt"], how="left")
    d["j_nige_rate"] = (d["jn_k"] / d["jn_n"]).where(d["jn_n"] >= 200)
    # 外国人騎手（名前がカタカナ・全角英字だけ）と短期免許（直近 365 日の騎乗が少ない）
    kat = d["jockey_name"].astype(str).str.fullmatch(r"[ァ-ヶー・．Ａ-Ｚ]+")
    d["j_foreign"] = kat
    d["j_short"] = kat & (d["j365_n"].fillna(0) < 300)
    # 減量（馬齢・定量戦で、同じ性・年齢の馬の斤量の最頻値との差）
    d["sexg"] = np.where(d["sex"] == "牝", "F", "M")
    base = d.groupby(["race_id", "sexg", "age"])["jw"].transform("max")
    d["allow"] = (base - d["jw"]).where(d["weight_rule"].isin(["馬齢", "定量"]))
    return d


def add_external(d: pd.DataFrame) -> pd.DataFrame:
    key = ["race_id", "horse_id"]
    # SmartRC（2020〜YEAR_END）
    sc = ["race_id", "hcode", "blinker", "tozai", "owner", "producer", "f_name", "ff_name", "fff_name", "mf_name", "cl_eval_cmb", "f_dirt_share",
          "f_1400_share", "ten1f_best_rank", "agari_has_rank", "ten_has_rank"]
    s = pd.concat([pd.read_parquet(TABLES / str(y) / "smartrc_race_flat.parquet", columns=sc) for y in range(2020, YEAR_END + 1)])
    s["race_id"] = s["race_id"].astype(str)
    s["horse_id"] = s["hcode"].astype(str)
    s = s.drop(columns=["hcode"]).drop_duplicates(key)
    d = d.merge(s, on=key, how="left")
    d["blinker"] = _f(d["blinker"])
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    d["p_blinker"] = d.groupby("horse_id")["blinker"].shift(1)
    d["blk_ever_prev"] = gshift_cum(d["blinker"].fillna(0), d["horse_id"], "cummax")
    # 東西: 調教師ごとの最頻値を全年に使う
    tz = d.dropna(subset=["tozai"]).groupby("trainer_id")["tozai"].agg(lambda s: s.mode().iloc[0])
    d["tozai"] = d["trainer_id"].map(tz).map({"1": "美浦", "2": "栗東"})
    # 父・母父: SmartRC の馬ごとの値、無ければ血統表
    hp = d.dropna(subset=["f_name"]).drop_duplicates("horse_id").set_index("horse_id")[["f_name", "ff_name", "fff_name", "mf_name", "cl_eval_cmb", "f_dirt_share", "f_1400_share"]]
    for c in hp.columns:
        d[c] = d["horse_id"].map(hp[c]).where(d[c].isna(), d[c]) if c in d else d["horse_id"].map(hp[c])
    try:
        import pyarrow.dataset as pads
        ds = pads.dataset(str(ROOT / "data/local/features/horse/ped_tbl"), format="parquet")
        cols = ["subject_horse_id", "ancestor_name"]
        t1 = ds.to_table(columns=cols, filter=(pads.field("generation") == 1) & (pads.field("path_fm") == "F")).to_pandas()
        t2 = ds.to_table(columns=cols, filter=(pads.field("generation") == 2) & (pads.field("path_fm") == "MF")).to_pandas()
        t3 = ds.to_table(columns=cols, filter=(pads.field("generation") == 2) & (pads.field("path_fm") == "FF")).to_pandas()
        def clean(x):
            return x.astype(str).str.replace(r"[A-Za-z].*$", "", regex=True).where(lambda s: s.str.len() > 0, x.astype(str).str.replace(r"\(.*$", "", regex=True))
        for t, c in ((t1, "f_name"), (t2, "mf_name"), (t3, "ff_name")):
            t = t.drop_duplicates("subject_horse_id")
            mp = pd.Series(clean(t["ancestor_name"]).to_numpy(), index=t["subject_horse_id"].astype(str))
            d[c] = d[c].fillna(d["horse_id"].map(mp))
    except Exception as e:  # noqa: BLE001
        print("ped_tbl skipped", e)
    for c in ["f_name", "ff_name", "fff_name", "mf_name"]:
        d[c] = d[c].astype(str).str.strip().replace({"nan": None, "None": None})
    chain = d[["f_name", "ff_name", "fff_name"]].astype(str)
    def line(names):
        return chain.apply(lambda col: col.isin(names) | col.str[:8].isin({n[:8] for n in names})).any(axis=1)
    d["line"] = np.select([line(DEEP), line(STAYGOLD), line(ROBERTO), line(KINGMAMBO)],
                          ["ディープ系", "ステイゴールド系", "ロベルト系", "キングマンボ系"], default="他")
    d.loc[d["f_name"].isna(), "line"] = None
    d["cl_cmb"] = d["cl_eval_cmb"].astype(str).str.replace("　", "")
    # 追い切り（最終）。乗り手・ラップ・強度・短評は 2016〜2019 だけ、評価は 2025 まで
    oc = ["race_id", "horse_id", "training_date", "course", "condition", "rider", "lap_times", "impression", "evaluation", "comment"]
    o = read_years("race_oikiri_flat", YEARS, oc)
    o["race_id"] = o["race_id"].astype(str)
    o["horse_id"] = o["horse_id"].astype(str)
    o = o.drop_duplicates(key)
    for c in oc[2:]:
        o[c] = o[c].fillna("").astype(str)
    tok = o["rider"].str.fullmatch(r"[美栗札函新福小東京阪中][坂ＷＰ芝ダＢＣＤＥプ障南北]")
    o["ok_course"] = np.where(o["course"] != "", o["course"], np.where(tok, o["rider"], ""))
    o["ok_rider"] = np.where(tok, "", o["rider"])
    o["ok_rider"] = o["ok_rider"].where(o["ok_rider"] != "", None)
    o["ok_eval"] = o["evaluation"].where(o["evaluation"].isin(list("ABCD")))
    o["ok_int"] = o["impression"].replace({"Ｇ強": "強め", "Ｇ一": "一杯", "直強": "強め", "直一": "一杯", "仕掛": "強め"}).where(o["impression"] != "")
    o["ok_tanpyo"] = o["comment"].where(o["comment"] != "")
    o["ok_cond"] = o["condition"].where(o["condition"] != "")
    td = pd.to_datetime(o["training_date"].str[:10], format="%Y/%m/%d", errors="coerce")
    o["ok_td"] = td
    o["ok_wd"] = o["training_date"].str.extract(r"\((.)\)")[0]
    def laps(s):
        v = [float(x) for x in re.findall(r"(\d+\.\d)\(", s)]
        last = re.findall(r"\((\d+\.\d)\)$", s)
        return v, (float(last[0]) if last else np.nan)
    lp = o["lap_times"].map(laps)
    o["ok_n_f"] = lp.map(lambda t: len(t[0]))
    o["ok_l1"] = lp.map(lambda t: t[1])
    def acc(s):
        segs = [float(x) for x in re.findall(r"\((\d+\.\d)\)", s)]
        cums = [float(x) for x in re.findall(r"(\d+\.\d)\(", s)]
        if len(cums) < 2:
            return np.nan
        # 最後から 2 番目の区間の 1F あたり（3F−1F を 2 で割る）と最後の 1F を比べる
        # cums の最後 = 終い 1F、その前 = 3F。3F−1F を 2 で割った 1F あたりより終い 1F が速ければ加速
        return float(cums[-1] < (cums[-2] - cums[-1]) / 2)
    o["ok_acc"] = o["lap_times"].map(acc)
    o["ok_slope"] = o["ok_course"].str.endswith("坂")
    o["ok_wood"] = o["ok_course"].str.endswith("Ｗ") | (~o["ok_slope"] & (o["ok_n_f"] >= 4))
    d = d.merge(o[key + ["ok_course", "ok_rider", "ok_eval", "ok_int", "ok_tanpyo", "ok_cond", "ok_td", "ok_wd", "ok_n_f", "ok_l1", "ok_acc", "ok_slope", "ok_wood"]],
                on=key, how="left")
    d["ok_days"] = (d["dt"] - d["ok_td"]).dt.days
    # 厩舎コメント（2020〜YEAR_END）
    c = read_years("race_trainer_comment_flat", range(2020, YEAR_END + 1), ["race_id", "horse_id", "comment", "questioner"])
    c["race_id"] = c["race_id"].astype(str)
    c["horse_id"] = c["horse_id"].astype(str)
    c = c.drop_duplicates(key)
    t = c["comment"].fillna("").astype(str)
    c["tc_has"] = (t.str.len() > 0).astype(float)
    c["tc_pos"] = t.str.contains("状態がいい|状態は良|状態は上|力を出せ|仕上がっ|好調|絶好|いい状態|万全|デキは良|デキはいい|上向|順調", regex=True).astype(float)
    c["tc_neg"] = t.str.contains("太め|重め|物足り|途上|今ひとつ|いまひとつ|疲れ|硬さ|もう一息|まだ", regex=True).astype(float)
    c["tc_next"] = t.str.contains("使いつつ|叩いて|先を見据え|ひと叩き|一叩き|次につなが|次に繋が|叩き台", regex=True).astype(float)
    c["tc_sp"] = c["questioner"].fillna("").astype(str)
    d = d.merge(c[key + ["tc_has", "tc_pos", "tc_neg", "tc_next", "tc_sp"]], on=key, how="left")
    # スピード指数（出走前の値）
    ix = read_years("race_index_flat", range(2019, YEAR_END + 1), ["race_id", "horse_id", "speed_max", "speed_avg", "speed_recent", "speed_course", "speed_distance"])
    ix["race_id"] = ix["race_id"].astype(str)
    ix["horse_id"] = ix["horse_id"].astype(str)
    ix = ix.drop_duplicates(key)
    for cc in ["speed_max", "speed_avg", "speed_recent", "speed_course", "speed_distance"]:
        ix[cc] = _f(ix[cc])
    d = d.merge(ix, on=key, how="left")
    # めぐ指数（その走の成績の指数）→ 履歴
    mg = read_years("megu_index_flat", range(2020, YEAR_END + 1), ["race_id", "horse_id", "megu_index_v3a"])
    mg["race_id"] = mg["race_id"].astype(str)
    mg["horse_id"] = mg["horse_id"].astype(str)
    d = d.merge(mg.drop_duplicates(key), on=key, how="left")
    d["megu"] = _f(d["megu_index_v3a"])
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    gm = d.groupby("horse_id")["megu"]
    d["mg_p1"] = gm.shift(1)
    d["mg_p2"] = gm.shift(2)
    d["mg_p3"] = gm.shift(3)
    d["mg_max5"] = gshift_roll(d["megu"], d["horse_id"], 5, 2, "max")
    d["mg_sd5"] = gshift_roll(d["megu"], d["horse_id"], 5, 3, "std")
    d["mg_mean3"] = gshift_roll(d["megu"], d["horse_id"], 3, 2, "mean")
    d["mg_maxall"] = gshift_cum(d["megu"], d["horse_id"], "cummax")
    # パドック・バロメーター
    pdk = read_years("race_paddock_flat", range(2020, YEAR_END + 1), ["race_id", "horse_id", "paddock_rank"])
    pdk["race_id"] = pdk["race_id"].astype(str)
    pdk["horse_id"] = pdk["horse_id"].astype(str)
    pdk["pd_rank"] = _f(pdk["paddock_rank"])
    d = d.merge(pdk.drop_duplicates(key)[key + ["pd_rank"]], on=key, how="left")
    bm = read_years("race_barometer_flat", range(2023, YEAR_END + 1), ["race_id", "horse_id", "index_total", "index_start", "index_chase", "index_closing"])
    bm["race_id"] = bm["race_id"].astype(str)
    bm["horse_id"] = bm["horse_id"].astype(str)
    for cc in ["index_total", "index_start", "index_chase", "index_closing"]:
        bm[cc] = _f(bm[cc])
    d = d.merge(bm.drop_duplicates(key), on=key, how="left")
    # 出馬表の過去 5 走（海外・地方の走歴）
    pp = read_years("race_shutuba_past_flat", range(2020, YEAR_END + 1), ["race_id", "horse_id", "past_races"])
    pp["race_id"] = pp["race_id"].astype(str)
    pp["horse_id"] = pp["horse_id"].astype(str)
    pp = pp.drop_duplicates(key)
    def last_region(s):
        try:
            arr = json.loads(s) if isinstance(s, str) else []
        except Exception:
            arr = []
        if not arr:
            return None, None
        raw = str(arr[0].get("raw", ""))
        reg = "海外" if any(v in raw for v in OVERSEAS) else ("地方" if any((" " + v + " ") in (" " + raw.replace("\xa0", " ") + " ") for v in NAR) else "中央")
        n_os = sum(1 for e in arr if any(v in str(e.get("raw", "")) for v in OVERSEAS))
        return reg, n_os
    lr = pp["past_races"].map(last_region)
    pp["pp_last_region"] = lr.map(lambda t: t[0])
    pp["pp_n_overseas"] = lr.map(lambda t: t[1])
    d = d.merge(pp[key + ["pp_last_region", "pp_n_overseas"]], on=key, how="left")
    # クッション値・含水率（場・日）
    cu = []
    for y in range(2018, YEAR_END + 1):
        f = ROOT / f"data/page_reference/cushion/cushion_{y}.json"
        if f.exists():
            cu += json.load(open(f, encoding="utf-8"))
    cu = pd.DataFrame(cu)
    if len(cu):
        cu["vc"] = cu["venue_code"].astype(str).str.zfill(2)
        cu["date"] = cu["date"].astype(str)
        cu = cu.groupby(["vc", "date"]).agg(cush=("cushion_value", "mean"), tmoist=("turf_moisture_goal", "mean"), dmoist=("dirt_moisture_goal", "mean")).reset_index()
        d = d.merge(cu, on=["vc", "date"], how="left")
    # 気象（場・日・発走時刻）
    w = pd.read_parquet(TABLES / "jma_weather_hourly_flat.parquet", columns=["venue_code", "date", "hour", "precip_mm", "temp_c", "wind_speed_ms"])
    w = w.loc[:, ~w.columns.duplicated()]
    w["vc"] = w["venue_code"].astype(str).str.zfill(2)
    w["date"] = w["date"].astype(str)
    w["hour"] = _f(w["hour"])
    hh = d["stime"].dt.hour
    d["hour"] = hh
    wt = w.groupby(["vc", "date", "hour"]).agg(temp=("temp_c", "mean"), wind=("wind_speed_ms", "mean")).reset_index()
    d = d.merge(wt, on=["vc", "date", "hour"], how="left")
    pr = w[w["hour"] <= 16].groupby(["vc", "date", "hour"])["precip_mm"].sum().reset_index().sort_values(["vc", "date", "hour"])
    pr["rain_cum"] = pr.groupby(["vc", "date"])["precip_mm"].cumsum()
    d = d.merge(pr[["vc", "date", "hour", "rain_cum"]], on=["vc", "date", "hour"], how="left")
    dmax = w.groupby(["vc", "date"])["temp_c"].max().rename("tmax").reset_index()
    d = d.merge(dmax, on=["vc", "date"], how="left")
    return d


def add_race_context(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("race_id")
    # 逃げ候補（as-of の逃げ率 0.3 以上、出走 3 回以上）
    d["nige_cand"] = ((d["nige_rate"] >= 0.3) & (d["starts_prev"] >= 3)).astype(float)
    d["n_nige_cand"] = g["nige_cand"].transform("sum")
    # 同じ厩舎・馬主の出走数と人気の順
    d["n_same_tr"] = d.groupby(["race_id", "trainer_id"])["horse_id"].transform("size")
    d["tr_pop_rank"] = d.groupby(["race_id", "trainer_id"])["pop"].rank(method="first")
    d["n_same_own"] = d.groupby(["race_id", "owner"])["horse_id"].transform("size")
    d["own_pop_rank"] = d.groupby(["race_id", "owner"])["pop"].rank(method="first")
    # 前走騎手が同じレースの別の馬に乗っているか
    jr = d[["race_id", "jockey_id", "horse_id"]].rename(columns={"jockey_id": "p_jockey_id", "horse_id": "other_h"})
    m = d[["race_id", "horse_id", "p_jockey_id"]].merge(jr, on=["race_id", "p_jockey_id"], how="left")
    m = m[m["other_h"].notna() & (m["other_h"] != m["horse_id"])].drop_duplicates(["race_id", "horse_id"])
    d["pj_rides_other"] = d.set_index(["race_id", "horse_id"]).index.isin(m.set_index(["race_id", "horse_id"]).index)
    # 騎手が前走で乗った馬が同じレースに 2 頭以上 → 騎手が選んだ馬
    d["p_jockey_id_now"] = d["jockey_id"]
    pairs = d[["race_id", "horse_id", "p_jockey_id", "jockey_id"]]
    cnt = pairs.groupby(["race_id", "p_jockey_id"])["horse_id"].transform("size")
    d["chosen_by_jockey"] = (cnt >= 2) & (d["jockey_id"] == d["p_jockey_id"])
    d["choice_set"] = cnt >= 2
    # レース内の順位（指数・持ち時計）
    d["si_rank_max"] = g["speed_max"].rank(ascending=False, method="min")
    d["si_rank_recent"] = g["speed_recent"].rank(ascending=False, method="min")
    d["mg_rank_p1"] = g["mg_p1"].rank(ascending=False, method="min")
    d["mg_rank_max5"] = g["mg_max5"].rank(ascending=False, method="min")
    d["mg_rank_mean3"] = g["mg_mean3"].rank(ascending=False, method="min")
    for c in ["index_start", "index_chase", "index_closing", "index_total"]:
        d[c + "_rank"] = g[c].rank(ascending=False, method="min")
    d["wt_rank"] = g["wt"].rank(ascending=False, method="min")
    d["jw_mean"] = g["jw"].transform("mean")
    # メンバーの格（出走馬の as-of の重賞・OP の出走数の合計）
    d["member_op"] = g["op_n_prev"].transform("sum")
    # 隣の枠に逃げ候補
    d = d.sort_values(["race_id", "hn"]).reset_index(drop=True)
    nc = d["nige_cand"]
    same = d["race_id"]
    left = nc.shift(1).where(same.shift(1) == same, 0)
    right = nc.shift(-1).where(same.shift(-1) == same, 0)
    d["nb_nige"] = (left.fillna(0) + right.fillna(0)) > 0
    # 内/外に同型
    tmp = d[d["nige_cand"] == 1][["race_id", "hn"]]
    out = tmp.merge(tmp, on="race_id", suffixes=("", "_o"))
    out = out[out["hn_o"] > out["hn"]].groupby(["race_id", "hn"]).size().rename("n_out")
    inn = tmp.merge(tmp, on="race_id", suffixes=("", "_o"))
    inn = inn[inn["hn_o"] < inn["hn"]].groupby(["race_id", "hn"]).size().rename("n_in")
    d = d.merge(out.reset_index(), on=["race_id", "hn"], how="left").merge(inn.reset_index(), on=["race_id", "hn"], how="left")
    d["n_out"] = d["n_out"].fillna(0)
    d["n_in"] = d["n_in"].fillna(0)
    # 開催の何日目（同じ場・同じ年の開催日の通し番号を、間が 14 日以上空いたら区切る）
    days = d[["vc", "dt"]].drop_duplicates().sort_values(["vc", "dt"])
    gap = days.groupby("vc")["dt"].diff().dt.days
    days["meet"] = (gap.isna() | (gap > 14)).astype(int).groupby(days["vc"]).cumsum()
    days["meet_day"] = days.groupby(["vc", "meet"]).cumcount() + 1
    days["meet_len"] = days.groupby(["vc", "meet"])["dt"].transform("size")
    d = d.merge(days, on=["vc", "dt"], how="left")
    # 同じ日・同じ面の『そのレースより前』のレースの勝ち馬・3 着内の 4 角位置と馬番の偏り
    r = d[d["fin"] <= 3][["race_id", "vc", "date", "surf", "stime", "pos4r", "hn", "field"]].copy()
    r["hnr"] = r["hn"] / r["field"]
    ra = r.groupby(["race_id", "vc", "date", "surf", "stime"]).agg(b_pos=("pos4r", "mean"), b_hn=("hnr", "mean")).reset_index().sort_values(["vc", "date", "surf", "stime"])
    gg = ra.groupby(["vc", "date", "surf"])
    ra["k"] = gg.cumcount()
    for c, o in (("b_pos", "bias_pos"), ("b_hn", "bias_hn")):
        ra[o] = (gg[c].cumsum() - ra[c]) / ra["k"].replace(0, np.nan)
    d = d.merge(ra[["race_id", "k", "bias_pos", "bias_hn"]].rename(columns={"k": "n_before"}), on="race_id", how="left")
    # 同じ日・同じ面の前のレースの勝ちタイムの速さ（クラス・コース基準比、2016〜2019 で基準を作る）
    win = d[d["fin"] == 1][["race_id", "course", "cls", "track", "tsec", "vc", "date", "surf", "stime", "year"]].drop_duplicates("race_id")
    base = win[win["year"] <= 2019].groupby(["course", "cls"])["tsec"].median().rename("par")
    win = win.merge(base.reset_index(), on=["course", "cls"], how="left")
    win["fast"] = win["tsec"] - win["par"]
    win = win.sort_values(["vc", "date", "surf", "stime"])
    gw = win.groupby(["vc", "date", "surf"])
    cnt = gw["fast"].cumcount()
    win["day_fast"] = (gw["fast"].cumsum() - win["fast"]) / cnt.replace(0, np.nan)
    d = d.merge(win[["race_id", "fast", "day_fast"]].rename(columns={"fast": "race_fast"}), on="race_id", how="left")
    # 芝の連続使用日数の代わり: 同じ場の同じ年の芝のレース日の通し番号
    return d


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id")
    for c in ["race_fast", "fast"]:
        if c in d:
            d["p_" + c] = g[c].shift(1)
    d["p_member_op"] = g["member_op"].shift(1)
    # 前走のメンバーのその後（前走の同走馬の『前走より後』の勝ち上がり）はリークを避けるため使わず、代わりに前走のメンバーの格を使う
    # 前走の同じレースの勝ち馬との対戦（再戦）
    # 遠征: 美浦所属が京都・阪神・中京・小倉、栗東所属が東京・中山・新潟・福島（北海道は両方とも遠征）
    east = {"04", "05", "06", "03"}
    west = {"07", "08", "09", "10"}
    d["away"] = np.where(d["vc"].isin(["01", "02"]), True,
                         np.where(d["tozai"] == "美浦", d["vc"].isin(west), np.where(d["tozai"] == "栗東", d["vc"].isin(east), np.nan)))
    d["away"] = d["away"].astype("float")
    a = d["away"].fillna(0)
    d["away_n_prev"] = a.groupby(d["horse_id"]).cumsum() - a
    # 3 歳の古馬混合の初戦・初の 1 番人気など
    d["first_vs_old"] = (d["age"] == 3) & d["mixed"] & (d["vs_old_prev"] == 0)
    # 前走の勝ちタイムのクラス基準比
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / "_stage_people.parquet"
    if not (ck.exists() and os.environ.get("FOLKLORE_RESUME", "1") == "1"):
        d = load_results()
        print("results", len(d), flush=True)
        d = parse_basic(d)
        d = add_history(d)
        print("history done", flush=True)
    else:
        d = None
    ck = OUT / "_stage_people.parquet"
    if d is None:
        d = pd.read_parquet(ck)
        print("resume from", ck, flush=True)
    else:
        d = add_people(d)
        d.to_parquet(ck, index=False)
    print("people done", flush=True)
    d = add_external(d)
    print("external done")
    d = add_race_context(d)
    d = add_derived(d)
    ran = d["fin"].notna() | (KEEP_DNF & (d["odds"] > 0))
    keep = d[(d["surf"].isin(["芝", "ダ"])) & ran & (d["odds"] > 0) & (d["field"] >= 5)].copy()
    if KEEP_DNF:   # 中止の馬は着順の生の値が -1 のときに y3・y1 を作っているので、着外として作り直す
        dnf = keep["fin"].isna()
        keep.loc[dnf, "y3"] = 0.0
        keep.loc[dnf, "y1"] = 0.0
    drop = [c for c in keep.columns if keep[c].dtype == object and c in ("pace", "lap_times", "passing_order")]
    keep = keep.drop(columns=drop)
    for c in keep.columns:
        if keep[c].dtype == object:
            keep[c] = keep[c].astype("string")
    keep.to_parquet(OUT / "master.parquet", index=False)
    print("master", keep.shape)


if __name__ == "__main__":
    main()
