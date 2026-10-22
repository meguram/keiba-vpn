"""tp-72 データ: 『買う要素』の網羅探索用の 1 出走 = 1 行の表（tp-59 の master に、季節・性別・クッション値・脚質・履歴由来の列を足す）。

python -m src.scripts.data.tp72_data build     # ホストで実行。出力 <TP72_OUT>/mining.parquet
python -m src.scripts.data.tp72_data check

履歴由来の列は、同じ馬の直前までの出走（2020 年以降の全 JRA 出走）から作る（as-of。同日の出走は使わない）。
騎手・調教師・種牡馬の集計も『その日より前』の累積だけ（merge_asof の allow_exact_matches=False）。
市場オッズは分析（市場に織り込まれているか）にだけ使い、特徴量にはしない。
クッション値・含水率の区分は 2020〜2021 年の三分位（探索期間 2022〜2024 を見る前に固定）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TP72_REPO", Path(__file__).resolve().parents[3]))
OUT = Path(os.environ.get("TP72_OUT", "/home/jovyan/work/tp72_out"))
MASTER = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out")) / "master.parquet"
TABLES = ROOT / "data/page_reference/tables"
CUSHION = ROOT / "data/page_reference/cushion"
YEARS = list(range(2020, 2027))
LOCAL_VENUES = {"01", "02", "03", "04", "10"}      # 札幌・函館・福島・新潟・小倉
SEASON4 = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def asof_merge(left: pd.DataFrame, src: pd.DataFrame, keys: list[str], cols: list[str], prefix: str) -> pd.DataFrame:
    """src（key, date, cols の日別合計）を累積し、left の各行に『その日より前』の累積値を付ける。"""
    g = src.groupby(keys + ["date"], sort=False)[cols].sum().reset_index().sort_values(["date"] + keys, kind="stable")
    cs = g.groupby(keys, sort=False)[cols].cumsum()
    g = pd.concat([g[keys + ["date"]], cs.add_prefix(prefix)], axis=1)
    l = left[keys + ["date"]].copy()
    l["_i"] = np.arange(len(l))
    l = l.sort_values("date", kind="stable")
    mg = pd.merge_asof(l, g.sort_values("date", kind="stable"), on="date", by=keys, allow_exact_matches=False, direction="backward")
    mg = mg.sort_values("_i")
    for c in [prefix + c for c in cols]:
        left[c] = mg[c].to_numpy()
    return left


def load_extra() -> pd.DataFrame:
    parts = []
    for y in YEARS:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["race_id", "horse_number", "bracket_number", "passing_order", "last_3f", "time_sec"]))
    d = pd.concat(parts, ignore_index=True)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce").astype("Int64")
    d["bracket_number"] = pd.to_numeric(d["bracket_number"], errors="coerce")
    d["last_3f"] = pd.to_numeric(d["last_3f"], errors="coerce")
    return d.drop_duplicates(["race_id", "horse_number"])


def load_oikiri() -> pd.DataFrame:
    """最終追い切りの評価（A〜D）と厩舎コメントの前向き語（出走前に出る情報）。"""
    parts = []
    for y in YEARS:
        f = TABLES / str(y) / "race_oikiri_flat.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["race_id", "horse_number", "evaluation"])
            parts.append(d)
    o = pd.concat(parts, ignore_index=True)
    o["race_id"] = o["race_id"].astype(str)
    o["horse_number"] = pd.to_numeric(o["horse_number"], errors="coerce").astype("Int64")
    o["ev_score"] = o["evaluation"].map({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    o = o.drop_duplicates(["race_id", "horse_number"])[["race_id", "horse_number", "ev_score"]]
    cp = []
    for y in YEARS:
        f = TABLES / str(y) / "race_trainer_comment_flat.parquet"
        if f.exists():
            cp.append(pd.read_parquet(f, columns=["race_id", "horse_number", "comment"]))
    if cp:
        c = pd.concat(cp, ignore_index=True)
        c["race_id"] = c["race_id"].astype(str)
        c["horse_number"] = pd.to_numeric(c["horse_number"], errors="coerce").astype("Int64")
        c = c.dropna(subset=["horse_number"]).drop_duplicates(["race_id", "horse_number"])
        t = c["comment"].fillna("").astype(str)
        c["comment_has"] = (t.str.len() > 0).astype(int)
        c["comment_pos"] = t.str.contains("良化|上向|状態が上がっ|順調|仕上がっ|動きが良|動きがいい|反応が良|気配が良|成長", regex=True).astype(int)
        c["comment_neg"] = t.str.contains("太め|太い|重め|まだ|物足り|途上|良化途上|今ひとつ|いまひとつ|疲れ|硬い", regex=True).astype(int)
        o = o.merge(c[["race_id", "horse_number", "comment_has", "comment_pos", "comment_neg"]], on=["race_id", "horse_number"], how="left")
    return o


def load_blinker() -> pd.DataFrame:
    """その走のブリンカー着用（SmartRC の馬ごとの行。装着は出走前に決まる事実）。2026 年は表が無い。"""
    parts = []
    for y in YEARS:
        f = TABLES / str(y) / "smartrc_race_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["race_id", "hcode", "blinker"]))
    b = pd.concat(parts, ignore_index=True)
    b["race_id"] = b["race_id"].astype(str)
    b["horse_id"] = b["hcode"].astype(str)
    b["blk"] = pd.to_numeric(b["blinker"], errors="coerce").clip(0, 1)
    return b.drop_duplicates(["race_id", "horse_id"])[["race_id", "horse_id", "blk"]]


def load_sire() -> pd.DataFrame | None:
    """父・母父（血統表の第 1 世代・父側 / 第 2 世代・母の父）。学習表の sire 列は空なので血統表（ped_tbl）から horse_id で引く。"""
    try:
        import pyarrow.dataset as pads

        base = ROOT / "data/local/features/horse/ped_tbl"
        ds = pads.dataset(str(base), format="parquet")
        cols = ["subject_horse_id", "ancestor_horse_id", "ancestor_name"]
        t1 = ds.to_table(columns=cols, filter=(pads.field("generation") == 1) & (pads.field("path_fm") == "F")).to_pandas()
        t2 = ds.to_table(columns=cols, filter=(pads.field("generation") == 2) & (pads.field("path_fm") == "MF")).to_pandas()
        t1 = t1.rename(columns={"subject_horse_id": "horse_id", "ancestor_horse_id": "sire", "ancestor_name": "sire_name"})
        t2 = t2.rename(columns={"subject_horse_id": "horse_id", "ancestor_horse_id": "dam_sire", "ancestor_name": "dam_sire_name"})
        for t in (t1, t2):
            t["horse_id"] = t["horse_id"].astype(str)
        return t1.drop_duplicates("horse_id").merge(t2.drop_duplicates("horse_id"), on="horse_id", how="outer")
    except Exception as exc:  # noqa: BLE001
        print("sire の読み込みに失敗:", exc)
        return None


def parse_pass(s) -> tuple[float, float]:
    try:
        xs = [int(x) for x in str(s).replace("(", "").replace(")", "").replace("*", "").split("-") if x.strip().isdigit()]
        return (float(xs[0]), float(xs[-1])) if xs else (np.nan, np.nan)
    except Exception:  # noqa: BLE001
        return (np.nan, np.nan)


def load_cushion() -> pd.DataFrame:
    rows = []
    for y in YEARS:
        f = CUSHION / f"cushion_{y}.json"
        if f.exists():
            rows += json.loads(f.read_text(encoding="utf-8"))
    c = pd.DataFrame(rows)
    c["venue_code"] = c["venue_code"].astype(str).str.zfill(2)
    c["date"] = pd.to_datetime(c["date"])
    keep = ["venue_code", "date", "cushion_value", "turf_moisture_goal", "turf_moisture_4corner", "dirt_moisture_goal", "dirt_moisture_4corner"]
    c = c[keep].copy()
    for k in keep[2:]:
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c = c.sort_values(["venue_code", "date"]).groupby(["venue_code", "date"], as_index=False).agg("first")
    return c


def tercile_cut(x: pd.Series, ref: pd.Series) -> tuple[float, float]:
    q = ref.dropna().quantile([1 / 3, 2 / 3]).to_numpy()
    return float(q[0]), float(q[1])


def build() -> pd.DataFrame:
    m = pd.read_parquet(MASTER)
    m["race_id"] = m["race_id"].astype(str)
    m["horse_id"] = m["horse_id"].astype(str)
    m["date"] = pd.to_datetime(m["date"])
    m["horse_number"] = pd.to_numeric(m["horse_number"], errors="coerce").astype("Int64")
    m = m[m["surface"].isin(["芝", "ダート"])].copy()
    m = m.merge(load_extra(), on=["race_id", "horse_number"], how="left")
    m = m.merge(load_oikiri(), on=["race_id", "horse_number"], how="left")
    m = m.merge(load_blinker(), on=["race_id", "horse_id"], how="left")
    for c in ("comment_has", "comment_pos", "comment_neg"):
        m[c] = m[c].fillna(0).astype(int) if c in m.columns else 0

    # --- 基本属性 ---
    m["sex"] = m["sex_age"].astype(str).str[0]
    m["age"] = pd.to_numeric(m["sex_age"].astype(str).str[1:], errors="coerce")
    m["month"] = m["date"].dt.month
    m["season"] = m["month"].map(SEASON4)
    m["is_summer"] = m["month"].isin([7, 8]).astype(int)
    m["is_local"] = m["venue_code"].isin(LOCAL_VENUES).astype(int)
    m["is_turf"] = (m["surface"] == "芝").astype(int)
    m["age_grp"] = np.where(m["age"] <= 3, "a3", "a4p")
    m["kai"] = pd.to_numeric(m["race_id"].str[6:8], errors="coerce")
    m["day_no"] = pd.to_numeric(m["race_id"].str[8:10], errors="coerce")
    m["day_max"] = m.groupby(["year", "venue_code", "kai"])["day_no"].transform("max")
    m["inner"] = (m["bracket_number"] <= 2).astype(int)
    m["outer"] = (m["bracket_number"] >= 7).astype(int)
    m["fs"] = m["field_size"]

    # --- G1 開催日の前後 ---
    g1days = pd.Series(sorted(m.loc[m["grade"].astype(str).str.contains("G1|GI|Ｇ１", regex=True), "date"].unique()))
    dts = m["date"].to_numpy()
    if len(g1days):
        g = g1days.to_numpy()
        idx = np.searchsorted(g, dts)                      # 最初の g >= date
        nxt = np.where(idx < len(g), g[np.clip(idx, 0, len(g) - 1)], np.datetime64("NaT"))
        prv_idx = np.searchsorted(g, dts, side="left") - 1
        prv = np.where(prv_idx >= 0, g[np.clip(prv_idx, 0, len(g) - 1)], np.datetime64("NaT"))
        dn = (pd.to_datetime(nxt) - m["date"]).dt.days
        dp = (m["date"] - pd.to_datetime(prv)).dt.days
        m["is_g1_day"] = (dn == 0).astype(int)
        m["d_before_g1"] = ((dn >= 1) & (dn <= 7)).astype(int)
        m["d_after_g1"] = ((dp >= 1) & (dp <= 7)).astype(int)
    else:
        m["is_g1_day"] = m["d_before_g1"] = m["d_after_g1"] = 0

    # --- クッション値・含水率 ---
    c = load_cushion()
    m = m.merge(c, on=["venue_code", "date"], how="left")
    ref = m[(m["year"] <= 2021)]
    t_lo, t_hi = tercile_cut(m["cushion_value"], ref.loc[ref["surface"] == "芝", "cushion_value"])
    d_lo, d_hi = tercile_cut(m["dirt_moisture_goal"], ref.loc[ref["surface"] == "ダート", "dirt_moisture_goal"])
    tm_lo, tm_hi = tercile_cut(m["turf_moisture_goal"], ref.loc[ref["surface"] == "芝", "turf_moisture_goal"])
    cuts = {"cushion_turf": [t_lo, t_hi], "dirt_moist": [d_lo, d_hi], "turf_moist": [tm_lo, tm_hi]}
    m["cush_bin"] = np.where(m["surface"] != "芝", np.nan, np.where(m["cushion_value"].isna(), np.nan, np.where(m["cushion_value"] >= t_hi, 2, np.where(m["cushion_value"] <= t_lo, 0, 1))))
    m["dmoist_bin"] = np.where(m["surface"] != "ダート", np.nan, np.where(m["dirt_moisture_goal"].isna(), np.nan, np.where(m["dirt_moisture_goal"] >= d_hi, 2, np.where(m["dirt_moisture_goal"] <= d_lo, 0, 1))))
    m["tmoist_bin"] = np.where(m["surface"] != "芝", np.nan, np.where(m["turf_moisture_goal"].isna(), np.nan, np.where(m["turf_moisture_goal"] >= tm_hi, 2, np.where(m["turf_moisture_goal"] <= tm_lo, 0, 1))))
    # 同じ競馬場の前の開催日からの変化
    cd = c.sort_values(["venue_code", "date"]).copy()
    cd["cush_prevday"] = cd.groupby("venue_code")["cushion_value"].shift(1)
    cd["gap_days"] = (cd["date"] - cd.groupby("venue_code")["date"].shift(1)).dt.days
    cd["cush_delta_day"] = np.where(cd["gap_days"] <= 8, cd["cushion_value"] - cd["cush_prevday"], np.nan)
    m = m.merge(cd[["venue_code", "date", "cush_delta_day"]], on=["venue_code", "date"], how="left")

    # --- 馬ごとの履歴（as-of） ---
    m = m.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    g = m.groupby("horse_id", sort=False)
    p = m["passing_order"].map(parse_pass)
    m["pos1"] = [a for a, _ in p]
    m["posL"] = [b for _, b in p]
    m["pos1_frac"] = (m["pos1"] - 1) / (m["fs"] - 1)
    m["posL_frac"] = (m["posL"] - 1) / (m["fs"] - 1)
    m["fin_pct"] = (m["finish_position"] - 1) / (m["fs"] - 1)
    m["agari_rank"] = m.groupby("race_id")["last_3f"].rank(method="min")
    g = m.groupby("horse_id", sort=False)
    m["style5"] = g["pos1_frac"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    m["prev_agari_rank"] = g["agari_rank"].shift(1)
    m["prev2_fin"] = g["finish_position"].shift(2)
    m["prev_pop"] = g["popularity"].shift(1)
    m["prev_surface"] = g["surface"].shift(1)
    m["prev_dist"] = g["distance"].shift(1)
    m["prev_jw"] = g["jockey_weight"].shift(1)
    m["prev_cush"] = g["cushion_value"].shift(1)
    m["prev_fs"] = g["fs"].shift(1)
    m["prev_posL_frac"] = g["posL_frac"].shift(1)
    m["prev_pos1"] = g["pos1"].shift(1)
    m["prev_posL"] = g["posL"].shift(1)
    m["pos1_chg"] = m["pos1_frac"] - m["style5"]          # 今回の初角位置 − 普段（負なら普段より前。結果由来なので機構の確認にだけ使う）
    m["prev_blk"] = g["blk"].shift(1)
    m["n_blk_prior"] = g["blk"].transform(lambda s: s.fillna(0).cumsum().shift(1).fillna(0))
    m["blk_state"] = np.select(
        [m["blk"].isna() | m["prev_blk"].isna(), (m["blk"] == 1) & (m["prev_blk"] == 1), (m["blk"] == 1) & (m["n_blk_prior"] == 0) & (m["run_no"] >= 3),
         (m["blk"] == 0) & (m["prev_blk"] == 1), (m["blk"] == 1)], ["na", "cont", "first", "off", "again"], "none")
    m["style4"] = np.select([m["style5"].isna(), m["style5"] <= 0.10, m["style5"] <= 0.30, m["style5"] <= 0.60], ["na", "nige", "senko", "sashi"], "oikomi")
    m["prev_ev"] = g["ev_score"].shift(1)
    m["ev_delta"] = m["ev_score"] - m["prev_ev"]
    m["ev_mean_prior"] = g["ev_score"].transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    m["ev_dev"] = m["ev_score"] - m["ev_mean_prior"]
    m["dist_chg"] = pd.to_numeric(m["distance"], errors="coerce") - pd.to_numeric(m["prev_dist"], errors="coerce")
    m["jw_chg"] = m["jockey_weight"] - m["prev_jw"]
    m["first_surface"] = (m.groupby(["horse_id", "surface"], sort=False).cumcount() == 0) & (m["run_no"] > 0)
    m["first_surface"] = m["first_surface"].astype(int)
    m["wt_mean_prior"] = g["weight"].transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    m["wt_dev"] = m["weight"] - m["wt_mean_prior"]
    # レースの隊列：逃げ・先行型（style5<=0.25）と差し・追込型（>=0.6）の頭数
    known = m["style5"].notna()
    nk = known.groupby(m["race_id"]).transform("sum")
    m["front"] = (m["style5"] <= 0.25).astype(float).where(known)
    m["closer"] = (m["style5"] >= 0.60).astype(float).where(known)
    m["n_front"] = m.groupby("race_id")["front"].transform("sum").where(nk >= 0.7 * m["fs"])
    m["n_closer"] = m.groupby("race_id")["closer"].transform("sum").where(nk >= 0.7 * m["fs"])
    m["style_grp"] = np.where(m["style5"].isna(), "na", np.where(m["style5"] <= 0.25, "front", np.where(m["style5"] >= 0.60, "closer", "mid")))

    # 馬ごとの季節・クッション値への as-of 親和度
    m["top3f"] = m["top3"].astype(float)
    m["win_f"] = m["win"].astype(float)
    m["one"] = 1.0
    h = m[["horse_id", "date", "one", "top3f"]].copy()
    m = asof_merge(m, h, ["horse_id"], ["one", "top3f"], "h_")
    hs = m[["horse_id", "season", "date", "one", "top3f"]].copy()
    m = asof_merge(m, hs, ["horse_id", "season"], ["one", "top3f"], "hs_")
    m["seas_aff"] = np.where((m["hs_one"] >= 2) & (m["h_one"] >= 5), m["hs_top3f"] / m["hs_one"] - m["h_top3f"] / m["h_one"], np.nan)
    # クッション値（芝）の硬い／柔らかい日での平均着順パーセンタイル
    tr = m[(m["surface"] == "芝") & m["cush_bin"].notna()].copy()
    for b, nm in ((2, "hard"), (0, "soft")):
        s = tr[tr["cush_bin"] == b][["horse_id", "date", "fin_pct"]].copy()
        s["n"] = 1.0
        s["fp"] = s["fin_pct"].fillna(0.5)
        m = asof_merge(m, s[["horse_id", "date", "n", "fp"]].rename(columns={"n": f"n_{nm}", "fp": f"fp_{nm}"}), ["horse_id"], [f"n_{nm}", f"fp_{nm}"], "c_")
    m["pref_hard"] = np.where((m["c_n_hard"] >= 2) & (m["c_n_soft"] >= 2), m["c_fp_soft"] / m["c_n_soft"] - m["c_fp_hard"] / m["c_n_hard"], np.nan)   # 正なら硬い日のほうが好走

    # 騎手・調教師の競馬場別の as-of 親和度
    for who in ("trainer_id", "jockey_id"):
        sub = m[[who, "venue_code", "date", "one", "top3f"]].copy()
        m = asof_merge(m, sub, [who, "venue_code"], ["one", "top3f"], f"{who[0]}v_")
        sub2 = m[[who, "date", "one", "top3f"]].copy()
        m = asof_merge(m, sub2, [who], ["one", "top3f"], f"{who[0]}a_")
        m[f"{who[0]}_local_aff"] = np.where((m[f"{who[0]}v_one"] >= 40) & (m[f"{who[0]}a_one"] >= 200), m[f"{who[0]}v_top3f"] / m[f"{who[0]}v_one"] - m[f"{who[0]}a_top3f"] / m[f"{who[0]}a_one"], np.nan)

    # 種牡馬 × クッション値（芝）／季節
    sire = load_sire()
    if sire is not None:
        m = m.merge(sire, on="horse_id", how="left")
    ref = m[m["year"] <= 2021]
    if "sire" in m.columns:
        top_s = ref["sire"].value_counts().head(15).index.tolist()
        top_ds = ref["dam_sire"].value_counts().head(10).index.tolist()
        m["sire_rank"] = m["sire"].map({k: i for i, k in enumerate(top_s)})
        m["damsire_rank"] = m["dam_sire"].map({k: i for i, k in enumerate(top_ds)})
        nm = m.dropna(subset=["sire"]).drop_duplicates("sire").set_index("sire")["sire_name"]
        nd = m.dropna(subset=["dam_sire"]).drop_duplicates("dam_sire").set_index("dam_sire")["dam_sire_name"]
        meta_names = {"sire_top15": [str(nm.get(k, k)) for k in top_s], "damsire_top10": [str(nd.get(k, k)) for k in top_ds]}
    else:
        meta_names = {}
    top_j = ref.groupby("jockey_id")["win"].sum().sort_values(ascending=False).head(12).index.tolist()
    jm = {k: i for i, k in enumerate(top_j)}
    m["jockey_rank"] = m["jockey_id"].astype(str).map({str(k): i for k, i in jm.items()})
    m["prev_jockey_rank"] = m["prev_jockey_id"].astype(str).map({str(k): i for k, i in jm.items()})
    meta_names["jockey_top12_ids"] = [str(k) for k in top_j]
    chg = m["chg_jockey"] == 1
    m["jchg_cls"] = np.select([chg & m["prev_jockey_rank"].isna() & m["jockey_rank"].notna(), chg & m["prev_jockey_rank"].notna() & m["jockey_rank"].isna(),
                               chg & m["prev_jockey_rank"].notna() & m["jockey_rank"].notna(), chg], ["up", "down", "top_to_top", "non_to_non"], "none")
    if "sire" in m.columns and m["sire"].notna().mean() > 0.5:
        ss = m[m["sire"].notna()][["sire", "date", "one", "top3f"]].copy()
        m = asof_merge(m, ss, ["sire"], ["one", "top3f"], "sa_")
        for b, nm in ((2, "hard"), (0, "soft")):
            sb = m[(m["cush_bin"] == b) & m["sire"].notna()][["sire", "date", "one", "top3f"]]
            m = asof_merge(m, sb, ["sire"], ["one", "top3f"], f"sc{nm}_")
        m["sire_hard_aff"] = np.where((m["schard_one"] >= 60) & (m["sa_one"] >= 300), m["schard_top3f"] / m["schard_one"] - m["sa_top3f"] / m["sa_one"], np.nan)
        m["sire_soft_aff"] = np.where((m["scsoft_one"] >= 60) & (m["sa_one"] >= 300), m["scsoft_top3f"] / m["scsoft_one"] - m["sa_top3f"] / m["sa_one"], np.nan)
        for sn in ("summer", "winter"):
            sb = m[(m["season"] == sn) & m["sire"].notna()][["sire", "date", "one", "top3f"]]
            m = asof_merge(m, sb, ["sire"], ["one", "top3f"], f"ss{sn}_")
            m[f"sire_{sn}_aff"] = np.where((m[f"ss{sn}_one"] >= 80) & (m["sa_one"] >= 300), m[f"ss{sn}_top3f"] / m[f"ss{sn}_one"] - m["sa_top3f"] / m["sa_one"], np.nan)
    sire_cov = float(m["sire"].notna().mean()) if "sire" in m.columns else 0.0

    m = m.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    m = add_pincer(m)
    m.attrs["cuts"] = cuts
    m.attrs["names"] = meta_names
    m.attrs["sire_coverage"] = sire_cov
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT / "mining.parquet", index=False)
    (OUT / "mining_meta.json").write_text(json.dumps({"cuts": cuts, "names": meta_names, "sire_coverage": sire_cov, "rows": int(len(m)), "cols": list(m.columns)}, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def add_pincer(m: pd.DataFrame) -> pd.DataFrame:
    """逃げはさみ: 馬番順（取消・除外後の確定した出走馬の並び）で隣り合う馬の前走位置取りが先行（定義 d1〜d3）か。
    d1: 前走の初角が 3 番手以内 / d2: 前走の 4 角が 3 番手以内 / d3: 前走の初角が 1 番手（逃げ）。
    当日の枠順（馬番）が確定した後にだけ分かる。隣の馬自身の前走の位置取り（過去の結果）だけを使い、今回の結果は使わない。
    *_pin: 両隣が先行（内外とも）/ *_p1: 内の隣または外の隣の片方だけが先行（両隣のある馬）/ *_edge: 外端（最大馬番）で内の隣が先行 /
    *_in1: 内端（最小馬番）で外の隣が先行 / *_self_nl: 自分は前走で先行ではない（前走の位置取りが分かっていて、定義の基準より後ろ）。"""
    m = m.sort_values(["race_id", "horse_number"], kind="stable").reset_index(drop=True)
    gr = m.groupby("race_id", sort=False)
    has_in = gr["horse_number"].shift(1).notna()
    has_out = gr["horse_number"].shift(-1).notna()
    defs = {"d1": (m["prev_pos1"] <= 3, m["prev_pos1"] > 3, m["prev_pos1"].notna()),
            "d2": (m["prev_posL"] <= 3, m["prev_posL"] > 3, m["prev_posL"].notna()),
            "d3": (m["prev_pos1"] <= 1, m["prev_pos1"] > 1, m["prev_pos1"].notna())}
    for k, (lead, notlead, known) in defs.items():
        f = lead.astype(float).where(known, 0.0)
        left = f.groupby(m["race_id"], sort=False).shift(1).fillna(0.0)          # 内側の隣（馬番が 1 つ小さい出走馬）
        right = f.groupby(m["race_id"], sort=False).shift(-1).fillna(0.0)        # 外側の隣
        m[f"{k}_cnt"] = (left + right)
        m[f"{k}_pin"] = ((left == 1) & (right == 1) & has_in & has_out).astype(int)
        m[f"{k}_p1"] = ((left + right == 1) & has_in & has_out).astype(int)
        m[f"{k}_edge"] = ((left == 1) & ~has_out).astype(int)
        m[f"{k}_in1"] = ((right == 1) & ~has_in).astype(int)
        m[f"{k}_self_nl"] = (notlead & known).astype(int)
    return m.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)


def check() -> None:
    m = pd.read_parquet(OUT / "mining.parquet")
    print(m.shape)
    for c in ["cushion_value", "cush_bin", "dmoist_bin", "style5", "n_front", "seas_aff", "pref_hard", "t_local_aff", "sire_hard_aff", "agari_rank", "wt_dev"]:
        if c in m.columns:
            print(c, round(float(m[c].notna().mean()), 3))
    print(m.groupby("year")[["p_win", "cush_bin"]].agg(lambda s: float(s.notna().mean())))
    print(json.loads((OUT / "mining_meta.json").read_text())["cuts"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    a = ap.parse_args()
    build() if a.cmd == "build" else check()
