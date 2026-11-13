"""重賞のルールベース予想家: 1 出走 = 1 行の表に、切り口に使う as-of の列を足す。

入力: folklore の master（`FOLKLORE_YEAR_END=2026 FOLKLORE_OUT=<GRADED_OUT>/master python -m src.scripts.data.folklore_build`）。
      ほかに手元の表（race_result_flat の course_type、race_shutuba_past_flat の過去 5 走）と 5 代血統（系統の当て方は
      src.research.pedigree.sire_dam_influence.pedigree_lines と同じ）。GCS は読まない。
出力: <GRADED_OUT>/frame.parquet（切り口の水準を作る前の列の表）

ここで足す列はすべて「その走より前」の情報だけで作る（馬ごとに shift してから累積、または merge_asof で前の日までを数える）。
その走の結果の列（fin・y1・y3 など）は、集計の目的変数としてだけ残す（catalog.ALLOWED_COLUMNS には入れない）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.graded import race_keys as RK

REPO = Path(__file__).resolve().parents[3]
TABLES = REPO / "data/page_reference/tables"
GRADED_OUT = Path(os.environ.get("GRADED_OUT", "/home/jovyan/work/graded_out"))
MASTER = GRADED_OUT / "master/master.parquet"
GRADES = ("G1", "G2", "G3")
YEARS = range(2016, 2027)

MASTER_COLS = [
    "race_id", "horse_id", "date", "dt", "year", "month", "venue", "vc", "surf", "dist", "dband", "direction", "grade", "cls", "race_class",
    "race_name", "field", "fin", "y1", "y3", "tsec", "tdiff", "hn", "bracket_number", "jw", "wt", "wtc", "sex", "age", "track", "io",
    "weight_rule", "fillies", "mixed", "pdiff", "jockey_id", "jockey_name", "trainer_id", "ok",
    # 前走・履歴（folklore が as-of で作った列）
    "p_dt", "p_fin", "p_tdiff", "p2_fin", "p3_fin", "r5_top3", "win_streak", "p_cls", "p_grade", "p_race_name", "p_month", "p_dist",
    "p_surf", "p_vc", "p_direction", "p_io", "p_track", "p_jw", "p_fillies", "p_field", "p_pos1", "p_pos1r", "p_pos4", "p_pos4r",
    "p_l3rank", "interval", "since_rest", "fb_n_prev", "fb_top3_prev", "nige_rate", "senko_rate", "agari1_prev", "nige_cand",
    "n_nige_cand", "course_n_prev", "course_top3_prev", "vs_n_prev", "vs_top3_prev", "dbs_n_prev", "dbs_top3_prev", "dir_n_prev",
    "dir_top3_prev", "bad_n_prev", "bad_top3_prev", "starts_prev", "top3_prev", "wins_prev", "vs_old_prev", "max_field_prev",
    "venue_n_prev", "op_n_prev", "best_dist_min", "best_dist_max", "dist_max_prev", "dist_min_prev", "since_geld", "tozai", "away",
    "blinker", "p_blinker", "blk_ever_prev", "since_transfer", "meet_day", "meet_len", "cush", "j_cont", "j_rate", "pj_rate",
    "j_foreign", "j_short", "jc_n", "jc_y3", "j_nige_rate", "jday_w_before", "t_rate", "t30_n", "t30_y3", "tc_n", "tc_y3", "tfb_n",
    "tfb_y3", "jt_n", "jt_y3", "n_same_tr", "f_name", "mf_name", "ok_eval", "mg_rank_p1", "mg_rank_max5",
]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_master(path: Path = MASTER) -> pd.DataFrame:
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema_arrow.names)
    d = pd.read_parquet(path, columns=[c for c in MASTER_COLS if c in have] + ["odds"])
    # 発走したか（中止の馬も含む）。着順の無い馬のうち、オッズが正なら発走して中止（着外）、0 なら取消・除外。
    # folklore の master はオッズが正の行だけを保存するので、今は全行 True（FOLKLORE_KEEP_DNF=1 の master では中止の馬の行が入る）。
    # オッズはこの判定にだけ使い、すぐに捨てる（切り口・点には使わない。catalog.FORBIDDEN_EXACT で検査する）
    d["started"] = (_num(d["fin"]) > 0) | (_num(d["odds"]) > 0)
    d = d.drop(columns=["odds"])
    # 中止の馬も「出走した走」として過去の成績（騎手・厩舎・父の産駒・重賞の実績）に着外で数える
    dnf = d["started"] & _num(d["fin"]).isna() & (d["surf"].astype(str) != "障")
    d["ok"] = d["ok"].astype(str).str.lower().isin(["true", "1", "1.0"]) | dnf
    d.loc[dnf, ["y3", "y1"]] = 0.0
    for c in ("race_id", "horse_id", "jockey_id", "trainer_id"):
        d[c] = d[c].astype(str)
    d["dt"] = pd.to_datetime(d["dt"])
    d["p_dt"] = pd.to_datetime(d["p_dt"])
    return d


def add_course(d: pd.DataFrame) -> pd.DataFrame:
    """レース名の鍵とコースの鍵。"""
    import pyarrow.parquet as pq
    parts = []
    for y in YEARS:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists() and "course_type" in pq.ParquetFile(f).schema_arrow.names:
            parts.append(pd.read_parquet(f, columns=["race_id", "course_type"]).drop_duplicates("race_id"))
    ct = pd.concat(parts, ignore_index=True)
    ct["race_id"] = ct["race_id"].astype(str)
    d = d.merge(ct, on="race_id", how="left")
    d["rkey"] = RK.canonical_keys(d["race_name"], d["month"])
    d["p_rkey"] = RK.canonical_keys(d["p_race_name"].fillna(""), d["p_month"]).where(d["p_race_name"].notna())
    d["io2"] = RK.resolve_io(d["vc"], d["surf"], d["dist"], d["io"], d["course_type"], d["cls"])
    d["course"] = RK.course_key(d["venue"], d["surf"], d["dist"], d["io2"])
    return d


def _last_region(s) -> str | None:
    from src.scripts.data.folklore_build import NAR, OVERSEAS
    try:
        arr = json.loads(s) if isinstance(s, str) else []
    except Exception:  # noqa: BLE001
        arr = []
    if not arr:
        return None
    raw = str(arr[0].get("raw", ""))
    if any(v in raw for v in OVERSEAS):
        return "海外"
    if any((" " + v + " ") in (" " + raw.replace("\xa0", " ") + " ") for v in NAR):
        return "地方"
    return "中央"


def add_prev_region(d: pd.DataFrame) -> pd.DataFrame:
    """前走が海外・地方か（出馬表の過去 5 走。2019 年〜）。2016〜2018 は判定できない（None）。"""
    parts = []
    for y in YEARS:
        f = TABLES / str(y) / "race_shutuba_past_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["race_id", "horse_id", "past_races"]))
    pp = pd.concat(parts, ignore_index=True)
    pp["race_id"] = pp["race_id"].astype(str)
    pp["horse_id"] = pp["horse_id"].astype(str)
    pp = pp.drop_duplicates(["race_id", "horse_id"])
    pp["prev_region"] = pp["past_races"].map(_last_region)
    return d.merge(pp[["race_id", "horse_id", "prev_region"]], on=["race_id", "horse_id"], how="left")


def add_horse_history(d: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの as-of の列（shift してから累積）。"""
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    # 着差（勝ち馬は 2 着とのタイム差をマイナスで）。その走の値 → 次の走の「前走の着差」。
    # folklore の tdiff は、取消・除外の馬（タイム 0）も入れた最小値から引いているレースがあり、そのレースでは走破タイムそのものになる。
    # ここでは着順の付いた馬（タイム > 0）だけで計算し直す。
    t = _num(d["tsec"]).where(lambda s: s > 0)
    t1 = t.groupby(d["race_id"]).transform("min")
    t2 = t.groupby(d["race_id"]).transform(lambda s: s.nsmallest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan)
    signed = (t - t1).where(d["fin"] > 1, -(t2 - t))
    d["p_margin"] = signed.groupby(d["horse_id"]).shift(1)
    # 前走のペース（前半 3F − 後半 3F の 3 分位。境目は 2016〜2017 のレースだけで決める）
    rr = d.drop_duplicates("race_id")[["race_id", "year", "surf", "dband", "pdiff"]]
    base = rr[rr["year"] <= 2017]
    q = base.groupby(["surf", "dband"])["pdiff"].quantile([1 / 3, 2 / 3]).unstack()
    rr = rr.join(q, on=["surf", "dband"])
    v = rr["pdiff"].to_numpy(float)
    lo, hi = rr[1 / 3].to_numpy(float), rr[2 / 3].to_numpy(float)
    rr["pace2"] = np.where(np.isnan(v) | np.isnan(lo), None, np.where(v < lo, "H", np.where(v > hi, "S", "M")))
    d = d.merge(rr[["race_id", "pace2"]], on="race_id", how="left")
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    d["p_pace2"] = g["pace2"].shift(1)
    d["p_mixed"] = g["mixed"].shift(1)
    d["p_race_id"] = g["race_id"].shift(1)
    # 重賞・G1 の実績、同じ重賞の実績（リピーター）
    okf = (d["ok"].fillna(False).astype(bool) & _num(d["y3"]).notna()).astype(float)
    y3, y1 = _num(d["y3"]).fillna(0), _num(d["y1"]).fillna(0)
    gr = d["grade"].isin(GRADES).astype(float) * okf
    g1 = (d["grade"] == "G1").astype(float) * okf

    def cum(v, keys):
        return v.groupby(keys).cumsum() - v
    d["g_n_prev"] = cum(gr, d["horse_id"])
    d["g_top3_prev"] = cum(gr * y3, d["horse_id"])
    d["g_win_prev"] = cum(gr * y1, d["horse_id"])
    d["g1_n_prev"] = cum(g1, d["horse_id"])
    d["g1_top3_prev"] = cum(g1 * y3, d["horse_id"])
    rk = d["rkey"].where(d["grade"].isin(GRADES), "__")
    d["rep_n_prev"] = cum(gr, [d["horse_id"], rk])
    d["rep_top3_prev"] = cum(gr * y3, [d["horse_id"], rk])
    return d


def add_race_level(d: pd.DataFrame) -> pd.DataFrame:
    """前走のレースレベル: 前走で一緒に走った馬（自分を除く）のうち、前走の翌日から今回の前日までに 1 勝以上した頭数。"""
    runners = d[["race_id", "horse_id", "dt"]].rename(columns={"race_id": "p_race_id", "horse_id": "mate", "dt": "p_dt0"})
    wins = d.loc[d["y1"] == 1, ["horse_id", "dt"]].rename(columns={"horse_id": "mate", "dt": "wdt"})
    tgt = d.loc[d["p_race_id"].notna(), ["race_id", "horse_id", "dt", "p_race_id"]]
    x = tgt.merge(runners, on="p_race_id", how="inner")
    x = x[x["mate"] != x["horse_id"]]
    x = x.merge(wins, on="mate", how="inner")
    x = x[(x["wdt"] > x["p_dt0"]) & (x["wdt"] < x["dt"])]
    lv = x.groupby(["race_id", "horse_id"])["mate"].nunique().rename("lvl_mates_won").reset_index()
    d = d.merge(lv, on=["race_id", "horse_id"], how="left")
    d["lvl_mates_won"] = d["lvl_mates_won"].fillna(0).where(d["p_race_id"].notna())
    return d


def asof_counts(d: pd.DataFrame, keys: list[str], mask: pd.Series, name: str) -> pd.DataFrame:
    """keys ごとに、その日より前（前の日まで）の出走数と 3 着内数。mask の走だけを数え、全行に付ける。"""
    m = d.loc[mask & d["ok"].fillna(False).astype(bool) & d[keys].notna().all(axis=1), keys + ["dt", "y3"]]
    agg = m.groupby(keys + ["dt"], sort=False).agg(n=("y3", "size"), k=("y3", "sum")).reset_index().sort_values("dt")
    agg[["n", "k"]] = agg.groupby(keys, sort=False)[["n", "k"]].cumsum()
    left = d[keys + ["dt"]].reset_index().sort_values("dt")
    for c in keys:
        left[c] = left[c].astype(str)
        agg[c] = agg[c].astype(str)
    r = pd.merge_asof(left, agg, on="dt", by=keys, allow_exact_matches=False).set_index("index").sort_index()
    d[f"{name}_n"] = r["n"].fillna(0).to_numpy()
    d[f"{name}_y3"] = r["k"].fillna(0).to_numpy()
    return d


def entry_weights() -> pd.DataFrame:
    """レース結果の表の全頭（取消・除外・中止も含む）の斤量。master は着順の付いた馬（と、FOLKLORE_KEEP_DNF=1 なら中止の馬）だけなので、
    レース内の斤量の順位は、ここから出走表の時点の全頭で作る。"""
    parts = []
    for y in YEARS:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["race_id", "horse_id", "jockey_weight"]))
    e = pd.concat(parts, ignore_index=True)
    e["race_id"], e["horse_id"] = e["race_id"].astype(str), e["horse_id"].astype(str)
    return e.drop_duplicates(["race_id", "horse_id"]).rename(columns={"jockey_weight": "jw"})


def add_entry_level(d: pd.DataFrame, entries: pd.DataFrame | None = None) -> pd.DataFrame:
    """出走表の時点（前日）で決まる、レース内の列。取消・除外・中止の馬も含めた全頭で数える（レース後に決まる情報を入れない）。
    - field_decl: 出走表の頭数（folklore の field = race_result_flat の field_size。取消・除外も含む全頭）
    - jw_rank・jw_max: entries（race_id, horse_id, jw。既定はレース結果の表の全頭）の中での斤量の順位。entries に無いレースは d の行だけで作る"""
    d = d.copy()
    d["field_decl"] = _num(d["field"])
    e = entries if entries is not None else entry_weights()
    e = e[e["race_id"].isin(set(d["race_id"]))].drop_duplicates(["race_id", "horse_id"]).copy()
    e["jw"] = _num(e["jw"])
    have = set(e["race_id"])
    own = d.loc[~d["race_id"].isin(have), ["race_id", "horse_id", "jw"]]
    e = pd.concat([e[["race_id", "horse_id", "jw"]], own], ignore_index=True)
    e["jw_rank"] = e.groupby("race_id")["jw"].rank(ascending=False, method="min", pct=True)
    e["jw_max"] = e["jw"] == e.groupby("race_id")["jw"].transform("max")
    d = d.drop(columns=[c for c in ("jw_rank", "jw_max") if c in d.columns])
    return d.merge(e[["race_id", "horse_id", "jw_rank", "jw_max"]], on=["race_id", "horse_id"], how="left")


def keep_runners(d: pd.DataFrame) -> pd.DataFrame:
    """発走した馬（着順の付いた馬と、中止の馬）だけにし、頭数をその数で数え直す（期待の 3 着内 = 3/頭数）。
    取消・除外の馬は、馬券も返還されるので評価から外す。中止の馬は着外（y3 = 0）として残す。
    前走などの履歴の列と、出走表の時点の列（add_entry_level）は、この前に全行で作っておく。"""
    d = d[d["started"].fillna(False).astype(bool)].copy()
    d["field"] = d.groupby("race_id")["horse_id"].transform("size")
    d["y3"] = (_num(d["fin"]) <= 3).astype(float)
    d["y1"] = (_num(d["fin"]) == 1).astype(float)
    return d.reset_index(drop=True)


def add_people_sire(d: pd.DataFrame) -> pd.DataFrame:
    gr = d["grade"].isin(GRADES)
    d = asof_counts(d, ["jockey_id"], gr, "jgr")          # 騎手の重賞での成績
    d = asof_counts(d, ["trainer_id"], gr, "tgr")         # 厩舎の重賞での成績
    alls = pd.Series(True, index=d.index)
    d = asof_counts(d, ["f_name", "course"], alls, "sc")   # 父の産駒の同じコースの成績（全クラス）
    d = asof_counts(d, ["f_name", "surf", "dband"], alls, "sd")   # 父の産駒の同じ芝ダ・距離帯の成績
    return d


def add_pedigree(d: pd.DataFrame) -> pd.DataFrame:
    """父・母父の小系統・大系統（sire_dam_influence.pedigree_lines と同じ当て方）。"""
    from src.research.pedigree.sire_dam_influence import pedigree_lines
    ids = sorted(d["horse_id"].unique())
    pl = pedigree_lines(ids)
    cols = {"父_小": "f_small", "父_大": "f_large", "母父_小": "mf_small", "母父_大": "mf_large"}
    pl = pl[list(cols)].rename(columns=cols)
    return d.join(pl, on="horse_id")


def build(master: Path = MASTER, out: Path | None = None, pedigree: bool = True) -> pd.DataFrame:
    d = load_master(master)
    print("master", d.shape, flush=True)
    d = add_course(d)
    d = add_prev_region(d)
    d = add_horse_history(d)
    d = add_race_level(d)
    d = add_entry_level(d)
    d = keep_runners(d)
    d = add_people_sire(d)
    if pedigree:
        d = add_pedigree(d)
    print("frame", d.shape, flush=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        for c in d.columns:
            if d[c].dtype == object:
                d[c] = d[c].astype("string")
        d.to_parquet(out, index=False)
    return d


def main() -> None:
    build(out=GRADED_OUT / "frame.parquet")


if __name__ == "__main__":
    main()
