"""tp-55: ラップ予想の学習・評価用のレース表（レース単位の特徴量＋実測ラップ）を作る。ホストで実行。

python -m src.scripts.data.tp55_build [--out data/local/modeling/lap_forecast] [--years 2020 ... 2026]
出力: races_feat.parquet（1 行 = 1 レース。特徴量・コース情報・laps(JSON)・日付）、build_report.json（被覆率・欠損）。
先読み検査: --verify-cut YYYY-MM-DD で、その日付以降の結果をすべて消した入力から作り直しても、それ以前のレースの特徴量が
完全に一致することを確かめる（as-of の完全一致テスト）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
YEARS = list(range(2020, 2027))
CATS = ["venue", "surface", "direction", "weather", "track_condition", "grade", "race_class"]


def _read(year: int, name: str, cols: list[str]) -> pd.DataFrame:
    p = TABLES / str(year) / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=cols)
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(p).schema_arrow.names)
    d = pd.read_parquet(p, columns=[c for c in cols if c in have])
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d


def load_inputs(years: list[int], cut: str | None = None):
    res_cols = ["race_id", "date", "venue", "surface", "distance", "direction", "grade", "race_class", "weather", "track_condition",
                "field_size", "round", "lap_times", "horse_number", "horse_id", "jockey_id", "passing_order", "sex_age", "jockey_weight"]
    sh_cols = ["race_id", "horse_number", "horse_id", "sex_age", "jockey_weight", "jockey_id"]
    sm_cols = ["race_id", "uno", "h1_ten1f", "ten1f_best"]
    res, shu, smr = [], [], []
    for y in years:
        res.append(_read(y, "race_result_flat", res_cols))
        shu.append(_read(y, "race_shutuba_flat", sh_cols))
        smr.append(_read(y, "smartrc_race_flat", sm_cols))
    res = pd.concat(res, ignore_index=True)
    shu = pd.concat(shu, ignore_index=True)
    smr = pd.concat(smr, ignore_index=True)
    res["date"] = pd.to_datetime(res["date"])
    if cut:
        res = res[res["date"] < pd.Timestamp(cut)]
    return res, shu, smr


def _race_base(df: pd.DataFrame) -> pd.DataFrame:
    races = df.drop_duplicates("race_id")[["race_id", "date", "venue", "surface", "distance", "direction", "grade", "race_class", "weather",
                                           "track_condition", "field_size", "round"]].copy()
    races["round_no"] = pd.to_numeric(races["round"], errors="coerce")
    rid = races["race_id"].astype(str)
    races["kai"] = pd.to_numeric(rid.str[6:8], errors="coerce")
    races["nichime"] = pd.to_numeric(rid.str[8:10], errors="coerce")
    races["month"] = pd.to_datetime(races["date"]).dt.month
    races["n_laps"] = races["distance"].map(lf.n_laps_of)
    races["d0"] = races["distance"].map(lf.first_lap_meters)
    return races


def _finish(races: pd.DataFrame, res: pd.DataFrame, shu: pd.DataFrame, smr: pd.DataFrame, rep: dict, done: pd.DataFrame) -> pd.DataFrame:
    """races: 対象レース（laps 列は無くてもよい）。done: ラップが分かっている終了済みレース（同日の傾向用）。"""
    ent = shu[shu["race_id"].isin(races["race_id"])].copy()
    miss = set(races["race_id"]) - set(ent["race_id"])
    if miss:
        alt = res[res["race_id"].isin(miss)][["race_id", "horse_number", "horse_id", "sex_age", "jockey_weight", "jockey_id"]]
        ent = pd.concat([ent, alt], ignore_index=True)
    rep["races_entrants_from_result_fallback"] = len(miss)
    ent = ent.merge(races[["race_id", "date"]], on="race_id", how="left")
    ent["horse_number"] = pd.to_numeric(ent["horse_number"], errors="coerce")
    ent = ent.dropna(subset=["horse_number", "horse_id"]).reset_index(drop=True)

    runs = res[["race_id", "horse_id", "date", "early", "jockey_id"]].dropna(subset=["horse_id"]).copy()
    hf = lf.asof_horse_features(ent, runs)
    ent = pd.concat([ent.reset_index(drop=True), hf.drop(columns=["race_id", "horse_id"]).reset_index(drop=True)], axis=1)
    ent["jockey_early"] = lf.asof_jockey_early(ent, runs).values
    smr = smr.dropna(subset=["race_id"]).copy()
    smr["horse_number"] = pd.to_numeric(smr["uno"], errors="coerce")
    for c in ("h1_ten1f", "ten1f_best"):
        smr[c] = pd.to_numeric(smr[c], errors="coerce") / 10.0
    smr = smr.drop_duplicates(["race_id", "horse_number"])
    ent = ent.merge(smr[["race_id", "horse_number", "h1_ten1f", "ten1f_best"]], on=["race_id", "horse_number"], how="left")
    ent = ent.rename(columns={"h1_ten1f": "ten1f"})
    ag = [lf.parse_sex_age(s) for s in ent["sex_age"]]
    ent["age"] = [a[0] for a in ag]
    ent["is_female"] = [a[1] for a in ag]
    ent["jockey_weight"] = pd.to_numeric(ent["jockey_weight"], errors="coerce")

    rows = []
    for race_id, g in ent.groupby("race_id", sort=False):
        d = lf.race_features_one(g)
        d["race_id"] = race_id
        rows.append(d)
    rf = pd.DataFrame(rows)
    rep["ten_coverage_share_races"] = float((rf["ten_cnt"] > 0).mean()) if len(rf) else None
    out = races.merge(rf, on="race_id", how="inner")
    par0_src = done[done["date"] < pd.Timestamp("2022-01-01")][["venue", "surface", "distance", "direction", "laps"]]
    par0 = lf.fit_par(par0_src)
    sd_in = pd.concat([done[["race_id", "date", "venue", "surface", "round_no", "laps", "distance", "direction"]],
                       out[~out["race_id"].isin(done["race_id"])][["race_id", "date", "venue", "surface", "round_no", "distance", "direction"]].assign(laps=None)],
                      ignore_index=True)
    sd_in = sd_in[sd_in["date"].isin(set(out["date"]))]
    sd = lf.sameday_features(sd_in, par0)
    out = out.merge(sd, on="race_id", how="left")
    out["course_key"] = [lf.course_key(v, s, d, di) for v, s, d, di in zip(out["venue"], out["surface"], out["distance"], out["direction"])]
    return out


def build(years: list[int], cut: str | None = None) -> tuple[pd.DataFrame, dict]:
    res, shu, smr = load_inputs(years, cut)
    res = res[res["surface"].isin(["芝", "ダート"])].copy()
    res["early"] = [lf.first_corner_norm(p, f) for p, f in zip(res["passing_order"], res["field_size"])]
    races = _race_base(res)
    laps = res.drop_duplicates("race_id")[["race_id", "lap_times"]].copy()
    laps["laps"] = laps["lap_times"].map(lf.parse_laps)
    races = races.merge(laps[["race_id", "laps"]], on="race_id", how="left")
    rep = {"n_races_all": int(len(races)), "n_races_no_laps": int(races["laps"].isna().sum())}
    races = races.dropna(subset=["laps"]).copy()
    races = races[races["laps"].map(len) == races["distance"].map(lf.n_laps_of)].copy()
    rep["n_races_laps_ok"] = int(len(races))
    races["date"] = pd.to_datetime(races["date"])
    done = races[["race_id", "date", "venue", "surface", "round_no", "laps", "distance", "direction"]]
    out = _finish(races.drop(columns=["laps"]), res, shu, smr, rep, done)
    out = out.merge(races[["race_id", "laps"]], on="race_id", how="left")
    out["laps_json"] = out["laps"].map(json.dumps)
    out = out.drop(columns=["laps"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    rep["n_races_features"] = int(len(out))
    rep["by_year"] = out["date"].str[:4].value_counts().sort_index().to_dict()
    return out.sort_values(["date", "race_id"]).reset_index(drop=True), rep


def build_infer(years: list[int], dates: list[str]) -> tuple[pd.DataFrame, dict]:
    """予測用: 指定日の全レース（結果が無くてもよい。出馬表から）の特徴量。ラップ列は無い。"""
    res, shu, smr = load_inputs(years, None)
    res = res[res["surface"].isin(["芝", "ダート"])].copy()
    res["early"] = [lf.first_corner_norm(p, f) for p, f in zip(res["passing_order"], res["field_size"])]
    sh_full = []
    for y in sorted({int(d[:4]) for d in dates}):
        sh_full.append(_read(y, "race_shutuba_flat", ["race_id", "date", "venue", "surface", "distance", "direction", "grade", "race_class", "weather",
                                                      "track_condition", "field_size", "round", "horse_number", "horse_id", "sex_age", "jockey_weight", "jockey_id"]))
    sh_full = pd.concat(sh_full, ignore_index=True)
    sh_full["date"] = pd.to_datetime(sh_full["date"])
    sh_full = sh_full[sh_full["date"].isin([pd.Timestamp(d) for d in dates]) & sh_full["surface"].isin(["芝", "ダート"])]
    races = _race_base(sh_full)
    races["date"] = pd.to_datetime(races["date"])
    laps = res.drop_duplicates("race_id")[["race_id", "lap_times"]].copy()
    laps["laps"] = laps["lap_times"].map(lf.parse_laps)
    done_r = _race_base(res).merge(laps[["race_id", "laps"]], on="race_id", how="left").dropna(subset=["laps"])
    done_r["date"] = pd.to_datetime(done_r["date"])
    done = done_r[["race_id", "date", "venue", "surface", "round_no", "laps", "distance", "direction"]]
    shu2 = pd.concat([shu, sh_full[shu.columns.intersection(sh_full.columns)]], ignore_index=True).drop_duplicates(["race_id", "horse_number"])
    rep: dict = {"mode": "infer", "dates": dates}
    out = _finish(races, res, shu2, smr, rep, done)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    rep["n_races_features"] = int(len(out))
    return out.sort_values(["date", "race_id"]).reset_index(drop=True), rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/local/modeling/lap_forecast"))
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--verify-cut", default=None, help="この日付以降の結果を消して作り直し、それ以前の特徴量が完全一致するか検査")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, rep = build(a.years)
    df.to_parquet(out / "races_feat.parquet", index=False)
    if a.verify_cut:
        df2, _ = build(a.years, cut=a.verify_cut)
        pre = df[df["date"] < a.verify_cut].set_index("race_id")
        pre2 = df2[df2["date"] < a.verify_cut].set_index("race_id")
        common = pre.index.intersection(pre2.index)
        num = [c for c in lf.RACE_NUM_COLS if c in pre.columns]
        # 結果を消した版は、cut 日以降の『同日の傾向』などに影響しない（cut より前のレースだけを比べる）
        diff = {}
        for c in num + ["sd_n", "sd_q_mean", "sd_f3_mean", "sd_l3_mean", "sd_q_last"]:
            a1 = pre.loc[common, c].astype(float).values
            b1 = pre2.loc[common, c].astype(float).values
            bad = ~((a1 == b1) | (np.isnan(a1) & np.isnan(b1)))
            if bad.any():
                diff[c] = int(bad.sum())
        rep["verify_cut"] = {"cut": a.verify_cut, "n_compared": int(len(common)), "columns_with_diff": diff}
    (out / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
