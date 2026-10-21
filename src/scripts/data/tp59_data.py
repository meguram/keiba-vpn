"""tp-59 データ: 騎手・調教師の回収率分析用の『1 出走 = 1 行』の表を作る（2020〜2026 年）。

python -m src.scripts.data.tp59_data build        # ホストで実行。出力 <TP59_OUT>/master.parquet
python -m src.scripts.data.tp59_data check        # 件数・還元率の簡易確認

列: race_id, date, year, horse_number, horse_id, jockey_id, trainer_id（すべて ID）, レース条件（venue_code, surface, distance, dist_band, cls, field_size, track_condition, sex_age）,
    履歴由来（prev_jockey_id, prev_trainer_id, chg_jockey, lay_days, run_no, prev_fin）, 結果（finish_position, win, top2, top3）,
    市場（win_odds, popularity, place_odds_min/max。確定値）, 払戻（ret_win, ret_place は 100 円あたりの確定払戻額）, モデル（p_win, p_top2, p_top3 = V4 の予測。2022 年以降のみ）。
履歴由来の列はその馬の直前の出走（2020 年以降の全 JRA 出走）から作る（as-of。同日の出走は使わない）。
市場オッズは分析（市場に織り込まれているか・馬券の評価）にだけ使い、特徴量にはしない。
取消（オッズ < 1）・払戻が曖昧なレースは除く（tp36_edge.load_frame と同じ規則）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BET = ROOT / "data/local/modeling/betting"
B26 = ROOT / "data/local/modeling/betting_2026"
TABLES = ROOT / "data/page_reference/tables"
FRAME = BET / "tp36/frame_all.parquet"
OUT = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
YEARS = list(range(2020, 2027))

RES_COLS = ["race_id", "date", "venue_code", "surface", "distance", "grade", "race_class", "track_condition", "field_size", "round",
            "finish_position", "horse_number", "horse_id", "sex_age", "jockey_weight", "jockey_id", "trainer_id", "weight", "weight_change"]


def class_of(race_class: pd.Series, grade: pd.Series) -> pd.Series:
    """クラス区分（新馬 / 未勝利 / 1勝 / 2勝 / 3勝 / オープン / G3 / G2 / G1）。障害は surface 側で別扱い。"""
    s = race_class.astype(str) + "|" + grade.astype(str)
    out = np.select(
        [s.str.contains("新馬"), s.str.contains("未勝利"), s.str.contains("1勝|１勝|500万"), s.str.contains("2勝|２勝|1000万"),
         s.str.contains("3勝|３勝|1600万"), s.str.contains("G1|GI|Ｇ１|Ｇ１", regex=True), s.str.contains("G2|GII|Ｇ２", regex=True),
         s.str.contains("G3|GIII|Ｇ３", regex=True)],
        ["c0_newcomer", "c1_maiden", "c2_1win", "c3_2win", "c4_3win", "g1", "g2", "g3"], "op")
    return pd.Series(out, index=race_class.index)


def dist_band(d: pd.Series) -> pd.Series:
    x = pd.to_numeric(d, errors="coerce")
    return pd.cut(x, [0, 1300, 1500, 1900, 2400, 99999], labels=["d_sprint", "d_mile_s", "d_mile", "d_mid", "d_long"]).astype(str)


def load_results() -> pd.DataFrame:
    parts = []
    for y in YEARS:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=RES_COLS)
            parts.append(d)
    r = pd.concat(parts, ignore_index=True)
    r["race_id"] = r["race_id"].astype(str)
    r["horse_id"] = r["horse_id"].astype(str)
    r["jockey_id"] = r["jockey_id"].astype("string")
    r["trainer_id"] = r["trainer_id"].astype("string")
    r["horse_number"] = pd.to_numeric(r["horse_number"], errors="coerce").astype("Int64")
    r["date"] = pd.to_datetime(r["date"])
    r["finish_position"] = pd.to_numeric(r["finish_position"], errors="coerce")
    return r.drop_duplicates(["race_id", "horse_number"])


def add_history(r: pd.DataFrame) -> pd.DataFrame:
    """前走の騎手・調教師・着順、間隔、出走回数（同じ馬の直前の出走から）。"""
    r = r.sort_values(["horse_id", "date", "race_id"], kind="stable").copy()
    g = r.groupby("horse_id", sort=False)
    r["prev_jockey_id"] = g["jockey_id"].shift(1)
    r["prev_trainer_id"] = g["trainer_id"].shift(1)
    r["prev_fin"] = g["finish_position"].shift(1)
    pd_ = g["date"].shift(1)
    r["lay_days"] = (r["date"] - pd_).dt.days
    r["run_no"] = g.cumcount()
    r["chg_jockey"] = np.where(r["prev_jockey_id"].notna(), (r["jockey_id"] != r["prev_jockey_id"]).astype(float), np.nan)
    r["chg_trainer"] = np.where(r["prev_trainer_id"].notna(), (r["trainer_id"] != r["prev_trainer_id"]).astype(float), np.nan)
    return r.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)


def load_market() -> pd.DataFrame:
    w = pd.concat([pd.read_parquet(BET / "odds_win_place.parquet"), pd.read_parquet(B26 / "odds_win_place.parquet")], ignore_index=True)
    pay = pd.concat([pd.read_parquet(BET / "payoff.parquet"), pd.read_parquet(B26 / "payoff.parquet")], ignore_index=True)
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    bad_odds = set(w.loc[~(w["win_odds"] >= 1.0), "race_id"])
    w = w[~w["race_id"].isin(bad | bad_odds)].drop_duplicates(["race_id", "horse_number"]).copy()
    for bt, col in (("tansho", "ret_win"), ("fukusho", "ret_place")):
        p = pay[pay["bet_type"] == bt][["race_id", "combo", "payout"]].copy()
        p["horse_number"] = pd.to_numeric(p["combo"], errors="coerce")
        p = p.dropna(subset=["horse_number"]).groupby(["race_id", "horse_number"], as_index=False)["payout"].sum()
        p["horse_number"] = p["horse_number"].astype(int)
        w["horse_number"] = w["horse_number"].astype(int)
        w = w.merge(p.rename(columns={"payout": col}), on=["race_id", "horse_number"], how="left")
        w[col] = w[col].fillna(0.0)
    w["race_id"] = w["race_id"].astype(str)
    return w[["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "popularity", "ret_win", "ret_place"]]


def build() -> pd.DataFrame:
    r = add_history(load_results())
    r["year"] = r["date"].dt.year
    r["cls"] = class_of(r["race_class"], r["grade"])
    r["dist_band"] = dist_band(r["distance"])
    fin = r["finish_position"]
    r["win"] = (fin == 1).astype(float)
    r["top2"] = ((fin >= 1) & (fin <= 2)).astype(float)
    r["top3"] = ((fin >= 1) & (fin <= 3)).astype(float)
    m = load_market()
    r["horse_number"] = r["horse_number"].astype(int)
    d = r.merge(m, on=["race_id", "horse_number"], how="inner")
    f = pd.read_parquet(FRAME, columns=["race_id", "horse_number", "p_win", "p_top2", "p_top3"])
    f["race_id"] = f["race_id"].astype(str)
    d = d.merge(f.drop_duplicates(["race_id", "horse_number"]), on=["race_id", "horse_number"], how="left")
    d["q_win"] = (1.0 / d["win_odds"]) / (1.0 / d["win_odds"]).groupby(d["race_id"]).transform("sum")
    d["place_eff"] = d["place_odds_min"] + 0.3 * (d["place_odds_max"] - d["place_odds_min"])
    d["race_id"] = d["race_id"].astype(str)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        d = build()
        d.to_parquet(OUT / "master.parquet", index=False)
        print(d.shape)
    d = pd.read_parquet(OUT / "master.parquet")
    g = d.groupby("year")
    print(pd.DataFrame({"rows": g.size(), "races": g["race_id"].nunique(), "roi_win": g["ret_win"].mean(), "roi_place": g["ret_place"].mean(),
                        "win_rate": g["win"].mean(), "p_win_na": g["p_win"].apply(lambda s: s.isna().mean()),
                        "jockey_na": g["jockey_id"].apply(lambda s: s.isna().mean())}).round(4))


if __name__ == "__main__":
    main()
