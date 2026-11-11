"""最終追い切り = レース週（月曜〜金曜）の平日の追い切りのうち、4F 時計が最も速い 1 本（tp-35 の『1〜6 日前で最も新しい 1 本』とは別の定義）。

- レース週の月曜 = レース日から曜日分戻した日（月曜開催は前の週に数える。tp35_lap_groups.prev_weekend と同じ）。平日 = その月曜〜金曜。
- 候補はラップ種別が読める坂路・ウッド・ポリの本だけ（t4 が無い本は除く）。時計は t4（4F の通し）で比べ、同じなら新しい本（days_before が小さい）。
  坂路とウッドの 4F は水準が違うが、馬ごとに同じ週の中の速い方を採る（種別は 1 本の中のラップで決まる）。
- すべてレース日より前の調教だけ（as-of）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PLACE = {"栗坂": "栗東坂路", "美坂": "美浦坂路", "CW": "Cウッド(栗東)", "美W": "美浦ウッド(北)", "南W": "美浦ウッド(南)", "函W": "函館ウッド",
         "美P": "美浦ポリ", "DP": "ポリ(DP)"}
OUT_COLS = ["BW_pat", "BW_cls", "BW_kind", "BW_course", "BW_place", "BW_t4", "BW_l1", "BW_l2", "BW_days", "BW_n"]


WOOD_EDGES = [50.0, 52.0, 54.0, 56.0, 58.0]   # ウッド・ポリの 4F 時計の区分（2 秒刻み）。両端は 50.0 未満・58.0 以上
WOOD_LABELS = ["W~49.9", "W50.0-51.9", "W52.0-53.9", "W54.0-55.9", "W56.0-57.9", "W58.0~"]


def wood_band(t4: pd.Series) -> pd.Series:
    """ウッド・ポリの 4F 時計の区分。時計は小数 1 位に丸めてから区切る。"""
    i = np.digitize(np.round(t4.to_numpy(dtype=float), 1), WOOD_EDGES)
    return pd.Series(np.array(WOOD_LABELS, dtype=object)[i], index=t4.index)


def pattern_of(kind: pd.Series, cls: pd.Series, t4: pd.Series) -> pd.Series:
    """パターン名。坂路はラップ種別（A1〜X）、ウッド・ポリ（2F の時計が無い）は 4F 時計の区分。"""
    return pd.Series(np.where(kind.eq("slope"), cls, wood_band(t4)), index=kind.index)


def place_of(course: pd.Series) -> pd.Series:
    """調教場所の区分。『…一番時計』の表記は元の場所に揃える。表に無い場所は元の名前のまま。"""
    c = course.astype(str).str.replace("一番時計", "", regex=False)
    return c.map(PLACE).fillna(c)


def race_week_monday(race_date: pd.Series) -> pd.Series:
    rd = pd.to_datetime(race_date)
    wd = rd.dt.weekday.to_numpy()
    return rd - pd.to_timedelta(np.where(wd == 0, 7, wd), unit="D")


def pick_best_weekday(tp: pd.DataFrame) -> pd.DataFrame:
    """prepare_laps 済みの調教表から、(horse_id, race_date) ごとにレース週の平日のベストタイムの 1 本を返す（BW_* 列）。"""
    t = tp[tp["cls"].ne("NA") & tp["t4"].notna()].copy()
    td = pd.to_datetime(t["train_date"])
    ws = race_week_monday(t["race_date"])
    in_week = (td >= ws) & (td <= ws + pd.Timedelta(days=4)) & (t["days_before"] > 0) & (td < pd.to_datetime(t["race_date"]))
    t = t[in_week]
    n = t.groupby(["horse_id", "race_date"]).size().rename("BW_n")
    t = t.sort_values(["horse_id", "race_date", "t4", "days_before"], kind="stable")
    g = t.groupby(["horse_id", "race_date"], sort=False).head(1)
    g = g[["horse_id", "race_date", "cls", "kind", "course", "t4", "l1", "l2", "days_before"]].copy()
    g.columns = ["horse_id", "race_date", "BW_cls", "BW_kind", "BW_course", "BW_t4", "BW_l1", "BW_l2", "BW_days"]
    g.insert(5, "BW_place", place_of(g["BW_course"]))
    g.insert(2, "BW_pat", pattern_of(g["BW_kind"], g["BW_cls"], g["BW_t4"]))
    return g.merge(n.reset_index(), on=["horse_id", "race_date"], how="left")
