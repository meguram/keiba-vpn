"""追い切りの列を、training_laps（馬別の調教ページ由来、1 行 1 本）から作り直す（通説の検証の B 分類・A39・D13・K02）。

従来の ok_*（最終追い切りの乗り手・強度・ラップ…）は race_oikiri 表由来で、2020 年以降は乗り手・ラップが空だった。
training_laps（data/local/modeling/training_pipeline/features/training_laps.parquet、build_training_laps.py）は
馬ごとの調教の履歴で、各行は『次に出走したレース』（race_date）に紐づく。2020〜2025 年の出走の 57〜97% の馬が載っている。

- 最終追い切り = そのレースの調教のうち days_before が最小の 1 本。ok_* の同名の列を上書きする（B01〜B37 の定義はそのまま使う）。
- 1 週前（days_before 5〜10 で 7 に最も近い 1 本）: w1_*。直近の本数・強め以上の本数・同じ日の複数本・自己ベスト・初めの追い切りの日: tl_*。
- 調教日は race_date より前だけ（as-of）。2016〜2019 年は、2020 年以降も走った馬だけが載る（生き残りの偏り）ので使わない（NaN にする）。
- position は調教ページの『位置』（ウッドの走行位置、1 = 内 … 9 = 外と読む。公式の定義は未確認）。併せ馬の相手・頭数は載っていない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "data/local/modeling/training_pipeline/features/training_laps.parquet"
FIRST_YEAR = 2020
TL_VERSION = 2  # 列の作り方を変えたら上げる（folklore_verify.run が古い列のまま使い続けないための印）
INT_MAP = {"馬也": "馬也", "強め": "強め", "一杯": "一杯", "G強": "強め", "直強": "強め", "仕掛": "強め", "G一": "一杯", "直一": "一杯"}
WD = "月火水木金土日"
HARD = ("強め", "一杯")


def load_sessions(max_days: int | None = 60) -> pd.DataFrame:
    """max_days: そのレースの何日前までの本を残すか。None なら全部（自己ベストの『それまでの全ての坂路』の計算用）。"""
    t = pd.read_parquet(TL)
    t["horse_id"] = t["horse_id"].astype(str)
    t = t[(t["race_date"] > "2000-01-01") & (t["days_before"] > 0)].copy()
    if max_days is not None:
        t = t[t["days_before"] <= max_days].copy()
    t["int"] = t["leg_color"].map(INT_MAP)
    t["train_dt"] = pd.to_datetime(t["train_date"], errors="coerce")
    t["n_f"] = t[["t1", "t2", "t3", "t4", "t5", "t6"]].notna().sum(axis=1)
    t["acc"] = np.where(t["accel"].isna(), np.nan, (t["accel"] <= -1.0).astype(float))  # 終い 1F が中間の区間の平均より 1 秒以上速い = 加速ラップ（坂路の約 30%）
    t["has_info"] = t["int"].notna() | t["l1"].notna()  # 強度か時計がある本（時計も強度も無い 1 日前の角馬場・ダートなどは最終として使わない）
    t["_kind_ord"] = t["kind"].map({"slope": 0, "wood": 1, "poly": 2, "other": 3, "unknown": 4}).fillna(5)
    # 同じ日に複数本あるときも順序が決まるように、情報のある本・坂路・ウッド・ポリ・時計の速い順で並べる
    t = t.sort_values(["horse_id", "race_date", "days_before", "has_info", "_kind_ord", "l1", "train_dt"], ascending=[True, True, True, False, True, True, True], kind="stable").reset_index(drop=True)
    return t


def _pick(t: pd.DataFrame, where: pd.Series, by_abs: float | None = None, need_info: bool = False) -> pd.DataFrame:
    """(horse_id, race_date) ごとに 1 本選ぶ。by_abs が None なら days_before が最小、あれば |days_before − by_abs| が最小。
    need_info なら強度か時計のある本を優先し、無ければ（ある本が 1 本も無い馬は）days_before が最小の本に戻す。同点は t の並びの先頭（決定的）。"""
    x = t[where].copy()
    x["_k"] = x["days_before"] if by_abs is None else (x["days_before"] - by_abs).abs()
    x["_ord"] = np.arange(len(x))
    x["_noinfo"] = (~x["has_info"]).astype(int) if need_info else 0
    return x.sort_values(["horse_id", "race_date", "_noinfo", "_k", "_ord"], kind="stable").drop_duplicates(["horse_id", "race_date"])


def add_training(d: pd.DataFrame) -> pd.DataFrame:
    """d は horse_id・date（YYYY-MM-DD）・dt・year を持つ。ok_*（最終追い切り）を上書きし、w1_*・tl_* を足す。"""
    t_all = load_sessions(None)
    t = t_all[t_all["days_before"] <= 60].copy()
    key = ["horse_id", "race_date"]
    last = _pick(t, pd.Series(True, index=t.index), need_info=True)
    last = last.rename(columns={"course": "ok_course", "rider": "ok_rider", "rank": "ok_eval", "int": "ok_int", "evaluation": "ok_tanpyo", "cond": "ok_cond",
                                "train_dt": "ok_td", "n_f": "ok_n_f", "l1": "ok_l1", "acc": "ok_acc", "kind": "_kind", "days_before": "ok_days", "position": "ok_pos"})
    last["ok_slope"] = last["_kind"] == "slope"
    last["ok_wood"] = last["_kind"] == "wood"
    last["ok_poly"] = last["_kind"] == "poly"
    last["ok_rider"] = last["ok_rider"].where(last["ok_rider"] != "")
    last["ok_course"] = last["ok_course"].where(last["ok_course"] != "")
    last["ok_eval"] = last["ok_eval"].where(last["ok_eval"].isin(list("ABCD")))
    last["ok_tanpyo"] = last["ok_tanpyo"].where(last["ok_tanpyo"] != "")
    last["ok_cond"] = last["ok_cond"].where(last["ok_cond"] != "")
    last["ok_wd"] = last["ok_td"].dt.dayofweek.map(lambda i: WD[int(i)] if pd.notna(i) else None)
    okc = ["ok_course", "ok_rider", "ok_eval", "ok_int", "ok_tanpyo", "ok_cond", "ok_td", "ok_wd", "ok_n_f", "ok_l1", "ok_acc", "ok_slope", "ok_wood", "ok_poly", "ok_days", "ok_pos"]
    fd = last.set_index(key)["ok_days"].rename("_final_days")  # 最終の 1 本の days_before。1 週前・中間はこれより前の本だけ
    t = t.merge(fd, left_on=key, right_index=True, how="left")
    w1 = _pick(t, t["days_before"].between(5, 10) & (t["days_before"] > t["_final_days"]), by_abs=7, need_info=True)
    w1 = w1.rename(columns={"int": "w1_int", "rider": "w1_rider", "l1": "w1_l1", "days_before": "w1_days", "kind": "w1_kind"})
    w1["w1_rider"] = w1["w1_rider"].where(w1["w1_rider"] != "")
    # レースごとの本数・強め以上の本数・同じ日の複数本・自己ベスト
    t["hard"] = t["int"].isin(HARD)
    t["ippai"] = t["int"] == "一杯"
    t["in14"] = t["days_before"] <= 14
    cnt = t.groupby(key).agg(tl_n=("days_before", "size"), tl_first_days=("days_before", "max"), tl_hard=("hard", "sum"), tl_ippai=("ippai", "sum")).reset_index()
    c14 = t[t["in14"]].groupby(key).agg(tl_n14=("days_before", "size"), tl_hard14=("hard", "sum")).reset_index()
    day = t.groupby(key + ["train_dt"]).agg(_n=("days_before", "size"), _slope=("kind", lambda s: (s == "slope").sum()), _poly=("kind", lambda s: (s == "poly").sum())).reset_index()
    day["tl_double"] = (day["_slope"] >= 2) | ((day["_slope"] >= 1) & (day["_poly"] >= 1))
    dbl = day.groupby(key)["tl_double"].max().reset_index()
    # 自己ベスト: 坂路の終い 1F が、その馬のそれまでの（前の日までの。60 日より前の本も含む）全ての坂路の最速より速い（それまでに 3 本以上）。同じ日の本どうしは比べない
    s = t_all[(t_all["kind"] == "slope") & t_all["l1"].notna()].copy()
    day_min = s.groupby(["horse_id", "train_dt"]).agg(_dmin=("l1", "min"), _dn=("l1", "size")).reset_index().sort_values(["horse_id", "train_dt"])
    day_min["_prev_min"] = day_min.groupby("horse_id")["_dmin"].transform(lambda v: v.shift(1).cummin())
    day_min["_prev_n"] = day_min.groupby("horse_id")["_dn"].transform(lambda v: v.cumsum() - v)
    s = s.merge(day_min[["horse_id", "train_dt", "_prev_min", "_prev_n"]], on=["horse_id", "train_dt"], how="left")
    s["best"] = (s["l1"] < s["_prev_min"]) & (s["_prev_n"] >= 3)
    s = s.merge(fd, left_on=key, right_index=True, how="left")
    mid = s[(s["days_before"] > s["_final_days"]) & (s["days_before"] <= 60)]  # 中間 = 最終の 1 本より前の本（曜日・レースの日で変わる固定の日数では切らない）
    best = mid.groupby(key)["best"].max().rename("tl_best_mid").reset_index()
    hist = mid.groupby(key)["_prev_n"].max().ge(3).rename("tl_slope_hist").reset_index()  # 中間に坂路の本があり、その時点で坂路の前歴が 3 本以上ある（自己ベストを判定できる）馬
    out = d[["horse_id", "date"]].copy()
    out["horse_id"] = out["horse_id"].astype(str)
    out["race_date"] = out["date"].astype(str).str[:10]
    for df_, cols in ((last, okc), (w1, ["w1_int", "w1_rider", "w1_l1", "w1_days", "w1_kind"]), (cnt, ["tl_n", "tl_first_days", "tl_hard", "tl_ippai"]),
                      (c14, ["tl_n14", "tl_hard14"]), (dbl, ["tl_double"]), (best, ["tl_best_mid"]), (hist, ["tl_slope_hist"])):
        out = out.merge(df_[key + cols], on=key, how="left")  # 左の行の並びは保たれる
    assert len(out) == len(d)
    has = out["tl_n"].notna()  # 追い切りが載る馬で、14 日以内の本が無いのは NaN ではなく 0 本
    for c in ("tl_n14", "tl_hard14"):
        out.loc[has, c] = out.loc[has, c].fillna(0)
    out.index = d.index
    ok_year = pd.Series(d["year"].to_numpy() >= FIRST_YEAR, index=d.index)
    for c in out.columns:
        if c not in ("horse_id", "date", "race_date"):
            d[c] = out[c].where(ok_year)
    for c in ("ok_slope", "ok_wood", "ok_poly", "tl_double", "tl_best_mid", "tl_slope_hist"):
        d[c] = d[c].astype(object).where(d[c].notna(), False).astype(bool)
    d["tl_has"] = d["ok_days"].notna()
    d["tl_version"] = TL_VERSION
    # 前走時の最終追い切り（B15・B16・B23・B34 など）
    d = d.sort_values(["horse_id", "dt", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    for c in ["ok_int", "ok_slope", "ok_l1", "ok_eval", "ok_wood"]:
        d["p_" + c] = g[c].shift(1)
    d["p_ok_l1"] = pd.to_numeric(d["p_ok_l1"], errors="coerce")
    for c in ("p_ok_slope", "p_ok_wood"):
        d[c] = d[c].astype(object).where(d[c].notna(), False).astype(bool)
    from src.scripts.data.folklore_verify import derive_training_cols

    return derive_training_cols(d)
