"""特徴量ファクトリー G1: 馬自身の履歴・状態（些細なものも含めて as-of で網羅的に作る）。

どの特徴量も「当該レースより前の走だけ」から作る（同じ馬は 1 日 1 走なので、馬ごとに日付順に並べて 1 つずらす / 排他的な累積を取る）。
市場オッズ・人気は使わない。入力 ``runs`` は ``load_runs`` の形（1 走 = 1 行、JRA の完走のみ）。戻り値は runs と同じ index の表。
群（group）ごとに関数を持ち、``build_groups`` が {群名: 特徴量表} を返す。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

KEY = ["race_id", "horse_id"]
RUN_COLS = ["race_id", "date", "start_time", "venue_code", "surface", "distance", "direction", "grade", "race_class", "weather", "track_condition",
            "field_size", "round", "bracket_number", "horse_number", "horse_name", "horse_id", "sex_age", "jockey_id", "trainer_id", "jockey_weight",
            "weight", "weight_change", "finish_position", "margin", "passing_order", "last_3f", "time_sec"]
CLASS_LEVEL = {"新馬": 0.0, "未勝利": 0.0, "1勝": 1.0, "2勝": 2.0, "3勝": 3.0, "OP": 4.0, "L": 4.5, "G3": 5.0, "G2": 6.0, "G1": 7.0}
GOING = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
_FRAC = re.compile(r"^(?:(\d+)\.)?(\d+)/(\d+)$")
_WORD = {"ハナ": 0.05, "アタマ": 0.1, "クビ": 0.2, "大": 12.0}


# ───────────────────────── 入力
def load_runs(tables_dir: str | Path, years: list[int]) -> pd.DataFrame:
    import polars as pl

    parts = []
    for y in years:
        f = Path(tables_dir) / str(y) / "race_result_flat.parquet"
        sch = pl.read_parquet_schema(f)
        d = pl.read_parquet(f, columns=[c for c in RUN_COLS if c in sch])
        parts.append(d.with_columns(pl.col("distance").cast(pl.Float64)))
    r = pl.concat(parts, how="diagonal_relaxed").to_pandas()
    r = r[pd.to_numeric(r["venue_code"], errors="coerce").between(1, 10) & (r["finish_position"] > 0)].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"])
    return r.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)


def parse_margin(s) -> float:
    """着差の表記（'1/2' '1.1/4' 'クビ' 'ハナ' '大' '3' など）を馬身の数値に。勝ち馬（空）は 0。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).strip()
    if t in ("", "0"):
        return 0.0
    if t in _WORD:
        return _WORD[t]
    m = _FRAC.match(t)
    if m:
        return float(m.group(1) or 0) + int(m.group(2)) / int(m.group(3))
    try:
        return float(t)
    except ValueError:
        return np.nan


def prep(r: pd.DataFrame) -> pd.DataFrame:
    """1 走ごとの派生列（その走の結果から作る。特徴量にするときは必ず 1 走ずらして使う）。"""
    d = r.copy()
    n = d["field_size"].astype(float).clip(lower=2)
    fin = d["finish_position"].astype(float)
    d["fp"] = (fin - 1.0) / (n - 1.0)                               # 0=1 着, 1=最下位
    d["win"] = (fin == 1).astype(float)
    d["top3"] = (fin <= 3).astype(float)
    d["fin"] = fin
    d["margin_len"] = d["margin"].map(parse_margin)
    d.loc[d["fin"] == 1, "margin_len"] = 0.0
    po = d["passing_order"].astype(str).str.findall(r"\d+")
    first = po.map(lambda x: float(x[0]) if len(x) else np.nan)
    last = po.map(lambda x: float(x[-1]) if len(x) else np.nan)
    d["pos_first"] = (first - 1.0) / (n - 1.0)
    d["pos_last"] = (last - 1.0) / (n - 1.0)
    d["pos_gain"] = d["pos_last"] - d["fp"]                         # 最後のコーナーから着順までに押し上げた分
    d["pos_move"] = d["pos_first"] - d["pos_last"]                  # 道中の押し上げ
    l3f = pd.to_numeric(d["last_3f"], errors="coerce")
    d["l3f"] = l3f.where(l3f.between(30, 50))
    d["l3f_gap"] = d["l3f"] - d.groupby("race_id")["l3f"].transform("min")
    d["l3f_rank"] = (d.groupby("race_id")["l3f"].rank(method="min") - 1.0) / (n - 1.0)
    t = pd.to_numeric(d["time_sec"], errors="coerce")
    t = t.where(t > 30)
    d["time_gap200"] = (t - t.groupby(d["race_id"]).transform("min")) / (d["distance"] / 200.0)
    w = pd.to_numeric(d["weight"], errors="coerce")
    d["wt"] = w.where(w.between(300, 700))
    d["wt_chg"] = pd.to_numeric(d["weight_change"], errors="coerce").where(lambda s: s.abs() < 60)
    d["jw"] = pd.to_numeric(d["jockey_weight"], errors="coerce")
    d["cls"] = d["grade"].map(CLASS_LEVEL)
    d["going"] = d["track_condition"].map(GOING)
    d["dist"] = d["distance"].astype(float)
    d["dnum"] = d["date"].values.astype("datetime64[D]").astype(np.int64).astype(float)
    sa = d["sex_age"].astype(str)
    d["age"] = pd.to_numeric(sa.str[1:], errors="coerce")
    d["sex"] = sa.str[0]
    d["kai"] = pd.to_numeric(d["race_id"].str[6:8], errors="coerce")
    d["nichi"] = pd.to_numeric(d["race_id"].str[8:10], errors="coerce")
    d["month"] = d["date"].dt.month
    d["wday"] = d["date"].dt.weekday
    d["hour"] = pd.to_numeric(d["start_time"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
    d["nfield"] = n
    return d


# ───────────────────────── 部品
def _g(d: pd.DataFrame):
    return d.groupby("horse_id", sort=False)


def lag(d: pd.DataFrame, col: str, k: int = 1) -> pd.Series:
    return _g(d)[col].shift(k)


def lags(d: pd.DataFrame, col: str, k: int) -> np.ndarray:
    """(n, k): 1 走前, 2 走前, … の値。"""
    return np.column_stack([lag(d, col, i).to_numpy(float) for i in range(1, k + 1)])


def ewm_prior(d: pd.DataFrame, col: str, halflife: float) -> pd.Series:
    s = lag(d, col)
    tmp = pd.DataFrame({"h": d["horse_id"].to_numpy(), "v": s.to_numpy()}, index=d.index)
    out = tmp.groupby("h", sort=False)["v"].ewm(halflife=halflife, ignore_na=True).mean()
    return pd.Series(out.droplevel(0).reindex(d.index).to_numpy(), index=d.index)


def roll_prior(d: pd.DataFrame, col: str, win: int, how: str) -> pd.Series:
    s = lag(d, col)
    tmp = pd.DataFrame({"h": d["horse_id"].to_numpy(), "v": s.to_numpy()}, index=d.index)
    r = tmp.groupby("h", sort=False)["v"].rolling(win, min_periods=1)
    out = getattr(r, how)()
    return pd.Series(out.droplevel(0).reindex(d.index).to_numpy(), index=d.index)


def slope_last(M: np.ndarray) -> np.ndarray:
    """M の列 0 が 1 走前、列 k-1 が k 走前。古い→新しいの時系列としての傾き（新しいほど値が増えるなら正）。"""
    k = M.shape[1]
    t = -np.arange(1, k + 1, dtype=float)[None, :]
    m = np.isfinite(M)
    cnt = m.sum(1)
    tt = np.where(m, t, 0.0)
    tbar = tt.sum(1) / np.maximum(cnt, 1)
    yy = np.where(m, M, 0.0)
    ybar = yy.sum(1) / np.maximum(cnt, 1)
    dt = np.where(m, t - tbar[:, None], 0.0)
    num = (dt * np.where(m, M - ybar[:, None], 0.0)).sum(1)
    den = (dt ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num / den
    return np.where((cnt >= 3) & (den > 0), s, np.nan)


def excl_cum(d: pd.DataFrame, keys: list[str], col: str, as_count: bool = False) -> np.ndarray:
    """keys ごとの排他的累積（その行より前だけ）。d は馬・日付順に並んでいること。"""
    v = d[col].astype(float)
    ok = v.notna().astype(float)
    if as_count:
        return (ok.groupby([d[k] for k in keys], sort=False, dropna=False).cumsum() - ok).to_numpy()
    z = v.fillna(0.0)
    return (z.groupby([d[k] for k in keys], sort=False, dropna=False).cumsum() - z).to_numpy()


def seg_affinity(d: pd.DataFrame, seg: pd.Series, name: str, vals: tuple[str, ...] = ("fp",), k: float = 3.0) -> pd.DataFrame:
    """馬 × 区分の過去成績を、馬の全体平均へ縮約した『その区分での得意度』と経験数。区分は今回のレースの値（レース前に分かる）。"""
    tmp = d[["horse_id"] + list(vals)].copy()
    tmp["_seg"] = seg.astype(str).to_numpy()
    out = {}
    n_all = excl_cum(tmp.assign(_one=1.0), ["horse_id"], "_one", as_count=True)
    for v in vals:
        s_all = excl_cum(tmp, ["horse_id"], v)
        n_v = excl_cum(tmp, ["horse_id"], v, as_count=True)
        mean_all = np.where(n_v > 0, s_all / np.maximum(n_v, 1), np.nan)
        s_seg = excl_cum(tmp, ["horse_id", "_seg"], v)
        n_seg = excl_cum(tmp, ["horse_id", "_seg"], v, as_count=True)
        with np.errstate(invalid="ignore"):
            aff = (s_seg + k * np.nan_to_num(mean_all, nan=0.5)) / (n_seg + k) - np.nan_to_num(mean_all, nan=0.5)
        out[f"{name}_aff_{v}"] = np.where(n_v > 0, aff, np.nan)
        if v == vals[0]:
            out[f"{name}_n"] = n_seg
            out[f"{name}_share"] = np.where(n_all > 0, n_seg / np.maximum(n_all, 1), np.nan)
    return pd.DataFrame(out, index=d.index)


def _f(d: pd.DataFrame, **cols) -> pd.DataFrame:
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=d.index)


# ───────────────────────── 群
def g_rest(d: pd.DataFrame, **_) -> pd.DataFrame:
    gap = d["dnum"] - lag(d, "dnum")
    gap2 = lag(d, "dnum") - lag(d, "dnum", 2)
    gap3 = lag(d, "dnum", 2) - lag(d, "dnum", 3)
    G = np.column_stack([gap, gap2, gap3])
    gm = np.where(np.isfinite(G), G, np.nan)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rest_mean3, rest_max3 = np.nanmean(gm, 1), np.nanmax(np.where(np.isfinite(gm), gm, -np.inf), 1)
    rest_max3 = np.where(np.isfinite(rest_max3), rest_max3, np.nan)
    ng = gap.notna()
    out = _f(d, rest_prev=gap, rest_prev2=gap2, rest_prev3=gap3, rest_mean3=rest_mean3, rest_max3=rest_max3, rest_ratio=gap / gap2.where(gap2 > 0),
             rest_log=np.log1p(gap), rest_is_renso=(gap <= 8).astype(float).where(ng), rest_is_nakaichi=((gap > 8) & (gap <= 15)).astype(float).where(ng),
             rest_is_long90=(gap >= 90).astype(float).where(ng), rest_is_long180=(gap >= 180).astype(float).where(ng),
             rest_is_long365=(gap >= 365).astype(float).where(ng))
    d2 = d.assign(_gap=gap.to_numpy())
    for w in (30, 60, 90, 180, 365):
        # w 日以内の出走数（今回を除く）: 日付を重みに累積する代わりに、過去走の日付 dnum との差で数える
        out[f"rest_n_last{w}d"] = _count_within(d, w)
    out["rest_n_layoff90"] = excl_cum(d2.assign(_l=(d2["_gap"] >= 90).astype(float)), ["horse_id"], "_l")
    out["rest_since_layoff"] = _runs_since(d2["_gap"].ge(90).to_numpy(), d["horse_id"].to_numpy())
    out["rest_since_win"] = _days_since(d, "win")
    out["rest_since_top3"] = _days_since(d, "top3")
    return out


def _count_within(d: pd.DataFrame, w: int) -> np.ndarray:
    h = d["horse_id"].to_numpy()
    x = d["dnum"].to_numpy()
    out = np.zeros(len(d))
    start = 0
    for i in range(len(d)):
        if i and h[i] != h[i - 1]:
            start = i
        j = i - 1
        c = 0
        while j >= start and x[i] - x[j] <= w:
            c += 1
            j -= 1
        out[i] = c
    return out


def _runs_since(flag: np.ndarray, h: np.ndarray) -> np.ndarray:
    """直近の flag が True だった走から何走目か（今回が flag 走なら 0。flag は『今回の前の休み』の意味で、その走を 0 とする）。"""
    out = np.full(len(h), np.nan)
    cnt = np.nan
    for i in range(len(h)):
        if i == 0 or h[i] != h[i - 1]:
            cnt = np.nan
        if flag[i]:
            cnt = 0.0
        elif not np.isnan(cnt):
            cnt += 1.0
        out[i] = cnt
    return out


def _days_since(d: pd.DataFrame, col: str) -> np.ndarray:
    """直近の col==1 の走から今回までの日数（今回を除く）。"""
    h, x, v = d["horse_id"].to_numpy(), d["dnum"].to_numpy(), d[col].to_numpy(float)
    out = np.full(len(d), np.nan)
    last = np.nan
    for i in range(len(d)):
        if i == 0 or h[i] != h[i - 1]:
            last = np.nan
        out[i] = x[i] - last if not np.isnan(last) else np.nan
        if v[i] == 1.0:
            last = x[i]
    return out


def g_age(d: pd.DataFrame, birthday: pd.Series | None = None, **_) -> pd.DataFrame:
    out = _f(d, age_yr=d["age"], sex_male=(d["sex"] == "牡").astype(float), sex_female=(d["sex"] == "牝").astype(float), sex_gelding=(d["sex"] == "セ").astype(float))
    out["age_x_month"] = d["age"].to_numpy(float) + d["month"].to_numpy(float) / 12.0
    out["sex_x_season"] = (d["sex"] == "牝").astype(float).to_numpy() * np.sin(2 * np.pi * d["month"].to_numpy(float) / 12.0)
    first_age = d.groupby("horse_id", sort=False)["age"].transform("first")
    out["age_debut"] = first_age.to_numpy(float)
    out["age_since_debut"] = d["age"].to_numpy(float) - first_age.to_numpy(float)
    out["days_since_debut"] = (d["dnum"] - d.groupby("horse_id", sort=False)["dnum"].transform("first")).to_numpy(float)
    if birthday is not None:
        b = pd.to_datetime(birthday.reindex(d["horse_id"]).to_numpy(), errors="coerce")
        days = (d["date"].to_numpy() - b) / np.timedelta64(1, "D")
        out["age_days"] = np.asarray(days, float)
        out["born_month"] = np.asarray(pd.Series(b).dt.month, float)
        out["born_early"] = np.where(np.isnan(out["born_month"]), np.nan, (out["born_month"] <= 3).astype(float))
        out["born_doy"] = np.asarray(pd.Series(b).dt.dayofyear, float)
        bdays = np.asarray((pd.Series(b) - pd.Timestamp("1970-01-01")).dt.days, float)
        out["age_days_at_debut"] = d.groupby("horse_id", sort=False)["dnum"].transform("first").to_numpy(float) - bdays
    return out


def g_weight_level(d: pd.DataFrame, **_) -> pd.DataFrame:
    wt = d["wt"]
    prior_mean = excl_cum(d, ["horse_id"], "wt") / np.maximum(excl_cum(d, ["horse_id"], "wt", True), 1)
    n = excl_cum(d, ["horse_id"], "wt", True)
    prior_mean = np.where(n > 0, prior_mean, np.nan)
    return _f(d, wt_now=wt, wt_now_vs_mean=wt - prior_mean, wt_now_per_dist=wt / d["dist"] * 1000, jw_over_wt=d["jw"] / wt * 100, jw_now=d["jw"],
              wt_x_age=wt * d["age"] / 1000.0, wt_rel_field=wt - wt.groupby(d["race_id"]).transform("mean"), wt_rank_field=wt.groupby(d["race_id"]).rank(pct=True),
              jw_rel_field=d["jw"] - d["jw"].groupby(d["race_id"]).transform("mean"), wt_chg_now=d["wt_chg"], wt_chg_abs=d["wt_chg"].abs(),
              wt_chg_big=(d["wt_chg"].abs() >= 10).astype(float).where(d["wt_chg"].notna()), wt_chg_up=(d["wt_chg"] >= 6).astype(float).where(d["wt_chg"].notna()),
              wt_chg_down=(d["wt_chg"] <= -6).astype(float).where(d["wt_chg"].notna()))


def g_weight_dyn(d: pd.DataFrame, **_) -> pd.DataFrame:
    W = lags(d, "wt", 5)
    C = lags(d, "wt_chg", 3)
    J = lags(d, "jw", 3)
    cur = d["wt"].to_numpy(float)
    out = _f(d, wt_lag1=W[:, 0], wt_lag2=W[:, 1], wt_slope5=slope_last(W), wt_std5=np.nanstd(W, 1), wt_max5=np.nanmax(np.where(np.isfinite(W), W, -np.inf), 1),
             wt_min5=np.nanmin(np.where(np.isfinite(W), W, np.inf), 1), wt_now_vs_max5=cur - np.nanmax(np.where(np.isfinite(W), W, -np.inf), 1),
             wt_now_vs_min5=cur - np.nanmin(np.where(np.isfinite(W), W, np.inf), 1), wt_chg_lag1=C[:, 0], wt_chg_lag2=C[:, 1], wt_chg_lag3=C[:, 2],
             wt_chg_sum3=np.nansum(C, 1), wt_chg_abs_mean3=np.nanmean(np.abs(C), 1), jw_lag1=J[:, 0], jw_chg=d["jw"].to_numpy(float) - J[:, 0],
             jw_mean3=np.nanmean(J, 1))
    out = out.replace([np.inf, -np.inf], np.nan)
    # 勝った走の馬体重（最後の勝ち）との差
    wwin = d["wt"].where(d["win"] == 1)
    last_win_wt = _g(d.assign(_ww=wwin))["_ww"].transform(lambda s: s.shift(1).ffill())
    out["wt_now_vs_lastwin"] = cur - last_win_wt.to_numpy(float)
    return out


def g_change(d: pd.DataFrame, **_) -> pd.DataFrame:
    def chg(col):
        p = lag(d, col)
        return (d[col].astype(str).to_numpy() != p.astype(str).to_numpy()).astype(float) * p.notna().to_numpy()
    first = lag(d, "dnum").isna().to_numpy()
    out = _f(d, chg_jockey=chg("jockey_id"), chg_trainer=chg("trainer_id"), chg_surface=chg("surface"), chg_venue=chg("venue_code"), chg_direction=chg("direction"),
             chg_going=chg("track_condition"), chg_weather=chg("weather"), chg_cls=d["cls"] - lag(d, "cls"), chg_cls_up=(d["cls"] > lag(d, "cls")).astype(float),
             chg_cls_down=(d["cls"] < lag(d, "cls")).astype(float), chg_dist=d["dist"] - lag(d, "dist"), chg_dist_abs=(d["dist"] - lag(d, "dist")).abs(),
             chg_dist_up=(d["dist"] > lag(d, "dist")).astype(float), chg_dist_down=(d["dist"] < lag(d, "dist")).astype(float),
             chg_field=d["nfield"] - lag(d, "nfield"), chg_bracket=d["bracket_number"] - lag(d, "bracket_number"), chg_hn=d["horse_number"] - lag(d, "horse_number"),
             chg_round=d["round"] - lag(d, "round"), chg_hour=d["hour"] - lag(d, "hour"), chg_going_num=d["going"] - lag(d, "going"))
    for c in out.columns:
        out.loc[first, c] = np.nan
    # 乗り替わり × 前走の結果、直近 5 走の騎手の種類数
    out["chg_jockey_x_prevfp"] = out["chg_jockey"] * lag(d, "fp").to_numpy(float)
    J5 = np.column_stack([lag(d, "jockey_id", i).astype(str).to_numpy() for i in range(1, 6)])
    out["n_jockeys_last5"] = [len({x for x in row if x not in ("nan", "None")}) for row in J5]
    out["chg_trainer_x_n"] = out["chg_trainer"] * excl_cum(d.assign(_o=1.0), ["horse_id"], "_o", True)
    return out


def g_career(d: pd.DataFrame, **_) -> pd.DataFrame:
    one = d.assign(_o=1.0)
    n = excl_cum(one, ["horse_id"], "_o", True)
    wins = excl_cum(d, ["horse_id"], "win")
    top3 = excl_cum(d, ["horse_id"], "top3")
    out = _f(d, car_n=n, car_wins=wins, car_top3=top3, car_win_rate=wins / np.maximum(n, 1), car_top3_rate=top3 / np.maximum(n, 1), car_is_debut=(n == 0).astype(float),
             car_is_maiden=(wins == 0).astype(float), car_n_after_first_win=np.where(wins > 0, n - 1, np.nan), car_log_n=np.log1p(n))
    out["car_win_x_n"] = wins * np.log1p(n)
    h = d["horse_id"].to_numpy()
    out["car_loss_streak"] = _streak(d["win"].to_numpy(float), h, target=0.0)
    out["car_win_streak"] = _streak(d["win"].to_numpy(float), h, target=1.0)
    out["car_top3_streak"] = _streak(d["top3"].to_numpy(float), h, target=1.0)
    out["car_out3_streak"] = _streak(d["top3"].to_numpy(float), h, target=0.0)
    out["car_n_same_cls"] = _same_key_count(d, ["horse_id", "cls"])
    out["car_n_same_cls_win"] = _same_key_sum(d, ["horse_id", "cls"], "win")
    out["car_max_cls"] = _g(d.assign(_c=d["cls"]))["_c"].transform(lambda s: s.shift(1).cummax()).to_numpy(float)
    out["car_cls_vs_max"] = d["cls"].to_numpy(float) - out["car_max_cls"].to_numpy(float)
    out["car_best_fp"] = _g(d.assign(_c=d["fp"]))["_c"].transform(lambda s: s.shift(1).cummin()).to_numpy(float)
    out["car_worst_fp"] = _g(d.assign(_c=d["fp"]))["_c"].transform(lambda s: s.shift(1).cummax()).to_numpy(float)
    return out


def _same_key_count(d: pd.DataFrame, keys: list[str]) -> np.ndarray:
    return d.assign(_o=1.0).groupby([d[k].astype(str) for k in keys], sort=False, dropna=False)["_o"].cumcount().to_numpy(float)


def _same_key_sum(d: pd.DataFrame, keys: list[str], col: str) -> np.ndarray:
    v = d[col].fillna(0.0).astype(float)
    return (v.groupby([d[k].astype(str) for k in keys], sort=False, dropna=False).cumsum() - v).to_numpy()


def _streak(v: np.ndarray, h: np.ndarray, target: float) -> np.ndarray:
    out = np.zeros(len(v))
    cur = 0
    for i in range(len(v)):
        if i == 0 or h[i] != h[i - 1]:
            cur = 0
        out[i] = cur
        cur = cur + 1 if v[i] == target else 0
    return out


def g_experience(d: pd.DataFrame, **_) -> pd.DataFrame:
    """同条件の経験数と初めて挑む条件のフラグ。"""
    out = {}
    d = d.copy()
    d["dist_rounded"] = (d["dist"] / 200).round() * 200
    d["dist_band5"] = pd.cut(d["dist"], [0, 1300, 1500, 1900, 2400, 99999], labels=False)
    d["course_key"] = d["venue_code"].astype(str) + "|" + d["surface"].astype(str) + "|" + d["dist"].fillna(0).astype(int).astype(str)
    d["vs"] = d["venue_code"].astype(str) + "|" + d["surface"].astype(str)
    d["sd"] = d["surface"].astype(str) + "|" + d["dist_band5"].astype(str)
    d["field_bucket"] = pd.cut(d["nfield"], [0, 9, 13, 99], labels=False)
    d["one"] = 1.0
    for nm, keys in {"venue": ["venue_code"], "surface": ["surface"], "dist_exact": ["dist"], "dist_round": ["dist_rounded"], "dist_band": ["dist_band5"], "course": ["course_key"],
                     "direction": ["direction"], "going": ["track_condition"], "venue_surface": ["vs"], "surf_dist": ["sd"], "cls": ["cls"], "field": ["field_bucket"],
                     "weather": ["weather"], "month": ["month"], "hour": ["hour"], "kai": ["kai"]}.items():
        c = _same_key_count(d, ["horse_id"] + keys)
        out[f"exp_n_{nm}"] = c
        out[f"exp_first_{nm}"] = (c == 0).astype(float)
        w = _same_key_sum(d, ["horse_id"] + keys, "win")
        out[f"exp_win_{nm}"] = w
    return pd.DataFrame(out, index=d.index)


def _lagblock(d: pd.DataFrame, cols: dict[str, int]) -> pd.DataFrame:
    out = {}
    for c, k in cols.items():
        L = lags(d, c, k)
        for i in range(k):
            out[f"lag{i + 1}_{c}"] = L[:, i]
    return pd.DataFrame(out, index=d.index)


def g_lags_fin(d: pd.DataFrame, **_) -> pd.DataFrame:
    return _lagblock(d, {"fin": 5, "fp": 5, "nfield": 3})


def g_lags_margin(d: pd.DataFrame, **_) -> pd.DataFrame:
    out = _lagblock(d, {"margin_len": 5, "l3f_gap": 5, "l3f_rank": 3, "time_gap200": 5})
    M = lags(d, "margin_len", 5)
    out["margin_mean5"] = np.nanmean(M, 1)
    out["margin_min5"] = np.nanmin(np.where(np.isfinite(M), M, np.inf), 1)
    out["margin_max5"] = np.nanmax(np.where(np.isfinite(M), M, -np.inf), 1)
    out["margin_n_close"] = np.nansum(M <= 1.0, 1) / np.maximum(np.isfinite(M).sum(1), 1)
    return out.replace([np.inf, -np.inf], np.nan)


def g_lags_pos(d: pd.DataFrame, **_) -> pd.DataFrame:
    out = _lagblock(d, {"pos_first": 5, "pos_last": 5, "pos_gain": 5, "pos_move": 3})
    P = lags(d, "pos_first", 5)
    out["pos_first_mean5"] = np.nanmean(P, 1)
    out["pos_first_std5"] = np.nanstd(P, 1)
    out["pos_front_rate5"] = np.nansum(P <= 0.2, 1) / np.maximum(np.isfinite(P).sum(1), 1)
    G = lags(d, "pos_gain", 5)
    out["pos_gain_mean5"] = np.nanmean(G, 1)
    out["pos_gain_pos_rate5"] = np.nansum(G > 0, 1) / np.maximum(np.isfinite(G).sum(1), 1)
    return out


def g_seq_stats(d: pd.DataFrame, **_) -> pd.DataFrame:
    out = {}
    for c in ("fp", "margin_len", "l3f_gap", "pos_first", "pos_gain", "time_gap200", "l3f_rank"):
        M = lags(d, c, 5)
        out[f"seq_slope5_{c}"] = slope_last(M)
        out[f"seq_std5_{c}"] = np.nanstd(M, 1)
        out[f"seq_best5_{c}"] = np.nanmin(np.where(np.isfinite(M), M, np.inf), 1)
        out[f"seq_worst5_{c}"] = np.nanmax(np.where(np.isfinite(M), M, -np.inf), 1)
        out[f"seq_last_vs_mean5_{c}"] = M[:, 0] - np.nanmean(M, 1)
        for hl in (1.0, 3.0, 8.0):
            out[f"seq_ewm{hl:g}_{c}"] = ewm_prior(d, c, hl).to_numpy(float)
    out["seq_top3_in5"] = np.nansum(lags(d, "top3", 5), 1)
    out["seq_win_in5"] = np.nansum(lags(d, "win", 5), 1)
    out["seq_top3_in3"] = np.nansum(lags(d, "top3", 3), 1)
    return pd.DataFrame(out, index=d.index).replace([np.inf, -np.inf], np.nan)


def g_aff_calendar(d: pd.DataFrame, **_) -> pd.DataFrame:
    segs = {"month": d["month"], "season": ((d["month"] % 12) // 3), "wday": d["wday"], "kai": d["kai"], "nichi": d["nichi"], "hour": d["hour"],
            "round_b": pd.cut(d["round"], [0, 4, 8, 12], labels=False), "weather": d["weather"]}
    return pd.concat([seg_affinity(d, s, f"affc_{k}", ("fp", "win")) for k, s in segs.items()], axis=1)


def g_aff_field(d: pd.DataFrame, **_) -> pd.DataFrame:
    n = d["nfield"]
    segs = {"field_b": pd.cut(n, [0, 9, 13, 99], labels=False), "bracket_z": pd.cut(d["bracket_number"], [0, 2, 6, 8], labels=False),
            "hn_rel": pd.cut(d["horse_number"] / n, [0, 0.34, 0.67, 1.01], labels=False), "going_full": d["track_condition"], "dir": d["direction"],
            "dist_round": (d["dist"] / 200).round(), "venue_surf": d["venue_code"].astype(str) + d["surface"].astype(str)}
    return pd.concat([seg_affinity(d, s, f"afff_{k}", ("fp", "win")) for k, s in segs.items()], axis=1)


def g_aff_state(d: pd.DataFrame, **_) -> pd.DataFrame:
    gap = d["dnum"] - lag(d, "dnum")
    segs = {"rest_b": pd.cut(gap, [-1, 14, 35, 70, 140, 100000], labels=False), "wt_b": pd.cut(d["wt"], [0, 440, 480, 520, 1000], labels=False),
            "wtchg_b": pd.cut(d["wt_chg"], [-100, -6, -1, 5, 100], labels=False), "jw_b": pd.cut(d["jw"], [0, 52, 54, 56, 58, 100], labels=False),
            "cls_prev": pd.Series(np.sign(d["cls"] - lag(d, "cls")), index=d.index)}
    return pd.concat([seg_affinity(d, s, f"affs_{k}", ("fp", "win")) for k, s in segs.items()], axis=1)


def g_calendar(d: pd.DataFrame, **_) -> pd.DataFrame:
    n, hn, br = d["nfield"], d["horse_number"], d["bracket_number"]
    return _f(d, cal_wday=d["wday"], cal_kai=d["kai"], cal_nichi=d["nichi"], cal_hour=d["hour"], cal_month=d["month"], cal_round=d["round"],
              cal_round_late=(d["round"] >= 9).astype(float), cal_main=(d["round"] == 11).astype(float), cal_last=(d["round"] == 12).astype(float),
              hn_rel=hn / n, hn_odd=(hn % 2).astype(float), hn_is1=(hn == 1).astype(float), hn_is_last=(hn == n).astype(float), hn_from_end=n - hn,
              br_rel=br / 8.0, br_inner=(br <= 2).astype(float), br_outer=(br >= 7).astype(float), br_x_n=br * n / 100.0,
              hn_in_br=hn - (br - 1) * n / 8.0, field_small=(n <= 8).astype(float), field_big=(n >= 16).astype(float), field_full=(n == 18).astype(float),
              dow_sin=np.sin(2 * np.pi * d["wday"] / 7.0), dow_cos=np.cos(2 * np.pi * d["wday"] / 7.0))


def g_name(d: pd.DataFrame, **_) -> pd.DataFrame:
    nm = d["horse_name"].fillna("").astype(str)
    return _f(d, name_len=nm.str.len(), name_has_long=nm.str.contains("ー").astype(float), name_ends_long=nm.str.endswith("ー").astype(float),
              name_n_small=nm.str.count("[ァィゥェォッャュョ]"), name_is_ascii=nm.str.match(r"^[A-Za-z0-9 ]+$").astype(float),
              name_has_kanji=nm.str.contains("[一-龥]").astype(float), name_first_code=nm.str[0].map(lambda c: ord(c) % 997 if c else 0))


def g_timedev(d: pd.DataFrame, **_) -> pd.DataFrame:
    """同じ (場・面・距離・馬場) の過去の時計の平均（その日より前）に対する、その走の走破時計の偏差を馬ごとに as-of 集計。"""
    t = pd.to_numeric(d["time_sec"], errors="coerce").where(lambda s: s > 30)
    key = (d["venue_code"].astype(str) + "|" + d["surface"].astype(str) + "|" + d["dist"].fillna(0).astype(int).astype(str) + "|" + d["track_condition"].astype(str))
    df = pd.DataFrame({"key": key.to_numpy(), "t": t.to_numpy(), "dnum": d["dnum"].to_numpy()})
    # 日ごとにまとめて、その日より前の累積平均を引く
    daily = df.groupby(["key", "dnum"])["t"].agg(["sum", "count"]).reset_index().sort_values(["key", "dnum"])
    daily["cs"] = daily.groupby("key")["sum"].cumsum() - daily["sum"]
    daily["cn"] = daily.groupby("key")["count"].cumsum() - daily["count"]
    m = df.merge(daily[["key", "dnum", "cs", "cn"]], on=["key", "dnum"], how="left")
    norm = np.where(m["cn"] >= 20, m["cs"] / np.maximum(m["cn"], 1), np.nan)
    dev = (m["t"].to_numpy() - norm) / (d["dist"].to_numpy() / 200.0)
    dd = d.assign(td=dev)
    out = {"td_ewm1": ewm_prior(dd, "td", 1.0).to_numpy(), "td_ewm3": ewm_prior(dd, "td", 3.0).to_numpy(), "td_lag1": lag(dd, "td").to_numpy(),
           "td_best5": roll_prior(dd, "td", 5, "min").to_numpy(), "td_mean5": roll_prior(dd, "td", 5, "mean").to_numpy(), "td_std5": roll_prior(dd, "td", 5, "std").to_numpy()}
    out["td_slope5"] = slope_last(lags(dd, "td", 5))
    return pd.DataFrame(out, index=d.index)


def g_pair(d: pd.DataFrame, **_) -> pd.DataFrame:
    """馬 × 騎手 / 調教師の過去（ID キー）。"""
    out = {}
    for nm, keys in {"hj": ["horse_id", "jockey_id"], "ht": ["horse_id", "trainer_id"], "hjv": ["horse_id", "jockey_id", "venue_code"]}.items():
        n = _same_key_count(d, keys)
        w = _same_key_sum(d, keys, "win")
        t3 = _same_key_sum(d, keys, "top3")
        out[f"pair_{nm}_n"] = n
        out[f"pair_{nm}_win"] = w
        out[f"pair_{nm}_win_rate"] = (w + 0.08 * 3) / (n + 3)
        out[f"pair_{nm}_top3_rate"] = (t3 + 0.25 * 3) / (n + 3)
        out[f"pair_{nm}_first"] = (n == 0).astype(float)
    # 前走と同じ騎手に連続で乗っている回数
    same = (d["jockey_id"].astype(str).to_numpy() == lag(d, "jockey_id").astype(str).to_numpy()).astype(float)
    out["pair_jockey_streak"] = _streak_true(same, d["horse_id"].to_numpy())
    return pd.DataFrame(out, index=d.index)


def _streak_true(v: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = np.zeros(len(v))
    cur = 0
    for i in range(len(v)):
        if i == 0 or h[i] != h[i - 1]:
            cur = 0
        cur = cur + 1 if v[i] == 1.0 else 0
        out[i] = cur
    return out


def g_attrs(d: pd.DataFrame, attrs: pd.DataFrame | None = None, frozen: dict | None = None, **_) -> pd.DataFrame:
    """馬の固定属性: 産地（レース前に変わらない・全馬で取れる）。毛色は生まれ年によって取れていない（最近生まれの馬だけ 100%）ので使わない。
    産地は頻度順の整数コードと、産地ごとの過去の成績（その日より前のレースだけ・縮約付き）。"""
    if attrs is None or "birthplace" not in attrs.columns:
        return pd.DataFrame(index=d.index)
    bp = attrs["birthplace"].reindex(d["horse_id"]).fillna("").astype(str).to_numpy()
    top = frozen["bp_top"] if frozen else pd.Series(bp).value_counts().index[:15]      # frozen: 推論では学習時（2020〜2025）の産地コードと平均を固定して使う
    code = {v: i for i, v in enumerate(top) if v != ""}
    out = {"attr_birthplace": np.array([code.get(v, np.nan if v == "" else 99) for v in bp], float)}
    df = pd.DataFrame({"key": bp, "fp": d["fp"].to_numpy(), "win": d["win"].to_numpy(), "dnum": d["dnum"].to_numpy()})
    daily = df.groupby(["key", "dnum"]).agg(sf=("fp", "sum"), sw=("win", "sum"), n=("fp", "count")).reset_index().sort_values(["key", "dnum"])
    for c in ("sf", "sw", "n"):
        daily[c] = daily.groupby("key")[c].cumsum() - daily[c]
    m = df.merge(daily, on=["key", "dnum"], how="left")
    n = m["n"].to_numpy(float)
    gm_fp, gm_w = (float(frozen["gm_fp"]), float(frozen["gm_w"])) if frozen else (float(np.nanmean(df["fp"])), float(np.nanmean(df["win"])))
    out["attr_bp_n"] = n
    out["attr_bp_fp"] = (m["sf"].to_numpy(float) + 200 * gm_fp) / (n + 200)
    out["attr_bp_win"] = (m["sw"].to_numpy(float) + 200 * gm_w) / (n + 200)
    r = pd.DataFrame(out, index=d.index)
    r.loc[bp == "", ["attr_bp_n", "attr_bp_fp", "attr_bp_win"]] = np.nan
    return r


GROUPS: dict[str, Callable] = {
    "rest": g_rest, "age": g_age, "weight_level": g_weight_level, "weight_dyn": g_weight_dyn, "change": g_change, "career": g_career, "experience": g_experience,
    "lags_fin": g_lags_fin, "lags_margin": g_lags_margin, "lags_pos": g_lags_pos, "seq_stats": g_seq_stats, "aff_calendar": g_aff_calendar,
    "aff_field": g_aff_field, "aff_state": g_aff_state, "calendar": g_calendar, "name": g_name, "timedev": g_timedev, "pair": g_pair, "attrs": g_attrs,
}


def attrs_constants(runs: pd.DataFrame, attrs: pd.DataFrame) -> dict:
    """g_attrs が全行から決める定数（産地コード上位 15 と平均着順・勝率）。学習時の runs（2020〜2025）から作って固定する。"""
    d = prep(runs)
    bp = attrs["birthplace"].reindex(d["horse_id"]).fillna("").astype(str).to_numpy()
    return {"bp_top": [str(x) for x in pd.Series(bp).value_counts().index[:15]], "gm_fp": float(np.nanmean(d["fp"])), "gm_w": float(np.nanmean(d["win"]))}


def build_groups(runs: pd.DataFrame, birthday: pd.Series | None = None, attrs: pd.DataFrame | None = None, only: list[str] | None = None,
                 frozen: dict | None = None) -> dict[str, pd.DataFrame]:
    d = prep(runs)
    out = {}
    for name, fn in GROUPS.items():
        if only and name not in only:
            continue
        f = fn(d, birthday=birthday, attrs=attrs, frozen=frozen)
        if f.shape[1]:
            out[name] = f.astype("float32").assign(race_id=d["race_id"].to_numpy(), horse_id=d["horse_id"].to_numpy())
    return out
