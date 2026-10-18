"""tp-33: 輸送（厩舎の拠点→開催場）と滞在（追い切りの場所）の特徴量。すべて「そのレース日より前」の情報だけで作る（as-of）。

輸送: 厩舎の拠点（栗東/美浦。調教師ごとに、追い切りの場所の多数派から決める）と開催場の組み合わせ、概算の道のり、前走の開催場からの移動。
滞在: レース前 1〜4 週の追い切りが、拠点ではない場所（遠征先・もう一方のトレセン・開催場の近く）で行われたか。
追い切りの場所は horse_training の「コース」（栗坂・CW・南W・函W・札ダ・小ダ 等）から読む。コースが読めない（空・地方競馬場など）ものは unknown。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
COORD = {"01": (43.08, 141.33), "02": (41.77, 140.72), "03": (37.76, 140.46), "04": (37.96, 139.20), "05": (35.66, 139.48), "06": (35.72, 139.96),
         "07": (35.06, 137.00), "08": (34.91, 135.73), "09": (34.78, 135.36), "10": (33.85, 130.88),
         "KU": (35.03, 136.03), "MI": (36.00, 140.33)}               # 栗東・美浦トレセン
ROAD = 1.25                                                          # 直線距離 → 道のりの概算
SEA = {"01", "02"}                                                   # 北海道（船・空路を伴う）

# 追い切りのコース名 → 場所
_PREFIX = {"栗": "KU", "美": "MI", "南": "MI", "北": "MI", "札": "01", "函": "02", "福": "03", "新": "04", "東": "05", "中": "06", "京": "08", "阪": "09", "小": "10", "門": "NAR"}
_EXACT = {"CW": "KU", "DP": "KU", "飛芝": "KU", "飛ダ": "KU", "障芝": "UNK", "札幌": "01", "福島": "03", "小倉": "10"}
NAR = {"佐賀", "船橋", "笠松", "西脇", "園田", "小林", "名古", "水沢", "大井", "門別", "小向", "野田", "川崎", "弥富", "盛岡", "高知", "金沢", "浦和", "白井"}
CENTER_LOCS = ("KU", "MI")


def course_loc(course: str | None) -> str:
    """追い切りのコース名から場所を返す。KU 栗東 / MI 美浦 / 01〜10 開催場 / NAR 地方 / UNK 不明。"""
    c = (course or "").replace("一番時計", "").strip()
    if not c:
        return "UNK"
    if c in _EXACT:
        return _EXACT[c]
    if c in NAR:
        return "NAR"
    if c[0] in _PREFIX:
        return _PREFIX[c[0]]
    return "UNK"


def km_between(a: str, b: str) -> float:
    la, lb = COORD.get(a), COORD.get(b)
    if la is None or lb is None:
        return float("nan")
    p1, p2 = math.radians(la[0]), math.radians(lb[0])
    dl = math.radians(lb[1] - la[1])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * 6371.0 * math.asin(math.sqrt(x)) * ROAD)


KM_HOME = {(h, v): km_between(h, v) for h in CENTER_LOCS for v in VENUES}
KM_VENUE = {(a, b): km_between(a, b) for a in VENUES for b in VENUES}


def tier_of_km(km: float, venue: str) -> int:
    """0 ホーム（~120km）/ 1 近距離（~350km）/ 2 中〜長距離（~700km）/ 3 遠方（それ以上、または北海道）。"""
    if not np.isfinite(km):
        return -1
    if venue in SEA:
        return 3
    return 0 if km <= 120 else 1 if km <= 350 else 2 if km <= 700 else 3


def trainer_home(laps: pd.DataFrame, races: pd.DataFrame) -> pd.Series:
    """調教師 ID → 拠点（KU/MI）。その調教師の馬の追い切りのうち、栗東/美浦の施設で行われたものの多数派。
    拠点は調教師の不変の属性なので全期間で数える（結果は使わない）。"""
    r = races[["horse_id", "date", "trainer_id"]].copy()
    r["horse_id"], r["date"] = r["horse_id"].astype(str), r["date"].astype(str)
    x = laps[["horse_id", "race_date", "loc"]].copy()
    x["horse_id"], x["race_date"] = x["horse_id"].astype(str), x["race_date"].astype(str)
    x = x[x["loc"].isin(CENTER_LOCS)].merge(r.rename(columns={"date": "race_date"}), on=["horse_id", "race_date"], how="inner")
    g = x.groupby(["trainer_id", "loc"]).size().unstack(fill_value=0)
    for c in CENTER_LOCS:
        if c not in g.columns:
            g[c] = 0
    home = np.where(g["KU"] >= g["MI"], "KU", "MI")
    return pd.Series(home, index=g.index, name="home")


def prep_laps(laps: pd.DataFrame) -> pd.DataFrame:
    x = laps[["horse_id", "race_date", "train_date", "course", "days_before"]].copy()
    x["horse_id"], x["race_date"] = x["horse_id"].astype(str), x["race_date"].astype(str)
    x["loc"] = x["course"].map(course_loc)
    return x


def _asof_rate(df: pd.DataFrame, key: str, flag: str, hit: str, prior_n: float = 20.0, name: str = "x") -> pd.DataFrame:
    """key 別の、日付が厳密に前の『flag==1 の走』の hit 率（経験ベイズ縮約）と件数。縮約先は同じ条件の、日付が前の全体の率（as-of。結果の先読みなし）。"""
    d = df[[key, "date", flag, hit]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["_i"] = np.arange(len(d))
    d["_n"] = (d[flag] == 1).astype(float)
    d["_k"] = d["_n"] * d[hit].astype(float)
    d["_all"] = "all"

    def asof(k: str) -> pd.DataFrame:
        daily = d.groupby([k, "date"], sort=True)[["_n", "_k"]].sum().reset_index()
        daily[["_n", "_k"]] = daily.groupby(k)[["_n", "_k"]].cumsum()
        daily = daily.sort_values("date")
        left = d[[k, "date", "_i"]].sort_values("date")
        return pd.merge_asof(left, daily, on="date", by=k, direction="backward", allow_exact_matches=False).sort_values("_i")

    g = asof("_all")
    p0 = np.where(g["_n"].fillna(0).to_numpy() > 0, g["_k"].fillna(0).to_numpy() / np.maximum(g["_n"].fillna(0).to_numpy(), 1), 0.22)
    m = asof(key)
    n = m["_n"].fillna(0).to_numpy()
    k = m["_k"].fillna(0).to_numpy()
    return pd.DataFrame({f"{name}_rate": (k + prior_n * p0) / (n + prior_n), f"{name}_n": n, f"{name}_prior": p0}, index=df.index)


def build(df: pd.DataFrame, laps: pd.DataFrame, home_map: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """df: race_id, horse_id, date, venue_code, trainer_id, finish_position を持つ行（RangeIndex）。laps: training_laps（horse_id, race_date, train_date, course, days_before）。
    戻り値: 群名 → 特徴量表（df と同じ index）。finish_position は『日付が厳密に前』の走の結果としてのみ使う。"""
    assert isinstance(df.index, pd.RangeIndex)
    d = df[["race_id", "horse_id", "date", "venue_code", "trainer_id", "finish_position"]].copy()
    d["horse_id"], d["date"] = d["horse_id"].astype(str), d["date"].astype(str).str.slice(0, 10)
    d["venue_code"] = d["venue_code"].astype(str).str.zfill(2)
    d["trainer_id"] = d["trainer_id"].astype(str)
    L = prep_laps(laps)
    home_map = home_map if home_map is not None else trainer_home(L, d)
    d["home"] = d["trainer_id"].map(home_map)
    ok = d["home"].notna() & d["venue_code"].isin(list(VENUES))
    km = np.array([KM_HOME.get((h, v), np.nan) for h, v in zip(d["home"], d["venue_code"])])
    tier = np.array([tier_of_km(k, v) for k, v in zip(km, d["venue_code"])])
    d["tier"] = np.where(ok, tier, -1)
    out_tier = pd.DataFrame({
        "tt_home_ku": np.where(d["home"].eq("KU"), 1.0, np.where(d["home"].eq("MI"), 0.0, np.nan)),
        "tt_km": np.where(ok, km, np.nan), "tt_tier": np.where(ok, tier, np.nan).astype(float),
        "tt_sea": d["venue_code"].isin(SEA).astype(float), "tt_far": np.where(ok, (tier >= 2).astype(float), np.nan),
    }, index=d.index)
    out_tier["tt_route"] = (d["home"].fillna("UNK") + ">" + d["venue_code"]).astype("category")

    # 前走からの移動（日付が前の走のうち直近）
    s = d.sort_values(["horse_id", "date", "race_id"], kind="stable")
    g = s.groupby("horse_id", sort=False)
    prev_v = g["venue_code"].shift(1)
    prev_tier = g["tier"].shift(1)
    prev_date = pd.to_datetime(g["date"].shift(1))
    cur_date = pd.to_datetime(s["date"])
    mv = pd.DataFrame(index=s.index)
    mv["tt_prev_km"] = [KM_VENUE.get((a, b), np.nan) if isinstance(a, str) else np.nan for a, b in zip(prev_v, s["venue_code"])]
    mv["tt_prev_same_venue"] = np.where(prev_v.isna(), np.nan, (prev_v == s["venue_code"]).astype(float))
    mv["tt_prev_tier"] = prev_tier.where(prev_tier >= 0)
    mv["tt_tier_change"] = s["tier"].where(s["tier"] >= 0) - mv["tt_prev_tier"]
    mv["tt_days_prev"] = (cur_date - prev_date).dt.days.astype(float)
    s["_far"] = (s["tier"] >= 2).astype(float)
    g = s.groupby("horse_id", sort=False)
    mv["tt_n_far_prev"] = g["_far"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    mv["tt_first_far"] = ((s["tier"] >= 2) & (mv["tt_n_far_prev"] == 0)).astype(float)
    s["_vseen"] = s.groupby(["horse_id", "venue_code"], sort=False).cumcount()
    mv["tt_first_at_venue"] = (s["_vseen"] == 0).astype(float)
    far_prev = g["_far"].shift(1).fillna(0)
    grp = (far_prev == 0).groupby(s["horse_id"]).cumsum()
    mv["tt_far_streak"] = far_prev.groupby([s["horse_id"], grp]).cumsum()          # 直前から遡って連続した遠征走の数
    out_move = mv.reindex(d.index)

    # 滞在（レース前 28 日の追い切りの場所）
    d["_k"] = d["horse_id"] + "|" + d["date"]
    W = L[(L["days_before"] >= 1) & (L["days_before"] <= 28)].copy()
    W["_k"] = W["horse_id"] + "|" + W["race_date"]
    W = W.merge(d[["_k", "home", "venue_code"]].drop_duplicates("_k"), on="_k", how="inner")
    W["away"] = (W["loc"] != W["home"]) & W["loc"].isin(list(CENTER_LOCS) + list(VENUES))
    W["known"] = W["loc"] != "UNK"
    W["in21"] = W["days_before"] <= 21
    W["at_venue"] = W["loc"] == W["venue_code"]
    # もう一方のトレセン、または開催場の近い方のトレセン以外で行った追い切り
    W["away21"] = (W["away"] & W["in21"]).astype(float)
    W["away_f"] = W["away"].astype(float)
    W["atv_f"] = W["at_venue"].astype(float)
    W["dba"] = W["days_before"].where(W["away"])
    agg = W.groupby("_k").agg(n_work=("loc", "size"), n_known=("known", "sum"), n_away21=("away21", "sum"), n_away28=("away_f", "sum"),
                              n_at_venue=("atv_f", "sum"), first_away=("dba", "max"))
    last = W.sort_values("days_before").groupby("_k").first()
    agg["last_away"] = last["away"].astype(float)
    agg["last_loc"] = last["loc"]
    st = d[["_k"]].merge(agg, left_on="_k", right_index=True, how="left")
    cov = st["n_work"].fillna(0)
    out_stay = pd.DataFrame({
        "tt_cov28": (cov > 0).astype(float), "tt_n_work28": cov.astype(float),
        "tt_stay_n21": st["n_away21"].where(cov > 0), "tt_stay_n28": st["n_away28"].where(cov > 0),
        "tt_stay_flag": (st["n_away28"].fillna(0) > 0).astype(float).where(cov > 0),
        "tt_stay_days": st["first_away"], "tt_stay_last_away": st["last_away"],
        "tt_stay_at_venue": (st["n_at_venue"].fillna(0) > 0).astype(float).where(cov > 0),
    }, index=d.index)
    out_stay["tt_stay_loc"] = st["last_loc"].where(st["last_away"] == 1, "home").fillna("none").astype("category")

    # 馬・調教師ごとの遠征/滞在の成績（as-of）
    r = d.assign(top3=(pd.to_numeric(d["finish_position"], errors="coerce") <= 3).astype(float), far=(d["tier"] >= 2).astype(float),
                 home_run=(d["tier"].between(0, 1)).astype(float))
    r["stay"] = out_stay["tt_stay_flag"].fillna(0).to_numpy()
    hf = _asof_rate(r, "horse_id", "far", "top3", 5.0, "tt_h_far")
    hh = _asof_rate(r, "horse_id", "home_run", "top3", 5.0, "tt_h_home")
    tf = _asof_rate(r, "trainer_id", "far", "top3", 30.0, "tt_tr_far")
    ts = _asof_rate(r, "trainer_id", "stay", "top3", 30.0, "tt_tr_stay")
    out_rate = pd.concat([hf, hh, tf, ts], axis=1)
    out_rate["tt_h_far_gap"] = out_rate["tt_h_far_rate"] - out_rate["tt_h_home_rate"]
    out_rate.loc[out_rate["tt_h_far_n"] == 0, "tt_h_far_gap"] = np.nan
    out_rate["tt_tr_far_gap"] = out_rate["tt_tr_far_rate"] - out_rate["tt_tr_far_prior"]
    out_rate = out_rate.drop(columns=[c for c in out_rate.columns if c.endswith("_prior")])
    return {"TT_tier": out_tier, "TT_move": out_move, "TT_stay": out_stay, "TT_rate": out_rate}
