"""好走距離レンジ（短）（長）: 馬ごとに、好走した距離の幅を過去走から作る（純関数・ファイルは読まない）。

点数: 走った距離 1 点。好走 3 点（古馬戦は 2・3 着、世代戦は 1〜3 着）。古馬戦の勝ち 5 点。
  着順 0（中止・取消など）は点を付けず、走数にも数えない。
相対値 = 距離（100m 刻み）ごとの点数 ÷ その馬の累計点数。相対値 >= τ の距離の最小・最大が レンジ（短）（長）。
  τ 以上の距離が無いとき（距離が多く点が散るとき）は、相対値が最大の距離（同値は全部）をレンジとする。
走ごとの値（as-of）は、その走の日より前の走だけを使う（同じ日の別レースは混ぜない）。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_EPS = 1e-9
_FULL = str.maketrans("０１２３４５６７８９", "0123456789")
GROUND = ("芝", "ダート")


def round_distance(d) -> np.ndarray:
    """100m 刻みに丸める（1150 → 1200、1149 → 1100）。"""
    return (np.floor(np.asarray(d, float) / 100.0 + 0.5) * 100).astype(int)


def age_of(sex_age: pd.Series) -> pd.Series:
    """sex_age（例 牡3）の数字。読めなければ NaN。"""
    return pd.to_numeric(sex_age.astype(str).str.translate(_FULL).str.extract(r"(\d+)")[0], errors="coerce")


def race_kind_of_class(race_class: object) -> str | None:
    """race_class の表記から。「３歳以上」「４歳以上」を含む→古馬戦、「２歳」「３歳」だけ→世代戦、読めなければ None。"""
    s = str(race_class or "").translate(_FULL)
    if "3歳以上" in s or "4歳以上" in s:
        return "古馬戦"
    if "2歳" in s or "3歳" in s:
        return "世代戦"
    return None


def run_points(df: pd.DataFrame) -> pd.DataFrame:
    """1 走 1 行に age・kind（世代戦／古馬戦）・points（1／3／5、着順 0 以下は NaN）を付けて返す。
    必要な列: race_id, race_class, sex_age, finish_position。race_class が空欄のレースは、同じレースの全馬が 3 歳以下なら世代戦、それ以外は古馬戦、年齢も読めなければ不明（1 点）。
    """
    out = df.copy()
    out["age"] = age_of(out["sex_age"])
    kind = out["race_class"].map(race_kind_of_class)
    blank = kind.isna()
    if blank.any():
        race_max_age = out.groupby("race_id")["age"].transform("max")
        # 区分の表記も年齢も読めないレースは「不明」とし、勝ち・好走の加点をしない（1 点）
        guess = np.where(race_max_age.isna(), "不明", np.where(race_max_age <= 3, "世代戦", "古馬戦"))
        kind = kind.where(~blank, guess)
    out["kind"] = kind
    fp = pd.to_numeric(out["finish_position"], errors="coerce")
    old = (out["kind"] == "古馬戦").to_numpy()
    pts = np.ones(len(out))
    good = np.where(old, (fp >= 2) & (fp <= 3), (fp >= 1) & (fp <= 3)).astype(bool)
    pts[good] = 3.0
    pts[old & (fp == 1).to_numpy()] = 5.0
    pts[(out["kind"] == "不明").to_numpy()] = 1.0
    pts[~(fp > 0).to_numpy()] = np.nan
    out["points"] = pts
    return out


def prepare_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """走ごとの表（horse_id, date, surface, distance, points）から、平地（芝・ダート）だけを残し、距離を丸め、日付を datetime にする。"""
    r = runs[runs["surface"].isin(GROUND)].copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r = r[r["date"].notna()].copy()          # 日付が欠けた走は履歴にも対象にも入れない（NaT が最小の整数になり全走の「前の走」に入るため）
    r["dist100"] = round_distance(r["distance"])
    return r


def _range_from_rel(rel: np.ndarray, dists: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """rel: [距離数] または [行, 距離数]。相対値 >= τ（無ければ最大の距離）の最小・最大の距離。"""
    rel = np.atleast_2d(rel)
    thr = np.minimum(tau, rel.max(axis=1)) - _EPS
    sel = rel >= thr[:, None]
    lo = np.where(sel, dists[None, :], np.iinfo(np.int64).max).min(axis=1)
    hi = np.where(sel, dists[None, :], -1).max(axis=1)
    return lo, hi


def as_of_ranges(runs: pd.DataFrame, taus: Sequence[float], by_surface: bool, min_runs: int = 1) -> dict[float, pd.DataFrame]:
    """走ごとの as-of レンジ。runs の列: horse_id, date, surface, distance, points（NaN は点なし・走数にも数えない。その走自身のレンジは返す）。
    各走について、日付が**その走より前**で（by_surface なら同じ馬場の）、points が有る走だけから距離別の点数を作る。
    戻り値は τ ごとの DataFrame（runs と同じ index）: kd_short, kd_long（m。前走までの走数が min_runs 未満なら NaN）, n_prior（前走までの走数）, pts_prior（累計点数）, rel_max（最大の相対値）。
    """
    r = prepare_runs(runs)
    r = r.sort_values(["horse_id", "date"], kind="stable")
    hid = r["horse_id"].to_numpy()
    dates = r["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    surf = (r["surface"].to_numpy() == "ダート").astype(int) if by_surface else np.zeros(len(r), int)
    dist = r["dist100"].to_numpy()
    pts = r["points"].to_numpy(float)
    valid = ~np.isnan(pts)
    pts0 = np.where(valid, pts, 0.0)
    cuts = np.flatnonzero(np.r_[True, hid[1:] != hid[:-1], True])
    n = len(r)
    res = {t: {"kd_short": np.full(n, np.nan), "kd_long": np.full(n, np.nan)} for t in taus}
    n_prior = np.zeros(n)
    pts_prior = np.zeros(n)
    rel_max = np.full(n, np.nan)
    for a, b in zip(cuts[:-1], cuts[1:]):
        d, s, ds, v, p = dates[a:b], surf[a:b], dist[a:b], valid[a:b], pts0[a:b]
        mask = (d[None, :] < d[:, None]) & (s[None, :] == s[:, None]) & v[None, :]
        udist = np.unique(ds)
        onehot = (ds[:, None] == udist[None, :]) * p[:, None]
        S = mask.astype(float) @ onehot
        tot = S.sum(axis=1)
        n_prior[a:b] = mask.sum(axis=1)
        pts_prior[a:b] = tot
        has = tot > 0
        if not has.any():
            continue
        rel = S[has] / tot[has, None]
        rel_max[a:b][has] = rel.max(axis=1)
        for t in taus:
            lo, hi = _range_from_rel(rel, udist, t)
            res[t]["kd_short"][a:b][has] = lo
            res[t]["kd_long"][a:b][has] = hi
    out = {}
    for t in taus:
        df = pd.DataFrame({**res[t], "n_prior": n_prior, "pts_prior": pts_prior, "rel_max": rel_max}, index=r.index)
        short = df["n_prior"] < min_runs
        df.loc[short, ["kd_short", "kd_long"]] = np.nan
        out[t] = df.reindex(runs.index)
    return out


def as_of_range(runs: pd.DataFrame, tau: float, by_surface: bool, min_runs: int = 1) -> pd.DataFrame:
    """as_of_ranges の τ が 1 つのとき。"""
    return as_of_ranges(runs, [tau], by_surface, min_runs)[tau]


def profile_table(runs: pd.DataFrame, by_surface: bool) -> pd.DataFrame:
    """全走（points が有るもの）を使った距離別の点数と相対値。列: horse_id, surface（合算のときは ""）, dist100, points, total, rel, n_runs。"""
    r = prepare_runs(runs)
    r = r[r["points"].notna()]
    r["surf"] = r["surface"] if by_surface else ""
    keys = ["horse_id", "surf"]
    g = r.groupby(keys + ["dist100"], as_index=False)["points"].sum()
    tot = r.groupby(keys).agg(total=("points", "sum"), n_runs=("points", "size")).reset_index()
    g = g.merge(tot, on=keys)
    g["rel"] = g["points"] / g["total"]
    return g.rename(columns={"surf": "surface"})


def latest_range(runs: pd.DataFrame, tau: float, by_surface: bool) -> pd.DataFrame:
    """馬ごと（by_surface なら馬×馬場）の最新のレンジ。列: horse_id, surface, kd_short, kd_long, n_runs, total_points。"""
    p = profile_table(runs, by_surface)
    keys = ["horse_id", "surface"]
    p = p.sort_values(keys + ["dist100"])
    thr = np.minimum(tau, p.groupby(keys)["rel"].transform("max")) - _EPS
    sel = p[p["rel"] >= thr]
    g = sel.groupby(keys).agg(kd_short=("dist100", "min"), kd_long=("dist100", "max")).reset_index()
    base = p.drop_duplicates(keys)[keys + ["n_runs", "total"]].rename(columns={"total": "total_points"})
    return base.merge(g, on=keys, how="left")[keys + ["kd_short", "kd_long", "n_runs", "total_points"]]


def range_before(runs: pd.DataFrame, horse_ids: Sequence[str], as_of, tau: float = 0.10, min_runs: int = 3) -> pd.DataFrame:
    """画面用: as_of の日より前の走だけで、芝ダ混合の馬ごとのレンジ。runs の列: horse_id, date, surface, distance, points。
    戻り値は horse_id 索引の short_m, long_m（m）と runs（点の付いた走数）。走数が min_runs 未満の馬は short_m・long_m が NaN。走が無い馬は行が無い。"""
    r = runs[runs["horse_id"].isin(set(horse_ids))]
    r = r[(pd.to_datetime(r["date"], errors="coerce") < pd.Timestamp(as_of)) & r["points"].notna()]
    if r.empty:
        return pd.DataFrame(columns=["short_m", "long_m", "runs"], dtype=float)
    r = r.assign(horse_id=r["horse_id"].astype(str))   # 区分型のままだと groupby が未使用の区分まで展開する
    lr = latest_range(r, tau, by_surface=False).set_index("horse_id")
    out = pd.DataFrame({"short_m": lr["kd_short"].astype(float), "long_m": lr["kd_long"].astype(float), "runs": lr["n_runs"].astype(int)})
    out.loc[out["runs"] < min_runs, ["short_m", "long_m"]] = np.nan
    return out
