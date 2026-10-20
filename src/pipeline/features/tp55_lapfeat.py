"""tp-55: ラップ予想（ハロンごとのラップ傾向別予測）の特徴量。出走表の段階（レース前）で分かる情報だけで作る。

馬ごとの過去走の特徴は『その走より前の走だけ』（日付が厳密に前）。同じ日の別レースの結果は使わない（同日の先のレースのラップは
『同日の傾向』として別に、ラウンド順で前のレースだけを使う）。市場オッズ・人気は使わない。
SmartRC のテン1F（h1_ten1f, ten1f_best）は『前走までの馬ごとのスタート1F』で、本走自身の値と一致する割合は偶然水準
（3%）。データの出どころは smartrc_race_flat（2020〜2025）。2026 年は表が無いので欠損として扱う。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FRONT_MAX = 0.15          # position_forecast と同じ脚質の境界
STALKER_MAX = 0.35
WEIGHTS = (0.35, 0.25, 0.20, 0.12, 0.08)
N_LAG = 8                 # 過去走の遡り数（早め先行の割合用）
MAX_LAPS = 18
M_SHRINK = 8.0            # コース別 par を距離×面の平均へ縮約する疑似件数

_PASS = re.compile(r"(\d{1,2})")


def parse_laps(s) -> list[float] | None:
    if isinstance(s, (list, tuple)):
        v = [float(x) for x in s]
    elif isinstance(s, str) and s.startswith("["):
        try:
            v = [float(x) for x in json.loads(s)]
        except Exception:
            return None
    else:
        return None
    if len(v) < 5 or any((not math.isfinite(x)) or x <= 0 for x in v):
        return None
    return v


def n_laps_of(distance) -> int:
    return int(math.ceil(float(distance) / 200.0))


def first_lap_meters(distance) -> int:
    return int(round(float(distance) - 200 * (n_laps_of(distance) - 1)))


def course_key(venue, surface, distance, direction) -> str:
    return f"{venue}|{surface}|{int(float(distance))}|{direction}"


def group_key(surface, distance) -> str:
    return f"{surface}|{int(float(distance))}"


def first_corner_norm(passing_order, field_size) -> float:
    """通過順位の最初の数字 → (順位-1)/(頭数-1)。読めなければ NaN。"""
    try:
        f = int(field_size)
        m = _PASS.search(str(passing_order or ""))
        if not m or f < 4:
            return float("nan")
        first = int(m.group(1))
        if not 1 <= first <= f:
            return float("nan")
        return (first - 1) / (f - 1)
    except Exception:
        return float("nan")


# ───────────────────────── par（コース別の基準ラップ）
@dataclass
class Par:
    course: dict            # course_key -> np.ndarray(n_laps)
    group: dict             # group_key -> np.ndarray(n_laps)

    def vec(self, venue, surface, distance, direction) -> np.ndarray | None:
        ck = course_key(venue, surface, distance, direction)
        gk = group_key(surface, distance)
        return self.course.get(ck, self.group.get(gk))


def fit_par(races: pd.DataFrame) -> Par:
    """races: columns venue, surface, distance, direction, laps(list)。距離×面の平均 par と、コース別 par（縮約）。"""
    grp: dict[str, list] = {}
    crs: dict[str, list] = {}
    for v, s, d, di, laps in zip(races["venue"], races["surface"], races["distance"], races["direction"], races["laps"]):
        a = np.asarray(laps, float)
        grp.setdefault(group_key(s, d), []).append(a)
        crs.setdefault(course_key(v, s, d, di), []).append(a)
    gmean = {k: np.mean(np.stack(v), axis=0) for k, v in grp.items() if len({len(x) for x in v}) == 1}
    cmean = {}
    for k, v in crs.items():
        if len({len(x) for x in v}) != 1:
            continue
        n = len(v)
        _, s, d, _ = k.split("|")[0], k.split("|")[1], k.split("|")[2], k.split("|")[3]
        g = gmean.get(f"{s}|{d}")
        m = np.mean(np.stack(v), axis=0)
        cmean[k] = (n * m + M_SHRINK * g) / (n + M_SHRINK) if g is not None else m
    return Par(course=cmean, group=gmean)


def lap_dev(laps, par_vec) -> np.ndarray:
    return np.asarray(laps, float) - np.asarray(par_vec, float)


def f3_l3(laps) -> tuple[float, float]:
    a = np.asarray(laps, float)
    return float(a[:3].sum()), float(a[-3:].sum())


# ───────────────────────── 馬ごとの過去走の特徴（as-of）
def asof_horse_features(entrants: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """entrants: race_id, horse_id, date(datetime)。runs: horse_id, date(datetime), early(最初のコーナーの正規化位置, 欠損可), race_id。
    各 entrant について、日付が厳密に前の過去走だけから特徴を作る（本走自身の結果は入らない）。"""
    r = runs[["horse_id", "date", "early"]].copy()
    r = r.sort_values(["horse_id", "date"], kind="stable")
    r["ri"] = r.groupby("horse_id").cumcount()
    cnt = r.groupby("horse_id").size().rename("n_total")
    e = entrants[["race_id", "horse_id", "date"]].copy()
    e["_k"] = np.arange(len(e))
    # その entrant より前にある run の件数 m = searchsorted(run_dates, date, 'left')
    out_m = np.zeros(len(e), dtype=np.int64)
    run_dates = {h: g["date"].values for h, g in r.groupby("horse_id")}
    for i, (h, d) in enumerate(zip(e["horse_id"].values, e["date"].values)):
        a = run_dates.get(h)
        out_m[i] = 0 if a is None else int(np.searchsorted(a, d, side="left"))
    e["m"] = out_m
    base = r.set_index(["horse_id", "ri"])["early"]
    lags = {}
    for k in range(1, N_LAG + 1):
        idx = pd.MultiIndex.from_arrays([e["horse_id"].values, e["m"].values - k])
        v = base.reindex(idx).values
        v = np.where(e["m"].values - k >= 0, v, np.nan)
        lags[f"lag{k}"] = v
    L = np.column_stack([lags[f"lag{k}"] for k in range(1, N_LAG + 1)])
    w = np.array(WEIGHTS)
    L5 = L[:, :5]
    ok = ~np.isnan(L5)
    wsum = (ok * w).sum(axis=1)
    early_w = np.where(wsum > 0, np.nansum(np.where(ok, L5, 0) * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    p_front = np.where(wsum > 0, (ok * (np.nan_to_num(L5, nan=9) <= FRONT_MAX) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    p_stalk = np.where(wsum > 0, (ok * (np.nan_to_num(L5, nan=9) <= STALKER_MAX) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    ok8 = ~np.isnan(L)
    n8 = ok8.sum(axis=1)
    front8 = np.where(n8 > 0, (ok8 * (np.nan_to_num(L, nan=9) <= FRONT_MAX)).sum(axis=1) / np.where(n8 > 0, n8, 1), np.nan)
    with np.errstate(all="ignore"):
        min5 = np.where(ok.any(axis=1), np.nanmin(np.where(ok, L5, np.inf), axis=1), np.nan)
        sd5 = np.where(ok.sum(axis=1) >= 2, np.nanstd(np.where(ok, L5, np.nan), axis=1), np.nan)
    feats = pd.DataFrame({
        "race_id": e["race_id"].values, "horse_id": e["horse_id"].values,
        "h_early_w": early_w, "h_p_front": p_front, "h_p_stalk": p_stalk, "h_front8": front8,
        "h_min5": min5, "h_sd5": sd5, "h_n_runs": e["m"].values, "h_n_known5": ok.sum(axis=1),
    })
    return feats


def asof_jockey_early(entrants: pd.DataFrame, runs: pd.DataFrame) -> pd.Series:
    """騎手の『最初のコーナーの正規化位置』の通算平均（日付が厳密に前まで）。index は entrants の位置。"""
    r = runs.dropna(subset=["early", "jockey_id"]).copy()
    r["jockey_id"] = r["jockey_id"].astype(str)
    daily = r.groupby(["jockey_id", "date"])["early"].agg(["sum", "count"]).reset_index().sort_values(["jockey_id", "date"])
    daily["cs"] = daily.groupby("jockey_id")["sum"].cumsum()
    daily["cc"] = daily.groupby("jockey_id")["count"].cumsum()
    q = entrants[["jockey_id", "date"]].copy()
    q["jockey_id"] = q["jockey_id"].fillna("").astype(str)
    q["_k"] = np.arange(len(q))
    q = q.sort_values("date", kind="stable")
    m = pd.merge_asof(q, daily[["jockey_id", "date", "cs", "cc"]].sort_values("date", kind="stable"), on="date", by="jockey_id", allow_exact_matches=False)
    val = np.where(m["cc"].fillna(0) >= 30, m["cs"] / m["cc"].replace(0, np.nan), np.nan)
    s = pd.Series(val, index=m["_k"].values).sort_index()
    return s


# ───────────────────────── レース単位の特徴
def _sorted_small(a: np.ndarray, k: int) -> np.ndarray:
    s = np.sort(a)
    out = np.full(k, np.nan)
    out[: min(k, len(s))] = s[:k]
    return out


def race_features_one(g: pd.DataFrame) -> dict:
    """g: 1 レースの entrants（列: horse_number, h_early_w, h_p_front, h_p_stalk, h_front8, h_n_known5, ten1f, ten1f_best, jockey_early, age, is_female, jockey_weight, h_n_runs）。"""
    n = len(g)
    hn = g["horse_number"].values.astype(float)
    e_raw = g["h_early_w"].values.astype(float)
    known = ~np.isnan(e_raw)
    e = np.where(known, e_raw, 0.5)
    out: dict = {"n_entrants": n, "n_known": int(known.sum()), "unknown_ratio": float(1 - known.mean()) if n else np.nan}
    es = _sorted_small(np.where(known, e_raw, np.inf), 4)
    es = np.where(np.isinf(es), np.nan, es)
    out.update({"e1": es[0], "e2": es[1], "e3": es[2], "e4": es[3], "e_mean": float(np.nanmean(e_raw)) if known.any() else np.nan,
                "e_std": float(np.nanstd(e_raw)) if known.sum() >= 2 else np.nan})
    out["n_front_hard"] = int((known & (e_raw <= FRONT_MAX)).sum())
    out["n_stalk_hard"] = int((known & (e_raw <= STALKER_MAX)).sum())
    out["exp_front"] = float(np.nansum(g["h_p_front"].values))
    out["exp_frontstalk"] = float(np.nansum(g["h_p_stalk"].values))
    out["press"] = float(np.nansum(np.exp(-e_raw[known] / 0.15))) if known.any() else np.nan
    out["no_leader"] = int(out["n_front_hard"] == 0)
    out["front8_sum"] = float(np.nansum(g["h_front8"].values))
    mx = float(hn.max()) if n else 1.0
    rel = (hn - 1) / max(mx - 1, 1)
    order = np.argsort(np.where(known, e_raw, np.inf), kind="stable")
    top = [i for i in order[:4] if known[i]]
    for j in range(3):
        out[f"gate_f{j+1}"] = float(rel[top[j]]) if len(top) > j else np.nan
        out[f"no_f{j+1}"] = float(hn[top[j]]) if len(top) > j else np.nan
    out["gate_gap12"] = abs(out["no_f1"] - out["no_f2"]) if len(top) >= 2 else np.nan
    pairs = 0
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            if abs(hn[top[a]] - hn[top[b]]) <= 2:
                pairs += 1
    out["front_adjacent_pairs"] = pairs
    out["front_inner"] = int(sum(rel[i] <= 0.35 for i in top))
    out["front_outer"] = int(sum(rel[i] >= 0.65 for i in top))
    out["gate_front_mean"] = float(np.mean([rel[i] for i in top[:3]])) if top else np.nan
    # テン1F（馬ごとの前走までのスタート1F。小さいほど速い）
    t = g["ten1f"].values.astype(float)
    tb = g["ten1f_best"].values.astype(float)
    tk = ~np.isnan(t)
    ts = np.sort(t[tk])
    out["ten_cnt"] = int(tk.sum())
    out["ten_min"] = float(ts[0]) if len(ts) else np.nan
    out["ten_2nd"] = float(ts[1]) if len(ts) > 1 else np.nan
    out["ten_gap12"] = float(ts[1] - ts[0]) if len(ts) > 1 else np.nan
    out["ten_mean3"] = float(ts[:3].mean()) if len(ts) else np.nan
    out["tenbest_min"] = float(np.nanmin(tb)) if np.isfinite(tb).any() else np.nan
    fm = known & (e_raw <= STALKER_MAX) & tk
    out["ten_front_min"] = float(t[fm].min()) if fm.any() else np.nan
    out["ten_front_mean"] = float(t[fm].mean()) if fm.any() else np.nan
    # 騎手の積極性（前に行った割合の通算）
    je = g["jockey_early"].values.astype(float)
    tj = [je[i] for i in top[:3] if np.isfinite(je[i])]
    out["jock_front_mean"] = float(np.mean(tj)) if tj else np.nan
    out["jock_min"] = float(np.nanmin(je)) if np.isfinite(je).any() else np.nan
    # 構成
    out["age_mean"] = float(np.nanmean(g["age"].values.astype(float))) if n else np.nan
    out["female_share"] = float(np.nanmean(g["is_female"].values.astype(float))) if n else np.nan
    out["age3_share"] = float(np.nanmean((g["age"].values.astype(float) <= 3).astype(float))) if n else np.nan
    out["jw_sd"] = float(np.nanstd(g["jockey_weight"].values.astype(float))) if n else np.nan
    out["runs_mean"] = float(np.nanmean(np.log1p(g["h_n_runs"].values.astype(float)))) if n else np.nan
    return out


def parse_sex_age(s) -> tuple[float, int]:
    m = re.search(r"(\d+)", str(s or ""))
    age = float(m.group(1)) if m else np.nan
    return age, int(str(s or "").startswith("牝"))


RACE_NUM_COLS = [
    "n_entrants", "n_known", "unknown_ratio", "e1", "e2", "e3", "e4", "e_mean", "e_std", "n_front_hard", "n_stalk_hard", "exp_front",
    "exp_frontstalk", "press", "no_leader", "front8_sum", "gate_f1", "gate_f2", "gate_f3", "no_f1", "no_f2", "no_f3", "gate_gap12",
    "front_adjacent_pairs", "front_inner", "front_outer", "gate_front_mean", "ten_cnt", "ten_min", "ten_2nd", "ten_gap12", "ten_mean3",
    "tenbest_min", "ten_front_min", "ten_front_mean", "jock_front_mean", "jock_min", "age_mean", "female_share", "age3_share", "jw_sd", "runs_mean",
]
FEATURE_GROUPS = {
    "style": ["e1", "e2", "e3", "e4", "e_mean", "e_std", "n_front_hard", "n_stalk_hard", "exp_front", "exp_frontstalk", "press", "no_leader",
              "front8_sum", "unknown_ratio", "n_known"],
    "gate": ["gate_f1", "gate_f2", "gate_f3", "no_f1", "no_f2", "no_f3", "gate_gap12", "front_adjacent_pairs", "front_inner", "front_outer",
             "gate_front_mean"],
    "ten": ["ten_cnt", "ten_min", "ten_2nd", "ten_gap12", "ten_mean3", "tenbest_min", "ten_front_min", "ten_front_mean"],
    "jockey": ["jock_front_mean", "jock_min"],
    "field": ["n_entrants", "age_mean", "female_share", "age3_share", "jw_sd", "runs_mean"],
    "sameday": ["sd_n", "sd_q_mean", "sd_f3_mean", "sd_l3_mean", "sd_q_last"],
    "course": ["distance", "n_laps", "d0", "course_code", "venue_code_c", "surface_c", "direction_c", "kai", "nichime", "round_no", "month"],
    "cond": ["weather_c", "track_c", "grade_c", "class_c"],
}


def sameday_features(races: pd.DataFrame, par0: Par) -> pd.DataFrame:
    """同日・同会場でラウンド順が前のレースのラップ傾向（as-of）。races: race_id, date, venue, surface, round_no, laps, distance, direction。"""
    r = races.copy()
    f3 = []
    l3 = []
    for v, s, d, di, laps in zip(r["venue"], r["surface"], r["distance"], r["direction"], r["laps"]):
        pv = par0.vec(v, s, d, di)
        if pv is None or laps is None or len(pv) != len(laps):
            f3.append(np.nan); l3.append(np.nan); continue
        dv = np.asarray(laps) - pv
        f3.append(dv[:3].sum()); l3.append(dv[-3:].sum())
    r["f3d"] = f3
    r["l3d"] = l3
    r["qd"] = r["l3d"] - r["f3d"]
    r = r.sort_values(["date", "venue", "round_no"], kind="stable")
    out = {}
    for (d, v), g in r.groupby(["date", "venue"], sort=False):
        qd = g["qd"].values; f = g["f3d"].values; l = g["l3d"].values; sf = g["surface"].values; rid = g["race_id"].values
        for i in range(len(g)):
            prev = [j for j in range(i) if np.isfinite(qd[j])]
            same = [j for j in prev if sf[j] == sf[i]]
            use = same[-4:]
            out[rid[i]] = (len(use), float(np.mean(qd[use])) if use else np.nan, float(np.mean(f[use])) if use else np.nan,
                           float(np.mean(l[use])) if use else np.nan, float(qd[use[-1]]) if use else np.nan)
    df = pd.DataFrame.from_dict(out, orient="index", columns=["sd_n", "sd_q_mean", "sd_f3_mean", "sd_l3_mean", "sd_q_last"])
    df.index.name = "race_id"
    return df.reset_index()
