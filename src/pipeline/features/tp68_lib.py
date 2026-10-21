"""tp-68（null importance・RFE 簡易版・欠損数/暦の特徴・疑似ラベル）の純関数。重い処理は src/scripts/data/tp68_tab.py。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


# ───────────────────────── 暦（国民の祝日 2019〜2027）
def _nth_monday(y: int, m: int, n: int) -> dt.date:
    d = dt.date(y, m, 1)
    d += dt.timedelta(days=(7 - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


_EQUINOX = {
    "spring": {2020: 20, 2021: 20, 2022: 21, 2023: 21, 2024: 20, 2025: 20, 2026: 20, 2027: 21},
    "autumn": {2020: 22, 2021: 23, 2022: 23, 2023: 23, 2024: 22, 2025: 23, 2026: 23, 2027: 23},
}


def jp_holidays(year: int) -> set[dt.date]:
    """その年の国民の祝日（振替休日・国民の休日を含む）。2020〜2027 年の規則（2020・2021 年の東京五輪に伴う海の日・スポーツの日・山の日の移動を含む）。"""
    h: set[dt.date] = {dt.date(year, 1, 1), _nth_monday(year, 1, 2), dt.date(year, 2, 11), dt.date(year, 4, 29), dt.date(year, 5, 3), dt.date(year, 5, 4),
                       dt.date(year, 5, 5), dt.date(year, 11, 3), dt.date(year, 11, 23), _nth_monday(year, 9, 3)}
    h.add(dt.date(year, 2, 23))
    h.add(dt.date(year, 3, _EQUINOX["spring"][year]))
    h.add(dt.date(year, 9, _EQUINOX["autumn"][year]))
    special = {2020: (dt.date(2020, 7, 23), dt.date(2020, 8, 10), dt.date(2020, 7, 24)), 2021: (dt.date(2021, 7, 22), dt.date(2021, 8, 8), dt.date(2021, 7, 23))}
    if year in special:
        h.update(special[year])
    else:
        h.update({_nth_monday(year, 7, 3), dt.date(year, 8, 11), _nth_monday(year, 10, 2)})
    # 国民の休日（前後が祝日の平日）
    for d in sorted(h):
        mid = d + dt.timedelta(days=1)
        if mid not in h and mid.weekday() < 5 and (mid + dt.timedelta(days=1)) in h and mid.year == year:
            h.add(mid)
    # 振替休日（日曜の祝日の翌日以降で最初に祝日でない日）
    for d in sorted(h):
        if d.weekday() == 6:
            s = d + dt.timedelta(days=1)
            while s in h:
                s += dt.timedelta(days=1)
            h.add(s)
    return h


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """祝日フラグと連休（土日祝の連続）の長さ・何日目か・連休明けか。日付だけから作るので先読みは無い。"""
    d = pd.to_datetime(dates).dt.date
    years = sorted({x.year for x in d} | {min(x.year for x in d) - 1, max(x.year for x in d) + 1})
    hol: set[dt.date] = set()
    for y in years:
        if 2020 <= y <= 2027:
            hol |= jp_holidays(y)

    def off(x: dt.date) -> bool:
        return x.weekday() >= 5 or x in hol

    uniq = sorted(set(d))
    info: dict[dt.date, tuple[int, int, int]] = {}
    for x in uniq:
        if not off(x):
            info[x] = (0, 0, 0)
            continue
        a = x
        while off(a - dt.timedelta(days=1)):
            a -= dt.timedelta(days=1)
        b = x
        while off(b + dt.timedelta(days=1)):
            b += dt.timedelta(days=1)
        info[x] = ((b - a).days + 1, (x - a).days + 1, int(x in hol))
    out = pd.DataFrame({"cal_run_len": [info[x][0] for x in d], "cal_day_in_run": [info[x][1] for x in d], "cal_is_holiday": [int(x in hol) for x in d]}, index=dates.index)
    out["cal_after_run"] = [int((x - dt.timedelta(days=1)) in hol and not off(x)) for x in d]
    return out.astype("float32")


# ───────────────────────── 欠損数
def prefix_groups(cols: list[str], min_size: int = 15) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for c in cols:
        g.setdefault(c.split("_")[0], []).append(c)
    return {k: v for k, v in g.items() if len(v) >= min_size}


def missing_features(X: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """出走ごとの欠損数（全体・群別）。群別は欠損割合（0〜1）。"""
    na = X.isna()
    out = pd.DataFrame(index=X.index)
    out["miss_n_all"] = na.sum(axis=1).astype("float32")
    for k, cols in groups.items():
        out[f"miss_frac_{k}"] = na[cols].mean(axis=1).astype("float32")
    return out


# ───────────────────────── null importance / RFE
def permute_within_race(y: np.ndarray, race: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """レース内でラベルを並べ替える（1 レース 1 頭の勝ちを保ち、馬と結果の対応だけを壊す）。"""
    out = y.copy()
    order = np.argsort(race, kind="stable")
    r = race[order]
    bounds = np.flatnonzero(np.r_[True, r[1:] != r[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        out[idx] = y[rng.permutation(idx)]
    return out


def null_summary(real: pd.Series, null: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """real: 列ごとの gain、null: (反復, 列) の gain。各しきい値の分位と、null が real 以上だった割合。"""
    q = {f"null_q{p}": np.percentile(null, p, axis=0) for p in (50, 75, 90, 95, 99)}
    df = pd.DataFrame({"col": cols, "real_gain": real.reindex(cols).to_numpy(), "null_mean": null.mean(axis=0), **q})
    df["frac_null_ge_real"] = (null >= df["real_gain"].to_numpy()[None, :]).mean(axis=0)
    df["gain_ratio"] = df["real_gain"] / df["null_mean"].clip(lower=1e-12)
    return df


def rfe_columns(real_gain: pd.Series, n_keep: int, min_per_group: int = 2) -> list[str]:
    """重要度の上位 n_keep 列。ただし接頭辞の群ごとに上位 min_per_group 列は必ず残す（群の全滅を防ぐ）。"""
    ranked = real_gain.sort_values(ascending=False)
    keep = list(ranked.index[:n_keep])
    seen = set(keep)
    per: dict[str, int] = {}
    for c in keep:
        per[c.split("_")[0]] = per.get(c.split("_")[0], 0) + 1
    for c in ranked.index:
        g = c.split("_")[0]
        if per.get(g, 0) < min_per_group and c not in seen:
            keep.append(c)
            seen.add(c)
            per[g] = per.get(g, 0) + 1
    return keep


# ───────────────────────── 疑似ラベル
def soft_label_augment(X_lab: pd.DataFrame, y_lab: dict, X_un: pd.DataFrame, p_un: dict) -> tuple[pd.DataFrame, dict, dict]:
    """ラベル無しの行を、目的ごとに『確率 p の重みで y=1、(1-p) の重みで y=0』の 2 行として足す（ソフトラベル）。
    戻り値: (結合した X, 目的ごとのラベル, 目的ごとの重み)。ラベル付き行の重みは 1。"""
    X = pd.concat([X_lab, X_un, X_un], axis=0, ignore_index=True)
    n, m = len(X_lab), len(X_un)
    y: dict = {}
    w: dict = {}
    for k in y_lab:
        p = np.clip(np.asarray(p_un[k], dtype=float), 1e-6, 1 - 1e-6)
        y[k] = np.concatenate([np.asarray(y_lab[k]), np.ones(m), np.zeros(m)]).astype(int)
        w[k] = np.concatenate([np.ones(n), p, 1 - p])
    return X, y, w
