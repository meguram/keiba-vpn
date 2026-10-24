"""v6 の追加候補（展開タイプ・枠順バイアス・クラス基準）を、年ごとの as-of（その年より前の年だけから作る基準）で出す純粋関数。

どの基準（脚質×展開の倍率・コース別の枠係数・クラス別の突破ライン）も、対象年より前の年のレースだけから推定する
（2020 年は基準が作れないので欠損）。推論時は同じ形の基準を『直近までの全期間』で作って当てる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PACE = ["slow", "mid", "high"]
SHAPE = ["compact", "normal", "stretched"]
STYLES = ["front", "stalker", "mid", "closer"]
STYLE_BINS = [-1, 0.15, 0.35, 0.65, 2]


def style_bucket(pf_early: pd.Series) -> pd.Series:
    """tp-61 / tp-53 と同じ脚質の区分（想定位置取り pf_early: 0=先頭〜1=後方）。"""
    return pd.cut(pf_early, STYLE_BINS, labels=STYLES).astype(object)


def asof_train_years(year: int, first: int = 2020, loyo_until: int = 2021, loyo_pool: tuple[int, ...] = (2020, 2021, 2022, 2023)) -> list[int]:
    """year の特徴を作るときに基準を推定してよい年。2022 年以降は『その年より前』だけ。
    2020・2021 年は前の年が無い（または少ない）ので、2020〜2023 年の他の年から作る（学習側の行だけに入る弱い先読み。評価年の値は厳密に過去のみ）。"""
    if year >= loyo_until + 1:
        return [y for y in range(first, year)]
    return [y for y in loyo_pool if y != year]


def style_lift_table(h: pd.DataFrame, min_races: int = 30) -> np.ndarray:
    """h: pf_style, pace_cls, shape_cls, is_top3, race_id（推定に使う年の馬）。戻り値 L[ペース, 隊列, 脚質] = その脚質の全体の3着内率に対する倍率。足りないセルは 1。"""
    L = np.ones((3, 3, 4))
    ref = h.groupby("pf_style")["is_top3"].mean()
    for i, pc in enumerate(PACE):
        for j, sc in enumerate(SHAPE):
            g = h[(h["pace_cls"] == pc) & (h["shape_cls"] == sc)]
            for k, st in enumerate(STYLES):
                gs = g[g["pf_style"] == st]
                if gs["race_id"].nunique() < min_races or st not in ref.index:
                    continue
                L[i, j, k] = float(gs["is_top3"].sum() / (ref[st] * len(gs)))
    return L


def expected_style_fit(pace_p: np.ndarray, shape_p: np.ndarray, style_idx: np.ndarray, L: np.ndarray) -> pd.DataFrame:
    """行ごとの展開の確率（pace_p: n×3, shape_p: n×3、独立な積）と脚質の番号（0〜3、欠損は -1）から、
    期待倍率・展開による振れ幅（9 通りの倍率の確率加重の標準偏差）・倍率の最大−最小（確率 0.1 以上の展開のうち）を返す。"""
    n = len(style_idx)
    ok = style_idx >= 0
    exp = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    rng = np.full(n, np.nan)
    if ok.any():
        k = style_idx[ok]
        Lk = np.transpose(L[:, :, k], (2, 0, 1))                     # (m, 3, 3)
        W = pace_p[ok][:, :, None] * shape_p[ok][:, None, :]          # (m, 3, 3)
        e = (W * Lk).sum(axis=(1, 2))
        exp[ok] = e
        sd[ok] = np.sqrt((W * (Lk - e[:, None, None]) ** 2).sum(axis=(1, 2)))
        hi = np.where(W >= 0.1, Lk, -np.inf).reshape(len(k), 9).max(axis=1)
        lo = np.where(W >= 0.1, Lk, np.inf).reshape(len(k), 9).min(axis=1)
        r = hi - lo
        rng[ok] = np.where(np.isfinite(r), r, np.nan)
    return pd.DataFrame({"pt_style_lift": exp, "pt_style_sd": sd, "pt_style_range": rng})


# ───────────────────────── 枠順バイアス

def gate_rel(horse_number: pd.Series, race_id: pd.Series) -> pd.Series:
    """(馬番−1)/(そのレースの最大馬番−1)。0=最内, 1=最外。"""
    mx = horse_number.groupby(race_id).transform("max")
    return ((horse_number - 1) / (mx - 1).replace(0, np.nan)).astype(float)


def gate_year_stats(h: pd.DataFrame, min_field: int = 8) -> pd.DataFrame:
    """h: race_id, year, course, horse_number, early_norm, field_size（過去走ではなく各走の最初のコーナー位置）。
    コース×年ごとに、レース内で中心化した (gate_rel−0.5) と early_norm の積和（傾きの推定用）を足し上げる。"""
    d = h[(h["field_size"] >= min_field) & h["early_norm"].notna()].copy()
    d["x"] = gate_rel(d["horse_number"], d["race_id"]) - 0.5
    gx = d.groupby("race_id")["x"].transform("mean")
    gy = d.groupby("race_id")["early_norm"].transform("mean")
    d["xc"], d["yc"] = d["x"] - gx, d["early_norm"] - gy
    d["sxx"], d["sxy"], d["syy"] = d["xc"] ** 2, d["xc"] * d["yc"], d["yc"] ** 2
    g = d.groupby(["course", "year"]).agg(sxx=("sxx", "sum"), sxy=("sxy", "sum"), syy=("syy", "sum"), n=("sxx", "size"), races=("race_id", "nunique")).reset_index()
    return g


def gate_coeff_asof(stats: pd.DataFrame, year: int, surface_of: dict[str, str], min_races: int = 40, tau_floor: float = 0.02) -> dict[str, float]:
    """year より前の年だけで、コースごとの傾き k（内端→外端で位置が何だけ後ろになるか）を出し、面の平均へ経験ベイズで縮約する。
    戻り値は course → k_shrunk（コースが無い・レースが少ないコースは面の平均）、'_surface:<面>' → 面の平均。"""
    s = stats[stats["year"] < year]
    if s.empty:
        return {}
    c = s.groupby("course").agg(sxx=("sxx", "sum"), sxy=("sxy", "sum"), syy=("syy", "sum"), n=("n", "sum"), races=("races", "sum"))
    c = c[c["sxx"] > 0].copy()
    c["k"] = c["sxy"] / c["sxx"]
    resid = (c["syy"] - c["sxy"] ** 2 / c["sxx"]).clip(lower=0) / (c["n"] - 1).clip(lower=1)
    c["se2"] = resid / c["sxx"]
    c["surf"] = [surface_of.get(x, "?") for x in c.index]
    out: dict[str, float] = {}
    for sf, g in c.groupby("surf"):
        gg = g[g["races"] >= min_races]
        if len(gg) < 3:
            m, tau2 = (float((g["k"] * 1 / g["se2"]).sum() / (1 / g["se2"]).sum()) if len(g) else 0.0), tau_floor ** 2
        else:
            w = 1.0 / gg["se2"]
            m = float((gg["k"] * w).sum() / w.sum())
            tau2 = max(float(((gg["k"] - m) ** 2).sum() / max(len(gg) - 1, 1) - gg["se2"].mean()), tau_floor ** 2 / 4)
        out[f"_surface:{sf}"] = m
        shr = tau2 / (tau2 + g["se2"])
        for course, row in g.iterrows():
            out[course] = m + float(shr[course]) * (float(row["k"]) - m) if row["races"] >= min_races else m
    return out


def gate_features(df: pd.DataFrame, coeff: dict[str, float]) -> pd.DataFrame:
    """df: race_id, course, surface, horse_number, field_size, pf_early, hn_firstcurve（任意）, lead5, n5。coeff は gate_coeff_asof の戻り値（同じ年の行だけに当てる）。"""
    rel = gate_rel(df["horse_number"].astype(float), df["race_id"])
    k = df["course"].map(coeff)
    k = k.fillna(df["surface"].map(lambda s: coeff.get(f"_surface:{s}", np.nan)))
    k = k.where(df["field_size"] >= 8, 0.0).where(k.notna())
    out = pd.DataFrame(index=df.index)
    out["gb_k"] = k.astype("float32")
    out["gb_gate_rel"] = rel.astype("float32")
    shift = k * (rel - 0.5)
    out["gb_shift"] = shift.astype("float32")
    out["gb_pos_adj"] = (pd.to_numeric(df["pf_early"], errors="coerce") + shift).clip(0, 1).astype("float32")
    fc = pd.to_numeric(df.get("hn_firstcurve"), errors="coerce") if "hn_firstcurve" in df else pd.Series(np.nan, index=df.index)
    out["gb_gate_x_fc"] = ((rel - 0.5) * fc / 1000.0).astype("float32")
    claim = (df["n5"] >= 3) & (df["lead5"] >= 0.4)
    out["gb_outer_lead"] = (claim & (rel >= 0.65)).astype("float32")
    out["gb_lead_x_rel"] = (df["lead5"] * rel).astype("float32")
    return out


# ───────────────────────── クラス基準

def same_condition(surface_a: str, dist_a: float, surface_b: str, dist_b: float) -> bool:
    """同条件走: 同じ面で、距離差 ±200m（今回が 2400m 以上なら ±400m）。"""
    if surface_a is None or surface_a != surface_b or not (dist_a == dist_a and dist_b == dist_b):
        return False
    tol = 400.0 if dist_a >= 2400 else 200.0
    return abs(dist_a - dist_b) <= tol


def prior_window_features(runs: pd.DataFrame, lines: pd.Series, lookback: int = 5) -> pd.DataFrame:
    """runs: horse_id, date, idx（有効な指数。無効は NaN）, surf, dist の全走（行ごとの順序は問わない）。lines: runs と同じ index の、その走（今回）のクラスの突破ライン。
    各走について『その走より日付が厳密に前の、指数のある直近 lookback 走』から、最高指数−突破ライン・基準以上の走の数・同条件走の最高指数−突破ラインを出す。"""
    r = runs.sort_values(["horse_id", "date"], kind="stable")
    n = len(r)
    mx = np.full(n, np.nan)
    cnt = np.full(n, np.nan)
    same = np.full(n, np.nan)
    nsame = np.zeros(n)
    nprior = np.zeros(n)
    idx = r["idx"].to_numpy(float)
    dt = r["date"].to_numpy("datetime64[ns]")
    surf = r["surf"].to_numpy(object)
    dist = r["dist"].to_numpy(float)
    ln = lines.reindex(r.index).to_numpy(float)
    starts = np.flatnonzero(np.r_[True, r["horse_id"].to_numpy()[1:] != r["horse_id"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        for i in range(a, b):
            j0 = np.searchsorted(dt[a:b], dt[i], side="left") + a          # 日付が厳密に前の走は [a, j0)
            prior = [j for j in range(j0 - 1, a - 1, -1) if idx[j] == idx[j]][:lookback]
            nprior[i] = len(prior)
            if not prior:
                continue
            vals = idx[prior]
            mx[i] = vals.max()
            if ln[i] == ln[i]:
                cnt[i] = float((vals >= ln[i]).sum())
            sm = [j for j in prior if same_condition(surf[i], dist[i], surf[j], dist[j])]
            nsame[i] = len(sm)
            if sm:
                same[i] = idx[sm].max()
    out = pd.DataFrame({"cb_max5": mx, "cb_cnt_above5": cnt, "cb_same_max": same, "cb_same_n": nsame, "cb_prior_n": nprior}, index=r.index)
    out["cb_line"] = ln
    out["cb_max5_gap"] = out["cb_max5"] - out["cb_line"]
    out["cb_same_gap"] = out["cb_same_max"] - out["cb_line"]
    return out.reindex(runs.index)


# ───────────────────────── 馬主名の正規化

_CORP = ("株式会社", "（株）", "(株)", "㈱", "有限会社", "（有）", "(有)", "㈲", "合同会社", "（合）", "(合)", "一般社団法人", "社団法人", "（社）")


def normalize_owner(name) -> str:
    """法人格の表記（（株）・株式会社・㈱ など）と空白・全角半角の違いを除く。空は ''。"""
    import re
    import unicodedata

    if name is None or name != name:
        return ""
    s = unicodedata.normalize("NFKC", str(name))
    for c in _CORP:
        s = s.replace(unicodedata.normalize("NFKC", c), "")
    s = re.sub(r"[\s　・･\.]", "", s)
    return s


def owner_asof(runs: pd.DataFrame, key: str, k: float = 50.0) -> pd.DataFrame:
    """runs: race_id, horse_id, date, field_size, finish_position, key 列（馬主のキー）。各走について、同じキーの馬主の『日付が厳密に前の走』だけの
    走数・馬の数・勝ちの超過（勝ち数−期待 1/頭数）と3着内の超過（3着内数−期待 min(3/頭数,1)）を n+k で縮約した値を返す。"""
    d = runs[["race_id", "horse_id", "date", "field_size", "finish_position", key]].copy()
    d["win"] = (pd.to_numeric(d["finish_position"], errors="coerce") == 1).astype(float)
    d["t3"] = (pd.to_numeric(d["finish_position"], errors="coerce") <= 3).astype(float)
    fs = pd.to_numeric(d["field_size"], errors="coerce")
    d["ew"] = 1.0 / fs
    d["e3"] = np.minimum(3.0 / fs, 1.0)
    d["one"] = 1.0
    d = d[d[key] != ""].copy()
    d = d.sort_values([key, "date"], kind="stable")
    g = d.groupby([key, "date"], sort=False)
    cols = ["one", "win", "t3", "ew", "e3"]
    day = g[cols].transform("sum")
    cum = d.groupby(key, sort=False)[cols].cumsum()
    end_of_day = cum.groupby([d[key], d["date"]], sort=False).transform("last")
    prior = end_of_day - day                            # 同じ日の走は含めない（日付が厳密に前）
    out = pd.DataFrame(index=d.index)
    out["n"] = prior["one"]
    out["ex_win"] = (prior["win"] - prior["ew"]) / (prior["one"] + k)
    out["ex_t3"] = (prior["t3"] - prior["e3"]) / (prior["one"] + k)
    return out.assign(race_id=d["race_id"], horse_id=d["horse_id"])
