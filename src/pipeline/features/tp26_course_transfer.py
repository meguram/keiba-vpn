"""tp-26: 馬の成績が実際に連動するコースの対応づけ（コース転移）。

第一段階（数値）: コース A, B の両方を走った馬について、馬ごとの『コース別の平均成績（その馬の全走平均との差）』の相関を
  『転移相関』とする。馬の能力（どのコースでも強い）は全走平均を引いて除くが、走数の少ない馬では引いた平均が自分の走を含むため負の偏りが出る。
  そこで、同じ馬の走のコースの割り当てを入れ替えた帰無分布（馬ごとのラベル置換）から期待値と標準偏差を求め、
  『超過相関 r_exc = r − 帰無の平均』と z = r_exc / 帰無の標準偏差 を使う。z から片側 p、Benjamini–Hochberg で q。
第二段階（絞り込み）: 統計（q と r_exc）、概形（コースマスタの形状距離の近傍）、積（AND）、和（OR）、全ペア（対照）。
特徴量: 今回のコース C に対し、馬の他コース B での (成績 − 全走平均) を、類似度 s_CB と件数の信頼度で重みづけ平均（as-of）。
  行の as-of: 類似度は『その行の年の 1 月 1 日より前』の走だけで作った行列、馬の成績は『その行の日付より前』の走だけを使う。

リーク防止: 類似度行列の年 y 版は date < y-01-01 の走だけ。馬の集計は厳密に過去。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.course_variant import add_course_variant

MASTER = Path(__file__).resolve().parents[3] / "data/meta/course_master/jra_courses.json"
METRICS = ("fin", "l3f", "pos")
N_MIN = 30           # 転移相関を出すために必要な、両方のコースを走った馬の数
R_MIN = 0.05         # 統計フィルタの超過相関の下限
Q_MAX = 0.05
K0 = 2.0             # 馬×コースの件数の縮約
LAM = 0.1            # 重みの合計に足す定数（根拠が薄いと 0 に寄せる）
GEO_K = 8            # 概形の近傍数（同じ面の中）


# ---------------------------------------------------------------- 走ごとの表
def course_key(df: pd.DataFrame) -> pd.Series:
    var = add_course_variant(df, parsed_col="course_type")
    surf = df["surface"].astype(str).map(lambda s: "ダート" if s.startswith("ダ") else s)
    return (df["venue_code"].astype(str).str.zfill(2) + "|" + surf + "|" + pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int).astype(str)
            + "|" + var.fillna("?").astype(str))


def build_runs(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """df: race_id, horse_id, date, venue_code, surface, distance, field_size, finish_position（内外はコースマスタから導出する）。
    labels: race_id, horse_id, l3f, pos_early（stage_labels）。戻り値: 走ごとの (course, fin/l3f/pos) の成績（高いほど良い。レース内で相対化）。"""
    d = df[["race_id", "horse_id", "date", "venue_code", "surface", "distance", "field_size", "finish_position"]].copy()
    d[["venue_code", "surface"]] = d[["venue_code", "surface"]].astype(str)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(pd.to_numeric(d["finish_position"], errors="coerce") > 0) & d["surface"].astype(str).isin(["芝", "ダ", "ダート"])].copy()
    n = pd.to_numeric(d["field_size"], errors="coerce").clip(lower=2)
    d["fin"] = 0.5 - (pd.to_numeric(d["finish_position"]) - 1.0) / (n - 1.0)
    lab = labels[["race_id", "horse_id", "l3f", "pos_early"]].copy()
    lab["race_id"], lab["horse_id"] = lab["race_id"].astype(str), lab["horse_id"].astype(str)
    d = d.merge(lab.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    g = d.groupby("race_id")
    d["l3f"] = -(pd.to_numeric(d["l3f"], errors="coerce") - g["l3f"].transform("mean"))          # 速い上がりほど大
    d["pos"] = -(pd.to_numeric(d["pos_early"], errors="coerce") - g["pos_early"].transform("mean"))  # 前にいるほど大
    d["course"] = course_key(d)
    d["year"] = d["date"].dt.year
    return d.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)[
        ["race_id", "horse_id", "date", "year", "course", "fin", "l3f", "pos"]]


# ---------------------------------------------------------------- 第一段階: 転移相関
def horse_course_matrix(horse_idx: np.ndarray, course_idx: np.ndarray, value: np.ndarray, n_h: int, n_c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """馬×コースの『平均 − 馬の全走平均』X、件数 N、有効フラグ M。value が NaN の走は数えない。"""
    ok = np.isfinite(value)
    h, c, v = horse_idx[ok], course_idx[ok], value[ok]
    cnt = np.zeros((n_h, n_c))
    sm = np.zeros((n_h, n_c))
    np.add.at(cnt, (h, c), 1.0)
    np.add.at(sm, (h, c), v)
    tot_n = cnt.sum(1)
    tot_s = sm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_all = np.where(tot_n > 0, tot_s / tot_n, 0.0)
        X = np.where(cnt > 0, sm / cnt - mean_all[:, None], 0.0)
    return X, cnt, (cnt > 0).astype(float)


def pair_corr(X: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全コース対の相関 r と、両方を走った馬の数 N。"""
    N = M.T @ M
    X0 = X * M
    Sxy = X0.T @ X0
    Sx = X0.T @ M
    Sxx = (X0 ** 2).T @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = Sx / N
        my = Sx.T / N
        cov = Sxy / N - mx * my
        vx = Sxx / N - mx ** 2
        vy = Sxx.T / N - my ** 2
        r = cov / np.sqrt(vx * vy)
    r = np.where(np.isfinite(r), r, 0.0)
    np.fill_diagonal(r, 0.0)
    return r, N


def shuffled_courses(horse_idx: np.ndarray, course_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """馬ごとに、走のコースの割り当てを入れ替える（成績との対応を壊す）。horse_idx は昇順に並んだ前提。"""
    order = np.lexsort((rng.random(len(horse_idx)), horse_idx))
    out = np.empty_like(course_idx)
    out[np.argsort(horse_idx, kind="stable")] = course_idx[order]
    return out


def similarity_snapshot(runs: pd.DataFrame, courses: list[str], metric: str, before: pd.Timestamp, n_perm: int = 40, seed: int = 0) -> dict:
    """date < before の走だけで作る類似度。戻り値: r, N, r_exc, z, p, q（いずれも C×C）。"""
    from scipy.stats import norm

    c2i = {c: i for i, c in enumerate(courses)}
    r_ = runs[(runs["date"] < before) & runs["course"].isin(c2i)]
    horse_codes, horses = pd.factorize(r_["horse_id"])
    ci = r_["course"].map(c2i).to_numpy()
    v = r_[metric].to_numpy(dtype=float)
    nh, nc = len(horses), len(courses)
    X, cnt, M = horse_course_matrix(horse_codes, ci, v, nh, nc)
    r, N = pair_corr(X, M)
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(n_perm):
        cs = shuffled_courses(horse_codes, ci, rng)
        Xp, _, Mp = horse_course_matrix(horse_codes, cs, v, nh, nc)
        perms.append(pair_corr(Xp, Mp)[0])
    P = np.stack(perms)
    mu0 = P.mean(0)
    sd0 = P.std(0, ddof=1)
    valid = (N >= N_MIN) & ~np.eye(nc, dtype=bool)
    # 標準偏差は 1/√N に比例するとして平滑化（置換回数が少ないと対ごとの推定がぶれる）
    a_hat = float(np.median((sd0 * np.sqrt(N))[valid & (sd0 > 0)])) if valid.any() else 1.0
    sd_fit = np.where(valid, a_hat / np.sqrt(np.maximum(N, 1.0)), np.inf)
    r_exc = np.where(valid, r - mu0, 0.0)
    z = np.where(valid, r_exc / sd_fit, 0.0)
    p = np.where(valid, norm.sf(z), 1.0)
    q = bh_qvalues(p, valid)
    return {"r": r, "N": N, "r_exc": r_exc, "z": z, "p": p, "q": q, "mu0": mu0, "a_hat": a_hat, "valid": valid, "n_horses": nh}


def bh_qvalues(p: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """上三角の対だけで Benjamini–Hochberg の q を出し、対称に戻す。"""
    iu = np.triu_indices_from(p, 1)
    pv = p[iu]
    ok = valid[iu]
    q_u = np.ones_like(pv)
    idx = np.flatnonzero(ok)
    if len(idx):
        pp = pv[idx]
        order = np.argsort(pp)
        m = len(pp)
        ranked = pp[order] * m / (np.arange(m) + 1)
        qq = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(qq, 1.0)
        q_u[idx] = out
    q = np.ones_like(p)
    q[iu] = q_u
    q = np.minimum(q, q.T)
    return q


# ---------------------------------------------------------------- 概形（コースマスタ）
GEO_NUM = ["log_dist", "lap_length_m", "final_straight_m", "elevation_range_m", "first_curve_m", "n_corners"]


def geo_table(courses: list[str], master_path: str | Path = MASTER) -> pd.DataFrame:
    """コースのキー（場|面|距離|内外）ごとの概形の属性。内外が不明（?）のキーは候補の平均。"""
    rows = json.loads(Path(master_path).read_text(encoding="utf-8"))
    m = pd.DataFrame([{"venue_code": str(r["venue_code"]).zfill(2), "surface": r["surface"], "distance": int(r["distance"]), "variant": r.get("variant") or "",
                       "direction": r.get("direction") or "", "lap_length_m": r.get("lap_length_m"), "final_straight_m": r.get("final_straight_m"),
                       "elevation_range_m": r.get("elevation_range_m"), "first_curve_m": r.get("first_curve_m"), "n_corners": r.get("n_corners_record_mode"),
                       "start_geometry": r.get("start_geometry") or ""} for r in rows])
    m["log_dist"] = np.log(m["distance"].astype(float))
    out = []
    for c in courses:
        v, s, d, var = c.split("|")
        sel = m[(m["venue_code"] == v) & (m["surface"] == s) & (m["distance"] == int(d))]
        if var != "?":
            sel2 = sel[sel["variant"] == var]
            sel = sel2 if len(sel2) else sel
        rec = {"course": c, "venue_code": v, "surface": s, "distance": int(d), "variant": var, "in_master": len(sel) > 0}
        if len(sel):
            for k in GEO_NUM:
                rec[k] = pd.to_numeric(sel[k], errors="coerce").mean()
            rec["direction"] = sel["direction"].mode().iloc[0]
            rec["start_geometry"] = sel["start_geometry"].mode().iloc[0]
        out.append(rec)
    return pd.DataFrame(out)


def geo_distance(geo: pd.DataFrame) -> np.ndarray:
    """標準化した数値の距離 + 回りが違えば +1 + 面が違えば +3。"""
    Z = geo[GEO_NUM].astype(float)
    Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0, 1.0)
    Z = Z.fillna(0.0).to_numpy()
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    dirs = geo["direction"].fillna("").to_numpy()
    surf = geo["surface"].to_numpy()
    d = d + (dirs[:, None] != dirs[None, :]) * 1.0 + (surf[:, None] != surf[None, :]) * 3.0
    np.fill_diagonal(d, 0.0)
    return d


def geo_close(d: np.ndarray, geo: pd.DataFrame, k: int = GEO_K) -> np.ndarray:
    """同じ面の中で近い k コース（どちらかの近傍に入ればよい＝対称）。master に無いコースは False。"""
    n = len(d)
    ok = geo["in_master"].to_numpy()
    close = np.zeros((n, n), dtype=bool)
    surf = geo["surface"].to_numpy()
    for i in range(n):
        if not ok[i]:
            continue
        cand = [j for j in range(n) if j != i and ok[j] and surf[j] == surf[i]]
        cand.sort(key=lambda j: d[i, j])
        for j in cand[:k]:
            close[i, j] = True
    return close | close.T


# ---------------------------------------------------------------- 重み行列（変種）
def weight_matrix(snap: dict | None, variant: str, close: np.ndarray, kernel: np.ndarray, r_scale: float = 0.1) -> np.ndarray:
    """変種ごとの類似度（C×C）。snap=None（学習データが無い年）は 0。対角は 0。"""
    n = close.shape[0]
    if snap is None:
        return np.zeros((n, n))
    r_exc, q, valid = snap["r_exc"], snap["q"], snap["valid"]
    stat = valid & (q < Q_MAX) & (r_exc >= R_MIN)
    if variant == "ALL":
        w = np.where(valid, np.maximum(r_exc, 0.0), 0.0)
    elif variant == "STAT":
        w = np.where(stat, r_exc, 0.0)
    elif variant == "GEO":
        w = np.where(close, kernel * r_scale, 0.0)
    elif variant == "AND":
        w = np.where(stat & close, r_exc, 0.0)
    elif variant == "OR":
        w = np.where(stat, r_exc, np.where(close, kernel * r_scale, 0.0))
    else:
        raise ValueError(variant)
    np.fill_diagonal(w, 0.0)
    return w


# ---------------------------------------------------------------- 行ごとの as-of 特徴量
def asof_vectors(runs: pd.DataFrame, courses: list[str], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """各走の時点（その走を含まない過去）での、馬のコース別 (平均 − 全走平均) と件数。runs は horse, date 昇順。
    戻り値 X, G（走数 × コース数、float32）。G は件数の信頼度 n/(n+K0)。"""
    c2i = {c: i for i, c in enumerate(courses)}
    nc = len(courses)
    ci = runs["course"].map(c2i).fillna(-1).astype(int).to_numpy()
    v = runs[metric].to_numpy(dtype=float)
    horse = runs["horse_id"].to_numpy()
    n = len(runs)
    X = np.zeros((n, nc), dtype=np.float32)
    G = np.zeros((n, nc), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, horse[1:] != horse[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        k = e - s
        if k < 2:
            continue
        sums = np.zeros(nc)
        cnts = np.zeros(nc)
        tot_s = 0.0
        tot_n = 0.0
        for t in range(k):
            if t > 0:
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_all = tot_s / tot_n if tot_n > 0 else 0.0
                    has = cnts > 0
                    X[s + t, has] = sums[has] / cnts[has] - mean_all
                    G[s + t, has] = cnts[has] / (cnts[has] + K0)
            c, val = ci[s + t], v[s + t]
            if c >= 0 and np.isfinite(val):
                sums[c] += val
                cnts[c] += 1.0
                tot_s += val
                tot_n += 1.0
    return X, G


def transfer_features(runs: pd.DataFrame, courses: list[str], metric: str, W_by_year: dict[int, np.ndarray], XG: tuple[np.ndarray, np.ndarray] | None = None,
                      lam: float = LAM) -> pd.DataFrame:
    """行ごとの 転移適性 ct と根拠の量 cn。今回のコース C の列で重みづけ（自分のコース C は含めない: 重み行列の対角が 0）。"""
    X, G = XG if XG is not None else asof_vectors(runs, courses, metric)
    c2i = {c: i for i, c in enumerate(courses)}
    ci = runs["course"].map(c2i).fillna(-1).astype(int).to_numpy()
    yr = runs["year"].to_numpy()
    ct = np.full(len(runs), np.nan, dtype=np.float32)
    cn = np.zeros(len(runs), dtype=np.float32)
    for y, W in W_by_year.items():
        for c in np.unique(ci[(yr == y) & (ci >= 0)]):
            idx = np.flatnonzero((yr == y) & (ci == c))
            w = W[c]
            if not np.any(w > 0):
                cn[idx] = 0.0
                continue
            a = G[idx] * w[None, :]
            den = a.sum(1)
            num = (a * X[idx]).sum(1)
            ct[idx] = np.where(den > 0, num / (den + lam), np.nan)
            cn[idx] = den
    return pd.DataFrame({f"ct_{metric}": ct, f"cn_{metric}": cn}, index=runs.index)
