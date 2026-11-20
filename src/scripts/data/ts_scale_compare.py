"""TS の「尺度」の比べ（読むだけ・JSON を書く）。現行（標準偏差で割る）/ 頑健な尺度 / 外れ値の打ち切り。

入力（ホストのローカルだけ。GCS は読まない）: 馬別ラップ（horse_lap の JSONL）。run_axes.with_ts と同じ作りで x（補正後ピーク）まで作る。
出力 JSON:
  outliers … 問題のコースの x の外れ値の正体（レース内の他馬の x と並べる。レース全体のずれ = offset の失敗 / 1 頭だけ = 実走・記録）
  compare  … 尺度の候補ごとの ① ts の分布 ② 1 走の再現性（馬単位ブートストラップ・現行との差の区間）③ レース内順位相関（レース単位ブートストラップ）
  尺度は 2024〜2025 年の走だけで当てる（mu・sd・打ち切り点）。2026 年は同じ尺度を当てて参考にだけ見る（採否に使わない）。
候補:
  cur    現行（コース平均・SD。コースの頭数が少ないと距離帯へ縮める）
  rob    SD を頑健な尺度 (p95-p05)/3.29 に替える（縮め方は現行と同じ）
  mad    SD を 1.4826×MAD に替える（同上）
  ex20   現行だが、ピーク 20 秒以上の走（歩く走）を mu・sd を出す走から外す
  打ち切りは x をコースの 0.5%/99.5% 点（01 は 1%/99%）で切ってから現行と同じ平均・SD にする。走数が 300 未満のコースの扱いで 3 通り:
  winabs 旧案。芝ダ×距離帯の x の分位点（絶対値）。帯の平均とずれたコースでは片側だけ切れる
  winskip 案 (1)。走数 300 未満のコースは打ち切らない
  winz   案 (2)。帯の z=(x−コース平均)/コース SD の分位点を、そのコースの平均・SD で x に戻す
  robwinz winz の尺度を頑健（rob）にしたもの
  評価は全尺度とも ピーク 20 秒未満の走だけで行う（20 秒以上の走は件数だけ別に出す）。
使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I /tmp/ts_scale_compare.py OUT.json /home/jovyan/acq/horse_lap/horse_lap_*.jsonl
  環境変数 TS_CACHE=/tmp/xxx.pkl があれば、with_ts の結果を保存して 2 回目から読む（試行錯誤用）。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from src.research.race import run_axes as ra  # noqa: E402

COURSE = ["venue", "surface", "distance"]
SHRINK_MEAN = 100.0
SHRINK_SD = 200.0
DIST_ORDER = ["〜1200m", "1300〜1400m", "1500〜1700m", "1800〜2000m", "2100〜2400m", "2500m〜"]
VARIANTS = ["cur", "ex20", "rob", "mad", "winabs", "winskip", "winz", "win01skip", "win01z", "robwinz"]
ABNORMAL_PEAK = 20.0     # ラスト 3 区間の最速が 20 秒/200m 以上 = 走っていない（run_axes.ABNORMAL_PACE と同じ値）
BOOT = 500
MIN_HORSES = 150
WIN_MIN_N = 300          # 打ち切り点をコース単位で出す最小の走数（足りなければ芝ダ・距離帯の点を使う）
FOCUS = [("京都", "芝", 1600), ("小倉", "芝", 1200), ("中山", "ダート", 2400)]
FIT_YEARS = (2024, 2025)


def dist_band(d: float) -> str:
    d = int(d)
    return DIST_ORDER[0] if d <= 1200 else DIST_ORDER[1] if d <= 1400 else DIST_ORDER[2] if d <= 1700 else DIST_ORDER[3] if d <= 2000 else DIST_ORDER[4] if d <= 2400 else DIST_ORDER[5]


# ── 尺度 ─────────────────────────────────────────────────────────────────

def spread_sd(s: pd.Series) -> float:
    return float(s.std())


def spread_rob(s: pd.Series) -> float:
    q = s.quantile([0.05, 0.95])
    return float((q[0.95] - q[0.05]) / 3.29)


def spread_mad(s: pd.Series) -> float:
    m = s.median()
    return float(1.4826 * (s - m).abs().median())


def row_params(h: pd.DataFrame, x: pd.Series, fit: np.ndarray, spread) -> tuple[np.ndarray, np.ndarray]:
    """horse_ts.course_ts と同じ縮め方のコース平均 mu・尺度 sd を各行に付けて返す（fit の走で出す。fit に無いコースは距離帯の値）。"""
    t = h[COURSE].assign(x=x.values, band=(h["distance"] // 400 * 400).astype(int).values)
    f = t[fit]
    g = f.groupby(COURSE)["x"]
    c = pd.DataFrame({"n": g.size(), "mu": g.mean(), "sp": g.agg(spread)}).reset_index()
    c["band"] = (c["distance"] // 400 * 400).astype(int)
    pool = f.groupby(["surface", "band"])["x"].agg(mu_p="mean", sp_p=spread).reset_index()
    c = c.merge(pool, on=["surface", "band"], how="left")
    c["mu_s"] = (c["n"] * c["mu"] + SHRINK_MEAN * c["mu_p"]) / (c["n"] + SHRINK_MEAN)
    c["sd_s"] = np.sqrt((c["n"] * c["sp"].fillna(c["sp_p"]) ** 2 + SHRINK_SD * c["sp_p"] ** 2) / (c["n"] + SHRINK_SD))
    m = t.merge(c[COURSE + ["mu_s", "sd_s"]], on=COURSE, how="left").merge(pool, on=["surface", "band"], how="left")
    return m["mu_s"].fillna(m["mu_p"]).to_numpy(), m["sd_s"].fillna(m["sp_p"]).to_numpy()


def winsor_bounds(h: pd.DataFrame, fit: np.ndarray, q: float, small: str, spread=spread_sd) -> tuple[np.ndarray, np.ndarray]:
    """各行に当てる打ち切りの下限・上限（q と 1-q の分位点）。fit の走で出す。NaN は「打ち切らない」。
    走数が WIN_MIN_N 以上のコースは、そのコースの x の分位点。それ未満のコース（small で選ぶ）:
      abs  … 芝ダ・距離帯の x の分位点（絶対値。帯の平均とずれたコースでは片側だけ切れる = 旧案）
      skip … 打ち切らない
      z    … 帯の z = (x − コース平均)/コース SD の分位点を、そのコースの平均・SD で x に戻す"""
    f = h[fit]
    fb = (f["distance"] // 400 * 400).astype(int)
    g = f.groupby(COURSE)["x"]
    c = pd.DataFrame({"lo": g.quantile(q), "hi": g.quantile(1 - q), "n": g.size()}).reset_index()
    key = h[COURSE].assign(band=(h["distance"] // 400 * 400).astype(int).values)
    m = key.merge(c, on=COURSE, how="left")
    use_c = (m["n"].fillna(0) >= WIN_MIN_N).to_numpy()
    lo_c, hi_c = m["lo"].to_numpy(), m["hi"].to_numpy()
    if small == "skip":
        return np.where(use_c, lo_c, np.nan), np.where(use_c, hi_c, np.nan)
    if small == "abs":
        pool = pd.DataFrame({"lo_p": f.groupby(["surface", fb])["x"].quantile(q), "hi_p": f.groupby(["surface", fb])["x"].quantile(1 - q)}).reset_index().rename(columns={"distance": "band"})
        m = key.merge(pool, on=["surface", "band"], how="left")
        return np.where(use_c, lo_c, m["lo_p"].to_numpy()), np.where(use_c, hi_c, m["hi_p"].to_numpy())
    assert small == "z"
    mu, sd = row_params(h, h["x"], fit, spread)
    z = pd.Series((h["x"].to_numpy() - mu) / sd, index=h.index)
    zf = z[fit]
    pz = pd.DataFrame({"lo_z": zf.groupby([f["surface"], fb]).quantile(q), "hi_z": zf.groupby([f["surface"], fb]).quantile(1 - q)}).reset_index().rename(columns={"distance": "band"})
    m = key.merge(pz, on=["surface", "band"], how="left")
    return np.where(use_c, lo_c, mu + m["lo_z"].to_numpy() * sd), np.where(use_c, hi_c, mu + m["hi_z"].to_numpy() * sd)


MODES = {
    # 名前: (異常走を当てる走から外す, 尺度, (打ち切りの下側の分位 q, 走数が少ないコースの扱い)。None は打ち切らない)
    "cur": (False, spread_sd, None),
    "ex20": (True, spread_sd, None),
    "rob": (False, spread_rob, None),
    "mad": (False, spread_mad, None),
    "winabs": (False, spread_sd, (0.005, "abs")),       # 旧案（走数が少ないコースは帯の絶対値）
    "winskip": (False, spread_sd, (0.005, "skip")),     # 案 (1)
    "winz": (False, spread_sd, (0.005, "z")),           # 案 (2)
    "win01skip": (False, spread_sd, (0.01, "skip")),
    "win01z": (False, spread_sd, (0.01, "z")),
    "robwinz": (False, spread_rob, (0.005, "z")),
}


def mode_bounds(h: pd.DataFrame, mode: str, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    exclude, spread, win = MODES[mode]
    if win is None:
        return None
    if exclude:
        fit = fit & (h["peak"].to_numpy() < ABNORMAL_PEAK)
    return winsor_bounds(h, fit, win[0], win[1], spread)


def scale_ts(h: pd.DataFrame, mode: str, fit: np.ndarray) -> pd.Series:
    """horse_ts.course_ts と同じ縮め方で ts を作る（mu・sd・打ち切り点は fit の走で出し、全行へ当てる）。mode は MODES のキー。
    ex20 系は、ピークが ABNORMAL_PEAK 以上の走（完走扱いだが歩くような走）を、mu・sd・打ち切り点を出す走から外す（ts はそれでも出す）。"""
    exclude, spread, win = MODES[mode]
    fit_eff = fit & (h["peak"].to_numpy() < ABNORMAL_PEAK) if exclude else fit
    x = h["x"].copy()
    bd = mode_bounds(h, mode, fit)
    if bd is not None:
        x = pd.Series(np.minimum(np.maximum(x.to_numpy(), np.nan_to_num(bd[0], nan=-np.inf)), np.nan_to_num(bd[1], nan=np.inf)), index=h.index)
    mu, sd = row_params(h, x, fit_eff, spread)
    return pd.Series(50 - 10 * (x.to_numpy() - mu) / sd, index=h.index)


def clip_rates(h: pd.DataFrame, mode: str, fit: np.ndarray, min_n: int = 50) -> dict:
    """fit の走のうち、打ち切り点の外にある割合（下側・上側）をコースごとに出し、走数の大小で要約する（理想は両側とも 0.5% 前後）。"""
    bd = mode_bounds(h, mode, fit)
    if bd is None:
        return {}
    lo, hi = bd
    x = h["x"].to_numpy()
    d = h[COURSE].assign(below=np.where(np.isnan(lo), False, x < lo), above=np.where(np.isnan(hi), False, x > hi))[fit]
    c = d.groupby(COURSE).agg(n=("below", "size"), lo=("below", "mean"), hi=("above", "mean")).reset_index()
    c = c[c["n"] >= min_n]
    out = {}
    for name, sub in (("n>=300", c[c["n"] >= WIN_MIN_N]), (f"{min_n}<=n<300", c[c["n"] < WIN_MIN_N])):
        if sub.empty:
            continue
        out[name] = {"courses": int(len(sub)), "lo_median": round(float(sub["lo"].median()) * 100, 2), "hi_median": round(float(sub["hi"].median()) * 100, 2),
                     "lo_max": round(float(sub["lo"].max()) * 100, 2), "hi_max": round(float(sub["hi"].max()) * 100, 2),
                     "courses_either_gt2pct": int(((sub["lo"] > 0.02) | (sub["hi"] > 0.02)).sum()),
                     "courses_one_sided": int((((sub["lo"] > 0.02) & (sub["hi"] < 0.0025)) | ((sub["hi"] > 0.02) & (sub["lo"] < 0.0025))).sum())}
    small = c[c["n"] < WIN_MIN_N].sort_values("n", ascending=False)
    if len(small):
        out["small_pooled_pct"] = {"lo": round(float((small["lo"] * small["n"]).sum() / small["n"].sum()) * 100, 2), "hi": round(float((small["hi"] * small["n"]).sum() / small["n"].sum()) * 100, 2),
                                   "runs": int(small["n"].sum())}
    out["small_courses"] = [{"course": f"{r.venue}{r.surface}{r.distance}", "n": int(r.n), "lo_pct": round(r.lo * 100, 2), "hi_pct": round(r.hi * 100, 2)} for r in small.itertuples()]
    return out


# ── 1 走の再現性（馬単位ブートストラップ。尺度どうしは同じ標本で比べる） ───────────

def rho_from_r(r: np.ndarray) -> np.ndarray:
    big_r = 2 * r / (1 + r)
    return big_r / (4 - 3 * big_r)


def four_runs(g: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    """馬ごとに古い順 4 走を取り (馬, 4, 尺度) の配列にする。4 走ない馬は除く。"""
    g = g.dropna(subset=cols).sort_values(["horse_id", "date"]).copy()      # NaN の走は n を数える前に落とす
    g["k"] = g.groupby("horse_id").cumcount()
    cnt = g.groupby("horse_id")["k"].transform("size")
    g = g[(cnt >= 4) & (g["k"] < 4)]
    if len(g) < 4 * MIN_HORSES:
        return None
    return g[cols].to_numpy().reshape(-1, 4, len(cols))


def colcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (n, V) → 列ごとの相関。"""
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def spearman_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra_ = pd.DataFrame(a).rank().to_numpy()
    rb_ = pd.DataFrame(b).rank().to_numpy()
    return colcorr(ra_, rb_)


def reliability(g: pd.DataFrame, names: list[str], rng: np.random.Generator) -> dict | None:
    cols = [f"ts_{n}" for n in names]
    arr = four_runs(g, cols)
    if arr is None:
        return None
    a, b = arr[:, [0, 2], :].mean(1), arr[:, [1, 3], :].mean(1)
    n = len(arr)
    point = rho_from_r(colcorr(a, b))
    sp = rho_from_r(spearman_cols(a, b))
    boots = np.empty((BOOT, len(names)))
    for t in range(BOOT):
        i = rng.integers(0, n, n)
        boots[t] = rho_from_r(colcorr(a[i], b[i]))
    out = {"horses4": int(n)}
    for k, nm in enumerate(names):
        d = {"rho": round(float(point[k]), 3), "ci": [round(float(np.percentile(boots[:, k], 2.5)), 3), round(float(np.percentile(boots[:, k], 97.5)), 3)],
             "rho_spearman": round(float(sp[k]), 3)}
        if nm != "cur":
            diff = boots[:, k] - boots[:, 0]
            d["diff_vs_cur"] = round(float(point[k] - point[0]), 4)
            d["diff_ci"] = [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)]
        out[nm] = d
    return out


# ── レース内順位相関（レース単位ブートストラップ） ───────────────────────────────

def race_spearman(h: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """レースごとの corr(ts のレース内順位, 着順)（horse_ts.main と同じ定義）。返り値: (race_id 配列, (レース, 尺度) の配列)。"""
    fo = h["finish_order"].astype(float)
    key = h["race_id"]
    cols = []
    for nm in names:
        rk = h[f"ts_{nm}"].groupby(key).rank(ascending=False)
        d = pd.DataFrame({"a": rk, "b": fo, "k": key})
        d["ab"], d["aa"], d["bb"] = d["a"] * d["b"], d["a"] ** 2, d["b"] ** 2
        s = d.groupby("k")[["a", "b", "ab", "aa", "bb"]].sum()
        n = d.groupby("k").size()
        cov = s["ab"] - s["a"] * s["b"] / n
        va, vb = s["aa"] - s["a"] ** 2 / n, s["bb"] - s["b"] ** 2 / n
        cols.append((cov / np.sqrt(va * vb)).rename(nm))
    m = pd.concat(cols, axis=1)
    return m.index.to_numpy(), m.to_numpy()


def race_boot(m: np.ndarray, names: list[str], rng: np.random.Generator) -> dict:
    # 注: 同じレースは同じコースで、尺度の違いは ts の正の一次変換（打ち切りを除く）なのでレース内順位は変わらない。diff_vs_cur は比較表に使わない（JSON には残す）。
    ok = np.isfinite(m).all(1)
    m = m[ok]
    n = len(m)
    mean = m.mean(0)
    boots = np.empty((BOOT, len(names)))
    for t in range(BOOT):
        boots[t] = m[rng.integers(0, n, n)].mean(0)
    out = {"races": int(n)}
    for k, nm in enumerate(names):
        d = {"mean": round(float(mean[k]), 4), "ci": [round(float(np.percentile(boots[:, k], 2.5)), 4), round(float(np.percentile(boots[:, k], 97.5)), 4)]}
        if nm != "cur":
            diff = boots[:, k] - boots[:, 0]
            d["diff_vs_cur"] = round(float(mean[k] - mean[0]), 5)
            d["diff_ci"] = [round(float(np.percentile(diff, 2.5)), 5), round(float(np.percentile(diff, 97.5)), 5)]
        out[nm] = d
    return out


def pooled_corr(h: pd.DataFrame, names: list[str], rng: np.random.Generator) -> dict:
    """レースをまたいだ全体の相関 corr(ts, 着順の位置 (着順-1)/(頭数-1))。レース単位ブートストラップ。コース間の尺度の揃い方が効く。"""
    rel = (h["finish_order"] - 1) / (h["nent"].clip(lower=2) - 1)
    key = h["race_id"].to_numpy()
    codes, uniq = pd.factorize(key)
    out = {}
    stats = {}
    for nm in names:
        t = h[f"ts_{nm}"].to_numpy()
        d = pd.DataFrame({"c": codes, "n": 1.0, "t": t, "r": rel.to_numpy(), "tt": t * t, "rr": rel.to_numpy() ** 2, "tr": t * rel.to_numpy()})
        stats[nm] = d.groupby("c").sum().sort_index().to_numpy()      # 列: n,t,r,tt,rr,tr
    nr = len(uniq)

    def corr_of(s: np.ndarray) -> float:
        n, st, sr, stt, srr, st_r = s.sum(0)[[0, 1, 2, 3, 4, 5]]
        return float((st_r - st * sr / n) / np.sqrt((stt - st ** 2 / n) * (srr - sr ** 2 / n)))

    pts = {nm: corr_of(stats[nm]) for nm in names}
    boots = {nm: np.empty(BOOT) for nm in names}
    for t in range(BOOT):
        i = rng.integers(0, nr, nr)
        for nm in names:
            boots[nm][t] = corr_of(stats[nm][i])
    for nm in names:
        d = {"corr": round(pts[nm], 4), "ci": [round(float(np.percentile(boots[nm], 2.5)), 4), round(float(np.percentile(boots[nm], 97.5)), 4)]}
        if nm != "cur":
            diff = boots[nm] - boots["cur"]
            d["diff_vs_cur"] = round(pts[nm] - pts["cur"], 5)
            d["diff_ci"] = [round(float(np.percentile(diff, 2.5)), 5), round(float(np.percentile(diff, 97.5)), 5)]
        out[nm] = d
    return out


# ── ts の分布 ────────────────────────────────────────────────────────────

def ts_dist(g: pd.DataFrame, names: list[str]) -> dict:
    out = {"n": int(len(g))}
    for nm in names:
        s = g[f"ts_{nm}"]
        q = s.quantile([0.05, 0.5, 0.95])
        out[nm] = {"sd": round(float(s.std()), 2), "skew": round(float(s.skew()), 2), "p05": round(float(q[0.05]), 1), "p50": round(float(q[0.5]), 1),
                   "p95": round(float(q[0.95]), 1), "min": round(float(s.min()), 1), "max": round(float(s.max()), 1)}
    return out


def course_spread_summary(g: pd.DataFrame, names: list[str], min_runs: int = 300) -> dict:
    """コースごとの ts の SD・歪みを出し、そのばらつき（理想は SD=10・歪み 0）を見る。"""
    out = {}
    for nm in names:
        s = g.groupby(COURSE)[f"ts_{nm}"].agg(["size", "std", "skew"])
        s = s[s["size"] >= min_runs]
        out[nm] = {"courses": int(len(s)), "sd_median": round(float(s["std"].median()), 2), "sd_min": round(float(s["std"].min()), 2), "sd_max": round(float(s["std"].max()), 2),
                   "sd_iqr": round(float(s["std"].quantile(0.75) - s["std"].quantile(0.25)), 2),
                   "skew_median": round(float(s["skew"].median()), 2), "skew_min": round(float(s["skew"].min()), 2), "skew_max": round(float(s["skew"].max()), 2),
                   "courses_abs_skew_gt1": int((s["skew"].abs() > 1).sum())}
    return out


# ── 外れ値の正体 ──────────────────────────────────────────────────────────

def classify(others_med_z: float, n_other_hi: int, finish: int, nent: int) -> str:
    if others_med_z >= 1.0:
        return "レース水準（offset の失敗か、馬場・ペースの読み違い）"
    if n_other_hi >= 2:
        return "同じレースの複数頭（その馬群の失速）"
    if finish > nent * 0.6:
        return "実走（1 頭の失速・故障疑い）"
    return "単独で上位（記録の誤りを疑う）"


def outlier_report(h: pd.DataFrame, top: int = 12) -> dict:
    """x（補正後ピーク。大きいほど遅い）が極端な走を、同じレースの他馬の x と並べる。z は コース内の 中央値・頑健 SD（(p95-p05)/3.29）で測る。"""
    h = h.copy()
    g = h.groupby(COURSE)
    h["x_med"] = g["x"].transform("median")
    h["x_rsd"] = g["x"].transform(spread_rob)
    h["z"] = (h["x"] - h["x_med"]) / h["x_rsd"]
    gp = h.groupby(COURSE)["peak"]
    h["pz"] = (h["peak"] - gp.transform("median")) / gp.transform(spread_rob)
    race_groups = {rid: g_ for rid, g_ in h.groupby("race_id")}
    day_key = h["date"].astype(str) + "|" + h["venue"] + "|" + h["surface"]
    h["day_key"] = day_key
    race_med = h.groupby("race_id")["z"].median()
    race_day = h.groupby("race_id")["day_key"].first()
    day_race_med = pd.DataFrame({"day": race_day, "m": race_med}).reset_index()

    def describe(row: pd.Series) -> dict:
        rg = race_groups[row["race_id"]]
        oth = rg[rg["horse_id"] != row["horse_id"]]
        omed = float(oth["z"].median()) if len(oth) else float("nan")
        n_hi = int((oth["z"] > 2.0).sum())
        same_day = day_race_med[(day_race_med["day"] == row["day_key"]) & (day_race_med["race_id"] != row["race_id"])]["m"]
        return {"race_id": row["race_id"], "date": str(row["date"].date()), "venue": row["venue"], "surface": row["surface"], "distance": int(row["distance"]),
                "track_condition": row["track_condition"], "horse_id": row["horse_id"], "horse_name": row["horse_name"], "finish": int(row["finish_order"]), "nent": int(row["nent"]),
                "peak": round(float(row["peak"]), 2), "offset": round(float(row["offset"]), 3), "x": round(float(row["x"]), 3), "z": round(float(row["z"]), 2),
                "peak_z_without_offset": round(float(row["pz"]), 2), "last4_laps": [round(float(v), 1) for v in row["lt"][-4:]],
                "others_med_z": round(omed, 2), "others_z_gt2": n_hi, "race_offset": round(float(row["offset"]), 3),
                "same_day_other_races_med_z": round(float(same_day.median()), 2) if len(same_day) else None,
                "class": classify(omed, n_hi, int(row["finish_order"]), int(row["nent"]))}

    out = {}
    for v, s, d in FOCUS:
        c = h[(h["venue"] == v) & (h["surface"] == s) & (h["distance"] == d)]
        if c.empty:
            continue
        ex = c.sort_values("x", ascending=False).head(top)
        fast = c.sort_values("x").head(3)
        out[f"{v}{s}{d}"] = {
            "runs": int(len(c)), "races": int(c["race_id"].nunique()), "x_sd": round(float(c["x"].std()), 3), "x_rsd": round(float(spread_rob(c["x"])), 3),
            "x_median": round(float(c["x"].median()), 3), "n_z_gt3": int((c["z"] > 3).sum()), "n_z_gt5": int((c["z"] > 5).sum()),
            "sd_without_top": {str(k): round(float(c["x"].sort_values().iloc[:-k].std()), 3) for k in (5, 10, 20)},
            "slow_tail": [describe(r) for _, r in ex.iterrows()],
            "fast_tail": [describe(r) for _, r in fast.iterrows()],
        }
    # 全コースで z>3 の走のうち、レース水準のずれによるものの割合
    big = h[h["z"] > 3.0]
    cls = []
    for _, r in big.iterrows():
        rg = race_groups[r["race_id"]]
        oth = rg[rg["horse_id"] != r["horse_id"]]
        cls.append(classify(float(oth["z"].median()) if len(oth) else 0.0, int((oth["z"] > 2.0).sum()), int(r["finish_order"]), int(r["nent"])))
    vc = pd.Series(cls).value_counts()
    out["all_courses_z_gt3"] = {"runs": int(len(big)), "share_of_runs": round(float(len(big) / len(h)), 4), "by_class": {k: int(v) for k, v in vc.items()}}
    # 同じ判定を レース単位に: レース中央値の z が大きいレースの数
    rm = pd.DataFrame({"med": race_med, "surface": h.groupby("race_id")["surface"].first(), "cond": h.groupby("race_id")["track_condition"].first(), "n": h.groupby("race_id").size()})
    rm = rm[rm["n"] >= 8]
    out["race_level_med_z"] = {"iqr_sd": round(float((rm["med"].quantile(0.75) - rm["med"].quantile(0.25)) / 1.349), 3), "gt1": int((rm["med"] > 1).sum()), "lt_m1": int((rm["med"] < -1).sum()),
                               "gt1_by_cond": {k: [int(v), int((rm["cond"] == k).sum())] for k, v in rm[rm["med"] > 1].groupby("cond").size().items()}}
    out["races_with_med_z_gt1"] = int((race_med > 1.0).sum())
    out["races"] = int(len(race_med))
    return out


# ── 入口 ─────────────────────────────────────────────────────────────────

def load_h(paths: list[str]) -> pd.DataFrame:
    cache = os.environ.get("TS_CACHE")
    if cache and Path(cache).exists():
        return pd.read_pickle(cache)
    runs = ra.read_runs(paths)
    h = ra.with_ts(runs, ra.default_years())
    h = h[h["surface"].isin(["芝", "ダート"]) & h["offset"].notna()].copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.reset_index(drop=True)
    if cache:
        h.to_pickle(cache)
    return h


def evaluate(h: pd.DataFrame, names: list[str], rng: np.random.Generator, bands: bool = True) -> dict:
    ok = h[[f"ts_{n}" for n in names]].notna().all(1)
    h = h[ok].copy()                      # ts が NaN の走は評価から外す（件数は出力に残す）
    res = {"dropped_nan_ts": int((~ok).sum()), "runs": int(len(h)), "races": int(h["race_id"].nunique())}
    res["dist_all"] = ts_dist(h, names)
    res["course_spread"] = course_spread_summary(h, names)
    res["reliability_all"] = reliability(h, names, rng)
    race_ids, m = race_spearman(h, names)
    res["race_spearman"] = race_boot(m, names, rng)
    res["pooled_corr"] = pooled_corr(h, names, rng)
    if bands:
        res["bands"] = []
        for s in ("芝", "ダート"):
            for b in DIST_ORDER:
                g = h[(h["surface"] == s) & (h["dband"] == b)]
                if len(g) < 500:
                    continue
                res["bands"].append({"surface": s, "band": b, "dist": ts_dist(g, names), "rel": reliability(g, names, rng)})
        res["focus_courses"] = {}
        for v, s, d in FOCUS:
            g = h[(h["venue"] == v) & (h["surface"] == s) & (h["distance"] == d)]
            if len(g):
                res["focus_courses"][f"{v}{s}{d}"] = ts_dist(g, names)
    return res


def main() -> None:
    out_path = Path(sys.argv[1])
    paths = sys.argv[2:]
    h = load_h(paths)
    h["dband"] = [dist_band(d) for d in h["distance"]]
    h["year"] = h["date"].dt.year
    rng = np.random.default_rng(20261118)

    fit = h["year"].isin(FIT_YEARS).to_numpy()
    # 検算: 全走で当てた winz が horse_ts.course_ts の ts と一致する（本番の TS は 2026-11-20 から winz）
    chk = scale_ts(h, "winz", np.ones(len(h), bool))
    max_diff = float((chk - h["ts"]).abs().max())
    print("check winz vs course_ts max abs diff:", max_diff, flush=True)
    assert max_diff < 1e-6, f"winz が horse_ts.course_ts と一致しない: {max_diff}"

    for nm in VARIANTS:
        h[f"ts_{nm}"] = scale_ts(h, nm, fit)
    print("variants built", flush=True)

    result = {"meta": {"built": date.today().isoformat(), "runs": int(len(h)), "races": int(h["race_id"].nunique()), "date_from": str(h["date"].min().date()),
                       "date_to": str(h["date"].max().date()), "fit_years": list(FIT_YEARS), "boot": BOOT, "check_cur_max_abs_diff": max_diff,
                       "note_2026": "with_ts の offset 係数（lap_correction）とコース平均は 2026 年を含む全期間で当てている。尺度（mu・sd・打ち切り点）だけを 2024-25 で当てているので、"
                                    "ref_2026 は完全な事後ではなく参考値（採否に使わない）"}}
    result["outliers"] = outlier_report(h)
    print("outliers done", flush=True)
    ab = h["peak"] >= ABNORMAL_PEAK
    result["abnormal_peak"] = {"threshold": ABNORMAL_PEAK, "runs": int(ab.sum()), "share": round(float(ab.mean()), 5),
                               "by_course": {f"{k[0]}{k[1]}{k[2]}": int(v) for k, v in h[ab].groupby(COURSE).size().items()}}
    result["clip_rates_fit_2024_2025"] = {nm: clip_rates(h, nm, fit) for nm in VARIANTS if MODES[nm][2] is not None}
    main_set = h[fit & ~ab.to_numpy()].copy()
    result["eval_2024_2025"] = evaluate(main_set, VARIANTS, rng)
    print("eval 2024-25 done", flush=True)
    ref = h[(h["year"] == 2026) & ~ab].copy()
    result["ref_2026"] = evaluate(ref, VARIANTS, rng, bands=False)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
