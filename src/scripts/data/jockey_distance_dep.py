"""距離の長さによって、騎手への依存度（騎手が成績・位置取りを動かす大きさ）が変わるかを調べる。

python -m src.scripts.data.jockey_distance_dep build     # tp-59 の master.parquet ＋ tp-61 の位置取りから as-of の特徴を作る（<JDD_OUT>/frame.parquet）
python -m src.scripts.data.jockey_distance_dep analyze   # 分散分解・騎手交代の対比・持ち越し・有力騎手の残差と回収率（<JDD_OUT>/result.json）
python -m src.scripts.data.jockey_distance_dep report    # 章 HTML（docs/html/modeling/chapters/jockey_distance_dependence.html）

事前に固定した設計（結果を見てから増やさない。実行前にコミットする）
- 対象: 平地（障害は除く）。距離帯 b1=1000〜1400 / b2=1500〜1800 / b3=1900〜2200 / b4=2300〜2800 / b5=3000 以上。芝・ダート別。
  ダートの b5 は少数のため b4 に含め、検定には b1・b2・b3・b4p（= b4 + b5）を使う。芝の b4・b5 は記述として単独でも出す。
- 成績の応答 z = レース内の着順の正規スコア Φ^-1((n−着順+0.5)/n)（大きいほど良い。n = 着順のついた頭数）。
- 定義 (a) 分散分解: 馬の効果＋騎手の効果を、バックフィッティング（弱いリッジ λ_馬=1・λ_騎手=2、40 反復）で推定。
  騎手の真の効果の分散は『馬で二分した 2 つの独立な標本で推定した同じ騎手の効果の共分散』（ノイズが独立なので不偏）。割合 = その共分散 / z の分散。
  馬の分け方は 5 通りの乱数分割の平均。区間は騎手単位のブートストラップ（400 回、距離帯をまたいで同じ抽出）。
  応答を『今走の初角位置 − その馬の過去 3 走の平均の初角位置（as-of、2 走以上、レース内で中心化）』にしたものを、位置取りへの影響とする。
- 定義 (b) 騎手交代の対比: 同じ馬が前走と違う騎手になった走だけ。z ~ 前走の z + 前々走以前の z の平均 + Δq（今走の騎手の as-of 勝率（直近 365 日、縮約 k=50）− 前走の騎手の同じ値、全体の標準偏差で割る）。レース単位のクラスタ頑健 SE。
- 定義 (c) 持ち越し: 騎手 × 距離帯 × 芝ダ × 年（50 騎乗以上）の z の平均、およびモデル予測に対する 3 着内の残差（実績 − p_top3）の平均を、翌年と比べた相関。
- 定義 (d) 有力騎手 = その日より前の直近 365 日の平地勝利数の上位 10 人（5・20 は感度）。jockey-change-ev と同じ。
  モデル予測に対する残差（勝ち・3 着内。モデルは tp-59 の master の p_*）、単勝・複勝の回収率（1 出走 100 円、レース単位ブートストラップ）、
  市場の単勝確率（q_win）で統制したロジスティック回帰の有力騎手の係数。
- 検定の族（K = 32）: 距離への依存の変化 14 本 = [分散割合の傾き(芝・ダ) 2, 位置取りの分散割合の傾き 2, 騎手交代の傾き b1−b4p 2, 持ち越し相関 b1−b4p 2,
  有力騎手の 3 着内残差 b1−b4p 2, 勝ち残差 b1−b4p 2, 市場統制後の係数 b1−b4p 2]（BH 補正）と、有力騎手の回収率 > 100% の 18 本 = [(芝 b1,b2,b3,b4,b5 + ダ b1,b2,b3,b4p) 9 セル × 単勝・複勝]（Holm 補正）。
- 期間: 発見 = 2020〜2024（モデル予測を使うものは 2022〜2024、有力騎手の順位が有効なのは 2021 年以降）、確認 = 2025 を 1 回、2026 は記述統計のみ（選択に使わない）。
  年をまたぐ再現（LOYO 相当）は、分散割合の傾きを年ごとに出して符号を見る。
- 特徴量化の見込み: 発見期間でロジスティック回帰（オフセット = logit p）に『有力 × 距離帯』を足し、2025 の対数損失の差（レース単位ブートストラップ）を 1 回見る。モデルの学習はしない。
- キーは jockey_id・horse_id。市場オッズは分析（統制・回収率）にのみ使い、特徴量にしない。
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.scripts.data import tp59_lib as L
from src.scripts.data.jockey_switch_ev import elite_flags

ROOT = Path(__file__).resolve().parents[3]
TP59_OUT = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
TP61 = Path(os.environ.get("TP61_DIR", str(ROOT / "data/local/modeling/tp61")))
OUT = Path(os.environ.get("JDD_OUT", "/home/jovyan/work/jdd_out"))
WINDOW_DAYS = 365
BANDS = ["b1", "b2", "b3", "b4", "b5"]
TEST_BANDS = ["b1", "b2", "b3", "b4p"]
BAND_LABEL = {"b1": "1000〜1400", "b2": "1500〜1800", "b3": "1900〜2200", "b4": "2300〜2800", "b5": "3000以上", "b4p": "2300以上"}
SURF = {"芝": "芝", "ダ": "ダート"}
Q_K = 50.0
N_SPLIT = 5
N_BOOT = 400
RATE_PRIOR = 0.07


def band_of(dist: pd.Series) -> pd.Series:
    x = dist.to_numpy(float)
    return pd.Series(np.select([x <= 1400, x <= 1800, x <= 2200, x <= 2800], ["b1", "b2", "b3", "b4"], default="b5"), index=dist.index, dtype=object)


# ------------------------------------------------------------------ 特徴の作成（as-of）
def jockey_rate(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日 × 騎手の『その日より前の直近 365 日』の平地の勝利数・騎乗数（当日を含まない）。"""
    daily = d.groupby(["date", "jockey_id"]).agg(w=("win", "sum"), n=("win", "size")).reset_index()
    days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    w = daily.pivot(index="date", columns="jockey_id", values="w").reindex(days).fillna(0.0)
    n = daily.pivot(index="date", columns="jockey_id", values="n").reindex(days).fillna(0.0)
    cw, cn = w.cumsum(), n.cumsum()
    return cw.shift(1).fillna(0.0) - cw.shift(1 + WINDOW_DAYS).fillna(0.0), cn.shift(1).fillna(0.0) - cn.shift(1 + WINDOW_DAYS).fillna(0.0)


def lookup(mat: pd.DataFrame, jid: pd.Series, date: pd.Series) -> np.ndarray:
    ci = mat.columns.get_indexer(jid.astype(object))
    ri = mat.index.get_indexer(date)
    arr = mat.to_numpy()
    ok = (ci >= 0) & (ri >= 0)
    out = np.full(len(jid), np.nan)
    out[ok] = arr[ri[ok], ci[ok]]
    return out


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(TP59_OUT / "master.parquet")
    d = d[(d["surface"] != "障") & d["jockey_id"].notna() & (d["finish_position"] > 0) & d["distance"].ge(1000)].copy()
    d["jockey_id"] = d["jockey_id"].astype("string")
    d["prev_jockey_id"] = d["prev_jockey_id"].astype("string")
    d["surf"] = d["surface"].map({"芝": "芝", "ダート": "ダ"})
    d = d[d["surf"].notna()]
    d["band"] = band_of(d["distance"])
    d.loc[(d["surf"] == "ダ") & (d["band"] == "b5"), "band"] = "b4"        # ダートの 3000 以上は少数のため b4 に含める
    d["band_t"] = d["band"].replace({"b4": "b4p", "b5": "b4p"})
    # 成績の正規スコア
    nfin = d.groupby("race_id")["finish_position"].transform("size")
    d["z"] = norm.ppf((nfin - d["finish_position"] + 0.5) / nfin)
    # 位置取り（tp-61）
    h = pd.read_parquet(TP61 / "horse.parquet", columns=["race_id", "horse_id", "early_norm", "pass_first"])
    d = d.merge(h, on=["race_id", "horse_id"], how="left")
    d = d.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    prev3 = pd.concat([g["early_norm"].shift(k) for k in (1, 2, 3)], axis=1)
    d["usual_pos"] = prev3.mean(axis=1).where(prev3.notna().sum(axis=1) >= 2)
    d["pos_chg"] = d["early_norm"] - d["usual_pos"]
    d["pos_chg_c"] = d["pos_chg"] - d.groupby("race_id")["pos_chg"].transform("mean")
    # 前走・前々走以前の成績
    d["z_prev"] = g["z"].shift(1)
    d["z_prev2m"] = pd.concat([g["z"].shift(k) for k in (2, 3, 4)], axis=1).mean(axis=1)
    d["z_prev2m"] = d["z_prev2m"].fillna(d["z_prev"])
    # 騎手の as-of
    d = d.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    ef = elite_flags(d)
    w365, n365 = jockey_rate(d)
    first_valid = d["date"].min() + pd.Timedelta(days=WINDOW_DAYS)
    d["rank_valid"] = d["date"] >= first_valid
    for top, mat in ef.items():
        d[f"el{top}"] = lookup(mat.astype(float), d["jockey_id"], d["date"])
        d[f"el{top}"] = d[f"el{top}"].fillna(0.0)
    wn, nn = lookup(w365, d["jockey_id"], d["date"]), lookup(n365, d["jockey_id"], d["date"])
    wp, npv = lookup(w365, d["prev_jockey_id"], d["date"]), lookup(n365, d["prev_jockey_id"], d["date"])
    d["q_now"] = (np.nan_to_num(wn) + RATE_PRIOR * Q_K) / (np.nan_to_num(nn) + Q_K)
    d["q_prev"] = (np.nan_to_num(wp) + RATE_PRIOR * Q_K) / (np.nan_to_num(npv) + Q_K)
    d["switch"] = d["prev_jockey_id"].notna() & (d["run_no"] > 0) & (d["jockey_id"] != d["prev_jockey_id"]) & d["rank_valid"] & d["z_prev"].notna()
    d["y_top3"] = d["top3"].astype(float)
    d["y_win"] = d["win"].astype(float)
    d.to_parquet(OUT / "frame.parquet", index=False)
    print("frame", d.shape, "switch rows", int(d["switch"].sum()), "years", d["year"].value_counts().sort_index().to_dict())


# ------------------------------------------------------------------ (a) 分散分解（馬で二分した標本の共分散）
def backfit(h: np.ndarray, j: np.ndarray, y: np.ndarray, nh: int, nj: int, lam_h: float = 1.0, lam_j: float = 2.0, iters: int = 40):
    ch = np.bincount(h, minlength=nh).astype(float)
    cj = np.bincount(j, minlength=nj).astype(float)
    b = np.zeros(nj)
    for _ in range(iters):
        a = np.bincount(h, weights=y - b[j], minlength=nh) / (ch + lam_h)
        b = np.bincount(j, weights=y - a[h], minlength=nj) / (cj + lam_j)
    return b, cj


def split_fits(h_all: np.ndarray, j_all: np.ndarray, y: np.ndarray, nj: int, seed: int, nmin: int = 20):
    nh = int(h_all.max()) + 1
    bA = np.zeros((N_SPLIT, nj)); bB = np.zeros((N_SPLIT, nj)); W = np.zeros((N_SPLIT, nj))
    for s in range(N_SPLIT):
        half = np.random.default_rng(seed + s).integers(0, 2, size=nh)[h_all]
        res = []
        for k in (0, 1):
            m = half == k
            hh, uh = pd.factorize(h_all[m])
            b, cj = backfit(hh, j_all[m], y[m], len(uh), nj)
            pos = cj > 0
            b = b - (b[pos] * cj[pos]).sum() / cj[pos].sum()
            res.append((b, cj))
        (a_, ca), (b_, cb) = res
        both = (ca >= nmin) & (cb >= nmin)
        bA[s, both], bB[s, both], W[s, both] = a_[both], b_[both], (ca + cb)[both]
    return bA, bB, W


def cov_boot(bA: np.ndarray, bB: np.ndarray, W: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """cnt: [D, nj] のブートストラップ重み。戻り値: [D]（分割をまたいで平均した、騎手効果の共分散）。"""
    wt = W[None, :, :] * cnt[:, None, :]
    sw = np.maximum(wt.sum(2), 1e-9)
    mA = (wt * bA[None]).sum(2) / sw
    mB = (wt * bB[None]).sum(2) / sw
    cov = (wt * (bA[None] - mA[..., None]) * (bB[None] - mB[..., None])).sum(2) / sw
    return cov.mean(1)


def boot_counts(nj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nj, size=(N_BOOT, nj))
    cnt = np.zeros((N_BOOT, nj))
    for i in range(N_BOOT):
        cnt[i] = np.bincount(idx[i], minlength=nj)
    return cnt


def slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y: [D, K] の x に対する回帰の傾き。"""
    xc = x - x.mean()
    return ((y - y.mean(1, keepdims=True)) * xc[None]).sum(1) / (xc**2).sum()


def pval2(draws: np.ndarray) -> float:
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


def ci(draws: np.ndarray) -> list[float]:
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def share_block(d: pd.DataFrame, resp: str, years: tuple[int, int], jcode: dict, cnt: np.ndarray, seed: int) -> dict:
    """芝・ダ × 距離帯の分散割合（点推定・区間）と、傾き・差の検定。"""
    out: dict = {"cells": {}, "tests": {}}
    nj = len(jcode)
    draws: dict[tuple, np.ndarray] = {}
    for sf in ("芝", "ダ"):
        cells = (BANDS if sf == "芝" else ["b1", "b2", "b3", "b4"]) + ["b4p"]
        for bd in cells:
            col = "band_t" if bd == "b4p" else "band"
            m = (d["surf"] == sf) & (d[col] == bd) & d["year"].between(*years) & d[resp].notna()
            sub = d[m]
            if len(sub) < 3000:
                continue
            y = sub[resp].to_numpy(float)
            y = y - y.mean() if resp != "z" else y
            h = pd.factorize(sub["horse_id"])[0]
            j = sub["jockey_id"].map(jcode).to_numpy(int)
            bA, bB, W = split_fits(h, j, y, nj, seed)
            var_y = float(np.var(y))
            point = float(cov_boot(bA, bB, W, np.ones((1, nj)))[0] / var_y)
            dr = cov_boot(bA, bB, W, cnt) / var_y
            draws[(sf, bd)] = dr
            out["cells"][f"{sf}:{bd}"] = {"n": int(len(sub)), "horses": int(len(np.unique(h))), "share": point, "ci": ci(dr), "var_y": var_y}
    for sf in ("芝", "ダ"):
        ks = [(sf, b) for b in TEST_BANDS if (sf, b) in draws]
        if len(ks) < 4:
            continue
        M = np.stack([draws[k] for k in ks], 1)
        sl = slope(np.arange(len(ks), dtype=float), M)
        pt = np.array([out["cells"][f"{sf}:{b}"]["share"] for _, b in ks])
        st = float(slope(np.arange(len(ks), dtype=float), pt[None])[0])
        con = M[:, 0] - M[:, -1]
        out["tests"][f"{sf}:trend"] = {"stat": st, "ci": ci(sl), "p": pval2(sl)}
        out["tests"][f"{sf}:b1-b4p"] = {"stat": float(pt[0] - pt[-1]), "ci": ci(con), "p": pval2(con)}
    return out


# ------------------------------------------------------------------ 実行
def analyze() -> None:
    d = pd.read_parquet(OUT / "frame.parquet")
    jids = pd.Index(sorted(d["jockey_id"].astype(str).unique()))
    jcode = {j: i for i, j in enumerate(jids)}
    d["jc"] = d["jockey_id"].astype(str).map(jcode)
    cnt = boot_counts(len(jcode), 11)
    res: dict = {"design": {"K": 32, "N_SPLIT": N_SPLIT, "N_BOOT": N_BOOT}, "n_rows": int(len(d))}

    # (a) 分散分解: 成績・位置取りの変化。発見 2020〜2024、確認 2025、記述 2026、年ごと
    res["share_z"] = {"disc": share_block(d, "z", (2020, 2024), jcode, cnt, 100),
                      "2025": share_block(d, "z", (2025, 2025), jcode, cnt, 200),
                      "2026": share_block(d, "z", (2026, 2026), jcode, cnt, 300)}
    res["share_pos"] = {"disc": share_block(d, "pos_chg_c", (2021, 2024), jcode, cnt, 400),
                        "2025": share_block(d, "pos_chg_c", (2025, 2025), jcode, cnt, 500),
                        "2026": share_block(d, "pos_chg_c", (2026, 2026), jcode, cnt, 600)}
    loyo = {}
    for y in range(2020, 2025):
        sb = share_block(d, "z", (y, y), jcode, cnt[:100], 700 + y)
        loyo[str(y)] = {"cells": {k: v["share"] for k, v in sb["cells"].items()}, "trend": {k: v["stat"] for k, v in sb["tests"].items()}}
    res["share_z_loyo"] = loyo
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("(a) done")
    analyze_rest(d, jcode, cnt, res)


def analyze_rest(d: pd.DataFrame, jcode: dict, cnt: np.ndarray, res: dict) -> None:
    import statsmodels.formula.api as smf

    # (b) 騎手交代の対比
    sw = d[d["switch"]].copy()
    sd = sw["q_now"].sub(sw["q_prev"]).std()
    sw["dq"] = (sw["q_now"] - sw["q_prev"]) / sd
    sw["lay"] = np.log1p(sw["lay_days"].fillna(sw["lay_days"].median()))
    res["switch_sd"] = float(sd)
    resB: dict = {"cells": {}, "tests": {}}
    for per, yrs in (("disc", (2021, 2024)), ("2025", (2025, 2025)), ("2026", (2026, 2026))):
        for sf in ("芝", "ダ"):
            sub = sw[(sw["surf"] == sf) & sw["year"].between(*yrs) & sw["z"].notna()]
            for bd in TEST_BANDS:
                s2 = sub[sub["band_t"] == bd]
                if len(s2) < 1500:
                    continue
                f = smf.ols("z ~ z_prev + z_prev2m + lay + dq", s2).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s2["race_id"])[0]})
                resB["cells"][f"{per}|{sf}:{bd}"] = {"n": int(len(s2)), "beta": float(f.params["dq"]), "ci": [float(x) for x in f.conf_int().loc["dq"]]}
            s3 = sub[sub["band_t"].isin(["b1", "b4p"])].copy()
            if len(s3) > 3000:
                s3["long"] = (s3["band_t"] == "b4p").astype(float)
                f = smf.ols("z ~ z_prev + z_prev2m + lay + dq + long + dq:long", s3).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s3["race_id"])[0]})
                b = float(f.params["dq:long"])
                resB["tests"][f"{per}|{sf}:long-minus-short"] = {"stat": b, "ci": [float(x) for x in f.conf_int().loc["dq:long"]], "p": float(f.pvalues["dq:long"])}
    res["switch"] = resB
    print("(b) done")

    # (c) 持ち越し
    resC: dict = {"cells": {}, "tests": {}}
    d["resid3"] = d["y_top3"] - d["p_top3"]
    for kind, col, pairs in (("z", "z", {"disc": [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)], "2025": [(2024, 2025)], "2026": [(2025, 2026)]}),
                              ("resid", "resid3", {"disc": [(2022, 2023), (2023, 2024)], "2025": [(2024, 2025)], "2026": [(2025, 2026)]})):
        for per, pl in pairs.items():
            tabs = {}
            for sf in ("芝", "ダ"):
                for bd in TEST_BANDS:
                    sub = d[(d["surf"] == sf) & (d["band_t"] == bd) & d[col].notna()]
                    g = sub.groupby(["jc", "year"])[col].agg(["mean", "size"]).reset_index()
                    g = g[g["size"] >= 50]
                    rows = []
                    for y0, y1 in pl:
                        a = g[g["year"] == y0].set_index("jc"); b2 = g[g["year"] == y1].set_index("jc")
                        both = a.index.intersection(b2.index)
                        rows.append(pd.DataFrame({"jc": both, "x": a.loc[both, "mean"].to_numpy(), "y": b2.loc[both, "mean"].to_numpy()}))
                    t = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["jc", "x", "y"])
                    tabs[(sf, bd)] = t
            draws = {}
            for k, t in tabs.items():
                if len(t) < 30:
                    continue
                jc, x, y = t["jc"].to_numpy(int), t["x"].to_numpy(float), t["y"].to_numpy(float)
                r0 = float(np.corrcoef(x, y)[0, 1])
                wts = cnt[:, jc]
                sw_ = np.maximum(wts.sum(1), 1e-9)
                mx, my = (wts * x).sum(1) / sw_, (wts * y).sum(1) / sw_
                cov = (wts * (x - mx[:, None]) * (y - my[:, None])).sum(1) / sw_
                vx = (wts * (x - mx[:, None]) ** 2).sum(1) / sw_; vy = (wts * (y - my[:, None]) ** 2).sum(1) / sw_
                r = cov / np.sqrt(np.maximum(vx * vy, 1e-12))
                draws[k] = r
                resC["cells"][f"{kind}|{per}|{k[0]}:{k[1]}"] = {"pairs": int(len(t)), "jockeys": int(t["jc"].nunique()), "r": r0, "ci": ci(r)}
            for sf in ("芝", "ダ"):
                if (sf, "b1") in draws and (sf, "b4p") in draws:
                    dd = draws[(sf, "b1")] - draws[(sf, "b4p")]
                    r1 = resC["cells"][f"{kind}|{per}|{sf}:b1"]["r"]; r4 = resC["cells"][f"{kind}|{per}|{sf}:b4p"]["r"]
                    resC["tests"][f"{kind}|{per}|{sf}:b1-b4p"] = {"stat": r1 - r4, "ci": ci(dd), "p": pval2(dd)}
    res["carry"] = resC
    print("(c) done")

    # (d) 有力騎手の残差・回収率・市場統制
    resD: dict = {"cells": {}, "tests": {}}
    d["elite"] = d["el10"]
    base = d[d["rank_valid"] & d["p_win"].notna()].copy()
    base["r_win"] = base["y_win"] - base["p_win"]
    base["r_top3"] = base["y_top3"] - base["p_top3"]
    per_years = {"disc": (2022, 2024), "2025": (2025, 2025), "2026": (2026, 2026)}
    rng = np.random.default_rng(77)

    def race_draws(race: np.ndarray, num: np.ndarray, den: np.ndarray, nd: int, rseed: int) -> np.ndarray:
        _, (cn, cd) = L.race_sums(race, num, den)
        idx = np.random.default_rng(rseed).integers(0, len(cn), size=(nd, len(cn)))
        return cn[idx].sum(1) / np.maximum(cd[idx].sum(1), 1e-9)

    roi_p = []
    diff_draws: dict[tuple, np.ndarray] = {}
    for per, yrs in per_years.items():
        for sf in ("芝", "ダ"):
            cells = (BANDS if sf == "芝" else ["b1", "b2", "b3", "b4"]) + ["b4p"]
            for bd in cells:
                col = "band_t" if bd == "b4p" else "band"
                sub = base[(base["surf"] == sf) & (base[col] == bd) & base["year"].between(*yrs)]
                if len(sub) < 2000:
                    continue
                e = sub[sub["elite"] == 1.0]; ne = sub[sub["elite"] == 0.0]
                if len(e) < 200:
                    continue
                key = f"{per}|{sf}:{bd}"
                cell = {"n": int(len(sub)), "n_elite": int(len(e)), "elite_share": float(len(e) / len(sub))}
                for met in ("r_win", "r_top3"):
                    de = race_draws(e["race_id"].to_numpy(), e[met].to_numpy(), np.ones(len(e)), N_BOOT, 1 + zlib.crc32((key + met + "e").encode()) % 1000)
                    dn = race_draws(ne["race_id"].to_numpy(), ne[met].to_numpy(), np.ones(len(ne)), N_BOOT, 2 + zlib.crc32((key + met + "n").encode()) % 1000)
                    dd = de - dn
                    cell[met] = {"elite": float(e[met].mean()), "non": float(ne[met].mean()), "diff": float(e[met].mean() - ne[met].mean()), "ci": ci(dd)}
                    diff_draws[(per, sf, bd, met)] = dd
                for bet, rc in (("win", "ret_win"), ("place", "ret_place")):
                    ee = e[e[rc].notna()]
                    r = L.boot_roi(ee["race_id"].to_numpy(), ee[rc].to_numpy(float), n_boot=N_BOOT, seed=5)
                    dr = race_draws(ee["race_id"].to_numpy(), ee[rc].to_numpy(float), 100.0 * np.ones(len(ee)), N_BOOT, 3 + zlib.crc32((key + bet).encode()) % 1000) * 100
                    p1 = float((dr <= 100).mean())
                    cell[f"roi_{bet}"] = {**r, "p_gt100": p1}
                    if per == "disc":
                        roi_p.append((f"{sf}:{bd}|{bet}", p1))
                    ne2 = ne[ne[rc].notna()]
                    cell[f"roi_{bet}_non"] = L.boot_roi(ne2["race_id"].to_numpy(), ne2[rc].to_numpy(float), n_boot=N_BOOT, seed=6)
                resD["cells"][key] = cell
    # 距離への依存の変化（b1 − b4p）
    for per in per_years:
        for sf in ("芝", "ダ"):
            for met in ("r_win", "r_top3"):
                k1, k4 = (per, sf, "b1", met), (per, sf, "b4p", met)
                if k1 in diff_draws and k4 in diff_draws:
                    dd = diff_draws[k1] - diff_draws[k4]
                    c1, c4 = resD["cells"][f"{per}|{sf}:b1"][met]["diff"], resD["cells"][f"{per}|{sf}:b4p"][met]["diff"]
                    resD["tests"][f"{per}|{sf}:{met}:b1-b4p"] = {"stat": c1 - c4, "ci": ci(dd), "p": pval2(dd)}
    # Holm（回収率 > 100%）
    roi_p.sort(key=lambda t: t[1])
    m = len(roi_p)
    holm, run = {}, 0.0
    for i, (k, p) in enumerate(roi_p):
        run = max(run, min(1.0, (m - i) * p))
        holm[k] = run
    resD["roi_holm"] = holm
    res["elite"] = resD
    print("(d) done")

    # 市場の単勝確率で統制したロジスティック回帰
    import statsmodels.api as sm

    resM: dict = {"cells": {}, "tests": {}}
    mk = d[d["rank_valid"] & d["q_win"].notna() & (d["q_win"] > 0) & (d["q_win"] < 1)].copy()
    mk["lq"] = np.log(mk["q_win"] / (1 - mk["q_win"]))
    for per, yrs in per_years.items():
        for sf in ("芝", "ダ"):
            for bd in TEST_BANDS:
                s = mk[(mk["surf"] == sf) & (mk["band_t"] == bd) & mk["year"].between(*yrs)]
                if len(s) < 4000 or s["elite"].sum() < 300:
                    continue
                X = sm.add_constant(s[["lq", "elite"]])
                for met, ycol in (("top3", "y_top3"), ("win", "y_win")):
                    try:
                        f = sm.GLM(s[ycol], X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["race_id"])[0]})
                    except Exception:
                        continue
                    resM["cells"][f"{per}|{sf}:{bd}|{met}"] = {"n": int(len(s)), "coef": float(f.params["elite"]), "ci": [float(x) for x in f.conf_int().loc["elite"]]}
            for met, ycol in (("top3", "y_top3"),):
                s = mk[(mk["surf"] == sf) & mk["band_t"].isin(["b1", "b4p"]) & mk["year"].between(*yrs)].copy()
                if len(s) < 8000:
                    continue
                s["long"] = (s["band_t"] == "b4p").astype(float)
                s["el_long"] = s["elite"] * s["long"]
                try:
                    f = sm.GLM(s[ycol], sm.add_constant(s[["lq", "elite", "long", "el_long"]]), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["race_id"])[0]})
                    resM["tests"][f"{per}|{sf}:{met}:long-minus-short"] = {"stat": float(f.params["el_long"]), "ci": [float(x) for x in f.conf_int().loc["el_long"]], "p": float(f.pvalues["el_long"])}
                except Exception:
                    pass
    res["market_ctrl"] = resM
    print("(d3) done")

    # BH（距離への依存の変化 14 本、発見期間）
    fam = []
    for blk, ks in (("share_z", [f"{sf}:trend" for sf in ("芝", "ダ")]), ("share_pos", [f"{sf}:trend" for sf in ("芝", "ダ")])):
        for k in ks:
            t = res[blk]["disc"]["tests"].get(k)
            if t:
                fam.append((f"{blk}|{k}", t["p"]))
    for k in (f"disc|{sf}:long-minus-short" for sf in ("芝", "ダ")):
        if k in res["switch"]["tests"]:
            fam.append((f"switch|{k}", res["switch"]["tests"][k]["p"]))
    for sf in ("芝", "ダ"):
        k = f"z|disc|{sf}:b1-b4p"
        if k in res["carry"]["tests"]:
            fam.append((f"carry|{k}", res["carry"]["tests"][k]["p"]))
        for met in ("r_top3", "r_win"):
            k = f"disc|{sf}:{met}:b1-b4p"
            if k in res["elite"]["tests"]:
                fam.append((f"elite|{k}", res["elite"]["tests"][k]["p"]))
        k = f"disc|{sf}:top3:long-minus-short"
        if k in res["market_ctrl"]["tests"]:
            fam.append((f"market|{k}", res["market_ctrl"]["tests"][k]["p"]))
    fam.sort(key=lambda t: t[1])
    mm = len(fam)
    bh, prev = {}, 1.0
    for i in range(mm - 1, -1, -1):
        prev = min(prev, fam[i][1] * mm / (i + 1))
        bh[fam[i][0]] = float(prev)
    res["bh_family"] = {"m": mm, "p": {k: p for k, p in fam}, "q": bh}

    # 特徴量化の見込み（発見期間で当てはめ、2025 の対数損失の差を 1 回）
    res["feature_potential"] = feature_potential(d)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("all done")


def feature_potential(d: pd.DataFrame) -> dict:
    import statsmodels.api as sm

    base = d[d["rank_valid"] & d["p_win"].notna() & d["band_t"].notna()].copy()
    out: dict = {}
    for ycol, pcol, nm in (("y_top3", "p_top3", "top3"), ("y_win", "p_win", "win")):
        b = base[(base[pcol] > 0) & (base[pcol] < 1)].copy()
        b["off"] = np.log(b[pcol] / (1 - b[pcol]))
        b["turf"] = (b["surf"] == "芝").astype(float)
        cols = ["elite"]
        for bd in ("b2", "b3", "b4p"):
            b[f"e_{bd}"] = b["elite"] * (b["band_t"] == bd)
            cols.append(f"e_{bd}")
        tr = b[b["year"].between(2022, 2024)]; te = b[b["year"] == 2025]
        f1 = sm.GLM(tr[ycol], sm.add_constant(tr[cols + ["turf"]]), family=sm.families.Binomial(), offset=tr["off"]).fit()
        f0 = sm.GLM(tr[ycol], np.ones((len(tr), 1)), family=sm.families.Binomial(), offset=tr["off"]).fit()
        def ll(p, y):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        lin1 = sm.add_constant(te[cols + ["turf"]]).to_numpy() @ f1.params.to_numpy() + te["off"].to_numpy()
        lin0 = f0.params.to_numpy()[0] + te["off"].to_numpy()
        l1 = ll(1 / (1 + np.exp(-lin1)), te[ycol].to_numpy()); l0 = ll(1 / (1 + np.exp(-lin0)), te[ycol].to_numpy())
        dl = l1 - l0
        _, (sd, cn) = L.race_sums(te["race_id"].to_numpy(), dl, np.ones(len(dl)))
        idx = np.random.default_rng(9).integers(0, len(sd), size=(N_BOOT, len(sd)))
        bs = sd[idx].sum(1) / cn[idx].sum(1)
        out[nm] = {"n_test": int(len(te)), "delta_logloss": float(dl.mean()), "ci": ci(bs), "coef": {k: float(v) for k, v in f1.params.items()}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "analyze", "report"])
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "analyze":
        analyze()
    else:
        from src.scripts.data.jockey_distance_dep_report import main as rep

        rep(json.loads((OUT / "result.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
