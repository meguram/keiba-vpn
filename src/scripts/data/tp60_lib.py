"""tp-60 共通: 血統キー付きの『1 出走 = 1 行』表（tp-59 の master + 血統）と、セルの集計・年またぎの再現性・レース単位ブートストラップ。

市場オッズ・払戻は分析（回収率・市場の織り込み・馬券の評価）にだけ使い、特徴量にはしない。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTD = Path(os.environ.get("TP60_OUT", "/home/jovyan/work/tp60_out"))
TP59 = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
PED = OUTD / "ped"
RES = OUTD / "roi"
RNG_SEED = 20261021


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_master(jump: bool = False) -> pd.DataFrame:
    m = pd.read_parquet(TP59 / "master.parquet")
    hs = pd.read_parquet(PED / "horse_static.parquet")
    use = ["horse_id", "sire_id", "dsire_id", "ssire_id", "dam_id", "sire_line", "dsire_line", "x_F", "x_n", "birth_year", "horse_foreign", "deep_cov"]
    m = m.merge(hs[use], on="horse_id", how="left")
    if not jump:
        m = m[m["surface"].astype(str).isin(["芝", "ダート"])].copy()
    sa = m["sex_age"].astype(str)
    m["sex"] = sa.str[0]
    m["age"] = pd.to_numeric(sa.str[1:], errors="coerce")
    m["sd"] = m["surface"].astype(str) + "|" + m["dist_band"].astype(str)
    m["going"] = m["track_condition"].astype(str)
    m["venue"] = m["venue_code"].astype(str)
    m["pop_band"] = pd.cut(m["popularity"], [0, 3, 6, 9, 99], labels=["p1-3", "p4-6", "p7-9", "p10+"]).astype(str)
    m["age_band"] = pd.cut(m["age"], [0, 2, 3, 4, 99], labels=["2yo", "3yo", "4yo", "5+"]).astype(str)
    m["all"] = "all"
    # 父の初年度産駒など（産駒の生年から。血統表は生まれた時点で公開なので先読みではない）
    uh = hs.dropna(subset=["birth_year"]).drop_duplicates("horse_id")
    first_by = uh.groupby("sire_id")["birth_year"].min()
    m["sire_first_by"] = m["sire_id"].map(first_by)
    crop = m["birth_year"] - m["sire_first_by"]
    new_sire = m["sire_first_by"] >= 2017
    m["crop"] = np.where(new_sire & (crop == 0), "crop0", np.where(new_sire & (crop == 1), "crop1", np.where(new_sire & (crop == 2), "crop2", np.where(new_sire, "crop3+", "established"))))
    m.loc[m["birth_year"].isna() | m["sire_id"].isna(), "crop"] = "unknown"
    m["sire_dsire"] = np.where(m["sire_id"].notna() & m["dsire_id"].notna(), m["sire_id"].astype(str) + "|" + m["dsire_id"].astype(str), None)
    m["line_pair"] = np.where(m["sire_line"].notna() & m["dsire_line"].notna(), m["sire_line"].astype(str) + "|" + m["dsire_line"].astype(str), None)
    m["xF_band"] = pd.cut(m["x_F"], [-1, 0, 0.0078, 0.0156, 0.03125, 1], labels=["no_cross", "F<0.8%", "F<1.6%", "F<3.1%", "F>=3.1%"]).astype(str)
    m.loc[m["x_F"].isna(), "xF_band"] = "unknown"
    m["foreign"] = np.where(m["horse_foreign"] == 1, "foreign_bred", "jp_bred")
    return m.reset_index(drop=True)


KEYS = ["sire_id", "dsire_id", "ssire_id", "sire_dsire", "sire_line", "dsire_line", "line_pair", "xF_band", "crop", "foreign"]
CONDS = ["all", "sd", "going", "venue", "cls", "age_band", "pop_band"]


def collapse(m: pd.DataFrame, key: str, cond: str = "all") -> pd.DataFrame:
    """(race_id, key, cond) ごとに合算（1 レースに同じセルの馬が複数いてもレース 1 件として扱うための前処理）。"""
    d = m.dropna(subset=[key])
    g = d.groupby(["race_id", key, cond], sort=False)
    out = g.agg(year=("year", "first"), n=("win", "size"), win=("win", "sum"), top2=("top2", "sum"), top3=("top3", "sum"),
                rw=("ret_win", "sum"), rp=("ret_place", "sum"), q=("q_win", "sum"), p=("p_win", "sum")).reset_index()
    out = out.rename(columns={key: "key", cond: "cond"})
    return out


def agg_cells(c: pd.DataFrame) -> pd.DataFrame:
    """セル（key, cond）ごとの集計。roi_* は 100 円あたり（1.0 = 100%）。race_roi_* はレース均等（レースごとに該当馬の平均を取ってから平均）。"""
    c = c.assign(rw_m=c["rw"] / c["n"], rp_m=c["rp"] / c["n"])
    g = c.groupby(["key", "cond"], sort=False)
    r = g.agg(races=("n", "size"), horses=("n", "sum"), win=("win", "sum"), top2=("top2", "sum"), top3=("top3", "sum"),
              rw=("rw", "sum"), rp=("rp", "sum"), q=("q", "sum"), p=("p", "sum"), rw_m=("rw_m", "mean"), rp_m=("rp_m", "mean"))
    r["win_rate"], r["top2_rate"], r["top3_rate"] = r["win"] / r["horses"], r["top2"] / r["horses"], r["top3"] / r["horses"]
    r["roi_win"], r["roi_place"] = r["rw"] / r["horses"] / 100.0, r["rp"] / r["horses"] / 100.0
    r["race_roi_win"], r["race_roi_place"] = r["rw_m"] / 100.0, r["rp_m"] / 100.0
    r["ex_mkt"] = (r["win"] - r["q"]) / r["horses"]          # 勝率 − 市場の勝率（単勝オッズ由来）
    r["ex_model"] = (r["win"] - r["p"]) / r["horses"]        # 勝率 − V4 の勝率
    return r.reset_index()


def eb_shrink(sum_ret: np.ndarray, n: np.ndarray, mu: float, k: float) -> np.ndarray:
    return (sum_ret + k * mu) / (n + k)


def boot_diff(m: pd.DataFrame, sel: np.ndarray, elig: np.ndarray, col: str, B: int = 2000, seed: int = RNG_SEED) -> dict:
    """elig（比較の母集団）の中の sel（選んだ）の回収率 − elig 全体の回収率。レースを単位にブートストラップ。col: ret_win / ret_place。"""
    d = m.loc[elig, ["race_id", col]].copy()
    d["sel"] = sel[elig]
    codes, uniq = pd.factorize(d["race_id"])
    k = len(uniq)
    v = d[col].to_numpy(dtype=float) / 100.0
    s = d["sel"].to_numpy(dtype=bool)
    a_sum = np.bincount(codes, weights=v, minlength=k)
    a_n = np.bincount(codes, minlength=k).astype(float)
    s_sum = np.bincount(codes, weights=v * s, minlength=k)
    s_n = np.bincount(codes, weights=s.astype(float), minlength=k)
    if s_n.sum() == 0:
        return {"sel_roi": np.nan, "all_roi": float(a_sum.sum() / a_n.sum()), "diff": np.nan, "lo": np.nan, "hi": np.nan, "p_le0": np.nan, "n_sel": 0, "races_sel": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(B, k))
    ss, sn, as_, an = s_sum[idx].sum(1), s_n[idx].sum(1), a_sum[idx].sum(1), a_n[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = ss / sn - as_ / an
        sel_roi = ss / sn
    diff = diff[np.isfinite(diff)]
    return {"sel_roi": float(s_sum.sum() / s_n.sum()), "all_roi": float(a_sum.sum() / a_n.sum()), "diff": float(s_sum.sum() / s_n.sum() - a_sum.sum() / a_n.sum()),
            "lo": float(np.percentile(sel_roi[np.isfinite(sel_roi)], 2.5)), "hi": float(np.percentile(sel_roi[np.isfinite(sel_roi)], 97.5)),
            "diff_lo": float(np.percentile(diff, 2.5)), "diff_hi": float(np.percentile(diff, 97.5)), "p_le0": float((diff <= 0).mean()),
            "n_sel": int(s.sum()), "races_sel": int((s_n > 0).sum())}


def roi_ci(m: pd.DataFrame, mask: np.ndarray, col: str, B: int = 2000, seed: int = RNG_SEED, race_equal: bool = True) -> dict:
    """mask の馬の回収率（レース均等: レースごとの平均 → 平均）と 95% 区間（レース単位ブートストラップ）。"""
    d = m.loc[mask, ["race_id", col]]
    if len(d) == 0:
        return {"roi": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "races": 0}
    codes, uniq = pd.factorize(d["race_id"])
    k = len(uniq)
    v = d[col].to_numpy(dtype=float) / 100.0
    s = np.bincount(codes, weights=v, minlength=k)
    n = np.bincount(codes, minlength=k).astype(float)
    per = s / n if race_equal else None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(B, k))
    if race_equal:
        bs = per[idx].mean(1)
        est = float(per.mean())
    else:
        bs = s[idx].sum(1) / n[idx].sum(1)
        est = float(s.sum() / n.sum())
    return {"roi": est, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": int(len(d)), "races": int(k)}
