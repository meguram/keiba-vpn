"""補助ヘッド NN の『ペース』『RPCI』（レース単位ラベル＝同じレースの馬を区別できない）を、馬単位の目的に置き換える検証。

python -m src.scripts.data.aux_h_eval labels   # 通過順（コーナーごとの位置）から馬単位の目的 h_drop / h_gain を作る
python -m src.scripts.data.aux_h_eval oof      # tp-25c と同じ設計（2021-24 は expanding 年別 OOF、2020 は時間ブロック k-fold）で補助 NN を学習 → aux_H.parquet
python -m src.scripts.data.aux_h_eval single   # 単独の精度（馬内の順位識別力・R²・主ヘッドの LogLoss を置き換え前と比較）
環境変数: SEEDS=0,1,2  THREADS=4  AUXH_OUT=出力先
目的変数（結果由来。学習のラベルにだけ使う）:
  h_drop = コーナー間の位置変化（0=先頭〜1=最後尾に正規化）の最大の後退幅（不利・進路ロス・バテの代理）
  h_gain = 同じく最大の前進幅（押し上げ・まくり）
障害・通過順が不正な行は欠損（損失から外す）。出遅れは ten_rel_actual（テン1F の相対）が既に担う。市場オッズ・人気は使わない。2026 年は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("AUXH_OUT") or ROOT / "data/local/modeling/training_pipeline/auxh")
SEEDS = [int(s) for s in (os.environ.get("SEEDS") or "0,1,2").split(",")]
THREADS = int(os.environ.get("THREADS") or 4)
KEY = ["race_id", "horse_id"]
AUX_OLD = ["t_pos_early", "t_pos_late", "t_l3f_dev", "t_time_dev", "t_rank_norm", "ten_rel_actual", "c1_rel_actual", "r_pace_bal", "r_rpci"]
AUX_H = ["t_pos_early", "t_pos_late", "t_l3f_dev", "t_time_dev", "t_rank_norm", "ten_rel_actual", "c1_rel_actual", "h_drop", "h_gain"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_labels(_a) -> None:
    import polars as pl

    from src.pipeline.features.horse_history.common import _valid_passing, load_runs

    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    d = runs.filter(pl.col("surface") != "障").with_columns(_valid_passing("n_fin", "passing_order").alias("ok"))
    d = d.filter(pl.col("ok")).select("race_id", "horse_id", "passing_order", "n_fin").to_pandas()
    po = d["passing_order"].str.split("-")
    n = d["n_fin"].to_numpy()
    drop, gain = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for i, (s, nn) in enumerate(zip(po, n)):
        if len(s) < 2:
            continue
        p = (np.asarray(s, dtype=float) - 1) / max(nn - 1, 1)
        st = np.diff(p)
        drop[i], gain[i] = max(st.max(), 0.0), max((-st).max(), 0.0)
    out = pd.DataFrame({"race_id": d["race_id"].astype(str), "horse_id": d["horse_id"].astype(str), "h_drop": drop.astype("float32"), "h_gain": gain.astype("float32")})
    out = out.drop_duplicates(KEY)
    out.to_parquet(OUT / "h_labels.parquet", index=False)
    _log(f"h_labels {out.shape} 欠損率 drop={out['h_drop'].isna().mean():.4f}")


def _frame():
    from src.scripts.data import tp25c_eval as c

    df, base, _sx = c.load_frame()
    h = pd.read_parquet(OUT / "h_labels.parquet")
    m = df[KEY].merge(h, on=KEY, how="left")
    assert len(m) == len(df)
    for k in ("h_drop", "h_gain"):
        df[k] = m[k].to_numpy()
    return df, base


def _rel(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    g = df.groupby("race_id", sort=False)
    out = {}
    for c in names:
        out[c + "_rk"] = g[c].rank(pct=True)
        out[c + "_gap"] = df[c] - g[c].transform("mean")
    return pd.DataFrame(out, index=df.index)


def cmd_oof(a) -> None:
    import torch

    from src.pipeline.models.tp25c_mtl import mtl_fit_predict
    from src.pipeline.models import win_models as wm
    from src.pipeline.models import zoo
    from src.scripts.data import tp25e_eval as e

    torch.set_num_threads(THREADS)
    aux_names = AUX_OLD if a.old else AUX_H
    tag = "C" if a.old else "H"
    df, base = _frame()
    X = df[base]
    A = df[aux_names].to_numpy(dtype="float64")
    race, fin, dates, years = df["race_id"].to_numpy(), df["finish_position"].to_numpy(), df["date"], df["year"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    k = len(aux_names)
    cache = OUT / f"oof_aux_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        oof, main = z["aux"], z["main"]
    else:
        oof = np.full((len(df), k), np.nan, dtype="float32")
        main = np.full((len(df), 3), np.nan, dtype="float32")        # 主ヘッド（勝率・連対・複勝）の OOF
        gap = pd.Timedelta(days=10)
        for yr in (2021, 2022, 2023, 2024):
            tr = np.flatnonzero((years >= 2020) & (dates < pd.Timestamp(f"{yr}-01-01") - gap))
            va = np.flatnonzero(years == yr)
            acc, mn = np.zeros((len(va), k)), np.zeros((len(va), 3))
            for seed in SEEDS:
                t0 = time.time()
                o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[va], race[va], seed=seed, aux_w=0.5)
                acc += o["aux"] / len(SEEDS)
                mn += np.column_stack([wm.softmax_by_race(race[va], o["win_score"]), zoo.sigmoid(o["top2_logit"]), zoo.sigmoid(o["top3_logit"])]) / len(SEEDS)
                _log(f"oof {tag} {yr} seed{seed} {time.time() - t0:.0f}s")
            oof[va], main[va] = acc, mn
        k20 = OUT / f"oof_aux_{tag}_2020_kfold.npy"
        if k20.exists():
            r20 = np.load(k20)
        else:
            e.SEEDS = SEEDS
            r20 = e.kfold_2020(df, base, A, Y, race)
            np.save(k20, r20)
        m20 = years == 2020
        oof[m20] = r20[m20]
        np.savez(cache, aux=oof, main=main)
    names = ["mtl_" + x for x in aux_names]
    t = pd.DataFrame(oof, columns=names, index=df.index)
    t = pd.concat([t, _rel(pd.concat([df[["race_id"]], t], axis=1), names)], axis=1)
    t.insert(0, "horse_id", df["horse_id"].to_numpy())
    t.insert(0, "race_id", df["race_id"].to_numpy())
    t.to_parquet(OUT / f"aux_{tag}.parquet", index=False)
    _log(f"aux_{tag}: {t.shape}")


def _within_race_corr(race: np.ndarray, p: np.ndarray, y: np.ndarray, min_n: int = 5) -> float:
    d = pd.DataFrame({"r": race, "p": p, "y": y}).dropna()
    d = d[d.groupby("r")["p"].transform("size") >= min_n]
    g = d.groupby("r")
    pc = g["p"].rank() - g["p"].rank().groupby(d["r"]).transform("mean")
    yc = g["y"].rank() - g["y"].rank().groupby(d["r"]).transform("mean")
    num = (pc * yc).groupby(d["r"]).sum()
    den = np.sqrt((pc ** 2).groupby(d["r"]).sum() * (yc ** 2).groupby(d["r"]).sum())
    return float((num / den).replace([np.inf, -np.inf], np.nan).mean())


def cmd_single(_a) -> None:
    from src.pipeline.models import win_models as wm

    df, base = _frame()
    years = df["year"].to_numpy()
    race = df["race_id"].to_numpy()
    fin = df["finish_position"].to_numpy()
    yy = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    res: dict = {}
    zh, zc = np.load(OUT / "oof_aux_H.npz"), np.load(OUT / "oof_aux_C.npz")
    for yr in (2022, 2023, 2024, "all"):
        m = np.isin(years, [2022, 2023, 2024]) if yr == "all" else years == yr
        row: dict = {}
        for tag, z, names in (("old", zc, AUX_OLD), ("new", zh, AUX_H)):
            main = z["main"]
            row[f"{tag}_ll"] = {q: float(wm.binary_logloss(main[m, i], yy[q][m])) for i, q in enumerate(("win", "top2", "top3"))}
        for tag, z, names in (("old", zc, AUX_OLD), ("new", zh, AUX_H)):
            for j, nm in enumerate(names):
                if nm in ("r_pace_bal", "r_rpci", "h_drop", "h_gain"):
                    # 各ヘッドの予測が、馬単位の h_* の真値（および着順）を同じレース内でどれだけ順位付けできるか
                    for tgt in ("h_drop", "h_gain"):
                        ok = m & np.isfinite(df[tgt].to_numpy())
                        row[f"{tag}:{nm}->{tgt}_within_race_spearman"] = _within_race_corr(race[ok], z["aux"][ok, j], df[tgt].to_numpy()[ok])
                    ok = m & np.isfinite(fin)
                    row[f"{tag}:{nm}->finish_within_race_spearman"] = _within_race_corr(race[ok], z["aux"][ok, j], fin[ok])
        for tgt, nm in (("h_drop", "h_drop"), ("h_gain", "h_gain")):
            j = AUX_H.index(nm)
            ok = m & np.isfinite(df[tgt].to_numpy())
            yv, pv = df[tgt].to_numpy()[ok], zh["aux"][ok, j]
            tr_mean = float(np.nanmean(df[tgt].to_numpy()[years < 2022]))
            # 標準化した単位の予測なので、R² は線形当てはめ（予測と真値の相関²）で見る
            row[f"new:{nm}_r2_linear"] = float(np.corrcoef(pv, yv)[0, 1] ** 2)
            row[f"new:{nm}_mean_baseline_mse"] = float(((yv - tr_mean) ** 2).mean())
        res[str(yr)] = row
    (OUT / "single.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["all"], ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("labels")
    o = sp.add_parser("oof")
    o.add_argument("--old", action="store_true", help="置き換え前（AUX_C と同じ 9 本）を同じ設計・同じ seed で再学習する対照")
    sp.add_parser("single")
    a = ap.parse_args()
    {"labels": cmd_labels, "oof": cmd_oof, "single": cmd_single}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
