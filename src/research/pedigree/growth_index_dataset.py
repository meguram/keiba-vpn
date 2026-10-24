"""成長指数（E）用の出走ごとの表を作る。ホスト（stg）で実行する。

  python -m src.research.pedigree.growth_index_dataset --root <repo> --dir <growth_weight の出力ディレクトリ>

入力: growth_weight_dataset の runs.parquet（2020〜2025 年）、学習パイプラインの追い切り（TL_A）と追い切り評価・厩舎コメント。
出力: gi_rows.parquet（1 出走 1 行。T-60 の入力・統制変数・目的変数）、ped_profile_<年>.json（その年より前のデータだけで作った父・母父の成長プロファイル）。

先読み禁止:
 - 直近水準 level は『その走より前の走』の指数だけ（前走が 7 日以内なら前走を除く＝指数の確定が遅れる想定）
 - 血統プロファイルは『その年の 1 月 1 日より前』の走だけで作る
 - 目的変数のうち将来側（次の 2 走の平均）は評価専用で、入力には使わない
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree import growth_weight_stats as S
from src.research.pedigree.growth_weight_analysis import horse_metrics, sire_table

log = logging.getLogger("growth_index_dataset")
DECAY = 0.7
K_PRIOR = 5
WORK = "data/local/modeling/training_pipeline"
TL_COLS = ["f_t4z_vs_self", "f_t1z_vs_self", "f_accelz_vs_self", "f_t1z", "f_accelz", "best_t1z_14", "mean_t4z_14", "n_work_14", "has_training"]
EXTRA_COLS = ["oikiri_eval", "comment_pos", "comment_neg"]
PED_MIN_N = 3


def level_columns(idx_v: np.ndarray) -> dict[str, np.ndarray]:
    """1 頭分（日付昇順）の指数から、直近水準（前走を含む/除く）・有効な過去走数・将来 3 走の平均を出す。"""
    n = len(idx_v)
    inc = np.full(n, np.nan)
    exc = np.full(n, np.nan)
    w2_inc = np.full(n, np.nan)
    w2_exc = np.full(n, np.nan)
    nvalid = np.zeros(n, dtype=int)
    fut = np.full(n, np.nan)
    m_fut = np.zeros(n, dtype=int)
    valid = ~np.isnan(idx_v)
    wts = DECAY ** np.arange(K_PRIOR)
    for i in range(n):
        pj = [j for j in range(i - 1, -1, -1) if valid[j]]
        nvalid[i] = len(pj)
        for out, w2, js in ((inc, w2_inc, pj[:K_PRIOR]), (exc, w2_exc, [j for j in pj if j <= i - 2][:K_PRIOR])):
            if len(js) >= 2:
                w = wts[: len(js)]
                out[i] = float(np.sum(w * idx_v[js]) / w.sum())
                w2[i] = float(np.sum((w / w.sum()) ** 2))
        f = idx_v[i : i + 3]
        f = f[~np.isnan(f)]
        m_fut[i] = len(f)
        if len(f) >= 2 and valid[i]:
            fut[i] = float(f.mean())
    return {"level_inc": inc, "level_exc": exc, "w2_inc": w2_inc, "w2_exc": w2_exc, "n_valid_prior": nvalid, "fut_mean": fut, "m_fut": m_fut}


def add_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    idx_v = df["idx"].where(~df["low_conf"].astype(bool)).to_numpy(float)
    _, first = np.unique(df["horse_id"].to_numpy(), return_index=True)
    bounds = np.r_[np.sort(first), len(df)]
    cols = {k: np.empty(len(df)) for k in ("level_inc", "level_exc", "w2_inc", "w2_exc", "n_valid_prior", "fut_mean", "m_fut")}
    for a, b in zip(bounds[:-1], bounds[1:]):
        r = level_columns(idx_v[a:b])
        for k in cols:
            cols[k][a:b] = r[k]
    for k, v in cols.items():
        df[k] = v
    df["idx_v"] = idx_v
    return df


def attach_inputs(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    rest = df["rest_days"]
    stale = (rest <= 7) | rest.isna()
    df["stale_prev"] = stale.astype(int)
    df["level"] = np.where(stale, df["level_exc"], df["level_inc"])
    df["w2"] = np.where(stale, df["w2_exc"], df["w2_inc"])
    df["y_now"] = df["idx_v"] - df["level"]
    df["y_fut"] = df["fut_mean"] - df["level"]
    wc = pd.to_numeric(df["weight_change"], errors="coerce")
    df["weight_change_in"] = df["dweight"].where(df["dweight"].notna(), wc)
    df["weight_dev"] = (df["weight"] - df["prior_mean_weight"]).where(df["prior_n_weight"] >= 1)
    tl = pd.read_parquet(root / WORK / "training_laps/cand_TL_A.parquet", columns=TL_COLS + ["race_id", "horse_id"])
    ex = pd.read_parquet(root / WORK / "features/prerace_extras.parquet", columns=EXTRA_COLS + ["race_id", "horse_id"])
    for t in (tl, ex):
        t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df.merge(tl.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    df = df.merge(ex.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    return df


def ped_tables(df: pd.DataFrame, as_of_year: int) -> dict:
    """as_of_year の 1 月 1 日より前のデータだけで、父・母父ごとの縮約済みプロファイルを作る。"""
    sub = df[df["date"] < pd.Timestamp(f"{as_of_year}-01-01")]
    h = horse_metrics(sub, range(2000, 2031), pd.Timestamp(f"{as_of_year - 1}-12-31"))
    metrics = ["late_minus_early_idx", "late_minus_early_wt"] + [c for c in h.columns if c.startswith(("dwt_", "didx_")) and not c.endswith("_dev")]
    out: dict = {}
    for key, nc, tag in (("sire_id", "sire_name", "sire"), ("dam_sire_id", "dam_sire_name", "dam_sire")):
        tabs = {m: sire_table(h, key, nc, m, min_n=PED_MIN_N) for m in metrics}
        out[tag] = tabs
    return out


def profile_json(tabs: dict) -> dict:
    res: dict = {}
    for tag, t in tabs.items():
        ids = set()
        for m in t.values():
            ids |= set(m.index)
        d = {}
        for sid in ids:
            e: dict = {"didx": {}, "dwt": {}}
            for m, tb in t.items():
                if sid not in tb.index:
                    continue
                v = float(tb.at[sid, "shrunk"])
                if m == "late_minus_early_idx":
                    e["banse"], e["n_idx"], e["name"] = round(v, 3), int(tb.at[sid, "n"]), tb.at[sid, "name"]
                elif m == "late_minus_early_wt":
                    e["wtgain"], e["n_wt"] = round(v, 3), int(tb.at[sid, "n"])
                elif m.startswith("didx_"):
                    e["didx"][m[5:]] = round(v, 3)
                elif m.startswith("dwt_"):
                    e["dwt"][m[4:]] = round(v, 3)
            d[str(sid)] = e
        res[tag] = d
    return res


def attach_ped(df: pd.DataFrame, tabs_by_year: dict[int, dict]) -> pd.DataFrame:
    cols = {k: np.full(len(df), np.nan) for k in ("sire_didx", "sire_dwt", "sire_banse", "sire_wtgain", "dam_sire_didx", "dam_sire_dwt", "dam_sire_banse", "dam_sire_wtgain")}
    prev_stage = {s: (S.STAGES[i - 1] if i > 0 else None) for i, s in enumerate(S.STAGES)}
    st = df["stage"].to_numpy(object)
    yr = df["year"].to_numpy()
    for tag, key in (("sire", "sire_id"), ("dam_sire", "dam_sire_id")):
        ids = df[key].astype(str).to_numpy(object)
        for y, tabs in tabs_by_year.items():
            t = tabs[tag]
            my = yr == y
            for m, name in (("late_minus_early_idx", "banse"), ("late_minus_early_wt", "wtgain")):
                cols[f"{tag}_{name}"][my] = pd.Series(ids[my]).map(t[m]["shrunk"]).to_numpy(float)
            for s in S.STAGES[1:]:
                mm = my & (st == s)
                if not mm.any():
                    continue
                tr = f"{prev_stage[s]}→{s}"
                for pre, name in (("didx_", "didx"), ("dwt_", "dwt")):
                    if pre + tr in t:
                        cols[f"{tag}_{name}"][mm] = pd.Series(ids[mm]).map(t[pre + tr]["shrunk"]).to_numpy(float)
    for k, v in cols.items():
        df["ped_" + k] = v
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, d = Path(a.root), Path(a.dir)
    df = pd.read_parquet(d / "runs.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["year"] <= 2025].copy()
    df["surface"] = df["surface"].map(lambda s: "ダ" if isinstance(s, str) and s.startswith("ダ") else s)
    df = add_levels(df)
    df = attach_inputs(df, root)
    log.info("行 %d / y_now あり %d / y_fut あり %d", len(df), df["y_now"].notna().sum(), df["y_fut"].notna().sum())
    tabs_by_year = {}
    for y in (2022, 2023, 2024, 2025):
        tabs_by_year[y] = ped_tables(df, y)
        log.info("血統プロファイル as-of %d: 父 %d / 母父 %d", y, len(tabs_by_year[y]["sire"]["late_minus_early_idx"]), len(tabs_by_year[y]["dam_sire"]["late_minus_early_idx"]))
    df = attach_ped(df, tabs_by_year)
    final = ped_tables(df, 2026)         # 2025 年末までの全データ（サーバ用）
    (d / "ped_profile.json").write_text(json.dumps(profile_json(final), ensure_ascii=False), encoding="utf-8")
    df.to_parquet(d / "gi_rows.parquet", index=False)
    log.info("保存: gi_rows.parquet %d 行 %d 列", *df.shape)


if __name__ == "__main__":
    main()
