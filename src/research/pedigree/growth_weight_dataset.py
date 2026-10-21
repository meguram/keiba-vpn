"""馬体重 × めぐ指数の調査用データセット（出走ごとの長い表）を作る。ホスト（stg）で実行する。

  python -m src.research.pedigree.growth_weight_dataset --root <repo> --out <出力ディレクトリ>

入力: data/page_reference/tables/<年>/{race_result_flat,megu_index_flat}.parquet（2020〜2025 年。2026 年は未完のため使わない）
      data/local/horse_pedigree_{5,10}gen/<馬ID先頭4桁>/<馬ID>.json（父・母父の ID）
出力: runs.parquet（1 出走 1 行。前走の体重・指数・日付、休養日数、成長段階、父・母父の ID つき）
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree.growth_weight_stats import age_from_id, stage_of

log = logging.getLogger("growth_weight_dataset")
YEARS = range(2020, 2026)
IDX_COL = "megu_index_v3a_abs_robust"


def load_runs(root: Path, years=YEARS) -> pd.DataFrame:
    frames = []
    for y in years:
        base = root / "data/page_reference/tables" / str(y)
        rr = pd.read_parquet(base / "race_result_flat.parquet",
                             columns=["race_id", "date", "venue", "surface", "distance", "field_size", "horse_id", "sex_age",
                                      "weight", "weight_change", "finish_position", "jockey_weight", "race_class", "grade"])
        mg = pd.read_parquet(base / "megu_index_flat.parquet",
                             columns=["race_id", "horse_id", IDX_COL, "megu_v3a_status", "megu_v3a_low_confidence"])
        frames.append(rr.merge(mg, on=["race_id", "horse_id"], how="left"))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace("/", "-"), errors="coerce")
    df = df.dropna(subset=["date", "horse_id"])
    df["horse_id"] = df["horse_id"].astype(str)
    df["sex"] = df["sex_age"].astype(str).str[:1]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    w = pd.to_numeric(df["weight"], errors="coerce")
    df["weight"] = w.where((w >= 300) & (w <= 700))
    df["idx"] = pd.to_numeric(df[IDX_COL], errors="coerce")
    # 指数が使えるのは状態が ok のものだけ（障害・未計算は NaN）
    bad = ~df["megu_v3a_status"].astype(str).str.startswith("ok")
    df.loc[bad, "idx"] = np.nan
    df["low_conf"] = df["megu_v3a_low_confidence"].fillna(False).astype(bool)
    df["age"] = [age_from_id(h, y) for h, y in zip(df["horse_id"], df["year"])]
    df["stage"] = [stage_of(a, m) for a, m in zip(df["age"], df["month"])]
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    return df.drop(columns=[IDX_COL, "megu_v3a_low_confidence"])


def add_previous(df: pd.DataFrame) -> pd.DataFrame:
    """同じ馬の前走（このデータの中での直前の出走）の値を付ける。前走がデータの先頭より前（2020 年以前）なら欠損。"""
    df = df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = df.groupby("horse_id", sort=False)
    for c in ["date", "weight", "idx", "race_id", "low_conf", "distance", "surface"]:
        df["prev_" + c] = g[c].shift(1)
    df["rest_days"] = (df["date"] - df["prev_date"]).dt.days
    df["dweight"] = df["weight"] - df["prev_weight"]
    df["didx"] = df["idx"] - df["prev_idx"]
    df["n_prior"] = g.cumcount()
    # 過去の体重・指数の平均（その走より前だけ）。適正体重からのずれに使う
    for c in ["weight", "idx"]:
        csum = g[c].cumsum() - df[c].fillna(0)
        cnt = g[c].transform(lambda s: s.notna().cumsum()) - df[c].notna().astype(int)
        df["prior_mean_" + c] = (csum / cnt.replace(0, np.nan)).where(cnt > 0)
        df["prior_n_" + c] = cnt
    return df


def _parents_of(args) -> tuple[str, str, str, str, str]:
    h, dirs = args
    for d in dirs:
        f = Path(d) / h[:4] / f"{h}.json"
        if not f.exists():
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        sire = dam_sire = sname = dname = ""
        for a in doc.get("ancestors") or []:
            g, p = a.get("generation"), a.get("position")
            if g == 1 and p == 0:
                sire, sname = str(a.get("horse_id") or ""), str(a.get("name") or doc.get("sire") or "")
            elif g == 2 and p == 2:
                dam_sire, dname = str(a.get("horse_id") or ""), str(a.get("name") or doc.get("dam_sire") or "")
        if sire or dam_sire:
            return h, sire, dam_sire, sname, dname
    return h, "", "", "", ""


def parent_table(root: Path, horse_ids, workers: int = 4) -> pd.DataFrame:
    dirs = [str(root / "data/local/horse_pedigree_5gen"), str(root / "data/local/horse_pedigree_10gen")]
    ids = sorted(set(horse_ids))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_parents_of, [(h, dirs) for h in ids], chunksize=64))
    log.info("血統 JSON %d 頭 %.0f 秒", len(ids), time.time() - t0)
    return pd.DataFrame(rows, columns=["horse_id", "sire_id", "dam_sire_id", "sire_name", "dam_sire_name"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = add_previous(load_runs(root))
    pt = parent_table(root, df["horse_id"].unique(), a.workers)
    df = df.merge(pt, on="horse_id", how="left")
    df.to_parquet(out / "runs.parquet", index=False)
    chk = df.dropna(subset=["dweight", "weight_change"])
    wc = pd.to_numeric(chk["weight_change"], errors="coerce")
    log.info("出走 %d 行 / 馬 %d 頭 / 体重あり %d / 指数あり %d / 前走あり(体重・指数とも) %d / 父ID あり %d 頭",
             len(df), df["horse_id"].nunique(), df["weight"].notna().sum(), df["idx"].notna().sum(),
             (df["dweight"].notna() & df["didx"].notna()).sum(), (pt["sire_id"] != "").sum())
    log.info("自前の前走比と発表の増減の一致率: %.3f（n=%d）", float((chk["dweight"] == wc).mean()), len(chk))


if __name__ == "__main__":
    main()
