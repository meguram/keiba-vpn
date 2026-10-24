"""推論時の LayerA 列（serving_layer_a）を、学習表（fixed）の同じ行と比べる。

python -m src.scripts.data.verify_serving_layer_a --days 20250302 20250614 --races-per-day 6 --out /tmp/serve_layera_parity.json
  A: その日の全レースを 1 回で（履歴はその日より前だけ）。同日の前のレースの結果が無いので、同日依存の列は差が出る。
  B: レースごとに 1 レース（履歴は同日の他レースの結果も含む。学習と同じ並びで使われる分だけ効く）。
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import serving_layer_a as sl

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "data/local/modeling/training_pipeline_fixed/features/train_table.parquet"
CAT = set(sl.CATEGORICAL)


def _reference(day: str) -> pd.DataFrame:
    d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    cols = ["race_id", "horse_id", "date", "sex_age", "start_time"] + [c for c in sl.OUTPUT_COLS if c not in sl.DERIVED_COLS]
    t = pl.read_parquet(TRAIN, columns=list(dict.fromkeys(cols))).filter(pl.col("date").cast(pl.Utf8) == d).to_pandas()
    t["sex"] = t["sex_age"].astype(str).str[:1]
    t["age"] = pd.to_numeric(t["sex_age"].astype(str).str[1:], errors="coerce")
    t["start_hour"] = pd.to_numeric(t["start_time"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
    for c in CAT:
        t[c] = t[c].astype("string").fillna("(なし)").astype(str)
    t["race_id"] = t["race_id"].astype(str)
    return t


def _card(day: str, tables: Path) -> pd.DataFrame:
    f = tables / day[:4] / "race_shutuba_flat.parquet"
    return pl.read_parquet(f).filter(pl.col("date").cast(pl.Utf8) == f"{day[:4]}-{day[4:6]}-{day[6:]}").to_pandas()


def _collision(ref: pd.DataFrame, raw_ids: pd.DataFrame) -> pd.DataFrame:
    """同じレースの中に、馬番の小さい同一騎手/調教師の馬がいる行（学習側はその結果まで数える）。"""
    m = raw_ids.sort_values(["race_id", "horse_number"]).copy()
    m["jk_dup"] = m.duplicated(["race_id", "jockey_id"], keep="first")
    m["tr_dup"] = m.duplicated(["race_id", "trainer_id"], keep="first")
    return m[["race_id", "horse_number", "jk_dup", "tr_dup"]]


def _compare(got: pd.DataFrame, ref: pd.DataFrame, coll: pd.DataFrame) -> dict:
    m = got.merge(ref, on=["race_id", "horse_id"], suffixes=("", "_ref")).merge(coll, on=["race_id", "horse_number"], how="left")
    out: dict = {"rows": int(len(m))}
    for c in sl.OUTPUT_COLS:
        a, b = m[c], m[c + "_ref"]
        if c in CAT:
            ok = (a.astype(str) == b.astype(str)).to_numpy()
            out[c] = {"match": float(ok.mean())}
            continue
        x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        ok = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-5, atol=1e-6)
        d = np.abs(x - y)
        loose = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=0, atol=2e-3)
        rec = {"match": float(ok.mean()), "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0, "match_atol_2e-3": float(loose.mean()), "nan_only_one_side": int((np.isnan(x) ^ np.isnan(y)).sum())}
        if c.startswith("jockey_prior"):
            k = ~m["jk_dup"].fillna(False).to_numpy(bool)
            rec["match_without_same_race_dup"] = float(ok[k].mean())
        if c.startswith("trainer_prior"):
            k = ~m["tr_dup"].fillna(False).to_numpy(bool)
            rec["match_without_same_race_dup"] = float(ok[k].mean())
        out[c] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--future-day", help="学習表に無い日（エラー無く値が出るかだけ確認）")
    ap.add_argument("--races-per-day", type=int, default=6)
    ap.add_argument("--tables", default=str(sl.DEFAULT_TABLES))
    ap.add_argument("--out", default="/tmp/serve_layera_parity.json")
    a = ap.parse_args()
    tables = ROOT / a.tables
    t0 = time.time()
    hist = sl.load_history(tables)
    rep: dict = {"history_rows": int(len(hist)), "history_load_sec": round(time.time() - t0, 1), "days": {}}
    print(f"履歴 {len(hist):,} 行 / {rep['history_load_sec']}s", flush=True)
    for day in a.days:
        ref = _reference(day)
        card = _card(day, tables)
        coll = _collision(ref, card[["race_id", "horse_number", "jockey_id", "trainer_id"]].assign(race_id=lambda d: d["race_id"].astype(str)))
        ref_races = sorted(ref["race_id"].unique())
        t1 = time.time()
        got = sl.build_layer_a_features(card[card["race_id"].astype(str).isin(ref_races)], hist)
        secA = time.time() - t1
        res = {"A_all_races_one_call": {"sec": round(secA, 1), **_compare(got, ref, coll)}}
        # B: 各開催場の先頭・中ほど・最終のレースなどを 1 レースずつ
        pick = []
        by_venue = {}
        for r in ref_races:
            by_venue.setdefault(r[4:6], []).append(r)
        for v, rs in by_venue.items():
            idx = sorted({0, len(rs) // 2, len(rs) - 1, min(4, len(rs) - 1)})
            pick += [rs[i] for i in idx]
        pick = pick[: a.races_per_day * max(1, len(by_venue))]
        parts, secs = [], []
        for r in pick:
            t2 = time.time()
            parts.append(sl.build_layer_a_features(card[card["race_id"].astype(str) == r], hist))
            secs.append(time.time() - t2)
        gotB = pd.concat(parts, ignore_index=True)
        res["B_single_races"] = {"races": pick, "sec_per_call": round(float(np.mean(secs)), 1), **_compare(gotB, ref, coll)}
        first = [min(rs) for rs in by_venue.values()]
        res["B_first_race_of_venue"] = _compare(gotB[gotB["race_id"].isin(first)], ref, coll) if set(first) & set(pick) else None
        rep["days"][day] = res
        print(day, "A", round(secA, 1), "s / B", round(float(np.mean(secs)), 1), "s/call", flush=True)
    if a.future_day:
        card = _card(a.future_day, tables)
        rids = sorted(card["race_id"].astype(str).unique())[:3]
        t3 = time.time()
        got = sl.build_layer_a_features(card[card["race_id"].astype(str).isin(rids)], hist)
        rep["future_day"] = {"day": a.future_day, "rows": int(len(got)), "sec": round(time.time() - t3, 1),
                             "null_share": {c: round(float(got[c].isna().mean()), 3) for c in sl.OUTPUT_COLS if got[c].isna().any()}}
    rep["max_rss_mb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("保存:", a.out, "max_rss_mb", rep["max_rss_mb"], flush=True)


if __name__ == "__main__":
    main()
