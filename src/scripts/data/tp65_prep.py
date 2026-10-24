"""tp-65: めぐ指数と netkeiba タイム指数を『走（馬×日付）』単位に揃えた表を作る。ホストで実行。

python -m src.scripts.data.tp65_prep [--out /home/jovyan/work/tp65_out] [--years 2020 ... 2026]

netkeiba のタイム指数は、各レースの出走前ページ（race_index_flat の speed_recent = 直近3走の指数）にだけ載る。
出馬表の過去走（race_shutuba_past_flat の past_races、新しい順）の日付と突き合わせて、各走（horse_id, 日付）の指数を作る。
同じ走は次走・次々走・3走先の出走前ページに最大 3 回現れる（lag0/1/2）ので、値の改訂（基準の更新）も測れる。
出力: runs.parquet（1 行 = 1 走。結果・めぐ指数・netkeiba 指数 lag0/1/2）、prep_report.json。
指数 0 は欠損として扱う（実データで 0 の混入を確認する）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
RES_COLS = ["race_id", "date", "venue", "surface", "distance", "grade", "race_class", "track_condition", "field_size",
            "horse_id", "finish_position", "time_sec", "margin", "last_3f", "passing_order", "jockey_weight"]
MEGU_COLS = ["race_id", "horse_id", "megu_index", "megu_index_raw", "megu_index_v3a", "megu_index_v3a_robust",
             "megu_index_v3a_abs", "megu_index_v3a_abs_robust", "megu_index_v3a_feat", "megu_v3a_status",
             "megu_v3a_low_confidence", "par_sample_count", "v3a_corrected_sec", "v3a_par_sec"]


def _read(year: int, name: str, cols: list[str]) -> pd.DataFrame:
    p = TABLES / str(year) / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=cols)
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(p).schema_arrow.names)
    d = pd.read_parquet(p, columns=[c for c in cols if c in have])
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d


def observations(years: list[int]) -> pd.DataFrame:
    """出走前ページの (レース, 馬) ごとに、過去走の日付と指数を縦に展開する。"""
    out = []
    for y in years:
        idx = _read(y, "race_index_flat", ["race_id", "horse_id", "date", "speed_recent"])
        pst = _read(y, "race_shutuba_past_flat", ["race_id", "horse_id", "past_races"])
        m = idx.merge(pst, on=["race_id", "horse_id"], how="inner")
        rows = []
        for race_id, horse_id, date, sr, pr in zip(m["race_id"], m["horse_id"], m["date"], m["speed_recent"], m["past_races"]):
            try:
                pr = json.loads(pr) if isinstance(pr, str) else (pr or [])
                sr = json.loads(sr) if isinstance(sr, str) else (list(sr) if sr is not None else [])
            except Exception:  # noqa: BLE001
                continue
            for lag in range(min(3, len(pr), len(sr))):
                d = str(pr[lag].get("date") or "").replace(".", "-")
                if len(d) != 10:
                    continue
                rows.append((horse_id, d, lag, float(sr[lag]), race_id, date))
        out.append(pd.DataFrame(rows, columns=["horse_id", "run_date", "lag", "val", "seen_at_race", "seen_at_date"]))
    return pd.concat(out, ignore_index=True)


def build(years: list[int]) -> tuple[pd.DataFrame, dict]:
    obs = observations(years)
    obs.loc[obs["val"] == 0, "val"] = np.nan
    wide = obs.pivot_table(index=["horse_id", "run_date"], columns="lag", values="val", aggfunc="first")
    wide.columns = [f"nk_lag{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    res = pd.concat([_read(y, "race_result_flat", RES_COLS) for y in years], ignore_index=True)
    mg = pd.concat([_read(y, "megu_index_flat", MEGU_COLS) for y in years], ignore_index=True)
    runs = res.merge(mg, on=["race_id", "horse_id"], how="left")
    runs["date"] = runs["date"].astype(str).str[:10]
    runs = runs.merge(wide, left_on=["horse_id", "date"], right_on=["horse_id", "run_date"], how="left").drop(columns=["run_date"])
    # netkeiba の値: 最初に出たもの（その走の次の出走前ページに載った値。予測時点で見える値）。nk_first_seen がその日付
    first_val = (obs.dropna(subset=["val"]).sort_values("seen_at_date").groupby(["horse_id", "run_date"])
                 .agg(nk=("val", "first"), nk_first_seen=("seen_at_date", "first")).reset_index())
    runs = runs.drop(columns=["nk_first_seen"], errors="ignore").merge(first_val, left_on=["horse_id", "date"], right_on=["horse_id", "run_date"], how="left").drop(columns=["run_date"])
    rep = {
        "years": years,
        "n_runs": int(len(runs)),
        "megu_v3a_cover": float(runs["megu_index_v3a"].notna().mean()),
        "nk_cover": float(runs["nk"].notna().mean()),
        "both_cover": float((runs["nk"].notna() & runs["megu_index_v3a"].notna()).mean()),
        "nk_lag_cover": {c: float(runs[c].notna().mean()) for c in ("nk_lag0", "nk_lag1", "nk_lag2")},
        "obs_zero_share": float((obs["val"].isna()).mean()),
    }
    return runs, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/tp65_out")
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2020, 2027)))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, rep = build(a.years)
    runs.to_parquet(out / "runs.parquet", index=False)
    (out / "prep_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
