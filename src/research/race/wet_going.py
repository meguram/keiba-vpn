"""道悪適性（走法マップの道悪の地図の軸。芝とダートは別に計算する）の 1 走ごとの材料。

定義（ユーザー指示 2026-11-05）: 稍重以上の馬場で、良馬場のときより走れるタイプ（道悪が上手）。
各走に (p, w) を付ける。p = そのレース内の着順スコア（+1 = 1 着 … -1 = 最下位）、w = 馬場状態の重さ（良 0・稍重 1・重 2・不良 3）。
馬ごとの道悪適性 = 同じ芝ダの直近の走での (p, w) の共分散（道悪ほど成績が上がるほど高い）。
共分散なので、馬の能力そのもの（p の平均）は引かれ、道悪の走が無い馬は 0（= 偏差値 50）になる。組み立て（加重・引き寄せ）は
running_profile.profiles_as_of / profile_table がやる。ここでは 1 走ごとの p, w を作る（芝は wtT_*、ダートは wtD_*。別の芝ダの走は NaN）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

STORE_VERSION = "1"
MIN_FIELD = 5
WET_LEVEL = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
_JUMP = r"障害|ジャンプ|JS\(|JG|J・G"
_COLS = ["race_id", "date", "surface", "race_name", "track_condition", "finish_position", "horse_id"]
OUT_COLS = ["wtT_p", "wtT_w", "wtD_p", "wtD_w"]


def _read(years: Iterable[int], base: Path) -> pd.DataFrame:
    parts = []
    for y in years:
        f = base / str(y) / "race_result_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=_COLS))
    if not parts:
        return pd.DataFrame(columns=_COLS)
    d = pd.concat(parts, ignore_index=True)
    d = d[d["surface"].isin(["芝", "ダート"]) & ~d["race_name"].fillna("").str.contains(_JUMP)].copy()
    d["finish_position"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d = d[d["finish_position"] > 0].copy()
    d["horse_id"] = d["horse_id"].astype(str)
    d["race_id"] = d["race_id"].astype(str)
    return d.reset_index(drop=True)


def compute(runs: pd.DataFrame) -> pd.DataFrame:
    """列: race_id, horse_id, wtT_p, wtT_w, wtD_p, wtD_w。馬場状態が読めない走・頭数 5 未満は p, w とも NaN。"""
    d = runs.copy()
    n = d.groupby("race_id")["horse_id"].transform("size")
    p = np.where(n >= MIN_FIELD, ((n + 1) / 2 - d["finish_position"]) / ((n - 1) / 2).replace(0, np.nan), np.nan)
    w = d["track_condition"].map(WET_LEVEL).astype(float)
    p = np.where(np.isnan(w), np.nan, p)
    w = np.where(np.isnan(p), np.nan, w)
    turf = (d["surface"] == "芝").to_numpy()
    out = d[["race_id", "horse_id"]].copy()
    out["wtT_p"], out["wtT_w"] = np.where(turf, p, np.nan), np.where(turf, w, np.nan)
    out["wtD_p"], out["wtD_w"] = np.where(~turf, p, np.nan), np.where(~turf, w, np.nan)
    return out.drop_duplicates(["race_id", "horse_id"]).reset_index(drop=True)


def load(years: Iterable[int], base: Path, store: Path, sig: dict) -> pd.DataFrame:
    """保存した 1 走ごとの p, w を読む。出走記録のファイルが変わっていたら作り直す。"""
    mf = store.with_suffix(".json")
    try:
        if store.exists() and mf.exists():
            meta = json.loads(mf.read_text(encoding="utf-8"))
            if meta.get("version") == STORE_VERSION and meta.get("source") == sig:
                return pd.read_parquet(store)
    except (OSError, ValueError):
        pass
    out = compute(_read(years, base))
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(store)
    mf.write_text(json.dumps({"version": STORE_VERSION, "source": sig, "rows": len(out)}, ensure_ascii=False), encoding="utf-8")
    return out
