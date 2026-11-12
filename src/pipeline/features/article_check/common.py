"""成績表の読み込み（平地のみ・障害除外）。キーは race_id・horse_id の ID。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES_DIR = Path(__file__).resolve().parents[4] / "data/page_reference/tables"
_COLS = ["race_id", "date", "venue_code", "surface", "distance", "field_size", "finish_position", "horse_number", "horse_id", "time_sec", "race_class", "race_name"]


def load_runs(years, tables_dir: Path = TABLES_DIR) -> pd.DataFrame:
    """1 走 1 行。取消・中止・失格（実データでは着順 -1・time_sec 0）は走として数えない（過去走・出走数・相手の後走から除く）。障害は race_class/race_name で除く。"""
    d = pd.concat([pd.read_parquet(Path(tables_dir) / str(y) / "race_result_flat.parquet", columns=_COLS) for y in years], ignore_index=True)
    d = d[d["surface"].isin(("芝", "ダート"))].copy()
    jump = d["race_class"].astype(str).str.contains("障害") | d["race_name"].astype(str).str.contains("障害|ジャンプ")
    d = d[~d["race_id"].isin(d.loc[jump, "race_id"])].drop(columns=["race_class", "race_name"])
    d = d[(d["finish_position"] > 0) & (d["time_sec"] > 0)]
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    d["venue_code"] = d["venue_code"].astype(str)
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:10])
    return d.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


def finish_pct(d: pd.DataFrame) -> pd.Series:
    """着順の相対位置（1 着 = 1、最下位 = 0）。着順なしは欠損。"""
    return 1.0 - (d["finish_position"] - 1.0) / (d["field_size"] - 1.0).clip(lower=1.0)
