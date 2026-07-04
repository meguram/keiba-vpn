"""
dataset_split_manifest.json に基づく race 単位の train / valid / test 割当。

企画書（horse_pre_race_dataset_spec / train_valid_test_split_strategy）と整合させる。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.data_paths import MODELING_META_DIR

DEFAULT_MANIFEST = MODELING_META_DIR / "dataset_split_manifest.json"


def race_calendar_year(race_id: Any) -> str:
    """race_id 先頭 4 桁を開催年とみなす（JRA 例: 202401010101）。"""
    s = str(race_id).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return "unknown"


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_MANIFEST
    if not p.is_file():
        raise FileNotFoundError(f"split manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def assign_split_column(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    protocol_key: str = "model_selection_primary",
    race_id_col: str = "race_id",
) -> pd.Series:
    """
    各行（馬行）に train / valid / test / unknown を付与。
    protocols.<protocol_key> の train_years / valid_years / test_years を使用。
    """
    if "protocols" not in manifest or protocol_key not in manifest["protocols"]:
        raise KeyError(f"protocol {protocol_key} not in manifest")

    proto = manifest["protocols"][protocol_key]
    train_y = set(proto.get("train_years", []))
    valid_y = set(proto.get("valid_years", []))
    test_y = set(proto.get("test_years", []))

    years = df[race_id_col].map(race_calendar_year)
    out = pd.Series("unknown", index=df.index, dtype=object)
    out[years.isin(train_y)] = "train"
    out[years.isin(valid_y)] = "valid"
    out[years.isin(test_y)] = "test"
    return out
