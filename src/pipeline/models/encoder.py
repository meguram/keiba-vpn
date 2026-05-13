"""
カテゴリ変数エンコーダー

予測パイプラインで使われるカテゴリ変数をモデル入力用に変換する。
学習時に fit → 推論時に transform する構造。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SURFACE_MAP = {"芝": 0, "ダート": 1, "ダ": 1, "障": 2}
DIRECTION_MAP = {"左": 0, "右": 1, "直": 2}
WEATHER_MAP = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 4, "雪": 5}
TRACK_COND_MAP = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3, "不": 3}
SEX_MAP = {"牡": 0, "牝": 1, "セ": 2}

VENUE_MAP = {
    "札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5,
    "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10,
}


class FeatureEncoder:
    """カテゴリ変数エンコーダー + 高頻度カテゴリの Target Encoding サポート"""

    def __init__(self):
        self.sire_map: dict[str, float] = {}
        self.dam_sire_map: dict[str, float] = {}
        self.jockey_map: dict[str, float] = {}
        self.trainer_map: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, target_col: str = "finish_position"):
        """学習データで target encoding のマッピングを構築する。"""
        if target_col not in df.columns:
            return self

        target = df[target_col].copy()
        is_top3 = (target >= 1) & (target <= 3)

        for col, attr in [("sire", "sire_map"), ("dam_sire", "dam_sire_map"),
                          ("jockey_name", "jockey_map"), ("trainer_name", "trainer_map")]:
            if col in df.columns:
                grouped = is_top3.groupby(df[col]).mean()
                setattr(self, attr, grouped.to_dict())

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ変数を数値に変換した新しいDataFrameを返す。"""
        out = df.copy()

        fixed_maps = {
            "surface": SURFACE_MAP, "direction": DIRECTION_MAP,
            "weather": WEATHER_MAP, "track_condition": TRACK_COND_MAP,
            "sex": SEX_MAP, "venue": VENUE_MAP,
        }

        for i in range(1, 6):
            fixed_maps[f"prev{i}_surface"] = SURFACE_MAP
            fixed_maps[f"prev{i}_track_cond"] = TRACK_COND_MAP

        for col, mapping in fixed_maps.items():
            if col in out.columns:
                out[col] = out[col].map(mapping).fillna(-1).astype(int)

        target_maps = {
            "sire": self.sire_map,
            "dam_sire": self.dam_sire_map,
            "jockey_name": self.jockey_map,
            "trainer_name": self.trainer_map,
        }
        global_mean = 0.2

        for col, mapping in target_maps.items():
            if col in out.columns:
                out[col + "_enc"] = out[col].map(mapping).fillna(global_mean)
                out.drop(columns=[col], inplace=True)

        drop_cols = {"race_id", "horse_name", "horse_id", "race_date"}
        out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True)

        return out

    def save(self, path: str):
        data = {
            "sire_map": self.sire_map,
            "dam_sire_map": self.dam_sire_map,
            "jockey_map": self.jockey_map,
            "trainer_map": self.trainer_map,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.sire_map = data.get("sire_map", {})
        self.dam_sire_map = data.get("dam_sire_map", {})
        self.jockey_map = data.get("jockey_map", {})
        self.trainer_map = data.get("trainer_map", {})
        return self
