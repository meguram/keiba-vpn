"""疑似特徴量ビルダー（ワークフロー動作確認用）。

本物の特徴量ビルダー（約1000特徴量）が完成するまでの代用。出走馬ごとに、``race_id`` と
``horse_id`` から決まる**再現可能な乱数**で N 列の特徴量を作る。中身に意味は無いが、
「出馬表 → 特徴量 → アンサンブル推論 → 結果保存」の流れが最後まで動くかの検証に使う。

本物のビルダーに差し替えるときは、``build(race_data) -> DataFrame`` と ``feature_names``
を満たすクラスを ``get_feature_builder`` から返せばよい（推論側のコードは変更不要）。
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

DEFAULT_N_FEATURES = 1000
META_COLUMNS = ["race_id", "horse_number", "horse_name", "horse_id"]


def pseudo_feature_names(n_features: int) -> list[str]:
    return [f"pf_{i:04d}" for i in range(n_features)]


class PseudoFeatureBuilder:
    """``race_data["race_card"]["entries"]`` の各馬に対し、決定的な疑似特徴量を作る。"""

    name = "pseudo"

    def __init__(self, n_features: int = DEFAULT_N_FEATURES, seed: int = 0):
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = n_features
        self.seed = seed

    @property
    def feature_names(self) -> list[str]:
        return pseudo_feature_names(self.n_features)

    def _row_rng(self, race_id: str, horse_id: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}:{race_id}:{horse_id}".encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def build(self, race_data: dict) -> pd.DataFrame:
        race_id = str(race_data.get("race_id") or "")
        card = race_data.get("race_card") or {}
        entries = [e for e in (card.get("entries") or []) if isinstance(e, dict)]

        meta = pd.DataFrame(
            {
                "race_id": [race_id] * len(entries),
                "horse_number": [int(e.get("horse_number") or 0) for e in entries],
                "horse_name": [str(e.get("horse_name") or "") for e in entries],
                "horse_id": [str(e.get("horse_id") or "") for e in entries],
            }
        )
        values = np.empty((len(entries), self.n_features), dtype="float32")
        for i, e in enumerate(entries):
            rng = self._row_rng(race_id, str(e.get("horse_id") or i))
            values[i] = rng.standard_normal(self.n_features).astype("float32")

        feats = pd.DataFrame(values, columns=self.feature_names)
        return pd.concat([meta, feats], axis=1)


def get_feature_builder(name: str | None = None):
    """使用する特徴量ビルダーを返す。``KEIBA_FEATURE_BUILDER``（既定 ``pseudo``）で切り替える。"""
    chosen = (name or os.environ.get("KEIBA_FEATURE_BUILDER") or "pseudo").strip().lower()
    if chosen == "pseudo":
        n = int(os.environ.get("KEIBA_PSEUDO_N_FEATURES") or DEFAULT_N_FEATURES)
        return PseudoFeatureBuilder(n_features=n)
    raise NotImplementedError(
        f"特徴量ビルダー {chosen!r} は未実装です（現状は 'pseudo' のみ）。"
        "本物のビルダー完成後に get_feature_builder へ追加してください。"
    )
