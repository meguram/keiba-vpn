"""Phase 4: LSTM ラップ系列予測（スケルトン）。"""

from __future__ import annotations

from typing import Any

import numpy as np


class LapLSTMModel:
    """LSTM ラップ予測 — Phase 4 で学習・推論を実装。"""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None

    def load(self) -> bool:
        if not self.model_path:
            return False
        try:
            import torch  # noqa: F401
            # TODO Phase 4: torch.load(self.model_path)
            return False
        except ImportError:
            return False

    def predict(self, features: np.ndarray) -> list[float]:
        """1F 毎ラップ秒の系列を返す。"""
        if self._model is None:
            raise RuntimeError("LSTM model not loaded — use lap_predictor (LightGBM) for Phase 3")
        return []

    def to_spec_payload(self, lap_secs: list[float]) -> dict[str, Any]:
        return {
            "lap_times": [
                {"furlong_index": i + 1, "predicted_lap_sec": round(sec, 2)}
                for i, sec in enumerate(lap_secs)
            ]
        }
