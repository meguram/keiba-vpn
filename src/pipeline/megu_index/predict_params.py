"""想定めぐ指数の学習済みチューニングパラメータ（1点=0.1秒は不変）。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline.megu_index.common import WEIGHTS_B
from src.pipeline.megu_index.condition_weights import DEFAULT_CONDITION_WEIGHTS

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "megu_predict_params.json"
)


@dataclass(frozen=True)
class PredictTuning:
    """
    予測キャリブレーション（すべて秒空間または指数の線形変換のみ）。

    - par_blend: 0=過去 megu 加重平均, 1=par 換算能力平均（補正済みタイム空間でブレンド）
    - ability_bias_sec: ブレンド後能力から減算する秒（+0.1秒 → 指数+1点）
    - transfer_strength: 条件転換 Δ秒 の倍率
    """

    par_blend: float = 0.0
    ability_bias_sec: float = 0.0
    transfer_strength: float = 1.0


@dataclass
class PredictParams:
    history_weights: list[float] = field(default_factory=lambda: list(WEIGHTS_B))
    condition_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONDITION_WEIGHTS)
    )
    tuning: PredictTuning = field(default_factory=PredictTuning)
    meta: dict[str, Any] = field(default_factory=dict)

    def normalized_history_weights(self, n: int) -> list[float]:
        w = self.history_weights[: max(n, 1)]
        if len(w) < n:
            w = w + [0.0] * (n - len(w))
        s = sum(w)
        return [x / s for x in w] if s > 0 else [1.0 / n] * n


def _parse_tuning(raw: dict[str, Any] | None) -> PredictTuning:
    if not raw:
        return PredictTuning()
    return PredictTuning(
        par_blend=float(raw.get("par_blend", 0.0)),
        ability_bias_sec=float(raw.get("ability_bias_sec", 0.0)),
        transfer_strength=float(raw.get("transfer_strength", 1.0)),
    )


def load_predict_params(path: Path | None = None) -> PredictParams:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        logger.debug("predict params not found: %s (defaults)", cfg_path)
        return PredictParams()
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        hw = data.get("history_weights") or WEIGHTS_B
        cw = {**DEFAULT_CONDITION_WEIGHTS, **(data.get("condition_weights") or {})}
        tuning = _parse_tuning(data.get("tuning"))
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        return PredictParams(
            history_weights=[float(x) for x in hw],
            condition_weights={k: float(v) for k, v in cw.items()},
            tuning=tuning,
            meta=meta,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("failed to load predict params %s: %s", cfg_path, e)
        return PredictParams()


def save_predict_params(params: PredictParams, path: Path | None = None) -> Path:
    cfg_path = path or DEFAULT_CONFIG_PATH
    payload = {
        "history_weights": params.history_weights,
        "condition_weights": {**DEFAULT_CONDITION_WEIGHTS, **params.condition_weights},
        "tuning": asdict(params.tuning),
        "meta": params.meta,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def load_predict_tuning(path: Path | None = None) -> PredictTuning:
    return load_predict_params(path).tuning
