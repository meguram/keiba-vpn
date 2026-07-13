"""想定めぐ指数: 過去走と今回レースの条件不一致に対する重み係数。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.pipeline.megu_index.common import MAJOR_DIST_CHANGE_M

logger = logging.getLogger(__name__)

# 最適化ノートブック (nb-megu-predict-condition-weight-opt) の出力先
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "megu_predict_condition_weights.json"
)

# 最適化結果 (2024–2025 STG, nb-megu-predict-condition-weight-opt)
DEFAULT_CONDITION_WEIGHTS: dict[str, float] = {
    "w_match": 1.0,
    "w_surface_only": 0.55,
    "w_distance_only": 0.70,
    "w_both": 0.30,
}


def condition_mismatch_multiplier(
    surface_hist: str,
    distance_hist: int,
    surface_target: str,
    distance_target: int,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """
    過去走1件に対する条件一致度の重み倍率（0〜1）。

    - 芝/ダ不一致 → w_surface_only（距離差 < 600m）
    - 同一馬場で距離差 ±600m 以上 → w_distance_only
    - 両方 → w_both
    - 一致 → w_match (=1.0)
    """
    w = {**DEFAULT_CONDITION_WEIGHTS, **(weights or {})}
    sf_diff = str(surface_hist or "").strip() != str(surface_target or "").strip()
    try:
        dist_diff = abs(int(distance_target) - int(distance_hist)) >= MAJOR_DIST_CHANGE_M
    except (TypeError, ValueError):
        dist_diff = False

    if sf_diff and dist_diff:
        return float(w["w_both"])
    if sf_diff:
        return float(w["w_surface_only"])
    if dist_diff:
        return float(w["w_distance_only"])
    return float(w["w_match"])


def apply_condition_weights(
    base_weights: list[float],
    history: list[dict[str, Any]],
    *,
    surface_target: str,
    distance_target: int,
    condition_weights: dict[str, float] | None = None,
) -> tuple[list[float], list[float]]:
    """
    ベース重み (WEIGHTS_B) に条件一致倍率を掛けて正規化する。

    Returns:
        (normalized_weights, raw_multipliers_per_row)
    """
    n = min(len(base_weights), len(history))
    if n == 0:
        return [], []

    multipliers: list[float] = []
    for h in history[:n]:
        mult = condition_mismatch_multiplier(
            str(h.get("surface") or ""),
            int(h.get("distance") or 0),
            surface_target,
            distance_target,
            weights=condition_weights,
        )
        multipliers.append(mult)

    raw = [float(base_weights[i]) * multipliers[i] for i in range(n)]
    total = sum(raw)
    if total <= 0:
        norm = [1.0 / n] * n
    else:
        norm = [w / total for w in raw]
    return norm, multipliers


def load_condition_weights(path: Path | None = None) -> dict[str, float]:
    """config/megu_predict_params.json または condition_weights JSON を読み込む。"""
    from src.pipeline.megu_index.predict_params import DEFAULT_CONFIG_PATH, load_predict_params

    if DEFAULT_CONFIG_PATH.is_file():
        pp = load_predict_params()
        return dict(pp.condition_weights)

    legacy_path = Path(__file__).resolve().parents[3] / "config" / "megu_predict_condition_weights.json"
    cfg_path = path or legacy_path
    if not cfg_path.is_file():
        logger.debug("condition weights config not found: %s (using defaults)", cfg_path)
        return dict(DEFAULT_CONDITION_WEIGHTS)
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        weights = data.get("weights") if isinstance(data, dict) else data
        if not isinstance(weights, dict):
            return dict(DEFAULT_CONDITION_WEIGHTS)
        merged = {**DEFAULT_CONDITION_WEIGHTS, **{k: float(v) for k, v in weights.items()}}
        return merged
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("failed to load condition weights from %s: %s", cfg_path, e)
        return dict(DEFAULT_CONDITION_WEIGHTS)


def save_condition_weights(
    weights: dict[str, float],
    *,
    path: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """最適化結果を JSON に保存。"""
    cfg_path = path or DEFAULT_CONFIG_PATH
    payload: dict[str, Any] = {
        "weights": {**DEFAULT_CONDITION_WEIGHTS, **weights},
        "meta": meta or {},
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg_path
