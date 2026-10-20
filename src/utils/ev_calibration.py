"""期待値（EV）の較正。モデルの EV は『モデル確率が市場より尖っている』ぶん実現回収率より高く出る（検証: ev_calibration_audit 章）ので、
実現回収率に合わせて、損益分岐（払戻率）に向けて縮める。

    EV_adj = base + k × (EV − base)      （base = その券種の払戻率、k は過去データで当てはめた 0〜1）

既定は無効（config/ev_calibration.json の enabled が false）。有効にすると、買い推奨度（buy_recommendation_tier）と
レース価値の色分け（race_value_summary）が縮約した値で判定される。k は audit_ev_calibration.py の結果から作る。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ev_calibration.json"
DEFAULT_BASE = {"win": 0.80, "top2": 0.775, "place": 0.80, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "portfolio": 0.80}


@lru_cache(maxsize=1)
def load_config() -> dict:
    try:
        d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def calibrated_ev(ev: float | None, kind: str, config: dict | None = None) -> float | None:
    """kind: win / top2 / place / umaren / wide / umatan / portfolio。無効、k が無い、EV が None のときはそのまま返す。"""
    if ev is None:
        return None
    cfg = load_config() if config is None else config
    if not cfg.get("enabled"):
        return ev
    k = (cfg.get("k") or {}).get(kind)
    if k is None:
        return ev
    base = (cfg.get("base") or {}).get(kind, DEFAULT_BASE.get(kind, 0.8))
    return base + float(k) * (float(ev) - float(base))
