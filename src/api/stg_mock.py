"""
stg 環境用モックモデル
======================
KEIBA_ENV=stg のとき、ML モデルを使わずにモックデータを返す。
各エンドポイントは実際のモデル計算に失敗したとき、
または stg 環境での応答補完としてこのモジュールを使用する。

提供するモック:
  - mock_predictions(race_id, horse_ids)  → race_predictions 形式
  - mock_tracking_difficulty(race_id, storage) → tracking-difficulty 形式
  - mock_race_quality_race(race_id, storage) → race-quality/race 形式
  - mock_race_quality_day(date_str, storage) → race-quality/day 形式
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

MOCK_MODEL_VERSION = "stg-mock-v1"

# 8 軸 ID (race_quality_model.py と同一順)
QUALITY_AXIS_IDS = [
    "closing", "baseline_speed", "stamina", "burst",
    "sustain_run", "position_draw", "class_flat", "mud_going",
]


def _rng(seed: str) -> random.Random:
    return random.Random(seed)


def _normalize(vals: list[float]) -> list[float]:
    total = sum(vals)
    if total <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / total for v in vals]


def mock_predictions(race_id: str, horse_ids: list[str]) -> dict[str, Any]:
    """race_predictions GCS 形式のモックデータ。"""
    rng = _rng(race_id)
    if not horse_ids:
        return {"race_id": race_id, "status": "no_entries", "horses": []}

    raw = [rng.uniform(0.5, 3.0) for _ in horse_ids]
    win_probs = _normalize(raw)

    horses_out = []
    for i, horse_id in enumerate(horse_ids):
        wp = win_probs[i]
        pp = min(wp * 2.2, 0.95)
        horses_out.append({
            "horse_id": horse_id,
            "post_no": i + 1,
            "win_prob": round(wp, 4),
            "win_probability": round(wp, 4),
            "place_prob": round(pp, 4),
            "show_prob": round(min(pp * 1.3, 0.98), 4),
            "place_probability": round(min(pp * 1.3, 0.98), 4),
            "predicted_win_odds": round(1.0 / max(wp, 0.01), 1),
            "predicted_place_odds": round(1.0 / max(pp, 0.01), 1),
            "expected_win_roi": round(rng.uniform(-0.3, 0.8), 2),
            "expected_show_roi": round(rng.uniform(-0.1, 0.5), 2),
            "predicted_position": i + 1,
            "predicted_running_style": rng.choice(["逃", "先", "差", "追"]),
            "is_value_bet": rng.random() > 0.7,
        })

    pace_cats = ["HIGH", "MIDDLE", "SLOW"]
    pace_cat = rng.choice(pace_cats)
    lap_times = []
    n_furlongs = rng.choice([6, 8, 10, 12])
    for fi in range(n_furlongs):
        lap_times.append({
            "furlong_index": fi + 1,
            "predicted_lap_sec": round(rng.uniform(11.0, 14.5), 2),
        })

    return {
        "race_id": race_id,
        "model_version": MOCK_MODEL_VERSION,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "pace_prediction": {
            "pace_category": pace_cat,
            "lap_times": lap_times,
        },
        "horses": horses_out,
        "_mock": True,
    }


def mock_tracking_difficulty(race_id: str, storage=None) -> dict[str, Any]:
    """tracking-difficulty 形式のモックデータ。"""
    rng = _rng(race_id + "_td")

    # race_shutuba から horse_ids を取得できれば使う
    horse_ids: list[str] = []
    if storage:
        try:
            card = storage.load("race_shutuba", race_id)
            if card:
                horse_ids = [e.get("horse_id", "") for e in (card.get("entries") or []) if e.get("horse_id")]
        except Exception:
            pass

    if not horse_ids:
        # フォールバック: ダミーエントリ
        horse_ids = [f"dummy_{i:04d}" for i in range(rng.randint(8, 16))]

    entries = []
    for i, hid in enumerate(horse_ids):
        entries.append({
            "horse_id": hid,
            "post_no": i + 1,
            "horse_name": f"馬{i+1}",
            "tracking_score": round(rng.uniform(40, 90), 1),
            "pace_score": round(rng.uniform(40, 90), 1),
            "position_score": round(rng.uniform(40, 90), 1),
            "difficulty_rank": i + 1,
            "running_style": rng.choice(["逃", "先", "差", "追"]),
            "pace_aptitude": rng.choice(["high", "middle", "slow"]),
            "corner_positions": [
                rng.randint(1, len(horse_ids)),
                rng.randint(1, len(horse_ids)),
                rng.randint(1, len(horse_ids)),
                rng.randint(1, len(horse_ids)),
            ],
        })

    pace_cats = ["high", "middle", "slow"]
    pace_cat = rng.choice(pace_cats)

    return {
        "race_id": race_id,
        "status": "ok",
        "pace_category": pace_cat,
        "pace_score": round(rng.uniform(50, 90), 1),
        "field_difficulty": round(rng.uniform(40, 85), 1),
        "entries": entries,
        "_mock": True,
        "_mock_model": MOCK_MODEL_VERSION,
    }


def mock_race_quality_race(race_id: str, storage=None) -> dict[str, Any]:
    """race-quality/race 形式のモックデータ（9確率）。"""
    rng = _rng(race_id + "_rq")

    raw = [rng.uniform(0.1, 2.0) for _ in QUALITY_AXIS_IDS] + [rng.uniform(0.05, 0.5)]
    probs = _normalize(raw)

    axes_out = []
    for i, axis_id in enumerate(QUALITY_AXIS_IDS):
        axes_out.append({
            "id": axis_id,
            "probability": round(probs[i], 4),
        })
    axes_out.append({
        "id": "unknown",
        "probability": round(probs[-1], 4),
    })

    dominant = max(axes_out, key=lambda x: x["probability"])

    return {
        "race_id": race_id,
        "axes": axes_out,
        "dominant_axis": dominant["id"],
        "dominant_probability": dominant["probability"],
        "confidence": round(rng.uniform(0.4, 0.9), 2),
        "segment_key": "芝_1200-1999",
        "_mock": True,
    }


def mock_race_quality_day(date_str: str, storage=None) -> dict[str, Any]:
    """race-quality/day 形式のモックデータ。"""
    # date_str 内の race_ids をローカル race_lists から取得
    import json
    from pathlib import Path
    import os

    project_root = Path(os.environ.get("KEIBA_ROOT", "/home/jovyan/work/keiba-vpn"))
    race_list_path = project_root / "data" / "page_reference" / "race_lists" / f"{date_str}.json"

    race_items = []
    if race_list_path.exists():
        try:
            rl = json.loads(race_list_path.read_text(encoding="utf-8"))
            for r in rl.get("races") or []:
                rid = r.get("race_id")
                if rid:
                    race_items.append({
                        "race_id": rid,
                        "race_name": r.get("race_name", ""),
                        "venue": r.get("venue", ""),
                        "round": r.get("round", 0),
                    })
        except Exception:
            pass

    results = []
    for r in race_items:
        rq = mock_race_quality_race(r["race_id"], storage)
        results.append({
            **r,
            "axes": rq["axes"],
            "dominant_axis": rq["dominant_axis"],
            "dominant_probability": rq["dominant_probability"],
            "confidence": rq["confidence"],
        })

    return {
        "date": date_str,
        "races": results,
        "_mock": True,
    }
