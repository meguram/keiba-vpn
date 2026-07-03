"""レース全体のハロンラップを MECE に分類する（馬場・距離別閾値）。

lap_times はスタート→ゴールの 1F 通過タイム（秒）。
着側: L1F=laps[-1], L3-4F=laps[-4:-2], L5-4F=laps[-5:-3], L2-1F=laps[-2:].

分類（優先順・1つのみ）:
  1. スロー瞬発力ラップ … L5-4F 平均 − L2-1F 平均 ≥ 閾値
  2. 消耗ラップ … L1F − L3-4F 平均 ≥ 閾値
  3. 持続ラップ … 後半5F の標準偏差・幅 ≤ 閾値
  4. 標準ラップ … 上記以外

閾値は馬場（芝/ダ）× 距離帯（短/マイル/中/長）ごとに較正。
較正元: data/local/meta/lap_pattern_thresholds_calibration.json
（``python -m src.scripts.data.calibrate_lap_pattern_thresholds`` で再生成）
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.distance_band import (
    distance_group_key,
    distance_group_label_ja,
    distance_m,
)

_PATTERN_META: dict[str, dict[str, str]] = {
    "slow_burst": {
        "label": "スロー瞬発力ラップ",
        "summary": "中盤（L5-4F）を抑え、終盤（L2-1F）でラップが大きく加速しています。",
    },
    "consuming": {
        "label": "消耗ラップ",
        "summary": "L3-4F から L1F にかけてラップが遅くなり、上がりで脚を使い切る展開です。",
    },
    "sustained": {
        "label": "持続ラップ",
        "summary": "後半 5F のラップ変動が小さく、均一なペースが続いています。",
    },
    "standard": {
        "label": "標準ラップ",
        "summary": "持続・消耗・スロー瞬発のいずれの型にも強く当てはまりません。",
    },
}

# 較正サンプル不足時の相対係数（baseline_lap = 後半5F 平均秒）
_REL_BURST_PCT = 0.028
_REL_CONSUME_PCT = 0.020
_REL_SUSTAIN_STD_PCT = 0.010
_REL_SUSTAIN_RANGE_PCT = 0.030

# 馬場×距離の倍率（2026 cache 235レース較正 + 理論補正）
_SURFACE_DIST_MULT: dict[str, dict[str, dict[str, float]]] = {
    "turf": {
        "sprint": {"burst": 0.90, "consume": 0.92, "sustain_std": 0.95, "sustain_range": 0.92},
        "mile": {"burst": 1.00, "consume": 1.00, "sustain_std": 1.00, "sustain_range": 1.00},
        "middle": {"burst": 1.05, "consume": 1.06, "sustain_std": 1.06, "sustain_range": 1.08},
        "long": {"burst": 1.10, "consume": 1.10, "sustain_std": 1.10, "sustain_range": 1.12},
    },
    "dirt": {
        "sprint": {"burst": 0.85, "consume": 1.05, "sustain_std": 1.12, "sustain_range": 1.08},
        "mile": {"burst": 0.92, "consume": 1.10, "sustain_std": 1.15, "sustain_range": 1.10},
        "middle": {"burst": 0.98, "consume": 1.12, "sustain_std": 1.18, "sustain_range": 1.12},
        "long": {"burst": 1.05, "consume": 1.15, "sustain_std": 1.20, "sustain_range": 1.15},
    },
}

# 較正 JSON から読み込む絶対値下限（キー: surface_g|dist_g）
_CALIBRATED_ABS: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class LapPatternThresholds:
    burst_delta_min: float
    consume_delta_min: float
    sustain_std_max: float
    sustain_range_max: float
    profile_key: str
    profile_label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "burst_delta_min": round(self.burst_delta_min, 3),
            "consume_delta_min": round(self.consume_delta_min, 3),
            "sustain_std_max": round(self.sustain_std_max, 3),
            "sustain_range_max": round(self.sustain_range_max, 3),
            "profile_key": self.profile_key,
            "profile_label": self.profile_label,
        }


def parse_lap_times_sec(raw: Any) -> list[float]:
    if raw is None or raw == "":
        return []
    out: list[float] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (int, float)):
                v = float(item)
            elif isinstance(item, dict):
                v = float(
                    item.get("time_sec")
                    or item.get("seconds")
                    or item.get("time")
                    or 0
                )
            else:
                continue
            if math.isfinite(v) and v > 0:
                out.append(v)
        return out
    if isinstance(raw, str):
        s = raw.replace("－", "-").replace("—", "-")
        for token in s.split("-"):
            try:
                v = float(token.strip())
                if math.isfinite(v) and v > 0:
                    out.append(v)
            except (TypeError, ValueError):
                continue
    return out


def surface_group(surface: Any) -> str:
    s = str(surface or "").strip()
    if "ダ" in s or s.lower() in ("dirt", "ダート"):
        return "dirt"
    if "芝" in s or s.lower() in ("turf", "grass"):
        return "turf"
    return "other"


def _profile_label(surface_g: str, dist_g: str) -> str:
    surf = {"turf": "芝", "dirt": "ダート", "other": "その他"}.get(surface_g, surface_g)
    return f"{surf}・{distance_group_label_ja(dist_g)}"


def _load_calibrated_abs() -> dict[str, dict[str, float]]:
    global _CALIBRATED_ABS
    if _CALIBRATED_ABS is not None:
        return _CALIBRATED_ABS
    path = Path(__file__).resolve().parents[2] / "data" / "local" / "meta" / "lap_pattern_thresholds_calibration.json"
    out: dict[str, dict[str, float]] = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for key, row in raw.items():
                if not isinstance(row, dict) or row.get("n", 0) < 8:
                    continue
                burst = row.get("burst_p80") or row.get("burst_p70")
                consume = row.get("consume_p70")
                std_p = row.get("std_p30")
                range_p = row.get("range_p30")
                if burst is None or consume is None:
                    continue
                out[key] = {
                    "burst": max(0.20, min(0.55, float(burst) * 0.88)),
                    "consume": max(0.15, min(0.42, float(consume) * 0.92)),
                    "sustain_std": max(0.08, min(0.22, float(std_p or 0.12) * 1.12)),
                    "sustain_range": max(0.22, min(0.58, float(range_p or 0.35) * 1.15)),
                }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    _CALIBRATED_ABS = out
    return out


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _stdev(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return 0.0 if vals else None
    return statistics.pstdev(vals)


def compute_lap_metrics(laps: list[float]) -> dict[str, Any] | None:
    n = len(laps)
    if n < 4:
        return None
    l1 = laps[-1]
    l34_avg = _mean(laps[-4:-2])
    l21_avg = _mean(laps[-2:])
    l54_avg = _mean(laps[-5:-3]) if n >= 5 else None
    last5 = laps[-5:] if n >= 5 else laps[-n:]
    last5_std = _stdev(last5)
    last5_range = max(last5) - min(last5) if last5 else None
    baseline = _mean(last5) or _mean(laps)
    burst_delta = (l54_avg - l21_avg) if l54_avg is not None and l21_avg is not None else None
    consume_delta = (l1 - l34_avg) if l34_avg is not None else None
    return {
        "n_furlongs": n,
        "l1_sec": l1,
        "l34_avg_sec": l34_avg,
        "l21_avg_sec": l21_avg,
        "l54_avg_sec": l54_avg,
        "burst_delta": burst_delta,
        "consume_delta": consume_delta,
        "last5_std": last5_std,
        "last5_range": last5_range,
        "baseline_lap": baseline,
    }


def resolve_thresholds(
    surface: Any = None,
    distance: Any = None,
    baseline_lap: float | None = None,
    n_furlongs: int | None = None,
) -> LapPatternThresholds:
    surface_g = surface_group(surface)
    dist_g = distance_group_key(distance, n_furlongs)
    if surface_g == "other":
        surface_g = "turf"
    profile_key = f"{surface_g}|{dist_g}"
    mult = _SURFACE_DIST_MULT.get(surface_g, _SURFACE_DIST_MULT["turf"]).get(
        dist_g, _SURFACE_DIST_MULT["turf"]["mile"]
    )
    bl = baseline_lap if baseline_lap and baseline_lap > 0 else 12.0

    rel_burst = max(0.22, _REL_BURST_PCT * bl * mult["burst"])
    rel_consume = max(0.16, _REL_CONSUME_PCT * bl * mult["consume"])
    rel_std = max(0.08, _REL_SUSTAIN_STD_PCT * bl * mult["sustain_std"] + 0.02)
    rel_range = max(0.22, _REL_SUSTAIN_RANGE_PCT * bl * mult["sustain_range"] + 0.04)

    cal = _load_calibrated_abs().get(profile_key)
    if cal:
        burst = max(rel_burst, cal["burst"])
        consume = max(rel_consume, cal["consume"])
        sustain_std = min(rel_std, cal["sustain_std"])
        sustain_range = min(rel_range, cal["sustain_range"])
    else:
        burst, consume, sustain_std, sustain_range = rel_burst, rel_consume, rel_std, rel_range

    return LapPatternThresholds(
        burst_delta_min=burst,
        consume_delta_min=consume,
        sustain_std_max=sustain_std,
        sustain_range_max=sustain_range,
        profile_key=profile_key,
        profile_label=_profile_label(surface_g, dist_g),
    )


def classify_race_lap_pattern(
    lap_times: Any,
    surface: Any = None,
    distance: Any = None,
) -> dict[str, Any] | None:
    laps = parse_lap_times_sec(lap_times)
    metrics = compute_lap_metrics(laps)
    if not metrics:
        return None

    n = metrics["n_furlongs"]
    th = resolve_thresholds(
        surface, distance, metrics.get("baseline_lap"), n_furlongs=n
    )

    burst_delta = metrics.get("burst_delta")
    consume_delta = metrics.get("consume_delta")
    last5_std = metrics.get("last5_std")
    last5_range = metrics.get("last5_range")

    code = "standard"
    if (
        n >= 5
        and burst_delta is not None
        and burst_delta >= th.burst_delta_min
    ):
        code = "slow_burst"
    elif consume_delta is not None and consume_delta >= th.consume_delta_min:
        code = "consuming"
    elif (
        n >= 5
        and last5_std is not None
        and last5_range is not None
        and last5_std <= th.sustain_std_max
        and last5_range <= th.sustain_range_max
    ):
        code = "sustained"

    meta = _PATTERN_META[code]
    detail_parts: list[str] = []
    if metrics.get("l54_avg_sec") is not None and metrics.get("l21_avg_sec") is not None:
        detail_parts.append(
            f"L5-4F平均 {metrics['l54_avg_sec']:.1f}秒 / L2-1F平均 {metrics['l21_avg_sec']:.1f}秒"
            + (f"（差 {burst_delta:+.2f}秒）" if burst_delta is not None else "")
        )
    if metrics.get("l34_avg_sec") is not None:
        detail_parts.append(
            f"L3-4F平均 {metrics['l34_avg_sec']:.1f}秒 / L1F {metrics['l1_sec']:.1f}秒"
            + (f"（差 {consume_delta:+.2f}秒）" if consume_delta is not None else "")
        )
    if n >= 5 and last5_std is not None and last5_range is not None:
        detail_parts.append(f"後半5F 標準偏差 {last5_std:.2f}秒・幅 {last5_range:.2f}秒")

    return {
        "code": code,
        "label": meta["label"],
        "summary": meta["summary"],
        "detail": " / ".join(detail_parts),
        "thresholds": th.as_dict(),
        "metrics": {
            "n_furlongs": n,
            "l1_sec": round(metrics["l1_sec"], 2),
            "l34_avg_sec": round(metrics["l34_avg_sec"], 2) if metrics.get("l34_avg_sec") is not None else None,
            "l21_avg_sec": round(metrics["l21_avg_sec"], 2) if metrics.get("l21_avg_sec") is not None else None,
            "l54_avg_sec": round(metrics["l54_avg_sec"], 2) if metrics.get("l54_avg_sec") is not None else None,
            "burst_delta_sec": round(burst_delta, 2) if burst_delta is not None else None,
            "consume_delta_sec": round(consume_delta, 2) if consume_delta is not None else None,
            "last5_std_sec": round(last5_std, 3) if last5_std is not None else None,
            "last5_range_sec": round(last5_range, 2) if last5_range is not None else None,
        },
    }


def attach_lap_pattern_to_result(result: dict[str, Any]) -> None:
    laps = parse_lap_times_sec(result.get("lap_times"))
    pattern = classify_race_lap_pattern(
        result.get("lap_times"),
        surface=result.get("surface"),
        distance=distance_m(result.get("distance"), len(laps)) or 1600,
    )
    if pattern:
        result["lap_pattern"] = pattern
        pace = dict(result.get("pace") or {})
        pace["lap_pattern"] = pattern
        result["pace"] = pace
