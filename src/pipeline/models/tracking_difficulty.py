"""
追走難度 (Tracking Difficulty) モジュール

「今回いつもと同じ位置を取るのにどれくらい楽か」「脚がどれだけたまるか」を数値化する。

== コンセプト ==
各馬の前走・前々走の通過順位割合を基礎データとし、
今回のペース・位置取り予測と組み合わせて「追走難度指数（スタミナ温存度）」を算出する。

== 分析データ ==
1. 前走位置取り割合（N番手/出走頭数）と分布ビン
2. 全馬の前走T1F（1コーナー通過順）データ（頭数・コース・前々走3番手以内馬数による正規化）
3. 前走との出走コース比較（馬場タイプ・距離変化・枠番推移）
4. ペースデータ（前半1F/3F予測・ペース圧力指数）
5. 各馬の追走難度指数（脚がたまる度 0-100）

== 特徴量設計 ==
1. 馬の脚質プロファイル: 直近走平均初角通過順位・安定性
2. 前走T1Fデータ: 前走1角通過順 / 前走頭数（前々走3番手以内馬数で正規化）
3. コース比較: 馬場タイプ一致・距離変化カテゴリ・枠番変化
4. ゲート要因: 馬番・枠番偶奇・ゾーン
5. 隣枠影響: 左右隣馬の傾向・スペーススコア
6. 場全体構成: 同脚質馬数・ペース圧力・先行馬比率

== ターゲット変数 ==
stamina_index = 脚がたまる度（100 = フル温存、0 = 消耗）
position_deviation = (typical_norm_pos - actual_norm_pos)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline.models.tracking_difficulty")


# ── 定数 ──

RUNNING_STYLE_THRESHOLDS = {
    "逃げ": 0.15,
    "先行": 0.35,
    "差し": 0.65,
    "追込": 1.0,
}

_STYLE_CODES = list(RUNNING_STYLE_THRESHOLDS.keys())

EXPERIMENT_NAME = "tracking-difficulty"
MODEL_NAME = "tracking-difficulty-lgbm"


# ── ユーティリティ ──

def _parse_passing(passing_order: str, field_size: int = 18) -> list[int]:
    """
    通過順をコーナー順位リストにパースする。

    ``3-4-4-2`` 形式のほか、ハイフン無し ``1055``（10-5-5）も field_size に合わせて復元する。
    """
    s = str(passing_order or "").strip()
    if not s or s in ("**", "-", "―", "—"):
        return []
    fs = max(int(field_size or 0), 2)

    if "-" in s:
        out: list[int] = []
        for part in s.split("-"):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return out

    if not re.fullmatch(r"\d+", s):
        return [int(x) for x in re.findall(r"\d+", s) if int(x) <= fs]

    out = []
    i = 0
    while i < len(s):
        if i + 1 < len(s):
            two = int(s[i : i + 2])
            if 10 <= two <= fs:
                out.append(two)
                i += 2
                continue
        one = int(s[i])
        if one <= fs:
            out.append(one)
        i += 1
    return out


def _norm_pos(position: float, field_size: int) -> float:
    if field_size <= 1:
        return 0.5
    return (position - 1) / (field_size - 1)


def _classify_style(norm_pos: float) -> str:
    for style, threshold in RUNNING_STYLE_THRESHOLDS.items():
        if norm_pos <= threshold:
            return style
    return "追込"


def _style_to_code(style: str) -> int:
    try:
        return _STYLE_CODES.index(style)
    except ValueError:
        return _STYLE_CODES.index("差し")


def _normalize_surface(surface: str) -> str:
    s = str(surface or "").strip()
    if s in ("ダ", "ダート"):
        return "ダート"
    if s in ("芝",):
        return "芝"
    return s or "芝"


def _distance_band(distance: int) -> int:
    d = int(distance or 0)
    if d <= 1400:
        return 0
    if d <= 1800:
        return 1
    if d <= 2200:
        return 2
    return 3


def _race_passing_metrics(passing: list[int], field_size: int) -> dict[str, Any]:
    """1走分の通過順から位置・脚質指標を抽出。"""
    fs = max(int(field_size or 0), 2)
    empty = {
        "t1f_raw": 0,
        "t1f_norm": 0.5,
        "final_raw": 0,
        "final_norm": 0.5,
        "best_norm": 0.5,
        "best_raw": 0,
        "corner_spread": 0.0,
        "ever_led_any": 0,
        "ever_led_before_final": 0,
        "style_jra": "差し",
        "style_final": "差し",
        "n_corners": 0,
    }
    if not passing or fs < 4:
        return empty

    norms = [_norm_pos(p, fs) for p in passing]
    t1f = int(passing[0])
    final = int(passing[-1])
    best_raw = min(passing)
    best_norm = min(norms)
    spread = round(_norm_pos(final, fs) - _norm_pos(t1f, fs), 4)
    ever_any = int(any(p == 1 for p in passing))
    ever_before_final = int(len(passing) >= 2 and any(p == 1 for p in passing[:-1]))

    style_jra = _classify_style_jra(passing, fs)
    style_final = _classify_style_final(passing, fs)

    return {
        "t1f_raw": t1f,
        "t1f_norm": round(_norm_pos(t1f, fs), 4),
        "final_raw": final,
        "final_norm": round(_norm_pos(final, fs), 4),
        "best_norm": round(best_norm, 4),
        "best_raw": best_raw,
        "corner_spread": spread,
        "ever_led_any": ever_any,
        "ever_led_before_final": ever_before_final,
        "style_jra": style_jra,
        "style_final": style_final,
        "n_corners": len(passing),
    }


def _classify_style_jra(passing: list[int], field_size: int) -> str:
    """JRA-VAN NEXT 系: 最終角以外で1位→逃げ、最終角4位以内→先行 等。"""
    fs = max(int(field_size or 0), 4)
    if not passing:
        return "差し"
    if len(passing) >= 2 and any(p == 1 for p in passing[:-1]):
        return "逃げ"
    if passing[-1] <= 4:
        return "先行"
    if fs >= 8 and passing[-1] <= fs * 2 / 3:
        return "差し"
    return "追込"


def _classify_style_final(passing: list[int], field_size: int) -> str:
    """最終コーナー通過順の割合で脚質分類。"""
    fs = max(int(field_size or 0), 2)
    if not passing:
        return "差し"
    return _classify_style(_norm_pos(passing[-1], fs))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights:
        return 0.5
    w_sum = sum(weights[: len(values)])
    if w_sum <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / w_sum


def _metrics_from_history_row(rec: dict) -> dict[str, Any] | None:
    fs = int(rec.get("field_size") or 0)
    passing = _parse_passing(rec.get("passing_order", ""), fs)
    if not passing or fs < 4:
        return None
    m = _race_passing_metrics(passing, fs)
    m["surface"] = _normalize_surface(rec.get("surface", ""))
    m["distance_band"] = _distance_band(int(rec.get("distance") or 0))
    m["jockey_id"] = str(rec.get("jockey_id") or "")
    return m


def _compute_jockey_position_bias(
    race_history: list[dict],
    current_jockey_id: str,
    horse_typical_norm: float,
    *,
    max_races: int = 24,
) -> float:
    """
    騎手の位置取りバイアス（負=前に乗る、正=後ろに乗る）。
    当該騎手で騎乗した走の初角 norm と馬の typical の差。
    """
    jid = str(current_jockey_id or "").strip()
    if not jid:
        return 0.0
    norms: list[float] = []
    weights: list[float] = []
    for i, rec in enumerate(race_history[: max_races * 2]):
        if str(rec.get("jockey_id") or "") != jid:
            continue
        m = _metrics_from_history_row(rec)
        if not m:
            continue
        norms.append(float(m["t1f_norm"]))
        weights.append(1.0 / (i + 1))
    if not norms:
        return 0.0
    jockey_avg = _weighted_mean(norms, weights)
    return round(jockey_avg - horse_typical_norm, 4)


def _aggregate_passing_history(
    race_history: list[dict],
    *,
    max_races: int = 10,
    surface: str = "",
    distance: int = 0,
) -> dict[str, Any]:
    """直近走の通過順指標を集計（全コーナー・ハナ経験・4角脚質）。"""
    surface_n = _normalize_surface(surface) if surface else ""
    dist_band = _distance_band(distance) if distance else -1

    t1f_norms: list[float] = []
    best_norms: list[float] = []
    final_norms: list[float] = []
    spreads: list[float] = []
    ever_any: list[int] = []
    ever_bf: list[int] = []
    cond_surface: list[float] = []
    cond_distance: list[float] = []
    weights: list[float] = []
    n_jra_nige = 0
    n_samples = 0
    recent_jra_style = "差し"

    for i, rec in enumerate(race_history[: max_races * 2]):
        m = _metrics_from_history_row(rec)
        if not m:
            continue
        if n_samples == 0:
            recent_jra_style = m["style_jra"]
        w = 1.0 / (i + 1)
        weights.append(w)
        t1f_norms.append(float(m["t1f_norm"]))
        best_norms.append(float(m["best_norm"]))
        final_norms.append(float(m["final_norm"]))
        spreads.append(float(m["corner_spread"]))
        ever_any.append(int(m["ever_led_any"]))
        ever_bf.append(int(m["ever_led_before_final"]))
        if m["style_jra"] == "逃げ":
            n_jra_nige += 1
        n_samples += 1
        if surface_n and m["surface"] == surface_n:
            cond_surface.append(float(m["t1f_norm"]))
        if dist_band >= 0 and m["distance_band"] == dist_band:
            cond_distance.append(float(m["t1f_norm"]))

    if not t1f_norms:
        return {
            "typical_norm_pos": 0.5,
            "best_norm_pos_avg": 0.5,
            "best_norm_pos_min": 0.5,
            "final_norm_pos_avg": 0.5,
            "corner_spread_avg": 0.0,
            "ever_led_rate": 0.0,
            "ever_led_before_final_rate": 0.0,
            "ever_led_any": 0,
            "ever_led_before_final": 0,
            "n_races": 0,
            "positions": [],
            "best_norms": [],
            "style": "差し",
            "style_jra": "差し",
            "style_final": "差し",
            "pos_std": 0.2,
            "start_tendency": 0.5,
            "nige_rate": 0.0,
            "same_surface_typical_norm": 0.5,
            "distance_band_typical_norm": 0.5,
        }

    w_sum = sum(weights)
    typical = _weighted_mean(t1f_norms, weights)
    best_avg = _weighted_mean(best_norms, weights)
    best_min = min(best_norms)
    final_avg = _weighted_mean(final_norms, weights)
    spread_avg = _weighted_mean(spreads, weights)
    ever_rate = sum(ever_any) / len(ever_any)
    ever_bf_rate = sum(ever_bf) / len(ever_bf)

    same_surf = (
        sum(cond_surface) / len(cond_surface) if cond_surface else typical
    )
    dist_typical = (
        sum(cond_distance) / len(cond_distance) if cond_distance else typical
    )

    return {
        "typical_norm_pos": round(typical, 4),
        "best_norm_pos_avg": round(best_avg, 4),
        "best_norm_pos_min": round(best_min, 4),
        "final_norm_pos_avg": round(final_avg, 4),
        "corner_spread_avg": round(spread_avg, 4),
        "ever_led_rate": round(ever_rate, 4),
        "ever_led_before_final_rate": round(ever_bf_rate, 4),
        "ever_led_any": int(any(ever_any)),
        "ever_led_before_final": int(any(ever_bf)),
        "n_races": n_samples,
        "positions": [round(p, 3) for p in t1f_norms[:5]],
        "best_norms": [round(p, 3) for p in best_norms[:5]],
        "style": _classify_style(typical),
        "style_jra": recent_jra_style,
        "style_final": _classify_style(final_avg),
        "pos_std": round(_safe_std(t1f_norms), 4),
        "start_tendency": round(typical, 4),
        "nige_rate": round(n_jra_nige / max(n_samples, 1), 4),
        "same_surface_typical_norm": round(same_surf, 4),
        "distance_band_typical_norm": round(dist_typical, 4),
    }


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _date_to_yyyymmdd(raw: str) -> str:
    """YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD → YYYYMMDD。"""
    s = str(raw or "").strip()
    if not s:
        return ""
    if len(s) >= 8 and s[:8].isdigit():
        return s[:8]
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}"
    return ""


def _filter_history_before_race(
    race_history: list[dict],
    *,
    before_date: str = "",
    exclude_race_id: str = "",
) -> list[dict]:
    """分析対象レースの開催日・race_id より前の戦績のみ（新しい順想定）。"""
    cutoff = _date_to_yyyymmdd(before_date)
    ex = (exclude_race_id or "").strip()
    out: list[dict] = []
    for rec in race_history or []:
        rid = str(rec.get("race_id") or "").strip()
        if ex and rid == ex:
            continue
        rdate = _date_to_yyyymmdd(str(rec.get("date") or ""))
        if cutoff and rdate and rdate >= cutoff:
            continue
        out.append(rec)
    return out


def _row_has_t1f_data(rec: dict) -> bool:
    fs = int(rec.get("field_size") or 0)
    passing = _parse_passing(rec.get("passing_order", ""), fs)
    return bool(passing) and fs >= 4


def _enrich_history_row_from_race_storage(
    rec: dict,
    horse_id: str,
    storage,
) -> dict:
    """
    horse_result 戦績行に通過順・上り3F が無い場合、race_result / race_shutuba から補完する。
    GCS・ローカルキャッシュの古い horse_result でもレースJSONがあれば埋められる。
    """
    if storage is None or not horse_id:
        return rec
    rid = str(rec.get("race_id") or "").strip()
    if not rid:
        return rec

    merged = dict(rec)
    fs_hint = int(merged.get("field_size") or 18)
    need_passing = not _parse_passing(merged.get("passing_order", ""), fs_hint)
    need_l3f = _safe_float(merged.get("last_3f")) <= 0
    if not need_passing and not need_l3f:
        return merged

    for storage_key in ("race_result", "race_shutuba"):
        rd = storage.load(storage_key, rid)
        if not rd:
            continue
        fs = int(rd.get("field_size") or merged.get("field_size") or 0)
        for ent in rd.get("entries") or []:
            if str(ent.get("horse_id")) != str(horse_id):
                continue
            if need_passing and ent.get("passing_order"):
                merged["passing_order"] = ent.get("passing_order")
                need_passing = not _parse_passing(
                    merged.get("passing_order", ""), fs or fs_hint
                )
            if need_l3f and _safe_float(ent.get("last_3f")) > 0:
                merged["last_3f"] = ent.get("last_3f")
                need_l3f = False
            if fs > 0:
                merged["field_size"] = fs
            break
        if not need_passing and not need_l3f:
            break
    return merged


def _pick_prev_and_prev_prev(
    race_history: list[dict],
    *,
    before_date: str = "",
    exclude_race_id: str = "",
    horse_id: str = "",
    storage=None,
) -> tuple[dict | None, dict | None]:
    """対象レースより前で、通過順が取れる最も新しい走＋その次走を返す。"""
    filtered = _filter_history_before_race(
        race_history,
        before_date=before_date,
        exclude_race_id=exclude_race_id,
    )
    prev: dict | None = None
    prev_prev: dict | None = None
    for i, raw in enumerate(filtered[:12]):
        rec = _enrich_history_row_from_race_storage(raw, horse_id, storage)
        if not _row_has_t1f_data(rec):
            continue
        if prev is None:
            prev = rec
            continue
        prev_prev = rec
        break
    return prev, prev_prev


# ── 前走データ特徴量 ──

def extract_prev_race_features(
    race_history: list[dict],
    *,
    before_date: str = "",
    exclude_race_id: str = "",
    horse_id: str = "",
    storage=None,
) -> dict:
    """
    前走・前々走の特徴量を抽出する。

    T1F = 前走1コーナー通過順位（初角ポジション）。
    正規化は build_field_prev_stats() で行う（全馬の前走top3馬数が必要）。
    分析対象レース当日以降の戦績は使わない（before_date / exclude_race_id）。
    """
    prev, prev_prev = _pick_prev_and_prev_prev(
        race_history,
        before_date=before_date,
        exclude_race_id=exclude_race_id,
        horse_id=horse_id,
        storage=storage,
    )
    if not prev:
        return _empty_prev_features()

    prev_fs = int(prev.get("field_size") or 0)
    prev_passing = _parse_passing(prev.get("passing_order", ""), prev_fs)
    if not prev_passing or prev_fs < 4:
        return _empty_prev_features()

    prev_metrics = _race_passing_metrics(prev_passing, prev_fs)
    prev_t1f_raw = prev_metrics["t1f_raw"]
    prev_pos_ratio = prev_t1f_raw / prev_fs

    prev_prev_was_top3 = 0
    prev_prev_passing: list[int] = []
    prev_prev_fs = 0
    if prev_prev:
        prev_prev_fs = int(prev_prev.get("field_size") or 0)
        prev_prev_passing = _parse_passing(
            prev_prev.get("passing_order", ""), prev_prev_fs
        )
        if prev_prev_passing and prev_prev_fs >= 4:
            prev_prev_was_top3 = 1 if prev_prev_passing[0] <= 3 else 0

    return {
        "prev_t1f_raw": prev_t1f_raw,
        "prev_pos_ratio": round(prev_pos_ratio, 4),
        "prev_field_size": prev_fs,
        "prev_surface": str(prev.get("surface") or ""),
        "prev_distance": int(prev.get("distance") or 0),
        "prev_bracket": int(prev.get("bracket_number") or prev.get("horse_number") or 0),
        "prev_horse_number": int(prev.get("horse_number") or 0),
        "prev_track_condition": str(prev.get("track_condition") or ""),
        "prev_last3f": _safe_float(prev.get("last_3f")),
        "prev_finish_pos": int(prev.get("finish_position") or 0),
        "prev_passing_order": str(prev.get("passing_order") or ""),
        "prev_style": _classify_style(_norm_pos(prev_t1f_raw, prev_fs)),
        "prev_style_jra": prev_metrics["style_jra"],
        "prev_style_final": prev_metrics["style_final"],
        "prev_best_norm": prev_metrics["best_norm"],
        "prev_final_norm": prev_metrics["final_norm"],
        "prev_corner_spread": prev_metrics["corner_spread"],
        "prev_ever_led_any": prev_metrics["ever_led_any"],
        "prev_ever_led_before_final": prev_metrics["ever_led_before_final"],
        "prev_prev_was_top3": prev_prev_was_top3,
        "prev_prev_t1f_raw": prev_prev_passing[0] if prev_prev_passing else 0,
        "prev_prev_field_size": prev_prev_fs,
    }


def _empty_prev_features() -> dict:
    return {
        "prev_t1f_raw": 0,
        "prev_pos_ratio": 0.5,
        "prev_field_size": 0,
        "prev_surface": "",
        "prev_distance": 0,
        "prev_bracket": 0,
        "prev_horse_number": 0,
        "prev_track_condition": "",
        "prev_last3f": 0.0,
        "prev_finish_pos": 0,
        "prev_passing_order": "",
        "prev_style": "",
        "prev_style_jra": "",
        "prev_style_final": "",
        "prev_best_norm": 0.5,
        "prev_final_norm": 0.5,
        "prev_corner_spread": 0.0,
        "prev_ever_led_any": 0,
        "prev_ever_led_before_final": 0,
        "prev_prev_was_top3": 0,
        "prev_prev_t1f_raw": 0,
        "prev_prev_field_size": 0,
    }


def _empty_recent3_features() -> dict:
    return {
        "n_races": 0,
        "t1f_raw_avg": 0.0,
        "pos_ratio_avg": 0.5,
        "field_size_avg": 0.0,
        "style": "",
        "typical_norm_pos": 0.5,
        "best_norm_pos_avg": 0.5,
        "best_norm_pos_min": 0.5,
        "ever_led_rate": 0.0,
        "t1f_raws": [],
        "pos_ratios": [],
        "best_norms": [],
    }


def _collect_form_race_rows(
    race_history: list[dict],
    *,
    before_date: str = "",
    exclude_race_id: str = "",
    horse_id: str = "",
    storage=None,
    max_rows: int = 3,
) -> list[dict]:
    """対象レースより前で通過順が取れる走を新しい順に最大 max_rows 件。"""
    filtered = _filter_history_before_race(
        race_history,
        before_date=before_date,
        exclude_race_id=exclude_race_id,
    )
    rows: list[dict] = []
    for raw in filtered[: max_rows * 3]:
        rec = _enrich_history_row_from_race_storage(raw, horse_id, storage)
        if _row_has_t1f_data(rec):
            rows.append(rec)
        if len(rows) >= max_rows:
            break
    return rows


def extract_recent3_features(
    race_history: list[dict],
    *,
    before_date: str = "",
    exclude_race_id: str = "",
    horse_id: str = "",
    storage=None,
) -> dict:
    """
    近3走（分析対象より前）の1角・位置割合・脚質を加重平均で集計する。

    新しい走ほど重み大（1, 1/2, 1/3）。
    """
    rows = _collect_form_race_rows(
        race_history,
        before_date=before_date,
        exclude_race_id=exclude_race_id,
        horse_id=horse_id,
        storage=storage,
        max_rows=3,
    )
    if not rows:
        return _empty_recent3_features()

    t1f_raws: list[int] = []
    pos_ratios: list[float] = []
    norm_positions: list[float] = []
    best_norms: list[float] = []
    field_sizes: list[int] = []
    weights: list[float] = []
    ever_any: list[int] = []

    for i, rec in enumerate(rows):
        fs = int(rec.get("field_size") or 0)
        passing = _parse_passing(rec.get("passing_order", ""), fs)
        if not passing or fs < 4:
            continue
        m = _race_passing_metrics(passing, fs)
        t1f_raws.append(m["t1f_raw"])
        pos_ratios.append(m["t1f_raw"] / fs)
        norm_positions.append(m["t1f_norm"])
        best_norms.append(m["best_norm"])
        ever_any.append(m["ever_led_any"])
        field_sizes.append(fs)
        weights.append(1.0 / (i + 1))

    if not norm_positions:
        return _empty_recent3_features()

    avg_norm = _weighted_mean(norm_positions, weights)
    avg_pos = _weighted_mean(pos_ratios, weights)
    avg_t1f = _weighted_mean([float(t) for t in t1f_raws], weights)
    avg_fs = _weighted_mean([float(f) for f in field_sizes], weights)
    avg_best = _weighted_mean(best_norms, weights)

    return {
        "n_races": len(norm_positions),
        "t1f_raw_avg": round(avg_t1f, 1),
        "pos_ratio_avg": round(avg_pos, 4),
        "field_size_avg": round(avg_fs, 1),
        "style": _classify_style(avg_norm),
        "typical_norm_pos": round(avg_norm, 4),
        "best_norm_pos_avg": round(avg_best, 4),
        "best_norm_pos_min": round(min(best_norms), 4),
        "ever_led_rate": round(sum(ever_any) / len(ever_any), 4),
        "t1f_raws": t1f_raws,
        "pos_ratios": [round(p, 4) for p in pos_ratios],
        "best_norms": [round(p, 4) for p in best_norms],
    }


def blend_form_style(style_prev: str, style_recent3: str, *, n_recent: int) -> str:
    """前走脚質と近3走脚質の合成（近3走の方をやや重視）。"""
    if not style_prev and not style_recent3:
        return "差し"
    if not style_recent3 or n_recent <= 0:
        return style_prev or "差し"
    if not style_prev:
        return style_recent3
    if style_prev == style_recent3:
        return style_prev
    # 脚質が食い違うときは近3走（トレンド）を優先しつつ前走をサブ表示用に残す
    return style_recent3


def build_field_prev_stats(
    all_prev_features: dict[int, dict],
    current_entries: list[dict],
    current_distance: int,
    current_surface: str,
) -> dict:
    """
    フィールド全体の前走統計を計算する。

    - 位置取り割合のビン化
    - 前走T1F正規化（前々走で3番手以内だった馬の頭数を使用）
    - コース比較統計

    Args:
        all_prev_features: horse_number → prev_features のマップ
        current_entries: 今回の出走エントリ
        current_distance: 今回の距離
        current_surface: 今回の馬場タイプ
    """
    field_size = len(current_entries)
    if field_size == 0:
        return {}

    pos_ratios = []
    t1f_raws = []
    prev_top3_count = 0   # 前走で3番手以内（前走pos_ratio <= 3/prev_fs）

    for e in current_entries:
        hn = e.get("horse_number", 0)
        pf = all_prev_features.get(hn, _empty_prev_features())
        pr = pf["prev_pos_ratio"]
        t1f_raw = pf["prev_t1f_raw"]

        pos_ratios.append(pr)
        if t1f_raw > 0:
            t1f_raws.append(t1f_raw)

        # 前走で1コーナー3番手以内を走っていた馬
        if pf["prev_field_size"] > 0 and t1f_raw > 0 and t1f_raw <= 3:
            prev_top3_count += 1

    # ── 位置取り割合ビン化（5区分）──
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["先頭圏 (0-20%)", "前目 (20-40%)", "中団 (40-60%)", "後方 (60-80%)", "最後方 (80-100%)"]
    bins: list[dict] = []
    for i, label in enumerate(bin_labels):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        horses_in_bin = [
            e.get("horse_number", 0)
            for e in current_entries
            if lo <= all_prev_features.get(e.get("horse_number", 0), _empty_prev_features())["prev_pos_ratio"] < hi
        ]
        bins.append({
            "label": label,
            "range": [lo, hi],
            "count": len(horses_in_bin),
            "horse_numbers": horses_in_bin,
        })

    # ── T1F正規化係数 ──
    # 前走で3番手以内だった馬の割合が高いほど、今回の前半争いが激化
    front_pressure_ratio = prev_top3_count / max(field_size, 1)
    # 正規化係数: front_pressure_ratio が高い = 各馬のT1F値の価値が下がる（競争が激しい）
    t1f_normalization_factor = 1.0 + front_pressure_ratio  # 1.0 ~ 2.0

    # T1F正規化値を各馬に適用
    normalized_t1f: dict[int, float] = {}
    for e in current_entries:
        hn = e.get("horse_number", 0)
        pf = all_prev_features.get(hn, _empty_prev_features())
        t1f_raw = pf["prev_t1f_raw"]
        prev_fs = pf["prev_field_size"]
        if t1f_raw > 0 and prev_fs > 0:
            # 前走T1F正規化 = (T1F順位/前走頭数) × 正規化係数
            # 値が小さいほど「前にいた馬」（前走前目＋今回も競争が少ない）
            normalized_t1f[hn] = round(
                (t1f_raw / prev_fs) * (1.0 + pf["prev_prev_was_top3"] * 0.15) / t1f_normalization_factor,
                4,
            )
        else:
            normalized_t1f[hn] = 0.5

    # ── コース比較統計 ──
    same_surface_count = sum(
        1 for e in current_entries
        if all_prev_features.get(e.get("horse_number", 0), _empty_prev_features())["prev_surface"] in (current_surface, "")
    )
    dist_changes = []
    ever_led_prev = 0
    ever_led_bf_prev = 0
    for e in current_entries:
        hn = e.get("horse_number", 0)
        pf = all_prev_features.get(hn, _empty_prev_features())
        if pf["prev_distance"] > 0 and current_distance > 0:
            dist_changes.append(current_distance - pf["prev_distance"])
        if pf.get("prev_ever_led_any"):
            ever_led_prev += 1
        if pf.get("prev_ever_led_before_final"):
            ever_led_bf_prev += 1

    return {
        "position_bins": bins,
        "prev_top3_count": prev_top3_count,
        "ever_led_prev_count": ever_led_prev,
        "ever_led_before_final_prev_count": ever_led_bf_prev,
        "front_pressure_ratio": round(front_pressure_ratio, 3),
        "t1f_normalization_factor": round(t1f_normalization_factor, 3),
        "normalized_t1f": normalized_t1f,
        "same_surface_count": same_surface_count,
        "avg_dist_change": round(sum(dist_changes) / len(dist_changes), 0) if dist_changes else 0,
        "dist_extension_count": sum(1 for d in dist_changes if d > 50),
        "dist_reduction_count": sum(1 for d in dist_changes if d < -50),
        "avg_prev_pos_ratio": round(sum(pos_ratios) / len(pos_ratios), 3) if pos_ratios else 0.5,
    }


def build_course_comparison(
    entry: dict,
    prev_features: dict,
    current_distance: int,
    current_surface: str,
) -> dict:
    """各馬の前走コースとの比較情報を構築する。"""
    hn = entry.get("horse_number", 0)
    current_bracket = entry.get("bracket_number", 0)
    prev_bracket = prev_features.get("prev_bracket", 0)
    prev_hn = prev_features.get("prev_horse_number", 0)
    prev_dist = prev_features.get("prev_distance", 0)
    prev_surf = prev_features.get("prev_surface", "")

    dist_change = (current_distance - prev_dist) if prev_dist > 0 and current_distance > 0 else 0
    if abs(dist_change) <= 50:
        dist_category = "同距離"
    elif dist_change > 50:
        dist_category = "距離延長"
    else:
        dist_category = "距離短縮"

    gate_change = (hn - prev_hn) if prev_hn > 0 else 0

    same_surface = False
    if prev_surf:
        surf_norm = {"ダ": "ダート", "芝": "芝"}.get(prev_surf, prev_surf)
        curr_norm = {"ダ": "ダート", "芝": "芝"}.get(current_surface, current_surface)
        same_surface = surf_norm == curr_norm

    return {
        "same_surface": same_surface,
        "prev_surface": prev_surf,
        "current_surface": current_surface,
        "prev_distance": prev_dist,
        "current_distance": current_distance,
        "dist_change": dist_change,
        "dist_category": dist_category,
        "gate_change": gate_change,
        "prev_horse_number": prev_hn,
        "current_horse_number": hn,
    }


# ── 追走難度指数（スタミナ温存度）──

def compute_stamina_index(
    predicted_position_norm: float,
    pace_index: float,
    pace_type: str,
    distance: int,
    surface: str,
    t1f_norm: float,
    profile: dict,
) -> dict:
    """
    追走難度指数（脚がたまる度）を計算する。

    脚がたまる = スタミナが温存される = 終盤に使える脚が多く残る

    理論:
    - 後方位置 × スローペース → 最大温存（差し・追込には不利）
    - 前方位置 × スローペース → 消耗は少ないが位置取りの意味が薄い
    - 前方位置 × ハイペース  → 最大消耗（逃げ・先行には苦しい）
    - 後方位置 × ハイペース  → 温存はできるが届きにくい

    Returns:
        {
            "stamina_index": 0-100 (高いほど脚がたまる),
            "stamina_label": str,
            "pace_load": float,   # ペース負荷
            "position_load": float,  # 位置取り負荷
            "distance_load": float,  # 距離負荷
        }
    """
    style = profile.get("style", "差し")
    pos_std = profile.get("pos_std", 0.2)

    # 1) ペース負荷（ハイペース = 負荷大）
    pace_load = (pace_index - 50) / 100  # -0.5 ~ +0.5
    pace_load = max(-0.5, min(0.5, pace_load))

    # 2) 位置取り負荷
    # 前方にいるほど負荷が大きいが、ペースとの組み合わせが重要
    # pos_norm: 0=1番手, 1=最後尾
    pos_front = 1.0 - predicted_position_norm  # 0=最後尾, 1=1番手
    position_load = pos_front * (1.0 + pace_load)  # ハイペースで前にいるほど負荷増

    # 3) 距離負荷（長距離ほどペース・位置取りの影響が大きい）
    if distance <= 1400:
        dist_factor = 0.8   # 短距離：消耗が短時間なので少し緩和
    elif distance <= 1800:
        dist_factor = 1.0
    elif distance <= 2200:
        dist_factor = 1.15
    else:
        dist_factor = 1.3

    # 4) ダートは芝より消耗が大きい
    surface_factor = 1.1 if surface in ("ダ", "ダート") else 1.0

    # 5) T1F正規化値による前半争い強度
    # t1f_norm が低い = 前に行きたい馬が多い = ペース争い激化
    t1f_pressure = (0.5 - min(t1f_norm, 1.0)) * 0.3  # -0.3 ~ +0.15

    # 総合消耗スコア（0=全消耗, 1=全温存）
    raw_load = (
        position_load * 0.4
        + pace_load * 0.3
        + t1f_pressure * 0.2
        + (pos_std - 0.2) * 0.1  # 不安定な脚質は追走で無駄な動きが多い
    )
    raw_load *= dist_factor * surface_factor

    # スタミナ温存度 = 100 - 消耗率×100
    stamina_index = max(0.0, min(100.0, 50.0 - raw_load * 100))

    if stamina_index >= 75:
        label = "最大温存"
    elif stamina_index >= 60:
        label = "温存"
    elif stamina_index >= 45:
        label = "標準"
    elif stamina_index >= 30:
        label = "消耗"
    else:
        label = "激消耗"

    return {
        "stamina_index": round(stamina_index, 1),
        "stamina_label": label,
        "pace_load": round(pace_load, 3),
        "position_load": round(position_load, 3),
        "distance_load": round(dist_factor, 2),
        "t1f_pressure": round(t1f_pressure, 3),
    }


# ── 馬の脚質プロファイル ──

def build_horse_profile(
    race_history: list[dict],
    max_races: int = 10,
    *,
    surface: str = "",
    distance: int = 0,
    current_jockey_id: str = "",
) -> dict:
    """直近走から馬の脚質・位置取りプロファイルを構築する。"""
    agg = _aggregate_passing_history(
        race_history[:max_races],
        max_races=max_races,
        surface=surface,
        distance=distance,
    )
    agg["jockey_position_bias"] = _compute_jockey_position_bias(
        race_history,
        current_jockey_id,
        agg["typical_norm_pos"],
        max_races=max_races * 2,
    )
    agg["style_code"] = _style_to_code(agg["style"])
    agg["style_jra_code"] = _style_to_code(agg.get("style_jra", agg["style"]))
    agg["style_final_code"] = _style_to_code(agg.get("style_final", agg["style"]))
    return agg


# ── 隣馬（前走1角ベース）──

def _prev_corner_top3(prev_feat: dict) -> bool:
    """前走1コーナーが3番手以内か。"""
    t1f = int(prev_feat.get("prev_t1f_raw") or 0)
    return t1f > 0 and t1f <= 3


def _prev_corner_norm(prev_feat: dict) -> float | None:
    t1f = int(prev_feat.get("prev_t1f_raw") or 0)
    fs = int(prev_feat.get("prev_field_size") or 0)
    if t1f > 0 and fs >= 4:
        return _norm_pos(t1f, fs)
    return None


def build_neighbor_gate_factors(
    horse_number: int,
    field_size: int,
    profile: dict,
    left_profile: dict,
    right_profile: dict,
    left_prev: dict | None = None,
    right_prev: dict | None = None,
) -> dict:
    """
    隣馬の脚質傾向と前走1角（3番手以内）からゲート周りの影響を算出。

    両隣が前走3番手以内のとき「挟まれて前目を取りやすい」プッシュを付与する。
    """
    left_prev = left_prev or _empty_prev_features()
    right_prev = right_prev or _empty_prev_features()
    fs = max(int(field_size or 0), 1)
    hn = int(horse_number or 0)
    typical = float(profile.get("typical_norm_pos", 0.5))

    lt = float(left_profile.get("start_tendency", 0.5))
    rt = float(right_profile.get("start_tendency", 0.5))
    left_std = float(left_profile.get("pos_std", 0.2))
    right_std = float(right_profile.get("pos_std", 0.2))

    left_top3 = _prev_corner_top3(left_prev)
    right_top3 = _prev_corner_top3(right_prev)
    left_t1f = int(left_prev.get("prev_t1f_raw") or 0)
    right_t1f = int(right_prev.get("prev_t1f_raw") or 0)
    front_neighbor_count = (1 if left_top3 else 0) + (1 if right_top3 else 0)

    has_left = hn > 1
    has_right = hn < fs

    sandwich_front = bool(
        has_left and has_right and left_top3 and right_top3
    )

    # 前目隣による「前への押し出し」（position_norm を下げる＝前に行きやすい）
    sandwich_push = 0.0
    if sandwich_front:
        sandwich_push = 0.18
    elif has_left and has_right and front_neighbor_count == 1:
        sandwich_push = 0.09
    elif hn == 1 and has_right and right_top3:
        sandwich_push = 0.06
    elif hn == fs and has_left and left_top3:
        sandwich_push = 0.06

    # 前走ベースの隣平均位置（取れるときは傾向より優先）
    prev_norms = []
    if has_left:
        ln = _prev_corner_norm(left_prev)
        if ln is not None:
            prev_norms.append(ln)
    if has_right:
        rn = _prev_corner_norm(right_prev)
        if rn is not None:
            prev_norms.append(rn)
    neighbor_avg_prev = (
        round(sum(prev_norms) / len(prev_norms), 4) if prev_norms else None
    )

    neighbor_slow = (1 if lt > 0.5 else 0) + (1 if rt > 0.5 else 0)

    # 速い隣馬: 前走3番手以内を優先し、従来の脚質傾向(<=0.2)をフォールバック
    neighbor_fast_tendency = 0
    if hn == 1:
        if rt <= 0.2:
            neighbor_fast_tendency = 1
    elif hn == fs:
        if lt <= 0.2:
            neighbor_fast_tendency = 1
    else:
        if lt <= 0.2 and rt <= 0.2:
            neighbor_fast_tendency = 1

    neighbor_fast_prev = 0
    if hn == 1:
        neighbor_fast_prev = 1 if right_top3 else 0
    elif hn == fs:
        neighbor_fast_prev = 1 if left_top3 else 0
    else:
        neighbor_fast_prev = 1 if left_top3 and right_top3 else 0

    neighbor_fast = max(neighbor_fast_tendency, neighbor_fast_prev)

    neighbor_avg_tendency = (lt + rt) / 2
    if neighbor_avg_prev is not None:
        neighbor_avg_blend = neighbor_avg_prev * 0.65 + neighbor_avg_tendency * 0.35
    else:
        neighbor_avg_blend = neighbor_avg_tendency

    neighbor_space_score = (
        neighbor_avg_blend
        - typical
        + neighbor_fast * 0.1
        - neighbor_slow * 0.05
        + sandwich_push * 0.35
    )

    return {
        "left_neighbor_tendency": round(lt, 4),
        "right_neighbor_tendency": round(rt, 4),
        "left_neighbor_std": round(left_std, 4),
        "right_neighbor_std": round(right_std, 4),
        "neighbor_slow_count": neighbor_slow,
        "neighbor_fast_count": neighbor_fast,
        "neighbor_avg_tendency": round(neighbor_avg_tendency, 4),
        "neighbor_space_score": round(neighbor_space_score, 4),
        "left_prev_t1f": left_t1f,
        "right_prev_t1f": right_t1f,
        "left_prev_top3": left_top3,
        "right_prev_top3": right_top3,
        "front_neighbor_count": front_neighbor_count,
        "sandwich_front": sandwich_front,
        "sandwich_push": round(sandwich_push, 4),
        "neighbor_avg_prev_norm": neighbor_avg_prev,
    }


def _apply_sandwich_ease_adjustment(
    ease_pct: float,
    style: str,
    sandwich_push: float,
) -> float:
    """挟まれ前目プッシュ: 差し・追込は消耗、逃げ・先行はやや有利。"""
    if sandwich_push <= 0:
        return ease_pct
    scale = sandwich_push / 0.18
    if style in ("差し", "追込"):
        ease_pct -= 7.0 * scale
    elif style in ("逃げ", "先行"):
        ease_pct += 3.5 * scale
    else:
        ease_pct -= 2.0 * scale
    return max(0.0, min(100.0, ease_pct))


# ── 学習データ構築 ──

def build_training_row(
    entry: dict,
    race_info: dict,
    horse_profile: dict,
    neighbor_profiles: dict,
    field_profiles: list[dict],
    *,
    left_prev: dict | None = None,
    right_prev: dict | None = None,
) -> dict | None:
    """1頭1レース分の学習用レコードを構築する。"""
    fs = int(race_info.get("field_size") or 0)
    passing = _parse_passing(entry.get("passing_order", ""), fs)
    if not passing or fs < 4:
        return None

    actual_norm = _norm_pos(passing[0], fs)
    typical = horse_profile["typical_norm_pos"]
    deviation = typical - actual_norm

    hn = entry.get("horse_number", 0)
    bn = entry.get("bracket_number", 0)

    front_runners = sum(
        1 for p in field_profiles if p["style"] in ("逃げ", "先行")
    )
    closers = sum(
        1 for p in field_profiles if p["style"] in ("差し", "追込")
    )
    same_style_count = sum(
        1 for p in field_profiles if p["style"] == horse_profile["style"]
    )

    nf = build_neighbor_gate_factors(
        hn, fs, horse_profile,
        neighbor_profiles.get("left", {}),
        neighbor_profiles.get("right", {}),
        left_prev=left_prev,
        right_prev=right_prev,
    )

    distance = race_info.get("distance", 1600)
    surface = race_info.get("surface", "芝")
    track_cond = race_info.get("track_condition", "良")

    row = {
        "race_id": race_info.get("race_id", ""),
        "horse_number": hn,
        "horse_id": entry.get("horse_id", ""),

        # ── ターゲット ──
        "position_deviation": round(deviation, 5),
        "actual_norm_pos": round(actual_norm, 4),

        # ── 馬の脚質・位置取り履歴 ──
        **_profile_model_features(horse_profile),

        # ── ゲート要因 ──
        "horse_number_norm": round(hn / fs, 4) if fs > 0 else 0.5,
        "bracket_number": bn,
        "is_even_gate": int(hn % 2 == 0),
        "gate_zone": (
            0 if hn <= fs * 0.25 else
            1 if hn <= fs * 0.5 else
            2 if hn <= fs * 0.75 else 3
        ),

        # ── 隣枠影響 ──
        "left_neighbor_tendency": nf["left_neighbor_tendency"],
        "right_neighbor_tendency": nf["right_neighbor_tendency"],
        "left_neighbor_std": nf["left_neighbor_std"],
        "right_neighbor_std": nf["right_neighbor_std"],
        "neighbor_slow_count": nf["neighbor_slow_count"],
        "neighbor_fast_count": nf["neighbor_fast_count"],
        "neighbor_avg_tendency": nf["neighbor_avg_tendency"],
        "neighbor_space_score": nf["neighbor_space_score"],
        "left_prev_top3": int(nf["left_prev_top3"]),
        "right_prev_top3": int(nf["right_prev_top3"]),
        "front_neighbor_count": nf["front_neighbor_count"],
        "sandwich_front": int(nf["sandwich_front"]),
        "sandwich_push": nf["sandwich_push"],

        # ── 場全体の構成 ──
        "field_size": fs,
        "front_runner_count": front_runners,
        "closer_count": closers,
        "same_style_count": same_style_count,
        "front_runner_ratio": round(front_runners / fs, 4) if fs > 0 else 0,
        "pace_pressure": round(front_runners / max(closers, 1), 4),

        # ── コース要因 ──
        "surface_code": 0 if surface == "芝" else 1,
        "distance": distance,
        "distance_category": (
            0 if distance <= 1400 else
            1 if distance <= 1800 else
            2 if distance <= 2200 else 3
        ),
        "track_condition_code": (
            0 if track_cond == "良" else
            1 if track_cond == "稍" else
            2 if track_cond == "重" else 3
        ),
    }

    return row


def build_training_dataset(storage, years: list[str] | None = None) -> pd.DataFrame:
    """GCSの過去レースデータから学習データセットを構築する。"""
    from src.scraper.storage import HybridStorage

    if years is None:
        years = ["2023", "2024", "2025", "2026"]

    result_keys = storage.list_keys("race_result")
    logger.info("race_result キー数: %d", len(result_keys))

    all_horse_histories: dict[str, list[dict]] = {}
    rows: list[dict] = []
    processed = 0
    skipped = 0

    for key in sorted(result_keys):
        year = key[:4]
        if year not in years:
            continue

        result_data = storage.load("race_result", key)
        if not result_data or not result_data.get("entries"):
            skipped += 1
            continue

        race_info = {
            "race_id": result_data.get("race_id", key),
            "field_size": result_data.get("field_size", len(result_data["entries"])),
            "distance": result_data.get("distance", 0),
            "surface": result_data.get("surface", ""),
            "track_condition": result_data.get("track_condition", ""),
            "venue": result_data.get("venue", ""),
            "date": result_data.get("date", ""),
        }

        entries = result_data["entries"]
        fs = race_info["field_size"]
        if fs < 4:
            skipped += 1
            continue

        horse_profiles_for_race: dict[int, dict] = {}
        for e in entries:
            hid = e.get("horse_id", "")
            hn = e.get("horse_number", 0)
            history = all_horse_histories.get(hid, [])
            horse_profiles_for_race[hn] = build_horse_profile(
                history,
                surface=race_info.get("surface", ""),
                distance=int(race_info.get("distance") or 0),
                current_jockey_id=str(e.get("jockey_id") or ""),
            )

        field_profiles = list(horse_profiles_for_race.values())
        prev_for_race: dict[int, dict] = {}
        for e in entries:
            hid = e.get("horse_id", "")
            hn = e.get("horse_number", 0)
            if hid:
                prev_for_race[hn] = extract_prev_race_features(
                    all_horse_histories.get(hid, [])
                )

        for e in entries:
            hn = e.get("horse_number", 0)
            hid = e.get("horse_id", "")
            profile = horse_profiles_for_race.get(hn)
            if not profile or profile["n_races"] < 2:
                continue

            left_hn = hn - 1
            right_hn = hn + 1
            neighbors = {
                "left": horse_profiles_for_race.get(left_hn, {}),
                "right": horse_profiles_for_race.get(right_hn, {}),
            }

            row = build_training_row(
                e, race_info, profile, neighbors, field_profiles,
                left_prev=prev_for_race.get(left_hn, _empty_prev_features()),
                right_prev=prev_for_race.get(right_hn, _empty_prev_features()),
            )
            if row is not None:
                rows.append(row)

        for e in entries:
            hid = e.get("horse_id", "")
            if hid:
                if hid not in all_horse_histories:
                    all_horse_histories[hid] = []
                all_horse_histories[hid].insert(0, {
                    "passing_order": e.get("passing_order", ""),
                    "field_size": fs,
                    "finish_position": e.get("finish_position", 0),
                    "bracket_number": e.get("bracket_number", 0),
                    "horse_number": e.get("horse_number", 0),
                    "distance": race_info["distance"],
                    "surface": race_info["surface"],
                    "track_condition": race_info["track_condition"],
                    "jockey_id": e.get("jockey_id", ""),
                })

        processed += 1
        if processed % 500 == 0:
            logger.info("処理済み: %d レース, 行数: %d", processed, len(rows))

    logger.info(
        "データセット構築完了: %d レース処理, %d スキップ, %d 行",
        processed, skipped, len(rows),
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ── モデル学習 ──

def _profile_model_features(profile: dict) -> dict[str, float]:
    """LGBM 入力行用のプロファイル特徴量スライス。"""
    return {
        "typical_norm_pos": float(profile.get("typical_norm_pos", 0.5)),
        "style_code": float(profile.get("style_code", _style_to_code(profile.get("style", "差し")))),
        "style_jra_code": float(profile.get("style_jra_code", 1)),
        "style_final_code": float(profile.get("style_final_code", 2)),
        "pos_std": float(profile.get("pos_std", 0.2)),
        "n_past_races": float(profile.get("n_races", 0)),
        "best_norm_pos_avg": float(profile.get("best_norm_pos_avg", 0.5)),
        "best_norm_pos_min": float(profile.get("best_norm_pos_min", 0.5)),
        "final_norm_pos_avg": float(profile.get("final_norm_pos_avg", 0.5)),
        "corner_spread_avg": float(profile.get("corner_spread_avg", 0)),
        "ever_led_rate": float(profile.get("ever_led_rate", 0)),
        "ever_led_before_final_rate": float(profile.get("ever_led_before_final_rate", 0)),
        "ever_led_any": float(profile.get("ever_led_any", 0)),
        "nige_rate": float(profile.get("nige_rate", 0)),
        "jockey_position_bias": float(profile.get("jockey_position_bias", 0)),
        "same_surface_typical_norm": float(profile.get("same_surface_typical_norm", 0.5)),
        "distance_band_typical_norm": float(profile.get("distance_band_typical_norm", 0.5)),
    }


FEATURE_COLUMNS = [
    "typical_norm_pos",
    "style_code",
    "style_jra_code",
    "style_final_code",
    "pos_std",
    "n_past_races",
    "best_norm_pos_avg",
    "best_norm_pos_min",
    "final_norm_pos_avg",
    "corner_spread_avg",
    "ever_led_rate",
    "ever_led_before_final_rate",
    "ever_led_any",
    "nige_rate",
    "jockey_position_bias",
    "same_surface_typical_norm",
    "distance_band_typical_norm",
    "horse_number_norm",
    "bracket_number",
    "is_even_gate",
    "gate_zone",
    "left_neighbor_tendency",
    "right_neighbor_tendency",
    "left_neighbor_std",
    "right_neighbor_std",
    "neighbor_slow_count",
    "neighbor_fast_count",
    "neighbor_avg_tendency",
    "neighbor_space_score",
    "left_prev_top3",
    "right_prev_top3",
    "front_neighbor_count",
    "sandwich_front",
    "sandwich_push",
    "field_size",
    "front_runner_count",
    "closer_count",
    "same_style_count",
    "front_runner_ratio",
    "pace_pressure",
    "surface_code",
    "distance",
    "distance_category",
    "track_condition_code",
]


class TrackingDifficultyTrainer:
    """追走難度モデルの学習と MLflow 登録。"""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.mlflow_uri = mlflow_tracking_uri

    def train(
        self,
        df: pd.DataFrame | None = None,
        storage=None,
        years: list[str] | None = None,
        test_ratio: float = 0.2,
    ) -> dict[str, Any]:
        t0 = time.time()

        if df is None:
            if storage is None:
                from src.scraper.storage import HybridStorage
                storage = HybridStorage()
            df = build_training_dataset(storage, years=years)

        if df.empty:
            return {"error": "学習データなし"}

        logger.info("学習データ: %d行", len(df))

        X = df[FEATURE_COLUMNS].fillna(0).astype(float)
        y = df["position_deviation"].values

        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM 未インストール")
            return {"error": "LightGBM required"}

        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.5,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=callbacks,
        )

        preds = model.predict(X_test)
        mae = float(np.mean(np.abs(preds - y_test)))
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        corr = float(np.corrcoef(preds, y_test)[0, 1]) if len(y_test) > 2 else 0

        importance = dict(
            zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain").tolist())
        )

        metrics = {
            "mae": round(mae, 5),
            "rmse": round(rmse, 5),
            "correlation": round(corr, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(FEATURE_COLUMNS),
            "best_iteration": model.best_iteration,
            "training_time_sec": round(time.time() - t0, 1),
            "top_features": dict(sorted(importance.items(), key=lambda x: -x[1])[:10]),
            "feature_names": FEATURE_COLUMNS,
            "target_mean": round(float(y.mean()), 5),
            "target_std": round(float(y.std()), 5),
        }

        self._log_mlflow(model, metrics, lgb_params)

        return metrics

    def _log_mlflow(self, model, metrics: dict, lgb_params: dict):
        try:
            import mlflow
            import mlflow.lightgbm

            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)

            run_name = f"td_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_type", "tracking_difficulty")
                mlflow.set_tag("algorithm", "lightgbm_regression")

                mlflow.log_params({k: str(v) for k, v in lgb_params.items()})
                mlflow.log_params({
                    "n_features": metrics["n_features"],
                    "n_train": metrics["n_train"],
                    "n_test": metrics["n_test"],
                })

                log_metrics = {}
                for k in ("mae", "rmse", "correlation", "best_iteration",
                          "training_time_sec"):
                    if k in metrics and isinstance(metrics[k], (int, float)):
                        log_metrics[k] = metrics[k]
                mlflow.log_metrics(log_metrics)

                mlflow.lightgbm.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=MODEL_NAME,
                )

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as f:
                    json.dump({
                        "feature_names": FEATURE_COLUMNS,
                        "importance": metrics.get("top_features", {}),
                        "target_stats": {
                            "mean": metrics["target_mean"],
                            "std": metrics["target_std"],
                        },
                    }, f, ensure_ascii=False, indent=2)
                    tmp_path = f.name
                mlflow.log_artifact(tmp_path, "feature_info")
                os.unlink(tmp_path)

                logger.info(
                    "MLflow 記録完了: run_id=%s  MAE=%.5f  RMSE=%.5f  corr=%.4f",
                    run.info.run_id,
                    metrics["mae"], metrics["rmse"], metrics["correlation"],
                )

        except Exception as e:
            logger.warning("MLflow 記録失敗 (学習結果はローカルに保持): %s", e)


# ── 推論 ──

_MLFLOW_MODEL_KEY = "tracking_difficulty"


def load_model():
    """MLflow カタログ経由で Booster をロード（後方互換ラッパー）。"""
    from src.pipeline.mlflow.runtime import load_lightgbm_booster

    return load_lightgbm_booster(_MLFLOW_MODEL_KEY)


def _lgbm_inference_columns() -> list[str]:
    """登録済み Booster の feature_name に合わせる（学習時より列が増えていても推論可能）。"""
    from src.pipeline.mlflow.runtime import booster_feature_names

    names = booster_feature_names(_MLFLOW_MODEL_KEY)
    if names:
        return names
    return FEATURE_COLUMNS


def predict_lgbm_batch(feature_rows: list[dict]) -> list[float]:
    """
    推論用特徴量行のバッチから LGBM 生スコアを返す。

    MLflow Model Serving → ローカル Booster → ヒューリスティック。
    """
    from src.pipeline.mlflow.runtime import get_serve_client, load_lightgbm_booster

    if not feature_rows:
        return []

    cols = _lgbm_inference_columns()

    try:
        client = get_serve_client(_MLFLOW_MODEL_KEY)
        if client is not None and client.is_available():
            matrix = [[r.get(c, 0) for c in cols] for r in feature_rows]
            return client.predict_dataframe(cols, matrix)
    except Exception as e:
        logger.warning("MLflow serve 推論失敗 → ローカルへ: %s", e)

    model = load_lightgbm_booster(_MLFLOW_MODEL_KEY)
    if model is not None:
        X = pd.DataFrame([{c: r.get(c, 0) for c in cols} for r in feature_rows])
        return [float(v) for v in model.predict(X)]

    return [_heuristic_score(r) for r in feature_rows]


def predict_tracking_difficulty(
    race_data: dict,
    horse_histories: dict[str, list[dict]] | None = None,
    storage=None,
) -> tuple[list[dict], dict]:
    """
    レースデータに対して各馬の追走難度を推定する。

    Args:
        race_data: race_shutuba + race_result 等のバンドルデータ
        horse_histories: horse_id -> race_history のマッピング
        storage: HybridStorage (horse_histories が None の場合に使用)

    Returns:
        (馬番順の追走難度スコアリスト, フィールド統計dict)
    """
    shutuba = race_data.get("race_shutuba") or race_data.get("race_card") or {}
    result = race_data.get("race_result") or {}
    pre_race_only = bool(race_data.get("_pre_race_only"))
    from src.utils.race_entries import normalize_race_entries

    entries = normalize_race_entries(
        shutuba.get("entries") or (
            [] if pre_race_only else (result.get("entries") or [])
        )
    )
    if shutuba.get("entries"):
        shutuba = {**shutuba, "entries": entries}
        race_data["race_shutuba"] = shutuba
    if not entries:
        return [], {}

    current_distance = int(shutuba.get("distance") or result.get("distance") or 1600)
    current_surface = str(shutuba.get("surface") or result.get("surface") or "芝")

    date_raw = str(shutuba.get("date") or result.get("date") or "")
    race_info = {
        "race_id": shutuba.get("race_id") or result.get("race_id", ""),
        "race_date": _date_to_yyyymmdd(date_raw),
        "field_size": len(entries),
        "distance": current_distance,
        "surface": current_surface,
        "track_condition": (
            shutuba.get("track_condition")
            or ("" if pre_race_only else result.get("track_condition", ""))
            or "良"
        ),
    }
    ctx_before = race_info["race_date"]
    ctx_rid = str(race_info.get("race_id") or "")

    if horse_histories is None:
        horse_histories = {}
        horses = race_data.get("horses") or {}
        for hid, hdata in horses.items():
            if isinstance(hdata, dict):
                horse_histories[hid] = hdata.get("race_history", [])

    if not horse_histories and storage:
        for e in entries:
            hid = e.get("horse_id", "")
            if hid and hid not in horse_histories:
                hr = storage.load("horse_result", hid)
                if hr and "race_history" in hr:
                    horse_histories[hid] = hr["race_history"]

    profiles: dict[int, dict] = {}
    prev_features_map: dict[int, dict] = {}
    recent3_features_map: dict[int, dict] = {}
    for e in entries:
        hid = e.get("horse_id", "")
        hn = e.get("horse_number", 0)
        history = horse_histories.get(hid, [])
        hist_before = _filter_history_before_race(
            history, before_date=ctx_before, exclude_race_id=ctx_rid
        )
        profiles[hn] = build_horse_profile(
            hist_before,
            surface=current_surface,
            distance=current_distance,
            current_jockey_id=str(e.get("jockey_id") or ""),
        )
        prev_features_map[hn] = extract_prev_race_features(
            history,
            before_date=ctx_before,
            exclude_race_id=ctx_rid,
            horse_id=hid,
            storage=storage,
        )
        recent3_features_map[hn] = extract_recent3_features(
            history,
            before_date=ctx_before,
            exclude_race_id=ctx_rid,
            horse_id=hid,
            storage=storage,
        )

    field_profiles = list(profiles.values())

    # フィールドレベルの前走統計
    field_prev_stats = build_field_prev_stats(
        prev_features_map, entries, current_distance, current_surface
    )

    results = []
    infer_rows: list[dict] = []
    horse_ctx: list[dict] = []

    for e in entries:
        hn = e.get("horse_number", 0)
        hid = e.get("horse_id", "")
        bn = e.get("bracket_number", 0)
        profile = profiles.get(hn, build_horse_profile([]))
        prev_feat = prev_features_map.get(hn, _empty_prev_features())
        r3_feat = recent3_features_map.get(hn, _empty_recent3_features())
        style_prev = prev_feat.get("prev_style") or ""
        style_r3 = r3_feat.get("style") or ""
        form_style = blend_form_style(
            style_prev, style_r3, n_recent=int(r3_feat.get("n_races") or 0)
        )

        left_hn = hn - 1
        right_hn = hn + 1
        neighbors = {
            "left": profiles.get(left_hn, {}),
            "right": profiles.get(right_hn, {}),
        }
        left_prev = prev_features_map.get(left_hn, _empty_prev_features())
        right_prev = prev_features_map.get(right_hn, _empty_prev_features())
        nf = build_neighbor_gate_factors(
            hn, race_info["field_size"], profile,
            neighbors["left"], neighbors["right"],
            left_prev=left_prev, right_prev=right_prev,
        )

        row = _build_inference_row(
            e, race_info, profile, neighbors, field_profiles, nf=nf
        )
        infer_rows.append(row)
        horse_ctx.append({
            "e": e,
            "hn": hn,
            "hid": hid,
            "bn": bn,
            "profile": profile,
            "prev_feat": prev_feat,
            "r3_feat": r3_feat,
            "form_style": form_style,
            "style_prev": style_prev,
            "style_r3": style_r3,
            "nf": nf,
            "row": row,
        })

    preds = predict_lgbm_batch(infer_rows)

    for ctx, pred in zip(horse_ctx, preds):
        e = ctx["e"]
        hn = ctx["hn"]
        hid = ctx["hid"]
        bn = ctx["bn"]
        profile = ctx["profile"]
        prev_feat = ctx["prev_feat"]
        r3_feat = ctx["r3_feat"]
        form_style = ctx["form_style"]
        style_prev = ctx["style_prev"]
        style_r3 = ctx["style_r3"]
        nf = ctx["nf"]
        row = ctx["row"]

        ease_pct = _deviation_to_ease(pred, profile)
        ease_pct = _apply_sandwich_ease_adjustment(
            ease_pct, profile.get("style", "差し"), nf["sandwich_push"]
        )

        # コース比較
        course_cmp = build_course_comparison(
            e, prev_feat, current_distance, current_surface
        )

        # T1F正規化値
        t1f_norm = field_prev_stats.get("normalized_t1f", {}).get(hn, 0.5)

        results.append({
            "horse_number": hn,
            "horse_id": hid,
            "horse_name": e.get("horse_name", ""),
            "bracket_number": bn,
            "tracking_difficulty": {
                "score": round(pred, 4),
                "ease_pct": round(ease_pct, 1),
                "label": _ease_label(ease_pct),
            },
            "profile": {
                "typical_position": profile["typical_norm_pos"],
                "style": profile["style"],
                "style_jra": profile.get("style_jra", profile["style"]),
                "style_final": profile.get("style_final", profile["style"]),
                "stability": round(1.0 - profile["pos_std"], 3),
                "n_races": profile["n_races"],
                "positions": profile.get("positions", []),
                "best_norms": profile.get("best_norms", []),
            },
            "position_ability": {
                "best_norm_pos_avg": profile.get("best_norm_pos_avg", 0.5),
                "best_norm_pos_min": profile.get("best_norm_pos_min", 0.5),
                "final_norm_pos_avg": profile.get("final_norm_pos_avg", 0.5),
                "ever_led_rate": profile.get("ever_led_rate", 0),
                "ever_led_before_final_rate": profile.get(
                    "ever_led_before_final_rate", 0
                ),
                "ever_led_any": bool(profile.get("ever_led_any")),
                "ever_led_before_final": bool(profile.get("ever_led_before_final")),
                "corner_spread_avg": profile.get("corner_spread_avg", 0),
                "nige_rate": profile.get("nige_rate", 0),
                "jockey_position_bias": profile.get("jockey_position_bias", 0),
                "same_surface_typical_norm": profile.get(
                    "same_surface_typical_norm", 0.5
                ),
                "distance_band_typical_norm": profile.get(
                    "distance_band_typical_norm", 0.5
                ),
            },
            "gate_factors": {
                "horse_number": hn,
                "bracket_number": bn,
                "is_even_gate": bool(hn % 2 == 0),
                "gate_zone": row.get("gate_zone", 0),
            },
            "neighbor_factors": {
                "left_tendency": nf["left_neighbor_tendency"],
                "right_tendency": nf["right_neighbor_tendency"],
                "space_score": nf["neighbor_space_score"],
                "slow_neighbors": nf["neighbor_slow_count"],
                "fast_neighbors": nf["neighbor_fast_count"],
                "left_prev_t1f": nf["left_prev_t1f"],
                "right_prev_t1f": nf["right_prev_t1f"],
                "left_prev_top3": nf["left_prev_top3"],
                "right_prev_top3": nf["right_prev_top3"],
                "front_neighbor_count": nf["front_neighbor_count"],
                "sandwich_front": nf["sandwich_front"],
                "sandwich_push": nf["sandwich_push"],
            },
            "field_factors": {
                "front_runner_count": row.get("front_runner_count", 0),
                "closer_count": row.get("closer_count", 0),
                "same_style_count": row.get("same_style_count", 0),
                "pace_pressure": round(row.get("pace_pressure", 0), 3),
            },
            "prev_race": {
                "t1f_raw": prev_feat["prev_t1f_raw"],
                "t1f_norm": round(t1f_norm, 4),
                "pos_ratio": prev_feat["prev_pos_ratio"],
                "field_size": prev_feat["prev_field_size"],
                "surface": prev_feat["prev_surface"],
                "distance": prev_feat["prev_distance"],
                "horse_number": prev_feat["prev_horse_number"],
                "last3f": prev_feat["prev_last3f"],
                "passing_order": prev_feat.get("prev_passing_order", ""),
                "finish_pos": prev_feat["prev_finish_pos"],
                "prev_was_top3": prev_feat["prev_prev_was_top3"],
                "style": style_prev,
                "style_jra": prev_feat.get("prev_style_jra", ""),
                "style_final": prev_feat.get("prev_style_final", ""),
                "best_norm": prev_feat.get("prev_best_norm", 0.5),
                "ever_led_any": bool(prev_feat.get("prev_ever_led_any")),
                "ever_led_before_final": bool(
                    prev_feat.get("prev_ever_led_before_final")
                ),
                "corner_spread": prev_feat.get("prev_corner_spread", 0),
            },
            "recent_3": {
                "n_races": r3_feat["n_races"],
                "t1f_raw_avg": r3_feat["t1f_raw_avg"],
                "pos_ratio_avg": r3_feat["pos_ratio_avg"],
                "field_size_avg": r3_feat["field_size_avg"],
                "style": style_r3,
                "t1f_raws": r3_feat.get("t1f_raws") or [],
                "best_norm_pos_avg": r3_feat.get("best_norm_pos_avg", 0.5),
                "best_norm_pos_min": r3_feat.get("best_norm_pos_min", 0.5),
                "ever_led_rate": r3_feat.get("ever_led_rate", 0),
                "best_norms": r3_feat.get("best_norms") or [],
            },
            "form": {
                "style": form_style,
                "style_prev": style_prev,
                "style_recent3": style_r3,
            },
            "course_comparison": course_cmp,
        })

    results.sort(key=lambda x: -x["tracking_difficulty"]["ease_pct"])
    return results, field_prev_stats


def _build_inference_row(
    entry: dict,
    race_info: dict,
    profile: dict,
    neighbors: dict,
    field_profiles: list[dict],
    nf: dict | None = None,
) -> dict:
    hn = entry.get("horse_number", 0)
    bn = entry.get("bracket_number", 0)
    fs = race_info["field_size"]

    if nf is None:
        nf = build_neighbor_gate_factors(
            hn, fs, profile,
            neighbors.get("left", {}),
            neighbors.get("right", {}),
        )

    front_runners = sum(
        1 for p in field_profiles if p.get("style") in ("逃げ", "先行")
    )
    closers = sum(
        1 for p in field_profiles if p.get("style") in ("差し", "追込")
    )
    same_style = sum(
        1 for p in field_profiles if p.get("style") == profile.get("style")
    )

    distance = race_info.get("distance", 1600)
    surface = race_info.get("surface", "芝")
    cond = race_info.get("track_condition", "良")

    return {
        **_profile_model_features(profile),
        "horse_number_norm": round(hn / fs, 4) if fs > 0 else 0.5,
        "bracket_number": bn,
        "is_even_gate": int(hn % 2 == 0),
        "gate_zone": (
            0 if hn <= fs * 0.25 else
            1 if hn <= fs * 0.5 else
            2 if hn <= fs * 0.75 else 3
        ),
        "left_neighbor_tendency": nf["left_neighbor_tendency"],
        "right_neighbor_tendency": nf["right_neighbor_tendency"],
        "left_neighbor_std": nf["left_neighbor_std"],
        "right_neighbor_std": nf["right_neighbor_std"],
        "neighbor_slow_count": nf["neighbor_slow_count"],
        "neighbor_fast_count": nf["neighbor_fast_count"],
        "neighbor_avg_tendency": nf["neighbor_avg_tendency"],
        "neighbor_space_score": nf["neighbor_space_score"],
        "left_prev_top3": int(nf["left_prev_top3"]),
        "right_prev_top3": int(nf["right_prev_top3"]),
        "front_neighbor_count": nf["front_neighbor_count"],
        "sandwich_front": int(nf["sandwich_front"]),
        "sandwich_push": nf["sandwich_push"],
        "field_size": fs,
        "front_runner_count": front_runners,
        "closer_count": closers,
        "same_style_count": same_style,
        "front_runner_ratio": round(front_runners / max(fs, 1), 4),
        "pace_pressure": round(front_runners / max(closers, 1), 4),
        "surface_code": 0 if surface == "芝" else 1,
        "distance": distance,
        "distance_category": (
            0 if distance <= 1400 else
            1 if distance <= 1800 else
            2 if distance <= 2200 else 3
        ),
        "track_condition_code": (
            0 if cond == "良" else
            1 if cond == "稍" else
            2 if cond == "重" else 3
        ),
    }


def _heuristic_score(row: dict) -> float:
    """モデルがない場合のヒューリスティックスコア。"""
    score = 0.0
    score += 0.02 * row.get("is_even_gate", 0)
    score += 0.03 * row.get("neighbor_slow_count", 0)
    score += 0.04 * row.get("neighbor_fast_count", 0)  # 速い隣馬の恩恵
    score += 0.05 * row.get("neighbor_space_score", 0)
    score += 0.06 * row.get("sandwich_push", 0)
    score -= 0.03 * row.get("front_neighbor_count", 0)  # 前目隣が多いと消耗
    score -= 0.02 * row.get("front_runner_ratio", 0)
    score -= 0.01 * (row.get("same_style_count", 0) - 1)
    score += 0.01 * (row.get("n_past_races", 0) / 10)
    score -= 0.04 * row.get("best_norm_pos_min", 0.5)
    score -= 0.03 * row.get("ever_led_before_final_rate", 0)
    score += 0.02 * row.get("jockey_position_bias", 0)
    score -= 0.02 * row.get("corner_spread_avg", 0)
    return score


def _deviation_to_ease(deviation: float, profile: dict) -> float:
    """deviation値をイージースコア（0-100）に変換。"""
    base = 50.0
    scaled = deviation * 200
    stability_bonus = (1.0 - profile.get("pos_std", 0.2)) * 10
    experience_bonus = min(profile.get("n_races", 0), 10) * 0.5
    ease = base + scaled + stability_bonus + experience_bonus
    return max(0, min(100, ease))


def _ease_label(ease_pct: float) -> str:
    if ease_pct >= 75:
        return "非常に楽"
    elif ease_pct >= 60:
        return "楽"
    elif ease_pct >= 45:
        return "普通"
    elif ease_pct >= 30:
        return "やや困難"
    else:
        return "困難"


# ── ペース予想（過去統計ベース）──

_PACE_COHORTS: dict[str, dict[str, Any]] | None = None
_EMPIRICAL_PACE_CACHE: dict[str, dict[str, float]] = {}


def _infer_grade_bucket(race_info: dict) -> str:
    name = str(race_info.get("race_name") or "")
    if "ヴィクトリア" in name or "有馬記念" in name or "ジャパンカップ" in name:
        return "G1"
    for g in ("G1", "G2", "G3"):
        if g in name.upper() or f"({g})" in name or f"（{g}）" in name:
            return g
    grade = str(race_info.get("grade") or "").strip().upper()
    if grade in ("G1", "G2", "G3", "L", "OP"):
        return grade
    return ""


def _load_pace_cohorts() -> dict[str, dict[str, Any]]:
    global _PACE_COHORTS
    if _PACE_COHORTS is not None:
        return _PACE_COHORTS
    from src.config.data_paths import TRACK_SPEED_PACE_BASELINES

    _PACE_COHORTS = {}
    path = TRACK_SPEED_PACE_BASELINES
    if not path.exists():
        logger.warning("ペース基準ファイルがありません: %s", path)
        return _PACE_COHORTS
    try:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            _PACE_COHORTS[str(row["context_key"])] = {
                "metric": str(row["metric"]),
                "mean": float(row["metric_mean"]),
                "std": float(row["metric_std"]),
                "beta": float(row["beta"]),
                "n": int(row.get("n", 0)),
            }
    except Exception as e:
        logger.warning("ペース基準読込失敗: %s", e)
    return _PACE_COHORTS


def _lookup_pace_cohort(
    venue: str,
    surface: str,
    distance: int,
    track_condition: str,
) -> dict[str, Any] | None:
    from src.research.race.track_speed_engine import (
        COND_CANDIDATES,
        load_course_layout,
        pace_coarse_context_key,
        pace_context_key,
    )

    cohorts = _load_pace_cohorts()
    if not cohorts:
        return None
    layout = load_course_layout(venue, surface, distance)
    tc = str(track_condition or "良").strip()
    pools = COND_CANDIDATES.get(tc, ())
    seen: set[str] = set()
    layouts = [layout, "-"] if layout != "-" else ["-"]
    keys: list[str] = []
    for pool in pools:
        for lay in layouts:
            keys.append(pace_context_key(venue, lay, surface, distance, pool))
        keys.append(pace_coarse_context_key(surface, distance, pool))
    for ctx in keys:
        if ctx in seen:
            continue
        seen.add(ctx)
        meta = cohorts.get(ctx)
        if meta:
            return {**meta, "context_key": ctx}
    return None


def _empirical_fh3f_stats(
    venue: str,
    surface: str,
    distance: int,
    grade_bucket: str = "",
) -> dict[str, float] | None:
    """track_speed レース Parquet から同条件の前半3F実績を集計。"""
    from src.config.data_paths import TRACK_SPEED_RACES_DIR

    cache_key = f"{venue}|{surface}|{distance}|{grade_bucket}"
    if cache_key in _EMPIRICAL_PACE_CACHE:
        return _EMPIRICAL_PACE_CACHE[cache_key]

    if not TRACK_SPEED_RACES_DIR.exists():
        return None

    frames: list[pd.Series] = []
    for path in sorted(TRACK_SPEED_RACES_DIR.glob("races_*.parquet")):
        try:
            df = pd.read_parquet(
                path,
                columns=["venue", "surface", "distance", "grade", "first_half_3f"],
            )
        except Exception:
            continue
        m = (
            (df["venue"] == venue)
            & (df["surface"] == surface)
            & (df["distance"] == int(distance))
            & df["first_half_3f"].notna()
            & (df["first_half_3f"] > 20)
        )
        if grade_bucket:
            m = m & (df["grade"] == grade_bucket)
        sub = df.loc[m, "first_half_3f"]
        if len(sub) >= 3:
            frames.append(sub.astype(float))

    if not frames:
        _EMPIRICAL_PACE_CACHE[cache_key] = {}
        return None

    allv = pd.concat(frames, ignore_index=True)
    stats = {
        "mean": float(allv.mean()),
        "std": float(allv.std()) if len(allv) > 1 else 0.6,
        "median": float(allv.median()),
        "p25": float(allv.quantile(0.25)),
        "p75": float(allv.quantile(0.75)),
        "n": int(len(allv)),
    }
    _EMPIRICAL_PACE_CACHE[cache_key] = stats
    return stats


def _resolve_pace_baseline(race_info: dict) -> dict[str, Any]:
    """コホート基準 + 同条件実績をブレンドした前半3F基準。"""
    venue = str(race_info.get("venue") or "").strip()
    surface = _normalize_surface(race_info.get("surface", "芝"))
    distance = int(race_info.get("distance") or 1600)
    track_cond = str(race_info.get("track_condition") or "良").strip()
    grade_bucket = _infer_grade_bucket(race_info)

    cohort = _lookup_pace_cohort(venue, surface, distance, track_cond)
    empirical = _empirical_fh3f_stats(venue, surface, distance, grade_bucket)
    if empirical is None or empirical.get("n", 0) < 5:
        empirical = _empirical_fh3f_stats(venue, surface, distance, "")
    empirical_g1 = (
        _empirical_fh3f_stats(venue, surface, distance, "G1")
        if grade_bucket == "G1"
        else None
    )

    if cohort and empirical and empirical.get("n", 0) >= 8:
        w_emp = min(0.65, 0.35 + empirical["n"] / 80)
        mean = cohort["mean"] * (1 - w_emp) + empirical["mean"] * w_emp
        if empirical_g1 and empirical_g1.get("n", 0) >= 3:
            mean = mean * 0.55 + empirical_g1["mean"] * 0.45
        std = max(
            cohort["std"] * (1 - w_emp * 0.5) + empirical["std"] * (w_emp * 0.5),
            0.35,
        )
        source = "cohort+empirical"
        if empirical_g1 and empirical_g1.get("n", 0) >= 3:
            source = "cohort+empirical+G1"
    elif empirical and empirical.get("n", 0) >= 5:
        mean = empirical["mean"]
        std = max(empirical["std"], 0.35)
        source = "empirical"
    elif cohort:
        mean = cohort["mean"]
        std = max(cohort["std"], 0.35)
        source = "cohort"
    else:
        mean = 35.2 if surface == "芝" else 34.8
        std = 1.0
        source = "fallback"

    return {
        "first_half_3f_mean": round(mean, 3),
        "first_half_3f_std": round(std, 3),
        "source": source,
        "cohort_key": cohort.get("context_key", "") if cohort else "",
        "empirical_n": int(empirical.get("n", 0)) if empirical else 0,
        "grade_bucket": grade_bucket,
    }


def _first_1f_from_fh3f(first_3f: float, distance: int, surface: str) -> float:
    """前半3Fから1F(200m)を同条件の典型比率で換算。"""
    ratio = 12.0 / 35.0
    if distance <= 1400:
        ratio = 11.8 / 34.0
    elif distance >= 2000:
        ratio = 12.5 / 36.5
    if surface == "ダート":
        ratio *= 1.02
    return first_3f * ratio


def _field_early_pressure(
    horse_profiles: dict[int, dict],
    field_prev_stats: dict | None,
) -> dict[str, float]:
    """脚質ラベルではなく連続値 + 前走T1F統計で先行意識（前半の争い強度）を推定。"""
    positions = [
        p.get("typical_norm_pos", 0.5)
        for p in horse_profiles.values()
        if p
    ]
    if not positions:
        return {
            "mean_norm_pos": 0.5,
            "forward_intent": 0.5,
            "front_share": 0.0,
            "stalk_share": 0.0,
        }

    mean_pos = sum(positions) / len(positions)
    forward_intent = max(0.0, min(1.0, 1.0 - mean_pos))

    front_share = sum(1 for x in positions if x <= 0.15) / len(positions)
    stalk_share = sum(1 for x in positions if 0.15 < x <= 0.35) / len(positions)
    led_bf_rate = sum(
        float(p.get("ever_led_before_final_rate") or 0)
        for p in horse_profiles.values()
        if p
    ) / max(len(horse_profiles), 1)
    best_front = sum(
        1
        for p in horse_profiles.values()
        if p and float(p.get("best_norm_pos_min", 0.5)) <= 0.12
    ) / max(len(horse_profiles), 1)

    if field_prev_stats:
        fpr = float(field_prev_stats.get("front_pressure_ratio", 0))
        avg_prev = float(field_prev_stats.get("avg_prev_pos_ratio", 0.5))
        forward_intent = min(
            1.0,
            max(
                0.0,
                0.50 * forward_intent
                + 0.22 * (1.0 - avg_prev)
                + 0.18 * min(fpr / 0.45, 1.0)
                + 0.10 * min(led_bf_rate / 0.35, 1.0),
            ),
        )
    else:
        forward_intent = min(
            1.0,
            max(
                0.0,
                forward_intent + 0.12 * min(led_bf_rate / 0.35, 1.0),
            ),
        )

    return {
        "mean_norm_pos": round(mean_pos, 3),
        "forward_intent": round(forward_intent, 3),
        "front_share": round(front_share, 3),
        "stalk_share": round(stalk_share, 3),
        "ever_led_before_final_share": round(led_bf_rate, 3),
        "best_front_share": round(best_front, 3),
    }


def predict_race_pace(
    entries: list[dict],
    horse_profiles: dict[int, dict],
    race_info: dict,
    field_prev_stats: dict | None = None,
) -> dict:
    """
    レース全体のペース予想（track_speed コホート + 同条件実績ベース）。

    前半3Fは ``track_speed_pace_baselines`` と ``track_speed/races_*.parquet`` の
    統計から決め、出走メンバーの先行意識で微調整する。
    """
    field_size = len(entries)
    if field_size == 0:
        return {"pace_type": "不明", "pace_index": 50, "front_runner_count": 0}

    distance = int(race_info.get("distance") or 1600)
    surface = _normalize_surface(race_info.get("surface", "芝"))
    track_cond = str(race_info.get("track_condition") or "良").strip()

    baseline = _resolve_pace_baseline(race_info)
    fh_mean = baseline["first_half_3f_mean"]
    fh_std = baseline["first_half_3f_std"]

    field_press = _field_early_pressure(horse_profiles, field_prev_stats)
    forward_intent = field_press["forward_intent"]

    # 先行意識が高いほど前半3Fは短く（速い）
    intent_shift = (forward_intent - 0.42) * fh_std * 0.85
    track_adj = 0.0
    if track_cond in ("重", "不良"):
        track_adj = 0.55
    elif track_cond in ("稍", "稍重"):
        track_adj = 0.25

    predicted_fh3f = fh_mean - intent_shift + track_adj
    predicted_fh3f = max(fh_mean - fh_std * 1.2, min(fh_mean + fh_std * 1.2, predicted_fh3f))

    z_pace = (predicted_fh3f - fh_mean) / fh_std if fh_std > 1e-6 else 0.0
    # z<0 → 速い前半 → ハイペース / 高い pace_index
    pace_index = max(0.0, min(100.0, 50.0 - z_pace * 22.0))

    if z_pace <= -0.55:
        pace_type = "ハイ"
        pattern = "前傾ラップ（統計より速い前半）"
    elif z_pace >= 0.55:
        pace_type = "スロー"
        pattern = "抑えラップ（統計より遅い前半）"
    else:
        pace_type = "ミドル"
        pattern = "同条件の平均的な前半ペース"

    front_count = sum(
        1 for p in horse_profiles.values()
        if p.get("style") == "逃げ"
    )
    stalker_count = sum(
        1 for p in horse_profiles.values()
        if p.get("style") == "先行"
    )
    eff_front = int(round(field_press["front_share"] * field_size))
    eff_stalk = int(round(field_press["stalk_share"] * field_size))

    if eff_front >= 2 and forward_intent >= 0.5:
        pattern = "先行意識が高く、先行争いが激しくなりやすい"
        if pace_type == "ミドル":
            pace_type = "ハイ"
    elif eff_front == 0 and forward_intent < 0.38:
        pattern = "前に行く馬が少なく、統計ベースではやや抑え気味"
    elif eff_front == 1 and eff_stalk <= 2 and forward_intent < 0.45:
        pattern = "単騎前付け → ややスロー寄りの流れもあり得る"

    first_3f = round(predicted_fh3f, 1)
    first_1f = round(_first_1f_from_fh3f(predicted_fh3f, distance, surface), 1)

    midpack = sum(1 for p in horse_profiles.values() if p.get("style") == "差し")
    closers = sum(1 for p in horse_profiles.values() if p.get("style") == "追込")

    return {
        "pace_type": pace_type,
        "pace_index": round(pace_index, 1),
        "front_runner_count": front_count,
        "stalker_count": stalker_count,
        "early_pressure": round(forward_intent, 3),
        "predicted_pattern": pattern,
        "pace_factors": {
            "baseline_fh3f": fh_mean,
            "baseline_std": fh_std,
            "baseline_source": baseline["source"],
            "z_pace": round(z_pace, 3),
            "forward_intent": forward_intent,
            "intent_shift_sec": round(-intent_shift, 2),
            "track_adj_sec": round(track_adj, 2),
            "empirical_n": baseline.get("empirical_n", 0),
            "cohort_key": baseline.get("cohort_key", ""),
        },
        "distribution": {
            "逃げ": front_count,
            "先行": stalker_count,
            "差し": midpack,
            "追込": closers,
        },
        "distribution_effective": {
            "逃げ": eff_front,
            "先行": eff_stalk,
            "差し": max(0, field_size - eff_front - eff_stalk - int(
                sum(1 for p in horse_profiles.values() if p.get("typical_norm_pos", 0.5) > 0.65)
            )),
            "追込": sum(
                1 for p in horse_profiles.values()
                if p.get("typical_norm_pos", 0.5) > 0.65
            ),
        },
        "lap_times": {
            "first_1f": first_1f,
            "first_3f": first_3f,
        },
    }


# ── 位置取り予測 ──

_STYLE_FORWARD_ADJ = {
    "逃げ": 0.0,
    "先行": 0.22,
    "差し": 0.48,
    "追込": 0.72,
}


def _field_percentile_ranks(
    scores: dict[int, float],
    *,
    lower_is_forward: bool = True,
) -> dict[int, float]:
    """
    フィールド内の相対順位（0=最も前寄り、1=最も後ろ寄り）。

    lower_is_forward=True のときスコアが小さい馬ほど 0 に近い。
    """
    if not scores:
        return {}
    ordered = sorted(
        scores.items(),
        key=lambda x: (x[1], x[0]),
        reverse=not lower_is_forward,
    )
    n = len(ordered)
    if n <= 1:
        return {hn: 0.0 for hn, _ in ordered}
    return {hn: i / (n - 1) for i, (hn, _) in enumerate(ordered)}


def _build_early_forward_composite(
    entries: list[dict],
    horse_profiles: dict[int, dict],
    td_by_hn: dict[int, dict],
    *,
    field_prev_stats: dict | None,
    field_size: int,
    pace_type: str,
) -> dict[int, float]:
    """
    序盤想定位置用の総合スコア（0に近いほど前）。

    追走容易度・前走T1F（正規化/生順位）・枠順・脚質をフィールド内で比較する。
    """
    fs = max(int(field_size or 1), 1)
    t1f_norm_scores: dict[int, float] = {}
    t1f_raw_ratio_scores: dict[int, float] = {}
    best_norm_scores: dict[int, float] = {}
    ease_scores: dict[int, float] = {}
    gate_scores: dict[int, float] = {}
    style_scores: dict[int, float] = {}
    typical_scores: dict[int, float] = {}

    for e in entries:
        hn = int(e.get("horse_number") or 0)
        if hn <= 0:
            continue
        td = td_by_hn.get(hn, {})
        profile = horse_profiles.get(hn, {})
        gate = td.get("gate_factors") or {}
        prev = td.get("prev_race") or {}
        pa = td.get("position_ability") or profile

        t1f_norm = 0.5
        if field_prev_stats:
            t1f_norm = float(field_prev_stats.get("normalized_t1f", {}).get(hn, 0.5))
        t1f_norm_scores[hn] = t1f_norm

        prev_fs = int(prev.get("field_size") or 0) or fs
        t1f_raw = int(prev.get("t1f_raw") or 0)
        if t1f_raw > 0 and prev_fs > 0:
            t1f_raw_ratio_scores[hn] = t1f_raw / prev_fs
        else:
            t1f_raw_ratio_scores[hn] = t1f_norm

        best_norm_scores[hn] = float(
            pa.get("best_norm_pos_min")
            or prev.get("best_norm")
            or profile.get("best_norm_pos_min", 0.5)
        )

        ease_scores[hn] = float((td.get("tracking_difficulty") or {}).get("ease_pct", 50))
        bn = int(gate.get("bracket_number") or max(1, (hn + 1) // 2))
        gate_scores[hn] = bn + (hn - 1) / max(fs - 1, 1) * 0.35
        style_scores[hn] = _STYLE_FORWARD_ADJ.get(
            profile.get("style_jra") or profile.get("style", "差し"), 0.48
        )
        typical_scores[hn] = float(profile.get("typical_norm_pos", 0.5))

    t1f_norm_rank = _field_percentile_ranks(t1f_norm_scores, lower_is_forward=True)
    t1f_raw_rank = _field_percentile_ranks(t1f_raw_ratio_scores, lower_is_forward=True)
    best_norm_rank = _field_percentile_ranks(best_norm_scores, lower_is_forward=True)
    ease_rank = _field_percentile_ranks(ease_scores, lower_is_forward=False)
    gate_rank = _field_percentile_ranks(gate_scores, lower_is_forward=True)
    typical_rank = _field_percentile_ranks(typical_scores, lower_is_forward=True)
    style_rank = _field_percentile_ranks(style_scores, lower_is_forward=True)

    composite: dict[int, float] = {}
    for e in entries:
        hn = int(e.get("horse_number") or 0)
        if hn <= 0:
            continue
        td = td_by_hn.get(hn, {})
        profile = horse_profiles.get(hn, {})
        pa = td.get("position_ability") or profile
        prev = td.get("prev_race") or {}
        neighbor = td.get("neighbor_factors") or {}
        score = (
            t1f_norm_rank.get(hn, 0.5) * 0.20
            + t1f_raw_rank.get(hn, 0.5) * 0.18
            + best_norm_rank.get(hn, 0.5) * 0.18
            + ease_rank.get(hn, 0.5) * 0.28
            + gate_rank.get(hn, 0.5) * 0.08
            + typical_rank.get(hn, 0.5) * 0.03
            + style_rank.get(hn, 0.5) * 0.03
        )
        score -= float(pa.get("ever_led_before_final_rate") or 0) * 0.06
        score += float(profile.get("jockey_position_bias") or 0) * 0.08
        if prev.get("ever_led_before_final"):
            score -= 0.04
        score -= float(neighbor.get("sandwich_push") or 0) * 0.12
        if pace_type == "ハイ" and profile.get("style") in ("逃げ", "先行"):
            score -= 0.03
        elif pace_type == "スロー" and profile.get("style") in ("差し", "追込"):
            score += 0.03
        composite[hn] = score

    return composite


def _assign_unique_positions(
    desire_scores: dict[int, float],
    field_size: int,
) -> dict[int, int]:
    """ desire_scores の昇順で 1〜field_size の通過順位を重複なく割当。"""
    fs = max(int(field_size or 1), 1)
    ordered = sorted(
        desire_scores.items(),
        key=lambda x: (x[1], x[0]),
    )
    ranks: dict[int, int] = {}
    for i, (hn, _) in enumerate(ordered):
        ranks[hn] = min(i + 1, fs)
    return ranks


def _rank_to_norm(rank: int, field_size: int) -> float:
    fs = max(int(field_size or 1), 1)
    if fs <= 1:
        return 0.0
    return (int(rank) - 1) / (fs - 1)


def _is_valid_last3f_sec(v: float) -> bool:
    return 30.0 < v < 50.0


def _horse_last3f_ability(
    race_history: list[dict] | None,
    *,
    prev_last3f: float = 0.0,
    max_rows: int = 5,
) -> dict[str, float]:
    """直近戦績から上がり3Fの基礎能力（秒・小さいほど速い）を推定。"""
    vals: list[float] = []
    weights: list[float] = []
    if prev_last3f > 0 and _is_valid_last3f_sec(prev_last3f):
        vals.append(prev_last3f)
        weights.append(1.2)

    for i, rec in enumerate((race_history or [])[: max_rows * 2]):
        l3 = _safe_float(rec.get("last_3f"))
        if _is_valid_last3f_sec(l3):
            vals.append(l3)
            weights.append(1.0 / (i + 1))

    if not vals:
        return {
            "ability_sec": 0.0,
            "recent_avg": 0.0,
            "best_recent": 0.0,
            "n_samples": 0,
        }

    w_sum = sum(weights)
    ability = sum(v * w for v, w in zip(vals, weights)) / w_sum
    return {
        "ability_sec": round(ability, 2),
        "recent_avg": round(ability, 2),
        "best_recent": round(min(vals), 2),
        "n_samples": len(vals),
    }


def _resolve_last3f_baseline(
    race_info: dict,
    pace_prediction: dict,
    horse_abilities: list[float],
) -> dict[str, float]:
    """コース・ペース・出走馬実績からフィールド基準の上がり3F（秒）を決める。"""
    distance = int(race_info.get("distance") or 1600)
    surface = _normalize_surface(race_info.get("surface", "芝"))
    lap = pace_prediction.get("lap_times") or {}
    fh3f = _safe_float(lap.get("first_3f"))

    valid = [a for a in horse_abilities if a > 0 and _is_valid_last3f_sec(a)]
    if valid:
        field_med = float(sorted(valid)[len(valid) // 2])
    elif surface == "ダート":
        field_med = 34.0 if distance <= 1600 else 35.0
    else:
        if distance <= 1400:
            field_med = 33.6
        elif distance <= 1800:
            field_med = 34.2
        else:
            field_med = 35.0

    if fh3f > 20:
        rpci = 1.05 if distance <= 1400 else 1.02 if distance <= 1800 else 0.98
        if surface == "ダート":
            rpci -= 0.03
        from_fh = fh3f / max(rpci, 0.85)
        field_med = field_med * 0.45 + from_fh * 0.55

    track_cond = str(race_info.get("track_condition") or "良").strip()
    if track_cond in ("重", "不良"):
        field_med += 0.35
    elif track_cond in ("稍", "稍重"):
        field_med += 0.15

    return {
        "baseline_sec": round(field_med, 2),
        "from_first_half_3f": round(fh3f, 1) if fh3f > 20 else 0.0,
    }


_STYLE_LAST3F_ADJ = {
    "逃げ": 0.18,
    "先行": 0.06,
    "差し": -0.14,
    "追込": -0.24,
}


def _last3f_pace_label(delta: float) -> str:
    """baseline との差（秒）。負=速い。"""
    if delta <= -0.35:
        return "鋭い上がり"
    if delta <= -0.15:
        return "やや速い"
    if delta < 0.15:
        return "標準"
    if delta < 0.35:
        return "やや遅い"
    return "上がり苦しい"


def predict_expected_last_3f(
    *,
    horse_number: int,
    profile: dict,
    td_data: dict,
    position_flow_row: dict,
    pace_prediction: dict,
    race_info: dict,
    baseline_sec: float,
    horse_ability: dict[str, float],
) -> dict[str, Any]:
    """
    想定上り3F（秒）を1頭分推定する。

    前走・近走の上がり実績、想定位置取り、ペース、脚質、温存度を合成する。
    """
    pace_type = pace_prediction.get("pace_type", "ミドル")
    pace_index = float(pace_prediction.get("pace_index", 50))
    style = profile.get("style", "差し")
    prev = td_data.get("prev_race") or {}
    course = td_data.get("course_comparison") or {}
    positions = position_flow_row.get("positions") or {}
    early_norm = float(positions.get("early", {}).get("position_norm", 0.5))
    late_norm = float(positions.get("late", {}).get("position_norm", 0.5))
    flow = position_flow_row.get("flow_pattern", "")
    stamina = position_flow_row.get("stamina") or {}
    stamina_index = float(stamina.get("stamina_index", 50))

    ability_sec = float(horse_ability.get("ability_sec") or 0)
    if ability_sec > 0:
        core = ability_sec * 0.62 + baseline_sec * 0.38
    else:
        core = baseline_sec

    adj = _STYLE_LAST3F_ADJ.get(style, 0.0)

    if pace_type == "ハイ":
        adj += early_norm * 0.42 - (1.0 - early_norm) * 0.28
        adj += (pace_index - 50) / 100 * 0.25
    elif pace_type == "スロー":
        adj += early_norm * 0.12 - (1.0 - early_norm) * 0.18
        adj -= (50 - pace_index) / 100 * 0.12
    else:
        adj += early_norm * 0.18 - (1.0 - early_norm) * 0.12

    if flow in ("追い込み", "差し"):
        adj -= 0.22 + max(0.0, early_norm - late_norm) * 0.15
    elif flow in ("後退", "微後退"):
        adj += 0.32
    elif flow == "粘り込み" and style in ("逃げ", "先行"):
        adj += 0.08

    adj += (50.0 - stamina_index) / 100.0 * 0.55

    dist_change = int(course.get("dist_change") or 0)
    if dist_change > 50:
        adj += 0.12
    elif dist_change < -50:
        adj -= 0.08

    if not bool(course.get("same_surface", True)):
        adj += 0.06

    predicted = max(32.0, min(38.5, core + adj))
    delta = predicted - baseline_sec

    return {
        "seconds": round(predicted, 1),
        "baseline_sec": round(baseline_sec, 1),
        "delta_sec": round(delta, 2),
        "label": _last3f_pace_label(delta),
        "ability_sec": ability_sec,
        "n_samples": int(horse_ability.get("n_samples") or 0),
        "best_recent": float(horse_ability.get("best_recent") or 0),
    }


def _assign_last3f_ranks(predictions: dict[int, dict]) -> None:
    """seconds の昇順で field 内ランク（1=最速）を付与。"""
    ordered = sorted(
        predictions.items(),
        key=lambda x: (x[1].get("seconds", 99), x[0]),
    )
    n = len(ordered)
    for i, (hn, row) in enumerate(ordered):
        row["rank"] = i + 1
        row["rank_pct"] = round((i + 1) / max(n, 1), 3)


def predict_position_flow(
    entries: list[dict],
    horse_profiles: dict[int, dict],
    tracking_results: list[dict],
    pace_prediction: dict,
    field_prev_stats: dict | None = None,
    race_info: dict | None = None,
    horse_histories: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """
    各馬の位置取りの流れ（序盤→中盤→終盤）を予測する。

    Returns:
        各馬の位置予測リスト [
            {
                "horse_number": int,
                "horse_name": str,
                "positions": {
                    "early": {"position": int, "position_norm": float},
                    "mid": {"position": int, "position_norm": float},
                    "late": {"position": int, "position_norm": float},
                },
                "flow_pattern": str,
                "stamina_concern": bool
            },
            ...
        ]
    """
    field_size = len(entries)
    pace_type = pace_prediction.get("pace_type", "ミドル")
    pace_index = pace_prediction.get("pace_index", 50)

    td_by_hn = {r["horse_number"]: r for r in tracking_results}
    early_composite = _build_early_forward_composite(
        entries,
        horse_profiles,
        td_by_hn,
        field_prev_stats=field_prev_stats,
        field_size=field_size,
        pace_type=pace_type,
    )
    mid_desire: dict[int, float] = {}
    late_desire: dict[int, float] = {}
    meta_by_hn: dict[int, dict] = {}

    for e in entries:
        hn = int(e.get("horse_number") or 0)
        if hn <= 0:
            continue
        profile = horse_profiles.get(hn, {})
        style = profile.get("style", "差し")
        pos_std = profile.get("pos_std", 0.2)
        td_data = td_by_hn.get(hn, {})
        meta_by_hn[hn] = {
            "e": e,
            "profile": profile,
            "style": style,
            "pos_std": pos_std,
            "td_data": td_data,
        }

    early_ranks = _assign_unique_positions(early_composite, field_size)

    for hn, meta in meta_by_hn.items():
        style = meta["style"]
        mid_desire[hn] = float(early_ranks[hn])
        if pace_type == "ハイ" and style in ("逃げ", "先行"):
            mid_desire[hn] += 1.1
        elif pace_type == "スロー" and style in ("差し", "追込"):
            mid_desire[hn] -= 0.6
        elif style in ("差し", "追込"):
            mid_desire[hn] -= 0.25

    mid_ranks = _assign_unique_positions(mid_desire, field_size)

    for hn, meta in meta_by_hn.items():
        style = meta["style"]
        late_desire[hn] = float(mid_ranks[hn])
        if style == "逃げ":
            late_desire[hn] += 1.4 if pace_type == "ハイ" else 0.5
        elif style == "先行":
            late_desire[hn] += 0.9 if pace_type == "ハイ" else 0.1
        elif style in ("差し", "追込"):
            if pace_type == "ハイ":
                late_desire[hn] -= 1.2
            elif pace_type == "スロー":
                late_desire[hn] += 0.5
            else:
                late_desire[hn] -= 0.8

    late_ranks = _assign_unique_positions(late_desire, field_size)

    results = []
    for hn in sorted(meta_by_hn.keys()):
        meta = meta_by_hn[hn]
        e = meta["e"]
        profile = meta["profile"]
        style = meta["style"]
        pos_std = meta["pos_std"]
        td_data = meta["td_data"]

        early_pos = early_ranks[hn]
        mid_pos = mid_ranks[hn]
        late_pos = late_ranks[hn]
        early_pos_norm = _rank_to_norm(early_pos, field_size)
        mid_pos_norm = _rank_to_norm(mid_pos, field_size)
        late_pos_norm = _rank_to_norm(late_pos, field_size)

        if late_pos < early_pos - 2:
            flow_pattern = "追い込み"
        elif late_pos < early_pos:
            flow_pattern = "差し"
        elif late_pos > early_pos + 2:
            flow_pattern = "後退"
        elif late_pos > early_pos:
            flow_pattern = "微後退"
        else:
            flow_pattern = "粘り込み"

        stamina_concern = (
            (pace_type == "ハイ" and style in ("逃げ", "先行"))
            or (pace_type == "スロー" and style in ("差し", "追込"))
        )

        t1f_norm = 0.5
        if field_prev_stats:
            t1f_norm = field_prev_stats.get("normalized_t1f", {}).get(hn, 0.5)

        distance = (race_info or {}).get("distance", 1600)
        surface = (race_info or {}).get("surface", "芝")

        stamina_data = compute_stamina_index(
            predicted_position_norm=early_pos_norm,
            pace_index=pace_index,
            pace_type=pace_type,
            distance=distance,
            surface=surface,
            t1f_norm=t1f_norm,
            profile=profile,
        )

        results.append({
            "horse_number": hn,
            "horse_name": e.get("horse_name", ""),
            "style": style,
            "positions": {
                "early": {
                    "position": early_pos,
                    "position_norm": round(early_pos_norm, 3),
                },
                "mid": {
                    "position": mid_pos,
                    "position_norm": round(mid_pos_norm, 3),
                },
                "late": {
                    "position": late_pos,
                    "position_norm": round(late_pos_norm, 3),
                },
            },
            "flow_pattern": flow_pattern,
            "stamina_concern": stamina_concern,
            "confidence": round(max(0, 1.0 - pos_std * 2), 2),
            "stamina": stamina_data,
            "allocation": {
                "early_composite": round(early_composite.get(hn, 0), 4),
                "t1f_norm": round(float(t1f_norm), 4),
                "best_norm_min": float(
                    (td_data.get("position_ability") or profile).get(
                        "best_norm_pos_min", 0.5
                    )
                ),
                "ever_led_before_final": bool(
                    (td_data.get("position_ability") or profile).get(
                        "ever_led_before_final"
                    )
                ),
                "ease_pct": round(
                    float((td_data.get("tracking_difficulty") or {}).get("ease_pct", 50)),
                    1,
                ),
            },
        })

    race_info = race_info or {}
    horse_abilities: dict[int, dict] = {}
    ability_secs: list[float] = []
    hist_map = horse_histories or {}

    for e in entries:
        hn = int(e.get("horse_number") or 0)
        if hn <= 0:
            continue
        td_data = td_by_hn.get(hn, {})
        prev = td_data.get("prev_race") or {}
        hid = str(e.get("horse_id") or td_data.get("horse_id") or "")
        prof = _horse_last3f_ability(
            hist_map.get(hid, []),
            prev_last3f=_safe_float(prev.get("last3f")),
        )
        horse_abilities[hn] = prof
        if prof["ability_sec"] > 0:
            ability_secs.append(prof["ability_sec"])

    l3f_baseline = _resolve_last3f_baseline(
        race_info, pace_prediction, ability_secs
    )
    baseline_sec = float(l3f_baseline["baseline_sec"])

    l3f_by_hn: dict[int, dict] = {}
    row_by_hn = {r["horse_number"]: r for r in results}
    for hn, row in row_by_hn.items():
        l3f_by_hn[hn] = predict_expected_last_3f(
            horse_number=hn,
            profile=horse_profiles.get(hn, {}),
            td_data=td_by_hn.get(hn, {}),
            position_flow_row=row,
            pace_prediction=pace_prediction,
            race_info=race_info,
            baseline_sec=baseline_sec,
            horse_ability=horse_abilities.get(hn, {}),
        )
    _assign_last3f_ranks(l3f_by_hn)

    for r in results:
        hn = r["horse_number"]
        r["expected_last_3f"] = l3f_by_hn.get(hn, {})

    return results
