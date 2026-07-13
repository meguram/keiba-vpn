"""めぐ指数のレース向け予測（adj_time ベース・1点=0.1秒厳守）。"""

from __future__ import annotations

import re
from typing import Any, Optional

from src.pipeline.megu_index.common import (
    MEGU_POINTS_PER_SEC,
    WEIGHTS_B,
    adjusted_time_to_megu,
    dist_band,
    is_major_condition_change,
    megu_to_adjusted_time,
)
from src.pipeline.megu_index.condition_weights import (
    apply_condition_weights,
    load_condition_weights,
)
from src.pipeline.megu_index.predict_params import PredictTuning, load_predict_params

_BASE_WEIGHT_MALE = 55.0
_BASE_WEIGHT_FEMALE = 53.0

# 条件転換: 低サンプル・極端値は適用しない
MIN_COND_TRANSFER_SAMPLES = 30
MAX_COND_TRANSFER_SEC = 1.0

# 想定指数の表示レンジ（過去走平均の異常値ガード）
MEGU_PREDICT_FLOOR = 50.0
MEGU_PREDICT_CEIL = 160.0


def base_weight_kg(sex_age: str | None) -> float:
    if sex_age and re.match(r"^牝", str(sex_age).strip()):
        return _BASE_WEIGHT_FEMALE
    return _BASE_WEIGHT_MALE


def weight_megu_delta(
    jockey_weight: float | None,
    sex_age: str | None,
    distance: int,
    beta_weight: float,
) -> float | None:
    """今回斤量での指数補正（点）。重いほど下がる。"""
    if jockey_weight is None or beta_weight == 0:
        return None
    weight_dev = float(jockey_weight) - base_weight_kg(sex_age)
    dist_scale = float(distance) / 2000.0
    delta_weight_sec = beta_weight * weight_dev * dist_scale
    return round(-delta_weight_sec * MEGU_POINTS_PER_SEC, 1)


def weight_sec_delta(
    jockey_weight: float | None,
    sex_age: str | None,
    distance: int,
    beta_weight: float,
) -> float:
    """今回斤量での走破タイム影響（秒）。"""
    if jockey_weight is None or beta_weight == 0:
        return 0.0
    weight_dev = float(jockey_weight) - base_weight_kg(sex_age)
    return beta_weight * weight_dev * (float(distance) / 2000.0)


def _normalize_weights(n: int, weights: list[float]) -> list[float]:
    w = weights[:n]
    s = sum(w)
    return [x / s for x in w] if s > 0 else [1.0 / n] * n


def weighted_ability_adj_time(
    history: list[dict[str, Any]],
    max_races: int = 5,
    weights: list[float] | None = None,
) -> tuple[float | None, float | None]:
    """
    過去走から加重平均 adjusted_time（秒）を返す。

    Returns:
        (ability_adj_sec, base_megu_at_target_par) — base_megu は par 未指定時 None
    """
    wts = weights or WEIGHTS_B
    valid = []
    for h in history[:max_races]:
        mi = h.get("megu_index")
        pt = h.get("par_time_sec")
        if mi is None or pt is None:
            continue
        try:
            adj = megu_to_adjusted_time(float(mi), float(pt))
            valid.append(adj)
        except (TypeError, ValueError):
            continue
    if not valid:
        return None, None

    nw = _normalize_weights(len(valid), wts)
    ability = sum(a * w for a, w in zip(valid, nw))
    return round(ability, 3), None


def _clamp_predict_megu(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(MEGU_PREDICT_FLOOR, min(MEGU_PREDICT_CEIL, float(value))), 1)


def _weighted_base_at_target_par(
    valid_hist: list[dict[str, Any]],
    nw: list[float],
    par_time_target: float,
    *,
    tuning: PredictTuning | None = None,
) -> tuple[float, float | None]:
    """
    過去走から base_megu を算出（秒空間ブレンド + bias。1点=0.1秒厳守）。
    """
    t = tuning or PredictTuning()
    raw_megu = sum(r["_megu"] * w for r, w in zip(valid_hist, nw))

    adj_terms: list[tuple[float, float]] = []
    for r, w in zip(valid_hist, nw):
        pt = r.get("par_time_sec")
        if pt is None:
            continue
        try:
            adj_terms.append((megu_to_adjusted_time(r["_megu"], float(pt)), w))
        except (TypeError, ValueError):
            continue

    if adj_terms:
        ability_par = sum(a * w for a, w in adj_terms)
    else:
        ability_par = megu_to_adjusted_time(raw_megu, par_time_target)

    raw_adj = megu_to_adjusted_time(raw_megu, par_time_target)
    blend = max(0.0, min(1.0, float(t.par_blend)))
    blended_adj = blend * ability_par + (1.0 - blend) * raw_adj - float(t.ability_bias_sec)
    base_megu = round(adjusted_time_to_megu(blended_adj, par_time_target), 1)
    return base_megu, round(blended_adj, 3)


def predict_megu_scores(
    history: list[dict[str, Any]],
    *,
    par_time_target: float,
    surface_target: str,
    distance_target: int,
    jockey_weight: float | None,
    sex_age: str | None,
    beta_weight: float,
    transfer_map: dict[tuple[str, str, str, str], dict[str, float]],
    max_races: int = 5,
    weights: list[float] | None = None,
    condition_weights: dict[str, float] | None = None,
    tuning: PredictTuning | None = None,
) -> dict[str, Any]:
    """
    未確定レース向け megu 予測。

    想定指数 (megu_final) は:
      1. 過去走 megu を各走 par で補正済みタイムに戻し加重平均
      2. 今回レースの par_time_target に換算 (base_megu)
      3. 芝↔ダート・距離大幅変更の条件転換を加算
      4. 今回斤量の補正 (weight_megu_delta) を加算

    実測めぐ指数と同様、ペース・馬場・斤量・レベル補正後の能力を
    今回条件のスケールで比較できる値とする。
    """
    wts = weights or load_predict_params().history_weights or WEIGHTS_B
    cw = condition_weights if condition_weights is not None else load_condition_weights()
    tune = tuning if tuning is not None else load_predict_params().tuning
    valid_hist = []
    for h in history[:max_races]:
        mi = h.get("megu_index")
        if mi is None:
            continue
        try:
            valid_hist.append({**h, "_megu": float(mi)})
        except (TypeError, ValueError):
            continue

    if not valid_hist:
        return {
            "base_megu": None,
            "megu_adjusted": None,
            "megu_final": None,
            "ability_adj_sec": None,
            "condition_delta_sec": 0.0,
            "weight_megu_delta": None,
            "condition_change": {"type": "none", "label": None},
            "condition_weight_multipliers": [],
        }

    nw, cond_mults = apply_condition_weights(
        wts,
        valid_hist,
        surface_target=surface_target,
        distance_target=distance_target,
        condition_weights=cw,
    )
    base_megu, ability_adj = _weighted_base_at_target_par(
        valid_hist, nw, par_time_target, tuning=tune,
    )

    cond_change: dict[str, Any] = {"type": "none", "label": None}
    cond_megu_pt = 0.0

    prev = valid_hist[0]
    sf = str(prev.get("surface") or "")
    st = str(surface_target)
    df_dist = int(prev.get("distance") or 0)
    dt_dist = int(distance_target)
    db_from = dist_band(df_dist)
    db_to = dist_band(dt_dist)

    if is_major_condition_change(sf, st, df_dist, dt_dist):
        change_type = (
            "both" if sf != st and abs(dt_dist - df_dist) >= 600
            else ("surface" if sf != st else "distance")
        )
        label = f"{sf}→{st}" if sf != st else (
            "距離大幅短縮" if dt_dist - df_dist <= -600 else "距離大幅延長"
        )
        cond_change = {
            "type": change_type,
            "label": label,
            "surface_from": sf,
            "surface_to": st,
            "dist_band_from": db_from,
            "dist_band_to": db_to,
            "dist_change": dt_dist - df_dist,
            "delta_mean": None,
            "delta_std": None,
            "transfer_sample_count": 0,
        }
        tr_key = (sf, db_from, st, db_to)
        if tr_key in transfer_map:
            tr = transfer_map[tr_key]
            sample_count = int(tr.get("sample_count") or 0)
            cond_change["transfer_sample_count"] = sample_count
            if sample_count >= MIN_COND_TRANSFER_SAMPLES:
                cond_delta_sec = float(tr.get("delta_sec") or tr.get("delta_mean") or 0.0)
                cond_delta_sec = max(
                    -MAX_COND_TRANSFER_SEC,
                    min(MAX_COND_TRANSFER_SEC, cond_delta_sec),
                )
                cond_megu_pt = cond_delta_sec * MEGU_POINTS_PER_SEC * float(tune.transfer_strength)
                cond_change["delta_mean"] = round(cond_megu_pt, 2)
                cond_change["delta_std"] = (
                    round(float(tr["delta_std"]) * MEGU_POINTS_PER_SEC, 2)
                    if tr.get("delta_std") is not None else None
                )

    w_delta = weight_megu_delta(jockey_weight, sex_age, distance_target, beta_weight)
    megu_adjusted = _clamp_predict_megu(base_megu + cond_megu_pt)
    megu_final = _clamp_predict_megu(base_megu + cond_megu_pt + (w_delta or 0.0))

    return {
        "base_megu": base_megu,
        "megu_adjusted": megu_adjusted,
        "megu_final": megu_final,
        "ability_adj_sec": ability_adj,
        "condition_delta_sec": round(cond_megu_pt / MEGU_POINTS_PER_SEC, 3),
        "weight_megu_delta": w_delta,
        "condition_change": cond_change,
        "condition_weight_multipliers": [round(m, 3) for m in cond_mults],
    }


# 後方互換
def megu_final_score(
    megu_adjusted: float | None,
    jockey_weight: float | None,
    sex_age: str | None,
    distance: int,
    beta_weight: float,
) -> float | None:
    if megu_adjusted is None:
        return None
    w_delta = weight_megu_delta(jockey_weight, sex_age, distance, beta_weight)
    if w_delta is None:
        return megu_adjusted
    return _clamp_predict_megu(megu_adjusted + w_delta)


def megu_margin_sec(megu_a: float | None, megu_b: float | None) -> float | None:
    """指数差 → 想定タイム差（秒）。megu_a が高いほど速い。"""
    if megu_a is None or megu_b is None:
        return None
    return round((float(megu_a) - float(megu_b)) / MEGU_POINTS_PER_SEC, 1)
