"""
6頭の重要祖先位置から馬の適性ベクトルを算出する。

祖先位置:
  父(1,0), 母父母父(4,10), 母父母母父(5,22),
  母母父(3,6), 母母母父(4,14), 母母母母父(5,30)

父は paternal 統計、他5頭は maternal 統計を参照。

optimize_weights: 過去レース結果との相関を最大化するウェイトを探索。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from research.sire_factor_stats import (
    AXIS_IDS,
    load_sire_factor_stats,
    load_snapshot_data,
    _collect_race_features,
)

logger = logging.getLogger(__name__)

ANCESTOR_SLOTS: list[dict[str, Any]] = [
    {"key": "father",    "gen": 1, "pos": 0,  "label_ja": "父",       "side": "paternal", "default_weight": 0.30},
    {"key": "mf_mf",     "gen": 4, "pos": 10, "label_ja": "母父母父",   "side": "maternal", "default_weight": 0.14},
    {"key": "mf_mm_f",   "gen": 5, "pos": 22, "label_ja": "母父母母父", "side": "maternal", "default_weight": 0.14},
    {"key": "mm_f",      "gen": 3, "pos": 6,  "label_ja": "母母父",     "side": "maternal", "default_weight": 0.14},
    {"key": "mmm_f",     "gen": 4, "pos": 14, "label_ja": "母母母父",   "side": "maternal", "default_weight": 0.14},
    {"key": "mmmm_f",    "gen": 5, "pos": 30, "label_ja": "母母母母父", "side": "maternal", "default_weight": 0.14},
]

DEFAULT_WEIGHTS = {s["key"]: s["default_weight"] for s in ANCESTOR_SLOTS}


def _find_ancestor(ancestors: list[dict], gen: int, pos: int) -> dict | None:
    for a in ancestors:
        if a.get("generation") == gen and a.get("position") == pos:
            return a
    return None


def zero_vector() -> dict[str, float]:
    return {k: 0.0 for k in AXIS_IDS}


def compute_horse_aptitude(
    storage: Any,
    horse_id: str,
    *,
    weights: dict[str, float] | None = None,
    stats_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    馬の適性ベクトルを6祖先のブレンドで算出。

    Returns:
        {
            "factors": {axis_id: float, ...},
            "ancestors": [{slot info + resolved sire data}, ...],
            "weights_used": {...},
        }
    """
    stats = stats_data or load_sire_factor_stats()
    sires_db = stats.get("sires", {})

    ped = storage.load("horse_pedigree_5gen", horse_id)
    ancestors_list = ped.get("ancestors", []) if ped else []

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)

    w_sum = sum(w.values())
    if w_sum <= 0:
        w_sum = 1.0

    blended = zero_vector()
    ancestor_details = []

    for slot in ANCESTOR_SLOTS:
        sk = slot["key"]
        weight = w.get(sk, 0.0) / w_sum

        anc = _find_ancestor(ancestors_list, slot["gen"], slot["pos"])
        sire_id = (anc.get("horse_id") or "").strip() if anc else ""
        sire_name = (anc.get("name") or "").strip() if anc else ""

        sire_data = sires_db.get(sire_id, {})
        side = slot["side"]
        side_data = sire_data.get(side, {})
        axes = side_data.get("axes", {})
        sample_size = side_data.get("sample_size", 0)

        for k in AXIS_IDS:
            blended[k] += axes.get(k, 0.0) * weight

        ancestor_details.append({
            "slot": sk,
            "label_ja": slot["label_ja"],
            "sire_id": sire_id,
            "sire_name": sire_name or sire_id,
            "side": side,
            "sample_size": sample_size,
            "found_in_db": bool(sire_data),
            "axes": {k: round(axes.get(k, 0.0), 4) for k in AXIS_IDS},
        })

    factors = {k: round(v, 4) for k, v in blended.items()}

    return {
        "factors": factors,
        "ancestors": ancestor_details,
        "weights_used": {k: round(v / w_sum, 4) for k, v in w.items()},
    }


# ---------------------------------------------------------------------------
# ウェイト自動チューニング
# ---------------------------------------------------------------------------

SLOT_KEYS = [s["key"] for s in ANCESTOR_SLOTS]


def _compute_aptitude_fast(
    ancestors_list: list[dict],
    weights_array: np.ndarray,
    sires_db: dict,
) -> np.ndarray:
    """高速版: pedigree ancestors + 重み配列 → 適性ベクトル (numpy)。"""
    w_sum = float(np.sum(weights_array))
    if w_sum <= 0:
        w_sum = 1.0
    vec = np.zeros(len(AXIS_IDS))
    for i, slot in enumerate(ANCESTOR_SLOTS):
        weight = weights_array[i] / w_sum
        anc = _find_ancestor(ancestors_list, slot["gen"], slot["pos"])
        if not anc:
            continue
        sid = (anc.get("horse_id") or "").strip()
        sire_data = sires_db.get(sid, {})
        side_data = sire_data.get(slot["side"], {})
        axes = side_data.get("axes", {})
        for j, k in enumerate(AXIS_IDS):
            vec[j] += axes.get(k, 0.0) * weight
    return vec


def _build_training_data(
    storage: Any,
    stats_data: dict,
    *,
    race_ids: list[str] | None = None,
    max_horses: int = 3000,
) -> list[tuple[list[dict], list[dict]]]:
    """
    チューニング用データセットを構築。
    Returns: [(ancestors_list, race_features), ...]
    """
    sires_db = stats_data.get("sires", {})

    if race_ids:
        horse_ids = set()
        for rid in race_ids:
            for src in ("race_shutuba", "race_result"):
                data = storage.load(src, rid)
                if data and data.get("entries"):
                    for e in data["entries"]:
                        hid = (e.get("horse_id") or "").strip()
                        if hid:
                            horse_ids.add(hid)
                    break
    else:
        snap = load_snapshot_data(storage)
        if snap:
            peds_raw, hrs_raw = snap
        else:
            return []

        hr_map = {}
        for hr in hrs_raw:
            hid = hr.get("horse_id", "")
            if hid and hr.get("race_history"):
                hr_map[hid] = hr["race_history"]

        dataset = []
        for ped in peds_raw:
            hid = ped.get("horse_id", "")
            ancestors = ped.get("ancestors", [])
            if not ancestors or hid not in hr_map:
                continue
            has_sire_data = False
            for slot in ANCESTOR_SLOTS:
                anc = _find_ancestor(ancestors, slot["gen"], slot["pos"])
                if anc:
                    sid = (anc.get("horse_id") or "").strip()
                    if sid in sires_db:
                        has_sire_data = True
                        break
            if not has_sire_data:
                continue
            features = _collect_race_features(hr_map[hid])
            if features:
                dataset.append((ancestors, features))
            if len(dataset) >= max_horses:
                break
        return dataset

    dataset = []
    for hid in list(horse_ids)[:max_horses]:
        ped = storage.load("horse_pedigree_5gen", hid)
        if not ped or not ped.get("ancestors"):
            continue
        hr = storage.load("horse_result", hid)
        if not hr or not hr.get("race_history"):
            continue
        features = _collect_race_features(hr["race_history"])
        if features:
            dataset.append((ped["ancestors"], features))
    return dataset


def optimize_weights(
    storage: Any,
    *,
    race_ids: list[str] | None = None,
    stats_data: dict | None = None,
    max_horses: int = 3000,
) -> dict[str, Any]:
    """
    過去レース結果（複勝率）との相関を最大化する6祖先ウェイトを探索。

    Parameters:
        race_ids: 特定レースの出走馬で最適化（None=グローバル）
        stats_data: 統計データ（None=自動ロード）
        max_horses: 最大馬数

    Returns:
        {"optimized_weights": {...}, "correlation": float, "sample_size": int, ...}
    """
    t0 = time.time()
    stats = stats_data or load_sire_factor_stats(storage=storage)
    sires_db = stats.get("sires", {})

    dataset = _build_training_data(
        storage, stats, race_ids=race_ids, max_horses=max_horses
    )
    if len(dataset) < 10:
        return {
            "error": f"データ不足: {len(dataset)}馬（最低10馬必要）",
            "optimized_weights": DEFAULT_WEIGHTS,
            "correlation": 0.0,
            "sample_size": len(dataset),
            "elapsed_sec": round(time.time() - t0, 2),
        }

    logger.info("チューニング開始: %d 馬のデータ", len(dataset))

    def _objective(w_raw: np.ndarray) -> float:
        w = np.exp(w_raw)
        w = w / np.sum(w)
        scores = []
        actuals = []
        for ancestors, features in dataset:
            vec = _compute_aptitude_fast(ancestors, w, sires_db)
            norm = float(np.linalg.norm(vec))
            for f in features:
                scores.append(norm)
                actuals.append(1.0 if f["top3"] else 0.0)
        if not scores or np.std(scores) < 1e-10:
            return 0.0
        from scipy.stats import spearmanr
        corr, _ = spearmanr(scores, actuals)
        return -corr if not math.isnan(corr) else 0.0

    x0 = np.log(np.array([s["default_weight"] for s in ANCESTOR_SLOTS]))

    result = minimize(
        _objective, x0,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 0.01, "fatol": 1e-5},
    )

    opt_w = np.exp(result.x)
    opt_w = opt_w / np.sum(opt_w)
    opt_weights = {k: round(float(opt_w[i]), 4) for i, k in enumerate(SLOT_KEYS)}

    final_corr = -result.fun if result.fun else 0.0
    total_samples = sum(len(f) for _, f in dataset)
    elapsed = time.time() - t0

    logger.info("チューニング完了: corr=%.4f, weights=%s (%.1f秒)",
                final_corr, opt_weights, elapsed)

    return {
        "optimized_weights": opt_weights,
        "correlation": round(final_corr, 4),
        "sample_size": len(dataset),
        "total_races": total_samples,
        "elapsed_sec": round(elapsed, 2),
        "scope": "race" if race_ids else "global",
    }
