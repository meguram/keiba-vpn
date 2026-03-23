"""
sire_aptitude_note を 5 世代血統（horse_pedigree_5gen）に展開してブレンドする。

父系／母系:
  各祖先の (generation, position) から「母方へ進んだ世代の割合」m を算出し、
  軸ごとの係数 k で「母方経路を強める／父方経路を強める」を掛ける。
  根拠・限界は docs/NOTE_APTITUDE_5GEN.md を参照。

基礎重み: (1/2)^generation — research.pedigree_similarity と整合。
"""

from __future__ import annotations

import logging
from typing import Any

from research.sire_aptitude_note import (
    axis_ids,
    blend_vectors,
    resolve_sire_name,
    vector_for_canonical,
    zero_vector,
)

logger = logging.getLogger(__name__)

# 軸 a について、母方経路割合 m が高い祖先をどれだけ強調するか。
# mult_a(m) = 1 + k_a * (2m - 1)。m=0.5 で中立、m=1 で 1+k、m=0 で 1-k。
# 正: スタミナ・晩成・タフ寄りを「母方の祖先経路」で少し強める
# 負: 早熟・スピード・ダート寄りを「父方の祖先経路」で少し強める
MATERNAL_PATH_AXIS_K: dict[str, float] = {
    "zaitech_stamina": 0.22,
    "stamina_mid": 0.2,
    "eu_power_stamina": 0.14,
    "late_maturity": 0.12,
    "wet_turf": 0.1,
    "aus_flat_outside": 0.08,
    "tokyo_2400_turf": 0.1,
    "kyoto_outer": 0.05,
    "early_maturity": -0.14,
    "dirt_power": -0.12,
    "us_speed_sustain": -0.1,
    "fast_turf": -0.08,
    "tokyo_mile_vm": -0.1,
    "straight_flat_speed": -0.06,
    "eu_burst": -0.08,
    "kyoto_inner_speed": -0.06,
    "de_power_inside": 0.0,
}

_MULT_MIN = 0.55
_MULT_MAX = 1.45


def maternal_path_fraction(generation: int, position: int) -> float:
    """
    netkeiba 5 代血統表の (generation, position) について、
    対象馬からその祖先へ至る経路のうち「母（牝）側」に進んだ世代の割合。

    generation=g の position は、g bit の二進表現とみなし、
    bit=1 を母側、0 を父側と読む（SMARTRC の f/m 系プレフィックスと整合）。
    """
    g = int(generation)
    if g <= 0:
        return 0.5
    mask = (1 << g) - 1
    p = int(position) & mask
    ones = p.bit_count()
    return ones / g


def _axis_path_multiplier(axis_id: str, maternal_frac: float) -> float:
    k = float(MATERNAL_PATH_AXIS_K.get(axis_id, 0.0))
    m = 1.0 + k * (2.0 * maternal_frac - 1.0)
    return max(_MULT_MIN, min(_MULT_MAX, m))


def blend_note_from_5gen_pedigree(
    storage: Any,
    horse_id: str,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """
    horse_pedigree_5gen があれば 17 軸ブレンドを返す。無ければ (None, meta)。

    Returns:
        (blended_axes | None, diagnostics)
    """
    hid = (horse_id or "").strip()
    if not hid:
        return None, {"reason": "no_horse_id"}

    try:
        rec = storage.load("horse_pedigree_5gen", hid)
    except Exception as e:
        logger.debug("horse_pedigree_5gen load %s: %s", hid, e)
        rec = None

    if not rec or not rec.get("ancestors"):
        return None, {"reason": "no_pedigree_record", "horse_id": hid}

    ancestors: list[dict[str, Any]] = list(rec["ancestors"])
    axes = axis_ids()
    numer = {a: 0.0 for a in axes}
    denom = {a: 0.0 for a in axes}
    hits = 0
    total_anc = len(ancestors)

    # 同名・同一解決キーでインブリード合算しやすいよう、まず (key, gen, pos, mf) 単位で重みを足す
    for anc in ancestors:
        gen = int(anc.get("generation") or 0)
        pos = int(anc.get("position") or 0)
        name = (anc.get("name") or "").strip()
        if gen < 1 or not name:
            continue

        w0 = (0.5) ** gen
        mf = maternal_path_fraction(gen, pos)
        canon = resolve_sire_name(name)
        if not canon:
            continue
        vec = vector_for_canonical(canon)
        if not any(vec.values()):
            continue
        hits += 1

        for ax in axes:
            v = float(vec.get(ax, 0.0))
            mult = _axis_path_multiplier(ax, mf)
            w = w0 * mult
            numer[ax] += w * v
            denom[ax] += w

    if hits == 0 or sum(denom.values()) <= 0:
        return None, {
            "reason": "no_resolved_stallions_in_pedigree",
            "horse_id": hid,
            "ancestor_count": total_anc,
            "hits": 0,
        }

    out: dict[str, float] = {}
    for ax in axes:
        d = denom[ax]
        out[ax] = (numer[ax] / d) if d > 0 else 0.0

    meta = {
        "horse_id": hid,
        "ancestor_count": total_anc,
        "resolved_ancestors_used": hits,
        "maternal_path_model": "bit_count(position,g)/g on dam=1 edges",
    }
    return out, meta


def blended_note_for_race_horse(
    storage: Any,
    horse_id: str,
    sire: str,
    dam_sire: str,
    broodmare_line: str | None,
    *,
    blend_mode: str = "shallow",
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    blend_mode:
      - shallow: 父 + 母父 + 牝系（従来）
      - 5gen: 5 世代が取れたらそれを採用、否则 shallow
    """
    mode = (blend_mode or "shallow").strip().lower()
    diag: dict[str, Any] = {"blend_mode_requested": mode}

    if mode in ("5gen", "fivegen", "full"):
        b5, m5 = blend_note_from_5gen_pedigree(storage, horse_id)
        diag["pedigree_5gen"] = m5
        if b5 is not None:
            diag["blend_mode_used"] = "5gen"
            return b5, diag

    diag["blend_mode_used"] = "shallow"
    blended = blend_vectors(sire, dam_sire, broodmare_line)
    return blended, diag
