"""
note「Pedigree investigation」系の種牡馬適性ナレッジを数値特徴量に変換する。

- データ: data/knowledge/sire_aptitude_note.json（著者観点の主観スコア、学習ラベルではない）
- 各馬: 父 + 母父 +（任意で）牝系ラインの線形ブレンド → 軸ごとの特徴量 + 距離帯との簡易整合スコア

下流のモデリングでは note_apt_* を追加特徴として concat し、
重みは学習データ側で推定するのが原則。
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# 距離帯ごとに「どの軸を重視するか」のヒューリスティック（記事の大枠に対応）
DIST_AXIS_WEIGHTS: dict[str, dict[str, float]] = {
    "sprint": {
        "eu_burst": 0.35,
        "us_speed_sustain": 0.3,
        "fast_turf": 0.2,
        "early_maturity": 0.15,
    },
    "mile": {
        "tokyo_mile_vm": 0.25,
        "us_speed_sustain": 0.25,
        "eu_burst": 0.2,
        "straight_flat_speed": 0.15,
        "fast_turf": 0.15,
    },
    "middle": {
        "stamina_mid": 0.3,
        "eu_power_stamina": 0.25,
        "kyoto_outer": 0.2,
        "us_speed_sustain": 0.15,
        "eu_burst": 0.1,
    },
    "long": {
        "zaitech_stamina": 0.3,
        "stamina_mid": 0.25,
        "eu_power_stamina": 0.2,
        "tokyo_2400_turf": 0.15,
        "straight_flat_speed": 0.1,
    },
}


def _json_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "knowledge" / "sire_aptitude_note.json"


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    with open(_json_path(), encoding="utf-8") as f:
        return json.load(f)


def axis_ids() -> tuple[str, ...]:
    b = load_bundle()
    return tuple(a["id"] for a in b.get("axes", []))


def _normalize_token(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", "", s)
    return s


def resolve_sire_name(name: str | None) -> str | None:
    if not name:
        return None
    n = _normalize_token(name)
    if not n:
        return None
    b = load_bundle()
    stallions = b.get("stallions", {})
    if n in stallions:
        return n
    aliases = b.get("aliases", {})
    if n in aliases:
        return str(aliases[n])
    lower = n.lower()
    for alt, canon in aliases.items():
        if str(alt).lower() == lower:
            return str(canon)
    return None


def resolve_broodmare_line_name(name: str | None) -> str | None:
    if not name:
        return None
    n = _normalize_token(name)
    if not n:
        return None
    b = load_bundle()
    bm = b.get("broodmare_lines", {})
    if n in bm:
        return n
    for canon in bm:
        if _normalize_token(canon) == n:
            return canon
    aliases = b.get("aliases", {})
    if n in aliases:
        c = str(aliases[n])
        if c in bm:
            return c
    lower = n.lower()
    for alt, canon in aliases.items():
        if str(alt).lower() == lower and str(canon) in bm:
            return str(canon)
    return None


def zero_vector() -> dict[str, float]:
    return {k: 0.0 for k in axis_ids()}


def vector_for_canonical(canonical: str | None) -> dict[str, float]:
    ax = zero_vector()
    if not canonical:
        return ax
    st = load_bundle().get("stallions", {}).get(canonical)
    if not st:
        return ax
    for k, v in (st.get("axes") or {}).items():
        if k in ax:
            try:
                ax[k] = float(v)
            except (TypeError, ValueError):
                pass
    return ax


def vector_for_broodmare_line(canonical: str | None) -> dict[str, float]:
    ax = zero_vector()
    if not canonical:
        return ax
    row = load_bundle().get("broodmare_lines", {}).get(canonical)
    if not row:
        return ax
    for k, v in (row.get("axes") or {}).items():
        if k in ax:
            try:
                ax[k] = float(v)
            except (TypeError, ValueError):
                pass
    return ax


def blend_vectors(
    sire: str,
    dam_sire: str,
    broodmare_line: str | None = None,
    *,
    sire_w: float = 1.0,
    bms_w: float = 0.55,
    bm_w: float = 0.35,
) -> dict[str, float]:
    vs = vector_for_canonical(resolve_sire_name(sire))
    vb = vector_for_canonical(resolve_sire_name(dam_sire))
    rbm = resolve_broodmare_line_name(broodmare_line)
    vbm = vector_for_broodmare_line(rbm) if rbm else zero_vector()
    return {
        k: sire_w * vs[k] + bms_w * vb[k] + bm_w * vbm[k]
        for k in axis_ids()
    }


def distance_bucket(meters: int) -> str:
    if meters <= 0:
        return "middle"
    if meters <= 1400:
        return "sprint"
    if meters <= 1800:
        return "mile"
    if meters <= 2200:
        return "middle"
    return "long"


def distance_fit(blended: dict[str, float], distance_m: int) -> float:
    bucket = distance_bucket(distance_m)
    wmap = DIST_AXIS_WEIGHTS[bucket]
    num = sum(blended.get(k, 0.0) * w for k, w in wmap.items())
    den = sum(abs(w) for w in wmap.values()) or 1.0
    return num / den


def zero_note_aptitude_features() -> dict[str, float]:
    """JSON と独立にキーを揃えたゼロベクトル（フォールバック用）。"""
    z = zero_vector()
    out = {"note_apt_dist_fit": 0.0, "note_apt_l2": 0.0}
    out.update({f"note_apt_{k}": 0.0 for k in z})
    return out


def compute_note_aptitude_features(
    sire: str,
    dam_sire: str,
    distance: int = 0,
    *,
    broodmare_line: str | None = None,
) -> dict[str, float]:
    """パイプライン用: 父・母父・（任意）牝系・レース距離からフラットな float 特徴量を返す。"""
    blended = blend_vectors(sire or "", dam_sire or "", broodmare_line)
    fit = distance_fit(blended, int(distance or 0))
    feats: dict[str, float] = {
        "note_apt_dist_fit": fit,
        "note_apt_l2": math.sqrt(sum(v * v for v in blended.values())),
    }
    for k, v in blended.items():
        feats[f"note_apt_{k}"] = v
    return feats


def predict_context_score(
    sire: str,
    dam_sire: str,
    *,
    venue: str = "",
    surface: str = "芝",
    distance_m: int = 0,
    going_heavy: bool = False,
    broodmare_line: str | None = None,
) -> dict[str, Any]:
    """API/デバッグ用: 開催条件をざっくり足し合わせた説明付きスコア（非確率）。"""
    blended = blend_vectors(sire, dam_sire, broodmare_line)
    score = 0.0
    hints: list[str] = []
    d = int(distance_m or 0)
    if d > 0:
        score += distance_fit(blended, d)
        hints.append(f"距離帯:{distance_bucket(d)}")
    if going_heavy:
        score += 0.4 * blended.get("wet_turf", 0.0)
        hints.append("道悪寄与")
    v = (venue or "").replace("競馬場", "")
    if "京都" in v and d >= 1800:
        score += 0.35 * blended.get("kyoto_outer", 0.0)
        score += 0.2 * blended.get("kyoto_inner_speed", 0.0)
        hints.append("京都長め")
    if "東京" in v:
        if 1900 <= d <= 2100:
            score += 0.3 * blended.get("tokyo_mile_vm", 0.0)
        if 2300 <= d <= 2500:
            score += 0.35 * blended.get("tokyo_2400_turf", 0.0)
        hints.append("東京")
    if "ダート" in surface or surface.strip() == "ダ":
        score += 0.45 * blended.get("dirt_power", 0.0)
        hints.append("ダート")
    return {
        "score": score,
        "hints": hints,
        "blended_axes": blended,
        "resolved_sire": resolve_sire_name(sire),
        "resolved_bms": resolve_sire_name(dam_sire),
        "resolved_broodmare_line": resolve_broodmare_line_name(broodmare_line),
    }


def stallion_table_rows() -> list[dict[str, Any]]:
    """種牡馬適性表（UI/CSV用）。"""
    b = load_bundle()
    axes = [a["id"] for a in b.get("axes", [])]
    rows = []
    for name, body in sorted(b.get("stallions", {}).items()):
        vec = vector_for_canonical(name)
        rows.append({
            "canonical_name": name,
            "summary_ja": body.get("summary_ja", ""),
            **{f"axis_{k}": vec[k] for k in axes},
        })
    return rows


def broodmare_table_rows() -> list[dict[str, Any]]:
    """牝系ライン×軸の適性表（UI/CSV用）。"""
    b = load_bundle()
    axes = [a["id"] for a in b.get("axes", [])]
    rows = []
    for name, body in sorted(b.get("broodmare_lines", {}).items()):
        vec = vector_for_broodmare_line(name)
        rows.append({
            "canonical_name": name,
            "kind": "broodmare_line",
            "summary_ja": body.get("summary_ja", ""),
            **{f"axis_{k}": vec[k] for k in axes},
        })
    return rows


if __name__ == "__main__":
    import pprint

    s = "ディープインパクト"
    bms = "キングカメハメハ"
    pprint.pprint(compute_note_aptitude_features(s, bms, distance=2000))
    pprint.pprint(
        compute_note_aptitude_features(s, bms, distance=2000, broodmare_line="Monevassia")
    )
    pprint.pprint(
        predict_context_score(
            s,
            bms,
            venue="京都",
            surface="芝",
            distance_m=2200,
            going_heavy=False,
        )
    )
