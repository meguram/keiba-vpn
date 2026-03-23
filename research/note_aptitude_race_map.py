"""
任意 race_id の出走馬を、sire_aptitude_note.json のブレンドベクトルから
3次元（パワー / 欧州瞬発 / TS）に射影し、フィールド内でクラスタリングする。

- パワー: ダート・欧州パワー・ドイツイン・中距離スタミナ・在来スタミナ寄り
- 欧州瞬発: eu_burst + 京都内回り型（記事上の欧州キレ）
- TS: 高速良芝・直線平坦・米型持続スピード・東京VM型（タイム・スピード指数の素地）

いずれも著者主観スコアの線形結合であり、netkeiba の実タイム指数ではない。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from research.note_aptitude_5gen import blended_note_for_race_horse
from research.sire_aptitude_note import (
    resolve_broodmare_line_name,
    resolve_sire_name,
)

logger = logging.getLogger(__name__)

# 各次元は -1〜1 付近の加重平均（元軸が -1〜1）
POWER_WEIGHTS: dict[str, float] = {
    "dirt_power": 0.22,
    "eu_power_stamina": 0.22,
    "de_power_inside": 0.20,
    "stamina_mid": 0.18,
    "zaitech_stamina": 0.18,
}

EU_BURST_WEIGHTS: dict[str, float] = {
    "eu_burst": 0.55,
    "kyoto_inner_speed": 0.45,
}

# TS = Time / speed index 素地（記事・軸ラベルからタイムが乗りやすい型）
TS_WEIGHTS: dict[str, float] = {
    "fast_turf": 0.28,
    "straight_flat_speed": 0.24,
    "us_speed_sustain": 0.30,
    "tokyo_mile_vm": 0.18,
}


def _weighted_axis(blended: dict[str, float], weights: dict[str, float]) -> float:
    s = sum(w for w in weights.values()) or 1.0
    return sum(float(blended.get(k, 0.0)) * w for k, w in weights.items()) / s


def blended_to_power_eu_ts(blended: dict[str, float]) -> dict[str, float]:
    return {
        "power": round(_weighted_axis(blended, POWER_WEIGHTS), 6),
        "eu_burst": round(_weighted_axis(blended, EU_BURST_WEIGHTS), 6),
        "ts": round(_weighted_axis(blended, TS_WEIGHTS), 6),
    }


def _enrich_pedigree(storage: Any, entry: dict) -> dict[str, str]:
    sire = (entry.get("sire") or "").strip()
    dam_sire = (entry.get("dam_sire") or "").strip()
    hid = (entry.get("horse_id") or "").strip()
    if (not sire or not dam_sire) and hid:
        try:
            hr = storage.load("horse_result", hid)
        except Exception as e:
            logger.debug("horse_result load %s: %s", hid, e)
            hr = None
        if hr:
            if not sire:
                sire = (hr.get("sire") or "").strip()
            if not dam_sire:
                dam_sire = (hr.get("dam_sire") or "").strip()
    return {"sire": sire, "dam_sire": dam_sire}


def load_race_entries(storage: Any, race_id: str) -> tuple[list[dict], str | None]:
    """出馬表優先。無ければ結果から馬一覧し、血統は horse_result で補完。"""
    rid = (race_id or "").strip()
    if not rid:
        return [], None

    card = storage.load("race_shutuba", rid)
    if card and card.get("entries"):
        return list(card["entries"]), "race_shutuba"

    res = storage.load("race_result", rid)
    if res and res.get("entries"):
        out = []
        for e in res["entries"]:
            if not e.get("horse_id"):
                continue
            merged = dict(e)
            ped = _enrich_pedigree(storage, merged)
            merged["sire"] = ped["sire"]
            merged["dam_sire"] = ped["dam_sire"]
            out.append(merged)
        return out, "race_result+horse_result"

    return [], None


def _cluster_labels_3d(coords: list[dict[str, float]]) -> list[int]:
    n = len(coords)
    if n == 0:
        return []
    if n == 1:
        return [0]
    X = np.array([[c["power"], c["eu_burst"], c["ts"]] for c in coords], dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(4, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw = km.fit_predict(Xs)
    # クラスタを「平均パワー」高い順に 0..k-1 へ並べ替え
    by_cluster: dict[int, list[float]] = {}
    for i, lab in enumerate(raw):
        by_cluster.setdefault(int(lab), []).append(X[i, 0])
    avg_power = {c: float(np.mean(v)) for c, v in by_cluster.items()}
    order = sorted(avg_power.keys(), key=lambda c: -avg_power[c])
    remap = {old: new for new, old in enumerate(order)}
    return [remap[int(raw[i])] for i in range(n)]


def _group_tag(p: float, e: float, t: float) -> str:
    """象限風の短文ラベル（閾値は 0）。"""
    hi = lambda x: x >= 0
    parts = []
    parts.append("パワー高" if hi(p) else "パワー低")
    parts.append("欧州瞬発高" if hi(e) else "欧州瞬発低")
    parts.append("TS高" if hi(t) else "TS低")
    return "・".join(parts)


def build_race_note_aptitude_map(
    storage: Any,
    race_id: str,
    *,
    broodmare_line_by_horse_id: dict[str, str] | None = None,
    distance_m: int = 0,
    blend_mode: str = "shallow",
) -> dict[str, Any]:
    """
    race_id の全出走馬について 3次元スコアとグループ（最大4）を返す。

    blend_mode:
      - shallow — 父・母父・牝系のみ（従来）
      - 5gen — horse_pedigree_5gen があれば 5 世代＋父／母経路重み、無ければ shallow
    """
    entries, source = load_race_entries(storage, race_id)
    entries = sorted(entries, key=lambda e: int(e.get("horse_number") or 999))

    dist = int(distance_m or 0)
    if not dist:
        card0 = storage.load("race_shutuba", race_id.strip())
        if card0:
            dist = int(card0.get("distance") or 0)
        if not dist:
            res0 = storage.load("race_result", race_id.strip())
            if res0:
                dist = int(res0.get("distance") or 0)

    bm_map = broodmare_line_by_horse_id or {}

    horses_raw: list[dict[str, Any]] = []
    coords: list[dict[str, float]] = []

    for ent in entries:
        hid = (ent.get("horse_id") or "").strip()
        ped = _enrich_pedigree(storage, ent)
        sire = ped["sire"]
        dam_sire = ped["dam_sire"]
        bm = bm_map.get(hid)
        blended, blend_diag = blended_note_for_race_horse(
            storage,
            hid,
            sire,
            dam_sire,
            bm,
            blend_mode=blend_mode,
        )
        pe = blended_to_power_eu_ts(blended)
        coords.append(pe)
        horses_raw.append({
            "horse_number": int(ent.get("horse_number") or 0),
            "bracket_number": int(ent.get("bracket_number") or 0),
            "horse_id": hid,
            "horse_name": (ent.get("horse_name") or "").strip(),
            "sire": sire,
            "dam_sire": dam_sire,
            "broodmare_line": bm or None,
            "resolved_sire": resolve_sire_name(sire),
            "resolved_bms": resolve_sire_name(dam_sire),
            "resolved_broodmare_line": resolve_broodmare_line_name(bm) if bm else None,
            "blend_mode_used": blend_diag.get("blend_mode_used"),
            "pedigree_5gen": blend_diag.get("pedigree_5gen"),
            "power": pe["power"],
            "eu_burst": pe["eu_burst"],
            "ts": pe["ts"],
            "aptitude_tag": _group_tag(pe["power"], pe["eu_burst"], pe["ts"]),
        })

    labels = _cluster_labels_3d(coords)
    for i, h in enumerate(horses_raw):
        h["group_id"] = labels[i] if i < len(labels) else 0

    # グループ要約
    groups: dict[int, list[dict[str, Any]]] = {}
    for h in horses_raw:
        gid = int(h["group_id"])
        groups.setdefault(gid, []).append(h)

    group_summary = []
    for gid in sorted(groups.keys()):
        members = groups[gid]
        npow = np.mean([m["power"] for m in members])
        neu = np.mean([m["eu_burst"] for m in members])
        nts = np.mean([m["ts"] for m in members])
        group_summary.append({
            "group_id": gid,
            "count": len(members),
            "centroid": {
                "power": round(float(npow), 4),
                "eu_burst": round(float(neu), 4),
                "ts": round(float(nts), 4),
            },
            "horse_names": [m["horse_name"] for m in members],
        })

    return {
        "race_id": race_id.strip(),
        "source": source,
        "distance_m": dist,
        "meta": {
            "knowledge": "data/knowledge/sire_aptitude_note.json",
            "blend_mode_requested": blend_mode,
            "blend_doc_5gen": "docs/NOTE_APTITUDE_5GEN.md",
            "axes_ja": {
                "power": "パワー（ダート・欧州パワー・ドイツイン・中距離S・在来S の加重）",
                "eu_burst": "欧州瞬発（eu_burst・京都内回り型）",
                "ts": "TS素地（高速芝・直平坦・米持続・東京VM の加重。実指数ではない）",
            },
            "weights": {
                "power": POWER_WEIGHTS,
                "eu_burst": EU_BURST_WEIGHTS,
                "ts": TS_WEIGHTS,
            },
        },
        "horses": horses_raw,
        "groups": group_summary,
    }


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scraper.storage import HybridStorage

    rid = ""
    blend = "shallow"
    for a in sys.argv[1:]:
        if a in ("--5gen", "--5g"):
            blend = "5gen"
        elif not rid and not a.startswith("-"):
            rid = a
    if not rid:
        print("使い方: python -m research.note_aptitude_race_map <race_id> [--5gen]")
        sys.exit(1)
    st = HybridStorage()
    out = build_race_note_aptitude_map(st, rid, blend_mode=blend)
    print(json.dumps(out, ensure_ascii=False, indent=2))
