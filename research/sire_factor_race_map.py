"""
レース出走馬の種牡馬因子適性 3D マップを構築する。

build_race_note_aptitude_map の代替として、実レースデータ統計ベースの
適性ベクトルを使用し、ユーザー選択可能な3軸で3D散布図を生成する。

比較機能:
  - 同条件上位馬マッピング (surface + distance, 馬場別)
  - 直近2週間同コースマッピング
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from research.sire_factor_aptitude import (
    ANCESTOR_SLOTS,
    DEFAULT_WEIGHTS,
    _find_ancestor,
    compute_horse_aptitude,
)
from research.sire_factor_stats import AXES, AXIS_IDS, load_sire_factor_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 血統スナップショットのメモリキャッシュ (horse_id → 6祖先slim dict)
# ---------------------------------------------------------------------------
_ped_cache: dict[str, dict[str, dict]] = {}
_ped_cache_lock = threading.Lock()
_ped_cache_ts: float = 0.0
_PED_CACHE_TTL = 3600  # 1時間

_LOCAL_PED_SNAPSHOT = Path("data/research/_ped_snapshot_cache.jsonl.gz")


def _build_slim_ancestors(ancestors: list[dict]) -> dict[str, dict]:
    """6スロットの祖先情報だけを抽出。"""
    slim: dict[str, dict] = {}
    for slot in ANCESTOR_SLOTS:
        anc = _find_ancestor(ancestors, slot["gen"], slot["pos"])
        if anc:
            slim[slot["key"]] = {
                "horse_id": (anc.get("horse_id") or "").strip(),
                "name": (anc.get("name") or "").strip(),
            }
    return slim


def _load_ped_cache(storage: Any) -> dict[str, dict[str, dict]]:
    """GCSスナップショットから血統キャッシュを構築/更新。"""
    global _ped_cache, _ped_cache_ts
    now = _time.time()
    if _ped_cache and (now - _ped_cache_ts) < _PED_CACHE_TTL:
        return _ped_cache

    with _ped_cache_lock:
        if _ped_cache and (now - _ped_cache_ts) < _PED_CACHE_TTL:
            return _ped_cache

        t0 = _time.time()
        cache: dict[str, dict[str, dict]] = {}

        _SNAP_DISK_TTL = 86400
        snap_ok = False
        if _LOCAL_PED_SNAPSHOT.exists():
            snap_age = now - _LOCAL_PED_SNAPSHOT.stat().st_mtime
            if snap_age <= _SNAP_DISK_TTL:
                snap_ok = True
            else:
                try:
                    _LOCAL_PED_SNAPSHOT.unlink()
                except OSError:
                    pass
        if snap_ok:
            try:
                raw = gzip.decompress(_LOCAL_PED_SNAPSHOT.read_bytes())
                for line in raw.decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    hid = rec.get("horse_id", "")
                    ancestors = rec.get("ancestors", [])
                    if hid and ancestors:
                        cache[hid] = _build_slim_ancestors(ancestors)
                if cache:
                    _ped_cache = cache
                    _ped_cache_ts = now
                    logger.info("ped_cache: ローカルから %d 馬ロード (%.1fs)",
                                len(cache), _time.time() - t0)
                    return _ped_cache
            except Exception as e:
                logger.warning("ped_cache: ローカル読込失敗: %s", e)

        # 2. GCSからダウンロード
        try:
            from research.sire_factor_stats import _download_snapshot_blob
            ped_gz = _download_snapshot_blob(storage, "horse_pedigree_5gen.jsonl.gz")
            if ped_gz:
                _LOCAL_PED_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
                _LOCAL_PED_SNAPSHOT.write_bytes(ped_gz)
                raw = gzip.decompress(ped_gz)
                for line in raw.decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    hid = rec.get("horse_id", "")
                    ancestors = rec.get("ancestors", [])
                    if hid and ancestors:
                        cache[hid] = _build_slim_ancestors(ancestors)
        except Exception as e:
            logger.warning("ped_cache: GCSロード失敗: %s", e)

        if cache:
            _ped_cache = cache
            _ped_cache_ts = now
            logger.info("ped_cache: GCSから %d 馬ロード (%.1fs)",
                        len(cache), _time.time() - t0)
        else:
            logger.warning("ped_cache: データなし")

        return _ped_cache


def _compute_apt_cached(
    hid: str,
    weights: dict[str, float] | None,
    stats_data: dict,
    ped_cache: dict[str, dict[str, dict]],
) -> dict[str, float]:
    """メモリキャッシュから高速aptitude計算。GCS I/O不要。"""
    slim = ped_cache.get(hid)
    if not slim:
        return {}

    sires_db = stats_data.get("sires", {})
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)
    w_sum = sum(w.values()) or 1.0

    blended = {k: 0.0 for k in AXIS_IDS}
    for slot in ANCESTOR_SLOTS:
        sk = slot["key"]
        weight = w.get(sk, 0.0) / w_sum
        anc = slim.get(sk)
        if not anc:
            continue
        sid = anc["horse_id"]
        sire_data = sires_db.get(sid, {})
        side_data = sire_data.get(slot["side"], {})
        axes = side_data.get("axes", {})
        for k in AXIS_IDS:
            blended[k] += axes.get(k, 0.0) * weight

    return {k: round(v, 4) for k, v in blended.items()}


def load_race_entries(storage: Any, race_id: str) -> tuple[list[dict], str | None]:
    """出馬表優先、無ければ結果から。horse_result で血統補完。"""
    rid = (race_id or "").strip()
    if not rid:
        return [], None

    card = storage.load("race_shutuba", rid)
    if card and card.get("entries"):
        return list(card["entries"]), "race_shutuba"

    res = storage.load("race_result", rid)
    if res and res.get("entries"):
        entries_need_hr = []
        for e in res["entries"]:
            if not e.get("horse_id"):
                continue
            hid = e["horse_id"].strip()
            sire = (e.get("sire") or "").strip()
            dam_sire = (e.get("dam_sire") or "").strip()
            entries_need_hr.append((dict(e), hid, not sire, not dam_sire))

        hids_to_load = [hid for _, hid, ns, nd in entries_need_hr if ns or nd]
        hr_map: dict[str, dict] = {}
        if hids_to_load:
            def _load_hr(hid: str):
                return hid, storage.load("horse_result", hid)
            with ThreadPoolExecutor(max_workers=min(len(hids_to_load), 8)) as pool:
                for hid, hr in pool.map(_load_hr, hids_to_load):
                    if hr:
                        hr_map[hid] = hr

        out = []
        for merged, hid, need_sire, need_dam in entries_need_hr:
            if need_sire or need_dam:
                hr = hr_map.get(hid)
                if hr:
                    if need_sire:
                        merged["sire"] = (hr.get("sire") or "").strip()
                    if need_dam:
                        merged["dam_sire"] = (hr.get("dam_sire") or "").strip()
            out.append(merged)
        return out, "race_result+horse_result"

    return [], None


def _cluster_labels(coords: np.ndarray) -> list[int]:
    n = len(coords)
    if n == 0:
        return []
    if n == 1:
        return [0]
    scaler = StandardScaler()
    xs = scaler.fit_transform(coords)
    k = min(4, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw = km.fit_predict(xs)
    by_cluster: dict[int, list[float]] = {}
    for i, lab in enumerate(raw):
        by_cluster.setdefault(int(lab), []).append(float(coords[i, 0]))
    avg_first = {c: float(np.mean(v)) for c, v in by_cluster.items()}
    order = sorted(avg_first.keys(), key=lambda c: -avg_first[c])
    remap = {old: new for new, old in enumerate(order)}
    return [remap[int(raw[i])] for i in range(n)]


def build_race_sire_factor_map(
    storage: Any,
    race_id: str,
    *,
    weights: dict[str, float] | None = None,
    axis_x: str = "ts",
    axis_y: str = "gear_change",
    axis_z: str = "ts_sustain",
) -> dict[str, Any]:
    entries, source = load_race_entries(storage, race_id)
    entries = sorted(entries, key=lambda e: int(e.get("horse_number") or 999))

    stats_data = load_sire_factor_stats()

    horses_raw: list[dict[str, Any]] = []
    all_factors: list[dict[str, float]] = []

    valid_entries = [(ent, (ent.get("horse_id") or "").strip())
                     for ent in entries if (ent.get("horse_id") or "").strip()]

    def _compute_one(pair):
        ent, hid = pair
        return ent, hid, compute_horse_aptitude(
            storage, hid, weights=weights, stats_data=stats_data)

    if valid_entries:
        with ThreadPoolExecutor(max_workers=min(len(valid_entries), 8)) as pool:
            results = list(pool.map(_compute_one, valid_entries))
        for ent, hid, apt in results:
            factors = apt["factors"]
            all_factors.append(factors)
            horses_raw.append({
                "horse_number": int(ent.get("horse_number") or 0),
                "bracket_number": int(ent.get("bracket_number") or 0),
                "horse_id": hid,
                "horse_name": (ent.get("horse_name") or "").strip(),
                "factors": factors,
                "ancestors": apt["ancestors"],
                "weights_used": apt["weights_used"],
            })

    if not horses_raw:
        return {
            "race_id": race_id.strip(),
            "source": source,
            "axes": AXES,
            "axis_x": axis_x, "axis_y": axis_y, "axis_z": axis_z,
            "weights_used": weights or DEFAULT_WEIGHTS,
            "ancestor_slots": [
                {"key": s["key"], "label_ja": s["label_ja"],
                 "default_weight": s["default_weight"]}
                for s in ANCESTOR_SLOTS
            ],
            "horses": [], "groups": [],
        }

    valid_axes = set(AXIS_IDS)
    ax = axis_x if axis_x in valid_axes else "ts"
    ay = axis_y if axis_y in valid_axes else "gear_change"
    az = axis_z if axis_z in valid_axes else "ts_sustain"

    coords = np.array([
        [f.get(ax, 0), f.get(ay, 0), f.get(az, 0)]
        for f in all_factors
    ], dtype=np.float64)

    labels = _cluster_labels(coords)
    for i, h in enumerate(horses_raw):
        h["group_id"] = labels[i] if i < len(labels) else 0

    groups: dict[int, list[dict]] = {}
    for h in horses_raw:
        gid = int(h["group_id"])
        groups.setdefault(gid, []).append(h)

    group_summary = []
    for gid in sorted(groups.keys()):
        members = groups[gid]
        centroid = {}
        for k in AXIS_IDS:
            centroid[k] = round(float(np.mean([
                m["factors"].get(k, 0) for m in members
            ])), 4)
        group_summary.append({
            "group_id": gid, "count": len(members),
            "centroid": centroid,
            "horse_names": [m["horse_name"] for m in members],
        })

    meta = stats_data.get("meta", {})
    return {
        "race_id": race_id.strip(),
        "source": source,
        "axes": AXES,
        "axis_x": ax, "axis_y": ay, "axis_z": az,
        "weights_used": horses_raw[0]["weights_used"] if horses_raw else DEFAULT_WEIGHTS,
        "ancestor_slots": [
            {"key": s["key"], "label_ja": s["label_ja"],
             "default_weight": s["default_weight"]}
            for s in ANCESTOR_SLOTS
        ],
        "stats_meta": {
            "generated_at": meta.get("generated_at", ""),
            "total_sires": meta.get("total_sires_filtered", 0),
            "horse_count": meta.get("horse_count", 0),
        },
        "horses": horses_raw,
        "groups": group_summary,
    }


# ---------------------------------------------------------------------------
# 比較データ: 同条件上位馬 / 直近2週間同コース
# ---------------------------------------------------------------------------

def _get_race_condition(storage: Any, race_id: str) -> dict[str, Any]:
    for cat in ("race_shutuba", "race_result"):
        data = storage.load(cat, race_id)
        if data:
            return {
                "surface": (data.get("surface") or "").strip(),
                "distance": int(data.get("distance") or 0),
                "track_condition": (data.get("track_condition") or "").strip(),
                "date": (data.get("date") or "").strip(),
                "venue": (data.get("venue") or "").strip(),
            }
    return {}


def _date_str_to_dt(d: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


def _list_race_dates(storage: Any, date_from: str, date_to: str) -> list[str]:
    keys = storage.list_keys("race_lists")
    return sorted(k for k in keys if date_from <= k <= date_to)


def _is_jra_race_id(rid: str) -> bool:
    return len(rid) >= 10 and rid[4:6] in {
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"
    }


def _is_good_track(tc: str) -> bool:
    return tc in ("\u826f", "")


_hr_cache: list[dict] | None = None
_hr_cache_lock = threading.Lock()
_hr_cache_ts: float = 0.0

_LOCAL_HR_SNAPSHOT = Path("data/research/_hr_snapshot_cache.jsonl.gz")


def _load_hr_snapshot(storage: Any) -> list[dict]:
    """horse_result スナップショットをメモリキャッシュで返す。"""
    global _hr_cache, _hr_cache_ts
    now = _time.time()
    if _hr_cache is not None and (now - _hr_cache_ts) < _PED_CACHE_TTL:
        return _hr_cache

    with _hr_cache_lock:
        if _hr_cache is not None and (now - _hr_cache_ts) < _PED_CACHE_TTL:
            return _hr_cache

        t0 = _time.time()
        records: list[dict] = []

        _SNAP_DISK_TTL = 86400
        snap_ok = False
        if _LOCAL_HR_SNAPSHOT.exists():
            snap_age = now - _LOCAL_HR_SNAPSHOT.stat().st_mtime
            if snap_age <= _SNAP_DISK_TTL:
                snap_ok = True
            else:
                try:
                    _LOCAL_HR_SNAPSHOT.unlink()
                except OSError:
                    pass
        if snap_ok:
            try:
                raw = gzip.decompress(_LOCAL_HR_SNAPSHOT.read_bytes())
                for line in raw.decode("utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
                if records:
                    _hr_cache = records
                    _hr_cache_ts = now
                    logger.info("hr_cache: ローカルから %d 馬ロード (%.1fs)",
                                len(records), _time.time() - t0)
                    return _hr_cache
            except Exception as e:
                logger.warning("hr_cache: ローカル読込失敗: %s", e)

        # GCSからダウンロード
        try:
            from research.sire_factor_stats import _download_snapshot_blob
            hr_gz = _download_snapshot_blob(storage, "horse_result.jsonl.gz")
            if hr_gz:
                _LOCAL_HR_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
                _LOCAL_HR_SNAPSHOT.write_bytes(hr_gz)
                raw = gzip.decompress(hr_gz)
                for line in raw.decode("utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            logger.warning("hr_cache: GCSロード失敗: %s", e)

        _hr_cache = records
        _hr_cache_ts = now
        logger.info("hr_cache: GCSから %d 馬ロード (%.1fs)", len(records), _time.time() - t0)
        return _hr_cache


def build_comparison_data(
    storage: Any,
    race_id: str,
    *,
    weights: dict[str, float] | None = None,
    axis_x: str = "ts",
    axis_y: str = "gear_change",
    axis_z: str = "ts_sustain",
    same_cond_from: str = "2025-01-01",
    same_cond_top_n: int = 3,
    recent_weeks: int = 2,
    override_surface: str = "",
    override_distance: int = 0,
    override_track_good: str = "",
) -> dict[str, Any]:
    """
    対象レースと比較するための2つのデータセットを返す。

    1. same_condition: 同surface+distance, 馬場カテゴリ(良/良以外)の上位N着
    2. recent_course: 直近N週間の同surface(芝orダ)複勝圏

    override_surface/distance/track_good でレース条件をオーバーライド可能。
    """
    t0 = _time.time()

    cond = _get_race_condition(storage, race_id)
    if not cond.get("surface") or not cond.get("distance"):
        return {"same_condition": {}, "recent_course": {}, "condition": {}}

    target_surface = override_surface or cond["surface"]
    target_dist = override_distance or cond["distance"]
    orig_tc = cond["track_condition"]
    if override_track_good == "good":
        target_good = True
    elif override_track_good == "bad":
        target_good = False
    else:
        target_good = _is_good_track(orig_tc)
    target_dt = _date_str_to_dt(cond["date"])
    is_turf = "\u829d" in target_surface

    stats_data = load_sire_factor_stats()

    # スナップショットキャッシュを並列ロード
    ped_cache = _load_ped_cache(storage)
    hr_records = _load_hr_snapshot(storage)

    t1 = _time.time()

    from_dt = _date_str_to_dt(same_cond_from)
    if target_dt and from_dt:
        df = from_dt.strftime("%Y%m%d")
        dt_end = target_dt.strftime("%Y%m%d")
    else:
        df = same_cond_from.replace("-", "")
        dt_end = "29991231"

    recent_from = ""
    if target_dt:
        recent_from = (target_dt - timedelta(days=recent_weeks * 7)).strftime("%Y%m%d")

    # horse_result スナップショットから比較候補を抽出
    same_cond_horses: list[dict] = []
    recent_horses: list[dict] = []
    seen_same: set[str] = set()
    seen_recent: set[str] = set()

    for hr in hr_records:
        hid = (hr.get("horse_id") or "").strip()
        if not hid:
            continue
        horse_name = (hr.get("name") or hr.get("horse_name") or "").strip()

        for race in hr.get("race_history") or []:
            rid = (race.get("race_id") or "").strip()
            if not rid or rid == race_id or not _is_jra_race_id(rid):
                continue

            s = (race.get("surface") or "").strip()
            d = int(race.get("distance") or 0)
            tc = (race.get("track_condition") or "").strip()
            race_date = (race.get("date") or "").strip()
            race_date_key = race_date.replace("-", "").replace("/", "")

            try:
                fp = int(race.get("finish_position") or 0)
            except (ValueError, TypeError):
                continue
            if fp < 1:
                continue

            # 同条件チェック
            if (s == target_surface and d == target_dist
                    and _is_good_track(tc) == target_good
                    and df <= race_date_key <= dt_end
                    and fp <= same_cond_top_n
                    and hid not in seen_same):
                seen_same.add(hid)
                factors = _compute_apt_cached(hid, weights, stats_data, ped_cache)
                if factors:
                    rn = horse_name or hid
                    hn = (race.get("horse_name") or rn).strip()
                    same_cond_horses.append({
                        "horse_id": hid, "horse_name": hn or rn,
                        "race_date": race_date, "finish_position": fp,
                        "factors": factors,
                    })

            # 直近チェック
            if (recent_from and race_date_key
                    and recent_from <= race_date_key <= dt_end
                    and fp <= 3
                    and "\u969c" not in s
                    and hid not in seen_recent):
                surface_match = False
                if is_turf and "\u829d" in s:
                    surface_match = True
                elif not is_turf and "\u30c0" in s:
                    surface_match = True
                if surface_match:
                    seen_recent.add(hid)
                    factors = _compute_apt_cached(hid, weights, stats_data, ped_cache)
                    if factors:
                        rn = horse_name or hid
                        hn = (race.get("horse_name") or rn).strip()
                        recent_horses.append({
                            "horse_id": hid, "horse_name": hn or rn,
                            "race_date": race_date, "finish_position": fp,
                            "distance": d,
                            "factors": factors,
                        })

    tc_label = "\u826f\u99ac\u5834" if target_good else "\u826f\u99ac\u5834\u4ee5\u5916"
    surface_label = "\u829d" if is_turf else "\u30c0\u30fc\u30c8"
    elapsed = round(_time.time() - t0, 1)
    scan_ms = round((_time.time() - t1) * 1000)

    return {
        "condition": {
            "surface": target_surface,
            "distance": target_dist,
            "track_condition": cond["track_condition"],
            "track_category": tc_label,
            "track_good": target_good,
            "date": cond["date"],
            "venue": cond["venue"],
        },
        "same_condition": {
            "label": f"{target_surface}{target_dist}m\uff08{tc_label}\uff09\u4e0a\u4f4d{same_cond_top_n}\u7740",
            "period_from": same_cond_from,
            "count": len(same_cond_horses),
            "horses": same_cond_horses,
        },
        "recent_course": {
            "label": f"\u76f4\u8fd1{recent_weeks}\u9031\u9593 {surface_label}\u8907\u52dd\u570f",
            "period_from": (target_dt - timedelta(days=recent_weeks * 7)).strftime("%Y-%m-%d") if target_dt else "",
            "period_to": cond["date"],
            "count": len(recent_horses),
            "horses": recent_horses,
        },
        "elapsed_sec": elapsed,
        "scan_ms": scan_ms,
    }
cond["date"],
            "count": len(recent_horses),
            "horses": recent_horses,
        },
        "elapsed_sec": elapsed,
        "scan_ms": scan_ms,
    }
