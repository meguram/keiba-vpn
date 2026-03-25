"""
種牡馬因子統計パイプライン

GCS上の horse_pedigree_5gen + horse_result から、種牡馬ごとの
paternal/maternal 別レース結果統計を集計する。

高速モード: GCS上の集約スナップショット (JSONL.gz) から再計算（~10秒）
フルモード:  個別GCSファイルを走査してスナップショットを更新 + 再計算（~5分）

出力: data/research/sire_factor_stats.json + GCS snapshots/
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AXES = [
    {"id": "ts", "label_ja": "TS（最高速度）",
     "hint": "上がり3Fの速さ。距離・馬場補正済み偏差値"},
    {"id": "gear_change", "label_ja": "ギアチェンジ",
     "hint": "後方から上がりの速さで追い込む力"},
    {"id": "ts_sustain", "label_ja": "TS持続力",
     "hint": "前半ペースの効率 + 長距離での成績"},
    {"id": "wet_turf", "label_ja": "道悪適性",
     "hint": "芝の重/不良での複勝率向上度"},
    {"id": "dirt", "label_ja": "ダート適性",
     "hint": "ダートでの複勝率向上度"},
    {"id": "start_speed", "label_ja": "スタート速度",
     "hint": "序盤の位置取り（前に行ける力）"},
    {"id": "distance_range", "label_ja": "距離適性幅",
     "hint": "好走距離の幅広さ（大=万能型）"},
    {"id": "consistency", "label_ja": "安定性",
     "hint": "総合的な複勝率"},
]

AXIS_IDS = tuple(a["id"] for a in AXES)

OUTPUT_PATH = Path("data/research/sire_factor_stats.json")
SNAPSHOT_PREFIX = "chuou/data/snapshots/sire_factor"

MIN_SAMPLE_DEFAULT = 10
SHRINKAGE_N = 20


# ---------------------------------------------------------------------------
# GCS スナップショット操作
# ---------------------------------------------------------------------------

def _get_bucket(storage: Any):
    """HybridStorage の内部バケットオブジェクトを取得。"""
    if hasattr(storage, "_get_bucket"):
        return storage._get_bucket()
    raise RuntimeError("storage に GCS バケットがありません")


def build_and_upload_snapshot(storage: Any, *, years: list[int] | None = None):
    """
    GCSから全 horse_pedigree_5gen + horse_result を読み込み、
    JSONL.gz 形式で GCS スナップショットにアップロードする。
    """
    t0 = time.time()
    if years is None:
        years = list(range(2015, 2027))

    bucket = _get_bucket(storage)

    # pedigree_5gen を JSONL.gz にまとめる
    logger.info("Snapshot: horse_pedigree_5gen 収集開始")
    ped_buf = io.BytesIO()
    ped_count = 0
    for yr in years:
        keys = storage.list_keys("horse_pedigree_5gen", year=str(yr))
        logger.info("  year=%d: %d keys", yr, len(keys))
        for key in keys:
            rec = storage.load("horse_pedigree_5gen", key)
            if rec and rec.get("ancestors"):
                ped_buf.write(
                    (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
                )
                ped_count += 1

    ped_raw = ped_buf.getvalue()
    ped_gz = gzip.compress(ped_raw, compresslevel=6)
    blob_ped = bucket.blob(f"{SNAPSHOT_PREFIX}/horse_pedigree_5gen.jsonl.gz")
    blob_ped.upload_from_string(ped_gz, content_type="application/gzip")
    logger.info("  pedigree snapshot: %d records, %.1f MB gz",
                ped_count, len(ped_gz) / 1024 / 1024)

    # horse_result: pedigree で必要な horse_id を収集してロード
    horse_ids = set()
    ped_buf.seek(0)
    for line in ped_buf:
        try:
            rec = json.loads(line)
            horse_ids.add(rec.get("horse_id", ""))
        except Exception:
            pass

    logger.info("Snapshot: horse_result 収集開始 (%d horses)", len(horse_ids))
    hr_buf = io.BytesIO()
    hr_count = 0

    def _load_hr(hid: str):
        return hid, storage.load("horse_result", hid)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_load_hr, hid): hid for hid in horse_ids}
        for fut in as_completed(futs):
            hid, hr = fut.result()
            if hr and hr.get("race_history"):
                hr_buf.write(
                    (json.dumps(hr, ensure_ascii=False) + "\n").encode("utf-8")
                )
                hr_count += 1

    hr_raw = hr_buf.getvalue()
    hr_gz = gzip.compress(hr_raw, compresslevel=6)
    blob_hr = bucket.blob(f"{SNAPSHOT_PREFIX}/horse_result.jsonl.gz")
    blob_hr.upload_from_string(hr_gz, content_type="application/gzip")
    logger.info("  horse_result snapshot: %d records, %.1f MB gz",
                hr_count, len(hr_gz) / 1024 / 1024)

    # メタ情報
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pedigree_count": ped_count,
        "horse_result_count": hr_count,
        "pedigree_gz_bytes": len(ped_gz),
        "horse_result_gz_bytes": len(hr_gz),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    blob_meta = bucket.blob(f"{SNAPSHOT_PREFIX}/snapshot_meta.json")
    blob_meta.upload_from_string(
        json.dumps(meta, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

    elapsed = time.time() - t0
    logger.info("Snapshot 完了: ped=%d, hr=%d (%.1f秒)", ped_count, hr_count, elapsed)
    return meta


def _download_snapshot_blob(storage: Any, name: str) -> bytes | None:
    """GCS スナップショットの blob をダウンロード。"""
    try:
        bucket = _get_bucket(storage)
        blob = bucket.blob(f"{SNAPSHOT_PREFIX}/{name}")
        return blob.download_as_bytes()
    except Exception as e:
        logger.warning("Snapshot download failed (%s): %s", name, e)
        return None


def load_snapshot_data(storage: Any) -> tuple[list[dict], list[dict]] | None:
    """
    GCS スナップショットから pedigree + horse_result を一括ロード。
    Returns (pedigree_records, horse_result_records) or None on failure.
    """
    t0 = time.time()
    logger.info("Snapshot からデータロード開始")

    ped_gz = _download_snapshot_blob(storage, "horse_pedigree_5gen.jsonl.gz")
    if not ped_gz:
        return None
    hr_gz = _download_snapshot_blob(storage, "horse_result.jsonl.gz")
    if not hr_gz:
        return None

    ped_raw = gzip.decompress(ped_gz)
    peds = []
    for line in ped_raw.decode("utf-8").splitlines():
        if line.strip():
            peds.append(json.loads(line))

    hr_raw = gzip.decompress(hr_gz)
    hrs = []
    for line in hr_raw.decode("utf-8").splitlines():
        if line.strip():
            hrs.append(json.loads(line))

    logger.info("  Snapshot ロード完了: ped=%d, hr=%d (%.1f秒)",
                len(peds), len(hrs), time.time() - t0)
    return peds, hrs


def upload_stats_to_gcs(storage: Any, data: dict):
    """sire_factor_stats.json を GCS にもアップロード。"""
    try:
        bucket = _get_bucket(storage)
        blob = bucket.blob(f"{SNAPSHOT_PREFIX}/sire_factor_stats.json")
        content = json.dumps(data, ensure_ascii=False, indent=2)
        blob.upload_from_string(content, content_type="application/json")
        logger.info("GCS に sire_factor_stats.json アップロード完了")
    except Exception as e:
        logger.warning("GCS アップロード失敗: %s", e)


def download_stats_from_gcs(storage: Any) -> dict | None:
    """GCS から sire_factor_stats.json をダウンロード。"""
    raw = _download_snapshot_blob(storage, "sire_factor_stats.json")
    if raw:
        return json.loads(raw.decode("utf-8"))
    return None


# ---------------------------------------------------------------------------
# レースフィーチャー抽出・統計計算
# ---------------------------------------------------------------------------

def _parse_passing_order(po_str: str) -> list[int]:
    if not po_str:
        return []
    parts = po_str.replace("　", "-").replace(" ", "-").split("-")
    out = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            out.append(int(p))
    return out


def _collect_race_features(races: list[dict]) -> list[dict]:
    """horse_result.race_history → 統計計算用のクリーンなレコード群。"""
    out = []
    for r in races:
        try:
            fp = int(r.get("finish_position") or 0)
        except (ValueError, TypeError):
            continue
        if fp <= 0:
            continue

        last_3f = float(r.get("last_3f") or 0)
        time_sec = float(r.get("time_sec") or 0)
        distance = int(r.get("distance") or 0)
        fs = int(r.get("field_size") or 0)
        surface = (r.get("surface") or "").strip()
        tc = (r.get("track_condition") or "").strip()
        po = _parse_passing_order(r.get("passing_order") or "")

        if distance <= 0 or fs <= 0:
            continue

        out.append({
            "fp": fp, "last_3f": last_3f, "time_sec": time_sec,
            "distance": distance, "field_size": fs,
            "surface": surface, "track_condition": tc,
            "passing_order": po, "top3": fp <= 3,
        })
    return out


def _compute_raw_vector(features: list[dict]) -> dict[str, float]:
    n = len(features)
    if n == 0:
        return {k: 0.0 for k in AXIS_IDS}

    last_3fs = [f["last_3f"] for f in features if f["last_3f"] > 0]
    turf_races = [f for f in features if "芝" in f["surface"]]
    dirt_races = [f for f in features if "ダ" in f["surface"]]

    ts = (38.0 - sum(last_3fs) / len(last_3fs)) if last_3fs else 0.0

    gear_vals = []
    for f in features:
        if f["passing_order"] and f["last_3f"] > 0:
            improvement = f["passing_order"][-1] - f["fp"]
            gear_vals.append(improvement * 0.3 + (38.0 - f["last_3f"]) * 0.7)
    gear_change = sum(gear_vals) / len(gear_vals) if gear_vals else 0.0

    sustain_vals = []
    for f in features:
        if f["time_sec"] > 0 and f["last_3f"] > 0 and f["distance"] > 0:
            front_time = f["time_sec"] - f["last_3f"]
            if front_time > 0:
                front_per_f = front_time / (f["distance"] / 200 - 3)
                if front_per_f > 0:
                    sustain_vals.append(max(0, 13.0 - front_per_f))
    ts_sustain = sum(sustain_vals) / len(sustain_vals) if sustain_vals else 0.0
    long_top3 = [f for f in features if f["distance"] >= 2000 and f["top3"]]
    long_total = [f for f in features if f["distance"] >= 2000]
    if long_total:
        ts_sustain += (len(long_top3) / len(long_total)) * 1.5

    wet_conditions = {"重", "不良"}
    turf_wet = [f for f in turf_races if f["track_condition"] in wet_conditions]
    turf_good = [f for f in turf_races if f["track_condition"] not in wet_conditions]
    wet_rate = (sum(1 for f in turf_wet if f["top3"]) / len(turf_wet)) if turf_wet else 0
    good_rate = (sum(1 for f in turf_good if f["top3"]) / len(turf_good)) if turf_good else 0
    wet_turf = wet_rate - good_rate * 0.5 if turf_wet else 0.0

    dirt_top3 = sum(1 for f in dirt_races if f["top3"])
    dirt_rate = dirt_top3 / len(dirt_races) if dirt_races else 0
    turf_top3_rate = (sum(1 for f in turf_races if f["top3"]) / len(turf_races)) if turf_races else 0
    dirt_val = dirt_rate - turf_top3_rate * 0.3 if dirt_races else 0.0

    start_vals = []
    for f in features:
        if f["passing_order"] and f["field_size"] > 1:
            first_pos = f["passing_order"][0]
            start_vals.append(1.0 - (first_pos - 1) / (f["field_size"] - 1))
    start_speed = sum(start_vals) / len(start_vals) if start_vals else 0.5

    top3_dists = [f["distance"] for f in features if f["top3"]]
    if len(top3_dists) >= 2:
        mean_d = sum(top3_dists) / len(top3_dists)
        var_d = sum((d - mean_d) ** 2 for d in top3_dists) / len(top3_dists)
        distance_range = math.sqrt(var_d) / 400.0
    else:
        distance_range = 0.0

    top3_count = sum(1 for f in features if f["top3"])
    consistency = top3_count / n

    return {
        "ts": round(ts, 4), "gear_change": round(gear_change, 4),
        "ts_sustain": round(ts_sustain, 4), "wet_turf": round(wet_turf, 4),
        "dirt": round(dirt_val, 4), "start_speed": round(start_speed, 4),
        "distance_range": round(distance_range, 4),
        "consistency": round(consistency, 4),
    }


def _z_normalize_sires(sires_raw: dict[str, dict]) -> dict[str, dict]:
    if not sires_raw:
        return {}
    all_vals: dict[str, list[float]] = {k: [] for k in AXIS_IDS}
    for info in sires_raw.values():
        for side in ("paternal", "maternal"):
            axes = info.get(side, {}).get("axes", {})
            for k in AXIS_IDS:
                all_vals[k].append(axes.get(k, 0.0))
    means, stds = {}, {}
    for k in AXIS_IDS:
        vals = all_vals[k]
        m = sum(vals) / len(vals) if vals else 0.0
        var = sum((v - m) ** 2 for v in vals) / len(vals) if vals else 0.0
        means[k] = m
        stds[k] = math.sqrt(var) if var > 0 else 1.0
    result = {}
    for sid, info in sires_raw.items():
        entry = {"name": info["name"]}
        for side in ("paternal", "maternal"):
            raw_axes = info.get(side, {}).get("axes", {})
            sample_size = info.get(side, {}).get("sample_size", 0)
            shrink = min(1.0, sample_size / SHRINKAGE_N) if sample_size > 0 else 0.0
            normalized = {}
            for k in AXIS_IDS:
                z = (raw_axes.get(k, 0.0) - means[k]) / stds[k]
                z = max(-3.0, min(3.0, z * shrink))
                normalized[k] = round(z, 4)
            entry[side] = {"sample_size": sample_size, "axes": normalized}
        result[sid] = entry
    return result


# ---------------------------------------------------------------------------
# コア集計: pedigree_records + horse_result_records → stats dict
# ---------------------------------------------------------------------------

def _aggregate_from_records(
    pedigree_records: list[dict],
    horse_result_records: list[dict],
    min_sample: int = MIN_SAMPLE_DEFAULT,
) -> dict[str, Any]:
    """メモリ上の pedigree + horse_result レコード群から統計を集計。"""
    t0 = time.time()

    sire_index: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"paternal": [], "maternal": []}
    )
    sire_names: dict[str, str] = {}
    horse_ids_needed: set[str] = set()

    for ped in pedigree_records:
        horse_id = ped.get("horse_id", "")
        if not horse_id:
            continue
        horse_ids_needed.add(horse_id)
        for anc in ped.get("ancestors", []):
            g = anc.get("generation", 0)
            p = anc.get("position", 0)
            sid = (anc.get("horse_id") or "").strip()
            name = (anc.get("name") or "").strip()
            if not sid:
                continue
            is_sire_position = (p & 1) == 0
            if g == 1 and p == 0:
                sire_index[sid]["paternal"].append(horse_id)
            elif is_sire_position:
                sire_index[sid]["maternal"].append(horse_id)
            if sid and name:
                sire_names[sid] = name

    hr_map: dict[str, list[dict]] = {}
    for hr in horse_result_records:
        hid = hr.get("horse_id", "")
        hist = hr.get("race_history", [])
        if hid and hist:
            hr_map[hid] = hist

    sires_raw: dict[str, dict] = {}
    for sid, groups in sire_index.items():
        entry: dict[str, Any] = {"name": sire_names.get(sid, sid)}
        for side in ("paternal", "maternal"):
            all_features = []
            for hid in groups[side]:
                all_features.extend(_collect_race_features(hr_map.get(hid, [])))
            entry[side] = {
                "sample_size": len(all_features),
                "axes": _compute_raw_vector(all_features),
            }
        sires_raw[sid] = entry

    sires_normalized = _z_normalize_sires(sires_raw)
    final_sires = {}
    for sid, info in sires_normalized.items():
        pat_n = info.get("paternal", {}).get("sample_size", 0)
        mat_n = info.get("maternal", {}).get("sample_size", 0)
        if pat_n >= min_sample or mat_n >= min_sample:
            final_sires[sid] = info

    elapsed = time.time() - t0
    return {
        "meta": {
            "version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_sec": round(elapsed, 1),
            "total_sires_raw": len(sires_raw),
            "total_sires_filtered": len(final_sires),
            "min_sample": min_sample,
            "horse_count": len(horse_ids_needed),
        },
        "axes": AXES,
        "sires": final_sires,
    }


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def build_sire_factor_stats_fast(
    storage: Any,
    *,
    min_sample: int = MIN_SAMPLE_DEFAULT,
) -> dict[str, Any]:
    """スナップショットから高速再計算（~10秒）。"""
    snap = load_snapshot_data(storage)
    if snap is None:
        raise RuntimeError(
            "スナップショットが見つかりません。先に mode=full で構築してください。"
        )
    peds, hrs = snap
    return _aggregate_from_records(peds, hrs, min_sample=min_sample)


def build_sire_factor_stats(
    storage: Any,
    *,
    min_sample: int = MIN_SAMPLE_DEFAULT,
    years: list[int] | None = None,
) -> dict[str, Any]:
    """個別GCSから全件読み直し（フルモード、~5分）。"""
    t0 = time.time()
    if years is None:
        years = list(range(2015, 2027))

    logger.info("Full mode: pedigree_5gen ロード開始")
    pedigree_records = []
    for yr in years:
        keys = storage.list_keys("horse_pedigree_5gen", year=str(yr))
        logger.info("  year=%d: %d keys", yr, len(keys))
        for key in keys:
            ped = storage.load("horse_pedigree_5gen", key)
            if ped and ped.get("ancestors"):
                pedigree_records.append(ped)

    horse_ids = {p.get("horse_id", "") for p in pedigree_records}
    logger.info("Full mode: horse_result ロード開始 (%d horses)", len(horse_ids))
    horse_result_records = []

    def _load_hr(hid: str):
        return storage.load("horse_result", hid)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_load_hr, hid): hid for hid in horse_ids}
        for fut in as_completed(futs):
            hr = fut.result()
            if hr and hr.get("race_history"):
                horse_result_records.append(hr)

    logger.info("  loaded %d horse_results", len(horse_result_records))

    result = _aggregate_from_records(
        pedigree_records, horse_result_records, min_sample=min_sample
    )
    result["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    return result


def save_sire_factor_stats(
    data: dict,
    path: Path | str | None = None,
    storage: Any = None,
) -> Path:
    p = Path(path) if path else OUTPUT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("保存: %s (%.1f MB)", p, p.stat().st_size / 1024 / 1024)
    if storage:
        upload_stats_to_gcs(storage, data)
    return p


_cached_stats: dict[str, Any] | None = None
_cached_mtime: float = 0.0


def load_sire_factor_stats(
    path: Path | str | None = None,
    storage: Any = None,
) -> dict[str, Any]:
    """ローカル → GCS フォールバックでロード。"""
    global _cached_stats, _cached_mtime
    p = Path(path) if path else OUTPUT_PATH
    if p.exists():
        mt = p.stat().st_mtime
        if _cached_stats is not None and mt == _cached_mtime:
            return _cached_stats
        with open(p, "r", encoding="utf-8") as f:
            _cached_stats = json.load(f)
        _cached_mtime = mt
        return _cached_stats
    if storage:
        logger.info("ローカルなし → GCS からダウンロード")
        data = download_stats_from_gcs(storage)
        if data:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            _cached_stats = data
            _cached_mtime = p.stat().st_mtime
            return data
    return {"meta": {}, "axes": AXES, "sires": {}}


def invalidate_cache():
    global _cached_stats, _cached_mtime
    _cached_stats = None
    _cached_mtime = 0.0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from scraper.storage import HybridStorage
    storage = HybridStorage(".")

    mode = "full"
    ms = MIN_SAMPLE_DEFAULT
    for a in sys.argv[1:]:
        if a == "--fast":
            mode = "fast"
        elif a == "--snapshot":
            mode = "snapshot"
        elif a.isdigit():
            ms = int(a)

    if mode == "snapshot":
        meta = build_and_upload_snapshot(storage)
        print(f"Snapshot 完了: {json.dumps(meta, ensure_ascii=False, indent=2)}")
    elif mode == "fast":
        data = build_sire_factor_stats_fast(storage, min_sample=ms)
        save_sire_factor_stats(data, storage=storage)
        print(f"Fast: {len(data['sires'])} 種牡馬 ({data['meta']['elapsed_sec']}秒)")
    else:
        data = build_sire_factor_stats(storage, min_sample=ms)
        save_sire_factor_stats(data, storage=storage)
        build_and_upload_snapshot(storage)
        print(f"Full: {len(data['sires'])} 種牡馬 ({data['meta']['elapsed_sec']}秒)")
