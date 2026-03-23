"""
GCS 上の前処理クッション（日次・会場別 JSON）を取り込み、
jra_cushion/{年}.json とローカル cushion_values.json を更新する。

preprocessed パス:
  chuou/data/preprocessed/others/cushion_data/{YYYY}/{会場名}/{YYYYMMDD}.json

出力（HybridStorage）:
  chuou/data/others/jra_cushion/{YYYY}.json

マージ方針:
  - 同一キー (date, venue_code, kai, race_day) の既存行は preprocessed 由来で上書き
  - preprocessed に無い列は既存値を維持（該当キーが無い場合）
  - preprocessed 専用の追加列（turf_course 等）は _preprocessed_extras に格納（任意）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scraper.storage import HybridStorage

logger = logging.getLogger(__name__)

PREPROCESSED_PREFIX = "chuou/data/preprocessed/others/cushion_data"

VENUE_JA_TO_CODE: dict[str, str] = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}

_WEEKDAY_SHORT = {
    "月": "月曜日",
    "火": "火曜日",
    "水": "水曜日",
    "木": "木曜日",
    "金": "金曜日",
    "土": "土曜日",
    "日": "日曜日",
}


def normalize_weekday(wd: str | None) -> str:
    if not wd:
        return ""
    s = str(wd).strip()
    if s in _WEEKDAY_SHORT:
        return _WEEKDAY_SHORT[s]
    if s.endswith("曜日"):
        return s
    if s.endswith("曜"):
        return s + "日"
    return s


def _is_race_weekday(wd: str) -> bool:
    return wd in ("土曜日", "日曜日", "月曜日")


def preprocessed_json_to_record(obj: dict, venue_folder: str) -> dict[str, Any] | None:
    """1ファイル分の dict → cushion_values 互換レコード。"""
    place = (obj.get("place") or venue_folder or "").strip()
    if not place:
        return None
    code = VENUE_JA_TO_CODE.get(place, "")
    if not code:
        logger.warning("未対応会場名をスキップ: %s", place)
        return None

    try:
        y = int(obj["year"])
        mo = int(obj["month"])
        d = int(obj["day"])
    except (KeyError, TypeError, ValueError):
        return None
    date_str = f"{y}-{mo:02d}-{d:02d}"

    wd = normalize_weekday(obj.get("weekday"))
    nichi = int(obj.get("nichi") or 0)
    kai = int(obj.get("kai") or 0)
    is_race = _is_race_weekday(wd)

    extras = {
        k: v
        for k, v in obj.items()
        if k
        in (
            "turf_course",
            "turf_course_info",
            "turf_status",
            "week_info",
            "water_info",
        )
    }

    rec: dict[str, Any] = {
        "year": y,
        "venue_code": code,
        "venue_name": place,
        "kai": kai,
        "pair_index": 0,
        "weekday": wd,
        "cushion_value": obj.get("cushion_value"),
        "turf_moisture_goal": obj.get("turf_moist_value_goal"),
        "turf_moisture_4corner": obj.get("turf_moist_value_4c"),
        "dirt_moisture_goal": obj.get("dart_moist_value_goal"),
        "dirt_moisture_4corner": obj.get("dart_moist_value_4c"),
        "is_race_day": is_race,
        "race_day": nichi if is_race else 0,
        "date": date_str,
        "course_position": (obj.get("turf_course") or "") or "",
        "measurement_time": obj.get("cushion_time") or "",
        "scraped_at": "",
        "_sync_source": "preprocessed_gcs",
    }
    if extras:
        rec["_preprocessed_extras"] = extras
    return rec


def record_merge_key(r: dict[str, Any]) -> tuple:
    """マージ・置換のキー。"""
    return (
        (r.get("date") or "").strip(),
        (r.get("venue_code") or "").strip(),
        int(r.get("kai") or 0),
        int(r.get("race_day") or 0),
    )


def _parse_preprocessed_blob_path(blob_name: str) -> tuple[int, str] | None:
    """(year, venue_folder) を返す。ファイル名は呼び出し側で扱う。"""
    parts = blob_name.split("/")
    try:
        i = parts.index("cushion_data")
    except ValueError:
        return None
    if i + 2 >= len(parts):
        return None
    y_str, venue_folder = parts[i + 1], parts[i + 2]
    if not y_str.isdigit() or len(y_str) != 4:
        return None
    return int(y_str), venue_folder


def iter_preprocessed_records(
    storage: HybridStorage,
    year: int,
) -> list[dict[str, Any]]:
    bucket = storage._get_bucket()
    prefix = f"{PREPROCESSED_PREFIX}/{year}/"
    out: list[dict[str, Any]] = []
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".json") or blob.name.endswith("/"):
            continue
        parsed = _parse_preprocessed_blob_path(blob.name)
        if not parsed:
            continue
        y, venue_folder = parsed
        if y != year:
            continue
        try:
            raw = json.loads(
                blob.download_as_text(timeout=getattr(storage, "_GCS_TIMEOUT", 10))
            )
        except Exception as e:
            logger.warning("読込失敗 %s: %s", blob.name, e)
            continue
        if not isinstance(raw, dict):
            continue
        rec = preprocessed_json_to_record(raw, venue_folder)
        if rec:
            out.append(rec)
    return out


def discover_preprocessed_years(storage: HybridStorage) -> list[int]:
    bucket = storage._get_bucket()
    years: set[int] = set()
    for blob in bucket.list_blobs(prefix=f"{PREPROCESSED_PREFIX}/"):
        p = _parse_preprocessed_blob_path(blob.name)
        if p:
            years.add(p[0])
    return sorted(years)


def merge_cushion_year(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    incoming（preprocessed）のキーに一致する既存行を置き換え、
    incoming に無い既存行はそのまま残す。
    """
    incoming_keys = {record_merge_key(r) for r in incoming}
    kept = [r for r in existing if record_merge_key(r) not in incoming_keys]
    merged = kept + incoming
    merged.sort(
        key=lambda r: (
            (r.get("date") or ""),
            r.get("venue_code") or "",
            int(r.get("kai") or 0),
            int(r.get("race_day") or 0),
            int(r.get("pair_index") or 0),
        )
    )
    stats = {
        "existing": len(existing),
        "incoming": len(incoming),
        "removed_overlap": len(existing) - len(kept),
        "result": len(merged),
    }
    return merged, stats


def _strip_internal_keys_for_local(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def rebuild_local_cushion_file(
    base_dir: Path,
    year_to_records: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """cushion_values.json を年別ブロックで差し替え。"""
    path = base_dir / "data" / "jra_baba" / "cushion_values.json"
    if not path.exists():
        all_rows: list[dict[str, Any]] = []
    else:
        all_rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(all_rows, list):
            all_rows = []

    for y, recs in year_to_records.items():
        all_rows = [r for r in all_rows if int(r.get("year") or 0) != y]
        for r in recs:
            all_rows.append(_strip_internal_keys_for_local(r))

    all_rows.sort(
        key=lambda r: (
            int(r.get("year") or 0),
            (r.get("date") or ""),
            r.get("venue_code") or "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "total_rows": len(all_rows)}


def sync_preprocessed_to_jra_cushion(
    *,
    years: list[int] | None = None,
    dry_run: bool = False,
    update_local_json: bool = True,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """
    preprocessed GCS → jra_cushion 年別 JSON 更新。
    years が None なら preprocessed に存在する年をすべて対象。
    """
    base = base_dir or Path.cwd()
    storage = HybridStorage(base_dir=str(base))

    if not storage.gcs_enabled:
        return {"ok": False, "error": "GCS が無効です"}

    target_years = sorted(years) if years else discover_preprocessed_years(storage)
    if not target_years:
        return {"ok": False, "error": "preprocessed 配下に対象年がありません"}

    per_year_stats: dict[str, Any] = {}
    year_payloads: dict[int, list[dict[str, Any]]] = {}

    for y in target_years:
        incoming = iter_preprocessed_records(storage, y)
        loaded = storage.load("jra_cushion", str(y))
        existing = (loaded or {}).get("records") or []
        if not isinstance(existing, list):
            existing = []

        if not incoming:
            per_year_stats[str(y)] = {
                "skipped": True,
                "reason": "preprocessed に該当年の JSON なし",
                "existing_rows": len(existing),
            }
            continue

        merged, st = merge_cushion_year(existing, incoming)
        per_year_stats[str(y)] = {**st, "preprocessed_blobs_rows": len(incoming)}
        year_payloads[y] = merged

        if dry_run:
            continue

        payload = {
            "records": merged,
            "_meta": {
                "kind": "cushion_year_merged",
                "year": y,
                "sync": "preprocessed_gcs",
            },
        }
        storage.save("jra_cushion", str(y), payload)

    if not dry_run:
        storage.invalidate_blob_cache("jra_cushion", "")

    local_info: dict[str, Any] | None = None
    if not dry_run and update_local_json:
        local_info = rebuild_local_cushion_file(base, year_payloads)

    return {
        "ok": True,
        "dry_run": dry_run,
        "years": target_years,
        "preprocessed_prefix": PREPROCESSED_PREFIX,
        "jra_cushion_prefix": f"{HybridStorage.GCS_OTHERS}/jra_cushion/",
        "per_year": per_year_stats,
        "local_cushion_values": local_info,
    }
