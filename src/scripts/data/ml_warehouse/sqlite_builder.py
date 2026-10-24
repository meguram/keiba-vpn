from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.scripts.data.ml_warehouse.paths import horse_shard_key, legacy_monolith_sqlite_path, local_tables_year
from src.scripts.data.ml_warehouse.sqlite_store import ShardWriter, rebuild_catalog

logger = logging.getLogger(__name__)


def _horse_result_json_paths(base: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (
        base / "data" / "cache" / "horse_result",
        base / "data" / "local" / "horse_result",
    ):
        if root.is_dir():
            paths.extend(root.rglob("*.json"))
    return paths


def _horse_training_json_paths(base: Path) -> list[Path]:
    paths: list[Path] = []
    root = base / "data" / "local" / "horse_training"
    if root.is_dir():
        paths.extend(root.rglob("*.json"))
    croot = base / "data" / "cache" / "horse_training"
    if croot.is_dir():
        paths.extend(croot.rglob("*.json"))
    return paths


def horse_ids_from_race_results(base: Path, years: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    for y in years:
        p = local_tables_year(base, y) / "race_result_flat.parquet"
        if not p.exists():
            continue
        try:
            t = pq.read_table(p, columns=["horse_id"])
        except Exception as e:
            logger.warning("race_result_flat 読込スキップ %s: %s", p, e)
            continue
        for h in t.column(0).to_pylist():
            if h is None:
                continue
            s = str(h).strip()
            if s:
                ids.add(s)
    logger.info("race_result 由来 horse_id: %d 頭", len(ids))
    return ids


def _row_horse_profile(
    horse_id: str,
    d: dict[str, Any],
    source_path: str,
    mtime: float,
) -> tuple[Any, ...]:
    now = time.time()
    return (
        horse_id,
        str(d.get("horse_name") or ""),
        str(d.get("name_en") or ""),
        str(d.get("status") or ""),
        str(d.get("sex") or ""),
        str(d.get("birthday") or ""),
        str(d.get("trainer") or ""),
        str(d.get("owner") or ""),
        str(d.get("breeder") or ""),
        str(d.get("birthplace") or ""),
        str(d.get("sire") or ""),
        str(d.get("dam") or ""),
        str(d.get("dam_sire") or ""),
        json.dumps(d.get("total_earnings"), ensure_ascii=False)
        if d.get("total_earnings") is not None else None,
        str(d.get("career") or ""),
        source_path,
        mtime,
        now,
    )


def _json_scalar(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (str, int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False)


class _ShardPool:
    def __init__(self, base: Path):
        self.base = base
        self._writers: dict[str, ShardWriter] = {}

    def get(self, horse_id: str) -> ShardWriter:
        sh = horse_shard_key(horse_id)
        if sh not in self._writers:
            self._writers[sh] = ShardWriter(self.base, sh)
        return self._writers[sh]

    def close_all(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for sh, w in self._writers.items():
            out[sh] = w.close()
        self._writers.clear()
        return out


def rebuild_horse_sqlite(
    base_dir: str | Path,
    years: Iterable[str],
    *,
    horse_id_filter: set[str] | None = None,
    include_training: bool = True,
    shard_filter: set[str] | None = None,
) -> dict[str, Any]:
    """馬データを horse_id 先頭4桁ごとの SQLite シャードに upsert。"""
    base = Path(base_dir)
    pool = _ShardPool(base)
    stats: dict[str, Any] = {
        "layout": "sharded_by_horse_id_prefix4",
        "profiles_upserted": 0,
        "history_rows": 0,
        "training_rows": 0,
        "skipped_files": 0,
        "shards_touched": 0,
    }
    now = time.time()

    def _shard_ok(hid: str) -> bool:
        if horse_id_filter is not None and hid not in horse_id_filter:
            return False
        if shard_filter is not None and horse_shard_key(hid) not in shard_filter:
            return False
        return True

    for jp in _horse_result_json_paths(base):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("horse_result 読込スキップ %s: %s", jp, e)
            stats["skipped_files"] += 1
            continue
        hid = str(d.get("horse_id") or jp.stem).strip()
        if not hid or not _shard_ok(hid):
            continue
        try:
            mtime = jp.stat().st_mtime
        except OSError:
            mtime = 0.0
        sp = str(jp)
        w = pool.get(hid)
        w.upsert_profile(_row_horse_profile(hid, d, sp, mtime))
        stats["profiles_upserted"] += 1

        hist = d.get("race_history")
        if isinstance(hist, list):
            for it in hist:
                if not isinstance(it, dict):
                    continue
                rid = str(it.get("race_id") or "").strip()
                if not rid:
                    continue
                w.upsert_history(
                    (
                        hid,
                        rid,
                        str(it.get("date") or ""),
                        str(it.get("venue") or ""),
                        str(it.get("race_name") or ""),
                        _json_scalar(it.get("race_round")),
                        str(it.get("surface") or ""),
                        _json_scalar(it.get("distance")),
                        str(it.get("track_condition") or ""),
                        str(it.get("weather") or ""),
                        _json_scalar(it.get("field_size")),
                        _json_scalar(it.get("bracket_number")),
                        _json_scalar(it.get("horse_number")),
                        _json_scalar(it.get("finish_position")),
                        str(it.get("jockey_name") or ""),
                        _json_scalar(it.get("jockey_weight")),
                        _json_scalar(it.get("odds")),
                        _json_scalar(it.get("popularity")),
                        str(it.get("finish_time") or ""),
                        float(it["time_sec"])
                        if isinstance(it.get("time_sec"), (int, float))
                        else None,
                        str(it.get("margin") or ""),
                        _json_scalar(it.get("passing_order")),
                        _json_scalar(it.get("last_3f")),
                        _json_scalar(it.get("weight")),
                        _json_scalar(it.get("weight_change")),
                        str(it.get("winner") or ""),
                        sp,
                        now,
                    )
                )
                stats["history_rows"] += 1

    if include_training:
        for jp in _horse_training_json_paths(base):
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
            except Exception as e:
                logger.debug("horse_training 読込スキップ %s: %s", jp, e)
                stats["skipped_files"] += 1
                continue
            hid = str(d.get("horse_id") or jp.stem).strip()
            if not hid or not _shard_ok(hid):
                continue
            try:
                mtime = jp.stat().st_mtime
            except OSError:
                mtime = 0.0
            sp = str(jp)
            pages = d.get("pages_fetched")
            pages_i = int(pages) if isinstance(pages, int) else None
            entries = d.get("entries")
            if not isinstance(entries, list):
                continue
            w = pool.get(hid)
            for seq, ent in enumerate(entries):
                if not isinstance(ent, dict):
                    continue
                dt = str(ent.get("date") or "").strip() or "__nodate__"
                w.upsert_training(
                    (
                        hid,
                        dt,
                        seq,
                        str(ent.get("race_info") or ""),
                        str(ent.get("day_of_week") or ""),
                        str(ent.get("course") or ""),
                        str(ent.get("track_condition") or ""),
                        str(ent.get("rider") or ""),
                        str(ent.get("time_raw") or ""),
                        _json_scalar(ent.get("lap_times")),
                        _json_scalar(ent.get("position")),
                        str(ent.get("leg_color") or ""),
                        str(ent.get("evaluation") or ""),
                        _json_scalar(ent.get("rank")),
                        str(ent.get("comment") or ""),
                        pages_i,
                        sp,
                        mtime,
                        now,
                    )
                )
                stats["training_rows"] += 1

    shard_stats = pool.close_all()
    stats["shards_touched"] = len(shard_stats)
    cat = rebuild_catalog(base, shard_stats)
    stats["catalog"] = str(cat)

    # 旧 monolith が残っていれば一度だけシャードへ移行（上書きはしない）
    leg = legacy_monolith_sqlite_path(base)
    if leg.exists() and stats["profiles_upserted"] == 0 and stats["training_rows"] == 0:
        from src.scripts.data.ml_warehouse.migrate_monolith import migrate_monolith_to_shards

        mig = migrate_monolith_to_shards(base)
        stats["monolith_migration"] = mig

    logger.info("SQLite シャード更新完了: %s", stats)
    return stats
