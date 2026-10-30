"""分析結果の事前計算。出馬表・結果が更新された時点で計算して保存し、画面（API）は保存済みの結果を読むだけにする。

再計算の要否は fingerprint（入力の指紋）で決める: 出走馬と枠・騎手の構成、レース条件、集計に使う結果データの最終日（そのレースの開催日より前）、
種牡馬の対応表、分析ロジックの版。どれも変わらなければ書き直さない。過去のレースの指紋は変わらないので、一度計算すれば再計算されない。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.research.race_analysis import runs as runs_mod
from src.research.race_analysis import store
from src.research.race_analysis.course_stats import lines_ready
from src.research.race_analysis.service import _load_card, _normalize_entries, analyze_race

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "17"      # 分析ロジック・列定義を変えたら上げる（保存済みの結果が作り直される）


def _asof_cache() -> dict[str, str | None]:
    return {}


def fingerprint(card: dict, entries: list[dict[str, Any]], asof: dict[str, str | None]) -> str:
    date = card.get("date")
    if date not in asof:
        runs = runs_mod.load_runs()
        before = runs[runs["date"] < date] if date else runs
        asof[date] = before["date"].max() if len(before) else None
    try:
        st = runs_mod.SIRE_MAP_PATH.stat()
        sire_sig = [st.st_mtime_ns, st.st_size]
    except OSError:
        sire_sig = None
    doc = {
        "v": ANALYSIS_VERSION,
        "race": [card.get(k) for k in ("venue", "surface", "distance", "date", "track_condition")],
        "entries": [[e.get("horse_id"), e.get("horse_number"), e.get("bracket_number"), e.get("jockey_id")] for e in entries],
        "data_until": asof[date],
        "sire_map": sire_sig,
        "lines_ready": lines_ready(),
    }
    return hashlib.sha1(json.dumps(doc, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def precompute_race(storage: Any, race_id: str, *, force: bool = False, asof: dict[str, str | None] | None = None) -> str:
    """1 レースぶんを計算して保存する。'written' / 'unchanged' / 'no_entries' を返す。"""
    asof = asof if asof is not None else _asof_cache()
    card, rows = _load_card(storage, race_id)
    entries, _ = _normalize_entries(rows)
    if not entries:
        return "no_entries"
    fp = fingerprint(card, entries, asof)
    old = store.read(race_id)
    if old and old.get("fingerprint") == fp and not force:
        return "unchanged"
    payload = analyze_race(storage, race_id)
    if payload.get("status") != "ok":
        return "no_entries"
    payload["fingerprint"] = fp
    payload["version"] = ANALYSIS_VERSION
    payload["computed_at"] = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S+09:00")   # ホストは UTC。画面は JST で見せる
    store.write(race_id, payload)
    return "written"


def precompute_races(storage: Any, race_ids: list[str], *, force: bool = False) -> dict[str, int]:
    asof = _asof_cache()
    out = {"written": 0, "unchanged": 0, "no_entries": 0, "error": 0}
    for rid in race_ids:
        try:
            out[precompute_race(storage, rid, force=force, asof=asof)] += 1
        except Exception:
            logger.exception("分析の事前計算に失敗: %s", rid)
            out["error"] += 1
    return out


def refresh_for_races(storage: Any, race_ids: list[str], *, force: bool = False, extra_horse_ids: set[str] | None = None) -> dict[str, int]:
    """出馬表が更新されたレースの分析を作り直す。先に、その出走馬の父の対応表（差分のみ）を更新する。"""
    from src.scraper import sire_line_store
    from src.scripts.data import build_horse_sire_map as sire

    ids: set[str] = set(extra_horse_ids or ())
    for rid in race_ids:
        try:
            _, rows = _load_card(storage, rid)
        except Exception:
            continue
        ids |= {str(r["horse_id"]) for r in rows if r.get("horse_id")}
    if ids:
        try:
            sire.build(horse_ids=ids, netkeiba_lines=sire_line_store.labels())
        except Exception:
            logger.exception("父の対応表の更新に失敗（分析は対応表の現状で続ける）")
    return precompute_races(storage, race_ids, force=force)
