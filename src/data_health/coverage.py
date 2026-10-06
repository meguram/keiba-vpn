"""レース／馬カテゴリの存在カバレッジ（どの時期・どのレースが足りないか）。

課金安全: GCS はカテゴリ×年ごとの ``batch_list_blobs``（list のみ）と、馬の評価窓の出馬表 ``load`` だけ。
dev は ``HybridStorage`` が ``data/dev_mock`` を読むので GCP には触れない。
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config.data_paths import LOCAL, calculated_data_root, page_reference_root
from src.data_health.spec import (
    HORSE_CATEGORIES,
    HORSE_RACES_MAX,
    HORSE_WINDOW_FUTURE_DAYS,
    HORSE_WINDOW_PAST_DAYS,
    RACE_CATEGORIES,
    CategorySpec,
)

logger = logging.getLogger("data_health.coverage")
JST = timezone(timedelta(hours=9))
JRA_PLACES = {f"{i:02d}" for i in range(1, 11)}
VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


# ── 期限（SLA）────────────────────────────────────────────────────────────

def _at(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=JST)


def due_time(rule: str, race_date: date) -> datetime:
    """このカテゴリが揃っているべき時刻（JST）。これを過ぎて無ければ「不足」、前なら「待機」。"""
    if rule == "shutuba":      # SLA1 前日17:00 ＋猶予
        return _at(race_date - timedelta(days=1), 18)
    if rule == "dayof":        # SLA3〜5 当日18:00 ＋猶予
        return _at(race_date, 19)
    if rule == "weekly":       # SLA6 レース後に最初に訪れる金曜17:00 ＋猶予
        days = (4 - race_date.weekday()) % 7 or 7
        return _at(race_date + timedelta(days=days), 18, 30)
    if rule == "horse":        # 出馬表取得後の馬取得
        return _at(race_date - timedelta(days=1), 21)
    raise ValueError(f"unknown due rule: {rule}")


# ── カレンダー（race_lists）──────────────────────────────────────────────

@dataclass
class Calendar:
    dates: dict[str, list[str]] = field(default_factory=dict)    # YYYYMMDD → race_id（JRA）
    anomalies: list[dict] = field(default_factory=list)
    source_dirs: list[str] = field(default_factory=list)
    no_race_dates: set[str] = field(default_factory=set)         # 「開催なし」と記録された日
    race_meta: dict[str, dict] = field(default_factory=dict)     # race_id → {round, venue, race_name}（race_lists の項目）

    def race_date(self) -> dict[str, str]:
        return {rid: d for d, ids in self.dates.items() for rid in ids}


def race_list_dirs() -> list[Path]:
    return [calculated_data_root() / "race_lists", page_reference_root() / "race_lists", LOCAL / "race_lists"]


def load_calendar(since: date, until: date, *, dirs: list[Path] | None = None) -> Calendar:
    from src.scraper.race_list_completeness import race_list_stats

    cal = Calendar()
    seen: set[str] = set()
    for d in (dirs if dirs is not None else race_list_dirs()):
        if not d.is_dir():
            continue
        cal.source_dirs.append(str(d))
        for p in sorted(d.glob("*.json")):
            key = p.stem
            if key in seen or not re.fullmatch(r"\d{8}", key):
                continue
            try:
                dt = datetime.strptime(key, "%Y%m%d").date()
            except ValueError:
                continue
            if dt < since or dt > until:
                continue
            seen.add(key)
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cal.anomalies.append({"date": key, "problem": "race_lists が読めない（JSON 破損）"})
                continue
            stats = race_list_stats(data)
            if stats.jra_count == 0 and stats.is_complete:
                cal.no_race_dates.add(key)
                continue
            ids = [str(r["race_id"]) for r in data.get("races", [])
                   if isinstance(r, dict) and str(r.get("race_id", ""))[4:6] in JRA_PLACES]
            for r in data.get("races", []):
                if isinstance(r, dict) and str(r.get("race_id", "")) in ids:
                    cal.race_meta[str(r["race_id"])] = {k: r[k] for k in ("round", "venue", "race_name", "grade", "surface", "distance")
                                                        if r.get(k) not in (None, "")}
            if ids:
                cal.dates[key] = sorted(set(ids))
            if not stats.is_complete:
                cal.anomalies.append({"date": key, "problem": f"race_lists 不完全: {stats.reason}"})
    return cal


def weekend_coverage(cal: Calendar, today: date, days_ahead: int = 7) -> list[str]:
    """今日〜days_ahead 日先の土日のうち、race_lists に記録（開催あり/なし）が無い日。"""
    out = []
    for i in range(days_ahead + 1):
        d = today + timedelta(days=i)
        key = d.strftime("%Y%m%d")
        if d.weekday() >= 5 and key not in cal.dates and key not in cal.no_race_dates:
            out.append(key)
    return out


# ── レース集合（universe）────────────────────────────────────────────────

@dataclass
class Race:
    race_id: str
    date: str | None          # YYYYMMDD（race_lists で確認できたものだけ）
    confidence: str           # confirmed(race_lists) / present(データ有) / inferred(開催回の連番から推定)


def _parse_rid(rid: str) -> tuple[str, str, int, int, int] | None:
    if len(rid) != 12 or not rid.isdigit():
        return None
    return rid[:4], rid[4:6], int(rid[6:8]), int(rid[8:10]), int(rid[10:12])


def infer_missing(observed: set[str]) -> set[str]:
    """開催回（year+場+回）の日は 1..最大日 が連続し、各日は 1..12R と推定して、欠けている race_id を返す。"""
    by_kai: dict[tuple[str, str, int], dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for rid in observed:
        p = _parse_rid(rid)
        if p:
            by_kai[(p[0], p[1], p[2])][p[3]].add(p[4])
    out: set[str] = set()
    for (yr, pl, kai), days in by_kai.items():
        for day in range(1, max(days) + 1):
            for r in range(1, 13):
                rid = f"{yr}{pl}{kai:02d}{day:02d}{r:02d}"
                if rid not in observed:
                    out.add(rid)
    return out


def build_universe(cal: Calendar, present: dict[str, dict[str, dict[str, float]]], years: list[str], *,
                   infer: bool, keytable: Any = None) -> dict[str, Race]:
    """対象レースの集合。日付は race_lists を優先し、無ければキーテーブル（取得済み JSON から収穫）を使う。"""
    cal_date = cal.race_date()

    class _DateOf(dict):
        def get(self, rid, default=None):           # noqa: A003
            return cal_date.get(rid) or (keytable.date_of(rid) if keytable is not None else None) or default

    date_of = _DateOf()
    observed: dict[str, Race] = {}
    for rid, d in cal_date.items():
        if rid[:4] in years:
            observed[rid] = Race(rid, d, "confirmed")
    for cat in ("race_shutuba", "race_result"):
        for y in years:
            for rid in present.get(cat, {}).get(y, {}):
                if rid[4:6] in JRA_PLACES and rid not in observed:
                    observed[rid] = Race(rid, date_of.get(rid), "present")
    if infer:
        for rid in infer_missing(set(observed)):
            if rid[:4] in years:
                observed[rid] = Race(rid, date_of.get(rid), "inferred")
    return observed


# ── 存在の収集（list のみ）───────────────────────────────────────────────

def collect_presence(storage: Any, categories: list[str], years: list[str], *, workers: int = 8
                     ) -> dict[str, dict[str, dict[str, float]]]:
    """{category: {year: {key: updated_ts}}}。一覧は毎回取り直す（キャッシュ無効化）。"""
    try:
        storage.invalidate_blob_cache()
    except Exception:
        pass

    def one(cat: str, year: str) -> tuple[str, str, dict[str, float]]:
        try:
            return cat, year, storage.batch_list_blobs(cat, year) or {}
        except Exception as e:  # 1 件の失敗で全体を止めない
            logger.warning("list 失敗 %s/%s: %s", cat, year, e)
            return cat, year, {}

    out: dict[str, dict[str, dict[str, float]]] = {c: {} for c in categories}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cat, year, res in pool.map(lambda a: one(*a), [(c, y) for c in categories for y in years]):
            out[cat][year] = res
    return out


def load_not_available(cat: str, year: str) -> set[str]:
    try:
        from src.scraper.date_coverage import load_not_available as _na
        return _na(cat, year)
    except Exception:
        return set()


# ── 判定・集計 ──────────────────────────────────────────────────────────

def _period(race: Race) -> str:
    return f"{race.date[:4]}-{race.date[4:6]}" if race.date else f"{race.race_id[:4]}-??"


def _in_scope(spec: CategorySpec, race: Race, now: datetime) -> bool:
    if spec.window_days is None:
        return True
    if not race.date:
        return False
    d = datetime.strptime(race.date, "%Y%m%d").date()
    return (now.date() - timedelta(days=spec.window_days)) <= d <= (now.date() + timedelta(days=HORSE_WINDOW_FUTURE_DAYS))


def classify(spec: CategorySpec, race: Race, present_keys: set[str], na: set[str], now: datetime, *,
             health: dict | None = None, unknown: str = "present") -> str:
    """存在（present）／健全（healthy）／不適合（invalid）／未検証（unvalidated）／待機／不足／取得不可。

    health（スキーマ検証の結果）があるカテゴリでは、存在するものを健全・不適合に分ける。検証がまだのものは
    ``unknown``（sample モードでは存在のみ=present、full モードでは unvalidated）。
    """
    if race.race_id in present_keys:
        if health is None:
            return "present"
        if race.race_id in health.get("na", ()):
            return "na"                                 # スクレイパーが「存在しない」と判断して保存したスタブ
        if race.race_id in health["ok"]:
            return "healthy"
        if race.race_id in health["bad"]:
            return "invalid"
        return unknown
    if race.race_id in na:
        return "na"
    if race.date:
        if now < due_time(spec.due, datetime.strptime(race.date, "%Y%m%d").date()):
            return "pending"
    return "missing"


def _blank() -> dict[str, int]:
    return {"present": 0, "healthy": 0, "invalid": 0, "unvalidated": 0, "missing": 0, "pending": 0, "na": 0}


def race_coverage(env: str, universe: dict[str, Race], present: dict[str, dict[str, dict[str, float]]],
                  now: datetime, *, max_gaps: int = 5000, health: dict[str, dict] | None = None,
                  unknown_state: str = "present") -> dict[str, Any]:
    cats = [c for c in RACE_CATEGORIES if c.level[env] != "skip"]
    by_year: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_blank))
    by_period: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_blank))
    gaps: list[dict] = []
    gap_total: dict[str, int] = defaultdict(int)
    invalid_total: dict[str, int] = defaultdict(int)
    unvalidated_total: dict[str, int] = defaultdict(int)
    race_ids: dict[str, dict[str, Any]] = {c.name: {"missing": [], "invalid": {}, "invalid_detail": {}, "unvalidated": [], "pending": []} for c in cats}
    race_status: dict[str, dict[str, str]] = defaultdict(dict)
    health = health or {}
    na_cache: dict[tuple[str, str], set[str]] = {}
    key_sets: dict[tuple[str, str], set[str]] = {}
    for rid, race in sorted(universe.items()):
        y = rid[:4]
        for spec in cats:
            if not _in_scope(spec, race, now):
                continue
            na = na_cache.setdefault((spec.name, y), load_not_available(spec.name, y))
            keys = key_sets.get((spec.name, y))
            if keys is None:
                keys = key_sets[(spec.name, y)] = set(present.get(spec.name, {}).get(y, {}))
            st = classify(spec, race, keys, na, now, health=health.get(spec.name), unknown=unknown_state)
            by_year[y][spec.name][st] += 1
            by_period[_period(race)][spec.name][st] += 1
            race_status[rid][spec.name] = st
            ids = race_ids[spec.name]
            if st == "missing":
                gap_total[spec.name] += 1
                ids["missing"].append(rid)
            elif st == "invalid":
                invalid_total[spec.name] += 1
                ids["invalid"][rid] = health[spec.name]["bad"].get(rid, [])
                ids["invalid_detail"][rid] = health[spec.name].get("examples", {}).get(rid, [])
            elif st == "unvalidated":
                unvalidated_total[spec.name] += 1
                ids["unvalidated"].append(rid)
            elif st == "pending":
                ids["pending"].append(rid)
            if st in ("missing", "invalid") and len(gaps) < max_gaps:
                gaps.append({"category": spec.name, "race_id": rid, "date": race.date, "period": _period(race),
                             "venue": VENUE.get(rid[4:6], rid[4:6]), "round": int(rid[10:12]),
                             "confidence": race.confidence, "level": spec.level[env], "task": spec.task, "status": st,
                             "issues": ids["invalid"].get(rid, []) if st == "invalid" else []})
    return {
        "categories": [{"name": c.name, "label": c.label, "level": c.level[env], "due": c.due} for c in cats],
        "by_year": {y: dict(v) for y, v in sorted(by_year.items())},
        "by_period": {p: dict(v) for p, v in sorted(by_period.items())},
        "gaps": gaps,
        "gap_total": dict(gap_total),
        "invalid_total": dict(invalid_total),
        "unvalidated_total": dict(unvalidated_total),
        "race_ids": race_ids,
        "race_status": {k: v for k, v in race_status.items()},
        "universe": {"races": len(universe),
                     "by_confidence": {k: sum(1 for r in universe.values() if r.confidence == k)
                                       for k in ("confirmed", "present", "inferred")}},
    }


def freshness(present: dict[str, dict[str, dict[str, float]]], now: datetime) -> dict[str, dict]:
    out = {}
    for cat, years in present.items():
        latest = max((ts for ys in years.values() for ts in ys.values()), default=0.0)
        if latest:
            dt = datetime.fromtimestamp(latest, tz=JST)
            out[cat] = {"latest": dt.strftime("%Y-%m-%d %H:%M"), "age_hours": round((now - dt).total_seconds() / 3600, 1),
                        "count": sum(len(ys) for ys in years.values())}
        else:
            out[cat] = {"latest": None, "age_hours": None, "count": sum(len(ys) for ys in years.values())}
    return out


# ── 馬（評価窓の出走馬）─────────────────────────────────────────────────

def horse_coverage(env: str, storage: Any, universe: dict[str, Race], now: datetime, *,
                   past_days: int = HORSE_WINDOW_PAST_DAYS, future_days: int = HORSE_WINDOW_FUTURE_DAYS,
                   races_max: int = HORSE_RACES_MAX) -> dict[str, Any]:
    specs = [c for c in HORSE_CATEGORIES if c.level[env] != "skip"]
    lo = (now.date() - timedelta(days=past_days)).strftime("%Y%m%d")
    hi = (now.date() + timedelta(days=future_days)).strftime("%Y%m%d")
    races = sorted((r for r in universe.values() if r.date and lo <= r.date <= hi), key=lambda r: r.race_id)
    truncated = len(races) > races_max
    horses: dict[str, str] = {}      # horse_id → 最初に出走するレース
    unreadable = 0
    for r in races[:races_max]:
        d = storage.load("race_shutuba", r.race_id)
        if not d:
            unreadable += 1
            continue
        for e in d.get("entries", []):
            hid = str(e.get("horse_id") or "")
            if hid:
                horses.setdefault(hid, r.race_id)
    years = sorted({h[:4] for h in horses})
    presence = collect_presence(storage, [c.name for c in specs], years) if specs else {}
    cats_out, gaps = [], []
    for spec in specs:
        miss = [h for h in sorted(horses) if h not in presence[spec.name].get(h[:4], {})]
        cats_out.append({"name": spec.name, "label": spec.label, "level": spec.level[env], "task": spec.task,
                         "expected": len(horses), "missing": len(miss)})
        gaps += [{"category": spec.name, "horse_id": h, "race_id": horses[h], "level": spec.level[env],
                  "task": spec.task} for h in miss]
    return {"window": {"from": lo, "to": hi}, "races_examined": min(len(races), races_max),
            "races_truncated": truncated, "shutuba_unreadable": unreadable, "horses": len(horses),
            "categories": cats_out, "gaps": gaps[:5000]}
