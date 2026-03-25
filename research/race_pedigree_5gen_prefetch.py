"""
レース単位で horse_pedigree_5gen を段階的に揃える（最優先キュー投入）。

1. **出走馬** — 出馬表の各馬の 5 世代血統を先に取得・保存する。
2. **第1階層** — 保存済み血統の ancestors のうち、sire_aptitude_note に解決できる「種牡馬」
   の horse_id をユニーク化し、欠損分だけ取得する。
3. **第2階層** — 第1階層の各頭の血統表から同様にナレッジ種牡馬 ID を集め、欠損分だけ取得する。

ブレンド本体は各出走馬の 1 レコード内の祖先名を解決するが、第2階層までの種牡馬の
血統JSONが揃っていると、将来の拡張や一貫したデータ整備に使える。いっぺんに積まず段階を分ける。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ストレージ確認フェーズ: GCS 往復が多いためバッチ並列 + 環境で調整可
_PED5_CLOSURE_MAX_WORKERS = max(4, min(64, int(os.environ.get("PED5_CLOSURE_MAX_WORKERS", "32"))))
_PED5_CLOSURE_BATCH = max(8, min(256, int(os.environ.get("PED5_CLOSURE_BATCH", "96"))))

from research.sire_aptitude_note import resolve_sire_name, vector_for_canonical
from scraper.job_queue import (
    PRIORITY_URGENT_PEDIGREE_5GEN,
    QUEUE_DIR,
    ScrapeJobQueue,
    _effective_dedupe_key,
)

logger = logging.getLogger(__name__)

SESSION_DIR = QUEUE_DIR / "ensure_5gen_sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# api.app._is_jra_race と同条件（循環 import 回避のためローカル定義）
_JRA_PLACE_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}


def _is_jra_race_id(race_id: str) -> bool:
    rid = (race_id or "").strip()
    return len(rid) >= 6 and rid[4:6] in _JRA_PLACE_CODES


def list_jra_race_ids_for_date(storage: Any, date_stem: str) -> list[str]:
    """race_lists の1日分から JRA レースIDのみ（重複なし・ファイル上の順）。"""
    data = storage.load("race_lists", date_stem)
    if not data:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for r in data.get("races") or []:
        rid = (r.get("race_id") or "").strip()
        if not rid or not _is_jra_race_id(rid):
            continue
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def _daterange_inclusive_ymd(date_from: str, date_to: str) -> list[str]:
    a = datetime.strptime(date_from.strip(), "%Y%m%d")
    b = datetime.strptime(date_to.strip(), "%Y%m%d")
    if a > b:
        a, b = b, a
    out: list[str] = []
    cur = a
    while cur <= b:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


def batch_race_pedigree_5gen_date_range(
    storage: Any,
    date_from: str,
    date_to: str,
    *,
    dry_run: bool = False,
    max_races: int = 500,
) -> dict[str, Any]:
    """
    開催日 YYYYMMDD（inclusive）の範囲について、race_lists の JRA 各レースに対し
    start_race_pedigree_prefetch を順に実行する（note-aptitude-race の race-ensure-5gen と同じ。
    """
    df = (date_from or "").strip()
    dt = (date_to or "").strip()
    if len(df) != 8 or not df.isdigit():
        return {"ok": False, "error": "date_from は YYYYMMDD で指定してください"}
    if len(dt) != 8 or not dt.isdigit():
        return {"ok": False, "error": "date_to は YYYYMMDD で指定してください"}
    max_races = max(1, min(int(max_races or 500), 5000))

    dates = _daterange_inclusive_ymd(df, dt)
    race_ids: list[str] = []
    seen: set[str] = set()
    for d in dates:
        for rid in list_jra_race_ids_for_date(storage, d):
            if rid in seen:
                continue
            seen.add(rid)
            race_ids.append(rid)
            if len(race_ids) >= max_races:
                break
        if len(race_ids) >= max_races:
            break

    meta: dict[str, Any] = {
        "date_from": df,
        "date_to": dt,
        "days_in_range": len(dates),
        "races_enumerated": len(race_ids),
        "capped": len(race_ids) >= max_races,
        "max_races": max_races,
    }

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "meta": meta,
            "sample_race_ids": race_ids[:50],
            "message": (
                f"ドライラン: {len(race_ids)} レースを処理対象に列挙しました（上限 {max_races}）。"
            ),
        }

    races_processed = 0
    total_enqueued = 0
    races_with_jobs = 0
    races_already_complete = 0
    errors: list[dict[str, str]] = []
    session_ids: list[str] = []

    for rid in race_ids:
        try:
            out = start_race_pedigree_prefetch(storage, rid)
            races_processed += 1
            enq = int(out.get("enqueued") or 0)
            total_enqueued += enq
            if enq > 0:
                races_with_jobs += 1
                sid = (out.get("session_id") or "").strip()
                if sid:
                    session_ids.append(sid)
            else:
                races_already_complete += 1
        except Exception as e:
            logger.exception("batch 5gen race_id=%s", rid)
            errors.append({"race_id": rid, "error": str(e)})

    return {
        "ok": True,
        "dry_run": False,
        "meta": meta,
        "races_processed": races_processed,
        "races_with_jobs": races_with_jobs,
        "races_already_complete": races_already_complete,
        "total_jobs_enqueued": total_enqueued,
        "session_ids_sample": session_ids[:20],
        "session_count": len(session_ids),
        "error_count": len(errors),
        "errors_sample": errors[:25],
    }


def _pedigree_ok(rec: dict | None) -> bool:
    if not rec:
        return False
    anc = rec.get("ancestors") or []
    return len(anc) >= 5


def _note_stallion_horse_ids_ordered(rec: dict | None) -> list[str]:
    """血統1件から、5gen ブレンドと同条件で因子を持つ種牡馬の horse_id（初出順・重複なし）。"""
    out: list[str] = []
    seen: set[str] = set()
    if not rec:
        return out
    for anc in rec.get("ancestors") or []:
        name = (anc.get("name") or "").strip()
        hid = (anc.get("horse_id") or "").strip()
        if not hid:
            continue
        canon = resolve_sire_name(name)
        if not canon:
            continue
        vec = vector_for_canonical(canon)
        if not any(vec.values()):
            continue
        if hid not in seen:
            seen.add(hid)
            out.append(hid)
    return out


def _parallel_scan_pedigree_missing(
    storage: Any, ordered_unique_ids: list[str]
) -> tuple[list[str], dict[str, int]]:
    """ordered_unique_ids についてストレージを並列読み、欠損 horse_id 一覧を返す。"""
    missing: list[str] = []
    miss_set: set[str] = set()
    stats: dict[str, int] = {
        "storage_loads": 0,
        "bfs_visits": 0,
        "graph_nodes_seen": len(ordered_unique_ids),
        "parallel_max_workers": _PED5_CLOSURE_MAX_WORKERS,
    }

    def _load_one(hid: str) -> tuple[str, dict | None]:
        return hid, storage.load("horse_pedigree_5gen", hid)

    if not ordered_unique_ids:
        return missing, stats

    with ThreadPoolExecutor(max_workers=_PED5_CLOSURE_MAX_WORKERS) as pool:
        for offset in range(0, len(ordered_unique_ids), _PED5_CLOSURE_BATCH):
            batch = ordered_unique_ids[offset : offset + _PED5_CLOSURE_BATCH]
            stats["bfs_visits"] += len(batch)
            stats["storage_loads"] += len(batch)
            for hid, rec in pool.map(_load_one, batch):
                if not _pedigree_ok(rec):
                    if hid not in miss_set:
                        miss_set.add(hid)
                        missing.append(hid)
    return missing, stats


def horse_ids_missing_pedigree_for_runners(
    storage: Any, runner_hids: list[str]
) -> tuple[list[str], dict[str, int]]:
    """
    出走馬だけを対象に horse_pedigree_5gen の有無を確認する。
    """
    seeds = list(
        dict.fromkeys((h or "").strip() for h in runner_hids if (h or "").strip())
    )
    missing, stats = _parallel_scan_pedigree_missing(storage, seeds)
    stats["runner_ids_checked"] = len(seeds)
    return missing, stats


def _collect_tier1_note_stallion_ids(storage: Any, runner_ids: list[str]) -> list[str]:
    """出走馬の保存済み血統から、ナレッジに載る種牡馬 horse_id を初出順でユニーク化。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for rid in runner_ids:
        rid = (rid or "").strip()
        if not rid:
            continue
        try:
            rec = storage.load("horse_pedigree_5gen", rid)
        except Exception:
            rec = None
        for hid in _note_stallion_horse_ids_ordered(rec):
            if hid not in seen:
                seen.add(hid)
                ordered.append(hid)
    return ordered


def _collect_tier2_note_stallion_ids(storage: Any, tier1_ids: list[str]) -> list[str]:
    """第1階層各頭の血統からナレッジ種牡馬 horse_id をユニーク化（ストレージ読みは並列）。"""
    if not tier1_ids:
        return []
    ordered: list[str] = []
    seen: set[str] = set()

    def _load_one(hid: str) -> dict | None:
        try:
            return storage.load("horse_pedigree_5gen", hid)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=_PED5_CLOSURE_MAX_WORKERS) as pool:
        recs = list(pool.map(_load_one, tier1_ids))
    for rec in recs:
        for hid in _note_stallion_horse_ids_ordered(rec):
            if hid not in seen:
                seen.add(hid)
                ordered.append(hid)
    return ordered


def enqueue_urgent_pedigree_jobs(
    horse_ids: list[str],
    *,
    source_label: str = "race_ensure_5gen",
) -> tuple[list[str], list[str]]:
    """
    各馬を最優先ジョブとして投入。戻り値: (job_ids, horse_ids_enqueued)
    """
    queue = ScrapeJobQueue()
    job_ids: list[str] = []
    enq: list[str] = []
    for hid in horse_ids:
        hid = (hid or "").strip()
        if not hid:
            continue
        body = {
            "job_kind": "horse",
            "target_id": hid,
            "tasks": ["horse_pedigree_5gen"],
            "smart_skip": False,
            "priority": PRIORITY_URGENT_PEDIGREE_5GEN,
            "job_label": f"urgent:5gen:{hid}",
        }
        try:
            r = queue.add_job(body)
        except ValueError as e:
            logger.warning("enqueue skip %s: %s", hid, e)
            continue
        jid = r.get("job_id")
        if jid:
            job_ids.append(str(jid))
        enq.append(hid)
    logger.info(
        "[%s] 5gen 最優先キュー投入: %d 頭 (job_ids=%d)",
        source_label,
        len(enq),
        len(job_ids),
    )
    return job_ids, enq


def save_prefetch_session(payload: dict) -> str:
    sid = str(uuid.uuid4())
    path = SESSION_DIR / f"{sid}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return sid


def load_prefetch_session(session_id: str) -> dict | None:
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def update_prefetch_session(session_id: str, updates: dict) -> None:
    data = load_prefetch_session(session_id) or {}
    data.update(updates)
    path = SESSION_DIR / f"{session_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cancel_race_pedigree_prefetch_session(session_id: str) -> dict[str, Any]:
    """
    セッションに紐づくジョブのうち、キュー上 status=pending のものだけ削除する。
    実行中 (running) のジョブは止められない。セッションに cancelled を記録する。
    """
    sid = (session_id or "").strip()
    if not sid:
        return {"ok": False, "error": "session_id が空です"}
    sess = load_prefetch_session(sid)
    if not sess:
        return {"ok": False, "error": "session_not_found", "session_id": sid}
    if sess.get("cancelled"):
        return {
            "ok": True,
            "session_id": sid,
            "removed_pending": 0,
            "already_cancelled": True,
        }
    job_id_set = {str(x) for x in (sess.get("job_ids") or []) if x}
    queue = ScrapeJobQueue()
    jobs = queue.load_queue()
    kept: list[dict] = []
    removed = 0
    for j in jobs:
        jid = str(j.get("job_id") or "")
        if jid in job_id_set and j.get("status") == "pending":
            removed += 1
            continue
        kept.append(j)
    if removed:
        queue.save_queue(kept)
    update_prefetch_session(
        sid,
        {"cancelled": True, "cancelled_at": datetime.now().isoformat()},
    )
    return {"ok": True, "session_id": sid, "removed_pending": removed}


def _horse_ped5_dedupe_key(horse_id: str) -> str:
    """ScrapeJobQueue._normalize_incoming_job と同じ dedupe（tasks ソート済み1件）。"""
    hid = (horse_id or "").strip()
    return f"horse:{hid}:horse_pedigree_5gen"


def _build_pedigree_name_map(storage: Any, runner_ids: list[str]) -> dict[str, str]:
    """出走馬 + ナレッジ種牡馬(tier1) の horse_pedigree_5gen から horse_id->name マップを構築。"""
    name_map: dict[str, str] = {}

    def _load_one(hid: str) -> dict | None:
        try:
            return storage.load("horse_pedigree_5gen", hid)
        except Exception:
            return None

    def _extract_names(recs: list[dict | None]) -> None:
        for rec in recs:
            if not rec:
                continue
            horse_name = (rec.get("horse_name") or "").strip()
            horse_id = (rec.get("horse_id") or "").strip()
            if horse_id and horse_name and horse_id not in name_map:
                name_map[horse_id] = horse_name
            for anc in rec.get("ancestors") or []:
                aid = (anc.get("horse_id") or "").strip()
                aname = (anc.get("name") or "").strip()
                if aid and aname and aid not in name_map:
                    name_map[aid] = aname

    all_ids = list(dict.fromkeys((h or "").strip() for h in runner_ids if (h or "").strip()))
    if not all_ids:
        return name_map

    mx = min(_PED5_CLOSURE_MAX_WORKERS, max(4, len(all_ids)))
    with ThreadPoolExecutor(max_workers=mx) as pool:
        runner_recs = list(pool.map(_load_one, all_ids))
    _extract_names(runner_recs)

    tier1_ids = _collect_tier1_note_stallion_ids(storage, all_ids)
    tier1_to_load = [h for h in tier1_ids if h not in {(r or {}).get("horse_id") for r in runner_recs if r}]
    if tier1_to_load:
        mx2 = min(_PED5_CLOSURE_MAX_WORKERS, max(4, len(tier1_to_load)))
        with ThreadPoolExecutor(max_workers=mx2) as pool:
            tier1_recs = list(pool.map(_load_one, tier1_to_load))
        _extract_names(tier1_recs)

    return name_map


def _display_label_for_horse(
    storage: Any, horse_id: str, race_disp_map: dict[str, str],
    pedigree_name_map: dict[str, str] | None = None,
) -> str:
    """UI 用の短い表示名（出走表 → horse_result → 血統ancestors → 馬名未取得）。"""
    hid = (horse_id or "").strip()
    if not hid:
        return ""
    if hid in race_disp_map:
        return race_disp_map[hid]
    if pedigree_name_map and hid in pedigree_name_map:
        return pedigree_name_map[hid]
    try:
        hr = storage.load("horse_result", hid)
    except Exception:
        hr = None
    name = (hr.get("horse_name") or "").strip() if isinstance(hr, dict) else ""
    if name:
        return name
    if pedigree_name_map is None:
        try:
            ped = storage.load("horse_pedigree_5gen", hid)
        except Exception:
            ped = None
        if ped:
            pname = (ped.get("horse_name") or "").strip()
            if pname:
                return pname
            for anc in ped.get("ancestors") or []:
                if (anc.get("horse_id") or "").strip() == hid:
                    aname = (anc.get("name") or "").strip()
                    if aname:
                        return aname
    return hid


def _summarize_batch_targets_ja(
    prefetch_stage: str,
    horse_ids: list[str],
    disp_map: dict[str, str],
    storage: Any | None = None,
    *,
    pedigree_name_map: dict[str, str] | None = None,
) -> str:
    """UI 用: 本バッチに含まれる馬の短い一覧。"""
    ids = [(h or "").strip() for h in horse_ids if (h or "").strip()]
    if not ids:
        return ""
    pnm = pedigree_name_map or {}
    stage = (prefetch_stage or "").strip()
    if stage == "runners":
        shown: list[str] = []
        for hid in ids[:10]:
            if disp_map.get(hid):
                shown.append(disp_map[hid])
            elif storage is not None:
                shown.append(_display_label_for_horse(storage, hid, disp_map, pnm))
            elif hid in pnm:
                shown.append(pnm[hid])
            else:
                shown.append(hid)
        more = len(ids) - len(shown)
        return (
            "本バッチの出走馬（欠損分・最大10件表示）: "
            + "、".join(shown)
            + (f" …他{more}頭" if more > 0 else "")
        )
    if storage is not None:
        shown_s: list[str] = []
        for hid in ids[:10]:
            shown_s.append(_display_label_for_horse(storage, hid, disp_map, pnm))
        more = len(ids) - len(shown_s)
        return (
            "本バッチの対象（種牡馬・最大10件表示）: "
            + "、".join(shown_s)
            + (f" …他{more}頭" if more > 0 else "")
        )
    return f"本バッチ: 種牡馬血統 {len(ids)} 件"


def _race_horse_display_map(storage: Any, race_id: str) -> dict[str, str]:
    """horse_id -> 「枠X Y番 馬名」"""
    from research.note_aptitude_race_map import load_race_entries

    rid = (race_id or "").strip()
    if not rid:
        return {}
    entries, _ = load_race_entries(storage, rid)
    m: dict[str, str] = {}
    for e in entries:
        hid = (e.get("horse_id") or "").strip()
        if not hid:
            continue
        bn = int(e.get("bracket_number") or 0)
        hn = int(e.get("horse_number") or 0)
        name = (e.get("horse_name") or "").strip() or hid
        segs: list[str] = []
        if bn > 0:
            segs.append(f"枠{bn}")
        if hn > 0:
            segs.append(f"{hn}番")
        segs.append(name)
        m[hid] = " ".join(segs)
    return m


def _race_runner_pedigree_progress(storage: Any, race_id: str) -> dict[str, Any]:
    """対象レースの出走馬のうち horse_pedigree_5gen が充足している頭数。"""
    from research.note_aptitude_race_map import load_race_entries

    rid = (race_id or "").strip()
    if not rid:
        return {
            "runner_count": 0,
            "runners_pedigree_ready": 0,
            "runner_pedigree_progress_pct": 100,
            "runner_progress_label_ja": "",
        }
    entries, _ = load_race_entries(storage, rid)
    runner_ids: list[str] = []
    seen: set[str] = set()
    for e in entries:
        hid = (e.get("horse_id") or "").strip()
        if hid and hid not in seen:
            seen.add(hid)
            runner_ids.append(hid)
    total = len(runner_ids)
    if total == 0:
        return {
            "runner_count": 0,
            "runners_pedigree_ready": 0,
            "runner_pedigree_progress_pct": 100,
            "runner_progress_label_ja": "",
        }

    def _one_ok(hid: str) -> bool:
        try:
            rec = storage.load("horse_pedigree_5gen", hid)
        except Exception:
            rec = None
        return _pedigree_ok(rec)

    mx = min(_PED5_CLOSURE_MAX_WORKERS, max(4, total))
    with ThreadPoolExecutor(max_workers=mx) as pool:
        bits = list(pool.map(_one_ok, runner_ids))
    ready = sum(1 for b in bits if b)
    pct = min(100, int(100 * ready / total))
    return {
        "runner_count": total,
        "runners_pedigree_ready": ready,
        "runner_pedigree_progress_pct": pct,
        "runner_progress_label_ja": (
            f"出走馬の5世代血統（レース全体）: {ready} / {total} 頭 取得済み"
        ),
    }


def _default_storage_for_session_progress() -> Any:
    from pathlib import Path

    from scraper.storage import HybridStorage

    root = Path(__file__).resolve().parent.parent
    return HybridStorage(str(root))


def session_progress(session_id: str, storage: Any | None = None) -> dict[str, Any]:
    """セッションに紐づくジョブの集計（キュー JSON を読む）。

    完了・失敗ジョブが scrape-queue/clear 等でキューから消えたあとも、
    job_id で引けない分は「クリア済み完了」として数える（進捗バーが 0 のまま固まるのを防ぐ）。
    job_id と不一致でも、同一馬の dedupe で現在のキュー行を引き当てる。
    """
    sess = load_prefetch_session(session_id)
    if not sess:
        return {"error": "session_not_found", "session_id": session_id}

    if storage is None:
        storage = _default_storage_for_session_progress()

    if sess.get("cancelled"):
        rid = str(sess.get("race_id") or "").strip()
        stage = str(sess.get("prefetch_stage") or "")
        disp_map = _race_horse_display_map(storage, rid) if rid else {}
        pnm_cn = sess.get("pedigree_name_map") or {}
        horse_ids_sess_cn = [(h or "").strip() for h in (sess.get("horse_ids") or [])]
        batch_targets_summary_ja = _summarize_batch_targets_ja(
            stage, horse_ids_sess_cn, disp_map, storage,
            pedigree_name_map=pnm_cn,
        )
        total_j = len(sess.get("job_ids") or [])
        q = ScrapeJobQueue()
        rpm_cn = _race_runner_pedigree_progress(storage, rid) if rid else {}
        return {
            "session_id": session_id,
            "race_id": sess.get("race_id", ""),
            "total_jobs": total_j,
            "completed": 0,
            "failed": 0,
            "pending": 0,
            "running_count": 0,
            "current_job_label": "",
            "is_running_worker": q.is_locked(),
            "horse_ids": sess.get("horse_ids", []),
            "all_done": True,
            "user_cancelled": True,
            "updated_at": datetime.now().isoformat(),
            "phase": "cancelled",
            "phase_label_ja": "一時停止",
            "phase_detail_ja": (
                "未実行のキュー待ちジョブは取り下げました。"
                "実行中のジョブは完了まで続く場合があります。"
            ),
            "progress_pct": 100 if total_j == 0 else 0,
            "scan_stats": sess.get("scan_stats"),
            "plan_duration_ms": sess.get("plan_duration_ms"),
            "prefetch_stage": sess.get("prefetch_stage"),
            "prefetch_stage_ja": sess.get("prefetch_stage_ja"),
            "current_target_horse_id": "",
            "current_scrape_display_ja": "",
            "batch_targets_summary_ja": batch_targets_summary_ja,
            "next_pending_display_ja": "",
            **rpm_cn,
        }

    queue = ScrapeJobQueue()
    jobs = queue.load_queue()
    jm = {str(j.get("job_id")): j for j in jobs if j.get("job_id")}
    by_dedupe: dict[str, dict] = {}
    for j in jobs:
        dk = _effective_dedupe_key(j)
        if dk:
            by_dedupe[dk] = j

    job_ids = [str(x) for x in (sess.get("job_ids") or [])]
    horse_ids_sess = [(h or "").strip() for h in (sess.get("horse_ids") or [])]

    completed = failed = pending = running = 0
    current_label = ""
    current_running_job: dict[str, Any] | None = None
    for idx, jid in enumerate(job_ids):
        j = jm.get(jid)
        if j is None and idx < len(horse_ids_sess):
            hid = horse_ids_sess[idx]
            if hid:
                j = by_dedupe.get(_horse_ped5_dedupe_key(hid))
        if j is None:
            # キューに行がない = 完了後に clear されたか、古い job_id のみ残存
            completed += 1
            continue
        st = j.get("status") or ""
        if st == "completed":
            completed += 1
        elif st == "failed":
            failed += 1
        elif st == "running":
            running += 1
            if current_running_job is None:
                current_running_job = j
        else:
            pending += 1

    total = len(job_ids)
    done = completed + failed
    # job_ids が空（投入スキップのみ等）のセッションは待機不要
    all_done = total == 0 or done >= total
    worker_locked = queue.is_locked()
    progress_pct = 100 if total == 0 else min(100, int(100 * done / total))

    urgent_worker_locked = queue.is_locked_urgent()

    phase = "complete"
    phase_label_ja = "完了"
    detail_ja = ""
    if not all_done and total > 0:
        if running > 0:
            phase = "scraping"
            phase_label_ja = "スクレイピング中"
            lane = "ファストレーン" if urgent_worker_locked else "最優先キュー"
            detail_ja = f"netkeiba から horse_pedigree_5gen を取得しています（{lane}）。"
        elif urgent_worker_locked or worker_locked:
            phase = "queued_busy"
            phase_label_ja = "キュー待ち"
            if urgent_worker_locked:
                detail_ja = "ファストレーンワーカーが他の最優先ジョブを処理中です。次に実行されます。"
            else:
                detail_ja = "通常ワーカーが他ジョブを処理中です。ファストレーンで割り込み実行を試みています。"
        else:
            phase = "queued_idle"
            phase_label_ja = "キュー待ち"
            detail_ja = "ワーカーがジョブを取りに行きます。しばらくお待ちください。"

    rid = str(sess.get("race_id") or "").strip()
    stage = str(sess.get("prefetch_stage") or "")
    disp_map = _race_horse_display_map(storage, rid) if rid else {}
    pnm = sess.get("pedigree_name_map") or {}
    rpm = _race_runner_pedigree_progress(storage, rid) if rid else {}

    current_target_horse_id = ""
    current_scrape_display_ja = ""
    if current_running_job is not None:
        current_target_horse_id = str(
            current_running_job.get("target_id") or ""
        ).strip()
        nm = (
            _display_label_for_horse(storage, current_target_horse_id, disp_map, pnm)
            if current_target_horse_id
            else ""
        )
        if current_target_horse_id:
            current_label = nm
        if stage == "runners":
            current_scrape_display_ja = f"実行中: {nm}" if nm else ""
        elif current_target_horse_id:
            current_scrape_display_ja = (
                f"実行中: {nm}（ナレッジ種牡馬・5世代血統）"
                if nm
                else ""
            )

    batch_targets_summary_ja = _summarize_batch_targets_ja(
        stage, horse_ids_sess, disp_map, storage,
        pedigree_name_map=pnm,
    )

    next_pending_display_ja = ""
    if not all_done and running == 0 and total > 0:
        for idx, jid in enumerate(job_ids):
            j = jm.get(jid)
            if j is None and idx < len(horse_ids_sess):
                h = horse_ids_sess[idx]
                if h:
                    j = by_dedupe.get(_horse_ped5_dedupe_key(h))
            if j is None or j.get("status") != "pending":
                continue
            tid = str(j.get("target_id") or "").strip()
            if not tid:
                break
            nm = _display_label_for_horse(storage, tid, disp_map, pnm)
            next_pending_display_ja = f"次に実行予定: {nm}"
            break

    return {
        "session_id": session_id,
        "race_id": sess.get("race_id", ""),
        "total_jobs": total,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "running_count": running,
        "current_job_label": current_label,
        "is_running_worker": worker_locked,
        "is_running_urgent_worker": urgent_worker_locked,
        "horse_ids": sess.get("horse_ids", []),
        "all_done": all_done,
        "updated_at": datetime.now().isoformat(),
        "phase": phase,
        "phase_label_ja": phase_label_ja,
        "phase_detail_ja": detail_ja,
        "progress_pct": progress_pct,
        "scan_stats": sess.get("scan_stats"),
        "plan_duration_ms": sess.get("plan_duration_ms"),
        "prefetch_stage": sess.get("prefetch_stage"),
        "prefetch_stage_ja": sess.get("prefetch_stage_ja"),
        "current_target_horse_id": current_target_horse_id,
        "current_scrape_display_ja": current_scrape_display_ja,
        "batch_targets_summary_ja": batch_targets_summary_ja,
        "next_pending_display_ja": next_pending_display_ja,
        **rpm,
    }


def plan_race_pedigree_prefetch(storage: Any, race_id: str) -> dict[str, Any]:
    """出走馬 → ナレッジ種牡馬第1階層 → 第2階層の順で欠損を計算（キューは投入しない）。"""
    import time

    from research.note_aptitude_race_map import load_race_entries

    rid = (race_id or "").strip()
    t0 = time.perf_counter()
    entries, src = load_race_entries(storage, rid)
    runner_ids: list[str] = []
    for e in entries:
        hid = (e.get("horse_id") or "").strip()
        if hid:
            runner_ids.append(hid)

    base = {
        "race_id": rid,
        "entry_source": src,
        "runner_count": len(runner_ids),
        "_runner_ids": runner_ids,
    }

    runner_missing, stats_run = horse_ids_missing_pedigree_for_runners(storage, runner_ids)
    if runner_missing:
        plan_ms = int((time.perf_counter() - t0) * 1000)
        stats_run["prefetch_stage"] = "runners"
        stats_run["prefetch_stage_ja"] = "出走馬の5世代血統"
        return {
            **base,
            "missing_horse_ids": runner_missing,
            "missing_count": len(runner_missing),
            "scan_stats": stats_run,
            "plan_duration_ms": plan_ms,
            "prefetch_stage": "runners",
            "prefetch_stage_ja": "出走馬の5世代血統",
        }

    tier1_ordered = _collect_tier1_note_stallion_ids(storage, runner_ids)
    tier1_missing, stats_t1 = _parallel_scan_pedigree_missing(storage, tier1_ordered)
    stats_t1["tier1_note_stallion_candidates"] = len(tier1_ordered)
    stats_t1["prefetch_stage"] = "stallion_tier1"
    stats_t1["prefetch_stage_ja"] = "第1階層（ナレッジ種牡馬）"
    if tier1_missing:
        plan_ms = int((time.perf_counter() - t0) * 1000)
        return {
            **base,
            "missing_horse_ids": tier1_missing,
            "missing_count": len(tier1_missing),
            "scan_stats": stats_t1,
            "plan_duration_ms": plan_ms,
            "prefetch_stage": "stallion_tier1",
            "prefetch_stage_ja": "第1階層（ナレッジ種牡馬）",
        }

    tier2_ordered = _collect_tier2_note_stallion_ids(storage, tier1_ordered)
    tier2_missing, stats_t2 = _parallel_scan_pedigree_missing(storage, tier2_ordered)
    stats_t2["tier1_note_stallion_candidates"] = len(tier1_ordered)
    stats_t2["tier2_note_stallion_candidates"] = len(tier2_ordered)
    stats_t2["prefetch_stage"] = "stallion_tier2"
    stats_t2["prefetch_stage_ja"] = "第2階層（種牡馬の血統内ナレッジ種牡馬）"
    if tier2_missing:
        plan_ms = int((time.perf_counter() - t0) * 1000)
        return {
            **base,
            "missing_horse_ids": tier2_missing,
            "missing_count": len(tier2_missing),
            "scan_stats": stats_t2,
            "plan_duration_ms": plan_ms,
            "prefetch_stage": "stallion_tier2",
            "prefetch_stage_ja": "第2階層（種牡馬の血統内ナレッジ種牡馬）",
        }

    plan_ms = int((time.perf_counter() - t0) * 1000)
    scan_done = {
        "prefetch_stage": "complete",
        "prefetch_stage_ja": "完了",
        "tier1_note_stallion_candidates": len(tier1_ordered),
        "tier2_note_stallion_candidates": len(tier2_ordered),
        "parallel_max_workers": _PED5_CLOSURE_MAX_WORKERS,
    }
    return {
        **base,
        "missing_horse_ids": [],
        "missing_count": 0,
        "scan_stats": scan_done,
        "plan_duration_ms": plan_ms,
        "prefetch_stage": "complete",
        "prefetch_stage_ja": "完了",
    }


def start_race_pedigree_prefetch(storage: Any, race_id: str) -> dict[str, Any]:
    """
    欠損があれば最優先ジョブを投入しセッションを返す。
    欠損ゼロなら session_id は空。
    """
    plan = plan_race_pedigree_prefetch(storage, race_id)
    missing = plan.get("missing_horse_ids") or []
    rid_plan = str(plan.get("race_id") or "")
    runner_ids = plan.pop("_runner_ids", [])
    disp_map = _race_horse_display_map(storage, rid_plan)
    ped_name_map = _build_pedigree_name_map(storage, runner_ids)
    rpm0 = _race_runner_pedigree_progress(storage, rid_plan)
    batch_targets_summary_ja = _summarize_batch_targets_ja(
        str(plan.get("prefetch_stage") or ""), missing, disp_map, storage,
        pedigree_name_map=ped_name_map,
    )
    if not missing:
        return {
            **plan,
            **rpm0,
            "session_id": "",
            "job_ids": [],
            "enqueued": 0,
            "message": "5世代血統は既に揃っています（出走馬・ナレッジ種牡馬2階層）",
            "batch_targets_summary_ja": batch_targets_summary_ja,
        }

    job_ids, _ = enqueue_urgent_pedigree_jobs(missing, source_label=plan["race_id"])
    stage_ja = str(plan.get("prefetch_stage_ja") or "").strip()
    payload = {
        "race_id": plan["race_id"],
        "created_at": datetime.now().isoformat(),
        "horse_ids": missing,
        "job_ids": job_ids,
        "runner_count": plan["runner_count"],
        "scan_stats": plan.get("scan_stats"),
        "plan_duration_ms": plan.get("plan_duration_ms"),
        "prefetch_stage": plan.get("prefetch_stage"),
        "prefetch_stage_ja": plan.get("prefetch_stage_ja"),
        "pedigree_name_map": ped_name_map,
    }
    sid = save_prefetch_session(payload)
    msg = f"最優先で {len(job_ids)} 件をキューに投入しました"
    if stage_ja:
        msg += f"（{stage_ja}）"
    return {
        **plan,
        **rpm0,
        "session_id": sid,
        "job_ids": job_ids,
        "enqueued": len(job_ids),
        "message": msg,
        "batch_targets_summary_ja": batch_targets_summary_ja,
    }
