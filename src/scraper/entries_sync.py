"""出馬表の段階的な収集（JRA の公示の流れに沿って、未来のレースの出馬表を取り、最新の状況を保持する）。

cron は 10 分おきにこのタスク（auto_scrape --task entries-sync）を呼ぶ。config/entries_schedule.json の checkpoint の時刻
（日曜夕の特別登録、月〜水の途中経過、木曜の出走馬決定、金曜16時／土曜10時の確定出馬表）から window 以内で、
未実施のときだけ実際に取りに行く（それ以外は即終了で netkeiba に触れない）。

- 取得は既存のキュー（race_shutuba、smart_skip=False で毎回取り直し。DEC-023・スロット・間隔はキューとクライアント側が守る）
- 取得後に段階（registered / entries / final / weighed）と差分（取消・騎手変更・斤量変更）を data/local/meta/entries_state/ に記録
  （最新状態＋履歴。race_shutuba の文書は最新で上書きされるが、履歴はここに残る）
- 今日のレースは対象外（開催日ランナーが担当）
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scraper import entries_phase as EP

logger = logging.getLogger("scraper.entries_sync")

ROOT = Path(__file__).resolve().parents[2]
HISTORY_KEEP = 40
CHANGES_KEEP = 60
DONE_KEEP = 200
WAIT_TIMEOUT_SEC = 1500.0  # 1 回の同期でキューを待つ最大（25 分）
ENTRIES_JOB_PRIORITY = -100  # 既定(0)より低い: 同じ時間帯の開催日ランナー（T-15 など）のジョブを先に処理させる


def state_dir() -> Path:
    return Path(os.environ.get("KEIBA_ENTRIES_STATE_DIR") or ROOT / "data" / "local" / "meta" / "entries_state")


def state_path(race_id: str) -> Path:
    return state_dir() / f"{race_id}.json"


def sync_state_path() -> Path:
    return state_dir().parent / "entries_sync_state.json"


def _atomic_write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_state(race_id: str) -> dict | None:
    try:
        return json.loads(state_path(race_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_sync_state() -> dict:
    try:
        return json.loads(sync_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"done": {}}


@contextmanager
def _lock():
    path = state_dir().parent / "entries_sync.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def update_state(prev: dict | None, prev_card: dict | None, card: dict | None, race_id: str, race_date: str, now: datetime,
                 list_entries_count: int | None, *, decided: tuple[str, str] = ("thu", "12:00")) -> tuple[dict, list[dict]]:
    """取得した出馬表から状態を更新する。戻り値: (新しい状態, 今回見つかった差分)。"""
    cls = EP.classify_entries(card, race_date, now, decided_weekday=decided[0], decided_time=decided[1])
    sig = EP.entries_signature(card)
    at = now.astimezone(EP.JST).isoformat(timespec="seconds")
    st = dict(prev or {})
    changes = EP.diff_entries(prev_card, card) if (prev_card and card) else []
    for c in changes:
        c["at"] = at
    history = list(st.get("history") or [])
    changed_content = (st.get("signature") != sig) or (st.get("phase") != cls["phase"])
    if changed_content:
        history.append({"at": at, "phase": cls["phase"], "n_entries": cls["n_entries"], "n_numbered": cls["n_numbered"], "signature": sig})
    st.update(
        race_id=race_id, date=EP.parse_date(race_date).isoformat() if EP.parse_date(race_date) else race_date,
        phase=cls["phase"], label=cls["label"], note=cls["note"], signature=sig,
        n_entries=cls["n_entries"], n_numbered=cls["n_numbered"], n_bracket=cls["n_bracket"], n_jockey=cls["n_jockey"],
        n_weight=cls["n_weight"], n_weight_carried=cls["n_weight_carried"], start_time=cls["start_time"],
        checked_at=at, list_entries_count=list_entries_count,
        history=history[-HISTORY_KEEP:],
        changes=(list(st.get("changes") or []) + changes)[-CHANGES_KEEP:],
    )
    if changed_content or not st.get("updated_at"):
        st["updated_at"] = at
    st.setdefault("first_seen_at", at)
    st.setdefault("first_seen_phase", cls["phase"])
    return st, changes


def _plan(storage: Any, dates: list[str], today, cp: dict, force_all: bool) -> list[dict]:
    """各レースを取り直すかを決める。"""
    plan: list[dict] = []
    for ymd in dates:
        rl = storage.load("race_lists", ymd) or {}
        rd = EP.parse_date(ymd)
        for r in rl.get("races", []) or []:
            rid = str(r.get("race_id") or "").strip()
            if not rid:
                continue
            from src.scraper.missing_races import is_jra_race_id

            if not is_jra_race_id(rid):
                continue
            prev = load_state(rid)
            cnt = r.get("entries_count")
            changed = prev is not None and cnt is not None and prev.get("list_entries_count") not in (None, cnt)
            go = force_all or EP.should_refetch(
                race_date=rd, today=today, prev_phase=(prev or {}).get("phase"),
                critical=bool(cp.get("critical")), list_count_changed=bool(changed),
            )
            plan.append({"race_id": rid, "date": ymd, "venue": str(r.get("venue") or ""), "round": int(r.get("round") or 0),
                         "race_name": str(r.get("race_name") or ""), "list_entries_count": cnt, "refetch": go,
                         "prev_phase": (prev or {}).get("phase")})
    return plan


def run_entries_sync(
    now: datetime | None = None, *, force: bool = False, dry_run: bool = False, dates: list[str] | None = None,
    storage: Any = None, queue: Any = None, schedule: dict | None = None, wait: bool = True,
    refresh_lists: bool = True,
) -> dict[str, Any]:
    now = (now or datetime.now(EP.JST)).astimezone(EP.JST)
    schedule = schedule or EP.load_schedule()
    sync = load_sync_state()
    cp = EP.due_checkpoint(now, schedule, sync.get("done", {}).keys())
    if force:
        cp = {"label": f"force {now:%Y-%m-%d %H:%M}", "critical": True, "weekday": EP._WEEKDAYS[now.weekday()], "time": f"{now:%H:%M}"}
    if cp is None:
        return {"status": "skipped", "reason": "not-due"}
    with _lock() as got:
        if not got:
            return {"status": "skipped", "reason": "already-running"}
        if storage is None:
            from src.scraper import auto_scrape_queue as _asq

            storage = _asq._storage()
        today = now.date()
        all_dates = list(storage.list_keys("race_lists"))
        tgt = dates or EP.target_dates(all_dates, today, int(schedule.get("lookahead_days", 10)))
        tgt = [d for d in tgt if EP.parse_date(d) and EP.parse_date(d) > today]  # 今日のレースは開催日ランナーが担当
        if not tgt:
            _mark_done(sync, cp, {"dates": []})
            return {"status": "ok", "checkpoint": cp["label"], "dates": [], "races": 0, "fetched": 0}

        refreshed: list[str] = []
        if refresh_lists and not dry_run:
            refreshed = _refresh_race_lists(storage, tgt, cp)

        plan = _plan(storage, tgt, today, cp, force_all=False)
        todo = [p for p in plan if p["refetch"]]
        result: dict[str, Any] = {
            "status": "ok", "checkpoint": cp["label"], "critical": bool(cp.get("critical")), "dates": tgt,
            "races": len(plan), "to_fetch": len(todo), "lists_refreshed": refreshed,
        }
        if dry_run:
            result["plan"] = [{k: p[k] for k in ("race_id", "date", "prev_phase", "refetch")} for p in plan]
            return result
        if not todo:
            _mark_done(sync, cp, result)
            return {**result, "fetched": 0}

        prev_cards = {p["race_id"]: _load_card(storage, p["race_id"]) for p in todo}
        specs = [
            {"job_kind": "race", "target_id": p["race_id"], "date": p["date"], "venue": p["venue"], "round": p["round"],
             "race_name": p["race_name"], "tasks": ["race_shutuba"], "smart_skip": False,
             "priority": ENTRIES_JOB_PRIORITY}
            for p in todo
        ]
        if queue is None:
            from src.scraper import auto_scrape_queue as _asq

            queue = _asq._queue()
        add = queue.bulk_add_jobs(specs)
        result["queue_bulk_add"] = add
        ok, msg = True, ""
        if wait:
            from src.scraper import auto_scrape_queue as _asq

            ok, msg, _ = _asq._kick_and_wait(queue, timeout_sec=WAIT_TIMEOUT_SEC)
        result["queue_ok"] = ok
        if not ok:
            result["queue_msg"] = msg

        decided = (schedule.get("entries_decided_at") or {}).get("weekday", "thu"), (schedule.get("entries_decided_at") or {}).get("time", "12:00")
        phases: dict[str, int] = {}
        notable: list[dict] = []
        for p in todo:
            card = _load_card(storage, p["race_id"], bypass=True)
            if not card:
                phases["none"] = phases.get("none", 0) + 1
                continue
            st, changes = update_state(load_state(p["race_id"]), prev_cards.get(p["race_id"]), card, p["race_id"], p["date"], now,
                                       p["list_entries_count"], decided=decided)
            _atomic_write(state_path(p["race_id"]), st)
            phases[st["phase"]] = phases.get(st["phase"], 0) + 1
            for c in changes:
                if c["type"] in ("removed", "jockey_change"):
                    notable.append({"race_id": p["race_id"], **c})
        result.update(fetched=len(todo), phases=phases, notable_changes=notable[:30])
        _mark_done(sync, cp, result)
        return result


def _load_card(storage: Any, race_id: str, bypass: bool = False) -> dict | None:
    try:
        return storage.load("race_shutuba", race_id, bypass_cache=True) if bypass else storage.load("race_shutuba", race_id)
    except TypeError:
        return storage.load("race_shutuba", race_id)
    except Exception:  # noqa: BLE001
        return None


def _refresh_race_lists(storage: Any, dates: list[str], cp: dict) -> list[str]:
    """レース一覧が不完全な日（特別登録の段階など）と、critical の時は全対象日を取り直す（1 日 1 リクエスト）。"""
    out: list[str] = []
    try:
        from src.scraper.run import ScraperRunner
    except Exception as exc:  # noqa: BLE001
        logger.warning("ScraperRunner を読み込めない（レース一覧の更新を見送り）: %s", exc)
        return out
    runner = None
    for ymd in dates:
        rl = storage.load("race_lists", ymd) or {}
        complete = bool((rl.get("_meta") or {}).get("race_list_complete"))
        if complete and not cp.get("critical"):
            continue
        try:
            if runner is None:
                runner = ScraperRunner()
            runner.scrape_race_list(ymd)
            out.append(ymd)
            time.sleep(1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("レース一覧の更新に失敗 %s: %s", ymd, exc)
    if out:
        try:
            storage.invalidate_load_cache("race_lists", "")
        except Exception:  # noqa: BLE001
            pass
    return out


def _mark_done(sync: dict, cp: dict, result: dict) -> None:
    done = dict(sync.get("done") or {})
    done[cp["label"]] = datetime.now(EP.JST).isoformat(timespec="seconds")
    if len(done) > DONE_KEEP:
        for k in sorted(done)[: len(done) - DONE_KEEP]:
            done.pop(k, None)
    _atomic_write(sync_state_path(), {"done": done, "last_run": done[cp["label"]], "last_checkpoint": cp["label"],
                                      "last_result": {k: result.get(k) for k in ("status", "races", "fetched", "phases", "dates")}})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="checkpoint の時刻に関わらず今すぐ実行")
    ap.add_argument("--dry-run", action="store_true", help="取得せず、取り直す予定のレースだけ表示")
    ap.add_argument("--dates", help="YYYYMMDD のカンマ区切り（既定: 今日の翌日から lookahead_days 先まで）")
    a = ap.parse_args(argv)
    from src.utils.keiba_logging import script_basic_config

    script_basic_config()
    res = run_entries_sync(force=a.force, dry_run=a.dry_run, dates=a.dates.split(",") if a.dates else None)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0 if res.get("status") in ("ok", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
