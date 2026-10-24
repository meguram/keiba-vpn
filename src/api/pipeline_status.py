"""開催日の予測パイプライン（出馬表 → 馬体重 → AI予測 → オッズ → 推奨買い目）の進み具合を API に付ける。

読み取りだけ（スクレイプ・推論は起動しない）。全工程が済んだとき ``complete=True`` になり、UI が『完了』を出せる。
予測の時刻は、ローカル予測なら ``data/local/inference/events/<日>.jsonl`` の ``at``、GCS の予測なら ``_meta.scraped_at``。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))
STAGES = ("entries", "weights", "prediction", "odds", "bets")
LABELS = {"entries": "出馬表", "weights": "馬体重", "prediction": "AI予測", "odds": "オッズ", "bets": "推奨買い目"}

# 馬体重ランナーが各レースを見る窓（発走の何分前）。src/scraper/raceday_weight_watch.py の既定と同じ
WINDOW_START_MIN, WINDOW_DEADLINE_MIN = 65, 30

_BETS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_BETS_TTL_SEC = 30.0


def _stage(key: str, state: str, detail: str = "", at: str | None = None) -> dict[str, Any]:
    return {"key": key, "label": LABELS[key], "state": state, "detail": detail, "at": at}


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, tz=JST).isoformat(timespec="seconds") if ts else None


def _start_dt(card: dict | None, race_date: str | None) -> datetime | None:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str((card or {}).get("start_time") or ""))
    d = re.match(r"^(\d{4})\D?(\d{2})\D?(\d{2})", str(race_date or (card or {}).get("date") or ""))
    if not (m and d):
        return None
    return datetime(int(d[1]), int(d[2]), int(d[3]), int(m[1]), int(m[2]), tzinfo=JST)


def _hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _prediction_event(resp: dict, race_id: str, race_date: str | None) -> dict[str, Any]:
    """予測を作った時刻（at）と、馬体重が揃っていたか（weights_known。分からなければ None）。"""
    if resp.get("prediction_source") in ("shadow", "backfill"):
        from src.api import local_predictions as LP

        day = re.sub(r"\D", "", str(race_date or ""))[:8]
        try:
            for line in reversed((LP.inference_dir() / "events" / f"{day}.jsonl").read_text(encoding="utf-8").splitlines()):
                ev = json.loads(line)
                if ev.get("race_id") == race_id and ev.get("v3_status", ev.get("status")) == "ok":
                    return {"at": ev.get("at"), "weights_known": ev.get("weights_known")}
        except (OSError, ValueError):
            pass
        found = LP.find_local_prediction(race_id, day)
        return {"at": _iso(found[0].stat().st_mtime) if found else None, "weights_known": None}
    fr = resp.get("freshness") or {}
    return {"at": _iso(float(fr["scraped_at"])) if fr.get("scraped_at") else None, "weights_known": None}


def _bets_state(race_id: str) -> dict[str, Any]:
    """推奨ポートフォリオを（既定の条件で）作れるか。30 秒キャッシュ。"""
    hit = _BETS_CACHE.get(race_id)
    if hit and time.time() - hit[0] < _BETS_TTL_SEC:
        return hit[1]
    from src.api.v1.delegates import DEFAULT_MAX_TOTAL_BET, optimize_betting

    try:
        payload, code = optimize_betting({
            "race_id": race_id, "bankroll": 100000, "odds_mode": "auto", "odds_phase": "latest",
            "max_total_bet": DEFAULT_MAX_TOTAL_BET, "bet_types": ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"],
        })
    except Exception as exc:  # noqa: BLE001
        payload, code = {"error": str(exc)}, 500
    if code != 200:
        out = {"ok": False, "reason": str(payload.get("error") or "計算できませんでした")[:80]}
    else:
        n = len(payload.get("candidates") or [])
        out = {"ok": True, "n": n, "odds_source": payload.get("odds_source"), "odds_phase": payload.get("odds_phase")}
    _BETS_CACHE[race_id] = (time.time(), out)
    return out


def pipeline_status(race_id: str, resp: dict, storage: Any, *, now: datetime | None = None) -> dict[str, Any]:
    now = (now or datetime.now(JST)).astimezone(JST)
    es = resp.get("entries_status") or {}
    try:
        card = storage.load("race_shutuba", race_id) or {}
    except Exception:  # noqa: BLE001
        card = {}
    race_date = es.get("race_date") or card.get("date")
    start = _start_dt(card, race_date)
    window = None
    if start:
        window = (start - timedelta(minutes=WINDOW_START_MIN), start - timedelta(minutes=WINDOW_DEADLINE_MIN))
    past_race = bool(start and now > start + timedelta(minutes=5))
    stages: list[dict[str, Any]] = []

    # 1. 出馬表
    phase = es.get("phase")
    if phase in ("final", "weighed"):
        stages.append(_stage("entries", "done", es.get("label") or "出馬表確定", es.get("updated_at")))
    else:
        stages.append(_stage("entries", "waiting", f"{es.get('label') or '出馬表待ち'}（枠順の確定は前日）", es.get("updated_at")))

    # 2. 馬体重（予測が済んでいれば、予測を作った時点の記録を優先する。出馬表は後の取得で上書きされるため）
    has_pred = resp.get("has_prediction") is True
    pev = _prediction_event(resp, race_id, race_date) if has_pred else {}
    n_entries, n_weight = int(es.get("n_entries") or 0), int(es.get("n_weight") or 0)
    if has_pred and pev.get("weights_known") is not None:
        stages.append(_stage("weights", "done", "馬体重あり" if pev["weights_known"] else "馬体重なしで予測（期限までに未発表）", pev.get("at")))
    elif phase == "weighed" or (n_entries and n_weight / n_entries >= 0.9):
        stages.append(_stage("weights", "done", f"{n_weight}/{n_entries}頭", es.get("updated_at")))
    elif window:
        stages.append(_stage("weights", "waiting", f"発走{WINDOW_START_MIN}〜{WINDOW_DEADLINE_MIN}分前（{_hhmm(window[0])}〜{_hhmm(window[1])}）に自動で確認"))
    else:
        stages.append(_stage("weights", "waiting", "発走の約1時間前に発表"))

    # 3. AI予測
    if has_pred:
        src = {"shadow": "検証用", "backfill": "過去分の算出"}.get(str(resp.get("prediction_source") or ""), "")
        stages.append(_stage("prediction", "done", src, pev.get("at")))
    elif window and now < window[1]:
        stages.append(_stage("prediction", "waiting", f"馬体重の確認後に自動実行（{_hhmm(window[0])}〜{_hhmm(window[1])}）"))
    else:
        stages.append(_stage("prediction", "waiting", "予測はまだありません"))

    # 4. オッズ
    try:
        odds = storage.load("race_odds", race_id)
    except Exception:  # noqa: BLE001
        odds = None
    if odds and odds.get("entries"):
        ph = odds.get("phase")
        detail = {"t60": "発走60分前", "t15": "発走15分前", "final": "確定", "t10": "10分前", "t05": "5分前", "t02": "2分前"}.get(str(ph), str(ph or "取得済み"))
        stages.append(_stage("odds", "done", detail, odds.get("snapshot_at")))
    else:
        stages.append(_stage("odds", "waiting", "発走60分前に取得（無いときは想定オッズ）"))

    # 5. 推奨買い目（予測が済んでいるときだけ計算する。オッズが無ければ想定オッズで計算される）
    if has_pred:
        b = _bets_state(race_id)
        if b["ok"]:
            stages.append(_stage("bets", "done", "見送り（推奨なし）" if b["n"] == 0 else "作成済み", None))
        else:
            stages.append(_stage("bets", "waiting", b["reason"]))
    else:
        stages.append(_stage("bets", "waiting", "予測の後に計算"))

    done = [s for s in stages if s["state"] == "done"]
    nxt = next((s for s in stages if s["state"] != "done"), None)
    complete = len(done) == len(stages)
    at = max((s["at"] for s in done if s.get("at")), default=None)
    return {
        "stages": stages, "complete": complete, "done": len(done), "total": len(stages),
        "next": nxt["key"] if nxt else None, "completed_at": at if complete else None,
        "window": [w.isoformat(timespec="minutes") for w in window] if window else None,
        "race_over": past_race, "as_of": now.isoformat(timespec="seconds"),
    }


def attach_pipeline_status(resp: dict, race_id: str, storage: Any = None) -> dict:
    """予測 API の応答に pipeline_status を足す（後方互換: 未知のキーとして付くだけ）。失敗しても応答は返す。"""
    try:
        if not isinstance(resp, dict) or resp.get("_mock"):
            return resp
        return {**resp, "pipeline_status": pipeline_status(race_id, resp, storage)}
    except Exception:  # noqa: BLE001
        return resp
