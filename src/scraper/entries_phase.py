"""出馬表の公示の段階（phase）の判定と、収集スケジュールの判定（純粋関数。ネットワーク・ストレージに触れない）。

JRA の公示の流れ（時刻は目安。実測で補正する。docs/operations/entries_timeline.md）:
  日曜夕        特別登録（重賞・特別競走の登録馬。平場は未定）
  月〜水        登録馬の更新（騎手・斤量が順次入る）
  木曜          出走馬の決定（平場も出る。枠順・馬番はまだ）
  前日 金16時／土10時  出馬表の確定（枠番・馬番）
  当日          馬体重の発表（発走の約 1 時間前）

phase:
  none        出馬表なし
  registered  特別登録・登録馬（馬番なし・出走馬決定の前）
  entries     出走馬決定（馬番なし）
  final       出馬表確定（全頭に馬番）
  weighed     馬体重発表済み（final の上に、馬体重が入っている）
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

PHASE_NONE = "none"
PHASE_REGISTERED = "registered"
PHASE_ENTRIES = "entries"
PHASE_FINAL = "final"
PHASE_WEIGHED = "weighed"

PHASE_RANK = {PHASE_NONE: 0, PHASE_REGISTERED: 1, PHASE_ENTRIES: 2, PHASE_FINAL: 3, PHASE_WEIGHED: 4}
PHASE_LABELS = {
    PHASE_NONE: "出馬表なし",
    PHASE_REGISTERED: "特別登録（暫定）",
    PHASE_ENTRIES: "出走馬決定",
    PHASE_FINAL: "出馬表確定",
    PHASE_WEIGHED: "馬体重発表済み",
}
PHASE_NOTES = {
    PHASE_NONE: "出馬表はまだ公開されていません。",
    PHASE_REGISTERED: "登録馬の一覧です。頭数・騎手・枠順は確定前で、変わることがあります。",
    PHASE_ENTRIES: "出走馬が決まりました。枠順（馬番）は前日に確定します。",
    PHASE_FINAL: "出馬表が確定しました（枠番・馬番）。馬体重は当日の発表後に入ります。",
    PHASE_WEIGHED: "馬体重が発表されました。",
}

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DEFAULT_SCHEDULE_PATH = Path(__file__).resolve().parents[2] / "config" / "entries_schedule.json"


def _num(v: Any) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _present(v: Any) -> bool:
    return v not in (None, "", 0, "0", "--", "-")


def parse_date(v: str | date | None) -> date | None:
    if isinstance(v, date):
        return v
    s = str(v or "").replace("-", "")[:8]
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


# ───────────────────────── 段階の判定 ─────────────────────────

def entries_decided_at(race_date: date, weekday: str = "thu", hhmm: str = "12:00") -> datetime:
    """そのレースの週の『出走馬決定』日時（race_date 以前で最も近い指定曜日の指定時刻）。"""
    wd = _WEEKDAYS.index(weekday)
    delta = (race_date.weekday() - wd) % 7
    d = race_date - timedelta(days=delta)
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(d, time(h, m), tzinfo=JST)


def classify_entries(
    card: dict | None,
    race_date: date | str | None,
    now: datetime | None = None,
    *,
    decided_weekday: str = "thu",
    decided_time: str = "12:00",
) -> dict[str, Any]:
    """出馬表（race_shutuba の文書）から段階と中身の集計を返す。

    - 馬番が全頭に入っている → final（馬体重が半数以上に入っていれば weighed）
    - 馬番が無い → 『出走馬決定』の日時（木曜 12:00 JST 既定）より前なら registered、以降は entries
    """
    now = (now or datetime.now(JST)).astimezone(JST)
    ents = [e for e in ((card or {}).get("entries") or []) if isinstance(e, dict)]
    n = len(ents)
    out: dict[str, Any] = {
        "n_entries": n, "n_numbered": 0, "n_bracket": 0, "n_jockey": 0, "n_weight": 0, "n_weight_carried": 0,
        "start_time": (card or {}).get("start_time") or None,
    }
    if n == 0:
        out.update(phase=PHASE_NONE)
        return {**out, "label": PHASE_LABELS[PHASE_NONE], "note": PHASE_NOTES[PHASE_NONE]}
    nums = [_num(e.get("horse_number")) for e in ents]
    out["n_numbered"] = sum(1 for x in nums if x > 0)
    out["n_bracket"] = sum(1 for e in ents if _num(e.get("bracket_number")) > 0)
    out["n_jockey"] = sum(1 for e in ents if _present(e.get("jockey_id")) or _present(e.get("jockey_name")))
    out["n_weight"] = sum(1 for e in ents if _present(e.get("weight")))
    out["n_weight_carried"] = sum(1 for e in ents if _present(e.get("jockey_weight")))
    all_numbered = out["n_numbered"] == n and len({x for x in nums if x > 0}) == n
    rd = parse_date(race_date)
    if all_numbered:
        phase = PHASE_WEIGHED if out["n_weight"] * 2 >= n else PHASE_FINAL
    else:
        decided = entries_decided_at(rd, decided_weekday, decided_time) if rd else None
        phase = PHASE_ENTRIES if (decided is not None and now >= decided) else PHASE_REGISTERED
    out.update(phase=phase, label=PHASE_LABELS[phase], note=PHASE_NOTES[phase])
    return out


def entries_signature(card: dict | None) -> str:
    """取消・騎手変更・馬番変更を検知するための内容の署名（馬体重・オッズは含めない）。"""
    import hashlib

    ents = [e for e in ((card or {}).get("entries") or []) if isinstance(e, dict)]
    rows = sorted(
        (str(e.get("horse_id") or e.get("horse_name") or ""), _num(e.get("horse_number")), _num(e.get("bracket_number")),
         str(e.get("jockey_id") or e.get("jockey_name") or ""), str(e.get("jockey_weight") or ""))
        for e in ents
    )
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[:12]


def diff_entries(old: dict | None, new: dict | None) -> list[dict[str, Any]]:
    """前回と今回の出馬表の差分（馬の追加・取消・騎手変更・斤量変更・馬番の確定）。"""
    def idx(card: dict | None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in (card or {}).get("entries") or []:
            if isinstance(e, dict):
                k = str(e.get("horse_id") or e.get("horse_name") or "")
                if k:
                    out[k] = e
        return out

    o, n = idx(old), idx(new)
    changes: list[dict[str, Any]] = []
    for k in n.keys() - o.keys():
        changes.append({"type": "added", "horse_id": k, "horse_name": n[k].get("horse_name")})
    for k in o.keys() - n.keys():
        changes.append({"type": "removed", "horse_id": k, "horse_name": o[k].get("horse_name")})
    for k in o.keys() & n.keys():
        a, b = o[k], n[k]
        ja, jb = (a.get("jockey_id") or a.get("jockey_name")), (b.get("jockey_id") or b.get("jockey_name"))
        if _present(ja) and _present(jb) and ja != jb:
            changes.append({"type": "jockey_change", "horse_id": k, "horse_name": b.get("horse_name"),
                            "from": a.get("jockey_name") or ja, "to": b.get("jockey_name") or jb})
        wa, wb = a.get("jockey_weight"), b.get("jockey_weight")
        if _present(wa) and _present(wb) and str(wa) != str(wb):
            changes.append({"type": "weight_carried_change", "horse_id": k, "horse_name": b.get("horse_name"), "from": wa, "to": wb})
    return sorted(changes, key=lambda c: (c["type"], c["horse_id"]))


# ───────────────────────── 収集スケジュール ─────────────────────────

def load_schedule(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or os.environ.get("KEIBA_ENTRIES_SCHEDULE") or DEFAULT_SCHEDULE_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def checkpoint_label(day: date, hhmm: str) -> str:
    return f"{day.isoformat()} {hhmm}"


def due_checkpoint(now: datetime, schedule: dict[str, Any], done: Iterable[str] = ()) -> dict[str, Any] | None:
    """今この時刻に実行すべき checkpoint（未実施で、時刻から window_minutes 以内の最新のもの）。無ければ None。

    戻り値: {"label": "2026-10-23 16:05", "critical": True}
    """
    now = now.astimezone(JST)
    window = timedelta(minutes=int(schedule.get("window_minutes", 25)))
    done_set = set(done)
    wd = _WEEKDAYS[now.weekday()]
    crit = set((schedule.get("critical") or {}).get(wd, []))
    best: tuple[datetime, str] | None = None
    for hhmm in (schedule.get("checkpoints") or {}).get(wd, []):
        h, m = (int(x) for x in hhmm.split(":"))
        at = datetime.combine(now.date(), time(h, m), tzinfo=JST)
        if at <= now < at + window and checkpoint_label(now.date(), hhmm) not in done_set:
            if best is None or at > best[0]:
                best = (at, hhmm)
    if best is None:
        return None
    return {"label": checkpoint_label(now.date(), best[1]), "critical": best[1] in crit, "weekday": wd, "time": best[1]}


def target_dates(race_dates: Iterable[str], today: date, lookahead_days: int = 10) -> list[str]:
    """今日以降 lookahead_days 日以内の開催日（YYYYMMDD、昇順）。"""
    out: list[str] = []
    limit = today + timedelta(days=lookahead_days)
    for s in race_dates:
        d = parse_date(s)
        if d and today <= d <= limit:
            out.append(d.strftime("%Y%m%d"))
    return sorted(set(out))


def should_refetch(
    *,
    race_date: date,
    today: date,
    prev_phase: str | None,
    critical: bool,
    list_count_changed: bool,
) -> bool:
    """このチェックポイントで、そのレースの出馬表を取り直すか。

    - 未取得、またはレース一覧の頭数が変わった → 取り直す
    - 段階が final 未満（登録馬・出走馬決定）→ 毎回取り直す（変化を追う）
    - final / weighed → 前日・当日（race_date - today <= 1）の critical なチェックポイントでだけ取り直す（取消・騎手変更の反映）
    """
    if prev_phase in (None, PHASE_NONE):
        return True
    if list_count_changed:
        return True
    if PHASE_RANK.get(prev_phase, 0) < PHASE_RANK[PHASE_FINAL]:
        return True
    return critical and (race_date - today).days <= 1
