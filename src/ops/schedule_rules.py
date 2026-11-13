"""定期実行ジョブの判定ルール（純関数）。監視ポータル・VPS の見張り・毎朝の点検が同じ関数を使う。

入力:
- ``jobs``: ``config/scheduled_jobs.json`` の jobs
- ``obs``: job id → ``{"runs": [{"start": 秒, "end": 秒|None, "ok": bool|None, "code": int|None}, …新しい順でなくてよい],
  "active": "active"|"activating"|…|None}``（VPS は systemd、GCP は Run Admin の実行の一覧・job_runs から作る）
- ``sources``: 場所 → ``{"seen_at": 秒|None, "error": str|None, "as_of": 秒|None}``（VPS は ops/heartbeat/vps/status.json の書かれた時刻）。
  as_of があれば、その場所のジョブは「その時刻に見た状態」として判定する（最大 45 分前に書かれた記録を今の時刻で見て、誤って未実行にしない）。
  記録を書く役（一覧の ``reporter: true``、VPS は vps-sched-status）が active でない場所は、記録が無くても途絶とみなさない（入れる前に赤にしない）。
- ``race_days``: 開催日（YYYY-MM-DD）の集合。None なら土日で代用（推定）
- ``unregistered``: 一覧に無い定期実行（Scheduler の名前など）

ラベル: green（正常）・yellow（caution）・red（異常）・gray（停止中・廃止・計画中）・unknown（判定不能・記録がまだ無い）。
全体はもっとも悪いもの（gray・unknown は全体を悪くしない）。
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from src.ops import scheduled_jobs as SJ

JST = timezone(timedelta(hours=9))
ORDER = {"gray": 0, "green": 1, "unknown": 1, "yellow": 2, "red": 3}
SOURCE_STALE_MIN = 45      # 記録の途絶とみなす（VPS の見張りは 15 分おき × 3）
SHORT_MIN = 15             # これより短い間隔は 1 回ずつ数えられない（最後の起動と結果だけで見る）
UNEXPECTED_WINDOW_S = 24 * 3600
LABEL = {"green": "正常", "yellow": "注意", "red": "異常", "gray": "停止中", "unknown": "判定不能"}


def worse(a: str, b: str) -> str:
    return a if ORDER.get(a, 1) >= ORDER.get(b, 1) else b


def _hm(t: float) -> str:
    return datetime.fromtimestamp(t, JST).strftime("%m/%d %H:%M")


def _mins(sec: float) -> str:
    m = int(sec // 60)
    return f"{m // 60} 時間 {m % 60} 分" if m >= 60 else f"{m} 分"


def _day_filter(job: dict[str, Any], race_days: set[str] | None):
    ap = job.get("applies") or "daily"
    if ap == "daily":
        return None
    if isinstance(ap, dict) and "weekdays" in ap:
        wd = set(ap["weekdays"])   # 0=月〜6=日（Python の weekday）
        return lambda iso: datetime.fromisoformat(iso).weekday() in wd
    if ap == "race_days":
        if race_days is None:
            return lambda iso: datetime.fromisoformat(iso).weekday() >= 5
        return lambda iso: iso in race_days
    return None


def _runs(o: dict[str, Any] | None) -> list[dict[str, Any]]:
    rs = [r for r in ((o or {}).get("runs") or []) if isinstance(r.get("start"), (int, float))]
    return sorted(rs, key=lambda r: r["start"], reverse=True)


def evaluate_job(job: dict[str, Any], o: dict[str, Any] | None, now: float, race_days: set[str] | None = None) -> dict[str, Any]:
    """1 件の判定 ``{"level", "reasons": [..], "last": run|None, "next_due": 秒|None, "estimated": bool}``。"""
    runs = _runs(o)
    last = runs[0] if runs else None
    st = job.get("status")
    res: dict[str, Any] = {"level": "green", "reasons": [], "last": last, "next_due": None, "estimated": False}

    def add(level: str, msg: str) -> None:
        res["level"] = worse(res["level"], level)
        res["reasons"].append(msg)

    if st != "active":
        res["level"] = "gray"
        recent = [r for r in runs if now - r["start"] <= UNEXPECTED_WINDOW_S]
        if recent and st in ("paused", "retired"):   # 6. 止めたはずなのに動いている（費用の観点）
            add("yellow", f"想定外の実行: {'停止中' if st == 'paused' else '廃止'}なのに直近 24 時間に {len(recent)} 回動いた（最後 {_hm(recent[0]['start'])}）")
        return res
    sch = job.get("schedule")
    if not sch:   # 常駐など、予定時刻の無いもの
        act = (o or {}).get("active")
        if act is None:
            res["level"] = "unknown"
            res["reasons"].append("記録がまだ無い")
        elif act not in ("active", "activating", "reloading"):
            add("red", f"常駐が止まっている（{act}）")
        return res
    if not runs:
        res["level"] = "unknown"
        res["reasons"].append("記録がまだ無い")
        return res

    grace = float(job.get("grace_min") or 30) * 60
    late = float(job.get("late_min") or grace / 120.0) * 60
    high = job.get("weight") == "high"
    every = sch.get("every_min")

    if every:   # 間隔だけの予定: 最後の起動からの経過で見る
        age = now - last["start"]
        if age > 2 * grace:
            add("red", f"未実行: 最後の起動 {_hm(last['start'])} から {_mins(age)}（{every} 分おきの予定）")
        elif age > grace:
            add("red" if high else "yellow", f"未実行: 最後の起動 {_hm(last['start'])} から {_mins(age)}（{every} 分おきの予定）")
    else:
        filt = _day_filter(job, race_days)
        res["estimated"] = job.get("applies") == "race_days" and race_days is None
        slots = SJ.slots_before(sch["cron"], now, n=3, day_filter=filt)
        if len(slots) >= 2 and 0 < slots[0] - slots[1] < grace + 120:
            # 間隔が猶予より短い cron（*/5 など）は、猶予を過ぎた回まで取る
            slots = SJ.slots_before(sch["cron"], now, n=min(500, int(grace / (slots[0] - slots[1])) + 3), day_filter=filt)
        due = [t for t in slots if t + grace <= now]   # 猶予を過ぎた予定（新しい順）
        if slots:
            res["next_due"] = slots[0]
        if due:
            d0 = due[0]
            nxt = slots[slots.index(d0) - 1] if slots.index(d0) > 0 else now + 1
            started = [r for r in runs if d0 - 120 <= r["start"] < nxt]
            if not started:   # 1. 未実行（5. 2 回続けば赤）
                d1 = due[1] if len(due) > 1 else None
                prev = [r for r in runs if d1 is not None and d1 - 120 <= r["start"] < d0 - 120]
                twice = d1 is not None and not prev
                add("red" if (high or twice) else "yellow",
                    f"未実行: {_hm(d0)} の予定から {_mins(now - d0)} すぎても開始なし" + ("（2 回続けて）" if twice else ""))
            else:
                first = min(r["start"] for r in started)
                if first - d0 > late:   # 2. 遅延
                    add("yellow", f"遅延: {_hm(d0)} の予定に対して {_mins(first - d0)} 遅れて開始")
                if len(started) > 1:   # 6. 二重起動
                    add("yellow", f"想定外の実行: {_hm(d0)} の回に {len(started)} 回起動（二重起動）")

    # 3. 終わらない
    mx = job.get("max_runtime_min")
    if last.get("end") is None and mx:
        run_for = now - last["start"]
        if run_for > 2 * mx * 60:
            add("red", f"長時間実行: {_hm(last['start'])} から {_mins(run_for)}（上限の目安 {mx} 分の 2 倍超）")
        elif run_for > mx * 60:
            add("yellow", f"長時間実行: {_hm(last['start'])} から {_mins(run_for)}（目安 {mx} 分）")
    # 4. 失敗（連続の回数）
    done = [r for r in runs if r.get("end") is not None]
    fails = 0
    for r in done:
        if r.get("ok") is False:
            fails += 1
        else:
            break
    if fails:
        code = done[0].get("code")
        add("red", f"失敗: 直近 {fails} 回続けて失敗" + (f"（終了コード {code}）" if code is not None else ""))
    # 7. 実行時間の異常（短い間隔のものは対象外）
    short = every is not None and every < SHORT_MIN
    durs = [r["end"] - r["start"] for r in done if r.get("ok") is not False and r["end"] >= r["start"]]
    if not short and len(durs) >= 4:
        med = statistics.median(durs[1:])
        if med > 0 and durs[0] > 3 * med and durs[0] > 60:
            add("yellow", f"実行時間の異常: 直近 {_mins(durs[0])}（ふだんの中央値 {_mins(med)} の 3 倍超）")
    return res


def evaluate(jobs: list[dict[str, Any]], obs: dict[str, dict[str, Any]], now: float, *, sources: dict[str, dict[str, Any]] | None = None,
             race_days: set[str] | None = None, unregistered: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """全体の判定。記録の途絶・読めない場所は、その場所のジョブを個別に並べず 1 件の注意にまとめる（8）。"""
    sources = sources or {}
    alerts: list[dict[str, Any]] = []
    down: dict[str, str] = {}
    for where, s in sources.items():
        if s.get("error"):
            down[where] = "unknown"
            alerts.append({"key": f"source:{where}", "level": "unknown", "where": where,
                           "title": f"{'GCP' if where == 'gcp' else where.upper()} の状態を読めない: {s['error']}"})
            continue
        seen = s.get("seen_at")
        stale = seen is None or now - float(seen) > SOURCE_STALE_MIN * 60
        reporters = [j for j in jobs if j.get("where") == where and j.get("reporter")]
        watched = all(j.get("status") == "active" for j in reporters) if reporters else True
        if stale and watched and any(j.get("where") == where and j.get("status") == "active" and not j.get("reporter") for j in jobs):
            down[where] = "red"
            age = "一度も届いていない" if seen is None else f"最後は {_hm(float(seen))}（{_mins(now - float(seen))} 前）"
            alerts.append({"key": f"source:{where}", "level": "red", "where": where,
                           "title": f"{'VPS' if where == 'vps' else where.upper()} の記録の途絶（{SOURCE_STALE_MIN} 分以上）: {age}。停止・回線断の疑い"})
    rows: list[dict[str, Any]] = []
    for j in jobs:
        if j.get("where") in down and j.get("status") == "active":
            r = {"level": "unknown", "reasons": ["記録の途絶のため判定なし"], "last": _runs(obs.get(j["id"]))[:1] or None, "next_due": None, "estimated": False}
            r["last"] = r["last"][0] if r["last"] else None
        else:
            r = evaluate_job(j, obs.get(j["id"]), float((sources.get(j.get("where")) or {}).get("as_of") or now), race_days)
        rows.append({"id": j["id"], **r, "label": LABEL[r["level"]]})
    for u in unregistered or []:
        alerts.append({"key": f"unregistered:{u.get('name')}", "level": "yellow", "where": u.get("where", "gcp"),
                       "title": f"未登録の定期実行: {u.get('name')}（{u.get('schedule') or '予定不明'}・{u.get('state') or ''}）。一覧に無い"})
    overall = "green"
    for r in rows:
        overall = worse(overall, r["level"]) if r["level"] not in ("gray", "unknown") else overall
    for a in alerts:
        overall = worse(overall, a["level"]) if a["level"] not in ("gray", "unknown") else overall
    counts = {k: sum(1 for r in rows if r["level"] == k) for k in ORDER}
    return {"overall": overall, "label": LABEL[overall], "rows": rows, "alerts": alerts, "counts": counts}


def _kind(reason: str) -> str:
    return reason.split(":", 1)[0].strip()


def red_keys(result: dict[str, Any]) -> dict[str, str]:
    """通知用: 赤の key → 1 行の説明。key はジョブと理由の種類（未実行・失敗など）ごと（理由が変われば知らせ直す）。"""
    out = {a["key"]: a["title"] for a in result["alerts"] if a["level"] == "red"}
    for r in result["rows"]:
        if r["level"] == "red":
            kinds = sorted({_kind(x) for x in r["reasons"]})
            out[f"job:{r['id']}:{'+'.join(kinds)}"] = f"{r['id']}: {' / '.join(r['reasons'])}"
    return out


def transitions(prev_red: dict[str, str], result: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """(新しく赤になったもの key→説明, 赤が解消した key)。同じ赤は繰り返し送らない。
    判定できなくなった（記録が無い・途絶）だけのジョブは「解消」としない（赤のまま持ち越す）。"""
    cur = red_keys(result)
    unknown = {r["id"] for r in result["rows"] if r["level"] == "unknown"}
    new = {k: v for k, v in cur.items() if k not in prev_red}
    gone = [k for k in prev_red if k not in cur and not (k.startswith("job:") and k.split(":")[1] in unknown)]
    return new, gone
