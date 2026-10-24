"""影モード（VPS の取得をホストと並走で検証）の日次レポートを作る。

    python -m src.scripts.ops.shadow_report --date YYYYMMDD [--write] [--slack]
    python -m src.scripts.ops.shadow_report --yesterday --write --slack      # 毎朝（Scheduler / Run Job）

読むもの（GCS）: 影のタスク表 ``ops/fetch_tasks_shadow/{done,failed,pending}/``、影の成果物 ``shadow/<日付>/…``、本番の同じパスのオブジェクト。
作るもの: ``ops/fetch_shadow/daily/<日付>.json``（取得の成功率・上限超過でスキップした数・本番との一致率・アクセス制限の疑い・合否 ``ok``）。
合否の基準は ``config/fetch_shadow.json`` の ``pass``。判定の利用側は ``src/ops/shadow_eval.py``（cron_cutover_status / cutover_plan / job_owner）。

本番との比較（オッズのスナップショットはタイミング差があるので許容規則つき）:
- ``race_odds``: 同じ phase の本番スナップショット（最新が違う phase なら ``snapshots`` から同じ phase を探す）と、出走馬の集合が一致し、
  単勝オッズが相対 ``odds_rtol`` 以内の馬が ``odds_horse_ok_ratio`` 以上。
- ``race_pair_odds``: 券種ごとに同じ組合せ集合のうち、オッズが相対 ``odds_rtol`` 以内のものが ``odds_horse_ok_ratio`` 以上（組合せ集合が違えば不一致）。
- ``race_shutuba``: 馬の集合と安定項目（馬番・枠・馬名・騎手・調教師・性齢）が一致。馬体重は両方に値があるときだけ比べる（片方が未発表なら無視）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from src.ops import fetch_shadow, shadow_eval
from src.ops import fetch_task as ft
from src.ops.gcs_io import read_json, write_json

JST = ZoneInfo("Asia/Tokyo")
STABLE_FIELDS = ("horse_number", "bracket_number", "horse_name", "jockey_name", "trainer_name", "sex_age")


# ───────────────────────── 比較（純関数）─────────────────────────

def _pick_snapshot(doc: dict[str, Any], phase: str | None, keys: tuple[str, ...]) -> dict[str, Any]:
    if not phase or doc.get("phase") == phase or not doc.get("phase"):
        return doc
    for s in doc.get("snapshots") or []:
        if isinstance(s, dict) and s.get("phase") == phase:
            return s
    return {}


def _close(a: Any, b: Any, rtol: float) -> bool:
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return abs(x - y) <= rtol * max(abs(y), 1e-9)


def compare_odds(shadow: dict[str, Any], prod: dict[str, Any], rtol: float, ok_ratio: float) -> tuple[bool, str]:
    snap = _pick_snapshot(prod, shadow.get("phase"), ("entries",))
    s_ent = {int(e["horse_number"]): e for e in shadow.get("entries") or [] if e.get("horse_number") is not None}
    p_ent = {int(e["horse_number"]): e for e in snap.get("entries") or [] if e.get("horse_number") is not None}
    if not p_ent:
        return False, "本番に同じ phase のオッズが無い"
    if set(s_ent) != set(p_ent):
        return False, f"出走馬の集合が違う（影 {sorted(set(s_ent) - set(p_ent))} / 本番 {sorted(set(p_ent) - set(s_ent))}）"
    near = sum(1 for n in s_ent if _close(s_ent[n].get("win_odds"), p_ent[n].get("win_odds"), rtol))
    ratio = near / len(s_ent) if s_ent else 0.0
    return ratio >= ok_ratio, f"単勝オッズが近い馬 {near}/{len(s_ent)}"


def compare_pair_odds(shadow: dict[str, Any], prod: dict[str, Any], rtol: float, ok_ratio: float) -> tuple[bool, str]:
    snap = _pick_snapshot(prod, shadow.get("phase"), tuple())
    total = near = 0
    for bt in ("umaren", "wide", "umatan", "sanrenpuku"):
        s_rows, p_rows = shadow.get(bt), snap.get(bt)
        if not s_rows or not p_rows:
            continue
        s_map = {tuple(r["combo"]): r.get("odds") for r in s_rows if "combo" in r}
        p_map = {tuple(r["combo"]): r.get("odds") for r in p_rows if "combo" in r}
        if set(s_map) != set(p_map):
            return False, f"{bt} の組合せ集合が違う（影 {len(s_map)} / 本番 {len(p_map)}）"
        total += len(s_map)
        near += sum(1 for k in s_map if _close(s_map[k], p_map[k], rtol))
    if total == 0:
        return False, "比べられる券種が無い"
    return near / total >= ok_ratio, f"オッズが近い組合せ {near}/{total}"


def compare_shutuba(shadow: dict[str, Any], prod: dict[str, Any]) -> tuple[bool, str]:
    s = {str(e.get("horse_id")): e for e in shadow.get("entries") or []}
    p = {str(e.get("horse_id")): e for e in prod.get("entries") or []}
    if set(s) != set(p):
        return False, f"馬の集合が違う（影のみ {len(set(s) - set(p))} / 本番のみ {len(set(p) - set(s))}）"
    for hid, e in s.items():
        q = p[hid]
        for f in STABLE_FIELDS:
            if e.get(f) != q.get(f):
                return False, f"{hid} の {f} が違う（影 {e.get(f)!r} / 本番 {q.get(f)!r}）"
        a, b = e.get("weight"), q.get("weight")
        if a and b and a != b:
            return False, f"{hid} の馬体重が違う（影 {a} / 本番 {b}）"
    return True, f"{len(s)} 頭一致"


def compare_doc(category: str, shadow: dict[str, Any], prod: dict[str, Any], p: dict[str, Any]) -> tuple[bool, str]:
    rtol, ratio = float(p.get("odds_rtol", 0.15)), float(p.get("odds_horse_ok_ratio", 0.9))
    if category == "race_odds":
        return compare_odds(shadow, prod, rtol, ratio)
    if category == "race_pair_odds":
        return compare_pair_odds(shadow, prod, rtol, ratio)
    if category == "race_shutuba":
        return compare_shutuba(shadow, prod)
    return True, "比較対象外"


# ───────────────────────── 集計 ─────────────────────────

def _date_of(ts: str | None) -> str | None:
    try:
        return ft.parse_ts(ts).astimezone(JST).strftime("%Y%m%d") if ts else None
    except (TypeError, ValueError):
        return None


def read_tasks(io: Any, date: str, now: datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for state in ("done", "failed", "pending"):
        for key in io.list(f"{fetch_shadow.SHADOW_QUEUE_PREFIX}/{state}/"):
            t = read_json(io, key)
            if not isinstance(t, dict) or _date_of(t.get("not_before")) != date:
                continue
            if state == "pending":
                try:
                    if ft.parse_ts(t["deadline"]) > now:
                        continue                       # まだ期限前（実行待ち）は数えない
                except (KeyError, TypeError, ValueError):
                    continue
                state_eff = "failed"
                t = {**t, "outcome": {"reason": "pending_overdue"}}
            else:
                state_eff = state
            out.append({**t, "_state": state_eff})
    return out


def classify(t: dict[str, Any]) -> str:
    res = ((t.get("outcome") or {}).get("result")) or {}
    if t["_state"] == "failed":
        return "failed"
    if res.get("skipped"):                      # 上限超過・直前オッズ無効など、実際には取得しなかったもの
        return "skipped"
    return "done_ok"


def build_day(io: Any, date: str, cfg: Any, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(JST)
    p = cfg.passing
    tasks = read_tasks(io, date, now)
    per_task: dict[str, dict[str, Any]] = {}
    for t in tasks:
        d = per_task.setdefault(t["task"], {"planned": 0, "done_ok": 0, "failed": 0, "skipped": 0, "restricted": 0})
        d["planned"] += 1
        d[classify(t)] += 1
        if (t.get("outcome") or {}).get("access_restriction"):
            d["restricted"] += 1
    # 影の成果物 ↔ 本番
    prefix = f"shadow/{date}/"
    cats = sorted({c for tcfg in cfg.tasks.values() for c in tcfg.get("categories") or []})
    by_cat: dict[str, dict[str, Any]] = {c: {"compared": 0, "matched": 0, "samples": []} for c in cats}
    for key in io.list(prefix):
        cat = next((c for c in cats if f"/{c}/" in key), None)
        if cat is None:
            continue
        s_doc, p_doc = read_json(io, key), read_json(io, key[len(prefix):])
        if not isinstance(s_doc, dict):
            continue
        by_cat[cat]["compared"] += 1
        if not isinstance(p_doc, dict):
            if len(by_cat[cat]["samples"]) < 3:
                by_cat[cat]["samples"].append({"key": key[len(prefix):], "detail": "本番に同じオブジェクトが無い"})
            continue
        ok, detail = compare_doc(cat, s_doc, p_doc, p)
        by_cat[cat]["matched"] += 1 if ok else 0
        if not ok and len(by_cat[cat]["samples"]) < 3:
            by_cat[cat]["samples"].append({"key": key[len(prefix):], "detail": detail})
    for task, d in per_task.items():
        tcats = (cfg.tasks.get(task) or {}).get("categories") or []
        comp = sum(by_cat[c]["compared"] for c in tcats if c in by_cat)
        mat = sum(by_cat[c]["matched"] for c in tcats if c in by_cat)
        exec_n = d["done_ok"] + d["failed"]
        d["success_rate"] = round(d["done_ok"] / exec_n, 4) if exec_n else None
        d["compared"], d["matched"] = comp, mat
        reasons = []
        if d["success_rate"] is not None and d["success_rate"] < float(p.get("min_success_rate", 0.95)):
            reasons.append(f"成功率 {d['success_rate']:.0%}")
        if tcats and d["done_ok"] and comp and mat / comp < float(p.get("min_match_ratio", 0.95)):
            reasons.append(f"一致率 {mat / comp:.0%}")
        if tcats and d["done_ok"] and not comp:
            reasons.append("比較できる成果物が無い")
        d["ok"], d["reason"] = not reasons, "; ".join(reasons)
    planned = sum(d["planned"] for d in per_task.values())
    done_ok = sum(d["done_ok"] for d in per_task.values())
    failed = sum(d["failed"] for d in per_task.values())
    skipped = sum(d["skipped"] for d in per_task.values())
    restricted = sum(d["restricted"] for d in per_task.values())
    alerts_extra = sum(1 for k in io.list(f"ops/alerts/{date}/") if "shadow" in k)
    compared = sum(v["compared"] for v in by_cat.values())
    matched = sum(v["matched"] for v in by_cat.values())
    expects = sum(1 for t, d in per_task.items() if (cfg.tasks.get(t) or {}).get("categories") and d["done_ok"])
    rep: dict[str, Any] = {
        "date": date, "generated_at": now.isoformat(timespec="seconds"), "planned": planned, "done_ok": done_ok, "failed": failed, "skipped": skipped,
        "success_rate": round(done_ok / (done_ok + failed), 4) if (done_ok + failed) else None,
        "compare_expected": expects, "compared": compared, "matched": matched, "match_ratio": round(matched / compared, 4) if compared else None,
        "alerts": restricted + alerts_extra, "access_restrictions": restricted,
        "is_raceday": any(t.startswith("race-") and d["done_ok"] > 0 for t, d in per_task.items()),
        "per_task": per_task, "by_category": {c: {k: v for k, v in d.items()} for c, d in by_cat.items()},
    }
    rep["success_rate"] = rep["success_rate"] if rep["success_rate"] is not None else 0.0
    rep["match_ratio"] = rep["match_ratio"] if rep["match_ratio"] is not None else 0.0
    rep["ok"], rep["reasons"] = shadow_eval.day_verdict(rep, p)
    return rep


def summarize(rep: dict[str, Any]) -> str:
    head = f"影モード日次 {rep['date']}: {'合格' if rep['ok'] else '不合格'}"
    body = [f"計画 {rep['planned']} / 成功 {rep['done_ok']} / 失敗 {rep['failed']} / 上限超過でスキップ {rep['skipped']}",
            f"本番との一致 {rep['matched']}/{rep['compared']}（{rep['match_ratio']:.0%}）・アクセス制限の疑い {rep['access_restrictions']} 件・開催日 {'あり' if rep['is_raceday'] else 'なし'}"]
    if rep["reasons"]:
        body.append("理由: " + "; ".join(rep["reasons"]))
    return head + "\n" + "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--yesterday", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--slack", action="store_true")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    from src.ops.gcs_io import default_io
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    now = datetime.now(JST)
    date = ns.date or (now - timedelta(days=1)).strftime("%Y%m%d") if (ns.date or ns.yesterday) else None
    if not date:
        ap.error("--date か --yesterday が必要です")
    io = default_io()
    rep = build_day(io, date, fetch_shadow.load_config(), now=now)
    if ns.write:
        write_json(io, shadow_eval.REPORT_KEY.format(date=date), rep)
    if ns.slack:
        try:
            from src.utils import slack_notify as SN

            text = summarize(rep)
            SN.send_payload(SN.build_message("影モード日次", [SN.section("*" + SN.esc(text.split("\n", 1)[0]) + "*\n" + SN.esc(text.split("\n", 1)[1]))],
                                             severity="normal" if rep["ok"] else "warning", fallback=text.split("\n", 1)[0]),
                            channel="report" if rep["ok"] else "alert")
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str) if ns.json else summarize(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
