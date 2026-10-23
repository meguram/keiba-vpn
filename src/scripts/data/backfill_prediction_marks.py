"""予測成績の日次集計（data/local/prediction_results/<日>.json）に、あとから足した印（▲ show_val・消 kesi）の着順を足す。

日次の集計を作り直すと、仮想購入の買い目が『その時点の最新オッズ』で変わってしまうため、作り直さない。
各レースの保存済みの予測（印）と結果から、足りない印の馬と着順だけを足し、日次の summary.marks を数え直す。bets・的中・LogLoss など他の項目は変えない。
使い方: python -m src.scripts.data.backfill_prediction_marks [--days 20260110 ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from src.pipeline.inference import prediction_results as pr


def _pick(e: dict, fm: dict) -> dict:
    hn = int(e.get("horse_number") or 0)
    return {"horse_number": hn, "horse_name": e.get("horse_name", ""), "finish": fm.get(hn),
            "win_prob": round(float(e.get("win_prob") or 0), 4), "win_odds": e.get("win_odds"),
            "ev_win": e.get("ev_win_cal") if e.get("ev_win_cal") is not None else e.get("ev_win"), "mark": e.get("mark_type")}


def recount_marks(races: list[dict]) -> dict:
    out = {m: pr._blank_counts() for m in pr.MARKS}
    for r in races:
        if not r.get("included"):
            continue
        for m in pr.MARKS:
            for p in (r.get("marks") or {}).get(m, []):
                b = pr._bucket(p.get("finish"))
                if b:
                    out[m]["n"] += 1
                    out[m][b] += 1
    return out


def backfill_day(doc: dict, storage) -> int:
    """足したレース数。"""
    added = 0
    for r in doc.get("races") or []:
        if not r.get("included"):
            continue
        marks = r.setdefault("marks", {})
        missing = [m for m in pr.MARKS if m not in marks]
        if not missing:
            continue
        pred = pr.load_predictions(storage, r["race_id"])
        res = pr.load_race_result(storage, r["race_id"])
        if not pred or not res:
            continue
        entries = [e for e in pred.get("predictions", []) if e.get("win_prob") is not None]
        for m in missing:
            marks[m] = [_pick(e, res["finish"]) for e in entries if e.get("mark_type") == m]
        added += 1
    if added:
        doc["summary"]["marks"] = recount_marks(doc["races"])
        doc["marks_backfilled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.gmtime(time.time() + 9 * 3600))
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    total = changed = 0
    for d in a.days or pr.list_days():
        doc = pr.read_day(d)
        if not doc:
            continue
        n = backfill_day(doc, storage)
        total += 1
        if n:
            changed += 1
            if not a.dry_run:
                pr.write_day(doc, force=True)
        print(json.dumps({"day": d, "races_updated": n, "marks": {m: doc["summary"]["marks"].get(m, {}).get("n") for m in pr.MARKS}}, ensure_ascii=False), flush=True)
    print(json.dumps({"days": total, "changed": changed, "dry_run": a.dry_run}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
