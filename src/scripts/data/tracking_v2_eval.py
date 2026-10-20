"""追走難度 v2 の保存と実測の突き合わせ（日次・週次）を手動で実行する。

python -m src.scripts.data.tracking_v2_eval --date YYYYMMDD            # その日（日次を保存）
python -m src.scripts.data.tracking_v2_eval --from A --to B             # 範囲の開催日（日次を保存し、含まれる週の週次も作る）
python -m src.scripts.data.tracking_v2_eval --week YYYYMMDD             # その日を含む週（月〜日）の週次
python -m src.scripts.data.tracking_v2_eval --from A --to B --dry-run   # 保存せず要約だけ表示

自動実行は KEIBA_TRACKING_V2_EVAL=1 のとき金曜の weekly-update の後（refresh_for_dates）。仕様: docs/operations/tracking_v2_precompute.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta


def _dates(frm: str, to: str) -> list[str]:
    a, b = datetime.strptime(frm, "%Y%m%d"), datetime.strptime(to, "%Y%m%d")
    return [(a + timedelta(days=i)).strftime("%Y%m%d") for i in range((b - a).days + 1)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--to")
    ap.add_argument("--week")
    ap.add_argument("--dry-run", action="store_true", help="保存せず、要約だけ出す")
    a = ap.parse_args(argv)
    from src.pipeline.inference import tracking_v2_eval as ev

    save = not a.dry_run
    if a.week:
        doc = ev.aggregate_week(a.week.replace("-", ""), save=save)
        print(json.dumps({k: v for k, v in doc.items() if k != "breakdown"}, ensure_ascii=False, indent=1))
        return 0
    frm, to = (a.date, a.date) if a.date else (a.frm, a.to)
    if not frm or not to:
        ap.error("--date / --from+--to / --week のどれかを指定してください")
    weeks, rows = set(), {}
    for d in _dates(frm.replace("-", ""), to.replace("-", "")):
        doc = ev.aggregate_day(d, save=save)
        if doc.get("races_saved") or doc.get("status") == "ok":
            rows[d] = {"status": doc["status"], "saved": doc.get("races_saved"), "held": doc.get("races_held"), "coverage": doc.get("coverage"),
                       "n_pairs": (doc.get("metrics") or {}).get("n_pairs"), "corr_drain_y": (doc.get("metrics") or {}).get("corr_drain_y")}
            if doc["status"] == "ok":
                weeks.add(ev.week_monday(d))
    if save:
        for w in sorted(weeks):
            ev.aggregate_week(w)
    print(json.dumps({"days": rows, "weeks": sorted(weeks)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
