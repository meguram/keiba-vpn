"""前夜の計算（追走難度・追走難度 v2・想定オッズ）。``raceday-eve`` の取得後半を分離したもの（取得は VPS の ``raceday-eve-fetch``）。

    python -m src.ops.jobs.eve_precompute --date YYYYMMDD

計算の中身は従来と同じ ``auto_scrape_queue._eve_precomputes``（環境変数 KEIBA_EVE_PRECOMPUTE_TRACKING / KEIBA_TRACKING_V2 /
KEIBA_EVE_PRECOMPUTE_FINAL_ODDS もそのまま効く）。対象の race_id は保存済みのレース一覧から取る（ここでは取得しない）。
冪等（計算済みは skip）。一部でも fail があれば終了コード 1。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def race_ids_for(storage: Any, date: str) -> list[str]:
    rl = storage.load("race_lists", date) or {}
    return [str(r["race_id"]).strip() for r in rl.get("races") or [] if isinstance(r, dict) and r.get("race_id")]


def run(date: str, storage: Any | None = None, precompute: Any | None = None) -> dict[str, Any]:
    if storage is None:
        from src.scraper.auto_scrape_queue import _storage

        storage = _storage()
    ids = race_ids_for(storage, date)
    if not ids:
        return {"status": "skipped", "reason": "no-races", "date": date}
    if precompute is None:
        from src.scraper.auto_scrape_queue import _eve_precomputes as precompute
    tracking, final_odds = precompute(date, ids)
    fails = int(tracking.get("fail", 0) or 0) + int(final_odds.get("fail", 0) or 0)
    v2 = tracking.get("tracking_v2") or {}
    fails += int(v2.get("fail", 0) or 0)
    errored = "error" in tracking or "error" in final_odds
    return {"status": "error" if (fails or errored) else "ok", "date": date, "races": len(ids), "tracking": tracking, "final_odds": final_odds}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYYMMDD（翌開催日）")
    ns = ap.parse_args(argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    res = run(ns.date)
    print(json.dumps(res, ensure_ascii=False, default=str))
    return 1 if res["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
