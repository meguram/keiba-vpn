"""
全未取得日付のバッチスクレイピング（新しい日付順）。
JST 8:00 を超えたら自動中断。
"""
import sys
import time
import json
import requests
from datetime import datetime, timezone, timedelta

BASE = "http://localhost:8001"
JST = timezone(timedelta(hours=9))
DEADLINE_HOUR = 8

def now_jst():
    return datetime.now(JST)

def past_deadline():
    n = now_jst()
    return n.hour >= DEADLINE_HOUR

def main():
    resp = requests.get(f"{BASE}/api/scrape-summary-all", timeout=30)
    data = resp.json()
    dates = sorted(data.get("dates", []), key=lambda d: d["date"], reverse=True)
    targets = [d["date"] for d in dates]

    print(f"Total targets: {len(targets)} dates", flush=True)
    print(f"Deadline: JST 08:00", flush=True)
    print(f"Current: {now_jst().strftime('%H:%M:%S JST')}", flush=True)
    print(flush=True)

    completed = 0
    skipped = 0

    for i, date_str in enumerate(targets):
        if past_deadline():
            print(f"\n⏰ JST {now_jst().strftime('%H:%M')} — 8時を超えたため中断", flush=True)
            break

        print(f"[{i+1}/{len(targets)}] {date_str} — {now_jst().strftime('%H:%M:%S')}", flush=True)

        try:
            r = requests.post(
                f"{BASE}/api/scrape-trigger",
                json={"race_id": date_str, "category": "date_all", "force": False},
                timeout=30,
            )
            rd = r.json()

            if rd.get("status") == "already_running":
                job_id = rd["job_id"]
                print(f"  already running", flush=True)
            else:
                job_id = rd["job_id"]

            poll = 0
            while True:
                if past_deadline():
                    print(f"\n⏰ 8時超過 — 中断 (ジョブは裏で継続)", flush=True)
                    print(f"完了: {completed}, スキップ: {skipped}", flush=True)
                    sys.exit(0)

                time.sleep(3)
                poll += 1

                jr = requests.get(f"{BASE}/api/scrape-jobs", timeout=10).json()
                job = next((j for j in jr.get("jobs", []) if j.get("job_id") == job_id), None)

                if job is None:
                    print(f"  ✓ done", flush=True)
                    completed += 1
                    break
                
                st = job.get("status", "")
                prog = job.get("progress", {})

                if st == "done":
                    elapsed = (job.get("finished_at") or 0) - (job.get("started_at") or 0)
                    print(f"  ✓ done in {elapsed:.0f}s", flush=True)
                    completed += 1
                    break
                elif st == "error":
                    print(f"  ✗ error: {job.get('error', '?')}", flush=True)
                    skipped += 1
                    break
                elif poll % 10 == 0:
                    cur = prog.get("current", 0)
                    tot = prog.get("total", 0)
                    lbl = prog.get("current_label", "")
                    if tot:
                        print(f"  ... {cur}/{tot} {lbl}", flush=True)

        except Exception as e:
            print(f"  ✗ exception: {e}", flush=True)
            skipped += 1
            time.sleep(5)

    print(f"\n=== Summary ===", flush=True)
    print(f"Completed: {completed}", flush=True)
    print(f"Skipped: {skipped}", flush=True)
    print(f"Remaining: {len(targets) - completed - skipped}", flush=True)
    print(f"Time: {now_jst().strftime('%H:%M:%S JST')}", flush=True)

if __name__ == "__main__":
    main()
