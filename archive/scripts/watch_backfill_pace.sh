#!/bin/bash
# race_result pace バックフィル進捗モニター（プログレスバー付き）
#
# Usage:
#   bash scripts/watch_backfill_pace.sh
#   bash scripts/watch_backfill_pace.sh --interval 2
#   bash scripts/watch_backfill_pace.sh --log logs/backfill_race_result_pace.log

set -euo pipefail
cd "$(dirname "$0")/.."

INTERVAL=2
LOG="logs/backfill_race_result_pace.log"
PROGRESS="logs/backfill_race_result_pace_progress.json"

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="${2:-2}"; shift 2 ;;
    --log) LOG="${2:-$LOG}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/watch_backfill_pace.sh [--interval SEC] [--log PATH]"
      exit 0
      ;;
    *) shift ;;
  esac
done

if [ ! -f "$LOG" ]; then
  echo "ログが見つかりません: $LOG"
  echo "先にバックフィルを起動してください:"
  echo "  nohup python3 -m src.scripts.data.backfill_race_result_pace --export > $LOG 2>&1 &"
  exit 1
fi

while true; do
  clear
  LOG="$LOG" PROGRESS="$PROGRESS" python3 << 'PY'
import json
import os
import re
import subprocess
import time
from pathlib import Path

log_path = Path(os.environ["LOG"])
prog_path = Path(os.environ.get("PROGRESS", "logs/backfill_race_result_pace_progress.json"))

def pgrep_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "src.scripts.data.backfill_race_result_pace"],
            capture_output=True, text=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False

def parse_log(text: str) -> dict:
    total = 0
    m = re.search(r"pace 欠損レース:\s*(\d+)\s*件", text)
    if m:
        total = int(m.group(1))

    done = ok = empty = fail = 0
    phase = "scrape"
    last_rid = ""
    export_year = ""

    # バックフィル本体の進捗行のみ（HTTP の "attempt 1/4" 等は無視）
    progress_lines = list(
        re.finditer(
            r"(?:^|\n)\[(\d+)/(\d+)\]\s+ok=(\d+)\s+empty=(\d+)\s+fail=(\d+)",
            text,
        )
    )
    if progress_lines:
        last = progress_lines[-1]
        done = int(last.group(1))
        total = int(last.group(2))
        ok = int(last.group(3))
        empty = int(last.group(4))
        fail = int(last.group(5))

    ids = re.findall(r"取得完了: race_result/(\d+)", text)
    if done == 0:
        done = len(ids)
    elif len(ids) > done:
        # 25件ごとの print よりログの完了行の方が進んでいる
        done = len(ids)

    for line in text.splitlines()[-30:]:
        if "export race_result " in line:
            phase = "export"
            m2 = re.search(r"export race_result(?:_lap)?\s+(\d{4})", line)
            if m2:
                export_year = m2.group(1)
        if "flat Parquet 再エクスポート完了" in line:
            phase = "done"
        if "完了:" in line and "ok" in line:
            phase = "done"

    for m in re.finditer(r"取得完了: race_result/(\d+)", text):
        last_rid = m.group(1)

    finished = phase == "done" or ("完了:" in text and "ok" in text.split("完了:")[-1][:80])

    return {
        "total": total,
        "done": done,
        "ok": ok,
        "empty": empty,
        "fail": fail,
        "phase": phase,
        "export_year": export_year,
        "last_rid": last_rid,
        "finished": finished,
    }

# progress JSON 優先
stats = {}
if prog_path.exists():
    try:
        stats = json.loads(prog_path.read_text(encoding="utf-8"))
    except Exception:
        stats = {}

log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
parsed = parse_log(log_text)

total = int(stats["total"]) if stats.get("total") else parsed["total"]
done = int(stats["done"]) if stats.get("done") is not None else parsed["done"]
ok = int(stats.get("ok", parsed["ok"]))
empty = int(stats.get("empty", parsed["empty"]))
fail = int(stats.get("fail", parsed["fail"]))

# ログ先頭に「pace 欠損レース」行が無い旧ジョブ向けフォールバック
if total <= 0:
    if done > 0:
        total = max(done, 531)
    else:
        total = 531
if total < done:
    total = done
phase = stats.get("phase") or parsed["phase"]
last_rid = stats.get("last_race_id") or parsed["last_rid"]
started_at = stats.get("started_at")
finished = parsed["finished"] or stats.get("finished")

running = pgrep_running()
if finished:
    running = False

elapsed = max(time.time() - started_at, 1) if started_at else None
rate_h = (done / elapsed * 3600) if elapsed and done > 0 else 0
remain = max(total - done, 0)
eta_sec = remain / (rate_h / 3600) if rate_h > 0 else 0
eta_s = time.strftime("%H:%M:%S", time.gmtime(int(eta_sec))) if eta_sec else "—"

bar_len = 40
pct = (done / total) if total > 0 else 0.0
pct = max(0.0, min(pct, 1.0))
filled = int(bar_len * pct)
bar = "█" * filled + "░" * (bar_len - filled)

print("=" * 62)
print(f" pace バックフィル監視   {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 62)
print()
state = "実行中" if running else ("完了" if finished else "停止?")
print(f"  状態:     {state}   フェーズ: {phase}" + (f" ({parsed['export_year']})" if parsed.get('export_year') else ""))
print(f"  [{bar}] {pct*100:5.1f}%")
print(f"  進捗:     {done:>5} / {total} レース")
print(f"  内訳:     ok={ok}  empty={empty}  fail={fail}")
if elapsed:
    print(f"  経過:     {elapsed/60:.1f} 分   速度: {rate_h:.0f} レース/時")
    print(f"  残り:     約 {remain} 件   ETA: {eta_s}")
if last_rid:
    print(f"  直近:     {last_rid}")
print()
print(f"  ログ:     {log_path}")
if prog_path.exists():
    print(f"  JSON:     {prog_path}")
print()
print("  Ctrl+C で終了（バックフィルは止まりません）")
if not running and not finished and done < total:
    print("  ※ プロセスが見つかりません。nohup が終了した可能性があります。")
PY
  sleep "$INTERVAL"
done
