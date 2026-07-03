#!/bin/bash
# ライブ進捗ダッシュボード
# Usage:
#   bash scripts/live_progress.sh             # 2 秒ごと更新 (Ctrl+C で抜ける)
#   bash scripts/live_progress.sh --interval 1 # 1 秒ごと更新

INTERVAL=2
if [ "$1" = "--interval" ] && [ -n "$2" ]; then
  INTERVAL=$2
fi

cd "$(dirname "$0")/.."

LOG="logs/scraping/scrape_missing_5gen.log"
STATUS="data/research/pedigree_10gen_3view/scrape_missing_5gen_progress.json"

while true; do
  clear
  echo "============================================================"
  echo " ライブ進捗ダッシュボード   $(date +'%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  echo

  # patch_sex 簡易表示
  if [ -f data/meta/patch_sex_status.json ]; then
    python3 -c "
import json
s = json.load(open('data/meta/patch_sex_status.json'))
state = '★ 完了' if not s['running'] else '進行中'
print(f'[patch_sex] {state}  {s[\"done\"]:,}/{s[\"total\"]:,}  失敗={s[\"failed\"]}')"
    echo
  fi

  # メイン: scrape_missing_5gen のライブ進捗
  if [ -f "$LOG" ] && [ -f "$STATUS" ]; then
    python3 << PY
import json, time, subprocess

# 進捗 JSON (50 件ごとに更新される)
s = json.load(open("$STATUS"))
total = s.get("initial_missing") or s.get("n_total", 1) or 1
started_at = s.get("run_started_at") or s.get("started_at", time.time())

# 累計完了数 (再開対応。無い場合はログ行数でフォールバック)
done = s.get("done_cumulative")
if done is None:
    try:
        n_log = int(subprocess.run(
            ["grep", "-c", "INFO GET.*attempt 1/4", "$LOG"],
            capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        n_log = 0
    c = s.get("counts", {})
    failed = c.get("failed", 0)
    done = max(n_log - failed, 0)
else:
    done = int(done)

# 失敗数 (累計優先)
cc = s.get("counts_cumulative") or s.get("counts", {})
failed = cc.get("failed", 0)

# 計算
elapsed = max(time.time() - started_at, 1)
rate = done / elapsed * 3600
eta_sec = (total - done) / (rate / 3600) if rate > 0 else 0
eta_h = eta_sec / 3600
fin = time.strftime("%m/%d %H:%M", time.localtime(time.time() + eta_sec))

bar_len = 50
filled = int(bar_len * done / total)
bar = "#" * filled + "-" * (bar_len - filled)
pct = done / total * 100

scope = s.get("scope_years") or []
scope_s = f"years={scope[0]}–{scope[-1]}" if len(scope) > 1 else (f"year={scope[0]}" if scope else "all")
print(f"[scrape_missing_5gen_for_10gen]   target_gen={s.get('target_gen','?')}  {scope_s}  race_horses={s.get('scope_race_horses','?')}")
print(f"  [{bar}] {pct:5.2f}%")
print(f"  進捗:    {done:>6,} / {total:,} 件 (失敗 {failed})")
print(f"  経過:    {elapsed/3600:5.2f}h ({elapsed/60:.1f}分)")
print(f"  速度:    {rate:.0f} 件/h (リアルタイム)")
print(f"  残り:    {eta_h:5.2f}h  ({eta_sec/60:.0f}分)")
print(f"  完了予定: {fin}")
PY
  else
    echo "[scrape_missing_5gen_for_10gen] (status / log ファイル無し)"
  fi
  echo

  # プロセス状況
  echo "--- 動作中プロセス ---"
  pids=$(pgrep -f "patch_sex|scrape_pedigree_10gen|scrape_missing_5gen" 2>/dev/null)
  if [ -z "$pids" ]; then
    echo "  (動作中プロセスなし)"
  else
    for pid in $pids; do
      ps -p "$pid" -o pid=,etime=,pcpu=,pmem=,cmd= 2>/dev/null | awk '{
        cmd=""
        for (i=5; i<=NF; i++) cmd = cmd " " $i
        printf "  PID %-7s  ETIME %-10s  CPU %4s%%  MEM %4s%%  CMD%s\n", $1, $2, $3, $4, cmd
      }'
    done
  fi
  echo

  # ライブログ (末尾 6 行)
  echo "--- 最新ログ (末尾 6 行) ---"
  tail -6 "$LOG" 2>/dev/null | sed 's/^/  /'
  echo
  echo "  (Ctrl+C で抜ける / ${INTERVAL}秒ごと更新)"

  sleep "$INTERVAL"
done
