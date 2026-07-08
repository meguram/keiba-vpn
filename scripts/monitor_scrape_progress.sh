#!/usr/bin/env bash
# 不完全データスクレイピングの進捗監視スクリプト
# 完了後に date_coverage を更新して最終カバレッジを表示する

LOGFILE="/home/jovyan/work/keiba-vpn/logs/scrape_progress_2026.log"
cd /home/jovyan/work/keiba-vpn

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=== スクレイピング進捗監視開始 ==="

while true; do
    STATUS=$(curl -s -m10 "http://127.0.0.1:8000/api/scrape-jobs" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        log "FastAPI 応答なし"
        sleep 60
        continue
    fi

    DONE=$(echo "$STATUS" | python3 -c "import sys,json; jobs=json.load(sys.stdin); print(sum(1 for j in (jobs if isinstance(jobs,list) else jobs.get('jobs',[])) if j.get('status')=='done'))" 2>/dev/null)
    RUNNING=$(echo "$STATUS" | python3 -c "import sys,json; jobs=json.load(sys.stdin); print(sum(1 for j in (jobs if isinstance(jobs,list) else jobs.get('jobs',[])) if j.get('status')=='running'))" 2>/dev/null)
    QUEUED=$(echo "$STATUS" | python3 -c "import sys,json; jobs=json.load(sys.stdin); print(sum(1 for j in (jobs if isinstance(jobs,list) else jobs.get('jobs',[])) if j.get('status')=='queued'))" 2>/dev/null)
    ERROR=$(echo "$STATUS" | python3 -c "import sys,json; jobs=json.load(sys.stdin); print(sum(1 for j in (jobs if isinstance(jobs,list) else jobs.get('jobs',[])) if j.get('status')=='error'))" 2>/dev/null)
    RUNNING_IDS=$(echo "$STATUS" | python3 -c "import sys,json; jobs=json.load(sys.stdin); [print(j.get('race_id','?')) for j in (jobs if isinstance(jobs,list) else jobs.get('jobs',[])) if j.get('status')=='running']" 2>/dev/null | tr '\n' ',')

    log "完了=$DONE 実行中=$RUNNING($RUNNING_IDS) 待機=$QUEUED エラー=$ERROR"

    # 全ジョブ完了またはエラーのみ残
    if [ "$RUNNING" = "0" ] && [ "$QUEUED" = "0" ]; then
        log "=== 全ジョブ完了 ==="
        break
    fi

    sleep 120
done

# date_coverage インデックスを再構築
log "date_coverage インデックス更新中..."
python3 -c "
import sys; sys.path.insert(0,'.')
from src.scraper.date_coverage import TRACK_CATEGORIES, load_year_coverage

incomplete = []
for y in [2026]:
    cov = load_year_coverage(y)
    for dt, data in sorted(cov.items()):
        if dt < '20260101' or dt > '20260630':
            continue
        total = data.get('total_races', 0)
        if total == 0:
            continue
        cats = data.get('categories') or {}
        missing = [c for c in TRACK_CATEGORIES if cats.get(c, 0) < total and c != 'race_barometer']
        if missing:
            pct = round(sum(cats.get(c,0) for c in TRACK_CATEGORIES) / (total * len(TRACK_CATEGORIES)) * 100, 1)
            incomplete.append((dt, total, pct, missing))

if incomplete:
    print(f'まだ不完全な日付: {len(incomplete)} 日')
    for dt, total, pct, m in incomplete[:10]:
        print(f'  {dt}: {pct}% missing={m[:2]}')
else:
    print('2026/1/1-6/30 の全データ完全取得を確認！')
" 2>/dev/null | tee -a "$LOGFILE"

log "=== 監視終了 ==="
