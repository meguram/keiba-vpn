#!/bin/bash
# 6月カバレッジ完了後に coverage を更新して結果を確認するスクリプト
source /home/jovyan/work/keiba-vpn/.env 2>/dev/null
LOG=/home/jovyan/work/keiba-vpn/logs/june_coverage_final.log
{
echo "=== $(date '+%Y-%m-%d %H:%M:%S') 6月カバレッジ最終確認 ==="
# キューが空になるまで待機
for i in $(seq 1 120); do
  JOBS=$(curl -s "http://127.0.0.1:8000/api/scrape-jobs" 2>/dev/null)
  RUNNING=$(echo "$JOBS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len([j for j in d.get('jobs',[]) if j.get('status') in ('running','queued')]))" 2>/dev/null || echo 99)
  if [ "$RUNNING" = "0" ]; then
    echo "$(date) 全ジョブ完了"
    break
  fi
  echo "$(date) 残りジョブ: $RUNNING"
  sleep 120
done

# カバレッジインデックス再構築をリクエスト
curl -s "http://127.0.0.1:8000/api/coverage-calendar?year=2026&refresh=true" -o /dev/null 2>/dev/null
sleep 5

# 最終カバレッジ確認
echo "=== 2026年6月 最終カバレッジ ==="
curl -s "http://127.0.0.1:8000/api/coverage-calendar?year=2026" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
june = [x for x in d.get('dates',[]) if x['date'].startswith('202606')]
for item in june:
    pct=item.get('pct',0)
    total=item['total_races']
    per_cat=item.get('per_cat',{})
    cats=d.get('categories',[])
    incomplete=[(c,per_cat.get(c,0)) for c in cats if per_cat.get(c,0) < total]
    status='✅' if pct==100 else ('⚠️' if pct>=80 else '❌')
    print(f'{status} {item[\"date\"]}  {pct:.0f}%  ({total}R)  不足={len(incomplete)}')
    for c,cnt in incomplete:
        print(f'      {c}: {cnt}/{total}')
" 2>/dev/null
} >> "$LOG" 2>&1
