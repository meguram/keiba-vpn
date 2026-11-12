#!/usr/bin/env bash
# バックフィル進捗の watch 用スクリプト
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/opt/conda/bin/python3
cd "$PROJECT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " バックフィル進捗  $(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S JST')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 実行中プロセス ──
echo ""
echo "▶ 実行中のバックフィルプロセス:"
PROCS=$(pgrep -a -f "^[^ ]*python[^ ]* .*(backfill|auto_scrape|fetch_sire_lines)" 2>/dev/null | grep -v "grep\|watch_backfill" || true)
if [ -n "$PROCS" ]; then
  echo "$PROCS" | awk '{printf "  PID %-7s  %s\n", $1, substr($0, index($0,$3))}'
else
  echo "  (なし)"
fi

# ── HALT ファイル ──
HALT="$PROJECT_DIR/data/local/meta/backfill_past_halt.json"
if [ -f "$HALT" ]; then
  echo ""
  python3 -c "
import json, datetime, os, sys
HALT = '$HALT'
d = json.load(open(HALT))
at_str = d.get('at','')
bad    = d.get('bad', 0)
te     = d.get('transport_errors', 0)
reason = d.get('reason', '?')
# 経過時間
elapsed_min = '?'
try:
    dt = datetime.datetime.fromisoformat(at_str)
    elapsed_min = int((datetime.datetime.now(dt.tzinfo) - dt).total_seconds() / 60)
except Exception:
    pass
if bad == 0:
    kind  = '通信エラー（自動解除待ち）'
    how   = f'→ 次回クロン（:05）で自動解除・再開します'
    mark  = '⚠️ '
else:
    kind  = f'HTTP エラー（bad={bad}件: 403/429/5xx）'
    how   = f'→ 人手確認後: rm {HALT}'
    mark  = '🚨'
print(f'  {mark} HALT 中: {kind}')
print(f'     理由: {reason}')
print(f'     停止から: {elapsed_min} 分')
print(f'     {how}')
" 2>/dev/null || echo "  ⛔ HALT ファイルあり（詳細取得失敗）"
fi

# ── phase past 詳細進捗 (2016-2019) ──
echo ""
echo "▶ 2016-2019 バックフィル (--phase past):"
$PY -m src.scraper.backfill --phase past --report 2>/dev/null | \
  grep -E "^\{|kaisai_days|fast|cards|lap|horse|累計|直近|想定" | \
  python3 -c "
import sys, json
lines = sys.stdin.read()
# JSON 行をパース
import re
m = re.search(r'\{.*\}', lines, re.S)
if m:
    d = json.loads(m.group())
    years = ['2019','2018','2017','2016']
    for yr in years:
        if yr not in d: continue
        v = d[yr]
        fast = v.get('fast','?')
        lap  = v.get('lap','?')
        cards= v.get('cards','?')
        print(f'  {yr}  fast={fast:12s}  lap={lap:12s}  cards={cards}')
    if 'horses' in d:
        h = d['horses']
        print(f'  horse: {h.get(\"collected\",0):,}頭取得済 / done={h.get(\"done\",0):,}')
else:
    # fallback: 生テキストをそのまま表示
    for l in lines.splitlines():
        if any(x in l for x in ['累計','直近','想定','fast','horse']):
            print(' ', l.strip())
" 2>/dev/null

# ── 年別 fast/full カバレッジ ──
echo ""
echo "▶ 年別カバレッジ (fast / full):"
$PY -c "
import json
with open('data/local/meta/backfill_progress.json') as f:
    d = json.load(f)
fast = {}; full = {}
for x in d.get('done_fast',[]): fast[x[:4]] = fast.get(x[:4],0)+1
for x in d.get('done_full',[]): full[x[:4]] = full.get(x[:4],0)+1
years = sorted(set(list(fast)+list(full)), reverse=True)
exp = {'2016':107,'2017':111,'2018':109,'2019':110,'2020':108,'2021':108,'2022':109,'2023':110,'2024':109,'2025':108}
for yr in years:
    f = fast.get(yr,0); fl = full.get(yr,0)
    e = exp.get(yr,'?')
    bar = '█'*(f*20//(e if isinstance(e,int) else 110)) + '░'*(20-f*20//(e if isinstance(e,int) else 110))
    flag = '✓' if isinstance(e,int) and f>=e else ' '
    print(f'  {yr}  fast={f:3d}/{e}  {flag} [{bar}]  full={fl:3d}')
" 2>/dev/null

# ── 最新ログ ──
echo ""
echo "▶ backfill_past.log (直近 6 行):"
tail -6 "$PROJECT_DIR/logs/backfill_past.log" 2>/dev/null | sed 's/^/  /'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " horse cron (backfill-all-horse): 毎日 21:00"
echo " sire_line: $(python3 -c "import json; d=json.load(open('data/local/pedigree_lines/sire_line_map.json')); print(len(d.get('lines',{})), '種牡馬  complete_at=', d.get('complete_at','未完了'))" 2>/dev/null)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
