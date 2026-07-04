#!/bin/bash
# patch_sex と scrape_10gen_ancestors の進捗をまとめて表示
# Usage:
#   bash scripts/show_scrape_progress.sh           # 一回だけ表示
#   bash scripts/show_scrape_progress.sh watch     # 10 秒ごとに自動更新

cd "$(dirname "$0")/.."

show() {
  clear
  echo "=========== スクレイピング進捗 ($(date +'%Y-%m-%d %H:%M:%S')) ==========="
  echo

  # patch_sex
  if [ -f data/meta/patch_sex_status.json ]; then
    python3 -c "
import json, time
try:
    s = json.load(open('data/meta/patch_sex_status.json'))
    d, t = s['done'], s['total']
    pct = d/t*100
    bar = '#'*int(40*d/t) + '-'*(40-int(40*d/t))
    eta_h = s['eta_sec']/3600
    fin = time.strftime('%m/%d %H:%M', time.localtime(time.time()+s['eta_sec']))
    state = '進行中' if s['running'] else '★ 完了'
    print(f'[patch_sex] {state}')
    print(f'  [{bar}] {pct:5.1f}%  {d:,}/{t:,}  失敗={s[\"failed\"]}')
    print(f'  速度: {s[\"rate_per_hour\"]:.0f} 件/h  残り: {eta_h:.1f}h  → 完了予定: {fin}')
except Exception as e:
    print(f'[patch_sex] (status 読込失敗: {e})')
"
  else
    echo "[patch_sex] (status JSON 無し)"
  fi
  echo

  # 10gen 用スクレイピング (新: scrape_missing_5gen_for_10gen / 旧: scrape_pedigree_10gen_ancestors)
  for label_path in \
      "scrape_missing_5gen_for_10gen|data/research/pedigree_10gen_3view/scrape_missing_5gen_progress.json" \
      "scrape_pedigree_10gen_ancestors|data/research/pedigree_10gen_3view/scrape_progress.json"; do
    label="${label_path%%|*}"
    path="${label_path##*|}"
    if [ -f "$path" ]; then
      python3 -c "
import json, time
try:
    s = json.load(open('$path'))
    d, t = s.get('done', 0), s.get('n_total', 1) or 1
    pct = d/t*100 if t else 0
    bar = '#'*int(40*d/t) + '-'*(40-int(40*d/t))
    eta_h = s.get('eta_hours', 0)
    fin = time.strftime('%m/%d %H:%M', time.localtime(time.time()+eta_h*3600))
    c = s.get('counts', {})
    extra = ''
    if 'target_gen' in s:
        extra += f'  target_gen={s[\"target_gen\"]}'
    if s.get('recursive'):
        extra += '  recursive=on'
    finished = 'finished_at' in s
    state = '★ 完了' if finished else '進行中'
    print(f'[$label] {state}{extra}')
    print(f'  [{bar}] {pct:5.1f}%  {d:,}/{t:,}  success={c.get(\"success\",0)} failed={c.get(\"failed\",0)}')
    if not finished:
        print(f'  速度: {s.get(\"rate_per_hour\",0)} 件/h  残り: {eta_h:.1f}h  → 完了予定: {fin}')
    else:
        eh = s.get('elapsed_hours', 0)
        print(f'  所要: {eh}h')
except Exception as e:
    print(f'[$label] (status 読込失敗: {e})')
"
    fi
  done
  echo

  # プロセス生存 (ETIME は実経過時間, ps -o etime から取得)
  echo "--- 動作中プロセス ---"
  pids=$(pgrep -f "patch_sex|scrape_pedigree_10gen|scrape_missing_5gen" 2>/dev/null)
  if [ -z "$pids" ]; then
    echo "  (動作中プロセスなし)"
  else
    for pid in $pids; do
      ps -p "$pid" -o pid=,etime=,pcpu=,cmd= 2>/dev/null | awk '{
        printf "  PID %-7s  ETIME %-12s  CPU %4s%%  CMD %s", $1, $2, $3, $4
        for (i=5; i<=NF; i++) printf " %s", $i
        printf "\n"
      }'
    done
  fi
}

if [ "$1" = "watch" ]; then
  while true; do
    show
    sleep 10
  done
else
  show
fi
