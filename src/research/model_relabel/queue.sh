#!/usr/bin/env bash
# 拡張ラベルの学習を 1 本ずつ流す（担当 model-relabel）。引数は「fs:target:variant[:split]」の並び。
# 各本の前に、load 8 以下・空きメモリ 20GB 以上を確かめ、全担当で共通の重い処理のロック（/tmp/claude-1000/heavy.lock）を取って 1 本ずつ流す（11/27 の決まり）。開催日（土日と mr_train.RACE_DAYS）の 07:30〜17:00（JST）は、
# mr_train が部品ごとに確かめて止める（終了コード 1 で queue も止まる）。部品ごとに保存するので、続きから流せる。
#   bash src/research/model_relabel/queue.sh v5bag:win:orig v5bag:top3:orig
set -u
cd "$(dirname "$0")/../../.."
export KEIBA_SESSION=model-relabel PYTHONPYCACHEPREFIX=/tmp/claude-1000/pyc MODEL_RELABEL_NTHREAD=2
export KEIBA_TABLES_DIR="${KEIBA_TABLES_DIR:-/home/jovyan/work/keiba-vpn/data/page_reference/tables}"
PY=/opt/conda/bin/python
for job in "$@"; do
  IFS=: read -r fs tg var split <<<"$job"; split=${split:-time}
  while :; do
    if $PY -c "import sys; from src.research.model_relabel.mr_train import blocked; sys.exit(0 if blocked() else 1)"; then
      echo "[queue] 開催日の日中なので止める: $job 以降"; exit 3
    fi
    load=$(cut -d' ' -f1 /proc/loadavg); mem=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if awk -v l="$load" 'BEGIN{exit !(l<=8)}' && [ "$mem" -ge 20 ]; then break; fi
    echo "[queue] 待つ（load $load・空き ${mem}GB）"; sleep 120
  done
  echo "[queue] 開始 $job $(TZ=Asia/Tokyo date '+%m/%d %H:%M')"
  flock /tmp/claude-1000/heavy.lock nice -n 19 $PY -m src.research.model_relabel.mr_train run --fs "$fs" --target "$tg" --variant "$var" --split "$split" || { echo "[queue] 失敗・停止 $job"; exit 1; }
done
echo "[queue] 全部終わり $(TZ=Asia/Tokyo date '+%m/%d %H:%M')"
