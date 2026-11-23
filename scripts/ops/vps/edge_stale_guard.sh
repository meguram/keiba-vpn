#!/usr/bin/env bash
# VPS: ホストの作り直し（edge_sync）が止まったら、フォルダ配信を自動で止める（keiba-edge-guard@<env>.timer が 1 分ごと。deploy で動く）。
# docs/operations/edge_cache_plan_20261121.md §11。
#
# - 見るのは VPS の中のファイルだけ（/srv/keiba-edge/<env>/current/_meta/heartbeat.json の更新時刻）。GCS・Cloud Run には触れない。
#   heartbeat はホストが作り直しの回ごとに書き、tar の展開で更新時刻（ホストで書いた時刻）が保たれる。
# - 確定前（current/pre/）: heartbeat が EDGE_STALE_PRE_SEC（既定 600 秒）より古ければ、pre_live を空のフォルダ（_empty）に付け替え、
#   印 _ctl/STALE_PRE.auto を置く（nginx・cache-warm は確定前を返さない）。開催日の予測・オッズで中身が変わるので、短く止める。
#   新しくなれば pre_live を current/pre に戻す。
# - 全部: EDGE_STALE_SEC（既定 1800 秒。深夜の全部の作り直しが長くかかっても止めないよう余裕を持つ）より古ければ _ctl/OFF.auto を置く。
# - 新しくなれば、自分の印（.auto）だけを外す。人が置いた _ctl/OFF には触れない（別の名前）。
set -euo pipefail
ENV_NAME="${1:?使い方: edge_stale_guard.sh prod|stg}"
D="/srv/keiba-edge/$ENV_NAME"
HB="$D/current/_meta/heartbeat.json"
PRE="${EDGE_STALE_PRE_SEC:-600}"
ALL="${EDGE_STALE_SEC:-1800}"
[ -d "$D/_ctl" ] || exit 0
age=999999
[ -f "$HB" ] && age=$(( $(date +%s) - $(stat -c %Y "$HB") ))

mark() {   # mark <印> <閾値> <説明>
  if [ "$age" -gt "$2" ]; then
    if [ ! -e "$D/_ctl/$1" ]; then
      touch "$D/_ctl/$1"
      logger -t keiba-edge-guard "$ENV_NAME: heartbeat が ${age} 秒前なので、$3を止めた（_ctl/$1）"
    fi
  elif [ -e "$D/_ctl/$1" ]; then
    rm -f "$D/_ctl/$1"
    logger -t keiba-edge-guard "$ENV_NAME: heartbeat が戻った（${age} 秒前）ので、$3を戻した"
  fi
}
mark STALE_PRE.auto "$PRE" "確定前のフォルダ配信"
# 確定前の切り替え（nginx は pre_live を見る。symlink の付け替えは mv -T で一度に）
mkdir -p "$D/_empty"
want="current/pre"; [ -e "$D/_ctl/STALE_PRE.auto" ] && want="_empty"
if [ "$(readlink "$D/pre_live" 2>/dev/null || true)" != "$want" ]; then
  ln -sfn "$want" "$D/.pre_live.tmp" && mv -T "$D/.pre_live.tmp" "$D/pre_live"
fi
mark OFF.auto "$ALL" "フォルダ配信の全部"
