#!/usr/bin/env bash
# prod（VPS）の状況の収集を、ホストの jovyan の crontab に 3 分おきで登録する（監視ポータルの /prod と /services が読む）。
#   手順は 2 つに分けてある（.env 編集や再起動と一括にしない）:
#     sudo -n bash scripts/ops/install_prod_status_cron.sh dir     # 出力先フォルダを作る（root。jovyan が書ける所有にする）
#     bash scripts/ops/install_prod_status_cron.sh cron            # jovyan の crontab に 1 行足す（冪等。タグ KEIBA-PROD-STATUS）
#     bash scripts/ops/install_prod_status_cron.sh once            # 今すぐ 1 回収集して結果の要約を出す
#     bash scripts/ops/install_prod_status_cron.sh remove          # crontab から外す
# 収集は読み取りだけ: VPS へは固定の採取スクリプトを ssh（keiba-vps）で流す／GCP は読み取り専用の gcloud_ro.sh。何も変更しない。
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
DIR="$ROOT/data/local/meta/prod_status"
PY="${PYTHON:-/opt/conda/bin/python}"
TAG="KEIBA-PROD-STATUS"
LINE="*/3 * * * * cd $ROOT && $PY -m src.scripts.ops.collect_prod_status >> $DIR/collect.log 2>&1 # $TAG"
case "${1:-}" in
  dir)
    [ "$(id -u)" = 0 ] || { echo "root で実行してください: sudo -n bash $0 dir" >&2; exit 1; }
    install -d -o jovyan -g users "$DIR" && echo "作成: $DIR" ;;
  cron)
    [ -w "$DIR" ] || { echo "$DIR に書けません。先に: sudo -n bash $0 dir" >&2; exit 1; }
    cur="$(crontab -l 2>/dev/null || true)"
    if printf '%s\n' "$cur" | grep -q "$TAG"; then echo "登録済みです（$TAG）"; exit 0; fi
    printf '%s\n%s\n' "$cur" "$LINE" | sed '/^$/N;/^\n$/D' | crontab -
    echo "登録しました:"; crontab -l | grep "$TAG" ;;
  once)
    cd "$ROOT" && "$PY" -m src.scripts.ops.collect_prod_status ;;
  remove)
    crontab -l 2>/dev/null | grep -v "$TAG" | crontab - && echo "外しました" ;;
  *) echo "使い方: $0 dir|cron|once|remove" >&2; exit 2 ;;
esac
