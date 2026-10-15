#!/usr/bin/env bash
# ホスト PC(stg)の最新データ健全性レポートをこの PC に取り込み、ダッシュボードを更新する。
# 読むだけ（ホストへは scp で latest.json を 1 つ取得するのみ。GCS には触れない）。
#
# Usage: bash scripts/server/fetch_stg_health.sh [ssh ホスト名(既定 keiba-remote)]
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"
HOST="${1:-keiba-remote}"
REMOTE_DIR="${KEIBA_REMOTE_DIR:-/home/jovyan/work/keiba-vpn}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
scp -q -o BatchMode=yes "$HOST:$REMOTE_DIR/data/local/meta/data_health/stg/latest.json" "$TMP/latest.json"
python3 -m src.data_health --import "$TMP/latest.json"
python3 -m src.data_health --dashboard
