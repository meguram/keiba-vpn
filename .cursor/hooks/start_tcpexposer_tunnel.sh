#!/usr/bin/env bash
# Cursor sessionStart — tcpexposer トンネル自動起動（dev + stg）
set -euo pipefail

if [[ ! -t 0 ]]; then
  cat >/dev/null || true
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export TZ="${TZ:-Asia/Tokyo}"

exec bash "$ROOT/scripts/server/tunnel_tcpexposer.sh" autostart-all
