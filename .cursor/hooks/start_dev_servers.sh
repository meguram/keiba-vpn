#!/usr/bin/env bash
# Cursor sessionStart — dev ローカルサーバ（モック UI）自動起動。KEIBA_ENV=dev のときだけ動く。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/scripts/server/autostart_dev.sh"
