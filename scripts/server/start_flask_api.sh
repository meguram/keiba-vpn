#!/usr/bin/env bash
# Flask /api/v1 API サーバー起動（仕様準拠）
set -euo pipefail
cd "$(dirname "$0")/../.."
export FLASK_PORT="${FLASK_PORT:-5000}"
exec python main.py --flask-api
