#!/usr/bin/env bash
# Flask /api/v1 API サーバー起動（仕様準拠）
set -euo pipefail
cd "$(dirname "$0")/../.."
# FLASK_PORT 未指定時は main.py が config/ports.env から現在の KEIBA_ENV 用ポートを決める
exec python main.py --flask-api
