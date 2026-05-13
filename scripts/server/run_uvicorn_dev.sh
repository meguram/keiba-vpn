#!/usr/bin/env bash
# 開発用: scripts/ 以下の編集で全リロードされないよう reload 対象を限定
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
mkdir -p logs
exec python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}" \
  --reload \
  --reload-dir src/api \
  --reload-dir src/scraper
