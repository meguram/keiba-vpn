#!/usr/bin/env bash
# AdSense の確かめ（外から見える ads.txt・robots・meta と、審査の状態）。統括役が 1 日 1 回、手で流す（cron は足さない）。
#   bash scripts/ops/adsense_check.sh            # 審査の状態も読む（資格が無ければ「未設定」）
#   bash scripts/ops/adsense_check.sh --no-api   # 外から見える所だけ
# 費用: 外への HTTP 数回と AdSense API 2〜3 回だけ（GCP・Cloudflare の課金は無い）。
set -uo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
exec "$PY" -m src.ops.adsense_check "$@"
