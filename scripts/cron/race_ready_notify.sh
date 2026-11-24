#!/usr/bin/env bash
# レースごとの「予測・推奨が画面に出た（見込み）」を stg・本番の Slack に送る（src/scripts/ops/race_ready_notify.py）。毎分 cron から。
# 読むのはホストの手元のファイルだけ（GCS・API に触れない。費用 0 円）。非開催日はレース一覧が無いのですぐ終わる。
# 送るのは KEIBA_SLACK_NOTIFY=1（.env）かつ目印 data/local/meta/race_ready_notify.on があるときだけ（無ければ送らずにログだけ）。
# 二重起動しない（flock -n）。本体の推論の邪魔をしないよう nice 19。手順: docs/operations/slack_race_ready_plan_20261119.md
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 1
# .env（Webhook の変数）は本体が race_ready_notify._load_env で読む（shell で source しない。上書きしない。値はログに出さない）
mkdir -p logs
PY="${KEIBA_PYTHON:-/opt/conda/bin/python3}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec 9>"/tmp/keiba-race-ready-notify.lock"
flock -n 9 || exit 0
export TZ=Asia/Tokyo PYTHONPATH="$PROJECT_DIR"
exec nice -n 19 "$PY" -m src.scripts.ops.race_ready_notify "${@:2}" >> logs/race_ready_notify.log 2>&1
