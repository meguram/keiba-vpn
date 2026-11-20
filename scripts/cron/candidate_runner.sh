#!/usr/bin/env bash
# 候補の影のランナー（新しい候補＋第 2 段＋凍結した買い方の判定。src/pipeline/inference/candidate_runner.py）を起動する。
# 既定はオフ（config/candidate_runtime.json の enabled か KEIBA_CANDIDATE_RUNNER=1 のときだけ動く）。開催の無い日はすぐ終わる。
# 二重起動しない（flock）。本体の推論より先に落ちるよう nice 10（プロセスの中で oom_score_adj=1000 も付ける）。
# 手順: docs/operations/weekend_launch_20261121.md
set -u
PROJECT_DIR="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
PY="${KEIBA_PYTHON:-/opt/conda/bin/python3}"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs
exec 9>"/tmp/keiba-candidate-runner.lock"
if ! flock -n 9; then
  echo "[$(date '+%Y-%m-%dT%H:%M:%S')] candidate_runner はすでに動いている。終わる"
  exit 0
fi
export TZ=Asia/Tokyo PYTHONPATH="$PROJECT_DIR"
nice -n 10 "$PY" -m src.pipeline.inference.candidate_runner "${@:2}" >> logs/candidate_runner.log 2>&1
rc=$?
echo "[$(date '+%Y-%m-%dT%H:%M:%S')] candidate_runner 終了 rc=$rc" >> logs/candidate_runner.log
exit $rc
