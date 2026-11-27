#!/usr/bin/env bash
# 血統予想家（めぐ分析の欄）。手元のファイルだけを読み、GCS を読まない（CLAUDE.md の費用の上限）。
#   eve … 前の夜に、翌日の出馬表の馬の生の値と欄（一覧は発走の約 60 分前に出す）を作る（重い・20:30 JST）。
#   t60 … 開催日の 09:00〜16:55 JST に 5 分おき。T-60 の単勝オッズの手元の写しが届いたレースの一覧を確定する（軽い）。
# 同時に 2 本走らない（flock）・nice 19。手順と費用の見積もり: 血統予想家の実装の計画（persona_impl_plan.md）
# Usage: bash scripts/cron/pedigree_persona.sh {eve|t60} [PROJECT_DIR]
set -u
MODE="${1:?eve|t60}"
PROJECT_DIR="${2:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
PY="${PYTHON:-/opt/conda/bin/python3}"
cd "$PROJECT_DIR" || exit 1
case "$MODE" in
  eve)
    exec flock -n /tmp/keiba_pedigree_persona_eve.lock env OMP_NUM_THREADS=4 nice -n 19 "$PY" -m src.research.pedigree_persona.eve ;;
  t60)
    exec nice -n 19 "$PY" -m src.research.pedigree_persona.t60 ;;          # flock は道具の中（対象が無ければ 1 秒以内に終わる）
  *)
    echo "usage: pedigree_persona.sh {eve|t60}" >&2; exit 2 ;;
esac
