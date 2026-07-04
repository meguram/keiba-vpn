#!/usr/bin/env bash
# =============================================================================
# リポジトリを 1 時間ごとに git pull で最新化（cron 用）
#
# Usage:
#   bash scripts/cron/git_pull_hourly.sh
#   bash scripts/cron/git_pull_hourly.sh --dry-run
#
# 環境変数:
#   KEIBA_GIT_PULL_BRANCH  追跡ブランチ（未設定時は現在ブランチ）
#   KEIBA_GIT_PULL_REMOTE  リモート名（既定 origin）
#   KEIBA_GIT_PULL_ON_DIRTY  1 なら未コミット変更があっても pull を試行（既定 0 = スキップ）
#
# cron 登録: bash scripts/cron/setup_git_pull_cron.sh install
#            または setup_all_cron.sh install
# =============================================================================

set -euo pipefail

export TZ="${TZ:-Asia/Tokyo}"

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/git_pull.log"
LOCK_FILE="${LOG_DIR}/git_pull.lock"
MAX_LOG_LINES=5000

REMOTE="${KEIBA_GIT_PULL_REMOTE:-origin}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h | --help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
  esac
done

log() {
  echo "$(date -Is) git_pull: $*" | tee -a "$LOG_FILE"
}

rotate_log_if_needed() {
  if [[ -f "$LOG_FILE" ]]; then
    local lines
    lines=$(wc -l <"$LOG_FILE" 2>/dev/null || echo 0)
    if [[ "$lines" -gt "$MAX_LOG_LINES" ]]; then
      tail -n 2000 "$LOG_FILE" >"${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
      log "log rotated (${lines} → 2000 lines)"
    fi
  fi
}

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "skip (lock held)"
  exit 0
fi

rotate_log_if_needed

if ! command -v git >/dev/null 2>&1; then
  log "ERROR: git コマンドがありません"
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log "skip (not a git repository)"
  exit 0
fi

BRANCH="${KEIBA_GIT_PULL_BRANCH:-$(git branch --show-current 2>/dev/null || true)}"
if [[ -z "$BRANCH" ]]; then
  log "skip (detached HEAD / no branch)"
  exit 0
fi

if [[ "${KEIBA_GIT_PULL_ON_DIRTY:-0}" != "1" ]]; then
  if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    log "skip (working tree dirty on branch=${BRANCH})"
    exit 0
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "dry-run fetch ${REMOTE} (branch=${BRANCH})"
  git fetch --dry-run "$REMOTE" "$BRANCH" 2>&1 | tee -a "$LOG_FILE" || true
  exit 0
fi

log "start fetch ${REMOTE} ${BRANCH}"
before_head="$(git rev-parse --short HEAD 2>/dev/null || echo none)"

if ! git fetch "$REMOTE" "$BRANCH" >>"$LOG_FILE" 2>&1; then
  log "ERROR: git fetch failed (branch=${BRANCH})"
  exit 1
fi

if ! git merge-base --is-ancestor HEAD "${REMOTE}/${BRANCH}" 2>/dev/null; then
  if ! git merge-base --is-ancestor "${REMOTE}/${BRANCH}" HEAD 2>/dev/null; then
    log "ERROR: diverged from ${REMOTE}/${BRANCH} — manual merge required (ff-only)"
    exit 1
  fi
fi

if ! git pull --ff-only "$REMOTE" "$BRANCH" >>"$LOG_FILE" 2>&1; then
  log "ERROR: git pull --ff-only failed (branch=${BRANCH})"
  exit 1
fi

after_head="$(git rev-parse --short HEAD 2>/dev/null || echo none)"
if [[ "$before_head" == "$after_head" ]]; then
  log "ok (already up to date ${after_head})"
else
  log "ok updated ${before_head} → ${after_head}"
fi
