#!/usr/bin/env bash
# ホスト（root）の crontab で、GCP / VPS に引き継いだジョブの行をコメントアウト／戻す。実行はユーザーが root で行う（このスクリプトは勝手に動かない）。
#
#   bash scripts/ops/cutover/host_cron_switch.sh status  --job-id ID [--tag TAG]
#   bash scripts/ops/cutover/host_cron_switch.sh disable --job-id ID [--tag TAG] [--dry-run]
#   bash scripts/ops/cutover/host_cron_switch.sh enable  --job-id ID [--tag TAG] [--dry-run]
#
# - 対象の行: コメントでなく、`cron_run.sh <ID> ` を含む行（--tag を付けたら、その行が `# <TAG>` タグを持つことも条件）。
#   KEIBA-VPN-ALL のように多数のジョブが共有するタグは、タグだけでは選ばない（--job-id が必須）。
# - 止め方: 行頭に `#KEIBA-CUTOVER:<ID># ` を付ける（enable はそれを外すだけ。元の行は一文字も変えない）。
# - 毎回 crontab を data/local/meta/cron_backups/ に退避する。対象が 0 行、または --max-lines（既定 4）を超えるときは何もしない。
# - disable は先に `python3 -m src.ops.job_owner can-stop <ID>` を通す（担当が GCP / VPS に渡っていて、ホストに残すジョブでないこと。
#   二重実行を避ける順序＝先に担当を渡し、観察してから行を止める）。--no-owner-check は検証用。
# - 止めた後に戻すときは enable → `python3 -m src.ops.job_owner unset <ID>` の順（行を戻してから担当を外す）。
set -euo pipefail

ACTION="${1:-}"; shift || true
ID=""; TAG=""; DRY=0; MAX_LINES=4; OWNER_CHECK=1
PROJECT_DIR="${KEIBA_PROJECT_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)}"
BACKUP_DIR="${KEIBA_CRON_BACKUP_DIR:-${PROJECT_DIR}/data/local/meta/cron_backups}"
CRONTAB="${CRONTAB_CMD:-crontab}"
while [ $# -gt 0 ]; do
  case "$1" in
    --job-id) ID="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --max-lines) MAX_LINES="$2"; shift 2 ;;
    --no-owner-check) OWNER_CHECK=0; shift ;;
    *) echo "不明な引数: $1" >&2; exit 2 ;;
  esac
done
case "$ACTION" in status|disable|enable) ;; *) echo "usage: $0 status|disable|enable --job-id ID [--tag TAG] [--dry-run]" >&2; exit 2 ;; esac
[ -n "$ID" ] || { echo "--job-id は必須です" >&2; exit 2; }
case "$ID" in *[!A-Za-z0-9._-]*) echo "job-id が不正です: $ID" >&2; exit 2 ;; esac

MARK="#KEIBA-CUTOVER:${ID}# "
CURRENT="$($CRONTAB -l 2>/dev/null || true)"

matches_active() {   # 有効な行で対象のもの
  local line="$1"
  case "$line" in "#"*|"") return 1 ;; esac
  case "$line" in *"cron_run.sh ${ID} "*) ;; *) return 1 ;; esac
  if [ -n "$TAG" ]; then case "$line" in *"${TAG}"*) ;; *) return 1 ;; esac; fi
  return 0
}
matches_disabled() {  # このスクリプトで止めた行
  case "$1" in "${MARK}"*) return 0 ;; *) return 1 ;; esac
}

n_active=0; n_disabled=0
while IFS= read -r line; do
  if matches_active "$line"; then n_active=$((n_active + 1)); fi
  if matches_disabled "$line"; then n_disabled=$((n_disabled + 1)); fi
done <<<"$CURRENT"

if [ "$ACTION" = status ]; then
  echo "job-id=${ID} tag=${TAG:-（指定なし）}: 有効 ${n_active} 行 / このスクリプトで停止中 ${n_disabled} 行"
  while IFS= read -r line; do
    if matches_active "$line" || matches_disabled "$line"; then echo "  $line"; fi
  done <<<"$CURRENT"
  exit 0
fi

if [ "$ACTION" = disable ]; then
  [ "$n_active" -gt 0 ] || { echo "対象の有効な行がありません（job-id=${ID}）。何もしません" >&2; exit 3; }
  [ "$n_active" -le "$MAX_LINES" ] || { echo "対象が ${n_active} 行で --max-lines(${MAX_LINES}) を超えます。タグ/ID を見直してください" >&2; exit 4; }
  if [ "$OWNER_CHECK" = 1 ]; then
    ( cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR" python3 -m src.ops.job_owner can-stop "$ID" ) || { echo "担当がまだ GCP / VPS に渡っていない、またはホストに残すジョブです。止めません" >&2; exit 5; }
  fi
else
  [ "$n_disabled" -gt 0 ] || { echo "戻す対象（#KEIBA-CUTOVER:${ID}# の行）がありません。何もしません" >&2; exit 3; }
fi

mkdir -p "$BACKUP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="${BACKUP_DIR}/crontab_before_cutover_${ACTION}_${ID}_${STAMP}.txt"
printf '%s\n' "$CURRENT" >"$BACKUP"
echo "crontab を退避しました: $BACKUP"

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
changed=0
while IFS= read -r line; do
  if [ "$ACTION" = disable ] && matches_active "$line"; then
    printf '%s\n' "${MARK}${line}" >>"$TMP"; changed=$((changed + 1))
  elif [ "$ACTION" = enable ] && matches_disabled "$line"; then
    printf '%s\n' "${line#"$MARK"}" >>"$TMP"; changed=$((changed + 1))
  else
    printf '%s\n' "$line" >>"$TMP"
  fi
done <<<"$CURRENT"

if [ "$DRY" = 1 ]; then
  echo "（dry-run）${ACTION}: ${changed} 行を変更する予定。差分:"
  diff <(printf '%s\n' "$CURRENT") "$TMP" || true
  exit 0
fi
$CRONTAB "$TMP"
echo "${ACTION} 完了: ${changed} 行"
$CRONTAB -l | grep -F -e "cron_run.sh ${ID} " -e "${MARK}" || true
echo "戻すとき: 退避 ${BACKUP} から crontab <ファイル> で復元できます"
