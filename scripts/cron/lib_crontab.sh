#!/usr/bin/env bash
# crontab の「タグの行だけ」を安全に入れ替える共通の関数（source して使う）。2026-11-13 に作った。
# それまでの登録スクリプトは `crontab -l 2>/dev/null ... || true` で、crontab -l が失敗すると root の crontab 全体を、登録する数行だけで置き換えうる作りだった。
#   crontab_replace_tag <TAG> <LINES> [--remove]
# - root でなければ中止（root 以外の利用者の crontab を、気付かずに変えないため）。テストでは KEIBA_CRON_ALLOW_NONROOT=1 で外す。
# - crontab -l が失敗したら中止（「no crontab」だけは空として続ける）。
# - 変える前の crontab を ${KEIBA_CRON_BACKUP_DIR:-$PROJECT_DIR/data/local/meta/cron_backups}/ に退避する（退避できなければ中止）。
# - 有効な行が KEIBA_CRON_MIN_LINES（既定 0）未満の crontab なら中止（root の crontab の取り違いに気付くため。ホストの root の登録スクリプトは 20 にする）。
# - 有効な行のうち「# TAG」で終わる行（またはその後に空白が続く行）だけを消し、--remove でなければ LINES を足す。コメントにした行・ほかのタグの行は残す。
# - 有効な行（空行とコメントを除く）の数が「前 − 消したタグの行 + 足した行」と合わなければ中止する。
crontab_replace_tag() {
  local tag="$1" lines="$2" mode="${3:-install}"
  local proj="${PROJECT_DIR:-$(pwd)}"
  if [ "${KEIBA_CRON_ALLOW_NONROOT:-0}" != 1 ] && [ "$(id -u)" != 0 ]; then
    echo "root で実行してください（sudo bash …）。root 以外の crontab は変えません。中止" >&2; return 1
  fi
  local now new err bkdir
  now="$(mktemp)"; new="$(mktemp)"; err="$(mktemp)"
  if ! crontab -l >"$now" 2>"$err"; then
    if grep -qi "no crontab" "$err"; then : >"$now"; else
      echo "crontab -l に失敗したので中止します: $(cat "$err")" >&2; rm -f "$now" "$new" "$err"; return 1
    fi
  fi
  bkdir="${KEIBA_CRON_BACKUP_DIR:-${proj}/data/local/meta/cron_backups}"
  if ! { mkdir -p "$bkdir" && cp "$now" "${bkdir}/crontab_before_${tag}_$(date +%Y%m%d_%H%M%S).txt"; }; then
    echo "crontab を退避できないので中止します（${bkdir}）" >&2; rm -f "$now" "$new" "$err"; return 1
  fi
  local min="${KEIBA_CRON_MIN_LINES:-0}" cur
  cur=$(grep -vcE '^[[:space:]]*(#|$)' "$now" || true)
  if [ "$cur" -lt "$min" ]; then
    echo "有効な行が少なすぎる crontab（${cur} 行 < ${min}）。取り違いの恐れがあるので中止" >&2; rm -f "$now" "$new" "$err"; return 1
  fi
  local re="# ${tag}([[:space:]]|\$)"
  awk -v re="$re" '!($0 ~ re && $0 !~ /^[[:space:]]*#/)' "$now" >"$new"
  local added=0
  if [ "$mode" != "--remove" ] && [ -n "$lines" ]; then
    printf '%s\n' "$lines" >>"$new"
    added=$(printf '%s\n' "$lines" | grep -vcE '^[[:space:]]*(#|$)' || true)
  fi
  local before tagged after want
  before=$(grep -vcE '^[[:space:]]*(#|$)' "$now" || true)
  tagged=$(grep -E "$re" "$now" | grep -vcE '^[[:space:]]*#' || true)
  after=$(grep -vcE '^[[:space:]]*(#|$)' "$new" || true)
  want=$(( before - tagged + added ))
  if [ "$after" -ne "$want" ]; then
    echo "行数が合いません（前 ${before}・消す ${tagged}・足す ${added}・後 ${after}）。中止" >&2; rm -f "$now" "$new" "$err"; return 1
  fi
  crontab "$new" || { rm -f "$now" "$new" "$err"; return 1; }
  rm -f "$now" "$new" "$err"
  return 0
}
