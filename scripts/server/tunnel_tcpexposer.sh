#!/usr/bin/env bash
# tcpexposer リバーストンネル（dev / stg）
#
# Usage:
#   ./scripts/server/tunnel_tcpexposer.sh [dev|stg]              # 前景接続
#   ./scripts/server/tunnel_tcpexposer.sh [dev|stg] autostart    # ワークスペース起動用
#   ./scripts/server/tunnel_tcpexposer.sh [dev|stg] background    # バックグラウンド + 自動再接続
#   ./scripts/server/tunnel_tcpexposer.sh [dev|stg] check        # 診断
#   ./scripts/server/tunnel_tcpexposer.sh [dev|stg] stop         # 停止
#   ./scripts/server/tunnel_tcpexposer.sh autostart-all          # dev + stg 両方
#   ./scripts/server/tunnel_tcpexposer.sh stop-all
#
# プロファイル（省略時 dev）:
#   dev  meguai-dev.tcpexposer.com  → localhost:3000  （モック UI）
#   stg  meguai-stg.tcpexposer.com  → localhost:3001  （本番相当 Next.js）
#
# 環境変数（任意）:
#   KEIBA_TCPEXPOSER_USER=megukeiba
#   KEIBA_TCPEXPOSER_KEY=/root/.ssh/keiba-vpn-local
#   KEIBA_AUTO_TCPEXPOSER=0   # autostart / autostart-all を無効化

set -euo pipefail

export TZ="${TZ:-Asia/Tokyo}"

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=tunnel_tcpexposer.profiles.sh
source "$ROOT/scripts/server/tunnel_tcpexposer.profiles.sh"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

TCPEXPOSER_PROFILE="${KEIBA_TCPEXPOSER_PROFILE:-dev}"
ACTION=""

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      dev|stg|prod)
        TCPEXPOSER_PROFILE="$1"
        shift
        ;;
      autostart|background|bg|check|-s|--status|stop|_loop|autostart-all|stop-all|-h|--help)
        ACTION="$1"
        shift
        ;;
      *)
        echo "不明な引数: $1" >&2
        exit 1
        ;;
    esac
  done
  ACTION="${ACTION:-foreground}"
}

parse_args "$@"

apply_tcpexposer_profile "$TCPEXPOSER_PROFILE" || exit 1

LOG_FILE="$LOG_DIR/tcpexposer_tunnel_${TCPEXPOSER_PROFILE}.log"
PID_FILE="$ROOT/.tcpexposer_tunnel.${TCPEXPOSER_PROFILE}.pid"
PUBLIC_URL="https://${DOMAIN}.tcpexposer.com/"

log() {
  echo "$(date -Is) [${TCPEXPOSER_PROFILE}] $*" | tee -a "$LOG_FILE"
}

ensure_key_permissions() {
  local dir key pub
  dir="$(dirname "$KEY_PATH")"
  key="$KEY_PATH"
  pub="${KEY_PATH}.pub"

  if [[ ! -f "$key" ]]; then
    log "ERROR: 秘密鍵がありません: $key"
    exit 1
  fi

  chmod 700 "$dir" 2>/dev/null || true
  chmod 600 "$key"
  if [[ -f "$pub" ]]; then
    chmod 644 "$pub"
  fi

  local key_mode dir_mode
  key_mode=$(stat -c '%a' "$key" 2>/dev/null || stat -f '%OLp' "$key")
  dir_mode=$(stat -c '%a' "$dir" 2>/dev/null || stat -f '%OLp' "$dir")
  if [[ "$key_mode" != "600" ]]; then
    log "ERROR: 秘密鍵の権限が 600 ではありません ($key_mode): $key"
    exit 1
  fi
  if [[ "$dir_mode" != "700" ]]; then
    log "ERROR: ~/.ssh の権限が 700 ではありません ($dir_mode): $dir"
    exit 1
  fi
}

check_local() {
  local code
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${LOCAL_PORT}/" 2>/dev/null || echo "000")
  if [[ "$code" =~ ^(200|304)$ ]]; then
    log "OK ローカル :${LOCAL_PORT} → HTTP ${code}"
    return 0
  fi
  log "WARN ローカル :${LOCAL_PORT} 未応答 (HTTP ${code})"
  case "$TCPEXPOSER_PROFILE" in
    dev) log "  → ./service_start --env dev" ;;
    stg) log "  → ./service_start --env stg" ;;
  esac
  return 1
}

check_external() {
  local code
  code=$(curl -s -m 8 -o /dev/null -w "%{http_code}" "$PUBLIC_URL" 2>/dev/null || echo "000")
  log "外部 ${PUBLIC_URL} → HTTP ${code}"
  [[ "$code" =~ ^(200|304)$ ]]
}

stop_tunnel() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      log "停止 PID=${pid}"
    fi
    rm -f "$PID_FILE"
  fi
  pkill -f "ssh.*-R ${DOMAIN}:.*@tcpexposer.com" 2>/dev/null || true
  pkill -f "ssh.*-R ${DOMAIN}:.*localhost:${LOCAL_PORT}.*@tcpexposer.com" 2>/dev/null || true
}

print_ssh_command() {
  cat <<EOF
ssh -i ${KEY_PATH} -N \\
  -o IdentitiesOnly=yes \\
  -o KexAlgorithms=diffie-hellman-group14-sha256 \\
  -o ServerAliveInterval=60 \\
  -o ExitOnForwardFailure=yes \\
  -R ${DOMAIN}:${REMOTE_PORT}:localhost:${LOCAL_PORT} \\
  ${USER_NAME}@tcpexposer.com
EOF
}

run_ssh_tunnel() {
  SSH_AUTH_SOCK= ssh \
    -i "$KEY_PATH" \
    -N \
    -o IdentitiesOnly=yes \
    -o KexAlgorithms=diffie-hellman-group14-sha256 \
    -o HostKeyAlgorithms=ssh-ed25519 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout=20 \
    -o BatchMode=yes \
    -R "${DOMAIN}:${REMOTE_PORT}:localhost:${LOCAL_PORT}" \
    "${USER_NAME}@tcpexposer.com"
}

run_check() {
  echo "=== tcpexposer 診断 (${TCPEXPOSER_PROFILE}) ==="
  describe_tcpexposer_profile
  ensure_key_permissions
  check_local || true
  echo "  秘密鍵: ${KEY_PATH} ($(stat -c '%a' "$KEY_PATH" 2>/dev/null || echo ?))"
  if [[ -f "${KEY_PATH}.pub" ]]; then
    echo "  公開鍵 fingerprint:"
    ssh-keygen -lf "${KEY_PATH}.pub" 2>/dev/null || true
  fi
  echo ""
  echo "  SSH コマンド:"
  print_ssh_command | sed 's/^/    /'
  echo ""
  echo "  公開 URL: ${PUBLIC_URL}"
  check_external || true
}

run_foreground() {
  ensure_key_permissions
  check_local || true
  stop_tunnel
  log "接続開始 ${DOMAIN}:${REMOTE_PORT} → localhost:${LOCAL_PORT}"
  echo ""
  echo "接続中... 成功するとこのターミナルは待機状態になります（正常）。"
  echo "別タブで開く: ${PUBLIC_URL}"
  echo ""
  while true; do
    log "ssh 接続試行"
    if run_ssh_tunnel >>"$LOG_FILE" 2>&1; then
      log "ssh 正常終了"
    else
      log "ssh 終了 (exit=$?)"
    fi
    local wait_time=$((RANDOM % 30 + 5))
    log "再接続まで ${wait_time}s"
    sleep "$wait_time"
  done
}

run_autostart() {
  if [[ "${KEIBA_AUTO_TCPEXPOSER:-1}" == "0" ]]; then
    log "autostart スキップ (KEIBA_AUTO_TCPEXPOSER=0)"
    return 0
  fi

  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "autostart スキップ (既に稼働 PID=${pid})"
      if pgrep -f "ssh.*-R ${DOMAIN}:.*localhost:${LOCAL_PORT}.*@tcpexposer.com" >/dev/null 2>&1; then
        return 0
      fi
      log "警告: ループは稼働中だが SSH が :${LOCAL_PORT} に未接続 — 再起動"
      stop_tunnel
    fi
  fi

  ensure_key_permissions
  check_local || log "警告: ローカル :${LOCAL_PORT} 未起動 — トンネルのみ先行起動"

  log "autostart: バックグラウンド + 自動再接続を開始"
  nohup "$0" "$TCPEXPOSER_PROFILE" _loop >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  log "autostart PID=$(cat "$PID_FILE") ログ=${LOG_FILE}"
}

run_background() {
  ensure_key_permissions
  check_local || true
  stop_tunnel
  log "バックグラウンド接続開始"
  nohup "$0" "$TCPEXPOSER_PROFILE" _loop >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 6
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    log "SSH ループ起動 PID=$(cat "$PID_FILE")"
    check_external || log "警告: 外部 URL がまだ 404 — 数秒待って再確認"
  else
    log "ERROR: バックグラウンド起動失敗。ログ: $LOG_FILE"
    tail -20 "$LOG_FILE"
    exit 1
  fi
}

run_loop() {
  ensure_key_permissions
  while true; do
    log "ssh 接続試行 (background loop)"
    run_ssh_tunnel >>"$LOG_FILE" 2>&1 || log "ssh 終了 (exit=$?)"
    local wait_time=$((RANDOM % 30 + 5))
    log "再接続まで ${wait_time}s"
    sleep "$wait_time"
  done
}

run_autostart_all() {
  if [[ "${KEIBA_AUTO_TCPEXPOSER:-1}" == "0" ]]; then
    log "autostart-all スキップ (KEIBA_AUTO_TCPEXPOSER=0)"
    return 0
  fi
  "$0" dev autostart
  "$0" stg autostart
}

run_stop_all() {
  "$0" dev stop
  "$0" stg stop
}

case "${ACTION:-foreground}" in
  check|-s|--status)
    run_check
    ;;
  stop)
    stop_tunnel
    log "トンネル停止"
    ;;
  stop-all)
    run_stop_all
    ;;
  background|bg)
    run_background
    ;;
  autostart)
    run_autostart
    ;;
  autostart-all)
    run_autostart_all
    ;;
  _loop)
    run_loop
    ;;
  -h|--help)
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    echo ""
    describe_tcpexposer_profile
    echo ""
    print_ssh_command
    ;;
  foreground)
    run_foreground
    ;;
esac
