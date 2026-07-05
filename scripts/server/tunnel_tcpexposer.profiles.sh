# shellcheck shell=bash
# tcpexposer トンネル — dev / stg プロファイル定義
#
# dev … モック UI 確認（localhost:3000 → meguai-dev）
# stg … 本番相当スタック（localhost:3001 → meguai-stg）
# prod … VPS クローン想定（将来 meguai 等。ローカル tcpexposer は通常オフ）

apply_tcpexposer_profile() {
  local profile="${1:-dev}"

  TCPEXPOSER_PROFILE="$profile"
  USER_NAME="${KEIBA_TCPEXPOSER_USER:-megukeiba}"
  REMOTE_PORT="${KEIBA_TCPEXPOSER_REMOTE_PORT:-80}"
  KEY_PATH="${KEIBA_TCPEXPOSER_KEY:-/root/.ssh/keiba-vpn-local}"

  case "$profile" in
    dev)
      DOMAIN="${KEIBA_TCPEXPOSER_DOMAIN:-meguai-dev}"
      LOCAL_PORT="${KEIBA_TCPEXPOSER_LOCAL_PORT:-3000}"
      ;;
    stg)
      DOMAIN="${KEIBA_TCPEXPOSER_DOMAIN:-meguai-stg}"
      LOCAL_PORT="${KEIBA_TCPEXPOSER_LOCAL_PORT:-3001}"
      ;;
    prod)
      DOMAIN="${KEIBA_TCPEXPOSER_DOMAIN:-meguai}"
      LOCAL_PORT="${KEIBA_TCPEXPOSER_LOCAL_PORT:-3001}"
      ;;
    *)
      echo "[tunnel_tcpexposer] 不明なプロファイル: $profile（dev / stg / prod）" >&2
      return 1
      ;;
  esac
}

describe_tcpexposer_profile() {
  echo "  profile=${TCPEXPOSER_PROFILE}  ${DOMAIN}.tcpexposer.com → localhost:${LOCAL_PORT}"
}
