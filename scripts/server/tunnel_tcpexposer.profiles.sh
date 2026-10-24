# shellcheck shell=bash
# tcpexposer トンネル — dev / stg プロファイル定義
#
# ローカルポートの正は config/ports.env（フロントエンド / 監視ポータル）
# dev     … モック UI 確認（dev フロントエンド → meguai-dev）
# stg     … 本番相当スタック（stg フロントエンド → meguai-stg）
# monitor … 監視ポータル（KEIBA_PORT_MONITOR → meguai-monitor）
# prod    … VPS クローン想定（将来 meguai 等。ローカル tcpexposer は通常オフ）

# shellcheck source=ports.sh
source "$(dirname "${BASH_SOURCE[0]}")/ports.sh"
load_keiba_ports "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

apply_tcpexposer_profile() {
  local profile="${1:-dev}"

  TCPEXPOSER_PROFILE="$profile"
  USER_NAME="${KEIBA_TCPEXPOSER_USER:-megukeiba}"
  REMOTE_PORT="${KEIBA_TCPEXPOSER_REMOTE_PORT:-80}"
  KEY_PATH="${KEIBA_TCPEXPOSER_KEY:-/root/.ssh/keiba-vpn-local}"

  case "$profile" in
    dev)
      DOMAIN="${KEIBA_TCPEXPOSER_DOMAIN:-meguai-dev}"
      LOCAL_PORT="${KEIBA_TCPEXPOSER_LOCAL_PORT:-$KEIBA_PORT_DEV_FRONTEND}"
      ;;
    stg)
      DOMAIN="${KEIBA_TCPEXPOSER_DOMAIN:-meguai-stg}"
      LOCAL_PORT="${KEIBA_TCPEXPOSER_LOCAL_PORT:-$KEIBA_PORT_STG_FRONTEND}"
      ;;
    monitor)
      DOMAIN="${KEIBA_TCPEXPOSER_DOMAIN:-meguai-monitor}"
      LOCAL_PORT="${KEIBA_TCPEXPOSER_LOCAL_PORT:-$KEIBA_PORT_MONITOR}"
      ;;
    prod)
      DOMAIN="${KEIBA_TCPEXPOSER_DOMAIN:-meguai}"
      LOCAL_PORT="${KEIBA_TCPEXPOSER_LOCAL_PORT:-$KEIBA_PORT_PROD_FRONTEND}"
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
