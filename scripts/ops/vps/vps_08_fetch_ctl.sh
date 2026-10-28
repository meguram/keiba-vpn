#!/usr/bin/env bash
# =============================================================================
# 取得の実行役の操作スクリプト（fetch_agent_ctl.sh）と、deploy ユーザー用の sudoers を入れる。VPS・root で 1 回だけ実行。冪等。
#
#   sudo bash vps_08_fetch_ctl.sh            # 入れる
#   sudo bash vps_08_fetch_ctl.sh --remove   # 外す（sudoers とスクリプトを消す）
#
# 入れた後、deploy ユーザーは `sudo -n /opt/keiba-ui/bin/fetch_agent_ctl.sh <操作>` だけを、パスワードなしで実行できる。
# それ以外の root 権限は増えない。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
[ "$(id -u)" = 0 ] || { echo "root で実行してください（sudo bash $0）" >&2; exit 1; }

BIN=/opt/keiba-ui/bin/fetch_agent_ctl.sh
SUDOERS=/etc/sudoers.d/keiba-fetch-ctl

if [ "${1:-}" = "--remove" ]; then
    rm -f "$SUDOERS" "$BIN"
    echo "外しました: $SUDOERS $BIN"
    exit 0
fi

SRC_CTL="$HERE/fetch_agent_ctl.sh"
SRC_SUDOERS="$HERE/templates/sudoers-keiba-fetch-ctl"
[ -f "$SRC_SUDOERS" ] || SRC_SUDOERS="$HERE/sudoers-keiba-fetch-ctl"      # 3 ファイルを同じフォルダに置いた場合
[ -f "$SRC_CTL" ] && [ -f "$SRC_SUDOERS" ] || { echo "fetch_agent_ctl.sh と sudoers-keiba-fetch-ctl が見つかりません" >&2; exit 1; }

bash -n "$SRC_CTL"
install -d -m 755 -o root -g root /opt/keiba-ui/bin
install -m 755 -o root -g root "$SRC_CTL" "$BIN"

tmp="$(mktemp)"
install -m 440 -o root -g root "$SRC_SUDOERS" "$tmp"
visudo -cf "$tmp" >/dev/null || { rm -f "$tmp"; echo "sudoers が不正のため入れませんでした" >&2; exit 1; }
mv "$tmp" "$SUDOERS"

echo "入れました: $BIN（root 所有）と $SUDOERS"
echo "確認: sudo -u deploy sudo -n $BIN status"
