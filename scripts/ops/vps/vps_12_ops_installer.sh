#!/usr/bin/env bash
# =============================================================================
# ops ファイルの入れ替えスクリプト（install_ops_files.sh）と、deploy ユーザー用の sudoers を入れる。VPS・root で 1 回だけ実行。冪等。
#
#   bash vps_12_ops_installer.sh            # 入れる（root で。ConoHa のコンソールで root でログインするか、su - で root になってから）
#   bash vps_12_ops_installer.sh --remove   # 外す（sudoers とスクリプトを消す。退避と記録は残す）
#
# 入れた後、deploy ユーザーは `sudo -n /opt/keiba-ui/bin/install_ops_files.sh install|rollback|status ...` だけを、パスワードなしで実行できる。
# 入れ替えられるのは install_ops_files.sh の中の許可リストのファイルだけで、元は取り込み済みの取得の配布物（root 所有）に限る。
# 3 ファイル（このスクリプト・install_ops_files.sh・sudoers-keiba-ops-install）は同じフォルダに置いてもよい（/tmp など）。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
[ "$(id -u)" = 0 ] || { echo "root で実行してください（su - で root になってから bash $0）" >&2; exit 1; }

BIN=/opt/keiba-ui/bin/install_ops_files.sh
SUDOERS=/etc/sudoers.d/keiba-ops-install

if [ "${1:-}" = "--remove" ]; then
    rm -f "$SUDOERS" "$BIN"
    echo "外しました: $SUDOERS $BIN（/var/lib/keiba-ops/backup と /var/log/keiba-ops-install.log は残しています）"
    exit 0
fi

SRC_BIN="$HERE/install_ops_files.sh"
SRC_SUDOERS="$HERE/templates/sudoers-keiba-ops-install"
[ -f "$SRC_SUDOERS" ] || SRC_SUDOERS="$HERE/sudoers-keiba-ops-install"
[ -f "$SRC_BIN" ] && [ -f "$SRC_SUDOERS" ] || { echo "install_ops_files.sh と sudoers-keiba-ops-install が見つかりません" >&2; exit 1; }

bash -n "$SRC_BIN"
command -v flock >/dev/null && command -v sha256sum >/dev/null && command -v python3 >/dev/null || { echo "flock・sha256sum・python3 が必要です" >&2; exit 1; }
install -d -m 755 -o root -g root /opt/keiba-ui/bin
# sudo で動かすスクリプトの置き場所とその親が root 所有で、group/other から書けないこと（書ければスクリプトを差し替えて root を取れる）
for d in /opt/keiba-ui/bin /opt/keiba-ui /opt /; do
    [ ! -L "$d" ] && [ "$(stat -c %u "$d")" = 0 ] && (( (8#$(stat -c %a "$d") & 8#022) == 0 )) \
        || { echo "中止: $d が root 所有でない・リンク・書ける（$(stat -c '%U %a' "$d")）。直してから実行してください" >&2; exit 1; }
done
install -d -m 700 -o root -g root /var/lib/keiba-ops /var/lib/keiba-ops/backup
install -m 755 -o root -g root "$SRC_BIN" "$BIN"

tmp="$(mktemp)"
install -m 440 -o root -g root "$SRC_SUDOERS" "$tmp"
visudo -cf "$tmp" >/dev/null || { rm -f "$tmp"; echo "sudoers が不正のため入れませんでした" >&2; exit 1; }
mv "$tmp" "$SUDOERS"

echo "入れました: $BIN（root 所有）と $SUDOERS"
echo "確認: sudo -u deploy sudo -n $BIN status"
