#!/usr/bin/env bash
# VPS の初期設定（Ubuntu 22.04 / 24.04 想定）。root で実行。何度実行しても同じ結果になるように書いてある。
#   bash vps_01_base_setup.sh --ssh-mode keep|ip:<自分のIP>|closed [--deploy-pubkey-file <公開鍵ファイル>] [--lock-ssh]
#
#   --ssh-mode keep       SSH ポートを全体に許可（初回だけ。最短。終わったら ip: か closed に変える）
#   --ssh-mode ip:1.2.3.4 SSH ポートをその IP だけに許可
#   --ssh-mode closed     受信を全拒否（SSH を公開しない。管理は Cloudflare Tunnel 経由の SSH か ConoHa のコンソール(VNC)）
#   --ssh-port N          SSH ポート。省略時は動いている sshd の設定から読む（この VPS は 2254）
# 締め出し防止: ufw を締める前に必ず別のターミナルで deploy の鍵ログインを確認する。ufw を有効にしても今の接続は切れない（新規接続だけが拒否される）。
#   --lock-ssh            パスワード認証と root ログインを無効化（deploy の鍵で入れることを確認してから付ける）
# 注意: ConoHa のコントロールパネル側（セキュリティグループ/ポート設定）にも別のフィルタがあることがある。両方確認すること。
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "root で実行してください" >&2; exit 1; }
SSH_MODE=""; PUBKEY=""; LOCK=0; SSH_PORT=""
while [ $# -gt 0 ]; do case "$1" in
  --ssh-mode) SSH_MODE="$2"; shift 2;;
  --deploy-pubkey-file) PUBKEY="$2"; shift 2;;
  --lock-ssh) LOCK=1; shift;;
  --ssh-port) SSH_PORT="$2"; shift 2;;
  *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ -n "$SSH_MODE" ] || { echo "--ssh-mode が必要です（keep | ip:<ADDR> | closed）" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
[ -n "$SSH_PORT" ] || SSH_PORT="$(sshd -T 2>/dev/null | awk '/^port /{print $2; exit}')"
SSH_PORT="${SSH_PORT:-22}"
echo "SSH ポート: $SSH_PORT"

echo "== 1/9 OS 更新とパッケージ"
apt-get update -y && apt-get -o Dpkg::Options::=--force-confold upgrade -y
apt-get install -y ufw fail2ban unattended-upgrades curl ca-certificates gnupg xz-utils rsync jq logrotate nginx xxd lsb-release
timedatectl set-timezone Asia/Tokyo || true

echo "== 2/9 ユーザー（deploy=配布用ログイン、keibaui=サービス実行用でログイン不可）"
id deploy >/dev/null 2>&1 || adduser --disabled-password --gecos "" deploy
id keibaui >/dev/null 2>&1 || useradd --system --home-dir /srv/keiba-ui --shell /usr/sbin/nologin keibaui
usermod -aG keibaui deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
if [ -n "$PUBKEY" ]; then
  [ -r "$PUBKEY" ] || { echo "公開鍵ファイルを読めません: $PUBKEY" >&2; exit 1; }
  grep -q '^ssh-\|^ecdsa-' "$PUBKEY" || { echo "公開鍵ではないようです（秘密鍵を渡さない）" >&2; exit 1; }
  grep -q 'PRIVATE KEY' "$PUBKEY" && { echo "秘密鍵が含まれています。公開鍵（.pub）だけを渡してください" >&2; exit 1; }
  touch /home/deploy/.ssh/authorized_keys
  while IFS= read -r line; do [ -n "$line" ] && ! grep -qxF "$line" /home/deploy/.ssh/authorized_keys && echo "$line" >> /home/deploy/.ssh/authorized_keys; done < "$PUBKEY"
  chown deploy:deploy /home/deploy/.ssh/authorized_keys; chmod 600 /home/deploy/.ssh/authorized_keys
fi

echo "== 3/9 swap（2GB の VPS 用）"
if [ "$(swapon --show --noheadings | wc -l)" = 0 ]; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
cat > /etc/sysctl.d/99-keiba.conf <<'SYS'
vm.swappiness=10
net.ipv4.conf.all.rp_filter=1
net.ipv4.tcp_syncookies=1
SYS
sysctl --system >/dev/null

echo "== 4/9 自動セキュリティ更新"
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'AU'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
AU

echo "== 5/9 SSH の設定"
install -d /etc/ssh/sshd_config.d
if [ "$LOCK" = 1 ]; then
  [ -s /home/deploy/.ssh/authorized_keys ] || { echo "deploy の authorized_keys が空です。鍵で入れることを確認してから --lock-ssh を付けてください" >&2; exit 1; }
  cat > /etc/ssh/sshd_config.d/99-keiba.conf <<'SSHD'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
PubkeyAuthentication yes
MaxAuthTries 3
AllowUsers deploy
SSHD
  sshd -t && systemctl reload ssh 2>/dev/null || systemctl reload sshd
  echo "  パスワード認証と root ログインを無効化しました（deploy の鍵のみ）"
else
  echo "  --lock-ssh なし: SSH の認証設定は変更していません（鍵で入れることを確認してから --lock-ssh で再実行）"
fi

echo "== 6/9 ファイアウォール（ufw）: 受信は既定で全拒否"
ufw --force reset >/dev/null
ufw default deny incoming >/dev/null; ufw default allow outgoing >/dev/null
case "$SSH_MODE" in
  keep) ufw allow "${SSH_PORT}/tcp" >/dev/null; echo "  ${SSH_PORT}/tcp を全体に許可（暫定）";;
  ip:*) ADDR="${SSH_MODE#ip:}"; ufw allow from "$ADDR" to any port "$SSH_PORT" proto tcp >/dev/null; echo "  ${SSH_PORT}/tcp を $ADDR だけに許可";;
  closed) echo "  SSH 受信なし（Cloudflare Tunnel 経由の SSH か、ConoHa のコンソールで管理）";;
  *) echo "--ssh-mode の値が不正です" >&2; exit 1;;
esac
# 80/443 は ufw 側で開けない（公開は cloudflared の外向き接続のみ）。ConoHa のセキュリティグループ IPv4v6-Web(80/443) は
# 新構成が動いてから外す（docs/operations/vps_ui_deploy.md 第 8 節）
ufw --force enable >/dev/null
ufw status verbose | head -12

echo "== 7/9 fail2ban"
cat > /etc/fail2ban/jail.d/keiba.local <<'F2B'
[sshd]
enabled = true
port = __SSH_PORT__
maxretry = 4
findtime = 10m
bantime = 1h
F2B
sed -i "s/__SSH_PORT__/${SSH_PORT}/" /etc/fail2ban/jail.d/keiba.local
systemctl enable --now fail2ban >/dev/null 2>&1 || true

echo "== 8/9 ディレクトリ"
install -d -m 755 /etc/keiba-ui /var/lib/keiba-ui /var/log/keiba-ui
install -d -m 755 -o deploy -g keibaui /srv/keiba-ui /srv/keiba-ui/releases /srv/keiba-ui/incoming /srv/keiba-ui/shared

echo "== 9/9 確認"
echo "  受信を待っているポート:"; ss -ltn | awk 'NR==1 || /0\.0\.0\.0|\[::\]/'
echo "完了。次は vps_02_install_stack.sh"
