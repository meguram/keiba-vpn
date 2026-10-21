#!/usr/bin/env bash
# VPS の初期設定（Ubuntu 22.04 / 24.04 想定）。root で実行。何度実行しても同じ結果になるように書いてある。
#   bash vps_01_base_setup.sh --ssh-mode keep|ip:<自分のIP>|closed [--deploy-pubkey-file <公開鍵ファイル>] [--lock-ssh]
#
#   --ssh-mode keep       SSH ポートを全体に許可（初回だけ。最短。終わったら ip: か closed に変える）
#   --ssh-mode ip:1.2.3.4 SSH ポートをその IP だけに許可
#   --ssh-mode closed     受信を全拒否（SSH を公開しない。管理は Cloudflare Tunnel 経由の SSH か ConoHa のコンソール(VNC)）
#   --admin-user NAME     今ログインしている管理ユーザー（sudo にパスワードが要るユーザー。既定: sudo を実行した人）。--lock-ssh のとき AllowUsers に残す
#   --ssh-port N          SSH ポート。省略時は動いている sshd の設定から読む（この VPS は 2254）
# 締め出し防止: ufw を締める前に必ず別のターミナルで deploy の鍵ログインを確認する。ufw を有効にしても今の接続は切れない（新規接続だけが拒否される）。
#   --lock-ssh            パスワード認証と root ログインを無効化（deploy の鍵で入れることを確認してから付ける）
# 注意: ConoHa のコントロールパネル側（セキュリティグループ/ポート設定）にも別のフィルタがあることがある。両方確認すること。
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "root で実行してください" >&2; exit 1; }
SSH_MODE=""; PUBKEY=""; LOCK=0; SSH_PORT=""; ADMIN="${SUDO_USER:-}"
while [ $# -gt 0 ]; do case "$1" in
  --ssh-mode) SSH_MODE="$2"; shift 2;;
  --deploy-pubkey-file) PUBKEY="$2"; shift 2;;
  --lock-ssh) LOCK=1; shift;;
  --ssh-port) SSH_PORT="$2"; shift 2;;
  --admin-user) ADMIN="$2"; shift 2;;
  *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ -n "$SSH_MODE" ] || { echo "--ssh-mode が必要です（keep | ip:<ADDR> | closed）" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
# awk を途中で exit させると sshd -T が SIGPIPE で落ち、pipefail + set -e で無言のまま終了する（実機で発生）。最後まで読む
[ -n "$SSH_PORT" ] || SSH_PORT="$(sshd -T 2>/dev/null | awk '/^port /{p=$2} END{print p}' || true)"
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

echo "== 3b/9 メモリ逼迫への備え（journald 100MB・earlyoom。architecture_target.md 3.2）"
install -d /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/99-keiba.conf <<'JD'
[Journal]
SystemMaxUse=100M
RuntimeMaxUse=50M
MaxRetentionSec=2week
JD
systemctl restart systemd-journald 2>/dev/null || true
# earlyoom: MemAvailable が 5% 未満かつ swap の空きが 10% 未満になったら、取得の子プロセス（python）→ node の順に先に止める（nginx・cloudflared・sshd は守る）
apt-get install -y earlyoom
cat > /etc/default/earlyoom <<'EO'
EARLYOOM_ARGS="-r 3600 -m 5 -s 10 --prefer '(^|/)(python3?)$' --avoid '(^|/)(nginx|cloudflared|sshd|systemd|systemd-journald)$'"
EO
systemctl enable --now earlyoom >/dev/null 2>&1 || true
systemctl restart earlyoom 2>/dev/null || true

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
AllowUsers deploy __ADMIN__
SSHD
  if [ -n "$ADMIN" ]; then sed -i "s/__ADMIN__/${ADMIN}/" /etc/ssh/sshd_config.d/99-keiba.conf; else sed -i "s/ __ADMIN__//" /etc/ssh/sshd_config.d/99-keiba.conf; echo "  警告: 管理ユーザーが分からないため AllowUsers は deploy だけ（--admin-user を付けて再実行を推奨）"; fi
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
ufw status verbose | head -12 || true

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
