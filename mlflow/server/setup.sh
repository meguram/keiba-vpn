#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════
# MLflow + Keiba API サーバー セットアップスクリプト
#
# 使い方:
#   chmod +x setup.sh
#   ./setup.sh
#
# 前提:
#   - Docker & Docker Compose がインストール済み
#   - (オプション) ドメインと SSL 証明書
# ═══════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║  MLflow + Keiba API セットアップ         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────
# 1. Basic認証のパスワードファイル生成
# ─────────────────────────────────────────
HTPASSWD_FILE="nginx/htpasswd"
if [ ! -f "$HTPASSWD_FILE" ]; then
    echo "[1/4] Basic認証の設定"
    read -p "  ユーザー名 (default: admin): " AUTH_USER
    AUTH_USER="${AUTH_USER:-admin}"
    read -sp "  パスワード: " AUTH_PASS
    echo ""

    if command -v htpasswd &> /dev/null; then
        htpasswd -cb "$HTPASSWD_FILE" "$AUTH_USER" "$AUTH_PASS"
    elif command -v openssl &> /dev/null; then
        HASH=$(openssl passwd -apr1 "$AUTH_PASS")
        echo "${AUTH_USER}:${HASH}" > "$HTPASSWD_FILE"
    else
        # Python フォールバック
        python3 -c "
import hashlib, base64, os
salt = os.urandom(8)
h = hashlib.sha1(b'$AUTH_PASS' + salt).digest()
entry = '{' + '$AUTH_USER' + '}:{SHA}' + base64.b64encode(h).decode()
print(entry)
" > "$HTPASSWD_FILE"
    fi
    echo "  → $HTPASSWD_FILE を生成しました"
else
    echo "[1/4] Basic認証: $HTPASSWD_FILE が既に存在 (スキップ)"
fi

# ─────────────────────────────────────────
# 2. SSL 証明書ディレクトリ
# ─────────────────────────────────────────
CERTS_DIR="nginx/certs"
mkdir -p "$CERTS_DIR"
if [ ! -f "$CERTS_DIR/fullchain.pem" ]; then
    echo "[2/4] SSL証明書"
    echo "  → $CERTS_DIR/ に証明書を配置してください"
    echo "    fullchain.pem + privkey.pem"
    echo "    (Let's Encrypt: sudo certbot certonly --standalone -d your-domain.com)"
    echo "  → 証明書なしでも HTTP で動作します (開発モード)"

    # 自己署名証明書を生成 (開発用)
    if command -v openssl &> /dev/null; then
        echo ""
        read -p "  自己署名証明書を生成しますか？ [y/N]: " GEN_CERT
        if [[ "$GEN_CERT" =~ ^[Yy]$ ]]; then
            openssl req -x509 -nodes -days 365 \
                -newkey rsa:2048 \
                -keyout "$CERTS_DIR/privkey.pem" \
                -out "$CERTS_DIR/fullchain.pem" \
                -subj "/CN=localhost" 2>/dev/null
            echo "  → 自己署名証明書を生成しました"
        fi
    fi
else
    echo "[2/4] SSL証明書: 既に配置済み (スキップ)"
fi

# ─────────────────────────────────────────
# 3. Docker Compose ビルド・起動
# ─────────────────────────────────────────
echo "[3/4] Docker Compose ビルド・起動"

if command -v docker compose &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "  ❌ Docker Compose が見つかりません。インストールしてください。"
    exit 1
fi

$COMPOSE_CMD build
$COMPOSE_CMD up -d

echo ""
echo "[4/4] 起動確認"

# MLflow の起動を待つ
echo -n "  MLflow サーバーの起動を待機中"
for i in $(seq 1 30); do
    if curl -sf http://localhost:5000/health > /dev/null 2>&1 || \
       curl -sf http://localhost:5000/api/2.0/mlflow/experiments/search > /dev/null 2>&1; then
        echo " ✅"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  セットアップ完了                               ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                  ║"
echo "║  MLflow UI:    http://localhost:5000              ║"
echo "║  外部アクセス: http://<your-ip>:80               ║"
echo "║  Keiba API:    http://localhost:8000              ║"
echo "║  外部API:      http://<your-ip>:80/keiba/        ║"
echo "║                                                  ║"
echo "║  認証: Basic認証 (nginx/htpasswd)                ║"
echo "║                                                  ║"
echo "║  管理コマンド:                                    ║"
echo "║    $COMPOSE_CMD logs -f          # ログ確認      ║"
echo "║    $COMPOSE_CMD restart          # 再起動        ║"
echo "║    $COMPOSE_CMD down             # 停止          ║"
echo "║                                                  ║"
echo "╚══════════════════════════════════════════════════╝"
