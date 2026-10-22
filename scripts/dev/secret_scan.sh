#!/usr/bin/env bash
# 公開リポジトリに秘密・内部情報を入れないための検査。push 前（.githooks/pre-push）と手動で使う。
#   bash scripts/dev/secret_scan.sh                 # origin/develop..HEAD の差分
#   bash scripts/dev/secret_scan.sh <rev範囲>        # 例: HEAD~3..HEAD
#   bash scripts/dev/secret_scan.sh --tracked       # 追跡中の全ファイル
# 検出したら非ゼロで終了。値そのものは表示しない（行番号とパターン名だけ）。
set -uo pipefail
RANGE="${1:-origin/develop..HEAD}"
declare -A PAT=(
  [slack_webhook]='hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{20,}'
  [private_key]='-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
  [jwt_or_cf_token]='eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{10,}'
  [aws_key]='AKIA[0-9A-Z]{16}'
  [slack_token]='xox[bpa]-[0-9A-Za-z-]{20,}'
  [github_token]='gh[pousr]_[A-Za-z0-9]{30,}'
  [google_api_key]='AIza[0-9A-Za-z_-]{35}'
  [secret_assignment]='(SECRET|TOKEN|PASSWORD|API_KEY)[A-Z_]*=["'"'"']?[A-Za-z0-9+/_-]{24,}'
  [private_ip_ssh_port]='ssh[^ ]* -p [0-9]{4,5} [^ ]*@[0-9]{1,3}(\.[0-9]{1,3}){3}'
)
# 許可: ダミー値・文書用 IP（RFC 5737）・テスト用
ALLOW='MASKED|REPLACE_WITH|verify-only|change-me|change_me|CHANGE|example|xxxx|XXXX|<[A-Z_]+>|203\.0\.113\.|198\.51\.100\.|192\.0\.2\.'
rc=0
if [ "$RANGE" = "--tracked" ]; then
  SRC() { git grep -InE "$1" -- . ':!*.lock' ':!docs/html/assets/*' ':!notebooks/*' ; }
else
  SRC() { git diff "$RANGE" -U0 | grep -E '^\+[^+]' | grep -nE "$1" ; }
fi
for name in "${!PAT[@]}"; do
  hits="$(SRC "${PAT[$name]}" 2>/dev/null | grep -vE "$ALLOW" | cut -d: -f1-2 | head -5 || true)"
  if [ -n "$hits" ]; then echo "検出: $name"; echo "$hits" | sed 's/^/   /'; rc=1; fi
done
[ $rc -eq 0 ] && echo "秘密らしい文字列は見つかりませんでした" || echo "→ 該当行を消す／環境変数・Secret Manager に移す。すでに公開した値は無効化(ローテーション)する"
exit $rc
