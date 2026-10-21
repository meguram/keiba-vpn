#!/usr/bin/env bash
# （任意・既定では使わない）承認フラグがあるときだけ、メンテ窓（JST 03:00–04:00）の中で prod に昇格する。cron への登録はユーザーが決める。
#   承認: echo '{"commit":"<hash>","expires_at":"2026-10-30T00:00:00+09:00"}' > data/local/deploy/promote_approved.json
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"; cd "$ROOT"
F=data/local/deploy/promote_approved.json
[ -f "$F" ] || exit 0
H="$(TZ=Asia/Tokyo date +%H)"; [ "$H" = 03 ] || { echo "メンテ窓の外（JST ${H} 時）のため実行しません"; exit 0; }
read -r COMMIT OK < <(python3 - "$F" <<'PY'
import json,sys,datetime
d=json.load(open(sys.argv[1])); e=datetime.datetime.fromisoformat(d["expires_at"])
print(d["commit"], "1" if datetime.datetime.now(e.tzinfo)<e else "0")
PY
)
[ "$OK" = 1 ] || { echo "承認の期限切れ"; mv "$F" "$F.expired"; exit 0; }
bash scripts/ops/vps/promote.sh --commit "$COMMIT" --yes && mv "$F" "data/local/deploy/promote_done_$(date +%Y%m%d%H%M).json"
