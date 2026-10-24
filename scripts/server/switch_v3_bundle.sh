#!/usr/bin/env bash
# 推論に使うアンサンブルのバンドル（KEIBA_V3_BUNDLE）と前段（KEIBA_V3_STAGE）を .env で切り替える。
# 使い方: bash scripts/server/switch_v3_bundle.sh <bundle_dir> [<stage_dir>]     （引数なしで現在の設定を表示）
# - 切り替え前に .env を env.before.<日時> に退避する。実行中のランナーは止めない（.env は次の起動で読まれる）。
# - 元に戻すには: cp env.before.<日時> .env
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/.env"
if [ "$#" -eq 0 ]; then
  grep -n "^KEIBA_V3_" "$ENV_FILE" || echo "(KEIBA_V3_* の設定なし)"
  exit 0
fi
BUNDLE="$1"
STAGE="${2:-}"
[ -f "$BUNDLE/bundle.json" ] || { echo "bundle.json がありません: $BUNDLE" >&2; exit 2; }
[ -z "$STAGE" ] || [ -f "$STAGE/stage_meta.json" ] || { echo "stage_meta.json がありません: $STAGE" >&2; exit 2; }
TS="$(date +%Y%m%d_%H%M%S)"
cp -a "$ENV_FILE" "$ROOT/env.before.$TS"
set_kv() {  # key value
  if grep -q "^$1=" "$ENV_FILE"; then sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"; else printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"; fi
}
set_kv KEIBA_V3_BUNDLE "$BUNDLE"
[ -z "$STAGE" ] || set_kv KEIBA_V3_STAGE "$STAGE"
echo "退避: $ROOT/env.before.$TS"
grep -n "^KEIBA_V3_" "$ENV_FILE"
