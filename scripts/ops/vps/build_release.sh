#!/usr/bin/env bash
# prod（ConoHa VPS）用の UI リリースを作る。ホスト機（メモリに余裕がある）か CI で実行する。VPS ではビルドしない（2GB で next build は不安定）。
#   bash scripts/ops/vps/build_release.sh [--commit <rev>] [--out <dir>] [--env-file <file>] [--upstream <url>]
#
# できるもの: <out>/keiba-ui-<commit8>-<日時>.tar.gz と .sha256 と .manifest.json
# - git のコミットから作る（作業ツリーの未コミット変更は入らない）。ビルドは /tmp の作業コピーで行う（9p のドライブは遅い）。
# - API の宛先はビルド時に固定: 既定 http://127.0.0.1:8080（VPS の nginx。nginx がホスト機へ中継する）。
# - stg と見られる情報を同じにするため、NEXT_PUBLIC_* は --env-file（既定 scripts/ops/vps/build.env があれば）で stg と揃える。
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"; cd "$ROOT"
REV=HEAD; OUT="$ROOT/data/local/releases"; ENVF=""; UP="http://127.0.0.1:8080"
while [ $# -gt 0 ]; do case "$1" in --commit) REV="$2"; shift 2;; --out) OUT="$2"; shift 2;; --env-file) ENVF="$2"; shift 2;; --upstream) UP="$2"; shift 2;; *) echo "不明な引数: $1" >&2; exit 1;; esac; done
[ -n "$ENVF" ] || { [ -r "$ROOT/scripts/ops/vps/build.env" ] && ENVF="$ROOT/scripts/ops/vps/build.env" || true; }
COMMIT="$(git rev-parse --verify "${REV}^{commit}")"; SHORT="${COMMIT:0:8}"; TS="$(date +%Y%m%d%H%M)"
ID="keiba-ui-${SHORT}-${TS}"
WORK="$(mktemp -d /tmp/keiba-ui-build.XXXXXX)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

echo "== 1/6 ソースを取り出す（$COMMIT）"
git archive "$COMMIT" frontend config/ports.env | tar -x -C "$WORK"
[ -d "$WORK/frontend" ] || { echo "frontend が取り出せません" >&2; exit 1; }

echo "== 2/6 依存パッケージ"
if [ -d "$ROOT/frontend/node_modules" ] && cmp -s "$ROOT/frontend/package-lock.json" "$WORK/frontend/package-lock.json"; then
  # ホスト機の frontend/node_modules は /tmp への symlink（9p の遅さ回避）。symlink のまま写すと standalone に実体が入らず VPS で 'next' が見つからない。末尾 /. で実体の中身を写す（-L は .bin の相対 symlink を壊すので使わない）
  mkdir -p "$WORK/frontend/node_modules"; cp -a "$ROOT/frontend/node_modules/." "$WORK/frontend/node_modules/"; echo "  既存の node_modules を再利用（package-lock.json が同一）"
else
  ( cd "$WORK/frontend" && npm ci --no-audit --no-fund ); fi

echo "== 3/6 version.json（画面から /version.json で見える。秘密なし）"
mkdir -p "$WORK/frontend/public"
printf '{"release":"%s","commit":"%s","short":"%s","built_at":"%s","env":"prod"}\n' "$ID" "$COMMIT" "$SHORT" "$(date -Is)" > "$WORK/frontend/public/version.json"

echo "== 4/6 ビルド（負荷が高いときは待つ）"
for i in $(seq 1 60); do
  l="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"; awk -v l="$l" 'BEGIN{exit !(l>14)}' && { echo "  load=$l が高いので 30 秒待つ ($i/60)"; sleep 30; } || break
done
(
  cd "$WORK/frontend"
  [ -n "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
  nice -n 19 env KEIBA_ENV=prod NODE_ENV=production NEXT_OUTPUT=standalone NEXT_BUILD_CPUS=2 NEXT_TELEMETRY_DISABLED=1 \
    KEIBA_API_URL="$UP" KEIBA_FASTAPI_URL="$UP" NEXT_PUBLIC_ENV_LABEL="" \
    npx next build
)

echo "== 5/6 パッケージ化"
DIST="$WORK/frontend/.next-prod"
[ -f "$DIST/standalone/server.js" ] || { echo "standalone の出力がありません（next.config.js の NEXT_OUTPUT を確認）" >&2; exit 1; }
BUILD_ID="$(cat "$DIST/BUILD_ID")"
PKG="$WORK/pkg"; mkdir -p "$PKG"
cp -a "$DIST/standalone/." "$PKG/"
[ ! -L "$PKG/node_modules" ] && [ -f "$PKG/node_modules/next/package.json" ] || { echo "standalone の node_modules に next の実体がありません（symlink のままか、トレース漏れ）。VPS では起動できないので中止します" >&2; exit 1; }
mkdir -p "$PKG/.next-prod"; cp -a "$DIST/static" "$PKG/.next-prod/static"
mkdir -p "$PKG/public"; [ -d "$WORK/frontend/public" ] && cp -a "$WORK/frontend/public/." "$PKG/public/"
python3 - "$PKG/public/version.json" "$BUILD_ID" <<'PY'
import json,sys
p,b=sys.argv[1],sys.argv[2]
d=json.load(open(p)); d["build_id"]=b; json.dump(d,open(p,"w"),ensure_ascii=False); open(p,"a").write("\n")
PY
tar -czf "$OUT/$ID.tar.gz" -C "$PKG" .
( cd "$OUT" && sha256sum "$ID.tar.gz" > "$ID.tar.gz.sha256" )
python3 - "$OUT/$ID.manifest.json" "$ID" "$COMMIT" "$BUILD_ID" "$UP" "$(node -v)" <<'PY'
import json,sys,datetime
p,i,c,b,u,n=sys.argv[1:7]
json.dump({"release":i,"commit":c,"build_id":b,"upstream":u,"node_build":n,"built_at":datetime.datetime.now().astimezone().isoformat()},open(p,"w"),ensure_ascii=False,indent=1)
PY

echo "== 6/6 できあがり"
ls -lh "$OUT/$ID".* | awk '{print "  "$5"  "$9}'
echo "$OUT/$ID.tar.gz"
