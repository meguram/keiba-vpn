#!/usr/bin/env bash
# ホスト（keiba-remote）の学習済みモデルを GCS(models/<component>/)へ公開する。ホストの git は古く・未コミットが多いので pull せず、
# origin/develop のコードだけを一時ディレクトリへ展開して model_publish を動かす（ホストの作業ツリーには触れない）。
#
#   手元から: ssh keiba-remote 'bash -s -- bundle_v6e stage_v6e rb_state_v6 tl_reference_v5' < scripts/gcp/publish_models_from_host.sh
#   ホスト上で（このファイルを置いた場合）: bash publish_models_from_host.sh bundle_v6e …
#   引数なし: 何も公開せず、公開できるコンポーネントと、ホスト上の実体のサイズだけを表示する（既定は確認のみ）
#
# 公開は latest.json を最後に書く（書いた瞬間が昇格）。同じ内容なら再アップロードしない。component の一覧は config/gcp_job_map.json の models。
# 前提: ホストの .env に GCS_BUCKET と書き込み可能な資格情報がある（切り離し後は publish 専用の別 SA に替える: decoupling_plan.md 9.4）。
set -euo pipefail
[ "${1:-}" != "--" ] || shift
REPO="${KEIBA_HOST_REPO:-/home/jovyan/work/keiba-vpn}"
PY="${PYTHON:-/opt/conda/bin/python3}"
NEW="$(mktemp -d /tmp/keiba_publish.XXXXXX)"
trap 'rm -rf "$NEW"' EXIT
cd "$REPO"
sudo -n git fetch -q origin
sudo -n git archive origin/develop src config scripts requirements.txt | tar -x -C "$NEW"
[ ! -f "$REPO/.env" ] || ln -sf "$REPO/.env" "$NEW/.env"

# job_map の models から、ホスト上の実体のパスを引く（local_path はリポジトリ相対）
list_models() {
  (cd "$NEW" && "$PY" - <<'EOF'
import json
for k, v in json.load(open("config/gcp_job_map.json"))["models"].items():
    print(k, v["local_path"])
EOF
  )
}

if [ "$#" -eq 0 ]; then
  echo "公開できるコンポーネント（ホスト上の実体とサイズ）:"
  list_models | while read -r c p; do
    if [ -e "$REPO/$p" ]; then printf '  %-18s %8s  %s\n' "$c" "$(du -sh "$REPO/$p" | cut -f1)" "$p"; else printf '  %-18s %8s  %s\n' "$c" "(無し)" "$p"; fi
  done
  echo "公開するには、コンポーネント名を引数に渡してください。"
  exit 0
fi

for c in "$@"; do
  p="$(list_models | awk -v c="$c" '$1==c{print $2}')"
  [ -n "$p" ] || { echo "未知のコンポーネント: $c（config/gcp_job_map.json の models に無い）" >&2; exit 2; }
  [ -e "$REPO/$p" ] || { echo "ホスト上に実体がありません: $REPO/$p" >&2; exit 2; }
  echo "== 公開: $c  ($p)"
  (cd "$NEW" && PYTHONPATH=. "$PY" -m src.ops.model_publish publish --component "$c" --from "$REPO/$p")
done
