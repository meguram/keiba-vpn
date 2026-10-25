#!/usr/bin/env bash
# IAM 条件（--condition-from-file 用の YAML）を標準出力に出す。バケットの接頭辞 1 つだけに、読み書き(get/create/delete)と一覧(list)を許す式。
#   読み書きは resource.name、一覧は objectListPrefix で、条件が別々に要る（片方だけだと、一覧が 403 になる／接頭辞の外が見える）。
#   BUCKET=<バケット> bash scripts/gcp/make_iam_condition.sh <接頭辞（末尾 /）> <タイトル> > /tmp/cond.yaml
# 使い方の例: gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member serviceAccount:… --role roles/storage.objectUser --condition-from-file /tmp/cond.yaml
set -euo pipefail
BUCKET="${BUCKET:?BUCKET（バケット名）を環境変数で渡してください}"; PREFIX="${1:?接頭辞（例 ops/）}"; TITLE="${2:?タイトル}"
case "$PREFIX" in */) ;; *) echo "接頭辞は / で終えてください: $PREFIX" >&2; exit 2 ;; esac
printf 'expression: resource.name.startsWith("projects/_/buckets/%s/objects/%s") || (resource.type == "storage.googleapis.com/Bucket" && api.getAttribute("storage.googleapis.com/objectListPrefix", "").startsWith("%s"))\n' "$BUCKET" "$PREFIX" "$PREFIX"
printf 'title: %s\n' "$TITLE"
