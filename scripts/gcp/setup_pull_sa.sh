#!/usr/bin/env bash
# VPS が GCS から画面のリリースを取りに行く（pull 型デプロイ。pull_release.py）ための、専用サービスアカウントを用意する。Cloud Shell で実行。
# 既定は『コマンドを表示するだけ』（何も変更しない）。実行は --apply。冪等（すでにあれば飛ばす）。
#
#   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_pull_sa.sh                # 表示だけ
#   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_pull_sa.sh --apply        # 実行
#   GCS_BUCKET=<バケット名> bash scripts/gcp/setup_pull_sa.sh --apply --make-key /tmp/keiba-pull-sa.json   # 鍵(JSON)も作る（VPS は GCP の外にあり、鍵が要る）
#
# 作るもの（最小権限）:
#   keiba-pull SA … バケットの ops/releases/ の読み取り（objectViewer）と、ops/releases/<種別>/applied/vps.json の書き込みだけ（IAM Conditions）。
#   他のオブジェクト（予測・データ・モデル・秘密）には一切触れない。VPS の画面が置き換えられるのは、ここで読めるリリースだけ。
# 鍵は --make-key のときだけ作る。画面には出さず、VPS へ安全に渡す（docs/operations/github_promote.md の 8 章）。渡し終えたら Cloud Shell のファイルは消す。
set -euo pipefail

APPLY=0; KEYFILE=""
while [ $# -gt 0 ]; do case "$1" in
  --apply) APPLY=1; shift;; --dry-run) APPLY=0; shift;; --make-key) KEYFILE="${2:?鍵の出力先}"; shift 2;;
  *) echo "不明な引数: $1" >&2; exit 2;; esac; done

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ] || { echo "PROJECT_ID が未設定です（gcloud config set project <ID>）" >&2; exit 1; }
BUCKET="${GCS_BUCKET:-}"; [ -n "$BUCKET" ] || { echo "GCS_BUCKET を指定してください" >&2; exit 1; }
SA_ID="${PULL_SA:-keiba-pull}"; SA_EMAIL="$SA_ID@$PROJECT_ID.iam.gserviceaccount.com"
PREFIX="ops/releases/"

run() { if [ "$APPLY" = 1 ]; then "$@"; else printf '[dry] '; printf '%q ' "$@"; echo; fi; }
echo "== project=$PROJECT_ID bucket=$BUCKET sa=$SA_EMAIL  ($([ "$APPLY" = 1 ] && echo 実行 || echo 表示のみ))"

echo "== 1/3 サービスアカウント"
if [ "$APPLY" = 1 ] && gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "  既にあります: $SA_EMAIL"
else
  run gcloud iam service-accounts create "$SA_ID" --display-name "keiba-pull (VPS の画面の取り込み。GCS の ops/releases/ だけ)" --project "$PROJECT_ID"
fi

echo "== 2/3 ops/releases/ の読み取り（objectViewer。この接頭辞だけ）"
READ_EXPR="resource.name.startsWith(\"projects/_/buckets/$BUCKET/objects/$PREFIX\")"
run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectViewer" \
  --condition="expression=$READ_EXPR,title=pull-read-releases,description=VPS の取り込みは ops/releases/ だけ読める"

echo "== 3/3 取り込み結果の報告（applied/vps.json）の書き込み。これ以外には書けない"
WRITE_EXPR="resource.name.startsWith(\"projects/_/buckets/$BUCKET/objects/$PREFIX\") && resource.name.endsWith(\"/applied/vps.json\")"
run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectUser" \
  --condition="expression=$WRITE_EXPR,title=pull-write-applied,description=取り込み結果の報告 applied/vps.json だけ書ける"

if [ -n "$KEYFILE" ]; then
  echo "== 鍵（JSON）を作る: $KEYFILE（権限 600。VPS へ渡したら消す）"
  if [ "$APPLY" = 1 ]; then ( umask 077; gcloud iam service-accounts keys create "$KEYFILE" --iam-account "$SA_EMAIL" --project "$PROJECT_ID" ); fi
  [ "$APPLY" = 1 ] || echo "[dry] gcloud iam service-accounts keys create $KEYFILE --iam-account $SA_EMAIL"
fi
echo
echo "VPS 側（root）: /etc/keiba-pull/pull.env に GCS_BUCKET=$BUCKET と GOOGLE_APPLICATION_CREDENTIALS=/etc/keiba-pull/sa.json、鍵を /etc/keiba-pull/sa.json（600）に置く。"
[ "$APPLY" = 1 ] || echo "→ 内容を確認して、実行するときは --apply を付けてください"
