#!/usr/bin/env bash
# Secret Manager に秘密を登録する。値は標準入力（非表示）から読み、コマンド履歴・画面・引数に残さない。
#
#   bash scripts/gcp/put_secret.sh SLACK_WEBHOOK_URL      # 値を貼り付けて Enter（画面には出ない）
#   printf %s "$VALUE" | bash scripts/gcp/put_secret.sh NAME   # パイプでも可（$VALUE を履歴に残さない場合のみ）
#
# 登録後、下の表のサービスアカウントにだけ読み取り権限（secretAccessor）を付ける。
# 古い版は --retire で無効化する（有効な版が 6 個を超えると月 約 9 円/版 かかる）。
set -euo pipefail

NAME="${1:-}"
[ -n "$NAME" ] || { echo "使い方: bash put_secret.sh <SECRET名> [--retire]" >&2; exit 2; }
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || { echo "プロジェクト未設定" >&2; exit 1; }

# 秘密名 → 読めるサービスアカウント（最小権限）
case "$NAME" in
  SLACK_WEBHOOK_URL|SLACK_WEBHOOK_URL_PROD|SLACK_WEBHOOK_URL_STG)    READERS="keiba-api keiba-jobs" ;;
  KEIBA_PROXY_KEY)      READERS="keiba-api keiba-deployer" ;;      # deployer: CI のスモークが鍵つきで API を叩く
  KEIBA_HEARTBEAT_KEY)  READERS="keiba-api" ;;
  APP_SECRET_KEY|JWT_SECRET|DEV_PASSWORD) READERS="keiba-api" ;;
  DEV_SECRET_KEY)       READERS="keiba-api keiba-jobs" ;;      # jobs: 共通ログインの月次更新(shared-login)がパスワードの元として使う
  KEIBA_SHARED_LOGIN_ID) READERS="keiba-jobs" ;;                 # 共通ログインの ID(shared-login)
  R2_CREDENTIALS)       READERS="keiba-jobs" ;;
  *) echo "未登録の名前です: $NAME（scripts/gcp/put_secret.sh の表と docs/operations/gcp_setup_runbook.md 5 章に足してから）" >&2; exit 2 ;;
esac

if [ "${2:-}" = "--retire" ]; then
  echo "有効な版:"; gcloud secrets versions list "$NAME" --project "$PROJECT" --filter="state=ENABLED" --format="table(name,createTime)"
  echo "最新以外を無効化します"
  gcloud secrets versions list "$NAME" --project "$PROJECT" --filter="state=ENABLED" --sort-by="~createTime" --format="value(name)" \
    | tail -n +2 | while read -r v; do gcloud secrets versions disable "$v" --secret "$NAME" --project "$PROJECT" --quiet; done
  exit 0
fi

if [ -t 0 ]; then
  read -r -s -p "$NAME の値（入力は表示されません）: " VALUE; echo
else
  VALUE="$(cat)"
fi
[ -n "$VALUE" ] || { echo "値が空です" >&2; exit 1; }

if ! gcloud secrets describe "$NAME" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud secrets create "$NAME" --replication-policy=user-managed --locations=asia-northeast1 --project "$PROJECT"
fi
printf %s "$VALUE" | gcloud secrets versions add "$NAME" --data-file=- --project "$PROJECT" >/dev/null
unset VALUE
for sa in $READERS; do
  gcloud secrets add-iam-policy-binding "$NAME" --member "serviceAccount:$sa@$PROJECT.iam.gserviceaccount.com" \
    --role roles/secretmanager.secretAccessor --project "$PROJECT" --quiet >/dev/null
done
echo "登録しました: $NAME（長さ・値は表示しません）。読めるのは: $READERS"
gcloud secrets versions list "$NAME" --project "$PROJECT" --format="table(name,state,createTime)" --limit 3
