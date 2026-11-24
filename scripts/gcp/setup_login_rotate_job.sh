#!/usr/bin/env bash
# 共有ログインの毎月の切り替え（Cloud Scheduler keiba-login-rotate → Cloud Run Job keiba-login-rotate）を作る。冪等。
# 既定は表示だけ（流すコマンドを出す）。--apply で実行する。秘密の値は扱わない（ペッパーの登録は docs の手順・put_secret.sh）。
#
#   IMAGE=<jobs の版> GCS_BUCKET=<バケット> OWNER=user:<オーナーのアカウント> DEPLOYER_SA=<CI の配備用 SA のメール> \
#     bash scripts/gcp/setup_login_rotate_job.sh [--apply]
#
# 作るもの（手順と順番: docs/operations/shared_login_rotation.md）:
#   1. SA keiba-login-rotator。名簿・目録・結果のファイルだけに書ける GCS の権限（IAM の条件つき）
#   2. 秘密 KEIBA_SHARED_LOGIN_PW（rotator は読み出し・版の追加と destroy。オーナーは読み出しだけ）と、ほかの秘密の読み出し
#   3. Job keiba-login-rotate（イメージは IMAGE。live-login-rotate のタグを付ける）
#   4. Scheduler keiba-login-rotate（0 6 1,25-31 * * Asia/Tokyo）。**PAUSED で作る**（旧経路を止めて確かめた後に resume）
# 費用（最悪）: Scheduler 15 円・秘密の版 3 つ（パスワード 2・ペッパー 1）で 27 円・Jobs 約 4 円・GCS 1 円未満 ≒ 月 47 円。
set -euo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-asia-northeast1}"
AR_REPO="${AR_REPO:-keiba-vpn}"
NAME=keiba-login-rotate
SA_NAME=keiba-login-rotator
: "${GCS_BUCKET:?GCS_BUCKET（名簿のバケット）を指定してください}"
: "${OWNER:?OWNER（下書きを取り出す人。例 user:someone@example.com）を指定してください}"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || { echo "プロジェクト未設定" >&2; exit 1; }
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
SCHED_SA="${SCHEDULER_SA_EMAIL:-keiba-scheduler@${PROJECT}.iam.gserviceaccount.com}"
REG="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}"
: "${IMAGE:?IMAGE（login_rotate の入った jobs の版。例 ${REG}/jobs:<最新の promote の短い SHA>）を指定してください。live-jobs は ENABLE_JOBS_DEPLOY のときしか進まないので使わない}"

run() { if [ "$APPLY" = 1 ]; then echo "+ $*"; "$@"; else printf '+'; printf ' %q' "$@"; echo; fi; }
exists() { [ "$APPLY" = 1 ] && "$@" >/dev/null 2>&1; }

echo "== project=$PROJECT region=$REGION bucket=$GCS_BUCKET image=$IMAGE（$([ "$APPLY" = 1 ] && echo 実行 || echo 表示だけ)）"

# 1) SA と GCS の権限（名前を絞る。名簿・名簿の目録・結果のファイルだけ）
exists gcloud iam service-accounts describe "$SA" --project "$PROJECT" || run gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" --display-name "shared-login monthly rotation"
OBJ="projects/_/buckets/${GCS_BUCKET}/objects"
# 名簿の GCS のキーは src/ops/gcs_mirror.py の定義から取る（書き写さない）
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.."
MEMBERS_KEY="$(PYTHONPATH=. "${PYTHON:-python3}" -c 'from src.ops import gcs_mirror as g; print(f"{g.PREFIX}/{g.MEMBERS_REL}")')"
INDEX_PREFIX="$(PYTHONPATH=. "${PYTHON:-python3}" -c 'from src.ops import gcs_mirror as g; print(g.index_key("auth").rsplit(".recent.json", 1)[0])')"
COND="title=login-rotate,expression=resource.name == \"${OBJ}/${MEMBERS_KEY}\" || resource.name.startsWith(\"${OBJ}/${INDEX_PREFIX}.\") || resource.name.startsWith(\"${OBJ}/ops/login_rotate/\")"
run gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" --member "serviceAccount:$SA" --role roles/storage.objectUser --condition "$COND"

# 2) 秘密
exists gcloud secrets describe KEIBA_SHARED_LOGIN_PW --project "$PROJECT" || \
  run gcloud secrets create KEIBA_SHARED_LOGIN_PW --replication-policy=user-managed --locations="$REGION" --project "$PROJECT"
for role in roles/secretmanager.secretAccessor roles/secretmanager.secretVersionManager; do
  run gcloud secrets add-iam-policy-binding KEIBA_SHARED_LOGIN_PW --member "serviceAccount:$SA" --role "$role" --project "$PROJECT" --quiet
done
run gcloud secrets add-iam-policy-binding KEIBA_SHARED_LOGIN_PW --member "$OWNER" --role roles/secretmanager.secretAccessor --project "$PROJECT" --quiet
for s in KEIBA_SHARED_LOGIN_ID SLACK_WEBHOOK_URL KEIBA_MEMBERS_PEPPER; do
  if [ "$APPLY" = 1 ] && ! gcloud secrets describe "$s" --project "$PROJECT" >/dev/null 2>&1; then
    echo "秘密 $s がありません。先に登録してください（KEIBA_MEMBERS_PEPPER は docs の手順 2）" >&2; exit 1
  fi
  run gcloud secrets add-iam-policy-binding "$s" --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor --project "$PROJECT" --quiet
done

# 3) Job（並走の包みを通さない専用の Job。1 タスク・失敗時は 2 回まで再試行＝処理は冪等）
run gcloud run jobs deploy "$NAME" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --memory 512Mi --cpu 1 --task-timeout 300s --max-retries 2 --parallelism 1 --tasks 1 \
  --service-account "$SA" \
  --set-env-vars "KEIBA_ENV=prod,KEIBA_RUN_LOCATION=gcp,KEIBA_MEMBERS_STORE=gcs,GCS_BUCKET=${GCS_BUCKET},GCS_PROJECT_ID=${PROJECT},TZ=Asia/Tokyo" \
  --set-secrets "KEIBA_SHARED_LOGIN_ID=KEIBA_SHARED_LOGIN_ID:latest,SLACK_WEBHOOK_URL=SLACK_WEBHOOK_URL:latest,KEIBA_MEMBERS_PEPPER=KEIBA_MEMBERS_PEPPER:latest" \
  --command python3 --args=-m,src.scripts.ops.login_rotate,run
if [ "$APPLY" = 1 ]; then
  d="$(gcloud run jobs describe "$NAME" --project "$PROJECT" --region "$REGION" --format='value(spec.template.spec.template.spec.containers[0].image)' 2>/dev/null || true)"
  run gcloud artifacts docker tags add "${d:-$IMAGE}" "${REG}/jobs:live-login-rotate" --quiet   # 自動掃除から守る（CLAUDE.md 再発防止 4）
else
  run gcloud artifacts docker tags add "$IMAGE" "${REG}/jobs:live-login-rotate" --quiet
fi
run gcloud run jobs add-iam-policy-binding "$NAME" --project "$PROJECT" --region "$REGION" --member "serviceAccount:$SCHED_SA" --role roles/run.invoker --quiet
if [ -n "${DEPLOYER_SA:-}" ]; then   # CI（promote の jobs の段）がイメージを進めるため
  run gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT" --member "serviceAccount:${DEPLOYER_SA}" --role roles/iam.serviceAccountUser --quiet
else
  echo "注意: DEPLOYER_SA が未指定です。CI が keiba-login-rotate のイメージを進められません（gh variable get WIF_SERVICE_ACCOUNT の値を渡す）"
fi

# 4) Scheduler（keiba-g- で始めない＝cost_guard に止められない。PAUSED で作る）
URI="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${NAME}:run"
if exists gcloud scheduler jobs describe "$NAME" --project "$PROJECT" --location "$REGION"; then
  echo "Scheduler $NAME はあります（状態は変えません）"
else
  run gcloud scheduler jobs create http "$NAME" --project "$PROJECT" --location "$REGION" \
    --schedule "0 6 1,25-31 * *" --time-zone Asia/Tokyo --uri "$URI" --http-method POST \
    --headers Content-Type=application/json --message-body '{}' \
    --oauth-service-account-email "$SCHED_SA" --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
    --attempt-deadline 180s --description "共有ログインの毎月の切り替え（準備・下書き・切り替え）"
  run gcloud scheduler jobs pause "$NAME" --project "$PROJECT" --location "$REGION"
fi
echo "済み。旧経路を止めて確かめた後に: gcloud scheduler jobs resume $NAME --location $REGION --project $PROJECT"
