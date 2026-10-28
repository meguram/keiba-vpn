#!/usr/bin/env bash
# =============================================================================
# CI（GitHub Actions）から呼ぶ、GCP への反映の手順。ワークフローは薄く保ち、手順はここに置く（手元で --dry-run して確かめられる）。
#   stg ブランチへの push  → .github/workflows/deploy-stg.yml   （--target stg）
#   master ブランチへの push → .github/workflows/promote.yml     （--target prod。承認の関門の後）
#
#   bash scripts/ci/deploy_gcp.sh <step> --target stg|prod [--dry-run]
#
#   build-api-image / check-jobs-image   stg 用に build-images を 2 つに分けたもの（API の push と、Jobs の読み込み確認(push なし)を並走させる）
#   build-images   API と Jobs のイメージを作って Artifact Registry へ push する。Jobs のイメージは、全ジョブの実行モジュールを読み込めるかまで確かめる
#   api-revision   通信 0% のタグ付きリビジョンを作る（stg は固定のタグ rstg、prod は rc<コミット8桁>）。タグ付き URL を GITHUB_OUTPUT に書く
#   smoke          タグ付き URL に対して、鍵なし 401・鍵あり 200 などを確かめる（src.scripts.ops.cloudrun_smoke）
#   promote        【prod のみ】通信を新リビジョンに 100% 切り替える。戻すのは rollback_cloud_run_api.sh
#   jobs           【prod のみ・ENABLE_JOBS_DEPLOY=true のときだけ】キュー・Run Job・Scheduler を更新する。JOBS_PARALLEL=1（既定）は並走(影モード)
#
# 必須の環境変数: PROJECT_ID, GCS_BUCKET（--dry-run のときは省略可）。任意: REGION(asia-northeast1) SERVICE(keiba-api) AR_REPO(keiba-vpn)
#   SMOKE_RACE_ID（smoke で使うレース ID） ENABLE_JOBS_DEPLOY JOBS_PARALLEL FETCH_SHADOW（1 で取得の影モードの予約・集計） GITHUB_SHA GITHUB_OUTPUT
# 鍵ファイルは使わない（認証は google-github-actions/auth の Workload Identity Federation が済ませてある前提）。
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

STEP="${1:-}"; shift || true
TARGET=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "不明な引数: $1" >&2; exit 2 ;;
  esac
done
[ -n "$STEP" ] || { sed -n '2,19p' "$0"; exit 2; }
case "$TARGET" in stg|prod) ;; *) echo "--target は stg か prod" >&2; exit 2 ;; esac

PROJECT_ID="${PROJECT_ID:-}"; GCS_BUCKET="${GCS_BUCKET:-}"
if [ "$DRY" = 1 ]; then PROJECT_ID="${PROJECT_ID:-PROJECT_ID}"; GCS_BUCKET="${GCS_BUCKET:-GCS_BUCKET}"; fi
[ -n "$PROJECT_ID" ] && [ -n "$GCS_BUCKET" ] || { echo "PROJECT_ID と GCS_BUCKET が必要です" >&2; exit 2; }
REGION="${REGION:-asia-northeast1}"; SERVICE="${SERVICE:-keiba-api}"; AR_REPO="${AR_REPO:-keiba-vpn}"
SHA="${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo 0000000000000000)}"; SHORT="${SHA:0:8}"
REG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
API_IMAGE="${REG}/keiba-api:${SHORT}"; JOBS_IMAGE="${REG}/jobs:${SHORT}"
if [ "$TARGET" = stg ]; then REV_TAG="stg"; else REV_TAG="c${SHORT}"; fi
export PROJECT_ID GCS_BUCKET REGION SERVICE AR_REPO

run() {  # 表示して、--dry-run でなければ実行。失敗したら、出力の末尾を GitHub の注釈にも出す（ログの閲覧に管理者権限が要らないように）
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$DRY" != 1 ] || return 0
  local log rc=0 msg; log="$(mktemp)"
  # `cmd | tee || true` だと失敗時に || true が走って PIPESTATUS が true の結果に上書きされ、失敗が成功になる（2026-10-26 に判明）。set +e で囲み、パイプの直後に取る
  set +e
  "$@" 2>&1 | tee "$log"
  rc="${PIPESTATUS[0]}"
  set -e
  if [ "$rc" -ne 0 ]; then
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      msg="$(tail -n 25 "$log" | cut -c1-300 | sed 's/%/%25/g; s/\r//g' | awk '{printf "%s%%0A", $0}')"
      echo "::error title=失敗（終了コード ${rc}）: ${1:-} ${2:-} ${3:-}::${msg:0:3500}"
    fi
    rm -f "$log"; exit "$rc"
  fi
  rm -f "$log"
}

fetch_fitted() {  # 当て込んだ係数（git に無い。config/fitted_artifacts.txt）のうち config/ 配下を、GCS(models/fitted_artifacts/)から取ってビルドの前に ./config へ重ねる
  if [ "$DRY" = 1 ]; then echo "+ （GCS models/fitted_artifacts/ の config/ 配下を検証つきで取得して ./config に重ねる。git 管理のファイルは上書きしない）"; return 0; fi
  local d out src
  d="$(mktemp -d)"
  if out="$(PYTHONPATH=. python3 -m src.ops.model_publish fetch --component fitted_artifacts --dest "$d" --only-prefix config/ 2>&1)"; then
    src="$(printf '%s\n' "$out" | grep '^{' | tail -n1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["path"])')"
    cp -rn "$src/config/." config/
    echo "係数を取り込みました: $(find "$src/config" -type f | wc -l) ファイル"
  else
    printf '%s\n' "$out" | tail -n 8
    local why; why="$(printf '%s\n' "$out" | grep -v -i 'warn' | tail -n 4 | cut -c1-240 | sed 's/%/%25/g; s/\r//g' | awk '{printf "%s%%0A", $0}')"
    if [ "$TARGET" = prod ]; then
      echo "::error title=係数の取得に失敗（prod は中止）::models/fitted_artifacts/ の config/ を取れませんでした。ホストで fitted_artifacts を publish し、CI の SA に GCS の読み取り権限があるか確認してください%0A${why:0:1500}"
      exit 1
    fi
    echo "::warning title=係数の取得に失敗（stg は続行）::係数なしのイメージになります（推奨馬券の新方式などが縮退）%0A${why:0:1500}"
  fi
}

case "$STEP" in
  build-api-image)   # stg 用: API のイメージだけ（Jobs は stg で更新しないので、待たずに check-jobs-image を別 job で並走させる）
    run gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    run gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" --project "$PROJECT_ID" --format="value(name)"
    fetch_fitted
    run env DOCKER_BUILDKIT=1 docker build --progress=plain -f deploy/docker/Dockerfile.cloudrun -t "$API_IMAGE" .
    run docker push "$API_IMAGE"
    ;;
  check-jobs-image)  # stg 用: Jobs のイメージを作って全ジョブのモジュールを読み込めるかだけ確かめる（push しない。GCP に接続しない）
    run env DOCKER_BUILDKIT=1 docker build --progress=plain -f deploy/docker/Dockerfile.jobs -t "$JOBS_IMAGE" .
    run docker run --rm -e KEIBA_ENV=prod -e KEIBA_RUN_LOCATION=gcp "$JOBS_IMAGE" -c \
      "import importlib, json; jobs=json.load(open('config/gcp_job_map.json'))['jobs']; mods=sorted({s['argv'][1] for j in jobs if j['destination']=='gcp' for s in j.get('steps',[]) if s.get('argv',[''])[0]=='-m'}); [importlib.import_module(m) for m in mods]; print('imported', len(mods), 'modules')"
    ;;
  build-images)
    run gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    # リポジトリは Cloud Shell での初回デプロイで作ってある（CI の SA には作成権限を付けない）
    run gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" --project "$PROJECT_ID" --format="value(name)"
    fetch_fitted
    run env DOCKER_BUILDKIT=1 docker build --progress=plain -f deploy/docker/Dockerfile.cloudrun -t "$API_IMAGE" .
    run docker push "$API_IMAGE"
    run env DOCKER_BUILDKIT=1 docker build --progress=plain -f deploy/docker/Dockerfile.jobs -t "$JOBS_IMAGE" .
    # イメージの中で、gcp 向けの全ジョブの実行モジュールを読み込めるか（足りないライブラリはここで落ちる）
    run docker run --rm -e KEIBA_ENV=prod -e KEIBA_RUN_LOCATION=gcp "$JOBS_IMAGE" -c \
      "import importlib, json; jobs=json.load(open('config/gcp_job_map.json'))['jobs']; mods=sorted({s['argv'][1] for j in jobs if j['destination']=='gcp' for s in j.get('steps',[]) if s.get('argv',[''])[0]=='-m'}); [importlib.import_module(m) for m in mods]; print('imported', len(mods), 'modules')"
    run docker push "$JOBS_IMAGE"
    ;;
  api-revision)
    # stg のリビジョンにだけ、テストユーザー(test/test。会員扱い)を有効にする。prod のリビジョンには付けない
    # CPU の割り当て: stg・prod ともリクエスト課金（アイドル中は課金されない。stg で速度・データの鮮度・課金を確かめ済み、2026-10-28）。
    #   prod だけ常時割り当てに戻すときは GitHub Variables の PROD_CPU_ALLOC=always（docs/operations/cost_recalc_20261028.md）
    run env TAG="$REV_TAG" IMAGE="$API_IMAGE" API_TEST_USER="$([ "$TARGET" = stg ] && echo 1 || echo 0)" \
      CPU_ALLOC="$([ "$TARGET" = stg ] && echo request || echo "${PROD_CPU_ALLOC:-request}")" bash scripts/gcp/deploy_cloud_run_api.sh deploy --apply
    if [ "$DRY" = 1 ]; then
      echo "+ （タグ付き URL を取得して GITHUB_OUTPUT の tag_url に書く: タグ r${REV_TAG}）"
    else
      url="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format=json \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t='r${REV_TAG}'; print(next((x['url'] for x in d['status'].get('traffic',[]) if x.get('tag')==t and x.get('url')), d['status']['url']))")"
      echo "タグ付き URL: $url"
      [ -z "${GITHUB_OUTPUT:-}" ] || echo "tag_url=$url" >> "$GITHUB_OUTPUT"
    fi
    ;;
  smoke)
    [ -n "${SMOKE_BASE:-}" ] || { [ "$DRY" = 1 ] && SMOKE_BASE="https://TAG_URL" || { echo "SMOKE_BASE（タグ付き URL）が必要です" >&2; exit 2; }; }
    [ -n "${SMOKE_RACE_ID:-}" ] || { [ "$DRY" = 1 ] && SMOKE_RACE_ID="RACE_ID" || { echo "SMOKE_RACE_ID（リポジトリの Variables）が未設定です。通信は切り替えません" >&2; exit 2; }; }
    if [ "$DRY" = 1 ]; then
      echo "+ KEIBA_PROXY_KEY=\$(gcloud secrets versions access latest --secret KEIBA_PROXY_KEY)   # 値は表示しない・マスクする"
    else
      KEIBA_PROXY_KEY="$(gcloud secrets versions access latest --secret KEIBA_PROXY_KEY --project "$PROJECT_ID")"
      echo "::add-mask::${KEIBA_PROXY_KEY}"
      export KEIBA_PROXY_KEY
    fi
    run env PYTHONPATH=. python3 -m src.scripts.ops.cloudrun_smoke --base "$SMOKE_BASE" --race-id "$SMOKE_RACE_ID"
    ;;
  promote)
    [ "$TARGET" = prod ] || { echo "promote は prod だけ（stg は通信を切り替えない）" >&2; exit 2; }
    run bash scripts/gcp/deploy_cloud_run_api.sh promote --apply
    ;;
  jobs)
    [ "$TARGET" = prod ] || { echo "jobs は prod だけ" >&2; exit 2; }
    if [ "${ENABLE_JOBS_DEPLOY:-}" != "true" ]; then
      echo "ENABLE_JOBS_DEPLOY が true ではないため、Jobs の更新は行いません（Cloud Shell でキュー・Run Job・IAM を整えたあとに有効化）"; exit 0
    fi
    PAR=(); [ "${JOBS_PARALLEL:-1}" = "1" ] && PAR=(--parallel)    # 切替(完全移行)まで並走。ホストの結果を上書きしない
    run env IMAGE="$JOBS_IMAGE" bash scripts/gcp/deploy_cloud_run_jobs.sh apply queue --yes
    run env IMAGE="$JOBS_IMAGE" bash scripts/gcp/deploy_cloud_run_jobs.sh apply jobs ${PAR[@]+"${PAR[@]}"} --yes
    run env IMAGE="$JOBS_IMAGE" bash scripts/gcp/deploy_cloud_run_jobs.sh apply scheduler ${PAR[@]+"${PAR[@]}"} --yes
    ;;
  *) echo "不明なステップ: $STEP（build-images / build-api-image / check-jobs-image / api-revision / smoke / promote / jobs）" >&2; exit 2 ;;
esac
