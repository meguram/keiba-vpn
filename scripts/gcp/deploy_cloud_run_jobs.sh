#!/usr/bin/env bash
# =============================================================================
# Cloud Run Jobs / Cloud Scheduler / Cloud Tasks キューの作成（config/gcp_job_map.json が正本）
#
# 既定は『表示だけ』。何も実行しない。実行するときだけ apply（確認つき）。dev PC は GCP に接続しないので、実行は Cloud Shell かホスト（stg）で。
#
#   bash scripts/gcp/deploy_cloud_run_jobs.sh                      # 全部のコマンドを表示（= show all）
#   bash scripts/gcp/deploy_cloud_run_jobs.sh show queue|jobs|scheduler|all [--parallel] [--include-unready]
#   bash scripts/gcp/deploy_cloud_run_jobs.sh --dry-run apply all  # apply の中身を表示だけ（show と同じ）
#   PROJECT_ID=… GCS_BUCKET=… bash scripts/gcp/deploy_cloud_run_jobs.sh apply queue|jobs|scheduler|all [--parallel] [--include-unready] [--yes]
#   bash scripts/gcp/deploy_cloud_run_jobs.sh build                # イメージのビルド手順（Cloud Build）を表示
#   PROJECT_ID=… bash scripts/gcp/deploy_cloud_run_jobs.sh run-now group g-0800 [--yes]   # 手動で 1 グループ／1 ジョブを今すぐ実行（動作確認）
#   bash scripts/gcp/deploy_cloud_run_jobs.sh list                 # グループ・ジョブの一覧（準備状況つき）
#
# 変数: PROJECT_ID / REGION(asia-northeast1) / GCS_BUCKET / IMAGE / RUN_SA_EMAIL(keiba-jobs@) / SCHEDULER_SA_EMAIL(keiba-scheduler@) / CLOUD_TASKS_QUEUE(keiba-predict)
#       SECRETS_EXTRA（追加で渡す Secret Manager の秘密名。カンマ区切り）
# 環境変数には KEIBA_ENV=prod と KEIBA_RUN_LOCATION=gcp が必ず入る（Slack の先頭に【PROD】・末尾に実行場所: GCP が付き、ホスト(stg)の通知と見分けられる）。
# --parallel は並走期間（KEIBA_PARALLEL_RUN=1。予測は影モード固定）。切替後は付けずに jobs を再 apply する。
# 手順の全体: docs/operations/gcp_jobs_migration.md
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
PY="${PYTHON:-python3}"
cd "$ROOT"

DRY=0; YES=0; ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --yes) YES=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
SUB="${ARGS[0]:-show}"
WHAT="${ARGS[1]:-all}"
EXTRA=()
for a in "${ARGS[@]:2}"; do EXTRA+=("$a"); done

gen() { PYTHONPATH="$ROOT" "$PY" -m src.ops.scheduler_gen "$WHAT" "${EXTRA[@]}"; }

need_real() {
  if [ "${PROJECT_ID:-}" = "" ] || [ "${PROJECT_ID:-}" = "YOUR_GCP_PROJECT_ID" ]; then echo "PROJECT_ID が未設定です" >&2; exit 1; fi
  if [ "${GCS_BUCKET:-}" = "" ] || [ "${GCS_BUCKET:-}" = "YOUR_BUCKET" ]; then echo "GCS_BUCKET が未設定です" >&2; exit 1; fi
}

confirm() {
  [ "$YES" = 1 ] && return 0
  read -r -p "上のコマンドを実行します。よければ yes と入力: " ans
  [ "$ans" = yes ] || { echo "中止しました"; exit 1; }
}

case "$SUB" in
  show)
    gen
    ;;
  list)
    PYTHONPATH="$ROOT" "$PY" - <<'PYEOF'
from src.ops import job_map
d = job_map.load()
for gid, g in d["scheduler_groups"].items():
    print(f"{gid:12s} {g['cron_jst']:22s} size={g['size']}  {g['desc']}")
    for jid in g["jobs"]:
        j = job_map.job_by_id(jid, d)
        print(f"    {'ready ' if j['ready'] else 'WAIT  '}{jid:30s} {'; '.join(j['blockers'])[:90]}")
PYEOF
    ;;
  build)
    cat <<'MSG'
# Cloud Shell で（無料枠: Cloud Build 2,500 分/月）
export PROJECT=$(gcloud config get-value project); export REGION=asia-northeast1
gcloud artifacts repositories describe keiba-vpn --location $REGION >/dev/null 2>&1 || gcloud artifacts repositories create keiba-vpn --repository-format docker --location $REGION
gcloud builds submit --config - . <<'YAML'
steps:
- name: gcr.io/cloud-builders/docker
  args: ['build','-f','deploy/docker/Dockerfile.jobs','-t','asia-northeast1-docker.pkg.dev/$PROJECT_ID/keiba-vpn/jobs:latest','.']
images: ['asia-northeast1-docker.pkg.dev/$PROJECT_ID/keiba-vpn/jobs:latest']
YAML
MSG
    ;;
  apply)
    need_real
    OUT="$(mktemp)"; trap 'rm -f "$OUT"' EXIT
    gen >"$OUT"
    cat "$OUT"
    if [ "$DRY" = 1 ]; then echo; echo "（--dry-run: 実行していません）"; exit 0; fi
    confirm
    bash "$OUT"
    ;;
  run-now)
    need_real
    KIND="${ARGS[1]:-}"; NAME="${ARGS[2]:-}"
    [ -n "$KIND" ] && [ -n "$NAME" ] || { echo "usage: run-now group <group_id> | job <job_id>" >&2; exit 2; }
    SIZE="$(PYTHONPATH="$ROOT" "$PY" - "$KIND" "$NAME" <<'PYEOF'
import sys
from src.ops import job_map
kind, name = sys.argv[1:3]
d = job_map.load()
if kind == "group":
    size = d["scheduler_groups"][name]["size"]
else:
    j = job_map.job_by_id(name, d)
    g = job_map.group_of(j["group"], d) if j.get("group") else None
    size = g["size"] if g else j.get("size", "l")
print(d["runtime"]["run_jobs"][size]["name"])
PYEOF
)"
    CMD=(gcloud run jobs execute "$SIZE" --project "$PROJECT_ID" --region "${REGION:-asia-northeast1}" --wait "--args=-m,src.ops.job_dispatch,$KIND,$NAME")
    echo "${CMD[*]}"
    if [ "$DRY" = 1 ]; then echo "（--dry-run: 実行していません）"; exit 0; fi
    confirm
    "${CMD[@]}"
    ;;
  -h|--help|help)
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "不明なサブコマンド: $SUB" >&2; exit 2 ;;
esac
