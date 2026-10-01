#!/usr/bin/env bash
# =============================================================================
# keiba-vpn GCP Cloud Run Jobs 一括デプロイ 設計図
#
# docs/operations/gcp-cloud-run-jobs.md の10ジョブそれぞれについて、
# `gcloud run jobs deploy` + `gcloud scheduler jobs create http` 相当のコマンドを
# 生成・表示する。
#
# !! 重要 !!
#   このスクリプトは「設計図」であり、既定では何も実行しない（show のみ）。
#   実際に GCP へデプロイ・スケジューラ登録する場合は、
#     1. PROJECT_ID / REGION / IMAGE / SCHEDULER_SA_EMAIL を実値に置き換える
#     2. 対象のコンテナイメージ（本リポジトリを含む Docker イメージ）を
#        事前に Artifact Registry へ push しておく
#     3. `--apply` を付けて明示的に実行する
#   の3点をユーザー側で確認してから使うこと。
#
# Usage:
#   bash scripts/gcp/deploy_cloud_run_jobs.sh                 # 全ジョブのコマンドを表示 (既定)
#   bash scripts/gcp/deploy_cloud_run_jobs.sh show             # 同上
#   bash scripts/gcp/deploy_cloud_run_jobs.sh show <job-name>   # 1ジョブのみ表示
#   bash scripts/gcp/deploy_cloud_run_jobs.sh list              # ジョブ名一覧のみ
#   PROJECT_ID=... REGION=... IMAGE=... SCHEDULER_SA_EMAIL=... \
#     bash scripts/gcp/deploy_cloud_run_jobs.sh --apply <job-name>
#       # 実際に gcloud を実行してデプロイ・スケジューラ登録する（要事前準備・自己責任）
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"

# ユーザー側で実値に置き換える前提のプレースホルダ（環境変数で上書き可）
PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"
REGION="${REGION:-asia-northeast1}"
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/keiba-vpn/cloud-jobs:latest}"
SCHEDULER_SA_EMAIL="${SCHEDULER_SA_EMAIL:-keiba-scheduler@${PROJECT_ID}.iam.gserviceaccount.com}"
RUN_SA_EMAIL="${RUN_SA_EMAIL:-keiba-cloud-run-jobs@${PROJECT_ID}.iam.gserviceaccount.com}"

# -----------------------------------------------------------------------------
# ジョブ定義: name|module|memory|timeout|cron(JST)|description
#   詳細・根拠は docs/operations/gcp-cloud-run-jobs.md を参照。
# -----------------------------------------------------------------------------
JOB_DEFS=(
  "structure-check|src.scraper.run structure-check|1Gi|900s|0 6 * * *|構造チェック（毎朝06:00 JST）"
  "queue-maintenance|src.scripts.cloud_jobs.queue_maintenance|256Mi|300s|0 * * * *|キュー定期メンテ（1時間ごと）"
  "daily-shutuba-enqueue|src.scripts.cloud_jobs.daily_shutuba_enqueue|512Mi|600s|0 7 * * *|出馬表自動取得（毎朝07:00 JST）"
  "weekly-sire-agg|src.scripts.maintenance.aggregate_sire_aptitude|2Gi|1800s|0 6 * * 1|週次種牡馬集計（毎週月曜06:00 JST）"
  "jockey-trainer-stats|src.pipeline.build_jockey_trainer_stats|2Gi|1800s|30 5 * * *|騎手・調教師統計（毎日05:30 JST）"
  "race-quality-day|src.scripts.cloud_jobs.race_quality_day|1Gi|600s|0 19 * * *|レース質日次一括推定（提案値: 毎日19:00 JST）"
  "tracking-difficulty-precompute|src.scripts.maintenance.precompute_tracking_difficulty_all --skip-existing|2Gi|3600s|0 8 * * *|追走難度precompute（提案値: 毎日08:00 JST）"
  "track-speed-rebuild-baselines|src.research.race.build_track_speed_baselines|2Gi|1800s|0 4 * * 0|track-speedベースライン再構築（提案値: 毎週日曜04:00 JST）"
  "myostatin-recalculate|src.scripts.cloud_jobs.myostatin_recalculate|256Mi|120s|0 5 1 * *|ミオスタチン再計算（提案値: 毎月1日05:00 JST）"
  "odds-train|src.scripts.data.train_final_odds_model|4Gi|7200s|0 3 * * 0|オッズ予測モデル学習（提案値: 毎週日曜03:00 JST）"
)

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

find_job_def() {
    local target="$1"
    for def in "${JOB_DEFS[@]}"; do
        local name="${def%%|*}"
        if [ "$name" = "$target" ]; then
            echo "$def"
            return 0
        fi
    done
    return 1
}

# `module` フィールド（"src.scripts.cloud_jobs.foo --bar" 等）を
# gcloud run jobs deploy の --command/--args に変換する。
module_to_args() {
    local module_and_args="$1"
    # python3 -m <module> [extra args...]
    echo "python3,-m,$(echo "$module_and_args" | sed 's/ /,/g')"
}

print_deploy_commands() {
    local def="$1"
    IFS='|' read -r name module memory timeout schedule desc <<<"$def"
    local args
    args="$(module_to_args "$module")"

    echo "# -------------------------------------------------------------"
    echo "# ${name} — ${desc}"
    echo "# -------------------------------------------------------------"
    cat <<CMD
gcloud run jobs deploy ${name} \\
  --project="${PROJECT_ID}" \\
  --region="${REGION}" \\
  --image="${IMAGE}" \\
  --command="python3" \\
  --args="-m,${module// /,}" \\
  --memory="${memory}" \\
  --task-timeout="${timeout}" \\
  --max-retries=1 \\
  --service-account="${RUN_SA_EMAIL}" \\
  --set-env-vars="KEIBA_ENV=prod"

gcloud scheduler jobs create http ${name}-scheduler \\
  --project="${PROJECT_ID}" \\
  --location="${REGION}" \\
  --schedule="${schedule}" \\
  --time-zone="Asia/Tokyo" \\
  --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${name}:run" \\
  --http-method=POST \\
  --oauth-service-account-email="${SCHEDULER_SA_EMAIL}"
CMD
    echo
}

cmd_list() {
    for def in "${JOB_DEFS[@]}"; do
        local name="${def%%|*}"
        echo "$name"
    done
}

cmd_show() {
    local target="${1:-}"
    if [ -n "$target" ]; then
        local def
        if ! def="$(find_job_def "$target")"; then
            echo "不明なジョブ名: $target" >&2
            echo "一覧: $(cmd_list | tr '\n' ' ')" >&2
            exit 1
        fi
        print_deploy_commands "$def"
    else
        for def in "${JOB_DEFS[@]}"; do
            print_deploy_commands "$def"
        done
    fi
}

cmd_apply() {
    local target="${1:-}"
    if [ -z "$target" ]; then
        echo "--apply には対象ジョブ名が必須です（全件一括適用はサポートしない。安全のため1件ずつ確認すること）" >&2
        exit 1
    fi
    if [ "$PROJECT_ID" = "YOUR_GCP_PROJECT_ID" ]; then
        echo "PROJECT_ID が未設定（placeholder のまま）。環境変数で実値を指定してください。" >&2
        exit 1
    fi
    local def
    if ! def="$(find_job_def "$target")"; then
        echo "不明なジョブ名: $target" >&2
        exit 1
    fi
    echo "=== 以下のコマンドを実行します ==="
    print_deploy_commands "$def"
    echo "=== 本当に実行しますか？ Ctrl-C で中断できます (5秒待機) ==="
    sleep 5
    IFS='|' read -r name module memory timeout schedule desc <<<"$def"
    gcloud run jobs deploy "${name}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --image="${IMAGE}" \
      --command="python3" \
      --args="-m,${module// /,}" \
      --memory="${memory}" \
      --task-timeout="${timeout}" \
      --max-retries=1 \
      --service-account="${RUN_SA_EMAIL}" \
      --set-env-vars="KEIBA_ENV=prod"
    gcloud scheduler jobs create http "${name}-scheduler" \
      --project="${PROJECT_ID}" \
      --location="${REGION}" \
      --schedule="${schedule}" \
      --time-zone="Asia/Tokyo" \
      --uri="https://${REGION}-run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${name}:run" \
      --http-method=POST \
      --oauth-service-account-email="${SCHEDULER_SA_EMAIL}"
}

main() {
    local apply=0
    local args=()
    for a in "$@"; do
        if [ "$a" = "--apply" ]; then
            apply=1
        else
            args+=("$a")
        fi
    done

    local sub="${args[0]:-show}"
    case "$sub" in
        show)
            cmd_show "${args[1]:-}"
            ;;
        list)
            cmd_list
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            # `deploy_cloud_run_jobs.sh --apply <job-name>` のように sub がジョブ名の場合
            if [ "$apply" -eq 1 ]; then
                cmd_apply "$sub"
                exit 0
            fi
            echo "不明なサブコマンド: $sub" >&2
            usage
            exit 1
            ;;
    esac

    if [ "$apply" -eq 1 ] && [ "$sub" = "show" ]; then
        echo "注意: --apply は <job-name> とセットで指定してください（例: --apply structure-check）" >&2
    fi
}

main "$@"
