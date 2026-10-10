#!/usr/bin/env bash
# =============================================================================
# keiba-vpn ログローテーション
#
# 2種類のローテーションポリシー:
#   TYPE-A: セッションログ (server_*.log, mlflow_*.log)
#     → 起動ごとにタイムスタンプ付きファイルが生成される。古いものを日数で削除。
#   TYPE-B: 追記型ログ (backfill_*.log, raceday_*.log など)
#     → 一つのファイルに追記し続ける。閾値を超えたら gzip 圧縮アーカイブ化。
#       アーカイブは直近 N 世代を保持、それ以上は削除。
#
# Usage:
#   bash scripts/cron/rotate_logs.sh          # 通常実行
#   bash scripts/cron/rotate_logs.sh --dry-run  # 実行確認のみ
#   bash scripts/cron/rotate_logs.sh --status   # 現在のログサイズ一覧
# =============================================================================

set -euo pipefail
export TZ="${TZ:-Asia/Tokyo}"

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"

DRY_RUN=0
STATUS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --status)  STATUS_ONLY=1 ;;
    esac
done

# -------------------------------------------------------------------
# ユーティリティ
# -------------------------------------------------------------------

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log()     { echo "$(_ts) [rotate_logs] $*"; }
dry_log() { echo "$(_ts) [rotate_logs][DRY-RUN] $*"; }

# バイト → 人が読める単位
human_size() {
    local bytes="${1:-0}"
    if   [ "$bytes" -ge $((1024*1024*1024)) ]; then
        awk "BEGIN{printf \"%.1fGB\", $bytes/1073741824}"
    elif [ "$bytes" -ge $((1024*1024)) ]; then
        awk "BEGIN{printf \"%.1fMB\", $bytes/1048576}"
    elif [ "$bytes" -ge 1024 ]; then
        awk "BEGIN{printf \"%.1fKB\", $bytes/1024}"
    else
        echo "${bytes}B"
    fi
}

file_size_bytes() {
    stat -c%s "$1" 2>/dev/null || echo 0
}

file_size_mb() {
    local bytes
    bytes="$(file_size_bytes "$1")"
    echo $(( bytes / 1048576 ))
}

file_age_days() {
    local mtime
    mtime="$(stat -c%Y "$1" 2>/dev/null || echo 0)"
    echo $(( ( $(date +%s) - mtime ) / 86400 ))
}

# -------------------------------------------------------------------
# TYPE-A: セッションログ管理
#   ・age_days > keep_days なら削除
#   ・age_days <= keep_days でも size > max_size_mb なら gzip アーカイブ化して truncate
#     （MLflow など長時間書き続けるセッションログへの対応）
# 引数: <パターン glob> <保持日数> [最大サイズMB=200]
# -------------------------------------------------------------------
cleanup_session_logs() {
    local pattern="$1"
    local keep_days="$2"
    local max_size_mb="${3:-200}"
    local deleted=0
    local freed_bytes=0

    while IFS= read -r -d '' f; do
        if [ -L "$f" ]; then continue; fi

        local age_days sz size_mb
        age_days="$(file_age_days "$f")"
        sz="$(file_size_bytes "$f")"
        size_mb="$(file_size_mb "$f")"

        if [ "$age_days" -gt "$keep_days" ]; then
            # 7日以上前 → 削除
            if [ "$DRY_RUN" -eq 1 ]; then
                dry_log "DEL session $(basename "$f") age=${age_days}d $(human_size "$sz")"
            else
                rm -f "$f"
                log "DEL session $(basename "$f") age=${age_days}d $(human_size "$sz")"
                deleted=$(( deleted + 1 ))
                freed_bytes=$(( freed_bytes + sz ))
            fi
        elif [ "$size_mb" -gt "$max_size_mb" ]; then
            # サイズ超過 → gzip アーカイブ化して truncate（セッションは継続）
            local ts archive
            ts="$(date +%Y%m%d_%H%M%S)"
            archive="${f}.${ts}.gz"
            if [ "$DRY_RUN" -eq 1 ]; then
                dry_log "ROTATE-session $(basename "$f") ${size_mb}MB > ${max_size_mb}MB → $(basename "$archive")"
            else
                gzip -c "$f" > "$archive"
                truncate -s 0 "$f"
                log "ROTATE-session $(basename "$f") ${size_mb}MB > ${max_size_mb}MB → $(basename "$archive")"
            fi
        fi
    done < <(find "$LOG_DIR" -maxdepth 1 -name "$pattern" -type f -print0 2>/dev/null)

    if [ "$deleted" -gt 0 ]; then
        log "セッションログ削除: ${deleted} 件 $(human_size "$freed_bytes") 解放"
    fi
}

# -------------------------------------------------------------------
# TYPE-B: 追記型ログのローテーション
# 引数: <ファイルパス> <最大MB> <保持世代数>
# -------------------------------------------------------------------
rotate_append_log() {
    local file="$1"
    local max_mb="${2:-20}"
    local keep_archives="${3:-3}"

    if [ ! -f "$file" ]; then return 0; fi
    if [ -L "$file" ]; then return 0; fi

    local size_mb
    size_mb="$(file_size_mb "$file")"

    if [ "$size_mb" -lt "$max_mb" ]; then
        return 0
    fi

    local ts archive
    ts="$(date +%Y%m%d_%H%M%S)"
    archive="${file}.${ts}.gz"

    if [ "$DRY_RUN" -eq 1 ]; then
        dry_log "ROTATE $(basename "$file") ${size_mb}MB → $(basename "$archive")"
    else
        gzip -c "$file" > "$archive"
        truncate -s 0 "$file"
        log "ROTATE $(basename "$file") ${size_mb}MB → $(basename "$archive")"
    fi

    # 古いアーカイブを削除（keep_archives 世代を残す）
    local old_list
    old_list="$(ls -t "${file}".*.gz 2>/dev/null | tail -n +"$((keep_archives + 1))" || true)"
    if [ -n "$old_list" ]; then
        while IFS= read -r old_archive; do
            local old_sz
            old_sz="$(file_size_bytes "$old_archive")"
            if [ "$DRY_RUN" -eq 1 ]; then
                dry_log "DEL old-archive $(basename "$old_archive") $(human_size "$old_sz")"
            else
                rm -f "$old_archive"
                log "DEL old-archive $(basename "$old_archive") $(human_size "$old_sz")"
            fi
        done <<< "$old_list"
    fi
}

# -------------------------------------------------------------------
# STATUS: ログサイズ一覧
# -------------------------------------------------------------------
cmd_status() {
    echo "=== keiba-vpn ログ一覧 ($(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S %Z')) ==="
    echo ""
    local total_bytes=0

    # 通常ファイル
    while IFS= read -r -d '' f; do
        if [ -L "$f" ]; then continue; fi
        local sz age_days
        sz="$(file_size_bytes "$f")"
        age_days="$(file_age_days "$f")"
        printf "  %-55s %8s  %dd前\n" "$(basename "$f")" "$(human_size "$sz")" "$age_days"
        total_bytes=$(( total_bytes + sz ))
    done < <(find "$LOG_DIR" -maxdepth 1 \( -name "*.log" -o -name "*.log.*.gz" \) -type f -print0 2>/dev/null | sort -z)

    # シンボリックリンク
    while IFS= read -r -d '' f; do
        local target
        target="$(readlink "$f")"
        printf "  %-55s %8s  (symlink → %s)\n" "$(basename "$f")" "" "$target"
    done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log" -type l -print0 2>/dev/null | sort -z)

    echo ""
    echo "合計（通常ファイル）: $(human_size "$total_bytes")"
}

# -------------------------------------------------------------------
# メインローテーション処理
# -------------------------------------------------------------------
cmd_rotate() {
    log "=== ローテーション開始 (DRY_RUN=${DRY_RUN}) ==="

    mkdir -p "$LOG_DIR"

    # ── TYPE-A: セッションログ（タイムスタンプ付き） ──
    # server_*.log: 7日以上前を削除、200MB 超は gzip rotate
    cleanup_session_logs "server_2*.log"  7  200
    # mlflow_*.log: 7日以上前を削除、100MB 超は gzip rotate（MLflow は冗長ログを出しやすい）
    cleanup_session_logs "mlflow_2*.log"  7  100

    # ── TYPE-B: 追記型ログ（サイズ閾値でローテーション、世代管理） ──

    # backfill 系 (大きくなりがち) → 20MB で rotate, 3世代保持
    while IFS= read -r -d '' f; do
        if [ ! -L "$f" ]; then rotate_append_log "$f" 20 3; fi
    done < <(find "$LOG_DIR" -maxdepth 1 \( -name "backfill_*.log" -o -name "backfill_full_*.log" \) -type f -print0 2>/dev/null)

    # スクレイパー系 → 20MB / 10MB で rotate, 5世代保持
    rotate_append_log "${LOG_DIR}/raceday_eve.log"             20 5
    rotate_append_log "${LOG_DIR}/raceday_runner.log"          20 5
    rotate_append_log "${LOG_DIR}/raceday_result_runner.log"   20 5
    rotate_append_log "${LOG_DIR}/raceday_evening.log"         10 5
    rotate_append_log "${LOG_DIR}/weekly_update.log"           10 5
    rotate_append_log "${LOG_DIR}/daily_race_lists_am.log"     10 3
    rotate_append_log "${LOG_DIR}/daily_race_lists_pm.log"     10 3
    rotate_append_log "${LOG_DIR}/jra_baba_morning.log"        10 3
    rotate_append_log "${LOG_DIR}/horse_name_index.log"         5 3
    rotate_append_log "${LOG_DIR}/jockey_trainer_stats.log"    10 3

    # cron ウィンドウ系
    rotate_append_log "${LOG_DIR}/cron_window_scrape.log"      20 3
    rotate_append_log "${LOG_DIR}/run_scrapes_in_cron_window_rerun.log" 20 3
    rotate_append_log "${LOG_DIR}/external_cron_month_coverage_skip_eve.log" 20 3

    # watchdog (行数は watchdog 自身が管理。サイズ保険)
    rotate_append_log "${LOG_DIR}/watchdog.log"                10 3

    # その他 .log → 50MB で rotate, 2世代保持（保険）
    while IFS= read -r -d '' f; do
        if [ -L "$f" ]; then continue; fi
        local base
        base="$(basename "$f")"
        # 既知パターンはすでに個別処理済み
        case "$base" in
            backfill_*.log|backfill_full_*.log) continue ;;
            server_2*.log|mlflow_2*.log)        continue ;;
            raceday_*.log|daily_race_lists_*.log) continue ;;
            weekly_update.log|jra_baba_morning.log) continue ;;
            horse_name_index.log|jockey_trainer_stats.log) continue ;;
            cron_window_scrape.log|run_scrapes_in_cron_window_rerun.log) continue ;;
            external_cron_month_coverage_skip_eve.log) continue ;;
            watchdog.log) continue ;;
        esac
        rotate_append_log "$f" 50 2
    done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -print0 2>/dev/null)

    # ── アーカイブ 30 日以上前のものを削除 ──
    local old_gz_count=0
    while IFS= read -r -d '' f; do
        local age_days sz
        age_days="$(file_age_days "$f")"
        sz="$(file_size_bytes "$f")"
        if [ "$age_days" -gt 30 ]; then
            if [ "$DRY_RUN" -eq 1 ]; then
                dry_log "DEL old-gz $(basename "$f") age=${age_days}d $(human_size "$sz")"
            else
                rm -f "$f"
                log "DEL old-gz $(basename "$f") age=${age_days}d $(human_size "$sz")"
                old_gz_count=$(( old_gz_count + 1 ))
            fi
        fi
    done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log.*.gz" -type f -print0 2>/dev/null)

    if [ "$old_gz_count" -gt 0 ]; then
        log "古い gzip アーカイブ削除: ${old_gz_count} 件"
    fi

    log "=== ローテーション完了 ==="
}

# -------------------------------------------------------------------
# エントリポイント
# -------------------------------------------------------------------
if [ "$STATUS_ONLY" -eq 1 ]; then
    cmd_status
else
    cmd_rotate
fi
