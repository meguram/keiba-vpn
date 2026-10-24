#!/usr/bin/env bash
# すべての予測モデルノートブックを順番に再実行するスクリプト
# Usage: bash scripts/run_all_model_notebooks.sh [--skip-nb01]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/data/local/modeling/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_all_${TIMESTAMP}.log"

SKIP_NB01=${1:-""}

run_nb() {
    local nb_in="$1"
    local nb_out="${nb_in%.ipynb}_executed.ipynb"
    local label="$(basename "$nb_in")"
    local nb_dir="$(dirname "$nb_in")"

    echo "[$(date '+%H:%M:%S')] ▶ START: $label" | tee -a "$LOG_FILE"
    # ノートブックのディレクトリから実行（../../ の相対パス解決のため）
    (cd "$REPO_ROOT/$nb_dir" && papermill "$(basename "$nb_in")" "$(basename "$nb_out")" \
        --kernel python3 \
        --no-progress-bar) \
        2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ OK: $label" | tee -a "$LOG_FILE"
    else
        echo "[$(date '+%H:%M:%S')] ✗ FAIL (exit=$exit_code): $label" | tee -a "$LOG_FILE"
        exit $exit_code
    fi
    echo "" | tee -a "$LOG_FILE"
}

echo "============================================================" | tee -a "$LOG_FILE"
echo " keiba-vpn 全モデルノートブック再計算" | tee -a "$LOG_FILE"
echo " 開始: $(date)" | tee -a "$LOG_FILE"
echo " ログ: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

NB_DIR="$REPO_ROOT/notebooks/modeling"
MEGU_DIR="$REPO_ROOT/notebooks/megu_index"

# ─── Step 1: マスターデータセット ─────────────────────────────────────────
if [ "$SKIP_NB01" != "--skip-nb01" ]; then
    run_nb "$NB_DIR/nb-01-master-dataset.ipynb"
else
    echo "[$(date '+%H:%M:%S')] SKIP: nb-01 (--skip-nb01 指定)" | tee -a "$LOG_FILE"
fi

# ─── Step 2: エンティティ統計 ───────────────────────────────────────────────
run_nb "$NB_DIR/nb-02-entity-stats.ipynb"

# ─── Step 3: 個別予測モデル（Stage 1 → Stage 2 の順序に従う）──────────────
# T-6 脚質（→ T-1 の入力）
run_nb "$NB_DIR/nb-03-t6-running-style.ipynb"

# T-8 ペース（→ T-1 の入力）
run_nb "$NB_DIR/nb-04-t8-pace.ipynb"

# T-4 上り3F（→ T-9 の入力）
run_nb "$NB_DIR/nb-05-t4-last3f.ipynb"

# T-5 位置取り（→ T-1 の入力）
run_nb "$NB_DIR/nb-06-t5-position.ipynb"

# T-9 走破タイム（T-4 OOF を使用）
run_nb "$NB_DIR/nb-07-t9-finish-time.ipynb"

# ─── Step 4: 総合モデル ────────────────────────────────────────────────────
run_nb "$NB_DIR/nb-08-t1-win-prob.ipynb"

# ─── Step 5: めぐ指数 β係数再推定 ─────────────────────────────────────────
run_nb "$MEGU_DIR/nb-02-pace-and-partime.ipynb"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 完了: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
