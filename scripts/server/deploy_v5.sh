#!/bin/bash
# v5 shadow の配備（コードの配置 + PX の状態の作成）。.env は触らない（/tmp/enable_v5_shadow.sh で別に追記）。DRY_RUN=1 で確認だけ。
set -e
R=/home/jovyan/work/keiba-vpn
TS=$(date +%Y%m%d_%H%M%S)
B=$R/data/local/meta/cron_backups/deploy_v5_$TS
SRC=/tmp/deploy_v5_files
FILES="src/pipeline/features/serving_v5.py src/pipeline/features/serving_v5_px.py src/pipeline/features/serving_v5_columns.json src/pipeline/features/tp38_pedigree.py src/pipeline/features/tp38_ridge.py src/pipeline/features/tp40_condition.py src/pipeline/inference/v3_serving.py src/pipeline/inference/weight_triggered_predict.py src/scripts/data/build_px_state.py"
echo "[deploy_v5] 対象: $FILES"
for f in $FILES; do [ -f $SRC/$f ] || { echo "ファイルが無い: $SRC/$f"; exit 2; }; done
n=$(pgrep -fc "weight_triggered|raceday_runner|auto_scrape_queue" || true); echo "[deploy_v5] 稼働中のランナー系: ${n:-0}（止めない。変更は次回起動から）"
if [ "$DRY_RUN" = 1 ]; then echo "[deploy_v5] DRY_RUN 終了"; exit 0; fi
sudo -n mkdir -p $B
for f in $FILES; do if [ -f $R/$f ]; then sudo -n mkdir -p $B/$(dirname $f); sudo -n cp -a $R/$f $B/$f; fi; done
echo "[deploy_v5] 退避: $B"
for f in $FILES; do sudo -n cp $SRC/$f $R/$f; done
cd $R
for f in $FILES; do case $f in *.py) /opt/conda/bin/python3 -c "import ast,sys; ast.parse(open('$f',encoding='utf-8').read())" ;; esac; done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /opt/conda/bin/python3 -c "import src.pipeline.features.serving_v5_px, src.pipeline.inference.v3_serving, src.pipeline.inference.weight_triggered_predict; print('import OK')"
echo "[deploy_v5] PX の状態を作る（数分）"
sudo -n env PYTHONPATH=. KEIBA_ENV=stg PYTHONDONTWRITEBYTECODE=1 /opt/conda/bin/python3 -m src.scripts.data.build_px_state --windows 2026-07-01 2>&1 | grep -v "Warning\|join_asof" | tail -5
echo "[deploy_v5] 完了。ロールバック: cp -a $B/* $R/ （新規ファイルは削除）"
