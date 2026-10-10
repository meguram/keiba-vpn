#!/usr/bin/env bash
# Layer 3 スナップショット + 推論パイプライン（T-15 トリガ相当）
set -euo pipefail
RACE_ID="${1:?usage: run_inference.sh RACE_ID}"
cd "$(dirname "$0")/../.."
python -m src.db.batch.stats_snapshot "$RACE_ID"
python -c "from src.pipeline.inference.inference_pipeline import run_inference_for_race; print(run_inference_for_race('$RACE_ID'))"
