#!/usr/bin/env bash
# T-060: config/settings.yaml の mlflow.models が MODEL_CATALOG と一致していること
source "$(dirname "$0")/_lib.sh"
TITLE="T-060 settings.yaml と MLflow カタログの一致"

echo "$TITLE"
check "pytest: tests/config/test_settings_mlflow_models_match_catalog.py" \
  python3 -m pytest tests/config/test_settings_mlflow_models_match_catalog.py -q -p no:cacheprovider

py_check "カタログ全モデルのポートが settings.yaml にある（表示: キー → ポート）" <<'PY'
import sys, yaml
from src.pipeline.mlflow.catalog import MODEL_CATALOG
cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))["mlflow"]["models"]
bad = [k for k, s in MODEL_CATALOG.items() if cfg.get(k, {}).get("serve_port") != s.default_serve_port]
for k, s in MODEL_CATALOG.items():
    print(f"         {k:22s} -> {cfg.get(k, {}).get('serve_port')} (catalog {s.default_serve_port})")
sys.exit(1 if bad else 0)
PY
finish
